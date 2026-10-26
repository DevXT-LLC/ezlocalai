#!/usr/bin/env python3
"""V2 supports post-finalize rollback recreation; original applied V1 files remain frozen. Operator-owned router lease required. Never logs full Docker config or dotenv.

apply PLAN [--dry-run] [--retain-original]
finalize RUN_DIRECTORY
rollback RUN_DIRECTORY

PLAN: run_id, changes, env_file(default /home/josh/ezlocalai/.env),
      output_dir(default /tmp/ezlocalai-env-update-RUN_ID), timeout(default180).
Only healthy `ezlocalai` with bind mounts and host or one simple local custom
bridge network is supported; dynamic IP/MAC/endpoint identities may change.
The original image tag must still resolve to the running immutable image.
Backups contain secrets: output_dir is0700; files are0600.
HostConfig remains exact except Docker's null/false OomKillDisable default.
The default equivalence never permits true, and memory/swap limits stay exact.
"""
import argparse, copy, fcntl, hashlib, http.client, json, os, pathlib, re, signal, socket, stat, sys, time, urllib.parse, secrets, contextlib

class UpdateError(RuntimeError): pass
class DockerHTTPError(UpdateError):
    def __init__(self, status, message): self.status=status; super().__init__(message)
class DockerTransportError(UpdateError): pass
@contextlib.contextmanager
def cleanup_signals():
    previous={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    try: yield
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)

class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path): super().__init__('localhost', timeout=90); self.path=path
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); self.sock.settimeout(self.timeout); self.sock.connect(self.path)
class Docker:
    def request(self, method, path, body=None):
        conn=UnixHTTP('/var/run/docker.sock')
        try:
            raw=None if body is None else json.dumps(body).encode()
            conn.request(method,path,body=raw,headers={} if raw is None else {'Content-Type':'application/json'})
            r=conn.getresponse(); data=r.read()
            if not 200<=r.status<300:
                raise DockerHTTPError(r.status,'Docker request failed: '+method+' '+path.split('?')[0]+' HTTP '+str(r.status))
            return json.loads(data) if data else None
        except (OSError,http.client.HTTPException) as e:
            raise DockerTransportError('Docker transport failure during '+method+' '+path.split('?')[0]) from None
        finally: conn.close()
    def inspect(self, name): return self.request('GET','/containers/'+quote(name)+'/json')
    def image(self, image): return self.request('GET','/images/'+quote(image)+'/json')
    def network(self, name): return self.request('GET','/networks/'+quote(name))
    def tag(self, image, tag):
        repo,name=tag.rsplit(':',1); self.request('POST','/images/'+quote(image)+'/tag?'+urllib.parse.urlencode({'repo':repo,'tag':name}))
    def create(self, name, config, hostconfig, networking=None):
        body={**config,'HostConfig':hostconfig}
        if networking is not None: body['NetworkingConfig']=networking
        return self.request('POST','/containers/create?'+urllib.parse.urlencode({'name':name}),body)['Id']
    def stop(self, ident): self.request('POST','/containers/'+quote(ident)+'/stop?t=60')
    def start(self, ident): self.request('POST','/containers/'+quote(ident)+'/start')
    def rename(self, ident, name): self.request('POST','/containers/'+quote(ident)+'/rename?'+urllib.parse.urlencode({'name':name}))
    def remove(self, ident): self.request('DELETE','/containers/'+quote(ident)+'?force=true&v=false')

def optional_inspect(engine, name):
    try: return engine.inspect(name)
    except DockerHTTPError as error:
        if error.status==404: return None
        raise

def quote(s): return urllib.parse.quote(str(s),safe='')
def digest(b): return hashlib.sha256(b).hexdigest()
def encoded(o): return json.dumps(o,sort_keys=True,separators=(',',':')).encode()
def atomic(path, data, mode=0o600):
    path=pathlib.Path(path); tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp,mode); os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass

def private_json(path, obj): atomic(path,json.dumps(obj,indent=2,sort_keys=True).encode()+b'\n')
def best_effort_json(path,obj):
    try: private_json(path,obj)
    except OSError: pass
def env_values(entries):
    result={}
    for entry in entries:
        if '=' not in entry: raise UpdateError('Container has a malformed environment entry')
        key,value=entry.split('=',1)
        if key in result: raise UpdateError('Container has duplicate environment keys')
        result[key]=value
    return result
ALLOWED={
    'LLM_UBATCH_SIZE':r'(?:auto|[1-9][0-9]{0,7})',
    'LLM_BATCH_SIZE':r'(?:auto|[1-9][0-9]{0,7})',
    'LLM_PROMPT_CACHE_MAX_RAM_FRACTION':r'(?:0?\.[0-9]{1,6}|0|1(?:\.0{1,6})?)',
    'GGML_CUDA_GRAPH_OPT':r'[01]',
    'EZLOCALAI_LLAMA_STATE_IO_BATCHING':r'[01]',
}
def validate_changes(changes):
    if not isinstance(changes,dict) or not changes: raise UpdateError('changes must be a nonempty object')
    for key,value in changes.items():
        if key not in ALLOWED or not isinstance(value,str) or not re.fullmatch(ALLOWED[key],value):
            raise UpdateError('Unsupported performance setting or value')
    return changes

def render_env(raw, changes):
    try: text=raw.decode('utf-8')
    except UnicodeDecodeError: raise UpdateError('Dotenv is not UTF-8') from None
    # Refuse multiline quoting rather than risk matching a selected-key-looking
    # line inside another variable's quoted value. Other bytes remain verbatim.
    for line in text.splitlines(keepends=True):
        opening=re.match(r'^[ \t]*(?:export[ \t]+)?[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*([\"\'])',line)
        if opening:
            quotechar=opening.group(1); index=opening.end(); closed=False
            while index<len(line.rstrip('\r\n')):
                if line[index]=='\\': index+=2; continue
                if line[index]==quotechar: closed=True; break
                index+=1
            if not closed: raise UpdateError('Multiline quoted dotenv values require manual handling')
    lines=text.splitlines(keepends=True); found={}; old={key:None for key in changes}
    for i,line in enumerate(lines):
        for key in changes:
            match=re.match(r'^([ \t]*(?:export[ \t]+)?'+re.escape(key)+r'[ \t]*=)([^\r\n]*)(\r\n|\n|\r)?$',line)
            if not match: continue
            if key in found: raise UpdateError('Selected dotenv setting is duplicated: '+key)
            value=match.group(2).strip()
            # Existing integer/batch parsing is unchanged. Only the selected
            # cache fraction accepts a bounded decimal or its empty default.
            token = r'(?:0?\.[0-9]{1,6}|0|1(?:\.0{1,6})?)?' if key == 'LLM_PROMPT_CACHE_MAX_RAM_FRACTION' else r'(?:auto|[0-9]+)'
            parsed=re.fullmatch(r'(?:(%s)|\'(%s)\'|"(%s)")[ \t]*(?:#.*)?' % (token,token,token),value)
            if not parsed: raise UpdateError('Selected dotenv setting has unsupported syntax: '+key)
            old[key]=next(v for v in parsed.groups() if v is not None); found[key]=i
            lines[i]=match.group(1)+changes[key]+(match.group(3) or '')
    newline='\r\n' if '\r\n' in text else '\n'
    if any(key not in found for key in changes):
        if lines and not lines[-1].endswith(('\n','\r')): lines[-1]+=newline
        lines += [key+'='+value+newline for key,value in changes.items() if key not in found]
    return ''.join(lines).encode(), old

def mount_view(x):
    return sorted(encoded({k:m.get(k) for k in ('Type','Source','Destination','Driver','Mode','RW','Propagation')}).decode() for m in x['Mounts'])
def network_config(container):
    mode=container['HostConfig'].get('NetworkMode')
    if mode=='host': return None
    networks=container.get('NetworkSettings',{}).get('Networks',{})
    if mode in ('bridge','none','default','') or not isinstance(mode,str) or mode.startswith('container:') or set(networks)!={mode}:
        raise UpdateError('Only host or one named custom bridge network is supported')
    if container['HostConfig'].get('PublishAllPorts'): raise UpdateError('Dynamic published host ports are unsupported')
    for bindings in (container['HostConfig'].get('PortBindings') or {}).values():
        for binding in bindings or []:
            port=binding.get('HostPort','')
            if not isinstance(port,str) or not port.isdecimal() or not 1<=int(port)<=65535:
                raise UpdateError('Dynamic/ranged published host ports are unsupported')
    endpoint=networks[mode]
    if endpoint.get('IPAMConfig') is not None or endpoint.get('Links') not in (None,[]) or endpoint.get('DriverOpts') not in (None,{}) or endpoint.get('GwPriority',0)!=0 or container['Config'].get('MacAddress'):
        raise UpdateError('Static/complex endpoint network configuration is unsupported')
    aliases=endpoint.get('Aliases')
    if not isinstance(aliases,list) or any(not isinstance(alias,str) or not alias for alias in aliases):
        raise UpdateError('Custom bridge endpoint aliases are unsupported')
    return {'EndpointsConfig':{mode:{'IPAMConfig':endpoint.get('IPAMConfig'),'Links':endpoint.get('Links'),'Aliases':copy.deepcopy(aliases),'DriverOpts':endpoint.get('DriverOpts'),'GwPriority':endpoint.get('GwPriority',0)}}}

def ports_view(container):
    return {key:sorted((v.get('HostIp',''),v.get('HostPort','')) for v in values) if values is not None else None for key,values in (container.get('NetworkSettings',{}).get('Ports') or {}).items()}

def verify_network(replacement, original):
    expected=network_config(original)
    if network_config(replacement)!=expected: raise UpdateError('Replacement static endpoint configuration differs')
    if expected is not None:
        name=next(iter(expected['EndpointsConfig'])); old=original['NetworkSettings']['Networks'][name]; new=replacement['NetworkSettings']['Networks'][name]
        if replacement['State'].get('Running'):
            if new.get('NetworkID')!=old.get('NetworkID'): raise UpdateError('Replacement attached to a different Docker network')
            if ports_view(replacement)!=ports_view(original): raise UpdateError('Replacement effective published ports differ')

def hostconfig_view(value):
    result=copy.deepcopy(value)
    # Docker's nullable API default and explicit false both leave the OOM
    # killer enabled. Docker29 can normalize null to false on cgroupv2.
    # https://docs.docker.com/engine/containers/runmetrics/#running-docker-on-cgroup-v2
    if 'OomKillDisable' in result and (result['OomKillDisable'] is None or result['OomKillDisable'] is False):
        result['OomKillDisable']=False
    return result

def verify(replacement, original, new_env):
    expected=copy.deepcopy(original['Config']); expected['Env']=new_env
    if replacement['Image']!=original['Image']: raise UpdateError('Replacement immutable image differs')
    if encoded(replacement['Config'])!=encoded(expected):
        keys=sorted(k for k in set(replacement['Config'])|set(expected) if encoded(replacement['Config'].get(k))!=encoded(expected.get(k)))
        raise UpdateError('Replacement Config differs beyond requested environment: '+','.join(keys))
    newhost=hostconfig_view(replacement['HostConfig']); oldhost=hostconfig_view(original['HostConfig'])
    if encoded(newhost)!=encoded(oldhost):
        keys=sorted(k for k in set(newhost)|set(oldhost) if encoded(newhost.get(k))!=encoded(oldhost.get(k)))
        raise UpdateError('Replacement HostConfig differs: '+','.join(keys))
    if mount_view(replacement)!=mount_view(original): raise UpdateError('Replacement mounts differ')
    verify_network(replacement,original)
    normalizations={}
    if replacement['HostConfig'].get('OomKillDisable') is not original['HostConfig'].get('OomKillDisable'):
        normalizations['OomKillDisable']={'original':original['HostConfig'].get('OomKillDisable'),'replacement':replacement['HostConfig'].get('OomKillDisable'),'meaning':'OOM killer remains enabled'}
    return {'immutable_image_preserved':True,'config_preserved_except_selected_env':True,'hostconfig_preserved':True,'hostconfig_default_normalizations':normalizations,'mounts_preserved':True,'static_network_configuration_preserved':True}

def wait_healthy(engine, ident, seconds, sleep=time.sleep):
    deadline=time.monotonic()+seconds
    while True:
        current=engine.inspect(ident)
        if not current['State'].get('Running'): raise UpdateError('Replacement exited before becoming healthy')
        if current['State'].get('OOMKilled'): raise UpdateError('Replacement was OOM-killed')
        status=current['State'].get('Health',{}).get('Status')
        if status=='healthy': return current
        if status=='unhealthy': raise UpdateError('Replacement reported unhealthy')
        if time.monotonic()>=deadline: raise UpdateError('Replacement health timeout')
        sleep(1)

def load_state(root):
    root=pathlib.Path(root)
    if not root.is_dir() or stat.S_IMODE(root.stat().st_mode)&0o077: raise UpdateError('Run directory must be private')
    state=json.loads((root/'state.json').read_text()); original=json.loads((root/'original-inspect.json').read_text())
    return root,state,original

def rollback_inputs(root, state, original):
    target=pathlib.Path(state['env_file'])
    if target.is_symlink() or not target.is_file(): raise UpdateError('Rollback dotenv is not an ordinary file')
    raw=target.read_bytes()
    if digest(raw) not in (state['original_env_file_sha256'],state['updated_env_file_sha256']):
        raise UpdateError('Dotenv changed externally; automatic rollback refuses to overwrite it')
    backup_path=root/'dotenv.backup'
    if backup_path.is_symlink() or not backup_path.is_file(): raise UpdateError('Private dotenv backup is missing or invalid')
    backup=backup_path.read_bytes()
    if digest(backup)!=state['original_env_file_sha256']: raise UpdateError('Private dotenv backup hash differs')
    if original.get('Id')!=state['original_id']: raise UpdateError('Private original container identity differs')
    recorded=state.get('original_inspect_sha256')
    if recorded is not None and recorded!=digest(encoded(original)): raise UpdateError('Private original inspect hash differs')
    # Legacy v1 states have no inspect hash. Bind their old metadata to the
    # still-recorded tuned configuration; non-selected Env cannot change.
    before=env_values(original['Config']['Env']);after=env_values(state['new_env'])
    if not set(before)<=set(after): raise UpdateError('Private environment lost an original setting')
    for key in set(before)|set(after):
        if before.get(key)==after.get(key): continue
        if key not in ALLOWED or not isinstance(after.get(key),str) or not re.fullmatch(ALLOWED[key],after[key]):
            raise UpdateError('Private environment differs beyond selected performance settings')
    return target,raw,backup

def rollback_named_guard(engine, allowed):
    named=optional_inspect(engine,'ezlocalai')
    if named is not None and named['Id'] not in allowed: raise UpdateError('External ezlocalai container conflicts with rollback ownership')
    return named

def rollback_existing(engine, root, state, original, old, target, raw, backup):
    if old['Image']!=original['Image'] or old['Config']!=original['Config'] or old['HostConfig']!=original['HostConfig']:
        raise UpdateError('Original container changed; rollback refused')
    if old['Name'] not in ('/ezlocalai','/'+state['rollback_name']): raise UpdateError('Original container name changed; rollback refused')
    replacement=state.get('replacement_id')
    rollback_named_guard(engine,{old['Id'],replacement})
    if replacement:
        current=optional_inspect(engine,replacement)
        if current is not None:
            if current['Id']!=replacement or current['Name'] not in ('/ezlocalai','/'+state['temporary_name']):
                raise UpdateError('Owned replacement identity changed; rollback refused')
            engine.remove(replacement)
        state['replacement_id']=None; best_effort_json(root/'state.json',state)
    if old['Name']!='/ezlocalai': engine.rename(old['Id'],'ezlocalai')
    env_restored=True
    try:
        if digest(raw)!=state['original_env_file_sha256']: atomic(target,backup,state['env_mode'])
    except OSError: env_restored=False
    finally:
        if not engine.inspect(old['Id'])['State'].get('Running'): engine.start(old['Id'])
    healthy=wait_healthy(engine,old['Id'],state['timeout']);checks=verify(healthy,original,original['Config']['Env'])
    state['status']='rolled_back' if env_restored else 'container_restored_dotenv_incomplete'; best_effort_json(root/'state.json',state)
    proof={'status':state['status'],'original_container_restored':True,'original_configuration_restored':True,
           'original_container_recreated':False,'original_container_id_preserved':True,'restored_container_id':old['Id'],
           'healthy':True,'dotenv_restored':env_restored,'rollback_image':state['rollback_tag'],'run_directory':str(root),**checks}
    failure=root/'failure.json'
    try:
        if failure.exists(): proof['failure']=json.loads(failure.read_text())
    except (OSError,ValueError): pass
    best_effort_json(root/'result.json',proof)
    if not env_restored: raise UpdateError('Original container is healthy; dotenv restoration still needs recovery')
    return proof

def recreated_reference(original):
    result=copy.deepcopy(original)
    # Docker stores the supplied Image string in Config.Image. Resolving the
    # recorded immutable ID avoids changing a shared tag after a later pull.
    result['Config']['Image']=original['Image']
    return result

def verify_recreated(current, original):
    checks=verify(current,recreated_reference(original),original['Config']['Env'])
    checks['config_preserved_except_image_reference']=True
    checks['original_environment_preserved']=True
    checks['config_representation_normalizations']={}
    if original['Config']['Image']!=original['Image']:
        checks['config_representation_normalizations']['Image']={'original':original['Config']['Image'],'restored':original['Image'],
            'meaning':'Same recorded immutable image; shared image aliases are not changed'}
    return checks

def recreate_ownership(engine, state, original):
    replacement=state.get('replacement_id');applied=optional_inspect(engine,replacement) if replacement else None
    clone_id=state.get('rollback_recreated_id');clone=optional_inspect(engine,clone_id) if clone_id else None
    if applied is not None:
        if applied['Name'] not in ('/ezlocalai','/'+state.get('rollback_applied_name',''),'/'+state['temporary_name']):
            raise UpdateError('Owned tuned container name changed; rollback refused')
        verify(applied,original,state['new_env'])
    if clone is not None:
        if clone['Id'] in (state['original_id'],replacement) or clone['Name'] not in ('/ezlocalai','/'+state.get('rollback_recreate_name','')):
            raise UpdateError('Recreated rollback container ownership differs')
        verify_recreated(clone,original)
    rollback_named_guard(engine,{x['Id']for x in (applied,clone)if x is not None})
    return applied,clone

def recover_tuned_after_recreate_failure(engine, root, state, original, target, raw):
    clone_id=state.get('rollback_recreated_id');clone=optional_inspect(engine,clone_id) if clone_id else None
    applied_id=state.get('replacement_id');applied=optional_inspect(engine,applied_id) if applied_id else None
    if applied is None: raise UpdateError('Recorded tuned container is unavailable for recovery')
    verify(applied,original,state['new_env'])
    if applied['Name'] not in ('/ezlocalai','/'+state['rollback_applied_name']): raise UpdateError('Tuned recovery container name differs')
    rollback_named_guard(engine,{applied_id,clone_id})
    if clone is not None:
        if clone['Name'] not in ('/ezlocalai','/'+state['rollback_recreate_name']): raise UpdateError('Recreated container name differs during recovery')
        engine.remove(clone_id)
        if optional_inspect(engine,clone_id) is not None: raise UpdateError('Recreated GPU container still exists; tuned restart refused')
    state['rollback_recreated_id']=None;best_effort_json(root/'state.json',state)
    rollback_named_guard(engine,{applied_id})
    if applied['Name']!='/ezlocalai': engine.rename(applied_id,'ezlocalai')
    now=target.read_bytes()
    if digest(now) not in (state['original_env_file_sha256'],state['updated_env_file_sha256']): raise UpdateError('Dotenv changed externally during rollback recovery')
    dotenv_restored=True
    try:
        if now!=raw: atomic(target,raw,state['env_mode'])
    except OSError: dotenv_restored=False
    finally:
        if not engine.inspect(applied_id)['State'].get('Running'):engine.start(applied_id)
    current=wait_healthy(engine,applied_id,state['timeout']);verify(current,original,state['new_env'])
    state['status']='applied';state['rollback_phase']='failed_previous_tuned_restored';best_effort_json(root/'state.json',state)
    return {'previous_tuned_container_restored':True,'previous_tuned_container_id':applied_id,'healthy':True,
            'dotenv_restored_to_pre_rollback':dotenv_restored,'original_configuration_restored':False}

def rollback_recreate(engine, root, state, original, target, raw, backup):
    if state['status'] not in ('applied','applied_cleanup_required','rolled_back','rolled_back_cleanup_required','container_restored_dotenv_incomplete'):
        raise UpdateError('Missing original container was not finalized; recreation refused')
    applied,clone=recreate_ownership(engine,state,original)
    if engine.image(state['rollback_tag'])['Id']!=original['Image'] or engine.image(original['Image'])['Id']!=original['Image']:
        raise UpdateError('Retained rollback image differs or is unavailable')
    if 'rollback_recreate_name'not in state:
        state['rollback_recreate_name']='ezlocalai-env-restore-'+state['run_id']+'-'+secrets.token_hex(6)
        state['rollback_applied_name']='ezlocalai-env-retained-'+state['run_id']+'-'+secrets.token_hex(6)
        private_json(root/'state.json',state)
    for name,owned in [(state['rollback_recreate_name'],clone),(state['rollback_applied_name'],applied)]:
        occupant=optional_inspect(engine,name)
        if occupant is not None and (owned is None or occupant['Id']!=owned['Id']): raise UpdateError('Rollback private container name is occupied externally')
    created=False;disrupted=False;restored_verified=False
    try:
        if clone is None:
            configuration=recreated_reference(original)['Config'];state['rollback_phase']='creating_stopped_original';private_json(root/'state.json',state)
            try:clone_id=engine.create(state['rollback_recreate_name'],configuration,copy.deepcopy(original['HostConfig']),network_config(original))
            except DockerTransportError:
                uncertain=optional_inspect(engine,state['rollback_recreate_name'])
                if uncertain is None or uncertain['State'].get('Running'): raise
                verify_recreated(uncertain,original);clone_id=uncertain['Id']
            if clone_id in (state['original_id'],state.get('replacement_id')): raise UpdateError('New rollback container ID collides with an existing transaction container')
            state['rollback_recreated_id']=clone_id;created=True;private_json(root/'state.json',state)
            clone=engine.inspect(clone_id)
        verify_recreated(clone,original)
        if clone['Name']!='/ezlocalai' and clone['State'].get('Running'): raise UpdateError('Stopped rollback preparation unexpectedly started')
        if target.read_bytes()!=raw: raise UpdateError('Dotenv changed during rollback preparation')
        applied,clone=recreate_ownership(engine,state,original)
        state['rollback_phase']='cutover_to_recreated_original';private_json(root/'state.json',state)
        if applied is not None:
            disrupted=True
            if applied['State'].get('Running'):engine.stop(applied['Id'])
            if applied['Name']!='/'+state['rollback_applied_name']:engine.rename(applied['Id'],state['rollback_applied_name'])
        rollback_named_guard(engine,{clone['Id']})
        if clone['Name']!='/ezlocalai':engine.rename(clone['Id'],'ezlocalai')
        disrupted=True
        if raw!=backup:atomic(target,backup,state['env_mode'])
        if not engine.inspect(clone['Id'])['State'].get('Running'):engine.start(clone['Id'])
        healthy=wait_healthy(engine,clone['Id'],state['timeout']);checks=verify_recreated(healthy,original)
        if healthy['Name']!='/ezlocalai' or target.read_bytes()!=backup:raise UpdateError('Recreated rollback identity or dotenv differs')
        restored_verified=True;state['status']='rolled_back_cleanup_required';state['rollback_phase']='original_configuration_verified';private_json(root/'state.json',state)
        if applied is not None:
            retained=optional_inspect(engine,applied['Id'])
            if retained is not None:
                if retained['State'].get('Running') or retained['Name']!='/'+state['rollback_applied_name']:raise UpdateError('Retained tuned container state changed before cleanup')
                verify(retained,original,state['new_env'])
                try:engine.remove(retained['Id'])
                except DockerTransportError:
                    if optional_inspect(engine,retained['Id']) is not None:raise
        state['replacement_id']=None;state['status']='rolled_back';state['rollback_phase']='complete';private_json(root/'state.json',state)
        proof={'status':'rolled_back','original_container_restored':False,'original_configuration_restored':True,
               'original_container_recreated':True,'original_container_id_preserved':False,'original_container_id':state['original_id'],
               'restored_container_id':clone['Id'],'healthy':True,'dotenv_restored':True,'retained_tuned_container_removed':True,
               'rollback_image':state['rollback_tag'],'run_directory':str(root),**checks}
        private_json(root/'result.json',proof);return proof
    except BaseException as failure:
        with cleanup_signals():
            proof={'status':'rollback_recreation_failed','failure_type':type(failure).__name__,
                   'failure':str(failure)if isinstance(failure,UpdateError)else'Rollback recreation exception','run_directory':str(root)}
            if restored_verified:
                # Never remove the only healthy restored original after tuned
                # cleanup (including a lost DELETE response or a full disk).
                proof.update(status='rolled_back_cleanup_required',original_configuration_restored=True,original_container_recreated=True,
                    original_container_id_preserved=False,restored_container_id=state.get('rollback_recreated_id'),healthy=True,dotenv_restored=True)
            elif disrupted:
                try:proof.update(recover_tuned_after_recreate_failure(engine,root,state,original,target,raw))
                except BaseException as recovery:proof.update(status='manual_recovery_required',recovery_failure_type=type(recovery).__name__)
            elif created:
                # Preparation failed while the tuned service stayed running.
                try:
                    candidate=optional_inspect(engine,state.get('rollback_recreated_id'))
                    if candidate is not None:
                        if candidate['Name']!='/'+state['rollback_recreate_name']:raise UpdateError('Prepared rollback ownership changed')
                        engine.remove(candidate['Id'])
                    state['rollback_recreated_id']=None;best_effort_json(root/'state.json',state)
                except BaseException:proof['stopped_preparation_cleanup_required']=True
            best_effort_json(root/'rollback-recreation-result.json',proof)
        raise

def rollback(engine, root, state, original):
    target,raw,backup=rollback_inputs(root,state,original)
    old=optional_inspect(engine,state['original_id'])
    if old is not None:return rollback_existing(engine,root,state,original,old,target,raw,backup)
    return rollback_recreate(engine,root,state,original,target,raw,backup)

def finalize(engine, root, state, original):
    if state['status'] not in ('awaiting_router_verification','applied_cleanup_required'): raise UpdateError('Run is not awaiting finalization')
    if digest(pathlib.Path(state['env_file']).read_bytes())!=state['updated_env_file_sha256']:
        raise UpdateError('Dotenv changed externally before finalization')
    current=wait_healthy(engine,state['replacement_id'],state['timeout']); checks=verify(current,original,state['new_env'])
    if current['Name']!='/ezlocalai': raise UpdateError('Replacement container name changed')
    old=optional_inspect(engine,state['original_id'])
    if old is None and state['status']!='applied_cleanup_required': raise UpdateError('Original rollback container disappeared before finalization')
    if old is not None:
        if old['State'].get('Running') or old['Name']!='/'+state['rollback_name']: raise UpdateError('Original rollback container state changed')
        if old['Config']!=original['Config'] or old['HostConfig']!=original['HostConfig'] or old['Image']!=original['Image']:
            raise UpdateError('Original rollback container configuration changed')
    state['status']='applied_cleanup_required'; private_json(root/'state.json',state)
    proof=json.loads((root/'result.json').read_text()); proof['status']=state['status']; private_json(root/'result.json',proof)
    if old is not None: engine.remove(state['original_id'])
    state['status']='applied'; private_json(root/'state.json',state)
    proof=json.loads((root/'result.json').read_text()); proof.update({'status':'applied','original_stopped_container_removed':True,**checks})
    private_json(root/'result.json',proof); return proof

def apply(engine, plan, dry_run=False, retain=False):
    run_id=plan.get('run_id','')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,39}',run_id): raise UpdateError('run_id must be2..40 lowercase letters/digits/hyphens')
    changes=validate_changes(plan.get('changes')); target=pathlib.Path(plan.get('env_file','/home/josh/ezlocalai/.env'))
    if target.is_symlink() or not target.is_file(): raise UpdateError('Dotenv must be an existing regular file without symlink')
    original=engine.inspect('ezlocalai')
    if plan.get('expected_original_id') is not None and original['Id'] != plan['expected_original_id']: raise UpdateError('Original container changed after qualified preflight')
    if plan.get('expected_image') is not None and original['Image'] != plan['expected_image']: raise UpdateError('Original image changed after qualification')
    if not original['State'].get('Running') or original['State'].get('Health',{}).get('Status')!='healthy': raise UpdateError('Original must be running and healthy')
    host=original['HostConfig']
    if host.get('AutoRemove') or host.get('ContainerIDFile') or host.get('Links') or host.get('VolumesFrom'):
        raise UpdateError('Unsupported Docker lifecycle/network configuration')
    networking=network_config(original)
    if networking is not None:
        network=engine.network(host['NetworkMode'])
        if network.get('Driver')!='bridge' or network.get('Scope')!='local' or network.get('Ingress'):
            raise UpdateError('Only local custom bridge networking is qualified')
    if not original.get('Mounts') or any(m.get('Type')!='bind' for m in original['Mounts']): raise UpdateError('Only existing bind mounts are supported')
    if engine.image(original['Config']['Image'])['Id']!=original['Image']: raise UpdateError('Original image tag moved; cannot preserve Config.Image safely')
    old_values=env_values(original['Config']['Env'])
    for key in changes:
        if key in old_values and not (key == 'LLM_PROMPT_CACHE_MAX_RAM_FRACTION' and old_values[key] == '') and not re.fullmatch(ALLOWED[key],old_values[key]): raise UpdateError('Existing selected container setting has unsupported syntax: '+key)
    new_env=[k+'='+changes.get(k,v) for k,v in old_values.items()]
    new_env += [k+'='+v for k,v in changes.items() if k not in old_values]
    raw=target.read_bytes(); rendered,file_old=render_env(raw,changes)
    proof={'status':'dry_run' if dry_run else 'prepared','changes':{k:{'container_old':old_values.get(k),'file_old':file_old[k],'new':v} for k,v in changes.items()},'image':original['Image'],'container_name':'ezlocalai'}
    if dry_run: return proof
    root=pathlib.Path(plan.get('output_dir','/tmp/ezlocalai-env-update-'+run_id)); root.mkdir(mode=0o700,parents=False,exist_ok=False)
    state={'status':'prepared','run_id':run_id,'env_file':str(target),'env_mode':stat.S_IMODE(target.stat().st_mode),'original_id':original['Id'],'original_inspect_sha256':digest(encoded(original)),'replacement_id':None,'rollback_name':'ezlocalai-env-rollback-'+run_id+'-'+secrets.token_hex(6),'temporary_name':'ezlocalai-env-new-'+run_id+'-'+secrets.token_hex(6),'rollback_tag':'ezlocalai:rollback-env-'+run_id,'original_env_file_sha256':digest(raw),'updated_env_file_sha256':digest(rendered),'new_env':new_env,'timeout':int(plan.get('timeout',180))}
    if not 30<=state['timeout']<=1800: raise UpdateError('timeout out of range')
    private_json(root/'original-inspect.json',original); atomic(root/'dotenv.backup',raw); private_json(root/'state.json',state)
    proof.update({'run_directory':str(root),'rollback_image':state['rollback_tag'],'backup_file':str(root/'dotenv.backup')}); private_json(root/'result.json',proof)
    committed=False; phase='preparing'
    try:
        if optional_inspect(engine,state['temporary_name']) is not None or optional_inspect(engine,state['rollback_name']) is not None:
            raise UpdateError('Transaction container name already exists')
        phase='tag_rollback_image'; engine.tag(original['Image'],state['rollback_tag'])
        config=copy.deepcopy(original['Config']); config['Env']=new_env
        phase='create_stopped_replacement'
        try:
            state['replacement_id']=engine.create(state['temporary_name'],config,copy.deepcopy(host),networking)
        except DockerTransportError:
            # A transport failure can hide a successful create. Adopt only the
            # exact requested config under this transaction's unique name.
            try:
                candidate=engine.inspect(state['temporary_name']); verify(candidate,original,new_env)
                state['replacement_id']=candidate['Id']; private_json(root/'state.json',state)
            except BaseException: pass
            raise
        private_json(root/'state.json',state)
        phase='verify_stopped_replacement'; verify(engine.inspect(state['replacement_id']),original,new_env)
        now=engine.inspect(original['Id'])
        if now['Config']!=original['Config'] or now['HostConfig']!=host or not now['State'].get('Running'): raise UpdateError('Original changed during preparation')
        if target.read_bytes()!=raw: raise UpdateError('Dotenv changed during preparation')
        phase='write_selected_dotenv'; atomic(target,rendered,state['env_mode'])
        phase='stop_original'; engine.stop(original['Id'])
        phase='rename_original'; engine.rename(original['Id'],state['rollback_name'])
        phase='rename_replacement'; engine.rename(state['replacement_id'],'ezlocalai')
        phase='start_replacement'; engine.start(state['replacement_id'])
        phase='wait_healthy'; current=wait_healthy(engine,state['replacement_id'],state['timeout'])
        phase='verify_running_replacement'; checks=verify(current,original,new_env)
        state['status']='awaiting_router_verification'; private_json(root/'state.json',state)
        proof.update({'status':state['status'],'healthy':True,**checks,'original_stopped_container_retained':True}); private_json(root/'result.json',proof)
        if retain: return proof
        committed=True; phase='finalize'
        return finalize(engine,root,state,original)
    except BaseException as failure:
        with cleanup_signals():
            info={'phase':phase,'type':type(failure).__name__,'reason':str(failure) if isinstance(failure,UpdateError) else 'Unexpected '+type(failure).__name__+' during transaction'}
            best_effort_json(root/'failure.json',info); proof['failure']=info; best_effort_json(root/'result.json',proof)
            if not committed:
                try:
                    with cleanup_signals(): rollback(engine,root,state,original)
                except BaseException:
                    proof.update({'status':'rollback_required','detail':'Automatic rollback did not finish; inspect private run state and original container.'}); best_effort_json(root/'result.json',proof)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=['apply','finalize','rollback']); parser.add_argument('path'); parser.add_argument('--dry-run',action='store_true'); parser.add_argument('--retain-original',action='store_true'); args=parser.parse_args()
    if args.action!='apply' and (args.dry_run or args.retain_original): parser.error('apply-only option')
    engine=Docker()
    def interrupted(signum,frame): raise UpdateError('Interrupted by signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupted)
    plan=json.loads(pathlib.Path(args.path).read_text()) if args.action=='apply' else None
    env_path=pathlib.Path(plan.get('env_file','/home/josh/ezlocalai/.env')) if plan is not None else pathlib.Path(json.loads((pathlib.Path(args.path)/'state.json').read_text())['env_file'])
    if args.dry_run:
        print(json.dumps(apply(engine,plan,True,args.retain_original),indent=2)); return
    lock=os.open(env_path.parent/'.ezlocalai-env-update.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise UpdateError('Another environment update holds the local lock') from None
        if args.action=='apply': proof=apply(engine,plan,args.dry_run,args.retain_original)
        else:
            root,state,original=load_state(args.path)
            if args.action=='finalize': proof=finalize(engine,root,state,original)
            else:
                with cleanup_signals(): proof=rollback(engine,root,state,original)
        print(json.dumps(proof,indent=2))
    finally: os.close(lock)
if __name__=='__main__':
    try: main()
    except (UpdateError,ValueError,KeyError,OSError) as error:
        # Do not emit Docker responses, config, dotenv contents, or raw OS errors.
        message=str(error) if isinstance(error,UpdateError) else type(error).__name__+' while reading operator plan/state'
        print(json.dumps({'status':'failed','error':message}),file=sys.stderr); sys.exit(1)
