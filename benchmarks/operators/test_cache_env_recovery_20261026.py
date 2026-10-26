"""Portable synthetic recovery guards; no Docker, SSH or real credentials.

All Docker objects, identifiers and environment values below are test fixtures.
File writes stay within unittest TemporaryDirectory instances.
"""
import copy,importlib.util,json,pathlib,tempfile,unittest
from unittest import mock
spec=importlib.util.spec_from_file_location('cache_env_recovery_fixture',pathlib.Path(__file__).with_name('cache_env_recovery_20261026.py')); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def setUpModule():
    global docker_request_guard
    docker_request_guard=mock.patch.object(mod.Docker,'request',side_effect=AssertionError('Offline fixtures must not contact Docker'))
    docker_request_guard.start()


def tearDownModule():
    docker_request_guard.stop()

class FakeDocker:
    def __init__(self):
        self.original={'Id':'original-id','Name':'/ezlocalai','Image':'sha256:fixed','Config':{'Hostname':'originalhost','Image':'registry/ezlocalai:latest','Env':['LLM_UBATCH_SIZE=auto','KEY=fixture-secret-marker-api','GGML_CUDA_GRAPH_OPT=0'],'Labels':{'com.docker.compose.service':'ezlocalai'},'User':'1000','Cmd':['serve'],'Healthcheck':{'Test':['CMD','check']}},'HostConfig':{'NetworkMode':'host','AutoRemove':False,'ContainerIDFile':'','Links':None,'VolumesFrom':None,'Binds':['/models:/app/models:rw'],'Memory':32212254720,'Ulimits':[{'Name':'memlock','Soft':-1,'Hard':-1}],'DeviceRequests':[{'Driver':'nvidia','Capabilities':[['gpu']],'Count':-1}],'RestartPolicy':{'Name':'unless-stopped','MaximumRetryCount':0}},'Mounts':[{'Type':'bind','Source':'/models','Destination':'/app/models','Mode':'rw','RW':True,'Propagation':'rprivate'}],'State':{'Running':True,'OOMKilled':False,'Health':{'Status':'healthy'}}}
        self.items={'original-id':copy.deepcopy(self.original)};self.calls=[];self.bad_health=False;self.bad_config=False;self.ambiguous_create=False;self.fail_rename=False;self.moved=False
    def inspect(self,key):
        for obj in self.items.values():
            if key in (obj['Id'],obj['Name'],obj['Name'].lstrip('/')):return copy.deepcopy(obj)
        raise mod.DockerHTTPError(404,'Docker request failed HTTP404')
    def network(self,name):return {'Driver':'bridge','Scope':'local','Ingress':False}
    def image(self,key):return {'Id':'sha256:other' if self.moved else 'sha256:fixed'}
    def tag(self,image,tag):self.calls.append(('tag',image,tag))
    def create(self,name,config,host,networking=None):
        self.calls.append(('create',name));obj=copy.deepcopy(self.original);obj.update({'Id':'new-id','Name':'/'+name,'Config':copy.deepcopy(config),'HostConfig':copy.deepcopy(host)});obj['State']={'Running':False,'OOMKilled':False,'Health':{'Status':'starting'}}
        if networking is not None:
            obj['NetworkSettings']={**copy.deepcopy(obj.get('NetworkSettings',{})),'Networks':{key:{**copy.deepcopy(endpoint),'NetworkID':'network-id','EndpointID':'new-endpoint','IPAddress':'172.22.0.3','MacAddress':'02:42:ac:16:00:03'} for key,endpoint in networking['EndpointsConfig'].items()}}
        if self.bad_config:obj['HostConfig']['Memory']=1
        self.items['new-id']=obj
        if self.ambiguous_create:raise mod.DockerTransportError('Docker transport failure during create')
        return 'new-id'
    def stop(self,key):self.calls.append(('stop',key));self.items[key]['State']['Running']=False
    def start(self,key):
        self.calls.append(('start',key));self.items[key]['State']['Running']=True;self.items[key]['State']['Health']['Status']='unhealthy' if key=='new-id' and self.bad_health else 'healthy'
    def rename(self,key,name):
        self.calls.append(('rename',key,name))
        if self.fail_rename and key=='original-id': self.fail_rename=False;raise mod.UpdateError('Rename failed')
        if any(v['Name']=='/'+name and k!=key for k,v in self.items.items()):raise mod.UpdateError('Name conflict')
        self.items[key]['Name']='/'+name
    def remove(self,key):self.calls.append(('remove',key));del self.items[key]

class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.dir=pathlib.Path(self.temp.name);self.env=self.dir/'.env';self.raw=b'# unmodified\r\nLLM_UBATCH_SIZE="auto" # tune\r\nAPI_KEY=fixture-secret-marker-env\r\nOTHER=unchanged\r\n';self.env.write_bytes(self.raw);self.env.chmod(0o640);self.engine=FakeDocker();self.plan={'run_id':'test-20261026','changes':{'LLM_UBATCH_SIZE':'1024'},'env_file':str(self.env),'output_dir':str(self.dir/'proof'),'timeout':30}
    def state(self):return mod.load_state(self.plan['output_dir'])
    def assert_restored(self):
        self.assertEqual(set(self.engine.items),{'original-id'});self.assertEqual(self.engine.inspect('ezlocalai'),self.engine.original);self.assertEqual(self.env.read_bytes(),self.raw)
    def bridge(self):
        obj=self.engine.items['original-id'];obj['HostConfig']['NetworkMode']='ezlocalai_default';obj['HostConfig']['PortBindings']={'8091/tcp':[{'HostIp':'','HostPort':'8091'}]};obj['Config']['ExposedPorts']={'8091/tcp':{}};obj['NetworkSettings']={'Ports':{'8091/tcp':[{'HostIp':'0.0.0.0','HostPort':'8091'},{'HostIp':'::','HostPort':'8091'}]},'Networks':{'ezlocalai_default':{'IPAMConfig':None,'Links':None,'Aliases':['ezlocalai','ezlocalai'],'DriverOpts':None,'GwPriority':0,'NetworkID':'network-id','EndpointID':'old-endpoint','IPAddress':'172.22.0.2','MacAddress':'02:42:ac:16:00:02'}}};self.engine.original=copy.deepcopy(obj)
    def test_simple_bridge_preserves_aliases_and_portbindings_with_new_dynamic_identity(self):
        self.bridge();proof=mod.apply(self.engine,self.plan,retain=True);self.assertTrue(proof['static_network_configuration_preserved']);new=self.engine.inspect('ezlocalai');old=self.engine.original;self.assertEqual(new['HostConfig'],old['HostConfig']);self.assertEqual(new['Config']['ExposedPorts'],old['Config']['ExposedPorts']);endpoint=new['NetworkSettings']['Networks']['ezlocalai_default'];self.assertEqual(endpoint['Aliases'],['ezlocalai','ezlocalai']);self.assertNotEqual(endpoint['IPAddress'],old['NetworkSettings']['Networks']['ezlocalai_default']['IPAddress']);self.assertNotEqual(endpoint['MacAddress'],old['NetworkSettings']['Networks']['ezlocalai_default']['MacAddress']);self.assertNotEqual(endpoint['EndpointID'],old['NetworkSettings']['Networks']['ezlocalai_default']['EndpointID']);root,state,original=self.state();mod.rollback(self.engine,root,state,original);self.assert_restored()
    def test_simple_bridge_finalize_preserves_port_mapping(self):
        self.bridge();proof=mod.apply(self.engine,self.plan);self.assertEqual(proof['status'],'applied');self.assertEqual(self.engine.inspect('ezlocalai')['HostConfig']['PortBindings'],{'8091/tcp':[{'HostIp':'','HostPort':'8091'}]})
    def test_complex_bridge_forms_refuse_before_any_mutation(self):
        for kind in ['static_ip','multiple_networks','priority','links','driveropts','driver','zeroport','emptyport','allports']:
            with self.subTest(kind=kind):
                self.engine=FakeDocker();self.bridge();obj=self.engine.items['original-id'];endpoint=obj['NetworkSettings']['Networks']['ezlocalai_default']
                if kind=='static_ip':endpoint['IPAMConfig']={'IPv4Address':'172.22.0.2'}
                elif kind=='multiple_networks':obj['NetworkSettings']['Networks']['another']={}
                elif kind=='priority':endpoint['GwPriority']=1
                elif kind=='links':endpoint['Links']=['other:alias']
                elif kind=='driveropts':endpoint['DriverOpts']={'custom':'setting'}
                elif kind=='zeroport':obj['HostConfig']['PortBindings']['8091/tcp'][0]['HostPort']='0'
                elif kind=='emptyport':obj['HostConfig']['PortBindings']['8091/tcp'][0]['HostPort']=''
                elif kind=='allports':obj['HostConfig']['PublishAllPorts']=True
                else:self.engine.network=lambda _: {'Driver':'overlay','Scope':'swarm','Ingress':False}
                with self.assertRaises(mod.UpdateError):mod.apply(self.engine,self.plan)
                self.assertEqual(self.engine.calls,[])
    def test_running_bridge_network_identity_mismatch_rolls_back(self):
        self.bridge();start=self.engine.start
        def changed_start(ident):
            start(ident)
            if ident=='new-id':self.engine.items[ident]['NetworkSettings']['Networks']['ezlocalai_default']['NetworkID']='wrong-network'
        self.engine.start=changed_start
        with self.assertRaisesRegex(mod.UpdateError,'different Docker network'):mod.apply(self.engine,self.plan)
        self.assert_restored()
    def test_running_bridge_effective_ports_mismatch_rolls_back(self):
        self.bridge();start=self.engine.start
        def changed_start(ident):
            start(ident)
            if ident=='new-id':self.engine.items[ident]['NetworkSettings']['Ports']['8091/tcp'][0]['HostPort']='8092'
        self.engine.start=changed_start
        with self.assertRaisesRegex(mod.UpdateError,'effective published ports'):mod.apply(self.engine,self.plan)
        self.assert_restored()
    def test_bridge_alias_loss_is_detected_before_stopping_original(self):
        self.bridge();create=self.engine.create
        def changed_create(*args,**kwargs):
            ident=create(*args,**kwargs);self.engine.items[ident]['NetworkSettings']['Networks']['ezlocalai_default']['Aliases']=['different'];return ident
        self.engine.create=changed_create
        with self.assertRaisesRegex(mod.UpdateError,'endpoint configuration'):mod.apply(self.engine,self.plan)
        self.assert_restored();self.assertNotIn(('stop','original-id'),self.engine.calls)
    def test_dry_run_has_no_writes_or_mutating_docker_calls(self):
        proof=mod.apply(self.engine,self.plan,dry_run=True);self.assertEqual(proof['status'],'dry_run');self.assertEqual(self.env.read_bytes(),self.raw);self.assertFalse((self.dir/'proof').exists());self.assertEqual(self.engine.calls,[]);self.assertNotIn('fixture-secret-marker',json.dumps(proof))
    def test_applied_preserves_entire_config_host_mounts_except_selected_env(self):
        proof=mod.apply(self.engine,self.plan);self.assertEqual(proof['status'],'applied');self.assertEqual(set(self.engine.items),{'new-id'});new=self.engine.inspect('ezlocalai');expected=copy.deepcopy(self.engine.original);expected['Config']['Env'][0]='LLM_UBATCH_SIZE=1024';self.assertEqual(new['Config'],expected['Config']);self.assertEqual(new['HostConfig'],expected['HostConfig']);self.assertEqual(new['Mounts'],expected['Mounts']);self.assertEqual(new['Image'],expected['Image']);self.assertIn(b'API_KEY=fixture-secret-marker-env\r\nOTHER=unchanged\r\n',self.env.read_bytes());self.assertNotIn('fixture-secret-marker',json.dumps(proof));self.assertEqual((self.dir/'proof').stat().st_mode&0o777,0o700);self.assertEqual((self.dir/'proof'/'dotenv.backup').stat().st_mode&0o777,0o600);self.assertEqual(self.env.stat().st_mode&0o777,0o640)
    def test_retained_original_then_finalize(self):
        proof=mod.apply(self.engine,self.plan,retain=True);self.assertEqual(proof['status'],'awaiting_router_verification');self.assertEqual(len(self.engine.items),2);self.assertFalse(self.engine.items['original-id']['State']['Running']);root,state,original=self.state();proof=mod.finalize(self.engine,root,state,original);self.assertEqual(proof['status'],'applied');self.assertEqual(set(self.engine.items),{'new-id'})
    def test_retained_original_can_roll_back_byte_exact_dotenv(self):
        mod.apply(self.engine,self.plan,retain=True);root,state,original=self.state();proof=mod.rollback(self.engine,root,state,original);self.assertEqual(proof['status'],'rolled_back');self.assert_restored()
    def test_failed_replacement_health_restores_original_and_file(self):
        self.engine.bad_health=True
        with self.assertRaisesRegex(mod.UpdateError,'unhealthy'):mod.apply(self.engine,self.plan)
        self.assert_restored();self.assertEqual(json.loads((self.dir/'proof'/'result.json').read_text())['status'],'rolled_back')
    def test_invalid_roundtrip_never_stops_original_and_removes_owned_new(self):
        self.engine.bad_config=True
        with self.assertRaisesRegex(mod.UpdateError,'HostConfig'):mod.apply(self.engine,self.plan)
        self.assert_restored();self.assertNotIn(('stop','original-id'),self.engine.calls)
    def test_failed_rename_after_stop_rolls_back(self):
        self.engine.fail_rename=True
        with self.assertRaisesRegex(mod.UpdateError,'Rename'):mod.apply(self.engine,self.plan)
        self.assert_restored()
    def test_ambiguous_create_recovers_only_exact_owned_container_then_cleans(self):
        self.engine.ambiguous_create=True
        with self.assertRaisesRegex(mod.UpdateError,'transport'):mod.apply(self.engine,self.plan)
        self.assert_restored();self.assertNotIn(('stop','original-id'),self.engine.calls)
    def test_moved_image_tag_rejected_before_any_mutation(self):
        self.engine.moved=True
        with self.assertRaisesRegex(mod.UpdateError,'image tag moved'):mod.apply(self.engine,self.plan)
        self.assertEqual(self.engine.calls,[]);self.assertFalse((self.dir/'proof').exists());self.assert_restored()
    def test_non_host_and_non_bind_are_rejected(self):
        for kind in ['network','mount']:
            with self.subTest(kind=kind):
                engine=FakeDocker()
                if kind=='network':engine.items['original-id']['HostConfig']['NetworkMode']='bridge'
                else:engine.items['original-id']['Mounts'][0]['Type']='volume'
                with self.assertRaises(mod.UpdateError):mod.apply(engine,self.plan)
                self.assertEqual(engine.calls,[])
    def test_external_dotenv_edit_is_not_clobbered_by_later_rollback(self):
        mod.apply(self.engine,self.plan,retain=True);self.env.write_bytes(self.env.read_bytes()+b'EXTERNAL=keep\n');root,state,original=self.state()
        with self.assertRaisesRegex(mod.UpdateError,'changed externally'):mod.rollback(self.engine,root,state,original)
        self.assertIn(b'EXTERNAL=keep',self.env.read_bytes());self.assertEqual(self.engine.inspect('ezlocalai')['Id'],'new-id')
    def test_unsupported_or_duplicate_selected_setting_rejected_without_secrets(self):
        self.env.write_bytes(self.raw+b'LLM_UBATCH_SIZE=2048\n')
        with self.assertRaisesRegex(mod.UpdateError,'duplicated'):mod.apply(self.engine,self.plan)
        self.assertEqual(self.engine.calls,[])
    def test_rollback_retries_when_owned_replacement_was_already_removed(self):
        mod.apply(self.engine,self.plan,retain=True);root,state,original=self.state();self.engine.remove(state['replacement_id']);proof=mod.rollback(self.engine,root,state,original);self.assertEqual(proof['status'],'rolled_back');self.assert_restored()
    def test_finalize_retries_when_old_delete_succeeded_but_response_was_lost(self):
        mod.apply(self.engine,self.plan,retain=True);root,state,original=self.state();state['status']='applied_cleanup_required';mod.private_json(root/'state.json',state);self.engine.remove(state['original_id']);proof=mod.finalize(self.engine,root,state,original);self.assertEqual(proof['status'],'applied');self.assertEqual(set(self.engine.items),{'new-id'})
    def test_docker29_nullable_oom_default_normalization_is_preserved(self):
        self.engine.items['original-id']['HostConfig']['OomKillDisable']=None;self.engine.original=copy.deepcopy(self.engine.items['original-id']);create=self.engine.create
        def normalized_create(*args,**kwargs):
            ident=create(*args,**kwargs);self.engine.items[ident]['HostConfig']['OomKillDisable']=False;return ident
        self.engine.create=normalized_create;proof=mod.apply(self.engine,self.plan);self.assertEqual(proof['status'],'applied');self.assertEqual(proof['hostconfig_default_normalizations']['OomKillDisable'],{'original':None,'replacement':False,'meaning':'OOM killer remains enabled'});self.assertEqual(mod.hostconfig_view(self.engine.inspect('ezlocalai')['HostConfig']),mod.hostconfig_view(self.engine.original['HostConfig']))
    def test_oom_true_or_nonbool_drift_still_refuses_before_stop_and_retains_reason(self):
        for changed in [True,0]:
            with self.subTest(changed=changed):
                self.engine=FakeDocker();self.engine.items['original-id']['HostConfig']['OomKillDisable']=None;self.engine.original=copy.deepcopy(self.engine.items['original-id']);self.plan['output_dir']=str(self.dir/('proof-'+str(changed)));create=self.engine.create
                def changed_create(*args,**kwargs):
                    ident=create(*args,**kwargs);self.engine.items[ident]['HostConfig']['OomKillDisable']=changed;return ident
                self.engine.create=changed_create
                with self.assertRaisesRegex(mod.UpdateError,'HostConfig differs: OomKillDisable'):mod.apply(self.engine,self.plan)
                self.assert_restored();self.assertNotIn(('stop','original-id'),self.engine.calls);proof=json.loads((pathlib.Path(self.plan['output_dir'])/'result.json').read_text());self.assertEqual(proof['failure']['phase'],'verify_stopped_replacement');self.assertEqual(proof['failure']['reason'],'Replacement HostConfig differs: OomKillDisable');self.assertNotIn('fixture-secret-marker',json.dumps(proof))
    def test_failure_logging_disk_full_never_skips_original_container_restoration(self):
        self.engine.bad_health=True;original_writer=mod.private_json;broken=False
        def failing_json(path,value):
            nonlocal broken
            if self.engine.items.get('new-id',{}).get('State',{}).get('Running'):broken=True
            if broken:raise OSError('injected disk full')
            return original_writer(path,value)
        with mock.patch.object(mod,'private_json',side_effect=failing_json):
            with self.assertRaisesRegex(mod.UpdateError,'unhealthy'):mod.apply(self.engine,self.plan)
        self.assert_restored()
    def test_dotenv_restore_io_failure_still_restarts_original_service(self):
        self.engine.bad_health=True;original_atomic=mod.atomic;failed_once=False
        def failing_restore(path,data,mode=0o600):
            nonlocal failed_once
            if pathlib.Path(path)==self.env and data==self.raw and not failed_once:
                failed_once=True;raise OSError('injected disk full')
            return original_atomic(path,data,mode)
        with mock.patch.object(mod,'atomic',side_effect=failing_restore):
            with self.assertRaisesRegex(mod.UpdateError,'unhealthy'):mod.apply(self.engine,self.plan)
        restored=self.engine.inspect('ezlocalai');self.assertEqual(restored['Id'],'original-id');self.assertTrue(restored['State']['Running']);self.assertEqual(restored['Config'],self.engine.original['Config']);self.assertNotEqual(self.env.read_bytes(),self.raw)
        root,state,original=self.state();mod.rollback(self.engine,root,state,original);self.assert_restored()
    def test_append_setting_preserves_other_bytes(self):
        rendered,old=mod.render_env(b'KEY=fixture-secret-marker-generic',{'GGML_CUDA_GRAPH_OPT':'1'});self.assertEqual(rendered,b'KEY=fixture-secret-marker-generic\nGGML_CUDA_GRAPH_OPT=1\n');self.assertEqual(old,{'GGML_CUDA_GRAPH_OPT':None})
    def test_multiline_other_variable_cannot_be_accidentally_edited(self):
        self.env.write_bytes(b'CERT="start\nLLM_UBATCH_SIZE=auto\nend"\n')
        with self.assertRaisesRegex(mod.UpdateError,'Multiline'):mod.apply(self.engine,self.plan)
        self.assertEqual(self.engine.calls,[])
    def test_no_other_env_key_is_permitted(self):
        self.plan['changes']={'API_KEY':'new-secret'}
        with self.assertRaisesRegex(mod.UpdateError,'Unsupported'):mod.apply(self.engine,self.plan)
        self.assertEqual(self.engine.calls,[])
class FractionUpdateTests(unittest.TestCase):
    state=UpdateTests.state
    assert_restored=UpdateTests.assert_restored
    def setUp(self):
        UpdateTests.setUp(self)
        self.plan['changes']={'LLM_PROMPT_CACHE_MAX_RAM_FRACTION':'0.4'}
        self.raw=b'# retained\r\nLLM_PROMPT_CACHE_MAX_RAM_FRACTION="0.25" # previous\r\nLLM_UBATCH_SIZE=1024\r\nAPI_KEY=fixture-secret-marker-env\r\n'
        self.env.write_bytes(self.raw)
        self.engine.items['original-id']['Config']['Env'] += ['LLM_PROMPT_CACHE_MAX_RAM_FRACTION=0.25','KV_CACHE_TYPE=q4_0','LLM_MAX_TOKENS=230000']
        self.engine.original=copy.deepcopy(self.engine.items['original-id'])
    def test_fraction_changes_only_selected_value_and_can_rollback(self):
        proof=mod.apply(self.engine,self.plan,retain=True)
        actual=self.engine.inspect('ezlocalai');expected=copy.deepcopy(self.engine.original['Config'])
        expected['Env']=[v.replace('LLM_PROMPT_CACHE_MAX_RAM_FRACTION=0.25','LLM_PROMPT_CACHE_MAX_RAM_FRACTION=0.4')for v in expected['Env']]
        self.assertEqual(actual['Config'],expected)
        self.assertIn(b'LLM_UBATCH_SIZE=1024\r\nAPI_KEY=fixture-secret-marker-env\r\n',self.env.read_bytes())
        self.assertNotIn('fixture-secret-marker',json.dumps(proof))
        root,state,original=self.state();mod.rollback(self.engine,root,state,original);self.assert_restored()
    def test_fraction_empty_default_is_supported_without_other_key_broadening(self):
        self.env.write_bytes(b'LLM_PROMPT_CACHE_MAX_RAM_FRACTION= # default\nOTHER=unchanged\n')
        self.engine.items['original-id']['Config']['Env']=[v if not v.startswith('LLM_PROMPT_CACHE_MAX_RAM_FRACTION=')else 'LLM_PROMPT_CACHE_MAX_RAM_FRACTION='for v in self.engine.items['original-id']['Config']['Env']]
        self.engine.original=copy.deepcopy(self.engine.items['original-id'])
        proof=mod.apply(self.engine,self.plan,dry_run=True);self.assertEqual(proof['changes']['LLM_PROMPT_CACHE_MAX_RAM_FRACTION']['file_old'],'')
    def test_fraction_domain_rejects_nonfinite_exponent_out_of_range_injection(self):
        for value in ['NaN','inf','Infinity','-0.1','1.1','2','0.4;echo bad','4e-1','0.1234567','0.4\nAPI_KEY=bad','auto']:
            with self.subTest(value=value),self.assertRaises(mod.UpdateError):mod.validate_changes({'LLM_PROMPT_CACHE_MAX_RAM_FRACTION':value})
        for key in ['LLM_UBATCH_SIZE','LLM_BATCH_SIZE','GGML_CUDA_GRAPH_OPT']:
            with self.subTest(key=key),self.assertRaises(mod.UpdateError):mod.validate_changes({key:'0.4'})
    def test_fraction_accepts_bounded_finite_decimal_syntax(self):
        for value in ['.4','0.4','0.25','0.000001','0','1','1.000000']:
            with self.subTest(value=value):self.assertEqual(mod.validate_changes({'LLM_PROMPT_CACHE_MAX_RAM_FRACTION':value}),{'LLM_PROMPT_CACHE_MAX_RAM_FRACTION':value})
    def test_changed_original_identity_is_rejected_before_mutation(self):
        for field,value in [('expected_original_id','another-id'),('expected_image','sha256:other')]:
            with self.subTest(field=field):
                self.plan[field]=value
                with self.assertRaisesRegex(mod.UpdateError,'changed after'):mod.apply(self.engine,self.plan)
                self.plan.pop(field);self.assertEqual(self.engine.calls,[])
    # Parent tests exercise integer-specific requests and run in their own class.



class MultiDocker(FakeDocker):
    def __init__(self):
        super().__init__()
        self.original['Image']='sha256:'+'a'*64
        self.items['original-id']=copy.deepcopy(self.original)
        self.alias=self.original['Config']['Image']
        self.images={self.alias:self.original['Image'],self.original['Image']:self.original['Image']}
        self.creations=0;self.clone_bad_health=False;self.clone_mutator=None;self.before_remove=None
    def image(self,key):
        if key not in self.images:raise mod.DockerHTTPError(404,'Image not found')
        return {'Id':self.images[key]}
    def tag(self,image,tag):
        self.calls.append(('tag',image,tag));self.images[tag]=self.images.get(image,image)
    def create(self,name,config,host,networking=None):
        self.creations+=1;ident='new-id'if self.creations==1 else'clone-'+str(self.creations)
        self.calls.append(('create',name,ident))
        obj=copy.deepcopy(self.original)
        obj.update({'Id':ident,'Name':'/'+name,'Image':self.image(config['Image'])['Id'],'Config':copy.deepcopy(config),'HostConfig':copy.deepcopy(host)})
        obj['State']={'Running':False,'OOMKilled':False,'Health':{'Status':'starting'}}
        if networking is not None:
            obj['NetworkSettings']={**copy.deepcopy(obj.get('NetworkSettings',{})),'Networks':{key:{**copy.deepcopy(endpoint),'NetworkID':'network-id','EndpointID':'endpoint-'+ident,'IPAddress':'172.22.0.3','MacAddress':'02:42:ac:16:00:03'}for key,endpoint in networking['EndpointsConfig'].items()}}
        if self.creations>1 and self.clone_mutator:self.clone_mutator(obj)
        self.items[ident]=obj
        return ident
    def start(self,ident):
        self.calls.append(('start',ident));self.items[ident]['State']['Running']=True
        self.items[ident]['State']['Health']['Status']='unhealthy'if ident.startswith('clone-')and self.clone_bad_health else'healthy'
    def remove(self,ident):
        if self.before_remove:self.before_remove(ident)
        super().remove(ident)

class FinalizedRollbackReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=pathlib.Path(self.temp.name);self.env=self.root/'.env'
        self.raw=b'# retained bytes\r\nLLM_PROMPT_CACHE_MAX_RAM_FRACTION="0.25"\r\nSECRET=fixture-secret-marker-proof\r\n'
        self.env.write_bytes(self.raw);self.env.chmod(0o640)
        self.engine=MultiDocker()
        self.engine.items['original-id']['Config']['Env'] += ['LLM_PROMPT_CACHE_MAX_RAM_FRACTION=0.25','KV_CACHE_TYPE=q4_0','LLM_MAX_TOKENS=230000']
        self.engine.original=copy.deepcopy(self.engine.items['original-id'])
        self.plan={'run_id':'extra-review-20261026','changes':{'LLM_PROMPT_CACHE_MAX_RAM_FRACTION':'0.4'},'env_file':str(self.env),'output_dir':str(self.root/'private-proof'),'timeout':30}
    def applied(self,retain=False):
        mod.apply(self.engine,self.plan,retain=retain)
        root,state,original=mod.load_state(self.plan['output_dir'])
        self.tuned=copy.deepcopy(self.engine.inspect('ezlocalai'));self.tuned_raw=self.env.read_bytes();self.engine.calls=[]
        return root,state,original
    def assert_tuned_retained(self):
        current=self.engine.inspect('ezlocalai')
        self.assertEqual(current,self.tuned)
        self.assertEqual(self.env.read_bytes(),self.tuned_raw)
        self.assertNotIn(('remove',self.tuned['Id']),self.engine.calls)
    def assert_original_configuration(self,current,original):
        expected=copy.deepcopy(original['Config']);expected['Image']=original['Image']
        self.assertEqual(current['Config'],expected)
        self.assertEqual(mod.hostconfig_view(current['HostConfig']),mod.hostconfig_view(original['HostConfig']))
        self.assertEqual(mod.mount_view(current),mod.mount_view(original))
        self.assertEqual(current['Image'],original['Image'])
        self.assertTrue(current['State']['Running']);self.assertEqual(current['State']['Health']['Status'],'healthy')
        self.assertEqual(current['Name'],'/ezlocalai')
    def test_finalized_recreation_restores_old_env_and_all_other_configuration(self):
        root,state,original=self.applied()
        self.assertNotIn(original['Id'],self.engine.items)
        proof=mod.rollback(self.engine,root,state,original)
        current=self.engine.inspect('ezlocalai');self.assertNotEqual(current['Id'],original['Id'])
        self.assert_original_configuration(current,original)
        self.assertEqual(self.env.read_bytes(),self.raw);self.assertEqual(self.env.stat().st_mode&0o777,0o640)
        self.assertTrue(proof['healthy']);self.assertTrue(proof['dotenv_restored'])
        self.assertNotIn('fixture-secret-marker-proof',json.dumps(proof));self.assertNotIn('fixture-secret-marker-api',json.dumps(proof))
    def test_prefinal_rollback_keeps_exact_original_id_and_config_image_alias(self):
        root,state,original=self.applied(retain=True)
        proof=mod.rollback(self.engine,root,state,original)
        self.assertEqual(self.engine.inspect('ezlocalai'),self.engine.original)
        self.assertEqual(self.env.read_bytes(),self.raw);self.assertTrue(proof['original_container_restored'])
        self.assertFalse(any(call[0]=='create'for call in self.engine.calls))
    def test_external_current_id_conflict_refuses_before_any_stop(self):
        root,state,original=self.applied()
        intruder=copy.deepcopy(self.tuned);intruder['Id']='external-id'
        self.engine.items.pop(self.tuned['Id']);self.engine.items[intruder['Id']]=intruder
        with self.assertRaises(mod.UpdateError):mod.rollback(self.engine,root,state,original)
        self.assertEqual(self.engine.inspect('ezlocalai'),intruder)
        self.assertEqual(self.env.read_bytes(),self.tuned_raw)
        self.assertFalse(any(call[0]in('stop','remove','rename','create')for call in self.engine.calls))
    def test_missing_or_corrupt_dotenv_backup_refuses_before_any_stop(self):
        for missing in (True,False):
            with self.subTest(missing=missing):
                root,state,original=self.applied()if missing else self.saved
                backup=root/'dotenv.backup';valid=backup.read_bytes()
                if missing:backup.unlink()
                else:backup.write_bytes(b'corrupt backup')
                with self.assertRaises((mod.UpdateError,OSError)):mod.rollback(self.engine,root,state,original)
                self.assert_tuned_retained()
                self.assertFalse(any(call[0]in('stop','remove','rename','create')for call in self.engine.calls))
                backup.write_bytes(valid);self.saved=(root,state,original)
    def test_corrupt_original_inspect_hash_refuses_before_any_stop(self):
        root,state,original=self.applied()
        require_hash='original_inspect_sha256'in state
        self.assertTrue(require_hash,'new transactions must fingerprint the original inspect')
        changed=copy.deepcopy(original);changed['Config']['Cmd']=['corrupt']
        (root/'original-inspect.json').write_text(json.dumps(changed))
        try:
            root,state,loaded=mod.load_state(root)
            with self.assertRaises(mod.UpdateError):mod.rollback(self.engine,root,state,loaded)
        except mod.UpdateError:pass
        self.assert_tuned_retained();self.assertFalse(any(call[0]=='stop'for call in self.engine.calls))
    def test_moved_latest_alias_uses_recorded_immutable_image_and_never_retags(self):
        root,state,original=self.applied()
        self.engine.images[self.engine.alias]='sha256:'+'b'*64
        proof=mod.rollback(self.engine,root,state,original)
        self.assert_original_configuration(self.engine.inspect('ezlocalai'),original)
        self.assertEqual(self.engine.images[self.engine.alias],'sha256:'+'b'*64)
        self.assertFalse(any(call[0]=='tag'for call in self.engine.calls))
        self.assertTrue(proof['healthy'])
    def test_missing_or_changed_retained_rollback_tag_refuses_before_stop(self):
        root,state,original=self.applied()
        self.engine.images[state['rollback_tag']]='sha256:'+'b'*64
        with self.assertRaises(mod.UpdateError):mod.rollback(self.engine,root,state,original)
        self.assert_tuned_retained();self.assertFalse(any(call[0]in('stop','create')for call in self.engine.calls))
    def test_tuned_is_retained_until_recreated_original_is_healthy(self):
        root,state,original=self.applied()
        def before_remove(ident):
            if ident==self.tuned['Id']:
                clones=[obj for obj in self.engine.items.values()if obj['Id'].startswith('clone-')]
                self.assertEqual(len(clones),1)
                self.assertTrue(clones[0]['State']['Running']);self.assertEqual(clones[0]['State']['Health']['Status'],'healthy')
                self.assert_original_configuration(clones[0],original)
        self.engine.before_remove=before_remove
        mod.rollback(self.engine,root,state,original)
    def test_clone_hostconfig_drift_refuses_before_stopping_tuned(self):
        root,state,original=self.applied()
        self.engine.clone_mutator=lambda obj:obj['HostConfig'].update({'Memory':1})
        with self.assertRaises(mod.UpdateError):mod.rollback(self.engine,root,state,original)
        self.assert_tuned_retained();self.assertNotIn(('stop',self.tuned['Id']),self.engine.calls)
    def test_clone_health_failure_restores_previous_working_tuned_container_and_file(self):
        root,state,original=self.applied();self.engine.clone_bad_health=True
        with self.assertRaises(mod.UpdateError):mod.rollback(self.engine,root,state,original)
        self.assert_tuned_retained()
        self.assertFalse(any(obj['Id'].startswith('clone-')for obj in self.engine.items.values()))

    def test_successful_recreated_rollback_is_idempotent_without_a_second_cutover(self):
        root,state,original=self.applied()
        mod.rollback(self.engine,root,state,original)
        current=copy.deepcopy(self.engine.inspect('ezlocalai'));self.engine.calls=[]
        root,state,original=mod.load_state(root)
        proof=mod.rollback(self.engine,root,state,original)
        self.assertEqual(self.engine.inspect('ezlocalai'),current)
        self.assertTrue(proof['healthy']);self.assertEqual(self.env.read_bytes(),self.raw)
        self.assertFalse(any(call[0]in('create','stop','remove','rename')for call in self.engine.calls))
    def test_lost_tuned_delete_response_keeps_the_only_healthy_clone(self):
        root,state,original=self.applied();remove=self.engine.remove;lost=False
        def lost_response(ident):
            nonlocal lost
            remove(ident)
            if ident==self.tuned['Id']and not lost:
                lost=True
                raise mod.DockerTransportError('Injected lost DELETE response')
        self.engine.remove=lost_response
        try:mod.rollback(self.engine,root,state,original)
        except mod.UpdateError:pass
        self.assertTrue(lost,'review fault must reach tuned deletion')
        self.assert_original_configuration(self.engine.inspect('ezlocalai'),original)
        self.assertEqual(self.env.read_bytes(),self.raw)
        self.assertNotIn(self.tuned['Id'],self.engine.items)
    def test_finalized_bridge_recreation_preserves_ports_aliases_and_network_identity(self):
        obj=self.engine.items['original-id']
        obj['HostConfig']['NetworkMode']='ezlocalai_default'
        obj['HostConfig']['PortBindings']={'8091/tcp':[{'HostIp':'','HostPort':'8091'}]}
        obj['Config']['ExposedPorts']={'8091/tcp':{}}
        obj['NetworkSettings']={'Ports':{'8091/tcp':[{'HostIp':'0.0.0.0','HostPort':'8091'},{'HostIp':'::','HostPort':'8091'}]},'Networks':{'ezlocalai_default':{'IPAMConfig':None,'Links':None,'Aliases':['ezlocalai','ezlocalai'],'DriverOpts':None,'GwPriority':0,'NetworkID':'network-id','EndpointID':'old-endpoint','IPAddress':'172.22.0.2','MacAddress':'02:42:ac:16:00:02'}}}
        self.engine.original=copy.deepcopy(obj)
        root,state,original=self.applied();mod.rollback(self.engine,root,state,original)
        current=self.engine.inspect('ezlocalai')
        self.assert_original_configuration(current,original)
        mod.verify_network(current,original)
        self.assertEqual(current['NetworkSettings']['Ports'],original['NetworkSettings']['Ports'])
        self.assertEqual(current['NetworkSettings']['Networks']['ezlocalai_default']['Aliases'],['ezlocalai','ezlocalai'])

    def test_existing_v1_state_without_inspect_hash_recreates_original_honestly(self):
        root,state,original=self.applied()
        state.pop('original_inspect_sha256');mod.private_json(root/'state.json',state)
        root,state,original=mod.load_state(root)
        proof=mod.rollback(self.engine,root,state,original)
        self.assert_original_configuration(self.engine.inspect('ezlocalai'),original)
        self.assertFalse(proof['original_container_id_preserved']);self.assertTrue(proof['original_container_recreated'])
        self.assertEqual(self.env.read_bytes(),self.raw)
    def test_postverification_state_write_failure_retains_healthy_clone_and_can_retry(self):
        root,state,original=self.applied();writer=mod.private_json;failed=False
        def injected(path,value):
            nonlocal failed
            if pathlib.Path(path)==root/'state.json'and value.get('rollback_phase')=='original_configuration_verified'and not failed:
                failed=True
                raise OSError('Injected state disk full after verified clone')
            return writer(path,value)
        with mock.patch.object(mod,'private_json',side_effect=injected):
            with self.assertRaises(OSError):mod.rollback(self.engine,root,state,original)
        self.assertTrue(failed,'fault must reach verified clone durability boundary')
        current=copy.deepcopy(self.engine.inspect('ezlocalai'));self.assert_original_configuration(current,original)
        self.assertIn(self.tuned['Id'],self.engine.items)
        self.assertFalse(self.engine.items[self.tuned['Id']]['State']['Running'])
        self.assertEqual(self.env.read_bytes(),self.raw)
        root,state,original=mod.load_state(root)
        proof=mod.rollback(self.engine,root,state,original)
        self.assertEqual(self.engine.inspect('ezlocalai'),current)
        self.assertNotIn(self.tuned['Id'],self.engine.items);self.assertTrue(proof['healthy'])


if __name__=='__main__':unittest.main(verbosity=2)
