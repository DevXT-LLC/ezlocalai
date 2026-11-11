"""One fresh baseline 64K Nsight capture, invoked through the fleet controller.

Bundle prefill_cases and prefill_probe as importable modules before this file.
The caller owns and renews the idle drain. Compatibility arguments never widen
this experiment. No code runs on import; all subprocesses require main().
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

import prefill_probe as probe


NSIGHT_ROOT = Path("/opt/nvidia/nsight-systems/2025.5.2")
NSYS = str(NSIGHT_ROOT / "target-linux-x64/nsys")
REPORT_ROOT = Path("/home/josh/prefill-experiments/profiles")
TARGET = 64000
WORK_SECONDS = 900
RESTORE_SECONDS = 360
WATCHDOG_SECONDS = 1500


def action(*args, timeout=60):
    return subprocess.check_output(
        args, text=True, stderr=subprocess.STDOUT, timeout=timeout
    ).strip()


def inspect_container(name, optional=False):
    result = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, timeout=30
    )
    if result.returncode:
        if optional and re.search(r"No such (object|container)", result.stderr):
            return None
        raise RuntimeError("Could not inspect container identity")
    return json.loads(result.stdout)[0]


def fingerprint(container):
    fields = ("Id", "Image", "Name", "Config", "HostConfig", "Mounts")
    data = {key: container.get(key) for key in fields}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def owned(container, state):
    labels = container["Config"].get("Labels") or {}
    return (
        labels.get("xt.prefill.owner") == state["owner"]
        and labels.get("xt.prefill.profile") == state["uuid"]
        and container["Image"] == state["image"]
    )


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


# Self-contained because the controller bundles modules in memory on SSH stdin.
# This child survives loss of that transport and does not need a staged script.
WATCHDOG = r"""
import glob,hashlib,json,pathlib,re,signal,subprocess,sys,time
s=json.loads(sys.argv[1])
time.sleep(s['watchdog_seconds'])
def expired(*_): raise TimeoutError('Watchdog recovery deadline')
signal.signal(signal.SIGALRM,expired)
signal.alarm(360)
def call(*args,timeout=45):
 return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
def inspect(name,optional=False):
 p=call('docker','inspect',name)
 if p.returncode:
  if optional and re.search(r'No such (object|container)',p.stderr): return None
  raise RuntimeError('Watchdog identity inspection failed')
 return json.loads(p.stdout)[0]
def fingerprint(c):
 return hashlib.sha256(json.dumps({k:c.get(k) for k in
  ('Id','Image','Name','Config','HostConfig','Mounts')},sort_keys=True).encode()).hexdigest()
receipt={'event':'watchdog_restoration','original_container':False,'original_argv':False}
try:
 c=inspect(s['name'],True)
 if c is not None:
  labels=c['Config'].get('Labels') or {}
  if (labels.get('xt.prefill.owner')!=s['owner'] or
      labels.get('xt.prefill.profile')!=s['uuid'] or c['Image']!=s['image']):
   raise RuntimeError('Watchdog refuses unowned container')
  if c['State']['Running']:
   for tail in [('stop','--session='+s['session']),
                ('shutdown','--session='+s['session'],'--kill=sigterm')]:
    try: call('docker','exec',s['name'],s['nsys'],*tail,timeout=30)
    except subprocess.TimeoutExpired: pass
  call('docker','rm','-f',c['Id'])
  if inspect(s['name'],True) is not None: raise RuntimeError('Probe still exists')
 c=inspect(s['production_id'])
 if fingerprint(c)!=s['fingerprint']: raise RuntimeError('Production identity changed')
 if call('docker','start',s['production_id']).returncode:
  raise RuntimeError('Production restart failed')
 deadline=time.monotonic()+240
 while time.monotonic()<deadline:
  c=inspect(s['production_id'])
  if fingerprint(c)!=s['fingerprint']: raise RuntimeError('Restored identity changed')
  argv=[]
  for path in glob.glob('/proc/[0-9]*/cmdline'):
   try:
    with open(path,'rb') as f: a=[v.decode() for v in f.read().split(b'\0') if v]
    if a and a[0]=='/usr/local/bin/llama-server': argv.append(a)
   except (OSError,UnicodeError): pass
  if (c['State']['Running'] and c['State'].get('Health',{}).get('Status')=='healthy'
      and all(a in argv for a in s['engines'])):
   receipt.update(original_container=True,original_argv=True,image=c['Image'])
   break
  time.sleep(2)
 else: raise RuntimeError('Watchdog restoration unverified')
except BaseException as e:
 receipt['error']=type(e).__name__+': '+str(e)
finally:
 signal.alarm(0)
 p=pathlib.Path(s['report_dir'])/'watchdog-restoration.json'
 p.write_text(json.dumps(receipt,indent=2)+'\n')
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned-drain", required=True)
    parser.add_argument("--variants", default="baseline")
    parser.add_argument("--targets", default="64000")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.owned_drain):
        parser.error("Invalid owned-drain identity")
    if args.repeats < 1:
        parser.error("Repeats must be positive")
    return args


def check_ports():
    for port in (18092, 18093):
        with socket.socket() as guard:
            guard.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            guard.bind(("127.0.0.1", port))


def runtime_identity(container):
    state = container["State"]
    return (state.get("Pid"), state.get("StartedAt"), container.get("RestartCount"))


def untouched_receipt(snapshot):
    original = snapshot["production"]
    current = inspect_container("ezlocalai")
    for container in (original, current):
        if (
            not container["State"]["Running"]
            or container["State"].get("Health", {}).get("Status") != "healthy"
        ):
            raise RuntimeError("Preflight production health is unverified")
    if (
        fingerprint(current) != fingerprint(original)
        or runtime_identity(current) != runtime_identity(original)
        or sorted(probe.engines()) != sorted(snapshot["engines"])
    ):
        raise RuntimeError("Preflight production snapshot changed")
    return {
        "image": current["Image"],
        "original_container": True,
        "original_argv": True,
        "original_untouched": True,
        "phase": "preflight",
    }


def prepare(args, snapshot=None):
    production = inspect_container("ezlocalai")
    if not production["State"]["Running"]:
        raise RuntimeError("Production must be running; never join another screen")
    if not production["Image"].startswith("sha256:"):
        raise RuntimeError("An immutable production image ID is required")
    available = probe.engines()
    qwen = [v for v in available if any("Qwen3.8-27B" in a for a in v)]
    voices = [v for v in available if any("LFM3" in a for a in v)]
    if len(qwen) != 1 or len(voices) > 1 or len(available) != 1 + len(voices):
        raise RuntimeError(
            "Unexpected inference processes; refusing ambiguous ownership"
        )
    if snapshot is not None:
        snapshot.update(
            json.loads(json.dumps({"production": production, "engines": available}))
        )
    if (
        not Path(NSYS).is_file()
        or not (NSIGHT_ROOT / "host-linux-x64/QdstrmImporter").is_file()
    ):
        raise RuntimeError("Complete Nsight 2025.5.2 installation required")
    version = action(NSYS, "--version", timeout=15)
    original = qwen[0]
    if TARGET + 2048 > int(original[original.index("-c") + 1]):
        raise ValueError("Insufficient baseline context for 64K profile")
    check_ports()
    baseline = probe.variant_args(original, "baseline")
    identifier = uuid.uuid4().hex
    directory = REPORT_ROOT / identifier
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(mode=0o700)
    state = {
        "uuid": identifier,
        "owner": args.owned_drain,
        "name": "prefill-profile-" + identifier,
        "session": "prefill" + identifier,
        "production_id": production["Id"],
        "image": production["Image"],
        "fingerprint": fingerprint(production),
        "engines": [original, *voices],
        "report_dir": str(directory),
        "nsys": NSYS,
        "watchdog_seconds": WATCHDOG_SECONDS,
    }
    probe.emit(
        "provenance",
        image=state["image"],
        engine_args=original,
        voice_args=voices,
        report_dir=str(directory),
        nsight_version=version,
        requested={
            k: getattr(args, k)
            for k in ("variants", "targets", "repeats", "skip_cache")
        },
        variants=["baseline"],
        targets=[TARGET],
        repeats=1,
        skip_cache=True,
        profiled_timings_not_throughput_evidence=True,
    )
    return production, baseline, state


def launch(production, baseline, state):
    env = dict(v.split("=", 1) for v in production["Config"]["Env"])
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        state["name"],
        "--label",
        "xt.prefill.owner=" + state["owner"],
        "--label",
        "xt.prefill.profile=" + state["uuid"],
        "--network",
        "host",
        "--gpus",
        "all",
        "--volumes-from",
        state["production_id"] + ":ro",
        "--mount",
        f"type=bind,src={NSIGHT_ROOT},dst={NSIGHT_ROOT},readonly",
        "--mount",
        f"type=bind,src={state['report_dir']},dst=/reports",
        "--entrypoint",
        "/bin/sleep",
    ]
    for key, value in env.items():
        if key.startswith(("GGML_", "CUDA_", "NVIDIA_")):
            command += ["--env", key + "=" + value]
    action(*command, state["image"], "infinity")
    # Keep server/profiler output separate from the controller's JSONL receipt.
    action(
        "docker",
        "exec",
        "-d",
        state["name"],
        "/bin/sh",
        "-c",
        'exec "$@" > /reports/nsys-launch.log 2>&1',
        "profile-launch",
        NSYS,
        "launch",
        "--session-new=" + state["session"],
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=node",
        "--show-output=true",
        state["engines"][0][0],
        *baseline,
    )
    probe.wait_health(18092)
    for voice in state["engines"][1:]:
        argv = voice.copy()
        argv[argv.index("--port") + 1] = "18093"
        action("docker", "exec", "-d", state["name"], *argv)
        probe.wait_health(18093)


def nsight(state, verb, *options, timeout=120):
    return action(
        "docker",
        "exec",
        state["name"],
        NSYS,
        verb,
        "--session=" + state["session"],
        *options,
        timeout=timeout,
    )


def has_kernels(stats):
    index = None
    for row in csv.reader(stats.splitlines()):
        if "Instances" in row and "Name" in row:
            index = row.index("Instances")
        elif index is not None and len(row) > index:
            try:
                if int(row[index].replace(",", "")) > 0:
                    return True
            except ValueError:
                pass
    return False


def capture(state):
    directory = Path(state["report_dir"])
    case, tokens = probe.calibrate(TARGET)
    if abs(tokens - TARGET) > TARGET * 0.01:
        raise RuntimeError(
            "Calibration did not reach 64K within 1%; no capture started"
        )
    start_log = nsight(
        state,
        "start",
        "--sample=none",
        "--cpuctxsw=none",
        "--gpu-metrics-devices=none",
        "--output=/reports/cold",
    )
    try:
        (directory / "nsys-start.log").write_text(start_log + "\n")
        original = state["engines"][0]
        sample = probe.measure(
            case, original[original.index("--alias") + 1], cache=False
        )
        probe.emit(
            "sample",
            block=0,
            variant="baseline",
            target_tokens=TARGET,
            template_tokens=tokens,
            repeat=0,
            kind="cold",
            profiled=True,
            **sample,
        )
        if sample["timings"]["cache_n"] != 0:
            raise RuntimeError("Strict cold profile reused prompt tokens")
        if abs(sample["timings"]["prompt_n"] - TARGET) > TARGET * 0.02:
            raise RuntimeError(
                "Native evaluated prompt length is not approximately 64K"
            )
    finally:
        stop_log = nsight(state, "stop")
        (directory / "nsys-stop.log").write_text(stop_log + "\n")
    report = directory / "cold.nsys-rep"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("Nsight did not produce a nonempty .nsys-rep")
    stats = action(
        "docker",
        "exec",
        state["name"],
        NSYS,
        "stats",
        "--report=cuda_gpu_kern_sum",
        "--format=csv",
        "/reports/cold.nsys-rep",
        timeout=120,
    )
    (directory / "kernel-summary.txt").write_text(stats + "\n")
    if not has_kernels(stats):
        raise RuntimeError("Report has no confirmed CUDA kernel activity")
    transfer_stats = action(
        "docker",
        "exec",
        state["name"],
        NSYS,
        "stats",
        "--report=cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum",
        "--format=csv",
        "/reports/cold.nsys-rep",
        timeout=120,
    )
    (directory / "transfer-api-summary.txt").write_text(transfer_stats + "\n")
    log = (directory / "nsys-launch.log").read_text(errors="replace")
    log += "\n" + start_log + "\n" + stop_log + "\n" + stats + "\n" + transfer_stats
    if re.search(
        r"CUPTI.*(ERROR|INVALID)|not all CUDA events|lost.*events|dropped.*events",
        log,
        re.I,
    ):
        raise RuntimeError(
            "Nsight reported incomplete collection; retain report as partial evidence"
        )
    return {
        "capture_success": True,
        "report": str(report),
        "quality_pass": sample["quality_pass"],
        "transfer_api_summary": str(directory / "transfer-api-summary.txt"),
        "aggregate_transfer_time_is_not_critical_path": True,
    }


def restore(state):
    container = inspect_container(state["name"], optional=True)
    if container is not None:
        if not owned(container, state):
            raise RuntimeError("Refusing cleanup of an unowned profile container")
        if container["State"]["Running"]:
            for verb, options in (("stop", ()), ("shutdown", ("--kill=sigterm",))):
                try:
                    nsight(state, verb, *options, timeout=30)
                except (subprocess.SubprocessError, OSError):
                    # A failed capture/session must not strand either inference engine.
                    pass
        action("docker", "rm", "-f", container["Id"], timeout=45)
        if inspect_container(state["name"], optional=True) is not None:
            raise RuntimeError("Profile container still exists; restoration unsafe")
    original = inspect_container(state["production_id"])
    if fingerprint(original) != state["fingerprint"]:
        raise RuntimeError("Retained production identity/configuration changed")
    action("docker", "start", state["production_id"])
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        restored = inspect_container(state["production_id"])
        if fingerprint(restored) != state["fingerprint"]:
            raise RuntimeError("Restored production identity/configuration changed")
        if (
            restored["State"]["Running"]
            and restored["State"].get("Health", {}).get("Status") == "healthy"
        ):
            actual = probe.engines()
            if all(argv in actual for argv in state["engines"]):
                return {
                    "image": restored["Image"],
                    "original_argv": True,
                    "original_container": True,
                }
        time.sleep(2)
    raise RuntimeError("Restoration unverified; independent watchdog retained")


def main(argv=None):
    args = parse_args(argv)
    snapshot = {}
    try:
        production, baseline, state = prepare(args, snapshot)
    except Exception as error:
        probe.emit(
            "profile",
            capture_success=False,
            phase="preflight",
            error_type=type(error).__name__,
        )
        try:
            receipt = untouched_receipt(snapshot)
        except Exception as verification_error:
            probe.emit(
                "restoration_failed",
                phase="preflight",
                error_type=type(verification_error).__name__,
                original_untouched=False,
            )
        else:
            # Controller consumes this final receipt even though the probe exits nonzero.
            probe.emit("restored", **receipt)
        raise
    watchdog = subprocess.Popen(
        [sys.executable, "-c", WATCHDOG, json.dumps(state)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    previous = {}
    status = {"capture_success": False, "report_dir": state["report_dir"]}
    failure = None

    def interrupted(*_):
        raise InterruptedError("Profile interrupted; restoring retained production")

    def expired(*_):
        raise TimeoutError("Profile phase deadline exceeded")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            previous[sig] = signal.signal(
                sig, expired if sig == signal.SIGALRM else interrupted
            )
        signal.alarm(WORK_SECONDS)
        if watchdog.poll() is not None:
            raise RuntimeError("Independent recovery watchdog exited before cutover")
        if (
            fingerprint(inspect_container(state["production_id"]))
            != state["fingerprint"]
        ):
            raise RuntimeError("Production changed before cutover")
        action("docker", "stop", "--time", "45", state["production_id"], timeout=60)
        stopped = inspect_container(state["production_id"])
        if stopped["State"]["Running"] or fingerprint(stopped) != state["fingerprint"]:
            raise RuntimeError("Retained production is not stopped and unchanged")
        launch(production, baseline, state)
        status.update(capture(state))
    except BaseException as error:
        failure = error
        # Do not put subprocess output (potential environment/argv data) in receipts.
        status["error_type"] = type(error).__name__
        if isinstance(error, (RuntimeError, ValueError, InterruptedError)):
            status["error"] = str(error)
    finally:
        signal.alarm(0)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(RESTORE_SECONDS)
        try:
            # Preserve capture failure evidence even when recovery needs the watchdog.
            # An evidence-write failure must never prevent production restoration.
            try:
                write_json(Path(state["report_dir"]) / "profile-status.json", status)
            except OSError:
                pass
            receipt = restore(state)
            watchdog.terminate()
            watchdog.wait(timeout=5)
            write_json(Path(state["report_dir"]) / "profile-status.json", status)
            write_json(
                Path(state["report_dir"]) / "restoration.json",
                dict(event="restored", **receipt),
            )
            probe.emit("profile", **status)
            probe.emit("restored", **receipt)
        except BaseException as error:
            probe.emit(
                "restoration_failed",
                error_type=type(error).__name__,
                watchdog_retained=watchdog.poll() is None,
            )
            raise
        finally:
            signal.alarm(0)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if failure is not None:
        raise RuntimeError(
            "Profile failed; production restored; inspect profile-status.json"
        ) from failure


if __name__ == "__main__":
    main()
