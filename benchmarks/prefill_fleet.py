"""Reserve one idle worker, run a bounded probe, and verify restoration.

No credentials are written to evidence. Remote targets use verified SSH keys;
local targets use --host local. This script never changes production settings.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


def request(base, path, body=None, key=""):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(
        urllib.request.Request(base + path, data=data, headers=headers), timeout=20
    ) as response:
        return json.load(response)


def bundle_source(root, probe_kind):
    """Bundle dependencies as inert modules; only the chosen entry point runs."""
    if probe_kind not in ("screen", "profile"):
        raise ValueError("probe_kind must be screen or profile")
    modules = ["prefill_cases"]
    if probe_kind == "profile":
        modules.append("prefill_probe")
    source = "import sys,types\n"
    for name in modules:
        source += f"m=types.ModuleType({json.dumps(name)})\n"
        source += f"sys.modules[{json.dumps(name)}]=m\n"
        source += "exec(" + repr((root / f"{name}.py").read_text()) + ",m.__dict__)\n"
    entry = "prefill_probe" if probe_kind == "screen" else "prefill_profile"
    return source + (root / f"{entry}.py").read_text()


class TargetAbsent(RuntimeError):
    """A stopped worker may disappear from the router until its heartbeat returns."""


def available_target(router, label):
    workers = request(router, "/v1/router/workers")["data"]
    target = next((w for w in workers if w.get("label") == label), None)
    others = {
        w["label"]
        for w in workers
        if w.get("label")
        and w["label"] != label
        and not w.get("external_provider")
        and not w.get("draining")
        and 0 <= w.get("last_heartbeat_age", 999) < 30
        and any("Qwen3.8-27B" in m for m in w.get("models", []))
    }
    if len(others) < 2:
        raise RuntimeError("Require two other healthy undrained Qwen workers")
    if target is None:
        raise TargetAbsent(f"Target {label!r} is absent from router registry")
    return target


def wait_probe(process, source, renewal_failed, timeout=2900, restore_timeout=300):
    """On cancellation, allow the probe's SIGTERM handler to restore first.

    Terminating SSH only closes the transport; it does not prove remote exit or
    restoration. The caller still requires an explicit final restoration receipt.
    """
    deadline = time.monotonic() + timeout
    try:
        while True:
            if renewal_failed.is_set():
                raise RuntimeError("Maintenance renewal failed; probe interrupted")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            try:
                process.communicate(input=source, timeout=min(1, remaining))
                return process.returncode
            except subprocess.TimeoutExpired:
                # communicate retains unsent input across timeout retries.
                source = None
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=restore_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=20)
        raise


def require_restoration(output):
    try:
        rows = [
            json.loads(line) for line in output.read_text().splitlines() if line.strip()
        ]
        valid = (
            rows
            and all(isinstance(row, dict) for row in rows)
            and rows[-1].get("event") == "restored"
            and rows[-1].get("original_argv") is True
            and rows[-1].get("original_container") is True
        )
    except (OSError, ValueError, RecursionError) as error:
        raise RuntimeError(
            "Unreadable restoration evidence; drain not released"
        ) from error
    if not valid:
        raise RuntimeError(
            "No confirmed final restoration receipt; inspect watchdog before releasing drain"
        )


def run_reserved(args, key, maintenance, target, source, command):
    """Own every post-acquisition path, including failures before probe launch."""
    done = threading.Event()
    renewal_failed = threading.Event()

    def renew():
        while not done.wait(45):
            try:
                request(args.router, "/v1/router/drain", maintenance, key)
            except Exception:
                renewal_failed.set()
                print("Maintenance renewal failed; interrupting probe", flush=True)

    thread = None
    thread_started = False
    process = None
    result = None
    try:
        thread = threading.Thread(target=renew, daemon=True)
        # The fleet may have changed while the idle reservation was in flight.
        target = available_target(args.router, args.label)
        with args.output.open("x") as output:
            thread.start()
            thread_started = True
            if renewal_failed.is_set():
                raise RuntimeError("Maintenance renewal failed before probe launch")
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, text=True, stdout=output
            )
            result = wait_probe(process, source, renewal_failed)
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    raise RuntimeError("Probe still alive; drain not released")
                require_restoration(args.output)
            # No receipt is needed when no process was launched. Production was
            # untouched, but verify health before returning it to the router.
            url = target["url"]
            if args.host != "local" and "172.18." in url:
                url = "http://" + args.host.split("@")[-1] + ":8091"
            health = request(url, "/health")
            if not isinstance(health, dict) or health.get("status") not in (
                "healthy",
                "ok",
            ):
                raise RuntimeError("Production health unconfirmed; drain not released")
        finally:
            done.set()
            if thread_started:
                thread.join(timeout=25)
                if thread.is_alive():
                    raise RuntimeError(
                        "Lease renewal still in flight; drain not released"
                    )
        request(
            args.router,
            "/v1/router/resume",
            {"label": args.label, "owner": maintenance["owner"]},
            key,
        )
        print(
            json.dumps(
                {
                    "label": args.label,
                    "restored": True,
                    "drain_released": True,
                    "renewal_failed": renewal_failed.is_set(),
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
    if renewal_failed.is_set():
        raise RuntimeError(
            "Maintenance renewal failed; production restored, inspect evidence"
        )
    if result is None or result:
        raise RuntimeError("Probe failed; production restored, inspect evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-kind", choices=("screen", "profile"), default="screen")
    parser.add_argument("--variants")
    parser.add_argument("--targets")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Pre-acquisition wait budget for a busy or temporarily absent target",
    )
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument(
        "--engine-root", help="Absolute remote engine bin/lib directory (screen only)"
    )
    parser.add_argument(
        "--candidate-model", help="Absolute remote external GGUF path (screen only)"
    )
    parser.add_argument(
        "--candidate-template", help="Absolute remote Jinja template path (screen only)"
    )
    parser.add_argument(
        "--tasks",
        default="infra",
        help="Comma-separated infra,rust,js tasks (screen only)",
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--router", default="http://192.168.1.214:8092")
    parser.add_argument("--key-env", default="EZLOCALAI_API_KEY")
    parser.add_argument(
        "--router-key-ssh", help="Read existing router API key into process memory only"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence output already exists; use a new path")
    if args.wait_seconds < 0:
        parser.error("Wait budget cannot be negative")
    remote_paths = {
        "--engine-root": args.engine_root,
        "--candidate-model": args.candidate_model,
        "--candidate-template": args.candidate_template,
    }
    if args.probe_kind == "profile":
        if any(value is not None for value in remote_paths.values()):
            parser.error("Engine and candidate options are screen-only")
        if args.tasks != "infra" or args.thinking or args.max_tokens != 1024:
            parser.error("Task, thinking, and token overrides are screen-only")
    else:
        args.variants = (
            args.variants
            if args.variants is not None
            else "baseline,ub512,ub2048,baseline"
        )
        args.targets = (
            args.targets if args.targets is not None else "8000,64000,128000,200000"
        )
        for flag, value in remote_paths.items():
            if value is not None and not Path(value).is_absolute():
                parser.error(f"{flag} must be an absolute remote path")
        variants = set(args.variants.split(","))
        if variants & {"gdn-off", "gdn-on"} and not args.engine_root:
            parser.error("gdn-off/gdn-on require --engine-root")
        if "nvfp4-ub1024" in variants and not args.candidate_model:
            parser.error("nvfp4-ub1024 requires --candidate-model")
        if not set(args.tasks.split(",")) <= {"infra", "rust", "js"}:
            parser.error("--tasks must contain only infra,rust,js")
        if not 128 <= args.max_tokens <= 8192:
            parser.error("--max-tokens must be between 128 and 8192")
    key = os.environ.get(args.key_env, "")
    if not key and args.router_key_ssh:
        data = json.loads(
            subprocess.check_output(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    args.router_key_ssh,
                    "docker",
                    "inspect",
                    "workerrouter-worker-router-1",
                ]
            )
        )[0]
        env = dict(v.split("=", 1) for v in data["Config"]["Env"])
        key = env.get("API_KEY", env.get("EZLOCALAI_API_KEY", ""))
    if not key and not args.router_key_ssh:
        raise RuntimeError("Router maintenance credential required")
    owner = "prefill-" + uuid.uuid4().hex
    maintenance = {"label": args.label, "owner": owner, "ttl_seconds": 300}
    root = Path(__file__).resolve().parent
    # Bundle the reviewed helper without writing any remote production files.
    source = bundle_source(root, args.probe_kind)
    options = [
        "--owned-drain",
        owner,
        "--repeats",
        str(args.repeats),
    ]
    # Omit unset profile options so its own defaults remain authoritative.
    if args.variants is not None:
        options += ["--variants", args.variants]
    if args.targets is not None:
        options += ["--targets", args.targets]
    if args.skip_cache:
        options += ["--skip-cache"]
    if args.probe_kind == "screen":
        for flag, value in remote_paths.items():
            if value is not None:
                options += [flag, value]
        options += ["--tasks", args.tasks, "--max-tokens", str(args.max_tokens)]
        if args.thinking:
            options += ["--thinking"]
    command = ["python3", "-u", "-", *options]
    if args.host != "local":
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            args.host,
            shlex.join(command),
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            target = available_target(args.router, args.label)
        except TargetAbsent as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TargetAbsent(
                    f"{error}; reservation wait budget exhausted"
                ) from None
            print(
                json.dumps(
                    {
                        "event": "maintenance_wait",
                        "label": args.label,
                        "reason": "target_absent",
                        "detail": str(error),
                    }
                ),
                flush=True,
            )
            time.sleep(min(30, remaining))
            continue
        try:
            request(
                args.router,
                "/v1/router/drain",
                dict(maintenance, require_idle=True),
                key,
            )
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            remaining = deadline - time.monotonic()
            if error.code != 409 or remaining <= 0:
                raise RuntimeError(
                    f"Maintenance reservation failed ({error.code}): {detail}"
                ) from None
            print(
                json.dumps(
                    {"event": "maintenance_wait", "label": args.label, "detail": detail}
                ),
                flush=True,
            )
            time.sleep(min(30, remaining))
    run_reserved(args, key, maintenance, target, source, command)


if __name__ == "__main__":
    main()
