"""Owned, idle-drained worker experiment. Invoke through prefill_fleet.py only.

The original container is retained, model mounts are read-only, and an independent
watchdog restores production if the controlling connection disappears.
"""

import argparse
import csv
import glob
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

from prefill_cases import build_case, build_code_case, estimate_record_count, grade


WORK_SECONDS = 2000
# Finish recovery within the controller's 300-second cancellation grace.
RESTORE_SECONDS = 280
WATCHDOG_SECONDS = 2400


# Pins for the isolated 2026-11-11 CUDA 12.8 GDN build, not a general engine policy.
GDN_BUILD_HASHES = {
    "libggml-base.so.0.29.0": "01165991ccb9a8317d0217fc922d31391b71c7e985f3b58b4b8b484b701c26a4",
    "libggml-cpu.so.0.29.0": "38a62771c7345cd95d20ffdc3b2983b22b4f957352f223f99b0209f68e2996f8",
    "libggml-cuda.so.0.29.0": "32a64b12e1bbc8a9d70e29ca60bc1ce0ba5005f5957d57401f09e5a1588df3d4",
    "libggml.so.0.29.0": "b819996e00de4990f519458e096a1a2a79244680b884d541767dd377df2d2e10",
    "libllama-common.so.0.9.0": "3ef50fd9dd50ca9841d2cb6628334ddd26046417f71845df05987bb5b9d86f39",
    "libllama-server-impl.so": "0e23d6fba388631aa5ab06655414b92ff3ced80f1b38b3d847624ff88fb00777",
    "libllama.so.0.9.0": "5eac20862e16f81c4c7efa3f3feae0b187cc4a27f18db58ff940737e803c15ce",
    "libmtmd.so.0.9.0": "80bc9c564bb143f8fe54d5d9d44c5da142847c88ad42ae53800dc2a48eeab97b",
    "llama-server": "b19b041625578c6f3c6c8d1294a361e3480c62d7403b896908397354eebc9db6",
    "test-backend-ops": "6bc78f74f048f4f41a34c3c6ff53049226756491cba9e9f44c9d31643633d73c",
}
GDN_ANCHORS = {"llama-server", "test-backend-ops", "libggml-cuda.so.0.29.0"}
GDN_OPERATOR_PARAMS = frozenset(
    f"type=f32,head_count=16,head_size=128,n_seq_tokens={tokens},"
    f"n_seqs={seqs},v_repeat={repeat},permuted={permuted},kda=0,K=1"
    for tokens, seqs, repeat, permuted in (
        (63, 1, 2, 0),
        (64, 1, 4, 0),
        (65, 1, 1, 0),
        (65, 3, 4, 0),
        (128, 1, 2, 0),
        (129, 1, 2, 0),
        (129, 2, 4, 1),
    )
)


def engine_provenance(engine_root, require_gdn=False):
    """Hash the staged bin tree; GDN pins and optional bin/artifact-sha256.txt must match.

    The optional sha256sum manifest accepts relative bin names, local absolute
    bin paths, or the original /validation/build/bin paths. It never replaces
    the trusted pins. Call before starting a watchdog or stopping production.
    """
    root = Path(engine_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Engine root must be a directory")
    hashes, resolved_hashes = {}, {}
    for path in sorted(root.rglob("*")):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or (path.is_symlink() and path.is_dir()):
            raise ValueError(f"Engine path escapes or redirects staging: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Engine artifact is not a regular file: {path}")
        if resolved not in resolved_hashes:
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            resolved_hashes[resolved] = digest.hexdigest()
        hashes[path.relative_to(root).as_posix()] = resolved_hashes[resolved]
    for name in (
        ("llama-server", "test-backend-ops") if require_gdn else ("llama-server",)
    ):
        if name not in hashes or not (root / name).stat().st_mode & 0o111:
            raise ValueError(f"Missing executable engine artifact: {name}")
    if require_gdn:
        expected = dict(GDN_BUILD_HASHES)
        # Check loadable SONAME aliases as well as the versioned payloads.
        for name, digest in GDN_BUILD_HASHES.items():
            if ".so." in name:
                stem, version = name.split(".so.", 1)
                expected[stem + ".so"] = digest
                expected[stem + ".so." + version.split(".")[0]] = digest
        for name, digest in expected.items():
            if hashes.get(name) != digest:
                raise ValueError(f"Missing or mismatched pinned GDN artifact: {name}")
        manifest = root / "artifact-sha256.txt"
        if manifest.exists():
            listed = set()
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
                if not match:
                    raise ValueError("Malformed GDN artifact manifest")
                digest, filename = match.groups()
                path = Path(filename)
                if path.is_absolute():
                    for prefix in (root, Path("/validation/build/bin")):
                        if path.is_relative_to(prefix):
                            path = path.relative_to(prefix)
                            break
                    else:
                        raise ValueError(
                            "Manifest path is outside the engine bin directory"
                        )
                name = path.as_posix()
                if (
                    ".." in path.parts
                    or name in listed
                    or hashes.get(name) != digest.lower()
                ):
                    raise ValueError(f"Stale or invalid GDN manifest entry: {name}")
                listed.add(name)
            if not GDN_ANCHORS <= listed:
                raise ValueError("GDN manifest is missing required pinned artifacts")
    result = {"engine_sha256": hashes["llama-server"], "engine_files_sha256": hashes}
    if require_gdn:
        result["gdn_build_verified"] = True
    return result


def run(*args, timeout=120):
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def inspect_container(name, optional=False):
    result = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, timeout=30
    )
    if result.returncode:
        if optional and re.search(r"no such (object|container)", result.stderr, re.I):
            return None
        raise RuntimeError("Could not inspect container identity")
    return json.loads(result.stdout)[0]


def remove_owned_probe(name, owner, image):
    container = inspect_container(name, optional=True)
    if container is None:
        return
    if (container["Config"].get("Labels") or {}).get(
        "xt.prefill.owner"
    ) != owner or container["Image"] != image:
        raise RuntimeError("Refusing cleanup of an unowned probe container")
    run("docker", "rm", "-f", container["Id"], timeout=45)
    if inspect_container(name, optional=True) is not None:
        raise RuntimeError("Probe container still exists; restoration unsafe")


def check_deadline(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Maintenance work deadline exhausted")
    return remaining


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def engines():
    result = []
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            argv = [v.decode() for v in open(path, "rb").read().split(b"\0") if v]
            if argv and argv[0] == "/usr/local/bin/llama-server":
                result.append(argv)
        except (OSError, UnicodeError):
            pass
    return result


def http(path, body=None, port=18092, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{port}" + path,
            data=data,
            headers={"Content-Type": "application/json"},
        ),
        timeout=timeout,
    )


def wait_health(port):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with http("/health", port=port, timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("Engine failed to become healthy")


def variant_args(original, variant):
    args = original[1:].copy()
    if variant in ("mtp2", "mtp3", "mtp4"):
        # Only qualify the explicit canonical MTP setup emitted by the worker.
        for flag in (
            "--port",
            "--spec-type",
            "--spec-draft-n-max",
            "--spec-draft-p-min",
        ):
            positions = [
                i
                for i, arg in enumerate(args)
                if arg == flag or arg.startswith(flag + "=")
            ]
            if len(positions) != 1:
                raise ValueError(f"MTP variant requires exactly one {flag}")
            index = positions[0]
            if (
                args[index] != flag
                or index + 1 == len(args)
                or not args[index + 1]
                or args[index + 1].startswith("-")
            ):
                raise ValueError(f"Unsupported MTP argument syntax: {flag}")
        if any(
            arg == "--draft-p-min" or arg.startswith("--draft-p-min=") for arg in args
        ):
            raise ValueError("Unsupported MTP p-min alias override")
        if args[args.index("--spec-type") + 1] != "draft-mtp":
            raise ValueError("MTP length variants require active draft-mtp")
        if not re.fullmatch(r"[1-9][0-9]*", args[args.index("--spec-draft-n-max") + 1]):
            raise ValueError("Unsupported original MTP maximum")
        args[args.index("--spec-draft-n-max") + 1] = variant[-1]
    args[args.index("--port") + 1] = "18092"
    if variant == "nvfp4-ub1024":
        if int(args[args.index("-b") + 1]) < 1024:
            raise ValueError("Physical batch exceeds logical batch")
        args[args.index("-m") + 1] = "/prefill-candidate.gguf"
        args[args.index("-ub") + 1] = "1024"
    elif variant.startswith("ub"):
        value = int(variant[2:])
        if value not in (512, 1024, 2048, 3072, 4096):
            raise ValueError("Unsupported physical batch")
        if value > int(args[args.index("-b") + 1]):
            raise ValueError("Physical batch exceeds logical batch")
        args[args.index("-ub") + 1] = str(value)
    elif variant == "mtp-off":
        args[args.index("--spec-type") + 1] = "none"
    elif variant not in ("baseline", "gdn-off", "gdn-on", "mtp2", "mtp3", "mtp4"):
        raise ValueError("Unknown variant")
    return args


def check_operators(production, name, owner, engine_root):
    """Qualify the isolated backend before using it for model timing."""
    for enabled in ("0", "1"):
        run(
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            "xt.prefill.owner=" + owner,
            "--network",
            "none",
            "--gpus",
            "all",
            "--mount",
            f"type=bind,src={engine_root},dst=/prefill-bin,readonly",
            "--env",
            "LD_LIBRARY_PATH=/prefill-bin:/usr/local/cuda/lib64",
            "--env",
            "GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=" + enabled,
            "--env",
            "OMP_NUM_THREADS=4",
            "--entrypoint",
            "/prefill-bin/test-backend-ops",
            production["Image"],
            "test",
            "-b",
            "CUDA0",
            "-o",
            "GATED_DELTA_NET",
            "-j",
            "1",
            "-p",
            "^type=f32,head_count=16,head_size=128,n_seq_tokens=(63|64|65|128|129),n_seqs=[123],v_repeat=[124],permuted=[01],kda=0,K=1$",
            "--output",
            "csv",
        )
        code = subprocess.check_output(
            ["docker", "wait", name], text=True, timeout=600
        ).strip()
        output = run("docker", "logs", name)
        # This printer omits `passed`; successful process exit is also required.
        cases = list(csv.DictReader(io.StringIO(output), strict=True))
        passed = [
            r
            for r in cases
            if r.get("op_name") == "GATED_DELTA_NET"
            and r.get("backend_name") == "CUDA0"
            and r.get("supported") == "1"
            and r.get("test_mode") == "test"
            and r.get("error_message") == ""
            and r.get("op_params") in GDN_OPERATOR_PARAMS
            and None not in r
        ]
        emit(
            "operator_check",
            slice64=enabled,
            exit_code=code,
            output=output,
            passed_cases=len(passed),
        )
        run("docker", "rm", name)
        if (
            code != "0"
            or len(cases) != len(GDN_OPERATOR_PARAMS)
            or len(passed) != len(GDN_OPERATOR_PARAMS)
            or {r["op_params"] for r in passed} != GDN_OPERATOR_PARAMS
        ):
            raise RuntimeError(
                "Experimental backend did not pass confirmed GDN operator checks"
            )


def calibrate(target, task="infra", chat_template_kwargs=None):
    # Same prompt on every variant. Coldness is verified from engine timings,
    # not assumed from the cache_prompt flag or a random suffix.
    nonce = "prefill-fleet-v1-" + str(target)
    count = estimate_record_count(target, nonce)
    for _ in range(3):
        case = (
            build_case(count, nonce)
            if task == "infra"
            else build_code_case(task, count, nonce)
        )
        template_body = {"messages": case["messages"]}
        if chat_template_kwargs is not None:
            template_body["chat_template_kwargs"] = chat_template_kwargs
        with http("/apply-template", template_body) as response:
            template = json.load(response)["prompt"]
        with http("/tokenize", {"content": template, "add_special": True}) as response:
            length = len(json.load(response)["tokens"])
        if abs(length - target) / target < 0.01:
            return case, length
        count = max(12, round(count * target / length))
    return case, length


def sampling_parameters(thinking=False, production_sampling=False):
    if production_sampling and not thinking:
        raise ValueError("--production-sampling requires --thinking")
    params = {
        "temperature": 0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if production_sampling:
        params.update(
            temperature=1,
            top_p=0.95,
            top_k=20,
            min_p=0,
            presence_penalty=0,
            repeat_penalty=1,
        )
        params["chat_template_kwargs"]["reasoning_effort"] = "xhigh"
    return params


def measure(
    case,
    model,
    cache=False,
    thinking=False,
    max_tokens=1024,
    deadline=None,
    production_sampling=False,
):
    body = {
        "model": model,
        "messages": case["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "cache_prompt": cache,
        **sampling_parameters(thinking, production_sampling),
    }
    begin = time.monotonic()
    first = None
    text = ""
    timing = None
    finish = None
    timeout = min(600, check_deadline(deadline)) if deadline is not None else 600
    with http("/v1/chat/completions", body, timeout=timeout) as response:
        for line in response:
            if deadline is not None:
                check_deadline(deadline)
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            item = json.loads(line[6:])
            if item.get("error"):
                raise RuntimeError("Inference error")
            timing = item.get("timings") or timing
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                token = delta.get("content") or ""
                if first is None and (token or delta.get("reasoning_content")):
                    first = time.monotonic() - begin
                text += token
                finish = choice.get("finish_reason") or finish
    if deadline is not None:
        check_deadline(deadline)
    if (
        not timing
        or first is None
        or any(
            k not in timing
            for k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second")
        )
    ):
        raise RuntimeError("Missing native timings or output")
    if not cache and timing.get("cache_n", 0) >= 100:
        raise RuntimeError("Cold sample reused a prefix")
    return {
        "ttft_seconds": first,
        "wall_seconds": time.monotonic() - begin,
        "timings": timing,
        "finish_reason": finish,
        "answer": text,
        "quality_pass": (
            (finish == "stop" and grade(text, case["expected"]))
            if "expected" in case
            else None
        ),
        "prompt_sha256": hashlib.sha256(
            json.dumps(body["messages"], sort_keys=True).encode()
        ).hexdigest(),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


WATCHDOG = r"""import json,re,signal,subprocess,sys,time
name,owner,production_id,image=sys.argv[1:5]
time.sleep(int(sys.argv[5]))
def expired(*_): raise TimeoutError('Watchdog recovery deadline exhausted')
signal.signal(signal.SIGALRM,expired)
signal.alarm(int(sys.argv[6]))
def inspect(name,optional=False):
 p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=30)
 if p.returncode:
  if optional and re.search(r'no such (object|container)',p.stderr,re.I): return None
  raise RuntimeError('Watchdog identity inspection failed')
 return json.loads(p.stdout)[0]
try:
 c=inspect(name,True)
 if c is not None:
  if (c['Config'].get('Labels') or {}).get('xt.prefill.owner')!=owner or c['Image']!=image:
   raise RuntimeError('Watchdog refuses unowned container')
  subprocess.run(['docker','rm','-f',c['Id']],check=True,stdout=subprocess.DEVNULL,timeout=45)
  if inspect(name,True) is not None: raise RuntimeError('Probe still exists')
 c=inspect(production_id)
 if c['Id']!=production_id or c['Image']!=image: raise RuntimeError('Retained production identity changed')
 subprocess.run(['docker','start',production_id],check=True,stdout=subprocess.DEVNULL,timeout=60)
finally:
 signal.alarm(0)
"""


def restore(production, name, owner, original, voices, watchdog):
    remove_owned_probe(name, owner, production["Image"])
    current = inspect_container(production["Id"])
    if current["Id"] != production["Id"] or current["Image"] != production["Image"]:
        raise RuntimeError("Retained production identity changed")
    run("docker", "start", production["Id"], timeout=60)
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        restored = inspect_container(production["Id"])
        if (
            restored["Id"] != production["Id"]
            or restored["Image"] != production["Image"]
        ):
            raise RuntimeError("Restored production identity changed")
        if (
            restored["State"]["Running"]
            and restored["State"].get("Health", {}).get("Status") == "healthy"
        ):
            actual = engines()
            if original in actual and all(v in actual for v in voices):
                if watchdog.poll() is None:
                    watchdog.terminate()
                watchdog.wait(timeout=5)
                emit(
                    "restored",
                    image=restored["Image"],
                    original_argv=True,
                    original_container=True,
                )
                return
        time.sleep(2)
    raise RuntimeError("Restoration unverified; watchdog retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned-drain", required=True)
    parser.add_argument("--variants", default="baseline,ub512,ub2048,baseline")
    parser.add_argument("--targets", default="8000,64000,128000,200000")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--candidate-template", type=Path)
    parser.add_argument("--tasks", default="infra")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--production-sampling", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    sampling = sampling_parameters(args.thinking, args.production_sampling)
    targets = [int(v) for v in args.targets.split(",")]
    variants = args.variants.split(",")
    tasks = args.tasks.split(",")
    if (
        any(t not in ("infra", "rust", "js") for t in tasks)
        or not 128 <= args.max_tokens <= 8192
    ):
        raise ValueError("Invalid task or output budget")
    for path in (args.engine_root, args.candidate_model, args.candidate_template):
        if path and (not path.is_absolute() or not path.exists() or "," in str(path)):
            raise ValueError("Experiment paths must be existing absolute mount paths")
    if any(v.startswith("gdn-") for v in variants) and not args.engine_root:
        raise ValueError("GDN experiments require an isolated compiled engine")
    if "nvfp4-ub1024" in variants and not args.candidate_model:
        raise ValueError("NVFP4 requires a separately verified model")
    experiment = (
        engine_provenance(
            args.engine_root, require_gdn=any(v.startswith("gdn-") for v in variants)
        )
        if args.engine_root
        else {}
    )
    production = json.loads(run("docker", "inspect", "ezlocalai"))[0]
    if not production["State"]["Running"] or args.repeats < 1:
        raise RuntimeError("Production must be running; repeats must be positive")
    all_engines = engines()
    qwen = [v for v in all_engines if any("Qwen3.8-27B" in a for a in v)]
    if len(qwen) != 1:
        raise RuntimeError("Expected exactly one Qwen engine")
    original = qwen[0]
    voices = [v for v in all_engines if any("LFM3" in a for a in v)]
    if len(voices) > 1 or any(
        t < 2000 or t + args.max_tokens + 1024 > int(original[original.index("-c") + 1])
        for t in targets
    ):
        raise ValueError("Invalid context or resident voice configuration")
    for variant in variants:
        variant_args(original, variant)
    for port in (18092, 18093):
        with socket.socket() as guard:
            guard.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            guard.bind(("127.0.0.1", port))
    env = dict(v.split("=", 1) for v in production["Config"]["Env"])
    forwarded = {
        k: v for k, v in env.items() if k.startswith(("GGML_", "CUDA_", "NVIDIA_"))
    }
    owner = args.owned_drain
    name = "prefill-probe-" + uuid.uuid4().hex
    model_path = original[original.index("-m") + 1]
    model_hash = run(
        "docker", "exec", production["Id"], "sha256sum", model_path
    ).split()[0]
    if args.candidate_model:
        experiment["candidate_sha256"] = run(
            "sha256sum", str(args.candidate_model)
        ).split()[0]
    if args.candidate_template:
        experiment["template_sha256"] = run(
            "sha256sum", str(args.candidate_template)
        ).split()[0]
    emit(
        "provenance",
        image=production["Image"],
        engine_args=original,
        voice_args=voices,
        model_sha256=model_hash,
        variants=variants,
        targets=targets,
        experiment=experiment,
        tasks=tasks,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        sampling_profile="production" if args.production_sampling else "greedy",
        sampling_parameters=sampling,
        gpu=run(
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,memory.total,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader",
        ),
    )
    watchdog = subprocess.Popen(
        [
            "python3",
            "-c",
            WATCHDOG,
            name,
            owner,
            production["Id"],
            production["Image"],
            str(WATCHDOG_SECONDS),
            str(RESTORE_SECONDS),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    def interrupted(*_):
        raise InterruptedError("Restore after signal")

    def expired(*_):
        raise TimeoutError("Maintenance phase deadline exhausted")

    previous = {}
    deadline = time.monotonic() + WORK_SECONDS
    cases = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            previous[sig] = signal.signal(
                sig, expired if sig == signal.SIGALRM else interrupted
            )
        signal.alarm(WORK_SECONDS)
        if watchdog.poll() is not None:
            raise RuntimeError("Independent recovery watchdog exited before cutover")
        run("docker", "stop", "--time", "45", production["Id"])
        if any(v.startswith("gdn-") for v in variants):
            check_operators(production, name, owner, args.engine_root)
        for block, variant in enumerate(variants):
            check_deadline(deadline)
            launch = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--label",
                "xt.prefill.owner=" + owner,
                "--network",
                "host",
                "--gpus",
                "all",
                "--volumes-from",
                production["Id"] + ":ro",
                "--entrypoint",
                "/prefill-bin/llama-server" if args.engine_root else original[0],
            ]
            if args.engine_root:
                launch += [
                    "--mount",
                    f"type=bind,src={args.engine_root},dst=/prefill-bin,readonly",
                    "--env",
                    "LD_LIBRARY_PATH=/prefill-bin:/usr/local/cuda/lib64",
                ]
            if args.candidate_model:
                launch += [
                    "--mount",
                    f"type=bind,src={args.candidate_model},dst=/prefill-candidate.gguf,readonly",
                ]
            if args.candidate_template:
                launch += [
                    "--mount",
                    f"type=bind,src={args.candidate_template},dst=/prefill-template.jinja,readonly",
                ]
            for key, value in forwarded.items():
                launch += ["--env", key + "=" + value]
            if variant.startswith("gdn-"):
                launch += [
                    "--env",
                    "GGML_CUDA_EXPERIMENTAL_GDN_SLICE64="
                    + ("1" if variant == "gdn-on" else "0"),
                ]
            engine_args = variant_args(original, variant)
            if args.candidate_template:
                engine_args += ["--chat-template-file", "/prefill-template.jinja"]
            run(*launch, production["Image"], *engine_args)
            wait_health(18092)
            for voice in voices:
                voice_args = voice.copy()
                voice_args[voice_args.index("--port") + 1] = "18093"
                run(
                    "docker",
                    "exec",
                    "-d",
                    "--env",
                    "LD_LIBRARY_PATH=" + env.get("LD_LIBRARY_PATH", ""),
                    name,
                    *voice_args,
                )
                wait_health(18093)
            if variant == "nvfp4-ub1024":
                free_mib = int(
                    run(
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    )
                )
                if free_mib < 1536:
                    raise RuntimeError(
                        "Candidate does not retain the 1.5 GiB VRAM reserve"
                    )
            emit(
                "ready",
                block=block,
                variant=variant,
                gpu=run(
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,temperature.gpu,power.draw",
                    "--format=csv,noheader",
                ),
            )
            for task, target in (
                (task, target) for task in tasks for target in targets
            ):
                check_deadline(deadline)
                if (task, target) not in cases:
                    cases[task, target] = calibrate(
                        target,
                        task,
                        chat_template_kwargs=sampling["chat_template_kwargs"],
                    )
                case, measured_tokens = cases[task, target]
                for repeat in range(args.repeats):
                    sample = measure(
                        case,
                        original[original.index("--alias") + 1],
                        thinking=args.thinking,
                        production_sampling=args.production_sampling,
                        max_tokens=args.max_tokens,
                        deadline=deadline,
                    )
                    emit(
                        "sample",
                        block=block,
                        variant=variant,
                        target_tokens=target,
                        task=task,
                        template_tokens=measured_tokens,
                        repeat=repeat,
                        kind="cold",
                        **sample,
                    )
                if not args.skip_cache:
                    sample = measure(
                        case,
                        original[original.index("--alias") + 1],
                        cache=True,
                        thinking=args.thinking,
                        production_sampling=args.production_sampling,
                        max_tokens=args.max_tokens,
                        deadline=deadline,
                    )
                    emit(
                        "sample",
                        block=block,
                        variant=variant,
                        target_tokens=target,
                        task=task,
                        template_tokens=measured_tokens,
                        repeat=0,
                        kind="repeat",
                        **sample,
                    )
            remove_owned_probe(name, owner, production["Image"])
    finally:
        signal.alarm(0)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(RESTORE_SECONDS)
        try:
            restore(production, name, owner, original, voices, watchdog)
        except BaseException as error:
            emit(
                "restoration_failed",
                error_type=type(error).__name__,
                watchdog_retained=watchdog.poll() is None,
            )
            raise
        finally:
            signal.alarm(0)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
