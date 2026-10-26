"""Compare actual worker HTTP preparation with an owned loopback mock engine.

This launches two CPU-only containers from one immutable image. Only their worker
executables differ. It measures HTTP latency, not model quality or GPU throughput.
All containers, temporary data and the mock engine are owned by this invocation.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.parse
import uuid


OWNER_LABEL = "devxt.ezlocalai.worker-input-benchmark"
MODEL = "input-benchmark"
ANSWER = "fixture benchmark response"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def open_loopback(request, timeout):
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password:
        raise ValueError("benchmark clients must use direct loopback HTTP")
    return DIRECT_OPENER.open(request, timeout=timeout)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def immutable_image(value):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("image must be an immutable local sha256 image ID")
    return value


def cases():
    def text(size):
        unit = 'Reference text with a quoted "value", slash \\ and Unicode café.\n'
        return (unit * (size // len(unit) + 1))[:size]

    result = {}
    for size in (1024, 102400, 1048576):
        result["text-" + str(size)] = [{"role": "user", "content": text(size)}]
    for count, size in ((16, 8192), (256, 4096)):
        result["messages-" + str(count)] = [
            {"role": "assistant" if i % 2 else "user", "content": text(size)}
            for i in range(count - 1)] + [{"role": "user", "content": "current turn"}]
    result["typed-last-256"] = [dict(item) for item in result["messages-256"]]
    result["typed-last-256"][-1]["content"] = ["current ", {"type": "text", "text": "turn"}]
    result["typed-all-256"] = [
        {"role": item["role"], "content": [{"type": "text", "text": item["content"]}]}
        for item in result["messages-256"]]
    return result


def payload(name, messages, streaming):
    return json.dumps({"model": MODEL, "messages": messages, "stream": streaming,
                       "max_tokens": 32, "temperature": 0, "seed": 42,
                       "prompt_cache_key": "http-input-" + name,
                       "stop": ["STOP"]}, ensure_ascii=False, separators=(",", ":")).encode()


def environment(port, engine_port, api_key, engine_key):
    return {
        "HOST": "127.0.0.1", "PORT": str(port), "EZLOCALAI_PORT": str(port),
        "EZLOCALAI_DOTENV_PATH": "", "EZLOCALAI_URL": "http://input-benchmark.invalid",
        "EZLOCALAI_API_KEY": api_key, "LLAMA_CPP_API_KEY": engine_key,
        "EZLOCALAI_NATIVE_BACKEND": "llama", "DEFAULT_MODEL": MODEL,
        "LLAMA_CPP_SERVER_URL": "http://127.0.0.1:" + str(engine_port),
        "LLAMA_CPP_MMPROJ_PATH": "", "LLAMA_CPP_MODEL_PATH": "",
        "LLAMA_CPP_MODEL_PATHS": "", "GPU_LAYERS": "0",
        "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        "ROUTER_URL": "", "ROUTER_API_KEY": "", "WORKER_TUNNEL": "false",
        "TEXT_SERVER": "", "FALLBACK_SERVER": "", "FALLBACK_API_KEY": "",
        "CHUTES_API_KEY": "", "OPENROUTER_API_KEY": "", "VOICE_SERVER": "",
        "IMAGE_ENABLED": "false", "VIDEO_ENABLED": "false", "MUSIC_ENABLED": "false",
        "TTS_ENABLED": "false", "STT_ENABLED": "false", "EMBEDDING_ENABLED": "false",
        "LLM_MODEL_RESIDENCY": "resident", "LLM_SPECULATIVE_TYPE": "none",
        "LLM_PROMPT_CACHE_MIB": "0", "LLM_MAX_TOKENS": "1048576",
        "LLAMA_CPP_CONTEXT": "1048576", "LLM_MAX_OUTPUT_TOKENS": "32",
        "N_PARALLEL": "1", "LOG_LEVEL": "WARNING", "HUGGING_FACE_HUB_TOKEN": "",
        "HF_TOKEN": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }


def docker_command(image, binary, data, env_file, name, owner, port):
    immutable_image(image)
    return ["docker", "run", "--detach", "--name", name, "--network", "host",
            "--runtime", "runc", "--label", OWNER_LABEL + "=" + owner,
            "--user", str(os.getuid()) + ":" + str(os.getgid()),
            "--env-file", str(env_file), "--workdir", "/benchmark/data",
            "--mount", "type=bind,src=" + str(binary) + ",dst=/benchmark/worker,readonly",
            "--mount", "type=bind,src=" + str(data) + ",dst=/benchmark/data",
            "--entrypoint", "/benchmark/worker", image,
            "--host", "127.0.0.1", "--port", str(port)]


def docker(arguments, required=True):
    result = subprocess.run(["docker", *arguments], capture_output=True, text=True)
    if required and result.returncode:
        raise RuntimeError("docker " + arguments[0] + " failed; no external resources were modified")
    return result


def owned_container(record, owner):
    if record.get("Config", {}).get("Labels", {}).get(OWNER_LABEL) != owner:
        raise RuntimeError("refusing to use or remove a container without this invocation's ownership label")
    container_id = record.get("Id", "")
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise RuntimeError("owned container has an invalid immutable ID")
    return container_id


def verify_isolation(record, owner, image=None, expected_environment=None):
    owned_container(record, owner)
    host = record["HostConfig"]
    if (host.get("NetworkMode") != "host" or host.get("Runtime") != "runc"
            or host.get("Privileged") or host.get("Devices") or host.get("DeviceRequests")):
        raise RuntimeError("benchmark container did not retain CPU-only isolation")
    if record["Config"]["Cmd"][:2] != ["--host", "127.0.0.1"]:
        raise RuntimeError("benchmark worker must bind only to loopback")
    if image is not None and record.get("Image") != image:
        raise RuntimeError("benchmark worker did not use the selected immutable image")
    observed = dict(item.split("=", 1) for item in record["Config"].get("Env", []))
    if expected_environment is not None and any(observed.get(key) != value
            for key, value in expected_environment.items()):
        raise RuntimeError("benchmark worker environment differs from the isolated plan")


def remove_owned(name, owner):
    result = docker(["inspect", name], required=False)
    if result.returncode:
        missing = {"Error: No such object: " + name,
                   "Error response from daemon: No such container: " + name}
        if result.stderr.strip().casefold() in {message.casefold() for message in missing}:
            return
        raise RuntimeError("could not inspect benchmark container during cleanup: " + name)
    record = json.loads(result.stdout)[0]
    # Remove by immutable ID after checking ownership, never by a reusable name.
    docker(["rm", "--force", owned_container(record, owner)])


def remove_all_owned(names, owner):
    failed = []
    for name in reversed(names):
        try:
            remove_owned(name, owner)
        except Exception:
            failed.append(name)
    if failed:
        raise RuntimeError("could not remove owned benchmark containers: " + ", ".join(failed))


class MockEngine:
    def __init__(self, key):
        self.records = []
        records = self.records

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def log_message(self, *args):
                pass

            def answer(self, body, content_type="application/json", status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def authorized(self):
                if secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
                    return True
                self.answer(b'{"error":"unauthorized"}', status=401)
                return False

            def do_GET(self):
                if self.authorized():
                    self.answer(json.dumps({"data": [{"id": MODEL}]}).encode())

            def do_POST(self):
                if not self.authorized():
                    return
                if self.path not in ("/v1/chat/completions", "/v1/completions"):
                    self.answer(b'{"error":"unexpected endpoint"}', status=404)
                    return
                count = int(self.headers.get("Content-Length", "0"))
                if count <= 0 or count > 8 * 1024 * 1024:
                    self.answer(b'{"error":"invalid length"}', status=400)
                    return
                body = self.rfile.read(count)
                value = json.loads(body)
                records.append({"sha256": digest(body), "bytes": len(body), "path": self.path})
                if value.get("stream"):
                    chunks = [{"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                              {"choices": [{"delta": {"content": ANSWER}, "finish_reason": None}]},
                              {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
                    data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                    self.answer(data.encode(), "text/event-stream")
                else:
                    self.answer(json.dumps({"model": MODEL, "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": ANSWER},
                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 1,
                        "completion_tokens": 1, "total_tokens": 2}}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def measure(url, key, body, streaming, timeout=30, clock=time.perf_counter):
    request = urllib.request.Request(url + "/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = clock()
    first = None
    content, reasons = [], []
    done = False
    with open_loopback(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError("worker returned non-success status")
        if streaming:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    continue
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError("worker stream returned an error")
                for choice in chunk.get("choices", []):
                    text = choice.get("delta", {}).get("content") or ""
                    if text:
                        if first is None:
                            first = clock()
                        content.append(text)
                    if choice.get("finish_reason") is not None:
                        reasons.append(choice["finish_reason"])
        else:
            value = json.load(response)
            choice = value["choices"][0]
            content = [choice["message"]["content"]]
            reasons = [choice["finish_reason"]]
    finished = clock()
    if "".join(content) != ANSWER or reasons != ["stop"] or (streaming and not done):
        raise RuntimeError("worker output identity or normal stream completion changed")
    elapsed = (finished - started) * 1000
    ttft = (first - started) * 1000 if first is not None else None
    if not math.isfinite(elapsed) or elapsed < 0 or (ttft is not None and
            (not math.isfinite(ttft) or ttft < 0 or ttft > elapsed)):
        raise RuntimeError("invalid latency measurement")
    return {"full_ms": elapsed, "first_output_ms": ttft,
            "output_sha256": digest(json.dumps({"content": ANSWER, "finish_reason": "stop"}).encode())}


def checked_capture(records, before, expected):
    if len(records) != before + 1:
        raise RuntimeError("each worker request must make exactly one mock-engine request")
    captured = records[-1]
    if expected is not None and captured != expected:
        raise RuntimeError("forwarded engine payload bytes differ across variants or repetitions")
    return captured


def blocks(count):
    for _ in range(count):
        yield from ("baseline", "candidate", "candidate", "baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--baseline-binary", required=True, type=Path)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=6, choices=range(1, 13))
    args = parser.parse_args()
    immutable_image(args.image)
    binaries = {name: path.resolve() for name, path in
                (("baseline", args.baseline_binary), ("candidate", args.candidate_binary))}
    for binary in binaries.values():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("both worker executables must be readable executable files")
    args.output.mkdir(parents=True, exist_ok=True)
    api_key, engine_key, owner = secrets.token_hex(32), secrets.token_hex(32), uuid.uuid4().hex
    names = []
    report = {"scope": "actual worker HTTP with a deterministic loopback mock engine; no GPU inference",
              "image_id": args.image, "worker_sha256": {name: digest(path.read_bytes())
              for name, path in binaries.items()}, "blocks": args.blocks,
              "samples_per_variant_per_case_mode": args.blocks * 2, "cases": []}
    with tempfile.TemporaryDirectory(prefix="ezlocalai-worker-input-") as temporary, MockEngine(engine_key) as engine:
        try:
            urls = {}
            for variant, binary in binaries.items():
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                data = Path(temporary, variant)
                data.mkdir()
                env_file = Path(temporary, variant + ".env")
                planned_environment = environment(port, engine.server.server_port, api_key, engine_key)
                env_file.write_text("".join(key + "=" + value + "\n" for key, value in planned_environment.items()))
                env_file.chmod(0o600)
                name = "ezlocalai-inputbench-" + owner[:12] + "-" + variant
                names.append(name)
                command = docker_command(args.image, binary, data, env_file, name, owner, port)
                docker(command[1:])
                record = json.loads(docker(["inspect", name]).stdout)[0]
                verify_isolation(record, owner, args.image, planned_environment)
                urls[variant] = "http://127.0.0.1:" + str(port)
                deadline = time.monotonic() + 30
                while True:
                    try:
                        with open_loopback(urls[variant] + "/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("owned benchmark worker did not become healthy")
                    time.sleep(0.05)
            for case, messages in cases().items():
                for streaming in (False, True):
                    body = payload(case, messages, streaming)
                    expected = None
                    samples = {name: [] for name in binaries}
                    # Warm both workers and check payload identity before timing.
                    for variant in binaries:
                        before = len(engine.records)
                        measure(urls[variant], api_key, body, streaming)
                        expected = checked_capture(engine.records, before, expected)
                    for variant in blocks(args.blocks):
                        before = len(engine.records)
                        sample = measure(urls[variant], api_key, body, streaming)
                        checked_capture(engine.records, before, expected)
                        samples[variant].append(sample)
                    result = {"case": case, "streaming": streaming, "request_bytes": len(body),
                              "request_sha256": digest(body), "engine_payload": expected, "samples": samples,
                              "median_full_ms": {name: statistics.median(item["full_ms"] for item in values)
                                                 for name, values in samples.items()}}
                    if streaming:
                        result["median_first_output_ms"] = {name: statistics.median(item["first_output_ms"]
                            for item in values) for name, values in samples.items()}
                    report["cases"].append(result)
                    print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)
        finally:
            remove_all_owned(names, owner)
    report["containers_removed"] = True
    args.output.joinpath("worker-input.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
