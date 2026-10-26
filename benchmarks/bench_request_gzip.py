"""Qualify negotiated request gzip on public source conversations.

The default mode measures the Python reference codec only. --run-route uses an
owned Rust router and CPU-only native worker with a deterministic mock engine.
A user-space loopback relay paces only request-body upload bytes; this is a
simulated link, not a production Wi-Fi or GPU throughput measurement.
"""
import argparse
import gzip
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib

import bench_worker_input as worker_http


MAX_BODY = 16 * 1024 * 1024
MIN_GZIP = 64 * 1024
RATE = 13_000_000
WORKER_ID = "gzip-benchmark"
MANIFEST = Path(__file__).resolve().parents[1] / "tests/fixtures/request_gzip_public_sources_20261026.json"
PUBLIC_CORPUS_REVISION = "c42148868a04d05171cdfc3bca7840f227b97216"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def strict_gzip(value, limit=MAX_BODY):
    if len(value) > limit:
        raise ValueError("compressed body exceeds limit")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    decoded = decoder.decompress(value, limit + 1)
    if len(decoded) > limit or decoder.unconsumed_tail:
        raise ValueError("decoded body exceeds limit")
    if not decoder.eof or decoder.unused_data:
        raise ValueError("gzip must contain exactly one complete member")
    return decoded


def load_corpus(path=MANIFEST):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    repositories = manifest["repositories"]
    if (len(repositories) != 1 or repositories[0]["id"] != "native"
            or repositories[0]["url"] != "https://github.com/DevXT-LLC/ezlocalai"
            or repositories[0]["revision"] != PUBLIC_CORPUS_REVISION or repositories[0]["license"] != "MIT"):
        raise ValueError("corpus must contain only the pinned public native repository")
    archive_name = manifest["archive"]
    if Path(archive_name).name != archive_name or manifest["schema_version"] != 1:
        raise ValueError("invalid public corpus manifest")
    archive = path.with_name(archive_name).read_bytes()
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("public corpus archive fingerprint differs")
    limit = manifest["archive_decoded_limit"]
    if type(limit) is not int or not 1 <= limit <= 2 * 1024 * 1024:
        raise ValueError("public corpus expansion limit differs")
    raw = strict_gzip(archive, limit)
    if len(raw) != manifest["archive_decoded_bytes"] or digest(raw) != manifest["archive_decoded_sha256"]:
        raise ValueError("decoded public corpus fingerprint differs")
    corpus = json.loads(raw)
    sources = corpus["sources"]
    if corpus["schema_version"] != 1 or len(sources) != len(manifest["sources"]):
        raise ValueError("public corpus source count differs")
    for source, expected in zip(sources, manifest["sources"]):
        content = source["content"].encode("utf-8")
        if (source["repo"] != "native" or {key: source[key] for key in expected} != expected
                or digest(content) != expected["sha256"] or len(content) != expected["bytes"]):
            raise ValueError("public source fingerprint differs")
    if sum(source["bytes"] for source in sources) != manifest["total_source_bytes"]:
        raise ValueError("public source byte count differs")
    if not 1 <= manifest["total_source_bytes"] <= 1024 * 1024:
        raise ValueError("public source corpus exceeds its bound")
    # Render each complete public source once; no private checkout is consulted.
    groups = {repo["id"]: iter([s for s in sources if s["repo"] == repo["id"]])
              for repo in manifest["repositories"]}
    ordered = []
    while groups:
        for name in list(groups):
            source = next(groups[name], None)
            if source is None:
                del groups[name]
            else:
                ordered.append("\n### " + name + ":" + source["path"] + "\n" + source["content"])
    return manifest, "".join(ordered)


def source_messages(corpus, byte_budget):
    raw = corpus.encode("utf-8")
    if byte_budget > len(raw):
        raise ValueError("public corpus is too short; repeated filler is forbidden")
    selected = raw[:byte_budget].decode("utf-8", errors="ignore")
    messages = [{"role": "system", "content": "Review the supplied public native-project source sections."}]
    # Each source byte appears only once in this conversation, split into turns.
    encoded = selected.encode("utf-8")
    cursor = 0
    while cursor < len(encoded):
        end = min(cursor + 24576, len(encoded))
        while end < len(encoded) and encoded[end] & 0xc0 == 0x80:
            end -= 1
        piece = encoded[cursor:end].decode("utf-8")
        messages.append({"role": "user" if len(messages) % 2 else "assistant", "content": piece})
        cursor = end
    messages.append({"role": "user", "content": "Summarize the review in one short sentence."})
    return messages


def cases(corpus):
    return {"public-source-" + str(size): source_messages(corpus, size)
            for size in (32768, 131072, 524288, 1000000)}


def payload(name, messages, streaming):
    return json.dumps({"model": worker_http.MODEL, "worker": WORKER_ID,
        "messages": messages, "stream": streaming, "max_tokens": 32,
        "temperature": 0, "seed": 42, "stop": ["STOP"],
        "routing_affinity_key": "gzip-public-fixture", "prompt_cache_key": "gzip-" + name,
        "disable_fallback": False}, ensure_ascii=False, separators=(",", ":")).encode()


def codec_measurements(corpus, samples=7):
    inputs = {name: payload(name, messages, False) for name, messages in cases(corpus).items()}
    # This raw-byte diagnostic is deliberately not sent to a JSON endpoint.
    inputs["incompressible-raw-131072"] = hashlib.shake_256(b"public-gzip-codec-control-v1").digest(131072)
    rows = []
    for name, raw in inputs.items():
        encoded = gzip.compress(raw, compresslevel=1, mtime=0)
        if strict_gzip(encoded) != raw:
            raise RuntimeError("reference codec changed bytes")
        encode_times, decode_times = [], []
        for _ in range(samples):
            started = time.perf_counter()
            trial = gzip.compress(raw, compresslevel=1, mtime=0)
            encode_times.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            decoded = strict_gzip(trial)
            decode_times.append((time.perf_counter() - started) * 1000)
            if decoded != raw:
                raise RuntimeError("reference codec changed bytes")
        rows.append({"case": name, "json_request": not name.startswith("incompressible-raw"),
            "input_bytes": len(raw), "gzip_bytes": len(encoded), "input_sha256": digest(raw),
            "gzip_sha256": digest(encoded), "decoded_sha256": digest(raw),
            "gzip_fraction": len(encoded) / len(raw),
            "eligible_and_smaller": MIN_GZIP <= len(raw) <= MAX_BODY and len(encoded) < len(raw),
            "samples": samples, "median_python_gzip1_encode_ms": statistics.median(encode_times),
            "median_python_gzip_decode_ms": statistics.median(decode_times),
            "python_gzip1_encode_samples_ms": encode_times, "python_gzip_decode_samples_ms": decode_times})
    return {"scope": "Python stdlib gzip/zlib diagnostic; not isolated Rust/C++ codec timings",
            "python_version": sys.version.split()[0], "zlib_version": zlib.ZLIB_VERSION,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION, "cases": rows}


def pace_send(send, body, rate=RATE, clock=time.monotonic, sleep=time.sleep, block=16384):
    if type(rate) is not int or rate < 0 or type(block) is not int or block <= 0:
        raise ValueError("nonnegative integer rate and positive block required")
    if rate == 0:
        for offset in range(0, len(body), block):
            send(body[offset:offset + block])
        return len(body)
    started = clock()
    sent = 0
    for offset in range(0, len(body), block):
        part = body[offset:offset + block]
        deadline = started + (sent + len(part)) / rate
        delay = deadline - clock()
        if delay > 0:
            sleep(delay)
        send(part)
        sent += len(part)
    return sent


class UploadRelay:
    """An owned loopback proxy pacing body bytes before forwarding them."""
    def __init__(self, target_port, expected_key, rate=RATE):
        self.records, self.errors = [], []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.settimeout(30)

            def log_message(self, *args):
                pass

            def do_POST(self):
                if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + expected_key):
                    self.send_error(401)
                    return
                if self.path != "/v1/chat/completions" or self.headers.get("Transfer-Encoding"):
                    self.send_error(400)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                    if not 0 <= size <= MAX_BODY:
                        raise ValueError("wire body length exceeds benchmark bound")
                    body = self.rfile.read(size)
                    if len(body) != size:
                        raise ValueError("incomplete benchmark upload")
                    encoding = self.headers.get("Content-Encoding", "")
                    connection = HTTPConnection("127.0.0.1", target_port, timeout=30)
                    try:
                        connection.putrequest("POST", self.path, skip_host=True, skip_accept_encoding=True)
                        for key, value in self.headers.items():
                            if key.lower() not in ("connection", "transfer-encoding"):
                                connection.putheader(key, value)
                        connection.putheader("Connection", "close")
                        connection.endheaders()
                        paced = time.perf_counter()
                        pace_send(connection.send, body, rate)
                        upload_seconds = time.perf_counter() - paced
                        response = connection.getresponse()
                        self.send_response(response.status)
                        for key, value in response.getheaders():
                            if key.lower() not in ("content-length", "transfer-encoding", "connection", "server", "date"):
                                self.send_header(key, value)
                        self.send_header("Transfer-Encoding", "chunked")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        while True:
                            part = response.read1(16384)
                            if not part:
                                break
                            self.wfile.write((format(len(part), "x") + "\r\n").encode() + part + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    finally:
                        connection.close()
                    # Verification belongs to the main thread after the client
                    # timer stops. Even work after flush can otherwise contend
                    # for Python's GIL before the timed client has returned.
                    owner.records.append({"path": self.path, "content_encoding": encoding,
                        "_body": body,
                        "upload_seconds": upload_seconds, "rate_bytes_per_second": rate,
                        "authentication_forwarded": True})
                except Exception as error:
                    owner.errors.append(type(error).__name__)
                    self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


def owns_listener(pid, port, proc=Path("/proc")):
    """Prove an ephemeral listener belongs to this invocation's process."""
    inodes = set()
    for line in (proc / str(pid) / "net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) > 9 and fields[1] == "0100007F:" + format(port, "04X") and fields[3] == "0A":
            inodes.add(fields[9])
    for fd in (proc / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if any(target == "socket:[" + inode + "]" for inode in inodes):
            return True
    return False


def wait_ready(base, pid, alive, timeout=30):
    port = int(base.rsplit(":", 1)[-1])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive():
            raise RuntimeError("owned benchmark server exited during startup")
        try:
            if owns_listener(pid, port):
                with worker_http.open_loopback(base + "/health", timeout=1) as response:
                    if response.status == 200 and alive():
                        return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.05)
    raise RuntimeError("owned benchmark server did not become ready")


class RouterProcess:
    def __init__(self, binary, data, client_key, register_key):
        self.base = "http://127.0.0.1:" + str(free_port())
        port = self.base.rsplit(":", 1)[-1]
        self.log = (data / "router.log").open("w")
        (data / ".env").write_text("")
        env = {key: value for key, value in os.environ.items() if not key.startswith(("UVICORN_", "ROUTER_", "CHUTES_", "OPENROUTER_"))}
        env.update({"HOST": "127.0.0.1", "PORT": port, "LOG_LEVEL": "WARNING",
            "EZLOCALAI_API_KEY": client_key, "ROUTER_REGISTER_KEY": register_key,
            "CHUTES_API_KEY": "", "OPENROUTER_API_KEY": "", "REQUEST_TIMEOUT": "30",
            "ROUTER_WAIT_TIMEOUT": "2", "ROUTER_RESERVATION_TTL": "15",
            "ROUTER_WORKER_TTL": "3600", "ROUTER_APP_DIR": str(data),
            "ROUTER_RESOLVE_WORKING_MODEL": "false", "TOKIO_WORKER_THREADS": "2",
            "USAGE_FILE": str(data / "usage.json"), "ROUTER_ERROR_FILE": str(data / "errors.json"),
            "ROUTER_DRAIN_FILE": str(data / "drains.json"), "ROUTER_OUTPUTS_DIR": str(data / "outputs")})
        self.process = subprocess.Popen([str(binary)], cwd=data, env=env, stdout=self.log, stderr=self.log)

    def __enter__(self):
        try:
            wait_ready(self.base, self.process.pid, lambda: self.process.poll() is None)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *args):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log.close()


def router_request(base, path, key, value=None):
    request = urllib.request.Request(base + path,
        data=None if value is None else json.dumps(value).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with worker_http.open_loopback(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict) or value.get("error"):
        raise RuntimeError("isolated router request failed")
    return value


def wait_idle(router, client_key):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rows = router_request(router.base, "/v1/router/workers", client_key)["data"]
        if len(rows) == 1 and not rows[0].get("active_requests") and not rows[0].get("router_in_flight"):
            return
        time.sleep(0.01)
    raise RuntimeError("isolated router reservation did not release")


def register(router, register_key, worker_key, relay_port, gzip_enabled):
    slot = {"capacity": 1, "in_flight": 0, "queued": 0, "instances": 1}
    extra = {"http_request_encodings": ["gzip"]} if gzip_enabled else {}
    return router_request(router.base, "/v1/router/register", register_key,
        {"worker_id": WORKER_ID, "label": WORKER_ID,
         "url": "http://127.0.0.1:" + str(relay_port), "port": relay_port,
         "api_key": worker_key, "models": [worker_http.MODEL], "capabilities": ["text"],
         "best_tier": 90, "queue_capacity": 1, "in_flight": 0, "queue_depth": 0,
         "cap_slots": {"text": slot}, "model_slots": {worker_http.MODEL: slot},
         "model_context": {worker_http.MODEL: 1048576}, "extra": extra})


def checked_record(relay, before, expected, encoding):
    deadline = time.monotonic() + 3
    while len(relay.records) == before and not relay.errors and time.monotonic() < deadline:
        time.sleep(0.001)
    if relay.errors or len(relay.records) != before + 1:
        raise RuntimeError("one routed request must create exactly one valid relay request")
    pending = relay.records[-1]
    if pending["content_encoding"] != encoding or encoding not in ("", "gzip"):
        raise RuntimeError("negotiated request encoding differs")
    # full_route invokes this only after worker_http.measure has returned. Drop
    # the raw bytes from the queue and expose only JSON-safe identity evidence.
    body = pending.pop("_body")
    decoded = strict_gzip(body) if encoding == "gzip" else body
    value = json.loads(decoded)
    if value.get("disable_fallback") is not True:
        raise RuntimeError("final router request lost disable_fallback")
    record = {"path": pending["path"], "content_encoding": encoding,
        "wire_bytes": len(body), "wire_sha256": digest(body),
        "decoded_bytes": len(decoded), "decoded_sha256": digest(decoded),
        "upload_seconds": pending["upload_seconds"], "rate_bytes_per_second": pending["rate_bytes_per_second"],
        "disable_fallback": True, "authentication_forwarded": pending["authentication_forwarded"]}
    identity = {key: record[key] for key in ("decoded_bytes", "decoded_sha256", "path", "disable_fallback")}
    if expected is not None and identity != expected:
        raise RuntimeError("decoded final router request bytes differ")
    rate = record["rate_bytes_per_second"]
    if rate and record["upload_seconds"] + 1e-9 < record["wire_bytes"] / rate:
        raise RuntimeError("simulated upload exceeded its configured rate")
    relay.records[-1] = record
    return record, identity


def full_route(args, corpus):
    worker_http.immutable_image(args.image)
    for binary in (args.worker_binary, args.router_binary):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("benchmark binaries must be executable files")
    client_key, register_key = secrets.token_hex(32), secrets.token_hex(32)
    worker_key, engine_key, owner = secrets.token_hex(32), secrets.token_hex(32), uuid.uuid4().hex
    names = []
    report = {"scope": "actual Rust router and native worker with a mock engine and an owned loopback body-upload relay",
        "gpu_inference": False, "production_wifi_measurement": False,
        "link_mode": "simulated-rate-limited" if args.bytes_per_second else "unshaped-loopback-control",
        "rate_bytes_per_second": args.bytes_per_second,
        "rate_scope": "request body upload only, paced from its first byte" if args.bytes_per_second else "no artificial sleeps or rate limit; loopback proxy overhead remains",
        "image_id": args.image, "worker_sha256": digest(args.worker_binary.read_bytes()),
        "router_sha256": digest(args.router_binary.read_bytes()), "blocks": args.blocks,
        "samples_per_variant_case_mode": args.blocks * 2, "cases": []}
    with tempfile.TemporaryDirectory(prefix="ezlocalai-request-gzip-") as temporary, worker_http.MockEngine(engine_key) as engine:
        temporary = Path(temporary)
        try:
            worker_data, router_data = temporary / "worker", temporary / "router"
            worker_data.mkdir()
            router_data.mkdir()
            port = free_port()
            environment = worker_http.environment(port, engine.server.server_port, worker_key, engine_key)
            name = "ezlocalai-gzipbench-" + owner[:12]
            names.append(name)
            command = worker_http.docker_command(args.image, args.worker_binary, worker_data,
                environment, name, owner, port)
            command.insert(2, "--pull=never")
            worker_http.docker(command[1:], worker_environment=environment)
            observed = json.loads(worker_http.docker(["inspect", name]).stdout)[0]
            worker_http.verify_isolation(observed, owner, args.image, environment)
            pid = observed["State"]["Pid"]
            wait_ready("http://127.0.0.1:" + str(port), pid,
                lambda: bool(json.loads(worker_http.docker(["inspect", name]).stdout)[0]["State"]["Running"]))
            with UploadRelay(port, worker_key, args.bytes_per_second) as relay, RouterProcess(
                    args.router_binary, router_data, client_key, register_key) as router:
                registered = False
                for case, messages in cases(corpus).items():
                    for streaming in (False, True):
                        body = payload(case, messages, streaming)
                        expected, engine_identity = None, None
                        samples = {"identity": [], "gzip": []}
                        order = ["identity", "gzip"] + list(worker_http.blocks(args.blocks))
                        # The reused ABBA helper names its first variant baseline.
                        order = ["identity" if value == "baseline" else "gzip" if value == "candidate" else value for value in order]
                        for index, variant in enumerate(order):
                            if registered:
                                wait_idle(router, client_key)
                            register(router, register_key, worker_key, relay.port, variant == "gzip")
                            registered = True
                            before, engine_before = len(relay.records), len(engine.records)
                            sample = worker_http.measure(router.base, client_key, body, streaming)
                            encoding = "gzip" if variant == "gzip" and len(body) >= MIN_GZIP else ""
                            record, expected = checked_record(relay, before, expected, encoding)
                            engine_identity = worker_http.checked_capture(engine.records, engine_before, engine_identity)
                            if index >= 2:
                                sample["upload"] = record
                                samples[variant].append(sample)
                        wait_idle(router, client_key)
                        row = {"case": case, "streaming": streaming, "request_bytes": len(body),
                            "request_sha256": digest(body), "decoded_router_payload": expected,
                            "engine_payload": engine_identity, "samples": samples,
                            "median_full_ms": {variant: statistics.median(value["full_ms"] for value in values)
                                               for variant, values in samples.items()}}
                        if streaming:
                            row["median_first_output_ms"] = {variant: statistics.median(value["first_output_ms"] for value in values)
                                                              for variant, values in samples.items()}
                        report["cases"].append(row)
                        print(json.dumps({key: value for key, value in row.items() if key != "samples"}), flush=True)
        finally:
            worker_http.remove_all_owned(names, owner)
    report.update(owned_containers_removed=True, owned_router_stopped=True)
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-route", action="store_true", help="explicitly launch owned local benchmark processes/containers")
    parser.add_argument("--image")
    parser.add_argument("--worker-binary", type=Path)
    parser.add_argument("--router-binary", type=Path)
    parser.add_argument("--blocks", type=int, choices=range(1, 13), default=6)
    parser.add_argument("--bytes-per-second", type=int, choices=(0, RATE), default=RATE,
                        help="0 for an unshaped loopback control;13000000 for simulated body upload")
    args = parser.parse_args(argv)
    if args.run_route:
        if args.image is None or args.worker_binary is None or args.router_binary is None:
            parser.error("--run-route requires immutable --image and both worker/router binaries")
        args.worker_binary = args.worker_binary.resolve()
        args.router_binary = args.router_binary.resolve()
    elif any(value is not None for value in (args.image, args.worker_binary, args.router_binary)):
        parser.error("binary/image arguments require explicit --run-route")
    return args


def main(argv=None):
    args = arguments(argv)
    manifest, corpus = load_corpus(args.manifest)
    report = {"schema_version": 1, "scope": "public-corpus codec diagnostics; optional owned simulated-link route",
        "manifest_sha256": digest(args.manifest.read_bytes()), "archive_sha256": manifest["archive_sha256"],
        "source_files": len(manifest["sources"]), "source_bytes": manifest["total_source_bytes"],
        "benchmark_sha256": digest(Path(__file__).read_bytes()), "python_codec": codec_measurements(corpus)}
    if args.run_route:
        report["full_route"] = full_route(args, corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request-gzip.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
