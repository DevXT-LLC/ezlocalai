#!/usr/bin/env python3
"""Linux loopback HTTP transport A/B using the Python standard library.

Start both immutable worker binaries with VOICE_SERVER=http://127.0.0.1:PORT,
where PORT is --voice-port. This driver owns only that fake voice listener;
it neither starts nor changes workers. Existing workers must have the isolated
Echo configuration checked below. No model or GPU work is performed.
"""

import argparse
import functools
import hashlib
import ipaddress
import json
import os
import platform
import re
import socket
import socketserver
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for part in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def assignments(values, parser, option):
    result = {}
    for value in values:
        if "=" not in value:
            parser.error(f"{option} requires label=value")
        label, value = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in result:
            parser.error(f"{option} labels must be unique and contain only letters, digits, _, . or -")
        result[label] = value
    return result


def worker_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.path not in ("", "/") or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("worker URLs must be plain HTTP loopback roots without credentials/query/fragment")
    host = parsed.hostname
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("only loopback worker addresses are allowed")
    return host, parsed.port or 80


def process_stat(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"start_ticks": int(fields[19]), "cpu_ticks": int(fields[11]) + int(fields[12])}


def guarded_worker(label, url, pid, voice_port):
    host, port = worker_url(url)
    environment = dict(entry.split(b"=", 1) for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in entry)
    expected = {
        b"EZLOCALAI_NATIVE_BACKEND": b"echo",
        b"EZLOCALAI_API_KEY": b"transport-bench",
        b"VOICE_SERVER": f"http://127.0.0.1:{voice_port}".encode(),
        b"TTS_ENABLED": b"true",
        b"NVIDIA_VISIBLE_DEVICES": b"void",
        b"CUDA_VISIBLE_DEVICES": b"",
        b"STT_ENABLED": b"false",
        b"IMAGE_ENABLED": b"false",
        b"VIDEO_ENABLED": b"false",
        b"MUSIC_ENABLED": b"false",
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            # Never echo arbitrary environment values, keys or authentication.
            raise ValueError(f"{label}: isolated worker configuration check failed for {key.decode()}")
    model = environment.get(b"DEFAULT_MODEL", b"").decode()
    if model != "transport-bench" and not re.fullmatch(r"transport-bench@[0-9]+", model):
        raise ValueError(f"{label}: DEFAULT_MODEL must be the isolated transport-bench model")
    for key in (b"ROUTER_URL", b"FALLBACK_SERVER", b"TEXT_SERVER", b"EMBEDDING_SERVER", b"IMAGE_SERVER"):
        if environment.get(key):
            raise ValueError(f"{label}: external forwarding/registration must be disabled ({key.decode()})")
    socket_inodes = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            socket_inodes.add(target[8:-1])
    owns_listener = False
    for table in ("tcp", "tcp6"):
        for line in Path(f"/proc/{pid}/net/{table}").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port and fields[9] in socket_inodes:
                owns_listener = True
    if not owns_listener:
        raise ValueError(f"{label}: loopback URL port is not owned by the verified worker PID")
    executable = Path(f"/proc/{pid}/exe")
    name = Path(os.readlink(executable)).name
    if not name.startswith("ezlocalai-native"):
        raise ValueError(f"{label}: PID executable is not an isolated native worker")
    state = process_stat(pid)
    binary_hash = file_sha256(executable)
    return {
        "pid": pid, "process_start_ticks": state["start_ticks"],
        "binary_name": name, "binary_sha256": binary_hash,
        "backend": "echo", "model": model, "host": host, "port": port,
        "voice_server": f"http://127.0.0.1:{voice_port}",
        "cpu_affinity": sorted(os.sched_getaffinity(pid)),
    }


def cpu_ticks(worker):
    state = process_stat(worker["pid"])
    if state["start_ticks"] != worker["process_start_ticks"]:
        raise RuntimeError("verified worker process identity changed during benchmark")
    return state["cpu_ticks"]


@functools.lru_cache(None)
def payload(size):
    pattern = bytes(range(251))
    return b"RIFF" + (pattern * ((size + 250) // 251))[:size - 4]


@functools.lru_cache(None)
def chunked(size, chunk_size):
    data = payload(size)
    return b"".join(
        f"{len(data[pos:pos + chunk_size]):x};bench=yes\r\n".encode()
        + data[pos:pos + chunk_size] + b"\r\n"
        for pos in range(0, len(data), chunk_size)
    ) + b"0\r\nX-End: yes\r\n\r\nignored trailing bytes"


@functools.lru_cache(None)
def voice_response(size, chunk_size):
    body = chunked(size, chunk_size) if chunk_size else payload(size)
    framing = b"Transfer-Encoding: chunked\r\n" if chunk_size else f"Content-Length: {size}\r\n".encode()
    return b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\n" + framing + b"Connection: close\r\n\r\n" + body


class VoiceHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(30)
        raw = b""
        while b"\r\n\r\n" not in raw:
            part = self.request.recv(4096)
            if not part:
                raise RuntimeError("incomplete fake voice request headers")
            raw += part
        header, body = raw.split(b"\r\n\r\n", 1)
        length = next((int(line.split(b":", 1)[1].strip()) for line in header.split(b"\r\n")
                       if line.lower().startswith(b"content-length:")), 0)
        while len(body) < length:
            part = self.request.recv(length - len(body))
            if not part:
                raise RuntimeError("incomplete fake voice request body")
            body += part
        if header.startswith(b"GET /v1/audio/voices "):
            body = b'{"voices":["availability"]}'
            response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        elif header.startswith(b"POST /v1/audio/speech "):
            marker = re.fullmatch(r"Native ezlocalai echo backend received: transport-bench:([0-9]+):([0-9]+)", json.loads(body)["input"])
            if marker is None:
                raise RuntimeError("unexpected fake voice input")
            response = voice_response(int(marker.group(1)), int(marker.group(2)))
        else:
            raise RuntimeError("unexpected fake voice route")
        self.request.sendall(response)


class VoiceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def receive_response(peer):
    parts = []
    while True:
        try:
            part = peer.recv(65536)
        except ConnectionResetError:
            break
        if not part:
            break
        parts.append(part)
    raw = b"".join(parts)
    header, body = raw.split(b"\r\n\r\n", 1)
    if not header.startswith(b"HTTP/1.1 200 "):
        raise AssertionError(header)
    return header, body


def connect(url):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.path not in ("", "/"):
        raise ValueError("benchmark URLs must be plain HTTP worker roots")
    peer = socket.create_connection((parsed.hostname, parsed.port or 80), timeout=30)
    peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return peer


def exchange(url, raw, expected, worker=None):
    started = time.perf_counter_ns()
    with connect(url) as peer:
        peer.sendall(raw)
        header, body = receive_response(peer)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    ending_cpu_ticks = cpu_ticks(worker) if worker is not None else None
    if expected is None:
        if (b"data: [DONE]" not in body or b"text/event-stream" not in header
                or b"Native ezlocalai echo backend received: SSE transport control" not in body):
            raise AssertionError("SSE control did not return a completed echo stream")
    elif raw.startswith(b"POST /v1/chat/completions "):
        content = json.loads(body)["choices"][0]["message"]["content"]
        asset = urlsplit(content.rsplit("\n", 1)[1])
        if not asset.path.startswith("/outputs/") or not asset.path.endswith(".wav") or asset.query or asset.fragment:
            raise AssertionError("chat did not attach the buffered speech asset")
        # Byte verification is outside the timed operation and CPU snapshot.
        # The chat operation itself includes writing its completed audio asset.
        with connect(url) as peer:
            peer.sendall(b"GET " + asset.path.encode() + b" HTTP/1.1\r\nHost: localhost\r\n\r\n")
            _, audio = receive_response(peer)
        if audio != expected:
            raise AssertionError(f"audio asset differs: {len(audio)} bytes; expected {len(expected)}")
    elif body != expected:
        raise AssertionError(f"HTTP body differs: {len(body)} bytes; expected {len(expected)}")
    return elapsed_ms, ending_cpu_ticks


def framing_controls(url):
    headers = b"GET /health HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n"
    healthy = b'{"status":"healthy"}'
    for prefix, finish, eof in [
        (b"1\r\nA\n", b"X", False),
        (b"0\r\nX-End: yes\r\n", b"\r\nignored", False),
        (b"0\r\n", b"", True),
        (b"1\r\nA\r", b"", True),
    ]:
        with connect(url) as peer:
            peer.sendall(headers + prefix)
            peer.settimeout(0.03)
            try:
                response = peer.recv(1)
            except socket.timeout:
                pass
            else:
                raise AssertionError(f"incomplete framing returned early: {response!r}")
            peer.settimeout(30)
            if finish:
                peer.sendall(finish)
            if eof:
                peer.shutdown(socket.SHUT_WR)
            _, body = receive_response(peer)
            if body != healthy:
                raise AssertionError("framing control route result changed")
    return {"valid_split_trailer": True, "split_malformed_terminator": True, "incomplete_zero_eof": True, "incomplete_data_eof": True}


def cases():
    healthy = b'{"status":"healthy"}'
    for size in (65536, 1048576, 8388608):
        # Prebuild all request/response bytes outside the timed region.
        for chunk_size in (0, 4096, 65536):
            framing = b"Transfer-Encoding: chunked\r\n" if chunk_size else f"Content-Length: {size}\r\n".encode()
            body = chunked(size, chunk_size) if chunk_size else payload(size)
            raw = b"GET /health HTTP/1.1\r\nHost: localhost\r\n" + framing + b"\r\n" + body
            yield {"path": "incoming", "payload_bytes": size, "chunk_bytes": chunk_size}, raw, healthy
            body = json.dumps({"model": "transport-bench", "messages": [{"role": "user", "content": f"transport-bench:{size}:{chunk_size}"}], "voice": "amy"}).encode()
            raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer transport-bench\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            voice_response(size, chunk_size)
            yield {"path": "buffered_response", "payload_bytes": size, "chunk_bytes": chunk_size}, raw, payload(size)
    body = b'{"model":"transport-bench","stream":true,"messages":[{"role":"user","content":"SSE transport control"}]}'
    raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer transport-bench\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    yield {"path": "sse_control", "payload_bytes": len(body), "chunk_bytes": 0}, raw, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="append", required=True, help="label=http://127.0.0.1:port; repeat for A/B")
    parser.add_argument("--worker-pid", action="append", required=True, help="label=PID for each worker; checked through Linux procfs before any HTTP")
    parser.add_argument("--source-commit", action="append", default=[], help="label=40-character Git commit; optional provenance")
    parser.add_argument("--voice-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-samples", type=int, default=7)
    parser.add_argument("--large-samples", type=int, default=3)
    args = parser.parse_args()
    workers = list(assignments(args.worker, parser, "--worker").items())
    labels = [label for label, _ in workers]
    if args.small_samples < 1 or args.large_samples < 1 or not 1024 <= args.voice_port <= 65535:
        parser.error("positive sample counts and an unprivileged voice port are required")
    pids = assignments(args.worker_pid, parser, "--worker-pid")
    commits = assignments(args.source_commit, parser, "--source-commit")
    if (set(pids) != set(labels) or len(set(pids.values())) != len(labels)
            or any(not re.fullmatch(r"[1-9][0-9]*", value) for value in pids.values())):
        parser.error("each variant must have a distinct verified worker PID")
    if not set(commits) <= set(labels) or any(not re.fullmatch(r"[0-9a-fA-F]{40}", sha) for sha in commits.values()):
        parser.error("source commits must belong to known labels and be full hexadecimal commit IDs")
    try:
        addresses = [worker_url(url) for _, url in workers]
        if len(set(addresses)) != len(workers) or any(port == args.voice_port for _, port in addresses):
            parser.error("workers and fake voice listener require distinct loopback ports")
        metadata = {label: guarded_worker(label, url, int(pids[label]), args.voice_port) for label, url in workers}
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if len({item["binary_sha256"] for item in metadata.values()}) != len(workers):
        parser.error("A/B variants must be different binary artifacts")
    for label, commit in commits.items():
        metadata[label]["source_commit"] = commit.lower()
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    server = VoiceServer(("127.0.0.1", args.voice_port), VoiceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = {
        "scope": "actual loopback HTTP transport; echo backend/fake speech only; no GPU/model benchmark",
        "started_utc": datetime.now(timezone.utc).isoformat(), "host": platform.platform(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "driver_cpu_affinity": sorted(os.sched_getaffinity(0)), "initial_load_average": os.getloadavg(),
        "workers": metadata, "timing": "connect + send + receive + close; prepared wire bytes; perf_counter_ns; audio asset verification excluded",
        "buffered_response_route": "chat with voice -> buffered fake speech response -> writes local audio asset",
        "cpu_clock_ticks_per_second": ticks_per_second, "variant_order": "alternating each sample",
        "controls": {}, "results": [],
        "status": "running",
    }
    try:
        for label, url in workers:
            report["controls"][label] = framing_controls(url)
        all_cases = list(cases())
        for case, raw, expected in all_cases:
            for _, url in workers:
                exchange(url, raw, expected)
            samples = {label: [] for label, _ in workers}
            cpu_samples = {label: [] for label, _ in workers}
            count = args.large_samples if case["payload_bytes"] >= 8388608 else args.small_samples
            for trial in range(count):
                ordered = workers if trial % 2 == 0 else list(reversed(workers))
                for label, url in ordered:
                    before_cpu = cpu_ticks(metadata[label])
                    wall_ms, after_cpu = exchange(url, raw, expected, metadata[label])
                    samples[label].append(wall_ms)
                    cpu_samples[label].append((after_cpu - before_cpu) * 1000 / ticks_per_second)
            result = dict(case, samples_ms=samples,
                          cpu_samples_ms=cpu_samples,
                          median_ms={label: statistics.median(values) for label, values in samples.items()},
                          body_sha256=hashlib.sha256(expected).hexdigest() if expected is not None else None)
            report["results"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: result[key] for key in ("path", "payload_bytes", "chunk_bytes", "median_ms")}), flush=True)
        report["status"] = "passed"
    finally:
        server.shutdown()
        server.server_close()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if report["status"] != "passed":
            report["status"] = "failed"
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
