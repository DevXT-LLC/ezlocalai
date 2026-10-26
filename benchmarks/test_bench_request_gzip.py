import contextlib
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error
import urllib.request

import bench_request_gzip as bench


class FakeClock:
    def __init__(self):
        self.now = 10.0
        self.sleeps = []
    def clock(self):
        return self.now
    def sleep(self, value):
        self.sleeps.append(value)
        self.now += value


class GzipBenchmarkGuards(unittest.TestCase):
    def test_public_corpus_provenance_and_bounds(self):
        manifest, text = bench.load_corpus()
        self.assertEqual(len(manifest["sources"]), 160)
        self.assertLessEqual(manifest["total_source_bytes"], 1024 * 1024)
        self.assertEqual({repo["url"] for repo in manifest["repositories"]},
            {"https://github.com/DevXT-LLC/ezlocalai"})
        self.assertTrue(all(repo["license"] == "MIT" and "Permission is hereby granted" in repo["license_text"]
                            for repo in manifest["repositories"]))
        small = bench.source_messages(text, 32768)
        combined = "".join(value["content"] for value in small)
        self.assertIn("### native:", combined)
        self.assertEqual({item["repo"] for item in manifest["sources"]}, {"native"})
        self.assertEqual(len({(item["repo"], item["path"]) for item in manifest["sources"]}), 160)
        expected = {(item["repo"], item["path"]) for item in manifest["sources"]}
        rendered = re.findall(r"^### ([^: \n]+):([^\n]+)$", text, re.MULTILINE)
        self.assertEqual(set(rendered), expected)
        self.assertEqual(len(rendered), len(expected))
        for repo, path in expected:
            self.assertEqual(text.count("### " + repo + ':' + path + '\n'), 1)
        self.assertIn('### native:src/http_body.cpp\n#include "ezlocalai/http_body.hpp"', text)
        self.assertEqual(set(bench.cases(text)), {"public-source-32768", "public-source-131072",
            "public-source-524288", "public-source-1000000"})

    def test_corpus_corruption_is_rejected_without_external_checkout(self):
        manifest = json.loads(bench.MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / bench.MANIFEST.name
            path.write_text(json.dumps(manifest))
            (root / manifest["archive"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                bench.load_corpus(path)
            manifest["archive"] = "../outside.gz"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest"):
                bench.load_corpus(path)

    def test_source_budget_refuses_repeated_filler_and_preserves_utf8(self):
        with self.assertRaisesRegex(ValueError, "repeated filler"):
            bench.source_messages("short", 32768)
        source = "é" * 20000
        value = bench.source_messages(source, 32769)
        pieces = [message["content"] for message in value[1:-1]]
        self.assertEqual("".join(pieces), "é" * 16384)

    def test_requests_are_stable_and_keep_both_routing_hints(self):
        messages = [{"role": "user", "content": "public source"}]
        first = bench.payload("case", messages, True)
        self.assertEqual(first, bench.payload("case", messages, True))
        value = json.loads(first)
        self.assertEqual(value["worker"], bench.WORKER_ID)
        self.assertEqual(value["prompt_cache_key"], "gzip-case")
        self.assertEqual(value["routing_affinity_key"], "gzip-public-fixture")
        self.assertIs(value["disable_fallback"], False)

    def test_strict_reference_decoder_rejects_trailing_truncated_and_corrupt(self):
        encoded = gzip.compress(b"public payload", mtime=0)
        self.assertEqual(bench.strict_gzip(encoded), b"public payload")
        for invalid in (encoded[:-1], encoded + b"junk", encoded + encoded,
                        encoded[:-8] + bytes([encoded[-8] ^ 1]) + encoded[-7:]):
            with self.assertRaises((ValueError, bench.zlib.error)):
                bench.strict_gzip(invalid)
        with self.assertRaisesRegex(ValueError, "decoded body exceeds"):
            bench.strict_gzip(gzip.compress(b"x" * 4096), 512)

    def test_pacing_starts_before_first_body_byte_and_preserves_bytes(self):
        clock = FakeClock()
        sends = []
        body = b"0123456789abcdef"
        bench.pace_send(lambda part: sends.append((clock.now, part)), body,
                        rate=8, clock=clock.clock, sleep=clock.sleep, block=4)
        self.assertEqual(b"".join(part for _, part in sends), body)
        self.assertEqual([at for at, _ in sends], [10.5, 11.0, 11.5, 12.0])
        self.assertEqual(sum(clock.sleeps), 2.0)
        self.assertGreater(sends[0][0], 10.0)

    def test_bad_pacing_and_implicit_route_invocation_are_rejected(self):
        for rate in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                bench.pace_send(lambda _: None, b"body", rate=rate)
        self.assertFalse(bench.arguments(["--output", "/tmp/codec-only"]).run_route)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.arguments(["--output", "/tmp/result", "--worker-binary", "/tmp/worker"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.arguments(["--output", "/tmp/result", "--run-route"])

    def test_unshaped_control_has_no_sleep_or_clock_rate_limit(self):
        sent = []
        def unexpected(*args):
            raise AssertionError("unshaped upload must not schedule a rate limit")
        self.assertEqual(bench.pace_send(sent.append, b"0123456789", rate=0,
            clock=unexpected, sleep=unexpected, block=4), 10)
        self.assertEqual(sent, [b"0123", b"4567", b"89"])
        self.assertEqual(bench.arguments(["--output", "/tmp/control", "--bytes-per-second", "0"]).bytes_per_second, 0)

    def test_listener_proof_rejects_a_foreign_process_socket(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "42/net").mkdir(parents=True)
            (root / "42/fd").mkdir()
            (root / "42/net/tcp").write_text("header\n0: 0100007F:1F90 00000000:0000 0A 0 0 0 0 0 123\n")
            (root / "42/fd/3").symlink_to("socket:[999]")
            self.assertFalse(bench.owns_listener(42, 8080, root))
            (root / "42/fd/3").unlink()
            (root / "42/fd/3").symlink_to("socket:[123]")
            self.assertTrue(bench.owns_listener(42, 8080, root))

    def test_startup_never_probes_an_unowned_listener(self):
        with mock.patch.object(bench, "owns_listener", return_value=False), \
             mock.patch.object(bench.worker_http, "open_loopback") as opened, \
             mock.patch.object(bench.time, "monotonic", side_effect=[0, 0, 2]), \
             mock.patch.object(bench.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                bench.wait_ready("http://127.0.0.1:8080", 42, lambda: True, timeout=1)
            opened.assert_not_called()

    def test_record_guards_encoding_identity_and_rate(self):
        body = gzip.compress(b'{"disable_fallback":true,"content":"verification-sentinel"}', mtime=0)
        def relay_value():
            return SimpleNamespace(records=[{"path": "/v1/chat/completions", "content_encoding": "gzip",
                "_body": body, "authentication_forwarded": True,
                "rate_bytes_per_second": 100, "upload_seconds": len(body) / 100}], errors=[])
        relay = relay_value()
        with self.assertRaises(TypeError):
            json.dumps(relay.records[0])
        record, identity = bench.checked_record(relay, 0, None, "gzip")
        self.assertNotIn("_body", record)
        self.assertNotIn("_body", relay.records[0])
        self.assertNotIn("verification-sentinel", json.dumps(record))
        self.assertEqual(json.loads(json.dumps(record)), record)
        with self.assertRaisesRegex(RuntimeError, "encoding"):
            bench.checked_record(relay_value(), 0, None, "")
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            bench.checked_record(relay_value(), 0, {**identity, "decoded_sha256": "other"}, "gzip")
        relay = relay_value()
        relay.records[0]["upload_seconds"] -= 0.01
        with self.assertRaisesRegex(RuntimeError, "rate"):
            bench.checked_record(relay, 0, None, "gzip")

    def test_router_credentials_use_subprocess_environment_only(self):
        before = dict(os.environ)
        fake = SimpleNamespace(poll=lambda: 0)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(bench.subprocess, "Popen", return_value=fake) as launch:
            root = Path(temporary)
            router = bench.RouterProcess(Path("/tmp/router"), root, "generated-client-secret", "generated-register-secret")
            try:
                args, kwargs = launch.call_args
                self.assertNotIn("generated-client-secret", repr(args))
                self.assertNotIn("generated-register-secret", repr(args))
                self.assertEqual(kwargs["env"]["EZLOCALAI_API_KEY"], "generated-client-secret")
                self.assertEqual(kwargs["env"]["ROUTER_REGISTER_KEY"], "generated-register-secret")
                self.assertEqual((root / ".env").read_bytes(), b"")
                self.assertEqual(os.environ, before)
            finally:
                router.__exit__()

    def test_relay_defers_decode_parse_and_hash_until_measurement_returns(self):
        response_body = json.dumps({"choices": [{"message": {"content": bench.worker_http.ANSWER},
            "finish_reason": "stop"}]}).encode()
        class Native(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Native)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        main_thread = threading.get_ident()
        measuring = [True]
        verified = []
        def observed(name, function):
            def verify(*args, **kwargs):
                self.assertEqual(threading.get_ident(), main_thread, "relay thread performed verification")
                self.assertFalse(measuring[0], "verification ran before measure returned")
                verified.append(name)
                return function(*args, **kwargs)
            return verify
        original_open = bench.worker_http.open_loopback
        def add_encoding(request, timeout):
            request.add_header("Content-Encoding", "gzip")
            return original_open(request, timeout)
        encoded = gzip.compress(b'{"disable_fallback":true,"messages":[]}', mtime=0)
        try:
            with bench.UploadRelay(server.server_port, "generated-worker-secret", rate=0) as relay, \
                 mock.patch.object(bench, "strict_gzip", observed("inflate", bench.strict_gzip)), \
                 mock.patch.object(bench, "digest", observed("sha256", bench.digest)), \
                 mock.patch.object(bench, "json", SimpleNamespace(loads=observed("parse", json.loads))), \
                 mock.patch.object(bench.worker_http, "open_loopback", side_effect=add_encoding):
                sample = bench.worker_http.measure("http://127.0.0.1:" + str(relay.port),
                    "generated-worker-secret", encoded, False)
                measuring[0] = False
                self.assertGreaterEqual(sample["full_ms"], 0)
                self.assertFalse(verified)
                record, _ = bench.checked_record(relay, 0, None, "gzip")
                self.assertEqual(set(verified), {"inflate", "parse", "sha256"})
                self.assertNotIn("_body", record)
                self.assertNotIn("_body", relay.records[0])
                json.dumps({"upload": record})
                self.assertFalse(relay.errors)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_real_loopback_relay_preserves_auth_and_raw_gzip_body(self):
        received = []
        class Native(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append((body, self.headers.get("Content-Encoding"), self.headers.get("Authorization")))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Native)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with bench.UploadRelay(server.server_port, "generated-worker-secret") as relay:
                decoded = b'{"disable_fallback":true,"messages":[]}'
                encoded = gzip.compress(decoded, mtime=0)
                request = urllib.request.Request("http://127.0.0.1:" + str(relay.port) + "/v1/chat/completions", data=encoded,
                    headers={"Authorization": "Bearer generated-worker-secret", "Content-Type": "application/json", "Content-Encoding": "gzip"})
                with bench.worker_http.open_loopback(request, timeout=2) as response:
                    self.assertEqual(response.read(), b"{}")
                record, _ = bench.checked_record(relay, 0, None, "gzip")
                self.assertEqual(received, [(encoded, "gzip", "Bearer generated-worker-secret")])
                self.assertEqual(record["decoded_sha256"], bench.digest(decoded))
                self.assertTrue(record["disable_fallback"])
                self.assertFalse(relay.errors)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
