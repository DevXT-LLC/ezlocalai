import base64
import io
import json
import struct
import unittest
from unittest.mock import patch
import zlib

from bench_capacity import (arguments, main, prepare_capacity, qualify, red_png, request_body,
                            validate_completion, validate_properties)
from bench_progression import digest


def props(context=230144):
    return {"default_generation_settings": {"n_ctx": context, "params": {}},
            "total_slots": 1, "modalities": {"vision": True}, "is_sleeping": False}


METADATA = {"original_parameters": {"-c": "230000", "-ub": "512"}, "overrides": {"-ub": 1024}}


def completion(count, word="Python", cached=0):
    return {"choices": [{"message": {"content": word}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": count},
            "timings": {"prompt_n": count, "cache_n": cached, "prompt_ms": 20,
                        "prompt_per_second": count * 50, "predicted_n": 2,
                        "predicted_ms": 1, "predicted_per_second": 1000}}


class CapacityArgumentsTests(unittest.TestCase):
    def argv(self):
        return ["http://127.0.0.1:28192", "absent-source.py", "230000", "1024", "gpu", "out.json",
                "--metadata", "absent-metadata.json"]

    def parse_without_io(self, argv):
        with patch("bench_capacity.Path.read_text", side_effect=AssertionError("parser read a file")), \
                patch("bench_capacity.urllib.request.urlopen", side_effect=AssertionError("parser used HTTP")), \
                patch("bench_capacity.Client", side_effect=AssertionError("parser created a client")), \
                patch("bench_capacity.atomic_report", side_effect=AssertionError("parser wrote a report")):
            return arguments(argv)

    def test_named_capacity_controls_are_retained_without_file_or_network_access(self):
        args = self.parse_without_io(self.argv() + ["--prompt-cache-key", "capacity-key",
            "--fraction", "0.94", "--timeout", "120", "--api-key-env", "CAPACITY_API_KEY",
            "--run-id", "qualification-run"])
        self.assertEqual((args.context, args.expected_ubatch), (230000, 1024))
        self.assertEqual(args.prompt_cache_key, "capacity-key")
        self.assertEqual((args.fraction, args.timeout), (.94, 120))
        self.assertEqual((args.api_key_env, args.run_id), ("CAPACITY_API_KEY", "qualification-run"))
        self.assertEqual(str(args.source), "absent-source.py")
        self.assertEqual(str(args.metadata), "absent-metadata.json")

    def test_default_capacity_controls_are_unchanged(self):
        args = self.parse_without_io(self.argv())
        self.assertIsNone(args.prompt_cache_key)
        self.assertEqual((args.fraction, args.timeout, args.api_key_env), (.95, 1800, "EZLOCALAI_API_KEY"))

    def test_invalid_options_fail_during_the_same_pure_preflight(self):
        cases = []
        for key in ("", "  ", "a\nb", "a\0b", "a\x7fb"):
            cases.append(self.argv() + ["--prompt-cache-key", key])
        for fraction in ("0.899", "1", "nan", "inf", "-inf"):
            cases.append(self.argv() + ["--fraction=" + fraction])
        for context in ("0", "511", "invalid"):
            argv = self.argv(); argv[2] = context; cases.append(argv)
        for ubatch in ("0", "-1", "invalid"):
            argv = self.argv(); argv[3] = ubatch; cases.append(argv)
        for timeout in ("0", "-1", "nan", "inf", "-inf"):
            cases.append(self.argv() + ["--timeout=" + timeout])
        for url in ("http://user@local", "https://user:secret@local", "http://local?key=secret",
                    "http://local#secret", "file:///tmp/engine", "http:///missing-host"):
            argv = self.argv(); argv[0] = url; cases.append(argv)
        for argv in cases:
            with self.subTest(argv=argv), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                self.parse_without_io(argv)


class CapacityGuardsTests(unittest.TestCase):
    def test_key_is_optional_and_default_request_fingerprint_is_unchanged(self):
        messages = [{"role": "user", "content": "hello"}]
        body = request_body(messages)
        self.assertNotIn("prompt_cache_key", body)
        self.assertEqual(digest(body), "3f92c625e56cf916ce18976fec776f18a6b28d06244ff6f1cc21928869232e7d")
        keyed = request_body(messages, "capacity-key")
        self.assertEqual(keyed, {**body, "prompt_cache_key": "capacity-key"})
        self.assertNotEqual(digest(body), digest(keyed))
        self.assertFalse(keyed["cache_prompt"])

    def test_invalid_explicit_cli_key_fails_before_reading_sources_or_sending_requests(self):
        argv = ["http://local", "absent-source.py", "4096", "1024", "gpu", "out.json",
                "--metadata", "absent-metadata.json", "--prompt-cache-key"]
        for key in ("", "  ", "a\nb", "a\0b", "a\x7fb"):
            with self.subTest(key=repr(key)), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                main(argv + [key])

    def test_declared_context_is_retained_with_only_upward_padding(self):
        result = validate_properties(props(), 230000, 1024, METADATA)
        self.assertEqual(result["effective_context"], 230144)
        self.assertEqual(result["expected_n_ubatch"], 1024)
        for value in [props(229888), props(262144), {**props(), "total_slots": 2},
                      {**props(), "modalities": {"vision": False}}]:
            with self.assertRaises(RuntimeError):
                validate_properties(value, 230000, 1024, METADATA)
        with self.assertRaisesRegex(RuntimeError, "launch metadata"):
            validate_properties(props(), 230000, 2048, METADATA)

    def test_tokenization_prepares_near_95_percent_without_context_reduction(self):
        def count(messages):
            return len(messages[-1]["content"]) // 4 + 30

        _, result = prepare_capacity("def example(): pass\n", 230000, count, "run")
        self.assertGreaterEqual(result["input_fraction"], .949)
        self.assertLessEqual(result["input_fraction"], .95)
        self.assertEqual(result["declared_context"], 230000)
        self.assertGreater(result["remaining_context_tokens"], 10000)
        self.assertLess(result["tokenizer_passes"], 35)
        with self.assertRaises(ValueError):
            prepare_capacity("x", 512, count, "run", .9999)

    def test_caching_truncation_wrong_answers_and_length_stops_fail(self):
        self.assertEqual(validate_completion(completion(100), 100, "Python", 1000)["answer"], "Python")
        bad = [completion(100, cached=1), completion(99), completion(100, "Java"),
               {**completion(100), "truncated": True},
               {**completion(100), "choices": [{"message": {"content": "Python"}, "finish_reason": "length"}]}]
        for value in bad:
            with self.assertRaises(RuntimeError):
                validate_completion(value, 100, "Python", 1000)
        with self.assertRaises(RuntimeError):
            validate_completion(completion(100), 100, "Python", 107)

    def test_png_contains_only_red_pixels_with_valid_chunk_checksums(self):
        png = red_png(2, 3)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        offset, compressed = 8, b""
        while offset < len(png):
            count = struct.unpack_from(">I", png, offset)[0]
            kind = png[offset + 4:offset + 8]
            value = png[offset + 8:offset + 8 + count]
            crc = struct.unpack_from(">I", png, offset + 8 + count)[0]
            self.assertEqual(zlib.crc32(kind + value) & 0xffffffff, crc)
            if kind == b"IDAT":
                compressed += value
            offset += count + 12
        self.assertEqual(zlib.decompress(compressed), (b"\0" + b"\xff\0\0" * 2) * 3)

    def test_http_qualification_evaluates_text_and_image_with_cache_disabled(self):
        calls, saved = [], {}

        def client(path, body=None):
            calls.append((path, body))
            if path == "/props":
                return props(4096)
            if path == "/apply-template":
                return {"prompt": body["messages"][-1]["content"]}
            if path == "/tokenize":
                return {"tokens": [1] * (len(body["content"]) // 4 + 30)}
            content = body["messages"][-1]["content"]
            count = len(content) // 4 + 30 if isinstance(content, str) else 300
            if path.endswith("input_tokens"):
                return {"input_tokens": count}
            if path == "/v1/chat/completions":
                self.assertFalse(body["cache_prompt"])
                self.assertEqual(body["temperature"], 0)
                if not isinstance(content, str):
                    url = content[1]["image_url"]["url"]
                    self.assertEqual(base64.b64decode(url.split(",", 1)[1]), red_png())
                return completion(count, "Python" if isinstance(content, str) else "red")
            raise AssertionError(path)

        qualify(client, "def f(): pass\n", 4096, 1024,
                {"n_ctx": 4096, "n_ubatch": 1024}, "test", lambda k, v: saved.update({k: v}))
        self.assertEqual(saved["capacity"]["answer"], "Python")
        self.assertEqual(saved["vision"]["answer"], "red")
        self.assertEqual(len([path for path, _ in calls if path == "/v1/chat/completions"]), 2)
        self.assertTrue(all("prompt_cache_key" not in body for _, body in calls if body and "messages" in body))

    def test_named_key_precedes_text_vision_token_counting_inference_and_proof_hashes(self):
        calls, saved = [], {}

        def client(path, body=None):
            calls.append((path, body))
            if path == "/props":
                return props(4096)
            if path == "/apply-template":
                self.assertEqual(body["prompt_cache_key"], "capacity-key")
                return {"prompt": body["messages"][-1]["content"]}
            if path == "/tokenize":
                return {"tokens": [1] * (len(body["content"]) // 4 + 30)}
            self.assertEqual(body["prompt_cache_key"], "capacity-key")
            self.assertFalse(body["cache_prompt"])
            content = body["messages"][-1]["content"]
            count = len(content) // 4 + 30 if isinstance(content, str) else 300
            if path.endswith("input_tokens"):
                return {"input_tokens": count}
            if path == "/v1/chat/completions":
                return completion(count, "Python" if isinstance(content, str) else "red")
            raise AssertionError(path)

        qualify(client, "def f(): pass\n", 4096, 1024, {"n_ctx": 4096, "n_ubatch": 1024},
                "test", lambda k, v: saved.update({k: v}), prompt_cache_key="capacity-key")
        counts = [body for path, body in calls if path.endswith("input_tokens")]
        requests = [body for path, body in calls if path == "/v1/chat/completions"]
        self.assertEqual(counts, requests)
        self.assertEqual(len(requests), 2)
        for name, body in zip(("capacity", "vision"), requests):
            self.assertEqual(saved[name + "_preparation"]["request_sha256"], digest(body))
            self.assertEqual(saved[name]["request_sha256"], digest(body))
            self.assertEqual(saved[name]["cached_prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
