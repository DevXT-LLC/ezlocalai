import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from bench_progression import (arguments, atomic_report, conversation_messages, digest,
                               main, measure_stream, native_metrics, sse_events, validate_metadata)


TIMINGS = {"prompt_n": 20, "cache_n": 18, "prompt_ms": 10,
           "prompt_per_second": 2000, "predicted_n": 6,
           "predicted_ms": 80, "predicted_per_second": 62.5}


def event(value):
    return "data: " + json.dumps(value) + "\r\n\r\n"


class Response(io.BytesIO):
    status = 200


def response(records):
    return Response("".join(records).encode())


def records():
    return [": heartbeat\r\n\r\n",
            event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
            event({"choices": [{"index": 0, "delta": {"content": ""}}]}),
            event({"choices": [{"index": 0, "delta": {"reasoning_content": "Think"}}]}),
            event({"choices": [{"index": 0, "delta": {"content": "Hello world"}}]}),
            event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "timings": TIMINGS}),
            event({"choices": [], "usage": {"prompt_tokens": 38, "completion_tokens": 6}}),
            "data: [DONE]\r\n\r\n"]


class StreamingMeasurementsTests(unittest.TestCase):
    def test_ttft_ignores_role_empty_usage_and_keepalive_and_uses_native_token_rate(self):
        times = iter(range(100))
        seen = []

        def opener(request, **kwargs):
            seen.append(request)
            return response(records())

        sample = measure_stream("http://127.0.0.1:1", "private-key", [{"role": "user", "content": "hello"}],
                                model="Model", cache_key="A", clock=lambda: next(times), opener=opener)
        self.assertEqual(sample["first_event_seconds"], 1)
        self.assertEqual(sample["ttft_seconds"], 3)
        self.assertEqual(sample["wall_seconds"], 8)
        # Two output deltas contain six generated tokens. A delta count is
        # intentionally never used as a token throughput measurement.
        self.assertEqual(sample["decode_tokens_per_second"], 62.5)
        self.assertNotIn("decode_wall_tps", sample)
        self.assertEqual(sample["output"], {"content": "Hello world", "reasoning_content": "Think"})
        self.assertEqual(sample["cache_fraction"], 18 / 38)
        payload = json.loads(seen[0].data)
        self.assertEqual(payload["prompt_cache_key"], "A")
        self.assertEqual(sample["request_sha256"], digest(payload))
        self.assertEqual(payload["model"], "Model")
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer private-key")
        self.assertNotIn("private-key", json.dumps(sample))

    def test_optional_key_changes_actual_request_hash_without_changing_historical_default(self):
        messages = [{"role": "user", "content": "hello"}]
        samples, payloads = [], []
        for cache_key in (None, "named-prefill"):
            def opener(request, **kwargs):
                payloads.append(json.loads(request.data))
                return response(records())

            samples.append(measure_stream("http://local", "", messages, cache_key=cache_key,
                clock=lambda: 1, opener=opener))
        self.assertNotIn("prompt_cache_key", payloads[0])
        self.assertEqual(samples[0]["request_sha256"],
            "0d02395b66f4241ed719c8bd66f01fdb3ce6d49369408f709e0ead9a715bf3be")
        self.assertEqual(payloads[1]["prompt_cache_key"], "named-prefill")
        self.assertEqual(samples[1]["request_sha256"], digest(payloads[1]))
        self.assertNotEqual(samples[0]["request_sha256"], samples[1]["request_sha256"])
        self.assertEqual(samples[0]["messages_sha256"], samples[1]["messages_sha256"])

    def test_stream_record_parser_handles_multiline_data_comments_crlf_and_bom(self):
        raw = '\ufeff: keepalive\r\n\r\nevent: token\r\nid: 3\r\ndata: {"a":\r\ndata: 1}\r\n\r\ndata:[DONE]\r\n\r\n'
        self.assertEqual(list(sse_events(Response(raw.encode()))), ['{"a":\n1}', '[DONE]'])

    def test_dropped_stream_cannot_become_a_fast_sample(self):
        with self.assertRaisesRegex(RuntimeError, "missing \\[DONE\\]"):
            measure_stream("http://x", "", [], opener=lambda *a, **k: response(records()[:-1]))

    def test_empty_output_has_no_ttft_and_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no content or reasoning"):
            measure_stream("http://x", "", [], opener=lambda *a, **k: response([
                event({"choices": [{"delta": {"role": "assistant"}, "finish_reason": "stop"}], "timings": TIMINGS}),
                "data: [DONE]\n\n"]))

    def test_stream_errors_and_missing_native_timings_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "stream returned an error"):
            measure_stream("http://x", "", [], opener=lambda *a, **k: response([
                event({"error": {"message": "capacity exceeded"}}), "data: [DONE]\n\n"]))
        missing = [r for r in records() if '"timings"' not in r]
        with self.assertRaisesRegex(RuntimeError, "native evaluated/cached"):
            measure_stream("http://x", "", [], opener=lambda *a, **k: response(missing))

    def test_invalid_native_counts_and_rates_are_rejected(self):
        for name, value in [("prompt_n", -1), ("cache_n", True), ("cache_n", 2.5),
                            ("predicted_per_second", float("nan")), ("prompt_ms", float("inf"))]:
            with self.subTest(name=name, value=value), self.assertRaises(RuntimeError):
                native_metrics({**TIMINGS, name: value})


class ReproducibilityTests(unittest.TestCase):
    def test_message_hash_uses_saved_historical_serialization(self):
        messages = [{"role": "user", "content": "café 日本語"}]
        self.assertEqual(digest(messages), hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest())

    def test_conversations_share_source_but_have_distinct_labels_and_stable_repeat_payloads(self):
        a = conversation_messages("abc", 10, "run", 1)
        b = conversation_messages("abc", 10, "run", 10)
        self.assertEqual(a, conversation_messages("abc", 10, "run", 1))
        self.assertNotEqual(a, b)
        self.assertIn("abcabcabca", a[-1]["content"])
        self.assertIn("BRANCH_1_END", a[-1]["content"])
        self.assertNotIn("BRANCH_1_END", b[-1]["content"])
        self.assertNotEqual(digest(a), digest(conversation_messages("abc", 10, "other-run", 1)))

    def test_report_replacement_is_atomic_and_invalid_json_does_not_destroy_prior_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            atomic_report(path, {"results": [1]})
            with self.assertRaises(ValueError):
                atomic_report(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"results": [1]})
            self.assertEqual(list(Path(directory).iterdir()), [path])
            atomic_report(path, {"results": [1, 2]})
            self.assertEqual(json.loads(path.read_text()), {"results": [1, 2]})

    def test_sanitized_metadata_may_include_fingerprints_but_not_credentials(self):
        validate_metadata({"image": "sha256:abc", "environment_sha256": "digest"})
        for value in [{"api_key": "private"}, {"nested": [{"Authorization": "Bearer private"}]},
                      {"HF_TOKEN": "private"}]:
            with self.assertRaises(ValueError):
                validate_metadata(value)

    def test_invalid_cli_values_and_unsafe_historical_layouts_are_rejected(self):
        base = ["http://127.0.0.1:1", "source.py", "report.json", "--label", "test"]
        for extra in [["--repeats", "0"], ["--timeout", "nan"], ["--conversations", "1"],
                      ["--exact-first-prompt"], ["--prompt-chars", "-1"]]:
            with self.subTest(extra=extra), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                arguments(base + extra)
        for url in ["http://name:password@host", "http://host?api_key=private", "file:///tmp/socket"]:
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                arguments([url] + base[1:])
        args = arguments(base + ["--suites", "prefill", "--prompt-chars", "680000", "--exact-first-prompt"])
        self.assertTrue(args.exact_first_prompt)
        self.assertIsNone(args.prompt_cache_key)
        for key in ("", "   ", "a\nb", "a\0b", "a\x7fb"):
            with self.subTest(key=repr(key)), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                arguments(base + ["--prompt-cache-key", key])
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            arguments(base + ["--suites", "decode", "--prompt-cache-key", "unused"])

    def test_explicit_key_only_names_prefill_and_preserves_other_suite_keys(self):
        seen = []

        def measured(*args, **kwargs):
            seen.append(args[5])
            marker = re.search(r"\bBRANCH_\d+_END\b", args[2][-1]["content"])
            content = marker.group() if marker else "Python"
            return {"wall_seconds": 1, "ttft_seconds": .5, "timings": dict(TIMINGS),
                    **native_metrics(TIMINGS), "output": {"content": content},
                    "output_sha256": digest(content)}

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.py", Path(directory) / "report.json"
            source.write_text("abc")
            argv = ["http://local", str(source), str(output), "--label", "gpu", "--run-id", "stable",
                    "--prompt-cache-key", "named-prefill", "--prompt-chars", "30", "--repeats", "1",
                    "--latency-repeats", "1", "--conversations", "2"]
            with patch("bench_progression.measure_stream", side_effect=measured), patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(argv), 0)
            report = json.loads(output.read_text())
        self.assertEqual(seen[:3], [None, None, None])  # warmup, latency, decode
        self.assertEqual(seen[3:7], ["named-prefill"] * 4)
        self.assertEqual(seen[7:], ["bench:stable:0", "bench:stable:1"] * 2)
        self.assertEqual(report["configuration"]["prompt_cache_key"], "named-prefill")

    def test_historical_run_keeps_message_hash_and_reports_native_metrics(self):
        from bench_prefill import prefill_cases

        def measured(*args, **kwargs):
            self.assertIsNone(args[5])
            return {"wall_seconds": 1, "ttft_seconds": 0.5, "timings": dict(TIMINGS),
                    **native_metrics(TIMINGS), "messages_sha256": digest(args[2]),
                    "output": {"content": "Python"}, "output_sha256": "answer"}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_text("abc")
            output = Path(directory) / "report.json"
            argv = ["http://local", str(source), str(output), "--label", "gpu", "--run-id", "stable",
                    "--suites", "prefill", "--prompt-chars", "30", "--repeats", "1", "--exact-first-prompt"]
            with patch("bench_progression.measure_stream", side_effect=measured), patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(argv), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["results"]), 4)  # warmup is excluded
            self.assertEqual(report["results"][0]["messages_sha256"], digest(prefill_cases("abc", 30)[0][1]))
            self.assertEqual(report["summary"]["prefill_cold:30"]["decode_tokens_per_second_median"], 62.5)

    def test_cached_cold_failure_is_checkpointed_without_a_false_sample(self):
        def warmed(*args, **kwargs):
            return {"timings": {**TIMINGS, "cache_n": 10000}}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_text("abc")
            output = Path(directory) / "report.json"
            argv = ["http://local", str(source), str(output), "--label", "gpu", "--suites", "prefill",
                    "--prompt-chars", "30", "--repeats", "1"]
            with patch("bench_progression.measure_stream", side_effect=warmed), self.assertRaisesRegex(RuntimeError, "reused a long prefix"):
                main(argv)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["results"], [])


if __name__ == "__main__":
    unittest.main()
