import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench_cache_workload import (Client, append_messages, arguments, branch_identity,
    conversation_messages, input_tokens, main, request_payload, run_workload,
    summary, validate_provenance, validate_runtime, validate_sample)
from bench_progression import digest


TIMINGS = {"prompt_n": 1000, "cache_n": 0, "prompt_ms": 500,
           "prompt_per_second": 2000, "predicted_n": 256,
           "predicted_ms": 2040, "predicted_per_second": 125,
           "draft_n": 200, "draft_n_accepted": 150}


def sample(branch=0, evaluated=1000, cached=0, text=None):
    marker, code = branch_identity(branch)
    output = {"content": text or marker + " " + code + "\nA hash map manages collisions.",
              "reasoning_content": ""}
    return {"status": 200, "wall_seconds": 3, "ttft_seconds": .4,
            "first_event_seconds": .1, "timings": {**TIMINGS, "prompt_n": evaluated, "cache_n": cached},
            "usage": {"prompt_tokens": evaluated + cached, "completion_tokens": 256},
            "prefill_tokens_per_second": 2000, "decode_tokens_per_second": 125,
            "evaluated_prompt_tokens": evaluated, "cached_prompt_tokens": cached,
            "output": output, "output_sha256": digest(output), "finish_reasons": ["length"]}


class Response(io.BytesIO):
    status = 200


def event(value):
    return "data: " + json.dumps(value) + "\r\n\r\n"


def stream_records(extra=None):
    rows = [": keepalive\r\n\r\n",
            event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
            event({"choices": [{"index": 0, "delta": {"content": ""}}]}),
            event({"choices": [{"index": 0, "delta": {"content": "BRANCH_0_END CODE_910000\n"}}]}),
            event({"choices": [{"index": 0, "delta": {"content": "Two chunks are not 256 token counts."}}]}),
            event({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                   "timings": TIMINGS, "usage": {"prompt_tokens": 1000, "completion_tokens": 256}})]
    if extra:
        rows.append(event(extra))
    return rows + ["data: [DONE]\r\n\r\n"]


class FakeClient:
    def __init__(self, bad_count=None, cold_cache=0, fail_after=None):
        self.props = {"total_slots": 1, "default_generation_settings": {"n_ctx": 8192}, "is_sleeping": False}
        self.bad_count, self.cold_cache, self.fail_after = bad_count, cold_cache, fail_after
        self.calls, self.counts, self.branch_calls = [], [], {}

    def json(self, path, body=None):
        if path == "/props":
            return self.props
        self.assert_path(path)
        self.counts.append(body)
        return {"input_tokens": self.bad_count if self.bad_count is not None else 1000}

    @staticmethod
    def assert_path(path):
        if path != "/v1/chat/completions/input_tokens":
            raise AssertionError(path)

    def measure(self, payload):
        self.calls.append(payload)
        if "warmup" in payload["messages"][0]["content"]:
            return sample()
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("controlled transport failure")
        marker = payload["prompt_cache_key"].rsplit(":", 1)[1]
        branch = int(marker)
        previous = self.branch_calls.get(branch, 0)
        self.branch_calls[branch] = previous + 1
        # Returning misses are legitimate under pressure. A branch's first
        # request optionally simulates accidental warm input for the cold guard.
        cached = self.cold_cache if previous == 0 else (996 if branch % 2 else 0)
        value = sample(branch, 1000 - cached, cached)
        value.update(request_sha256=digest(payload), messages_sha256=digest(payload["messages"]))
        return value


def cli(extra=None):
    return arguments(["http://127.0.0.1:28192", "source.py", "result.json", "--label", "test",
                      "--context", "8192", "--prompt-chars", "100", "--run-id", "fixed"] + (extra or []))


class StreamTests(unittest.TestCase):
    def test_ttft_and_native_rates_do_not_count_role_empty_frames_or_chunks(self):
        seen = []
        def opener(request, **kwargs):
            seen.append(request)
            return Response("".join(stream_records()).encode())
        clock = iter(range(100))
        payload = request_payload([{"role": "user", "content": "test"}], 256, "Model", "branch")
        value = Client("http://engine", "private", 30, opener).measure(payload, lambda: next(clock))
        self.assertEqual(value["ttft_seconds"], 3)
        self.assertEqual(value["first_event_seconds"], 1)
        self.assertEqual(value["decode_tokens_per_second"], 125)
        self.assertEqual(value["timings"]["predicted_n"], 256)
        self.assertEqual(value["request_sha256"], digest(payload))
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer private")
        self.assertNotIn("private", json.dumps(value))

    def test_incomplete_truncated_and_unsupported_finish_streams_fail(self):
        cases = [stream_records()[:-1], stream_records({"truncated": True}),
                 stream_records({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
                 stream_records({"choices": [{"index": 0, "delta": {}}, {"index": 0, "delta": {}}]})]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                Client("http://engine", "", 30, lambda *a, **k: Response("".join(rows).encode())).measure(request_payload([], 256))


class MetricAndBudgetTests(unittest.TestCase):
    def test_metadata_rejects_raw_docker_environments_and_credentials(self):
        validate_provenance({"image": "sha256:x", "original_parameters": {"-c": "8192"},
                             "environment": {"GGML_CUDA_GRAPH_OPT": "0"}, "model_sha256": "abc"})
        for metadata in ({"Config": {"Env": ["API_KEY=private"]}},
                         {"environment": ["API_KEY=private"]},
                         {"environment": {"LLM_API_KEY": "private"}},
                         {"environment": {"UNRELATED_TOKEN": "private"}},
                         {"authorization": "private"}, {"cache_mib": float("nan")}, []):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_provenance(metadata)

    def test_draft_acceptance_uses_valid_native_counters_and_missing_is_optional(self):
        value = validate_sample(sample(), 1000, 256, 0, 64, True)
        self.assertEqual(value["draft_acceptance_ratio"], .75)
        self.assertEqual(value["generated_tokens"], 256)
        self.assertTrue(value["generated_budget_reached"])
        value = sample()
        for name in ("draft_n", "draft_n_accepted"):
            value["timings"].pop(name)
        self.assertNotIn("draft_acceptance_ratio", validate_sample(value, 1000, 256, 0, 64))
        with self.assertRaisesRegex(RuntimeError, "draft counters"):
            validate_sample(value, 1000, 256, 0, 64, True)

    def test_invalid_native_mtp_or_input_counts_and_short_decodes_fail(self):
        for key, value in [("draft_n", True), ("draft_n", float("nan")),
                           ("draft_n_accepted", 2.5), ("draft_n_accepted", 201), ("predicted_n", 1)]:
            data = sample()
            data["timings"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                validate_sample(data, 1000, 256, 0, 64)
        for count in (999, 1001):
            with self.assertRaisesRegex(RuntimeError, "possible truncation"):
                validate_sample(sample(), count, 256, 0, 64)
        data = sample()
        data["usage"]["completion_tokens"] = 255
        with self.assertRaisesRegex(RuntimeError, "generated token counts differ"):
            validate_sample(data, 1000, 256, 0, 64)

    def test_identity_checks_correct_first_line_and_other_branch_contamination(self):
        for text in ("BRANCH_1_END CODE_910017\nA summary", "BRANCH_0_END CODE_910000\nBRANCH_1_END",
                     "BRANCH_0_END CODE_910000\nCODE_910017", "Summary only"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                validate_sample(sample(text=text), 1000, 256, 0, 64)
        data = sample()
        data["output"]["reasoning_content"] = "Unexpected reasoning"
        with self.assertRaisesRegex(RuntimeError, "thinking being disabled"):
            validate_sample(data, 1000, 256, 0, 64)

    def test_fixed_context_padding_single_slot_and_headroom_guards(self):
        self.assertEqual(validate_runtime({"n_ctx": 230144, "total_slots": 1}, 230000)["effective_context"], 230144)
        for props in ({"n_ctx": 229888, "total_slots": 1}, {"n_ctx": 230144, "total_slots": 2},
                      {"n_ctx": 230144, "total_slots": True},
                      {"n_ctx": 230144, "total_slots": 1, "is_sleeping": True}):
            with self.assertRaises(RuntimeError):
                validate_runtime(props, 230000)
        client = FakeClient(bad_count=8000)
        with self.assertRaisesRegex(RuntimeError, "unchanged context"):
            input_tokens(client, request_payload([], 256), 8192)
        self.assertEqual(client.calls, [])


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        silence = patch("sys.stdout", new=io.StringIO())
        silence.start()
        self.addCleanup(silence.stop)

    def test_independent_prefix_is_early_and_replay_rewind_are_deterministic(self):
        a = conversation_messages("abc", 100, "run", 0, "independent", 90)
        b = conversation_messages("abc", 100, "run", 1, "independent", 90)
        self.assertNotEqual(a[0]["content"][:32], b[0]["content"][:32])
        self.assertEqual(a, conversation_messages("abc", 100, "run", 0, "independent", 90))
        rewind = conversation_messages("abc", 100, "run", 0, "independent", 40)
        self.assertEqual(a[0], rewind[0])
        self.assertLess(len(rewind[1]["content"]), len(a[1]["content"]))

    def test_default_resident_schedule_records_native_acceptance_and_append_dependencies(self):
        args = cli()
        report = {"results": []}
        client = FakeClient()
        run_workload(client, "source", args, report, lambda: None)
        self.assertEqual(len(report["results"]), 12)
        self.assertEqual([v["kind"] for v in report["results"]], ["seed"] * 3 + ["replay"] * 3 + ["append"] * 3 + ["rewind"] * 3)
        by_id = {row["sample_id"]: row for row in report["results"]}
        for row in report["results"]:
            if row["kind"] == "append":
                self.assertTrue(row["model_output_dependent_request"])
                dep = by_id[row["history_dependency_sample_id"]]
                self.assertEqual(dep["kind"], "replay")
                self.assertEqual(row["history_dependency_output_sha256"], dep["output_sha256"])
            else:
                self.assertFalse(row["model_output_dependent_request"])
        self.assertEqual(report["summary"]["seed"]["draft_acceptance_ratio"], .75)
        self.assertNotIn("pending_request", report)
        self.assertNotIn("last_response", report)

    def test_pressure_misses_are_recorded_not_rejected_or_called_proven_evictions(self):
        args = cli(["--mode", "pressure"])
        self.assertEqual((args.conversations, args.prefix_mode, args.cases), (8, "independent", ["replay"]))
        report = {"results": []}
        run_workload(FakeClient(), "source", args, report, lambda: None)
        self.assertEqual(len(report["results"]), 16)
        self.assertEqual(report["summary"]["replay"]["cache_outcomes"], {"tail_reuse": 4, "prefix_reuse": 0, "full_reevaluation": 4})
        self.assertEqual(sum(row.get("possible_eviction_or_unavailable_state", False) for row in report["results"]), 4)

    def test_accidental_warm_seed_and_preflight_overflow_fail_without_false_success(self):
        for client, error in [(FakeClient(cold_cache=200), "reused a long prefix"),
                              (FakeClient(bad_count=8000), "unchanged context")]:
            report = {"results": []}
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                run_workload(client, "source", cli(), report, lambda: None)
            self.assertEqual(report["results"], [])
            if client.bad_count:
                self.assertEqual(client.calls, [])

    def test_failed_main_checkpoints_complete_prior_samples_and_pending_request(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.py", Path(directory) / "out.json"
            source.write_text("source")
            client = FakeClient(fail_after=2)
            argv = ["http://engine", str(source), str(output), "--label", "test", "--context", "8192",
                    "--prompt-chars", "100", "--run-id", "fixed"]
            with patch("bench_cache_workload.Client", return_value=client), patch("sys.stdout", new=io.StringIO()), self.assertRaisesRegex(RuntimeError, "controlled transport"):
                main(argv)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["pending_request"]["sample_id"], "seed:r0:b1")
            self.assertEqual(report["inference_retries"], 0)

    def test_successful_main_fingerprints_imported_helpers_and_marks_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.py", Path(directory) / "out.json"
            source.write_text("source")
            argv = ["http://engine", str(source), str(output), "--label", "test", "--context", "8192",
                    "--prompt-chars", "100", "--run-id", "fixed"]
            with patch("bench_cache_workload.Client", return_value=FakeClient()):
                self.assertEqual(main(argv), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["results"]), 12)
            self.assertEqual(set(report["harness_dependencies_sha256"]), {"bench_prefill.py", "bench_progression.py"})
            self.assertTrue(all(len(value) == 64 for value in report["harness_dependencies_sha256"].values()))
            self.assertNotIn("pending_request", report)
            self.assertTrue(report["warmup_excluded"])

    def test_nonfinite_optional_native_data_cannot_break_failed_atomic_checkpoint(self):
        class NonfiniteClient(FakeClient):
            def measure(self, payload):
                value = super().measure(payload)
                value["timings"]["draft_n"] = float("inf")
                return value
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.py", Path(directory) / "out.json"
            source.write_text("source")
            argv = ["http://engine", str(source), str(output), "--label", "test", "--context", "8192",
                    "--prompt-chars", "100", "--run-id", "fixed"]
            with patch("bench_cache_workload.Client", return_value=NonfiniteClient()), self.assertRaisesRegex(ValueError, "finite JSON"):
                main(argv)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["results"], [])
            self.assertEqual(report["pending_request"]["sample_id"], "seed:r0:b0")

    def test_keyboard_interrupt_is_not_left_as_a_running_report(self):
        class InterruptedClient(FakeClient):
            def measure(self, payload):
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.py", Path(directory) / "out.json"
            source.write_text("source")
            argv = ["http://engine", str(source), str(output), "--label", "test", "--context", "8192",
                    "--prompt-chars", "100"]
            with patch("bench_cache_workload.Client", return_value=InterruptedClient()), self.assertRaises(KeyboardInterrupt):
                main(argv)
            self.assertEqual(json.loads(output.read_text())["status"], "cancelled")

    def test_cli_rejects_unsafe_urls_counts_and_duplicate_cases(self):
        for extra in (["--rounds", "1"], ["--cases", "replay", "replay"], ["--timeout", "nan"],
                      ["--rewind-records", "90"], ["--minimum-generated", "257"]):
            with self.subTest(extra=extra), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                cli(extra)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            arguments(["http://user:secret@engine", "source", "out", "--label", "x", "--context", "8192"])


if __name__ == "__main__":
    unittest.main()
