"""Offline behavioral guards for the optional finite cache-control workload."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench_cache_workload as benchmark
from bench_progression import digest


def expected_content(branch, turn):
    return "BRANCH_" + str(branch) + "_END CODE_" + str(910000 + 17 * branch) + "\n" + turn


def measured_sample(branch=0, turn="TURN_ALPHA", trailing_newline=False, cached=0):
    output = {"content": expected_content(branch, turn) + ("\n" if trailing_newline else ""),
              "reasoning_content": ""}
    return {"status": 200, "wall_seconds": .5, "ttft_seconds": .1,
            "first_event_seconds": .05,
            "timings": {"prompt_n": 1000 - cached, "cache_n": cached, "prompt_ms": 100,
                        "prompt_per_second": 10000, "predicted_n": 16,
                        "predicted_ms": 80, "predicted_per_second": 200,
                        "draft_n": 12, "draft_n_accepted": 8},
            "usage": {"prompt_tokens": 1000, "completion_tokens": 16},
            "prefill_tokens_per_second": 10000, "decode_tokens_per_second": 200,
            "evaluated_prompt_tokens": 1000 - cached, "cached_prompt_tokens": cached,
            "output": output, "output_sha256": digest(output), "finish_reasons": ["stop"]}


def cli(extra=None):
    return benchmark.arguments(["http://engine", "source.py", "result.json", "--label", "test",
        "--context", "8192", "--prompt-chars", "120", "--run-id", "fixed"] + (extra or []))


class ScriptedClient:
    """Supply a fixed response schedule, independent of prompt-construction helpers."""

    def __init__(self, schedule, mutate=None):
        self.schedule, self.mutate = list(schedule), mutate
        self.calls, self.preflights, self.responses = [], [], []

    def json(self, path, payload=None):
        if path == "/props":
            return {"total_slots": 1, "default_generation_settings": {"n_ctx": 8192},
                    "is_sleeping": False}
        if path != "/v1/chat/completions/input_tokens":
            raise AssertionError("Unexpected HTTP path: " + path)
        self.preflights.append(digest(payload))
        return {"input_tokens": 1000}

    def measure(self, payload):
        if digest(payload) not in self.preflights:
            raise AssertionError("Final request body was not preflighted")
        if "prompt_cache_key" not in payload:
            return measured_sample()
        index = len(self.calls)
        if index >= len(self.schedule):
            raise AssertionError("Unexpected generation after scripted responses")
        branch, turn, trailing_newline = self.schedule[index]
        if payload["prompt_cache_key"].rsplit(":", 1)[1] != str(branch):
            raise AssertionError("Conversation scheduling changed")
        self.calls.append(copy.deepcopy(payload))
        prior = any(row[0] == branch for row in self.schedule[:index])
        sample = measured_sample(branch, turn, trailing_newline, 996 if prior else 0)
        if self.mutate:
            self.mutate(index, sample)
        sample.update(request_sha256=digest(payload), messages_sha256=digest(payload["messages"]),
                      output_sha256=digest(sample["output"]))
        self.responses.append(copy.deepcopy(sample))
        return sample


class FiniteArgumentsTests(unittest.TestCase):
    def test_finite_budget_defaults_and_explicit_caps(self):
        legacy = cli()
        self.assertFalse(legacy.finite_control)
        self.assertEqual((legacy.decode_tokens, legacy.minimum_generated), (256, 64))
        for extra, expected_cap in [([], 64), (["--decode-tokens", "64"], 64),
                                    (["--decode-tokens", "128", "--minimum-generated", "1"], 128)]:
            with self.subTest(extra=extra):
                args = cli(["--finite-control"] + extra)
                self.assertTrue(args.finite_control)
                self.assertEqual((args.decode_tokens, args.minimum_generated), (expected_cap, 1))

    def test_finite_mode_rejects_other_decode_caps_or_minimum_lengths(self):
        for extra in [["--decode-tokens", str(value)] for value in (1, 63, 65, 127, 129, 256)] + [
                ["--minimum-generated", str(value)] for value in (0, 2, 64)]:
            with self.subTest(extra=extra), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                cli(["--finite-control"] + extra)


class FinitePromptTests(unittest.TestCase):
    def test_legacy_request_bodies_retain_prechange_golden_hashes(self):
        # Recorded before finite-control implementation; protects the published
        # long-decode protocol, not merely equivalence between two new branches.
        golden = {
            "common": {"seed": "aeb83eef5c02327296c48773f609e7448f4df7a51430dde336a4cb008c04d716",
                       "append": "1d3403fd30c69e0a2f56067a2951fbc87f748e0b691c86b37c109d8869f5d4ad",
                       "rewind": "95682427ba833845711593416dae5cee4365ac6d09d6fb565d59edc9bbd7dfc6"},
            "independent": {"seed": "19e848f0c339c99779c713679201483498409dc69ab52f1c1bf920874b894874",
                            "append": "925194eda597c3a22be84d51883baa663d710ac1f8f45a009837dd4c5d94e8f2",
                            "rewind": "dc3a627321477e7099ddeb5e8f59d154463c28c2a974b117611bf6a810e0968d"}}
        previous = {"output": {"content": "BRANCH_1_END CODE_910017\nHistorical explanation."}}
        for mode in golden:
            seed = benchmark.conversation_messages("def f(x): return x + 1\n", 120, "fixed", 1, mode, 4)
            messages = {"seed": seed,
                "append": benchmark.append_messages(seed, previous, "fixed", 1, 1),
                "rewind": benchmark.conversation_messages("def f(x): return x + 1\n", 120, "fixed", 1, mode, 2)}
            for kind, body in messages.items():
                with self.subTest(mode=mode, kind=kind):
                    payload = benchmark.request_payload(body, 256, "TestModel", "cache-workload:fixed:1")
                    self.assertEqual(digest(payload), golden[mode][kind])

    def test_finite_turns_do_not_conflict_with_persistent_system_or_long_prefix(self):
        seed = benchmark.conversation_messages("reference()\n", 120, "fixed", 1, "common", 4,
                                               finite_control=True, control_turn="TURN_ALPHA")
        rewind = benchmark.conversation_messages("reference()\n", 120, "fixed", 1, "common", 2,
                                                 finite_control=True, control_turn="TURN_CHARLIE")
        self.assertEqual(seed[0], rewind[0])
        self.assertNotIn("TURN_ALPHA", seed[0]["content"])
        self.assertNotIn("TURN_CHARLIE", rewind[0]["content"])
        self.assertTrue(seed[1]["content"].startswith("Reference code:\n" + "reference()\n" * 10))
        self.assertIn("TURN_ALPHA", seed[-1]["content"])
        self.assertIn("TURN_CHARLIE", rewind[-1]["content"])
        appended = benchmark.append_messages(seed, measured_sample(1, trailing_newline=True),
                                             "fixed", 1, 2, finite_control=True)
        self.assertEqual(appended[-2], {"role": "assistant",
                                        "content": "BRANCH_1_END CODE_910017\nTURN_ALPHA\n"})
        self.assertIn("TURN_BRAVO_r2", appended[-1]["content"])
        for message in [*seed, *rewind, appended[-1]]:
            self.assertNotIn("800 words", message["content"])
            self.assertNotIn("programming question", message["content"])

    def test_control_turn_selection_distinguishes_continuation_from_replay(self):
        for kind, round_number, expected in [("seed", 0, "TURN_ALPHA"), ("replay", 2, "TURN_ALPHA"),
                ("rewind", 1, "TURN_CHARLIE"), ("append", 1, "TURN_BRAVO_r1"),
                ("append", 3, "TURN_BRAVO_r3")]:
            with self.subTest(kind=kind, round_number=round_number):
                self.assertEqual(benchmark.finite_control_turn(kind, round_number), expected)


class FiniteResponseTests(unittest.TestCase):
    def test_only_exact_two_lines_with_optional_one_final_newline_are_accepted(self):
        for branch, turn in [(0, "TURN_ALPHA"), (1, "TURN_BRAVO_r2"), (2, "TURN_CHARLIE")]:
            for newline in (False, True):
                with self.subTest(branch=branch, turn=turn, newline=newline):
                    sample = measured_sample(branch, turn, newline)
                    benchmark.validate_finite_control(sample, branch, turn)
                    result = benchmark.validate_sample(sample, 1000, 64, branch, 1,
                                                       require_draft=True, expected_turn=turn)
                    self.assertTrue(result["identity_correct"])
                    self.assertTrue(result["finite_control_correct"])

    def test_control_rejects_prose_fences_partial_lines_and_extra_bytes(self):
        valid = "BRANCH_0_END CODE_910000\nTURN_ALPHA"
        invalid = ["", "BRANCH_0_END CODE_910000", "TURN_ALPHA", valid[:-1],
                   "\n" + valid, " " + valid, valid + " ", valid + "\n\n",
                   "```\n" + valid + "\n```", "Here is the answer:\n" + valid,
                   valid + "\nAn explanation.", valid.replace("\n", "\r\n"),
                   valid.replace("CODE_910000", "CODE_910017"), valid.replace("BRANCH_0", "BRANCH_1"),
                   valid + "\nBRANCH_1_END CODE_910017"]
        for content in invalid:
            sample = measured_sample()
            sample["output"]["content"] = content
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                benchmark.validate_finite_control(sample, 0, "TURN_ALPHA")

    def test_correct_branch_with_stale_or_wrong_turn_is_rejected(self):
        for actual, expected in [("TURN_ALPHA", "TURN_BRAVO_r1"), ("TURN_BRAVO_r1", "TURN_BRAVO_r2"),
                                  ("TURN_ALPHA", "TURN_CHARLIE"), ("TURN_CHARLIE", "TURN_ALPHA")]:
            with self.subTest(actual=actual, expected=expected), self.assertRaises(RuntimeError):
                benchmark.validate_sample(measured_sample(turn=actual), 1000, 64, 0, 1,
                                          expected_turn=expected)

    def test_exact_visible_content_still_requires_stop_and_no_reasoning(self):
        for finishes in (["length"], ["stop", "length"], [], ["tool_calls"], None):
            sample = measured_sample()
            sample["finish_reasons"] = finishes
            with self.subTest(finishes=finishes), self.assertRaises(RuntimeError):
                benchmark.validate_finite_control(sample, 0, "TURN_ALPHA")
        sample = measured_sample()
        sample["output"]["reasoning_content"] = "Hidden reasoning is still unwanted output."
        with self.assertRaises(RuntimeError):
            benchmark.validate_finite_control(sample, 0, "TURN_ALPHA")

    def test_finite_output_does_not_bypass_native_prompt_decode_or_mtp_guards(self):
        mutations = [("timings", "prompt_n", 999), ("usage", "prompt_tokens", 999),
                     ("timings", "predicted_n", 65), ("usage", "completion_tokens", 15),
                     ("timings", "draft_n_accepted", 13)]
        for container, field, value in mutations:
            sample = measured_sample()
            sample[container][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                benchmark.validate_sample(sample, 1000, 64, 0, 1, require_draft=True,
                                          expected_turn="TURN_ALPHA")


class FinitePipelineTests(unittest.TestCase):
    def setUp(self):
        silence = patch("sys.stdout", new=io.StringIO())
        silence.start()
        self.addCleanup(silence.stop)

    def assert_row_provenance(self, rows, client):
        by_id = {row["sample_id"]: row for row in rows}
        for row, payload in zip(rows, client.calls):
            self.assertEqual(row["request_sha256"], digest(payload))
            self.assertEqual(row["messages_sha256"], digest(payload["messages"]))
            self.assertEqual(row["output_sha256"], digest(row["output"]))
            self.assertEqual(row["finite_control_expected_content"],
                             expected_content(row["branch"], row["expected_control_turn"]))
            self.assertTrue(payload["messages"][-1]["content"].endswith(row["finite_control_expected_content"]))
            self.assertTrue(row["finite_control_correct"])
            if row["kind"] == "append":
                previous = by_id[row["history_dependency_sample_id"]]
                self.assertTrue(row["model_output_dependent_request"])
                self.assertEqual(row["history_dependency_output_sha256"], previous["output_sha256"])
                self.assertEqual(payload["messages"][-2], {"role": "assistant", "content": previous["output"]["content"]})
            else:
                self.assertFalse(row["model_output_dependent_request"])

    def test_consecutive_appends_preserve_actual_seed_then_previous_assistant_bytes(self):
        schedule = [(0, "TURN_ALPHA", True), (1, "TURN_ALPHA", False),
                    (0, "TURN_BRAVO_r1", False), (1, "TURN_BRAVO_r1", True),
                    (0, "TURN_BRAVO_r2", True), (1, "TURN_BRAVO_r2", False)]
        client = ScriptedClient(schedule)
        args = cli(["--finite-control", "--conversations", "2", "--rounds", "3", "--cases", "append"])
        report = {"results": []}
        benchmark.run_workload(client, "source", args, report, lambda: None)
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual([row["expected_control_turn"] for row in report["results"]],
                         [turn for _, turn, _ in schedule])
        self.assertEqual(report["results"][2]["history_dependency_sample_id"], "seed:r0:b0")
        self.assertEqual(report["results"][4]["history_dependency_sample_id"], "append:r1:b0")
        self.assertEqual(client.calls[2]["messages"][-2]["content"], "BRANCH_0_END CODE_910000\nTURN_ALPHA\n")
        self.assert_row_provenance(report["results"], client)

    def test_reordered_rewind_then_append_uses_charlie_output_not_stale_alpha(self):
        schedule = [(0, "TURN_ALPHA", False), (1, "TURN_ALPHA", True),
                    (0, "TURN_CHARLIE", True), (1, "TURN_CHARLIE", False),
                    (0, "TURN_BRAVO_r1", False), (1, "TURN_BRAVO_r1", True),
                    (0, "TURN_ALPHA", True), (1, "TURN_ALPHA", False)]
        client = ScriptedClient(schedule)
        args = cli(["--finite-control", "--conversations", "2", "--cases", "rewind", "append", "replay"])
        report = {"results": []}
        benchmark.run_workload(client, "source", args, report, lambda: None)
        self.assertEqual([row["kind"] for row in report["results"]], ["seed"] * 2 + ["rewind"] * 2 + ["append"] * 2 + ["replay"] * 2)
        self.assertEqual(report["results"][4]["history_dependency_sample_id"], "rewind:r1:b0")
        self.assertEqual(client.calls[4]["messages"][-2]["content"], "BRANCH_0_END CODE_910000\nTURN_CHARLIE\n")
        self.assertEqual(digest(client.calls[0]), digest(client.calls[6]))
        self.assert_row_provenance(report["results"], client)

    def test_append_hashes_diverge_when_actual_valid_assistant_bytes_differ(self):
        args = cli(["--finite-control", "--conversations", "2", "--cases", "append", "rewind", "replay"])
        runs = []
        for newline in (False, True):
            schedule = [(0, "TURN_ALPHA", newline), (1, "TURN_ALPHA", False),
                        (0, "TURN_BRAVO_r1", False), (1, "TURN_BRAVO_r1", False),
                        (0, "TURN_CHARLIE", False), (1, "TURN_CHARLIE", False),
                        (0, "TURN_ALPHA", False), (1, "TURN_ALPHA", False)]
            report, client = {"results": []}, ScriptedClient(schedule)
            benchmark.run_workload(client, "source", args, report, lambda: None)
            runs.append(report["results"])
        for index in (0, 1, 3, 4, 5, 6, 7):
            self.assertEqual(runs[0][index]["request_sha256"], runs[1][index]["request_sha256"])
        self.assertNotEqual(runs[0][0]["output_sha256"], runs[1][0]["output_sha256"])
        self.assertNotEqual(runs[0][2]["request_sha256"], runs[1][2]["request_sha256"])

    def test_rejected_append_is_checkpointed_and_stops_before_next_conversation(self):
        # Correct branch but a stale seed control token is cache contamination.
        schedule = [(0, "TURN_ALPHA", False), (1, "TURN_ALPHA", False), (0, "TURN_ALPHA", False)]
        client = ScriptedClient(schedule)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.py", Path(directory) / "result.json"
            source.write_text("source")
            argv = ["http://engine", str(source), str(output), "--label", "test", "--context", "8192",
                    "--prompt-chars", "120", "--run-id", "fixed", "--finite-control",
                    "--conversations", "2", "--cases", "append"]
            with patch.object(benchmark, "Client", return_value=client), self.assertRaises(RuntimeError):
                benchmark.main(argv)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["configuration"]["finite_control"])
            self.assertEqual(len(report["results"]), 2)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(report["pending_request"]["sample_id"], "append:r1:b0")
            self.assertEqual(report["pending_request"]["expected_control_turn"], "TURN_BRAVO_r1")
            self.assertEqual(report["last_response"]["output"]["content"], "BRANCH_0_END CODE_910000\nTURN_ALPHA")
            self.assertEqual(report["inference_retries"], 0)


if __name__ == "__main__":
    unittest.main()
