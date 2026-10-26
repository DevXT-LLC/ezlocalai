"""Offline protocol and regression checks for finite text disk continuations."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench_state_roundtrip as benchmark
from bench_progression import digest, native_metrics
from test_bench_state_roundtrip import FakeClient, args as legacy_args, metadata, report


# Captured with the complete historical pipeline before this option was added,
# using its existing FakeClient, source='source', run_id='fixed', prompt_chars=100.
HISTORICAL_HASHES = {
    "warmup": "bd0a62d1b50c89dfbb27b2a1d804f2eded2d4eca1039c7cf6d418100232cc715",
    "text_seed": "18a863fef3b2da8c8d54257301f329e1e33caf6c16b212b346bd17a37823db95",
    "text_reference": "306706ee8f00433dc04dad218fa4cfbb87e0213a329754f397b4fd691afc48d9",
    "text_stomp": "54c7b8d263609a926d1994af38769956b3d104eea371bd90b0f0938385d81507",
    "text_restored": "306706ee8f00433dc04dad218fa4cfbb87e0213a329754f397b4fd691afc48d9",
    "vision_seed": "e9d2810fb4602ab4e3711492d3a65034caa20e137b26abfa3505ff85a317a3d0",
    "vision_reference": "1134dd1a73a412f6d5385d436bc802fc1a3f9823aeaa3db32365eb51fef26442",
    "vision_stomp": "b9867b63e34d964c18db3dafb70711b38a91f959cc6a1eb1ceb554ccdb92a25a",
    "vision_restored": "1134dd1a73a412f6d5385d436bc802fc1a3f9823aeaa3db32365eb51fef26442"}


def cli(extra=None):
    return benchmark.arguments(["http://engine", "source.py", "out.json", "--label", "test",
        "--context", "8192", "--metadata", "meta.json", "--prompt-chars", "100", "--run-id", "fixed",
        "--require-draft-counters"] + (extra or []))


def control_content(branch, turn, newline=True):
    return "BRANCH_" + str(branch) + "_END CODE_" + str(910000 + 17 * branch) + "\n" + turn + ("\n" if newline else "")


class FiniteClient(FakeClient):
    """Keep the existing disk-state fake while independently scripting text output."""

    def __init__(self, mutate=None, **kwargs):
        super().__init__(**kwargs)
        self.mutate = mutate
        self.preflight_hashes = []

    def json(self, path, payload=None):
        if path == "/v1/chat/completions/input_tokens":
            self.preflight_hashes.append(digest(payload))
        return super().json(path, payload)

    def measure(self, payload):
        if digest(payload) not in self.preflight_hashes:
            raise AssertionError("The final generation request was not preflighted")
        index = len(self.generations)
        value = super().measure(payload)
        controls = {1: (0, "TURN_ALPHA"), 2: (0, "TURN_BRAVO_r1"),
                    3: (1, "TURN_ALPHA"), 4: (0, "TURN_BRAVO_r1")}
        if index in controls:
            branch, turn = controls[index]
            value["output"] = {"content": control_content(branch, turn), "reasoning_content": ""}
            value["timings"].update(predicted_n=16, draft_n=12, draft_n_accepted=8)
            value["usage"]["completion_tokens"] = 16
            value["finish_reasons"] = ["stop"]
        if self.mutate:
            self.mutate(index, payload, value)
        value.update(native_metrics(value["timings"]))
        value["output_sha256"] = digest(value["output"])
        return value


class StateFiniteArgumentsTests(unittest.TestCase):
    def test_legacy_and_finite_defaults_are_explicit_and_only_finite_caps_are_restricted(self):
        self.assertFalse(cli().finite_control)
        self.assertEqual(cli().decode_tokens, 256)
        self.assertEqual(cli(["--decode-tokens", "512"]).decode_tokens, 512)
        for extra, expected in [([], 64), (["--decode-tokens", "64"], 64), (["--decode-tokens", "128"], 128)]:
            with self.subTest(extra=extra):
                args = cli(["--finite-control"] + extra)
                self.assertTrue(args.finite_control)
                self.assertEqual(args.decode_tokens, expected)

    def test_finite_caps_are_bounded_and_the_option_cannot_be_abbreviated(self):
        for extra in [["--finite-control", "--decode-tokens", str(value)] for value in (1, 63, 65, 129, 256)] + [["--finite"]]:
            with self.subTest(extra=extra), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                cli(extra)


class StateFiniteProtocolTests(unittest.TestCase):
    def setUp(self):
        silence = patch("sys.stdout", new=io.StringIO())
        silence.start()
        self.addCleanup(silence.stop)

    def test_default_full_pipeline_retains_all_historical_request_hashes(self):
        client, value = FakeClient(), report()
        benchmark.run_roundtrips(client, "source", legacy_args(), value, lambda: None)
        names = ["warmup"] + [row["kind"] for row in value["results"]]
        self.assertEqual({name: digest(payload) for name, payload in zip(names, client.generations)}, HISTORICAL_HASHES)
        self.assertTrue(all("expected_control_turn" not in row for row in value["results"]))

    def test_finite_text_roundtrip_preserves_actual_history_and_identical_continuation(self):
        for budget in (64, 128):
            with self.subTest(budget=budget):
                client, value = FiniteClient(), report()
                benchmark.run_roundtrips(client, "source", cli(["--finite-control", "--decode-tokens", str(budget)]), value, lambda: None)
                rows = value["results"][:4]
                self.assertEqual([row["kind"] for row in rows], ["text_seed", "text_reference", "text_stomp", "text_restored"])
                self.assertEqual([row["expected_control_turn"] for row in rows], ["TURN_ALPHA", "TURN_BRAVO_r1", "TURN_ALPHA", "TURN_BRAVO_r1"])
                for index, row in enumerate(rows, 1):
                    payload = client.generations[index]
                    branch = 1 if row["kind"] == "text_stomp" else 0
                    self.assertTrue(row["finite_control_correct"])
                    self.assertEqual(row["requested_generated_tokens"], budget)
                    self.assertEqual(row["finite_control_expected_content"], control_content(branch, row["expected_control_turn"], False))
                    self.assertTrue(payload["messages"][-1]["content"].endswith(row["finite_control_expected_content"]))
                    self.assertEqual(row["request_sha256"], digest(payload))
                    self.assertEqual(row["output_sha256"], digest(row["output"]))
                self.assertEqual(client.generations[2]["messages"][-2], {"role": "assistant", "content": "BRANCH_0_END CODE_910000\nTURN_ALPHA\n"})
                self.assertEqual(client.generations[2], client.generations[4])
                self.assertTrue(value["comparisons"]["text"]["saved_seed_prefix_reused"])
                self.assertTrue(value["comparisons"]["text"]["output_identical"])
                self.assertEqual([item["status"] for item in value["snapshots"]], ["restored", "restored"])
                self.assertTrue(all(item["save"]["n_written"] == item["restore"]["n_read"] for item in value["snapshots"]))
                self.assertEqual(value["launch"]["cache_ram_mib"], 0)

    def test_finite_text_leaves_warmup_and_every_vision_request_historically_unchanged(self):
        client, value = FiniteClient(), report()
        benchmark.run_roundtrips(client, "source", cli(["--finite-control"]), value, lambda: None)
        self.assertEqual(digest(client.generations[0]), HISTORICAL_HASHES["warmup"])
        for index, row in enumerate(value["results"][4:], 5):
            self.assertEqual(digest(client.generations[index]), HISTORICAL_HASHES[row["kind"]])
            self.assertNotIn("expected_control_turn", row)
            self.assertNotIn("finite_control_expected_content", row)
            self.assertNotIn("finite_control_correct", row)
        stomp = value["results"][6]
        self.assertEqual((stomp["requested_generated_tokens"], stomp["generated_tokens"]), (128, 128))
        self.assertEqual(stomp["finish_reasons"], ["length"])
        self.assertIn("800 words", client.generations[7]["messages"][-1]["content"])

    def test_invalid_reference_or_restored_control_never_enters_results_or_starts_vision(self):
        for stage in (2, 4):
            for invalid in (control_content(0, "TURN_ALPHA"), control_content(1, "TURN_BRAVO_r1"),
                            control_content(0, "TURN_BRAVO_r2")):
                def mutate(index, payload, sample):
                    if index == stage:
                        sample["output"]["content"] = invalid
                client, value = FiniteClient(mutate=mutate), report()
                with self.subTest(stage=stage, invalid=invalid), self.assertRaises(RuntimeError):
                    benchmark.run_roundtrips(client, "source", cli(["--finite-control"]), value, lambda: None)
                self.assertEqual(len(value["results"]), stage - 1)
                self.assertEqual(len(client.generations), stage + 1)
                self.assertEqual(len(value["snapshots"]), 1)
                self.assertNotIn("text", value["comparisons"])
                self.assertEqual(value["pending_request"]["expected_control_turn"], "TURN_BRAVO_r1")

    def test_correct_control_content_cannot_hide_length_finish_or_reasoning(self):
        for stage in (2, 4):
            for failure in ("length", "reasoning"):
                def mutate(index, payload, sample):
                    if index == stage:
                        if failure == "length":
                            sample["finish_reasons"] = ["length"]
                        else:
                            sample["output"]["reasoning_content"] = "Unexpected hidden output."
                client, value = FiniteClient(mutate=mutate), report()
                with self.subTest(stage=stage, failure=failure), self.assertRaises(RuntimeError):
                    benchmark.run_roundtrips(client, "source", cli(["--finite-control"]), value, lambda: None)
                self.assertEqual(len(value["results"]), stage - 1)
                self.assertNotIn("text", value["comparisons"])

    def test_vision_stomp_retains_legacy_minimum_despite_global_finite_option(self):
        def mutate(index, payload, sample):
            if index == 7:
                sample["timings"].update(predicted_n=1, draft_n=1, draft_n_accepted=0)
                sample["usage"]["completion_tokens"] = 1
                sample["finish_reasons"] = ["stop"]
        client, value = FiniteClient(mutate=mutate), report()
        with self.assertRaisesRegex(RuntimeError, "decode bounds"):
            benchmark.run_roundtrips(client, "source", cli(["--finite-control"]), value, lambda: None)
        self.assertEqual(len(value["results"]), 6)
        self.assertEqual(value["pending_request"]["kind"], "vision_stomp")
        self.assertNotIn("expected_control_turn", value["pending_request"])

    def test_finite_identical_outputs_cannot_mask_ignored_restore(self):
        client, value = FiniteClient(ignore_restore=True), report()
        with self.assertRaisesRegex(RuntimeError, "full recomputation"):
            benchmark.run_roundtrips(client, "source", cli(["--finite-control"]), value, lambda: None)
        self.assertEqual(value["results"][1]["output_sha256"], value["results"][3]["output_sha256"])
        self.assertEqual(value["results"][3]["cached_prompt_tokens"], 0)
        self.assertNotIn("text", value["comparisons"])

    def test_both_reference_and_restored_must_meet_original_cached_prefix_threshold(self):
        for stage in (2, 4):
            def mutate(index, payload, sample):
                if index == stage:
                    sample["timings"]["cache_n"] = 983
                    sample["timings"]["prompt_n"] = sample["usage"]["prompt_tokens"] - 983
            value = report()
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, "full recomputation"):
                benchmark.run_roundtrips(FiniteClient(mutate=mutate), "source", cli(["--finite-control"]), value, lambda: None)
            self.assertNotIn("text", value["comparisons"])
        def boundary(index, payload, sample):
            if index in (2, 4):
                sample["timings"]["cache_n"] = 984
                sample["timings"]["prompt_n"] = sample["usage"]["prompt_tokens"] - 984
        value = report()
        benchmark.run_roundtrips(FiniteClient(mutate=boundary), "source", cli(["--finite-control"]), value, lambda: None)
        self.assertEqual(value["comparisons"]["text"]["minimum_cached_tokens"], 984)

    def test_two_individually_valid_answers_still_require_identical_actual_output_hashes(self):
        def mutate(index, payload, sample):
            if index == 4:
                sample["output"]["content"] = control_content(0, "TURN_BRAVO_r1", False)
        client, value = FiniteClient(mutate=mutate), report()
        with self.assertRaisesRegex(RuntimeError, "output differs"):
            benchmark.run_roundtrips(client, "source", cli(["--finite-control"]), value, lambda: None)
        self.assertTrue(value["results"][1]["finite_control_correct"])
        self.assertTrue(value["results"][3]["finite_control_correct"])
        self.assertEqual(value["results"][1]["finite_control_expected_content"], value["results"][3]["finite_control_expected_content"])
        self.assertNotEqual(value["results"][1]["output_sha256"], value["results"][3]["output_sha256"])
        self.assertEqual(len(client.generations), 5)
        self.assertEqual(value["snapshots"][0]["status"], "restored")

    def test_changed_request_hash_fails_even_for_valid_identical_finite_answers(self):
        client, value = FiniteClient(), report()
        benchmark.run_roundtrips(client, "source", cli(["--finite-control"]), value, lambda: None)
        reference, restored = value["results"][1], copy.deepcopy(value["results"][3])
        restored["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "request differs"):
            benchmark.compare_continuation(reference, restored, 1000)

    def test_atomic_failure_retains_expected_answer_and_snapshot_rpc_evidence(self):
        for stage, status in ((2, "saved"), (4, "restored")):
            def mutate(index, payload, sample):
                if index == stage:
                    sample["output"]["content"] = control_content(0, "TURN_ALPHA")
            client = FiniteClient(mutate=mutate)
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, meta, output = root / "source.py", root / "meta.json", root / "out.json"
                source.write_text("source")
                meta.write_text(json.dumps(metadata()))
                argv = ["http://engine", str(source), str(output), "--label", "test", "--context", "8192",
                        "--metadata", str(meta), "--run-id", "fixed", "--prompt-chars", "100", "--finite-control"]
                with patch.object(benchmark, "Client", return_value=client), self.assertRaises(RuntimeError):
                    benchmark.main(argv)
                saved = json.loads(output.read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertTrue(saved["configuration"]["finite_control"])
                self.assertEqual(saved["configuration"]["decode_tokens"], 64)
                self.assertEqual(saved["pending_request"]["expected_control_turn"], "TURN_BRAVO_r1")
                self.assertEqual(saved["pending_request"]["finite_control_expected_content"], control_content(0, "TURN_BRAVO_r1", False))
                self.assertEqual(saved["last_response"]["output"]["content"], control_content(0, "TURN_ALPHA"))
                self.assertEqual(saved["counts"]["recorded_samples"], stage - 1)
                self.assertEqual(len(saved["snapshots"]), 1)
                snapshot = saved["snapshots"][0]
                self.assertRegex(snapshot["filename"], r"^ezl-state-text-[a-f0-9]{32}\.bin$")
                self.assertEqual(snapshot["status"], status)
                self.assertEqual(snapshot["save"]["filename"], snapshot["filename"])
                self.assertEqual("restore" in snapshot, stage == 4)
                self.assertNotIn("text", saved["comparisons"])
                self.assertEqual(saved["inference_retries"], 0)


if __name__ == "__main__":
    unittest.main()
