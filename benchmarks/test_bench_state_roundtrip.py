import copy
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from bench_cache_workload import branch_identity, request_payload
from bench_progression import digest, native_metrics
from bench_state_roundtrip import (arguments, compare_continuation, main, run_roundtrips,
    snapshot_filename, validate_launch, validate_slot_response, validate_vision)


def metadata():
    return {"image": "sha256:image", "model_sha256": "a" * 64,
            "original_parameters": {"-c": "8192", "--cache-ram": "7680"},
            "overrides": {"--cache-ram": 0, "--slot-save-path": "/state-proof/"}}


def measured(payload, expected, cached=0, text=None):
    maximum = payload["max_tokens"]
    vision = any(isinstance(message.get("content"), list) for message in payload["messages"])
    generated = 2 if vision or "warmup" in payload["messages"][0]["content"] else maximum
    if text is None:
        if vision:
            text = "Red"
        else:
            match = re.search(r"branch (\d+)", payload["messages"][0]["content"])
            branch = int(match.group(1)) if match else 0
            marker, code = branch_identity(branch)
            text = marker + " " + code + "\nHash maps handle collisions with chaining or open addressing."
    output = {"content": text, "reasoning_content": ""}
    timings = {"prompt_n": expected - cached, "cache_n": cached, "prompt_ms": 50,
               "prompt_per_second": 2000, "predicted_n": generated, "predicted_ms": 100,
               "predicted_per_second": 100, "draft_n": generated, "draft_n_accepted": generated // 2}
    return {"status": 200, "wall_seconds": .2, "ttft_seconds": .1, "first_event_seconds": .05,
            "timings": timings, "usage": {"prompt_tokens": expected, "completion_tokens": generated},
            **native_metrics(timings), "request_sha256": digest(payload), "messages_sha256": digest(payload["messages"]),
            "output": output, "output_sha256": digest(output), "finish_reasons": ["stop" if generated < maximum else "length"]}


class FakeClient:
    def __init__(self, ignore_restore=False, changed_output=False, vision=True):
        self.props = {"total_slots": 1, "default_generation_settings": {"n_ctx": 8192}, "modalities": {"vision": vision}}
        self.events, self.generations, self.files = [], [], {}
        self.current = None
        self.ignore_restore, self.changed_output, self.restored = ignore_restore, changed_output, False

    def count(self, payload):
        if "warmup" in payload["messages"][0]["content"]:
            return 20
        image = any(isinstance(message.get("content"), list) for message in payload["messages"])
        if image:
            return 1088 if len(payload["messages"]) == 2 else 1120
        if len(payload["messages"]) > 2:
            return 1100
        return 1000 if "branch 0;" in payload["messages"][0]["content"] else 500

    def json(self, path, payload=None):
        if path == "/props":
            return self.props
        if path == "/v1/chat/completions/input_tokens":
            return {"input_tokens": self.count(payload)}
        filename = payload["filename"]
        if path == "/slots/0?action=save":
            self.events.append("save:" + self.current)
            self.files[filename] = self.current
            return {"id_slot": 0, "filename": filename, "n_saved": 1255 if self.current == "text" else 1089,
                    "n_written": 1000000, "timings": {"save_ms": 10}}
        if path == "/slots/0?action=restore":
            self.events.append("restore:" + self.files[filename])
            if not self.ignore_restore:
                self.current = self.files[filename]
            self.restored = True
            return {"id_slot": 0, "filename": filename, "n_restored": 1255 if self.files[filename] == "text" else 1089,
                    "n_read": 1000000, "timings": {"restore_ms": 10}}
        raise AssertionError(path)

    def measure(self, payload):
        self.generations.append(copy.deepcopy(payload))
        image = any(isinstance(message.get("content"), list) for message in payload["messages"])
        expected = self.count(payload)
        continuation = len(payload["messages"]) > 2
        case = "vision" if image else "text" if "branch 0;" in payload["messages"][0]["content"] else "stomp"
        if "warmup" in payload["messages"][0]["content"]:
            case = "warmup"
        cached = (1088 if image else 1000) if continuation and self.current == case else 0
        value = measured(payload, expected, cached)
        if continuation and self.restored and self.changed_output:
            value["output"]["content"] += " Changed."
            value["output_sha256"] = digest(value["output"])
        self.events.append("generation:" + case + (":continuation" if continuation else ""))
        self.current = case
        self.restored = False
        return value


def args():
    return arguments(["http://engine", "source.py", "out.json", "--label", "test", "--context", "8192",
                      "--metadata", "meta.json", "--prompt-chars", "100", "--run-id", "fixed",
                      "--require-draft-counters"])


def report():
    return {"metadata": metadata(), "results": [], "snapshots": [], "comparisons": {},
            "counts": {"generation_calls": 0, "warmup_generation_calls": 0, "recorded_samples": 0,
                       "input_token_preflights": 0, "save_calls": 0, "restore_calls": 0}}


class StateRoundtripTests(unittest.TestCase):
    def setUp(self):
        silence = patch("sys.stdout", new=io.StringIO())
        silence.start()
        self.addCleanup(silence.stop)

    def test_requires_ram_cache_zero_fixed_context_and_owned_save_path(self):
        self.assertEqual(validate_launch(metadata(), 8192)["cache_ram_mib"], 0)
        for overrides in ({"--cache-ram": 7680, "--slot-save-path": "/state-proof/"},
                          {"--cache-ram": False, "--slot-save-path": "/state-proof/"},
                          {"--cache-ram": 0, "--slot-save-path": "/app/models/"},
                          {"--cache-ram": 0}, {"--cache-ram": 0, "-c": 4096, "--slot-save-path": "/state-proof/"}):
            with self.subTest(overrides=overrides), self.assertRaises(RuntimeError):
                validate_launch({**metadata(), "overrides": overrides}, 8192)

    def test_generated_names_are_bounded_basenames_not_caller_paths(self):
        filenames = [snapshot_filename(kind) for kind in ("text", "vision") for _ in range(10)]
        self.assertEqual(len(filenames), len(set(filenames)))
        self.assertTrue(all(re.fullmatch(r"ezl-state-(?:text|vision)-[0-9a-f]{32}\.bin", name) for name in filenames))
        with self.assertRaises(ValueError):
            snapshot_filename("../../models")

    def test_snapshot_rpc_must_return_same_slot_filename_and_positive_counts(self):
        good = {"id_slot": 0, "filename": "safe.bin", "n_saved": 100, "n_written": 1000, "timings": {"save_ms": 1}}
        validate_slot_response(good, "safe.bin", "save")
        for altered in ({"id_slot": False}, {"filename": "../other"}, {"n_written": 0},
                        {"n_saved": True}, {"timings": {"save_ms": float("nan")}}):
            with self.subTest(altered=altered), self.assertRaises((RuntimeError, ValueError)):
                validate_slot_response({**good, **altered}, "safe.bin", "save")

    def test_real_protocol_order_stomps_cpu_and_gpu_before_each_restore(self):
        client, value = FakeClient(), report()
        run_roundtrips(client, "source", args(), value, lambda: None)
        self.assertEqual(client.events, ["generation:warmup", "generation:text", "save:text", "generation:text:continuation",
            "generation:stomp", "restore:text", "generation:text:continuation", "generation:vision", "save:vision",
            "generation:vision:continuation", "generation:stomp", "restore:vision", "generation:vision:continuation"])
        self.assertEqual(value["counts"], {"generation_calls": 9, "warmup_generation_calls": 1, "recorded_samples": 8,
                                         "input_token_preflights": 9, "save_calls": 2, "restore_calls": 2})
        self.assertEqual([sample["kind"] for sample in value["results"]], ["text_seed", "text_reference", "text_stomp", "text_restored",
            "vision_seed", "vision_reference", "vision_stomp", "vision_restored"])
        for item in value["comparisons"].values():
            self.assertTrue(item["request_identical"] and item["output_identical"] and item["saved_seed_prefix_reused"])
        self.assertEqual(client.generations[2], client.generations[4])
        self.assertEqual(client.generations[6], client.generations[8])

    def test_successful_restore_rpc_cannot_mask_full_recomputation(self):
        value = report()
        with self.assertRaisesRegex(RuntimeError, "full recomputation"):
            run_roundtrips(FakeClient(ignore_restore=True), "source", args(), value, lambda: None)
        self.assertEqual(len(value["results"]), 4)
        self.assertNotIn("text", value["comparisons"])

    def test_changed_exact_continuation_is_a_failure_even_when_identity_matches(self):
        value = report()
        with self.assertRaisesRegex(RuntimeError, "output differs"):
            run_roundtrips(FakeClient(changed_output=True), "source", args(), value, lambda: None)
        self.assertEqual(value["snapshots"][0]["status"], "restored")

    def test_no_projector_prevents_generation(self):
        client = FakeClient(vision=False)
        with self.assertRaisesRegex(RuntimeError, "projector loaded"):
            run_roundtrips(client, "source", args(), report(), lambda: None)
        self.assertEqual(client.generations, [])

    def test_restore_cannot_silently_ignore_appended_state_bytes(self):
        class ShortRead(FakeClient):
            def json(self, path, payload=None):
                value = super().json(path, payload)
                if path == "/slots/0?action=restore":
                    value["n_read"] -= 1
                return value
        with self.assertRaisesRegex(RuntimeError, "consume all bytes"):
            run_roundtrips(ShortRead(), "source", args(), report(), lambda: None)

    def test_vision_counts_color_and_native_draft_guards(self):
        payload = request_payload([{"role": "user", "content": []}], 8)
        value = measured(payload, 1088, text="Red")
        validate_vision(value, 1088, 8, True)
        bad = copy.deepcopy(value)
        bad["output"]["content"] = "Blue"
        with self.assertRaisesRegex(RuntimeError, "color red"):
            validate_vision(bad, 1088, 8, True)
        with self.assertRaisesRegex(RuntimeError, "possible truncation"):
            validate_vision(value, 1089, 8, True)
        bad = copy.deepcopy(value)
        bad["timings"].pop("draft_n")
        with self.assertRaisesRegex(RuntimeError, "draft counters"):
            validate_vision(bad, 1088, 8, True)

    def test_main_complete_report_has_full_counts_and_dependency_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("source")
            (root / "meta.json").write_text(json.dumps(metadata()))
            argv = ["http://engine", str(root / "source.py"), str(root / "report.json"), "--label", "test",
                    "--context", "8192", "--metadata", str(root / "meta.json"), "--prompt-chars", "100", "--run-id", "fixed"]
            with patch("bench_state_roundtrip.Client", return_value=FakeClient()):
                self.assertEqual(main(argv), 0)
            value = json.loads((root / "report.json").read_text())
            self.assertEqual(value["status"], "complete")
            self.assertEqual(value["counts"]["recorded_samples"], 8)
            self.assertEqual(len(value["harness_dependencies_sha256"]), 4)
            self.assertEqual(len(value["snapshots"]), 2)
            self.assertEqual(value["inference_retries"], 0)

    def test_inference_failure_retains_requested_snapshot_filename_for_operator_cleanup(self):
        class BrokenRestore(FakeClient):
            def json(self, path, payload=None):
                if path == "/slots/0?action=restore":
                    raise RuntimeError("controlled restore failure")
                return super().json(path, payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("source")
            (root / "meta.json").write_text(json.dumps(metadata()))
            argv = ["http://engine", str(root / "source.py"), str(root / "report.json"), "--label", "test",
                    "--context", "8192", "--metadata", str(root / "meta.json"), "--prompt-chars", "100"]
            with patch("bench_state_roundtrip.Client", return_value=BrokenRestore()), self.assertRaisesRegex(RuntimeError, "controlled restore"):
                main(argv)
            value = json.loads((root / "report.json").read_text())
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["snapshots"][0]["status"], "restore_requested")
            self.assertEqual(value["counts"]["restore_calls"], 1)
            self.assertRegex(value["snapshots"][0]["filename"], r"^ezl-state-text-[0-9a-f]{32}\.bin$")

    def test_cli_refuses_paths_in_url_context_clamps_and_abbreviations(self):
        base = ["http://engine", "source", "out", "--label", "test", "--context", "8192", "--metadata", "meta.json"]
        for extra in (["--decode-tokens", "8"], ["--context", "128"], ["--timeout", "nan"], ["--lab", "other"]):
            with self.subTest(extra=extra), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                arguments(base + extra)
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            arguments(["http://user:secret@engine"] + base[1:])


if __name__ == "__main__":
    unittest.main()
