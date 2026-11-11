import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import grade_prefill_code as grader


def sample(task="js", **kwargs):
    return dict(
        event="sample",
        task=task,
        answer=json.dumps({"code": "MODEL_CODE"}),
        finish_reason="stop",
        block=2,
        variant="nvfp4-ub1024",
        target_tokens=64000,
        template_tokens=63999,
        repeat=1,
        kind="cold",
        prompt_sha256="prompt",
        **kwargs
    )


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.run = self.enterContext(patch.object(grader.subprocess, "run"))
        self.run.side_effect = [
            subprocess.CompletedProcess([], 0, "PREFILL_CODE_TESTS_PASSED", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]

    def test_sandbox_is_locked_down_and_code_only_passed_as_stdin(self):
        row = sample()
        original = copy.deepcopy(row)
        self.assertTrue(grader.evaluate(row)[0])
        self.assertEqual(row, original)
        call, cleanup = self.run.call_args_list
        command = call.args[0]
        self.assertEqual(command[:2], ["docker", "run"])
        for flag in ("--rm", "--pull=never", "--read-only"):
            self.assertIn(flag, command)
        for flag, value in (
            ("--network", "none"),
            ("--cap-drop", "ALL"),
            ("--security-opt", "no-new-privileges"),
            ("--memory", "512m"),
            ("--memory-swap", "512m"),
            ("--cpus", "1"),
            ("--pids-limit", "64"),
            ("--user", "65534:65534"),
            ("--log-driver", "none"),
            ("--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=128m,mode=1777"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("node:22-alpine", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--volume", command)
        self.assertNotIn("-v", command)
        self.assertNotIn("MODEL_CODE", " ".join(command))
        self.assertTrue(call.kwargs["input"].startswith("MODEL_CODE\n"))
        self.assertIn(grader.JS_TESTS, call.kwargs["input"])
        self.assertEqual(call.kwargs["timeout"], 20)
        name = command[command.index("--name") + 1]
        self.assertTrue(name.startswith("prefill-code-grade-"))
        self.assertEqual(cleanup.args[0], ["docker", "rm", "-f", name])
        self.assertEqual(cleanup.kwargs["timeout"], 15)

    def test_rust_image_compiler_and_test_completion(self):
        self.run.side_effect = [
            subprocess.CompletedProcess(
                [], 0, "test result: ok. 2 passed; 0 failed", ""
            ),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
        self.assertTrue(grader.evaluate(sample("rust"))[0])
        call = self.run.call_args_list[0]
        self.assertIn("rust:1.93-slim-bookworm", call.args[0])
        self.assertIn("rustc --edition=2021 --test", call.args[0][-1])
        self.assertIn(grader.RUST_TESTS, call.kwargs["input"])

    def test_fenced_code_uses_existing_envelope_parser(self):
        row = sample()
        row["answer"] = "```json\n" + row["answer"] + "\n```"
        with patch.object(grader, "extract_code", wraps=grader.extract_code) as extract:
            self.assertTrue(grader.evaluate(row)[0])
        extract.assert_called_once_with(row["answer"])

    def test_incomplete_and_invalid_envelopes_never_start_docker(self):
        for finish in (None, "length", "error", "tool_calls", "x" * 2000):
            row = sample()
            row["finish_reason"] = finish
            passed, diagnostic = grader.evaluate(row)
            self.assertFalse(passed)
            self.assertLessEqual(len(diagnostic), 1500)
        for answer in (
            None,
            "bad",
            "{}",
            '{"code":""}',
            '{"code":1}',
            '{"code":"a","code":"b"}',
            '{"code":"a","extra":true}',
            'prose {"code":"x"}',
            '{"code":"x"} trailing',
        ):
            row = sample()
            row["answer"] = answer
            self.assertFalse(grader.evaluate(row)[0])
        self.run.assert_not_called()

    def test_failure_and_missing_marker_fail_even_if_exit_zero(self):
        for code, stdout, stderr in (
            (1, "PREFILL_CODE_TESTS_PASSED", "assertion failed"),
            (0, "", ""),
            (125, "", "cached image missing"),
        ):
            with self.subTest(code=code, stdout=stdout):
                self.run.side_effect = [
                    subprocess.CompletedProcess([], code, stdout, stderr),
                    subprocess.CompletedProcess([], 0, "", ""),
                ]
                self.assertFalse(grader.evaluate(sample())[0])

    def test_diagnostics_are_bounded(self):
        self.run.side_effect = [
            subprocess.CompletedProcess([], 1, "a" * 9000, "z" * 9000),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
        passed, diagnostic = grader.evaluate(sample())
        self.assertFalse(passed)
        self.assertLessEqual(len(diagnostic), 1500)
        self.assertIn("sandbox exit 1", diagnostic)

    def test_timeout_and_launch_failure_clean_up_owned_container(self):
        for error in (
            subprocess.TimeoutExpired("docker", 20),
            FileNotFoundError("docker"),
        ):
            with self.subTest(error=error):
                self.run.reset_mock()
                self.run.side_effect = [
                    error,
                    subprocess.CompletedProcess([], 0, "", ""),
                ]
                passed, diagnostic = grader.evaluate(sample())
                self.assertFalse(passed)
                self.assertEqual(self.run.call_args.args[0][:3], ["docker", "rm", "-f"])
                self.assertLessEqual(len(diagnostic), 1500)

    def test_cleanup_failure_cannot_report_pass(self):
        for result in (
            subprocess.TimeoutExpired("cleanup", 15),
            OSError("daemon missing"),
            subprocess.CompletedProcess([], 1, "", "daemon failure"),
        ):
            with self.subTest(result=result):
                self.run.side_effect = [
                    subprocess.CompletedProcess([], 0, "PREFILL_CODE_TESTS_PASSED", ""),
                    result,
                ]
                passed, diagnostic = grader.evaluate(sample())
                self.assertFalse(passed)
                self.assertIn("cleanup failed", diagnostic)

    def test_already_auto_removed_container_is_normal(self):
        for message in (
            "Error: No such container: owned",
            "Error response from daemon: no such container: owned",
        ):
            with self.subTest(message=message):
                self.run.side_effect = [
                    subprocess.CompletedProcess([], 0, "PREFILL_CODE_TESTS_PASSED", ""),
                    subprocess.CompletedProcess([], 1, "", message),
                ]
                self.assertTrue(grader.evaluate(sample())[0])

    def test_interrupt_still_cleans_up_and_propagates(self):
        self.run.side_effect = [
            KeyboardInterrupt(),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
        with self.assertRaises(KeyboardInterrupt):
            grader.evaluate(sample())
        self.assertEqual(self.run.call_args.args[0][:3], ["docker", "rm", "-f"])

    def test_names_are_unique(self):
        self.run.side_effect = lambda *a, **kw: subprocess.CompletedProcess(
            [], 0, "PREFILL_CODE_TESTS_PASSED", ""
        )
        grader.evaluate(sample())
        grader.evaluate(sample())
        names = [
            call.args[0][call.args[0].index("--name") + 1]
            for call in self.run.call_args_list
            if call.args[0][1] == "run"
        ]
        self.assertEqual(len(set(names)), 2)

    def test_infra_is_not_handled_by_code_grader(self):
        with self.assertRaises(ValueError):
            grader.evaluate(sample("infra"))
        self.run.assert_not_called()


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "probe.jsonl"
        self.output = self.path.with_name("grades.json")
        self.evaluate = self.enterContext(
            patch.object(grader, "evaluate", return_value=(True, "passed"))
        )
        self.run = self.enterContext(
            patch.object(
                grader.subprocess,
                "run",
                side_effect=AssertionError("unit tests never run Docker"),
            )
        )

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as output:
            output.write("\n")
            for row in rows:
                output.write(json.dumps(row) + "\n")

    def test_hash_identifiers_and_infra_untouched(self):
        rows = [
            {"event": "provenance"},
            sample("infra", quality_pass=True),
            sample("rust"),
            sample("js"),
        ]
        self.write_rows(rows)
        raw = self.path.read_bytes()
        self.evaluate.side_effect = [(True, "passed"), (False, "assertion failed")]
        report = grader.grade_file(self.path)
        self.assertEqual(report["input_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            report["summary"],
            {"graded": 2, "passed": 1, "failed": 1, "skipped_infra": 1},
        )
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(len(self.evaluate.call_args_list), 2)
        for index, result in enumerate(report["samples"]):
            row = rows[index + 2]
            for key in grader.IDENTIFIERS:
                self.assertEqual(result[key], row[key])
            self.assertEqual(result["line"], index + 4)
            self.assertEqual(
                result["answer_sha256"],
                hashlib.sha256(row["answer"].encode()).hexdigest(),
            )
            self.assertNotIn("answer", result)
            self.assertNotIn("quality_pass", result)

    def test_all_jsonl_validated_before_starting_sandboxes(self):
        for tail in ("truncated", "[]", "null"):
            self.write_rows([sample()])
            with self.path.open("a") as output:
                output.write(tail)
            with self.subTest(tail=tail), self.assertRaisesRegex(ValueError, "line 3"):
                grader.grade_file(self.path)
            self.evaluate.assert_not_called()

    def test_missing_answer_hash_and_legacy_infra(self):
        row = sample()
        del row["answer"]
        self.write_rows([row, {"event": "sample", "quality_pass": True}])
        report = grader.grade_file(self.path)
        self.assertIsNone(report["samples"][0]["answer_sha256"])
        self.assertEqual(report["summary"]["skipped_infra"], 1)

    def test_cli_writes_separate_json(self):
        self.write_rows([sample()])
        raw = self.path.read_bytes()
        with patch("sys.stdout", new_callable=io.StringIO):
            grader.main([str(self.path), str(self.output)])
        report = json.loads(self.output.read_text())
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_cli_refuses_overwriting_evidence_or_grades(self):
        self.write_rows([sample()])
        with self.output.open("w") as output:
            output.write("existing grades")
        for path in (self.path, self.output):
            with self.subTest(path=path), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    grader.main([str(self.path), str(path)])
        self.evaluate.assert_not_called()
        self.assertEqual(self.output.read_text(), "existing grades")


if __name__ == "__main__":
    unittest.main()
