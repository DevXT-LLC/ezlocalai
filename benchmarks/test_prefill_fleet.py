import ast
import io
import json
from pathlib import Path
import subprocess
import shlex
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

import prefill_fleet as fleet


RECEIPT = {
    "event": "restored",
    "original_argv": True,
    "original_container": True,
}


def workers():
    return [
        {
            "label": name,
            "url": "http://" + name + ":8091",
            "last_heartbeat_age": 1,
            "models": ["Qwen3.8-27B"],
        }
        for name in ("target", "other-a", "other-b")
    ]


class ReservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "evidence.jsonl"
        self.argv = [
            "prefill_fleet.py",
            "--host",
            "local",
            "--label",
            "target",
            "--output",
            str(self.output),
        ]
        self.enterContext(patch("sys.argv", self.argv))
        self.enterContext(patch.dict("os.environ", {"EZLOCALAI_API_KEY": "secret"}))
        self.enterContext(patch("builtins.print"))
        self.request = self.enterContext(patch.object(fleet, "request"))
        self.request.side_effect = lambda base, path, *a: (
            {"data": workers()} if path.endswith("workers") else {}
        )
        self.run = self.enterContext(patch.object(fleet, "run_reserved"))
        self.sleep = self.enterContext(patch.object(fleet.time, "sleep"))
        self.popen = self.enterContext(patch.object(fleet.subprocess, "Popen"))

    def test_source_read_failure_precedes_acquisition(self):
        with patch.object(Path, "read_text", side_effect=OSError("missing source")):
            with self.assertRaisesRegex(OSError, "missing source"):
                fleet.main()
        self.request.assert_not_called()
        self.run.assert_not_called()

    def test_mkdir_failure_precedes_acquisition(self):
        with patch.object(Path, "mkdir", side_effect=OSError("read-only")):
            with self.assertRaisesRegex(OSError, "read-only"):
                fleet.main()
        self.request.assert_not_called()

    def test_capacity_rechecked_after_busy_wait_before_next_acquisition(self):
        self.argv += ["--wait-seconds", "60"]
        busy = urllib.error.HTTPError("url", 409, "busy", {}, io.BytesIO(b"busy"))
        self.request.side_effect = [{"data": workers()}, busy, {"data": workers()[:2]}]
        with self.assertRaisesRegex(RuntimeError, "two other"):
            fleet.main()
        self.assertEqual(
            [c.args[1] for c in self.request.call_args_list],
            [
                "/v1/router/workers",
                "/v1/router/drain",
                "/v1/router/workers",
            ],
        )
        self.run.assert_not_called()

    def test_successful_reservation_hands_off_owner_and_latest_target(self):
        fleet.main()
        args, key, maintenance, target, source, command = self.run.call_args.args
        self.assertEqual(key, "secret")
        self.assertEqual(target["label"], "target")
        self.assertIn(maintenance["owner"], command)
        self.assertEqual(command[:3], ["python3", "-u", "-"])
        self.assertIn('sys.modules["prefill_cases"]', source)
        self.assertTrue(self.request.call_args.args[2]["require_idle"])
        self.popen.assert_not_called()

    def test_absent_target_waits_before_acquisition_and_uses_fresh_target(self):
        self.argv += ["--wait-seconds", "60"]
        returned = workers()
        returned[0]["url"] = "http://returned:8091"
        self.request.side_effect = [{"data": workers()[1:]}, {"data": returned}, {}]
        with patch.object(fleet.time, "monotonic", side_effect=[0, 0]):
            fleet.main()
        self.sleep.assert_called_once_with(30)
        self.assertEqual(self.run.call_args.args[3]["url"], "http://returned:8091")
        self.assertEqual(
            [c.args[1] for c in self.request.call_args_list],
            ["/v1/router/workers", "/v1/router/workers", "/v1/router/drain"],
        )
        self.assertFalse(self.output.exists())
        self.popen.assert_not_called()

    def test_absent_target_without_wait_budget_fails_without_mutation(self):
        self.request.return_value = {"data": workers()[1:]}
        self.request.side_effect = None
        with self.assertRaisesRegex(fleet.TargetAbsent, "wait budget exhausted"):
            fleet.main()
        self.request.assert_called_once_with(
            "http://192.168.1.214:8092", "/v1/router/workers"
        )
        self.sleep.assert_not_called()
        self.run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_absent_target_retry_is_bounded_and_final_sleep_is_clipped(self):
        self.argv += ["--wait-seconds", "35"]
        self.request.side_effect = None
        self.request.return_value = {"data": workers()[1:]}
        with patch.object(fleet.time, "monotonic", side_effect=[0, 0, 30, 35]):
            with self.assertRaisesRegex(fleet.TargetAbsent, "wait budget exhausted"):
                fleet.main()
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [30, 5])
        self.assertEqual(self.request.call_count, 3)
        self.assertTrue(
            all(c.args[1].endswith("workers") for c in self.request.call_args_list)
        )
        self.run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_absence_and_busy_reservations_share_one_wait_budget(self):
        self.argv += ["--wait-seconds", "35"]

        def busy():
            return urllib.error.HTTPError("url", 409, "busy", {}, io.BytesIO(b"busy"))

        self.request.side_effect = [
            {"data": workers()},
            busy(),
            {"data": workers()[1:]},
            {"data": workers()},
            busy(),
        ]
        with patch.object(fleet.time, "monotonic", side_effect=[0, 0, 30, 35]):
            with self.assertRaisesRegex(RuntimeError, "reservation failed.*409"):
                fleet.main()
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [30, 5])
        self.run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_absent_target_does_not_hide_lost_spare_capacity(self):
        self.argv += ["--wait-seconds", "60"]
        self.request.side_effect = [{"data": workers()[1:]}, {"data": workers()[1:2]}]
        with self.assertRaisesRegex(RuntimeError, "two other"):
            fleet.main()
        self.sleep.assert_called_once()
        self.assertEqual(self.request.call_count, 2)
        self.run.assert_not_called()

    def test_worker_lookup_errors_are_not_blindly_retried(self):
        self.argv += ["--wait-seconds", "60"]
        for error in (
            ValueError("bad JSON"),
            OSError("unavailable"),
            StopIteration("not a missing target signal"),
            urllib.error.HTTPError("url", 409, "lookup conflict", {}, io.BytesIO()),
            urllib.error.HTTPError("url", 503, "unavailable", {}, io.BytesIO()),
        ):
            with self.subTest(error=error):
                self.request.reset_mock()
                self.request.side_effect = error
                with self.assertRaises(type(error)):
                    fleet.main()
                self.request.assert_called_once()
                self.sleep.assert_not_called()
                self.run.assert_not_called()

    def test_malformed_worker_registry_is_not_absence(self):
        self.argv += ["--wait-seconds", "60"]
        self.request.side_effect = None
        self.request.return_value = {"data": [None]}
        with self.assertRaises(AttributeError):
            fleet.main()
        self.sleep.assert_not_called()
        self.run.assert_not_called()

    def test_explicit_ssh_opt_in_allows_verified_empty_production_key(self):
        self.argv += ["--router-key-ssh", "key-host"]
        with patch.dict("os.environ", {"EZLOCALAI_API_KEY": ""}), patch.object(
            fleet.subprocess, "check_output", return_value=b'[{"Config":{"Env":[]}}]'
        ):
            fleet.main()
        self.assertEqual(self.run.call_args.args[1], "")
        self.assertEqual(self.request.call_args.args[3], "")

    def test_empty_key_without_explicit_ssh_opt_in_is_rejected(self):
        with patch.dict("os.environ", {"EZLOCALAI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "credential required"):
                fleet.main()
        self.request.assert_not_called()

    def test_screen_defaults_preserved(self):
        fleet.main()
        args, _, _, _, _, command = self.run.call_args.args
        self.assertEqual(args.probe_kind, "screen")
        for flag, value in (
            ("--variants", "baseline,ub512,ub2048,baseline"),
            ("--targets", "8000,64000,128000,200000"),
            ("--repeats", "1"),
            ("--tasks", "infra"),
            ("--max-tokens", "1024"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        for flag in (
            "--engine-root",
            "--candidate-model",
            "--candidate-template",
            "--thinking",
        ):
            self.assertNotIn(flag, command)

    def test_profile_keeps_own_defaults_and_does_not_forward_screen_options(self):
        self.argv += ["--probe-kind", "profile"]
        with patch.object(
            fleet, "bundle_source", return_value="profile source"
        ) as bundle:
            fleet.main()
        bundle.assert_called_once_with(Path(fleet.__file__).resolve().parent, "profile")
        command = self.run.call_args.args[-1]
        for flag in (
            "--variants",
            "--targets",
            "--tasks",
            "--thinking",
            "--max-tokens",
            "--engine-root",
            "--candidate-model",
            "--candidate-template",
        ):
            self.assertNotIn(flag, command)

    def test_profile_forwards_explicit_shared_options(self):
        self.argv += [
            "--probe-kind",
            "profile",
            "--variants",
            "baseline",
            "--targets",
            "64000",
            "--repeats",
            "2",
            "--skip-cache",
        ]
        with patch.object(fleet, "bundle_source", return_value="profile source"):
            fleet.main()
        command = self.run.call_args.args[-1]
        for flag, value in (
            ("--variants", "baseline"),
            ("--targets", "64000"),
            ("--repeats", "2"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--skip-cache", command)

    def test_remote_screen_options_preserve_quoted_paths(self):
        self.argv[self.argv.index("--host") + 1] = "user@remote"
        options = {
            "--engine-root": "/opt/engine with spaces/bin",
            "--candidate-model": "/models/model;$(not-a-command).gguf",
            "--candidate-template": "/models/template's file.jinja",
            "--variants": "gdn-off,gdn-on,nvfp4-ub1024",
            "--tasks": "infra,rust,js",
            "--max-tokens": "4096",
        }
        for flag, value in options.items():
            self.argv += [flag, value]
        self.argv += ["--thinking"]
        fleet.main()
        command = self.run.call_args.args[-1]
        self.assertEqual(
            command[:-1],
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "user@remote"],
        )
        remote = shlex.split(command[-1])
        self.assertEqual(remote[:3], ["python3", "-u", "-"])
        for flag, value in options.items():
            self.assertEqual(remote[remote.index(flag) + 1], value)
        self.assertIn("--thinking", remote)

    def test_invalid_screen_options_fail_before_acquisition(self):
        invalid = [
            ["--variants", "gdn-off"],
            ["--variants", "gdn-on"],
            ["--variants", "baseline,nvfp4-ub1024"],
            ["--engine-root", "relative/bin"],
            ["--candidate-model", "relative.gguf"],
            ["--candidate-template", "relative.jinja"],
            ["--tasks", ""],
            ["--tasks", "infra,python"],
            ["--tasks", "infra,"],
            ["--max-tokens", "0"],
            ["--max-tokens", "-1"],
            ["--max-tokens", "127"],
            ["--max-tokens", "8193"],
            ["--probe-kind", "invalid"],
        ]
        for options in invalid:
            with self.subTest(options=options), patch(
                "sys.argv", self.argv + options
            ), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    fleet.main()
                self.assertEqual(error.exception.code, 2)
                self.request.assert_not_called()
                self.run.assert_not_called()

    def test_token_limit_boundaries_are_forwarded(self):
        for limit in ("128", "8192"):
            with self.subTest(limit=limit), patch(
                "sys.argv", self.argv + ["--max-tokens", limit]
            ):
                fleet.main()
                command = self.run.call_args.args[-1]
                self.assertEqual(command[command.index("--max-tokens") + 1], limit)

    def test_profile_rejects_screen_overrides_before_acquisition(self):
        for options in (
            ["--engine-root", "/opt/engine"],
            ["--candidate-model", "/model.gguf"],
            ["--candidate-template", "/template.jinja"],
            ["--thinking"],
            ["--tasks", "rust"],
            ["--max-tokens", "2048"],
        ):
            with self.subTest(options=options), patch(
                "sys.argv", self.argv + ["--probe-kind", "profile"] + options
            ), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    fleet.main()
                self.request.assert_not_called()

    def test_failed_acquisition_never_starts_probe_or_resumes(self):
        for error in (
            ValueError("bad reservation JSON"),
            OSError("router unavailable"),
            urllib.error.HTTPError(
                "url", 403, "forbidden", {}, io.BytesIO(b"forbidden")
            ),
        ):
            self.request.reset_mock()
            self.request.side_effect = [{"data": workers()}, error]
            with self.subTest(error=error), self.assertRaises(
                (ValueError, OSError, RuntimeError)
            ):
                fleet.main()
            self.run.assert_not_called()
            self.assertFalse(
                any(c.args[1].endswith("resume") for c in self.request.call_args_list)
            )


class BundleTests(unittest.TestCase):
    def test_selected_entry_and_module_order_without_executing_probes(self):
        sources = {
            "prefill_cases.py": 'MARKER = "cases"\n',
            "prefill_probe.py": 'from prefill_cases import MARKER\nif __name__ == "__main__":\n    raise RuntimeError("screen must not execute in this test")\n',
            "prefill_profile.py": 'import prefill_probe\nraise RuntimeError("profile must not execute in this test")\n',
        }
        for kind in ("screen", "profile"):
            reads = []

            def read(path):
                reads.append(path.name)
                return sources[path.name]

            with self.subTest(kind=kind), patch.object(Path, "read_text", read):
                source = fleet.bundle_source(Path("/source"), kind)
                tree = ast.parse(source)
                compile(tree, "<bundle>", "exec")  # Syntax validation only.
            modules = ["prefill_cases"] + (
                ["prefill_probe"] if kind == "profile" else []
            )
            entry = "prefill_profile.py" if kind == "profile" else "prefill_probe.py"
            self.assertEqual(reads, [name + ".py" for name in modules] + [entry])
            exec_calls = [
                node.value
                for node in tree.body
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "exec"
            ]
            self.assertEqual(
                [ast.literal_eval(call.args[0]) for call in exec_calls],
                [sources[name + ".py"] for name in modules],
            )
            for call in exec_calls:
                self.assertEqual(ast.unparse(call.args[1]), "m.__dict__")
            self.assertTrue(source.endswith(sources[entry]))
            for name in modules:
                self.assertIn(f'types.ModuleType("{name}")', source)
                self.assertIn(f'sys.modules["{name}"]=m', source)

    def test_unknown_kind_rejected_without_reading_files(self):
        with patch.object(Path, "read_text") as read:
            with self.assertRaises(ValueError):
                fleet.bundle_source(Path("/source"), "unknown")
        read.assert_not_called()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.args = SimpleNamespace(
            router="http://router",
            label="target",
            host="local",
            output=Path(temporary.name) / "evidence.jsonl",
        )
        self.maintenance = {"label": "target", "owner": "owned", "ttl_seconds": 300}
        self.target = workers()[0]
        self.receipt = json.dumps(RECEIPT) + "\n"
        self.health = {"status": "healthy"}
        self.process = Mock(args=["python3"], returncode=0)
        self.process.poll.side_effect = lambda: self.process.returncode
        self.thread = Mock()
        self.thread.is_alive.return_value = False
        self.thread_factory = self.enterContext(
            patch.object(fleet.threading, "Thread", return_value=self.thread)
        )
        self.wait = self.enterContext(patch.object(fleet, "wait_probe", return_value=0))
        self.request = self.enterContext(
            patch.object(fleet, "request", side_effect=self.respond)
        )
        self.popen = self.enterContext(
            patch.object(fleet.subprocess, "Popen", side_effect=self.launch)
        )
        self.enterContext(patch("builtins.print"))

    def respond(self, base, path, *args):
        if path.endswith("workers"):
            return {"data": workers()}
        if path == "/health":
            return self.health
        return {}

    def launch(self, *args, **kwargs):
        kwargs["stdout"].write(self.receipt)
        kwargs["stdout"].flush()
        return self.process

    def run_reserved(self):
        fleet.run_reserved(
            self.args, "secret", self.maintenance, self.target, "source", ["python3"]
        )

    def resumed(self):
        return [c for c in self.request.call_args_list if c.args[1].endswith("resume")]

    def test_success_requires_receipt_exit_health_then_stops_renewal_before_resume(
        self,
    ):
        def respond(base, path, *args):
            if path.endswith("resume"):
                self.thread.join.assert_called_once_with(timeout=25)
                self.assertEqual(self.process.poll(), 0)
            return self.respond(base, path, *args)

        self.request.side_effect = respond
        self.run_reserved()
        self.assertEqual(len(self.resumed()), 1)
        self.assertEqual(
            self.resumed()[0].args[2], {"label": "target", "owner": "owned"}
        )
        self.assertEqual(
            [c.args[1] for c in self.request.call_args_list],
            [
                "/v1/router/workers",
                "/health",
                "/v1/router/resume",
            ],
        )

    def test_capacity_lost_after_acquisition_releases_only_untouched_healthy_target(
        self,
    ):
        def respond(base, path, *args):
            if path.endswith("workers"):
                return {"data": workers()[:2]}
            return self.respond(base, path, *args)

        self.request.side_effect = respond
        with self.assertRaisesRegex(RuntimeError, "two other"):
            self.run_reserved()
        self.popen.assert_not_called()
        self.thread.start.assert_not_called()
        self.assertEqual(len(self.resumed()), 1)

    def test_open_failure_releases_untouched_healthy_target(self):
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_reserved()
        self.popen.assert_not_called()
        self.assertEqual(len(self.resumed()), 1)

    def test_absent_target_after_acquisition_is_not_retried_or_launched(self):
        def respond(base, path, *args):
            if path.endswith("workers"):
                return {"data": workers()[1:]}
            return self.respond(base, path, *args)

        self.request.side_effect = respond
        with patch.object(fleet.time, "sleep") as sleep:
            with self.assertRaises(fleet.TargetAbsent):
                self.run_reserved()
        sleep.assert_not_called()
        self.popen.assert_not_called()
        self.thread.start.assert_not_called()
        self.assertFalse(self.args.output.exists())
        # Preserve existing cleanup for a never-launched, verified healthy target.
        self.assertEqual(len(self.resumed()), 1)

    def test_absent_target_after_acquisition_unhealthy_stays_fail_closed(self):
        def respond(base, path, *args):
            if path.endswith("workers"):
                return {"data": workers()[1:]}
            return {"status": "unhealthy"}

        self.request.side_effect = respond
        with self.assertRaisesRegex(RuntimeError, "health unconfirmed"):
            self.run_reserved()
        self.popen.assert_not_called()
        self.assertFalse(self.resumed())

    def test_thread_setup_failure_releases_untouched_healthy_target(self):
        self.thread_factory.side_effect = RuntimeError("thread setup")
        with self.assertRaisesRegex(RuntimeError, "thread setup"):
            self.run_reserved()
        self.popen.assert_not_called()
        self.assertEqual(len(self.resumed()), 1)

    def test_launch_failure_releases_untouched_healthy_target(self):
        self.popen.side_effect = OSError("cannot launch")
        with self.assertRaisesRegex(OSError, "cannot launch"):
            self.run_reserved()
        self.assertEqual(len(self.resumed()), 1)
        self.thread.join.assert_called_once()

    def test_nonzero_exit_with_restoration_resumes_but_reports_failure(self):
        self.process.returncode = 1
        self.wait.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "Probe failed; production restored"):
            self.run_reserved()
        self.assertEqual(len(self.resumed()), 1)

    def test_timeout_with_restoration_resumes_but_preserves_timeout(self):
        self.wait.side_effect = subprocess.TimeoutExpired("probe", 2900)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_reserved()
        self.assertEqual(len(self.resumed()), 1)

    def test_live_probe_never_resumes_even_with_receipt(self):
        self.process.returncode = None
        self.wait.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            self.run_reserved()
        self.assertFalse(self.resumed())
        self.thread.join.assert_called_once()

    def test_renewal_failure_does_not_resume_live_probe(self):
        def failed(process, source, event):
            event.set()
            process.returncode = None
            raise RuntimeError("renewal failed")

        self.wait.side_effect = failed
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            self.run_reserved()
        self.assertFalse(self.resumed())

    def test_renewal_failure_after_confirmed_restoration_releases_and_reports(self):
        def failed(process, source, event):
            event.set()
            return 0

        self.wait.side_effect = failed
        with self.assertRaisesRegex(
            RuntimeError, "renewal failed; production restored"
        ):
            self.run_reserved()
        self.assertEqual(len(self.resumed()), 1)

    def test_missing_malformed_or_nonfinal_receipts_fail_closed(self):
        for receipt in (
            "",
            "garbage\n",
            "[]\n",
            "null\n",
            '{"event":"restored"}\n',
            json.dumps(dict(RECEIPT, original_argv=1)),
            json.dumps(dict(RECEIPT, original_container=False)),
            json.dumps(RECEIPT) + '\n{"event":"ready"}\n',
            json.dumps(RECEIPT) + '\n{"partial":',
        ):
            with self.subTest(receipt=receipt):
                self.args.output.unlink(missing_ok=True)
                self.request.reset_mock()
                self.receipt = receipt
                with self.assertRaisesRegex(RuntimeError, "restoration"):
                    self.run_reserved()
                self.assertFalse(self.resumed())

    def test_evidence_read_failure_fails_closed(self):
        with patch.object(Path, "read_text", side_effect=OSError("unreadable")):
            with self.assertRaisesRegex(
                RuntimeError, "Unreadable restoration evidence"
            ):
                self.run_reserved()
        self.assertFalse(self.resumed())

    def test_unhealthy_production_fails_closed(self):
        for health in ({"status": "unhealthy"}, {}, [], None):
            with self.subTest(health=health):
                self.args.output.unlink(missing_ok=True)
                self.health = health
                with self.assertRaisesRegex(RuntimeError, "health unconfirmed"):
                    self.run_reserved()
                self.assertFalse(self.resumed())

    def test_health_request_failure_fails_closed(self):
        def respond(base, path, *args):
            if path == "/health":
                raise OSError("unreachable")
            return self.respond(base, path, *args)

        self.request.side_effect = respond
        with self.assertRaisesRegex(OSError, "unreachable"):
            self.run_reserved()
        self.assertFalse(self.resumed())

    def test_inflight_renewal_blocks_resume(self):
        self.thread.is_alive.return_value = True
        with self.assertRaisesRegex(RuntimeError, "renewal still in flight"):
            self.run_reserved()
        self.assertFalse(self.resumed())

    def test_remote_health_uses_remote_host_for_container_url(self):
        self.args.host = "user@remote"
        remote = workers()
        remote[0]["url"] = "http://172.18.0.2:8091"

        def respond(base, path, *args):
            if path.endswith("workers"):
                return {"data": remote}
            return self.respond(base, path, *args)

        self.request.side_effect = respond
        self.run_reserved()
        self.request.assert_any_call("http://remote:8091", "/health")


class ProbeWaitTests(unittest.TestCase):
    def setUp(self):
        self.process = Mock(args=["python3"], returncode=None)
        self.process.poll.side_effect = lambda: self.process.returncode
        self.event = threading.Event()

    def finish(self, **kwargs):
        self.process.returncode = 0
        return (None, None)

    def test_communication_retry_does_not_resend_source(self):
        calls = 0

        def communicate(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise subprocess.TimeoutExpired("probe", 1)
            return self.finish()

        self.process.communicate.side_effect = communicate
        self.assertEqual(fleet.wait_probe(self.process, "source", self.event), 0)
        self.assertEqual(
            self.process.communicate.call_args_list[0].kwargs["input"], "source"
        )
        self.assertIsNone(self.process.communicate.call_args_list[1].kwargs["input"])
        self.process.terminate.assert_not_called()

    def test_timeout_gracefully_terminates_and_waits_for_restoration(self):
        self.process.communicate.side_effect = self.finish
        with patch.object(fleet.time, "monotonic", side_effect=[0, 2901]):
            with self.assertRaises(subprocess.TimeoutExpired):
                fleet.wait_probe(self.process, "source", self.event)
        self.process.terminate.assert_called_once()
        self.process.communicate.assert_called_once_with(timeout=300)
        self.process.kill.assert_not_called()

    def test_renewal_failure_interrupts_then_waits_instead_of_blind_resume(self):
        def communicate(**kwargs):
            if "input" in kwargs:
                self.event.set()
                raise subprocess.TimeoutExpired("probe", 1)
            return self.finish()

        self.process.communicate.side_effect = communicate
        with self.assertRaisesRegex(RuntimeError, "renewal failed"):
            fleet.wait_probe(self.process, "source", self.event)
        self.process.terminate.assert_called_once()
        self.assertEqual(self.process.returncode, 0)

    def test_stuck_cleanup_is_killed_and_reaped(self):
        self.event.set()
        self.process.communicate.side_effect = [
            subprocess.TimeoutExpired("probe", 300),
            (None, None),
        ]
        with self.assertRaisesRegex(RuntimeError, "renewal failed"):
            fleet.wait_probe(self.process, "source", self.event)
        self.process.terminate.assert_called_once()
        self.process.kill.assert_called_once()
        self.assertEqual(self.process.communicate.call_args.kwargs, {"timeout": 20})

    def test_keyboard_interrupt_also_waits_for_cleanup(self):
        self.process.communicate.side_effect = [KeyboardInterrupt(), (None, None)]
        with self.assertRaises(KeyboardInterrupt):
            fleet.wait_probe(self.process, "source", self.event)
        self.process.terminate.assert_called_once()
        self.assertEqual(self.process.communicate.call_args.kwargs, {"timeout": 300})


class CapacityTests(unittest.TestCase):
    def test_unhealthy_stale_drained_external_and_duplicate_capacity_rejected(self):
        for changes in (
            {"draining": True},
            {"external_provider": True},
            {"last_heartbeat_age": 30},
            {"last_heartbeat_age": -1},
            {"models": ["other-model"]},
            {"label": "other-a"},
        ):
            rows = workers()
            rows[2].update(changes)
            with self.subTest(changes=changes), patch.object(
                fleet, "request", return_value={"data": rows}
            ):
                with self.assertRaisesRegex(RuntimeError, "two other"):
                    fleet.available_target("http://router", "target")


if __name__ == "__main__":
    unittest.main()
