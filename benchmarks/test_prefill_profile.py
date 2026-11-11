"""Offline lifecycle tests. Docker, Nsight, HTTP and watchdog spawning are mocked."""

import io
import errno
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, Mock, patch

import prefill_profile as profile


class ProfileLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        nsight = root / "nsight"
        for relative in ("target-linux-x64/nsys", "host-linux-x64/QdstrmImporter"):
            path = nsight / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.stack.enter_context(patch.object(profile, "NSIGHT_ROOT", nsight))
        self.stack.enter_context(
            patch.object(profile, "NSYS", str(nsight / "target-linux-x64/nsys"))
        )
        self.stack.enter_context(patch.object(profile, "REPORT_ROOT", root / "reports"))
        self.original = [
            "/usr/local/bin/llama-server",
            "-m",
            "/app/models/Qwen3.8-27B.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "8092",
            "--alias",
            "qwen",
            "-c",
            "262144",
            "-b",
            "4096",
            "-ub",
            "1024",
            "--spec-type",
            "draft-mtp",
            "-ctk",
            "q4_0",
            "-ctv",
            "q4_0",
        ]
        self.voice = [
            "/usr/local/bin/llama-server",
            "-m",
            "LFM3.gguf",
            "--port",
            "8093",
        ]
        self.production = {
            "Id": "production-id",
            "Image": "sha256:immutable",
            "Name": "/ezlocalai",
            "Config": {
                "Env": [
                    "GGML_FLAG=1",
                    "CUDA_FLAG=2",
                    "NVIDIA_FLAG=3",
                    "API_KEY=private",
                ],
                "Labels": {},
            },
            "HostConfig": {"Memory": 0},
            "Mounts": [{"Source": "/models", "Destination": "/app/models", "RW": True}],
            "State": {"Running": True, "Health": {"Status": "healthy"}},
        }
        self.container = None
        self.calls = []
        self.events = []
        self.failure = None
        self.bad_owner = False
        self.missing_report = False
        self.warning = ""
        self.stop_warning = ""
        self.kernel_stats = (
            '"Time (%)","Total Time (ns)","Instances","Name"\n100,1234,5,kernel\n'
        )
        self.cache_n = 0
        self.tokens = 64000
        self.watchdog = Mock()
        self.watchdog.poll.return_value = None
        self.stack.enter_context(
            patch.object(profile.subprocess, "Popen", return_value=self.watchdog)
        )
        self.stack.enter_context(
            patch.object(profile.subprocess, "check_output", side_effect=self.command)
        )
        self.stack.enter_context(
            patch.object(profile.subprocess, "run", side_effect=self.inspect_command)
        )
        self.stack.enter_context(patch.object(profile, "check_ports"))
        self.stack.enter_context(
            patch.object(
                profile.probe, "engines", return_value=[self.original, self.voice]
            )
        )
        self.stack.enter_context(
            patch.object(profile.probe, "wait_health", side_effect=self.health)
        )
        self.stack.enter_context(
            patch.object(profile.probe, "calibrate", side_effect=self.calibrate)
        )
        self.measure_mock = self.stack.enter_context(
            patch.object(profile.probe, "measure", side_effect=self.measure)
        )
        self.stack.enter_context(
            patch.object(profile.probe, "emit", side_effect=self.emit)
        )
        self.alarm = self.stack.enter_context(patch.object(profile.signal, "alarm"))
        self.handlers = {}
        self.stack.enter_context(
            patch.object(profile.signal, "signal", side_effect=self.install_signal)
        )
        self.stack.enter_context(patch.object(profile.time, "sleep"))

    def install_signal(self, sig, handler):
        previous = self.handlers.get(sig, signal.SIG_DFL)
        self.handlers[sig] = handler
        return previous

    def emit(self, event, **values):
        self.events.append(dict(event=event, **values))

    def inspect_command(self, args, **kwargs):
        self.assertGreater(kwargs["timeout"], 0)
        self.assertEqual(args[:2], ["docker", "inspect"])
        name = args[2]
        if name in ("ezlocalai", self.production["Id"]):
            result = self.production
        else:
            result = self.container
        if result is None:
            return subprocess.CompletedProcess(args, 1, "", "Error: No such object")
        return subprocess.CompletedProcess(args, 0, json.dumps([result]), "")

    def command(self, args, **kwargs):
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 120)
        args = list(args)
        self.calls.append(args)
        if args == [profile.NSYS, "--version"]:
            return "Nsight Systems 2025.5.2"
        if args[:2] == ["docker", "stop"]:
            self.production["State"]["Running"] = False
            if self.failure == "stop-lost-response":
                raise subprocess.TimeoutExpired(args, 60)
        elif args[:2] == ["docker", "run"]:
            labels = dict(
                args[i + 1].split("=", 1)
                for i, arg in enumerate(args)
                if arg == "--label"
            )
            self.directory = Path(
                next(
                    arg.split("src=", 1)[1].split(",dst=", 1)[0]
                    for arg in args
                    if "dst=/reports" in arg
                )
            )
            self.container = {
                "Id": "owned-id",
                "Image": self.production["Image"],
                "Config": {"Labels": labels},
                "State": {"Running": True},
            }
            if self.failure == "run-lost-response":
                raise subprocess.TimeoutExpired(args, 60)
        elif args[:2] == ["docker", "exec"]:
            if "launch" in args:
                (self.directory / "nsys-launch.log").write_text(self.warning)
            if "start" in args:
                if self.failure == "start":
                    raise subprocess.CalledProcessError(1, args)
            if "stop" in args:
                if self.failure == "flush":
                    raise subprocess.TimeoutExpired(args, 120)
                if not self.missing_report:
                    (self.directory / "cold.nsys-rep").write_bytes(b"mock report")
                return self.stop_warning or "ok"
            if "stats" in args:
                if self.failure == "transfer-stats" and any(
                    "cuda_gpu_mem_time_sum" in arg for arg in args
                ):
                    raise subprocess.CalledProcessError(1, args)
                return self.kernel_stats
        elif args[:3] == ["docker", "rm", "-f"]:
            self.assertEqual(args[3], "owned-id")
            self.container = None
        elif args[:2] == ["docker", "start"]:
            self.assertEqual(args[2], self.production["Id"])
            self.production["State"]["Running"] = True
        else:
            self.fail_test(args)
        return "ok"

    def fail_test(self, args):
        raise AssertionError("Unexpected subprocess: " + repr(args))

    def health(self, port):
        self.calls.append(["health", port])
        if self.failure == "health":
            raise TimeoutError("No health")
        if self.failure == "voice-health" and port == 18093:
            raise TimeoutError("No voice health")

    def calibrate(self, target):
        self.calls.append(["calibrate", target])
        return {"messages": [], "expected": {}}, self.tokens

    def measure(self, case, model, cache):
        self.calls.append(["request", model, cache])
        if self.failure == "request":
            raise OSError("Connection closed")
        if self.failure == "interrupt":
            self.handlers[signal.SIGHUP](signal.SIGHUP, None)
        if self.failure == "deadline":
            self.handlers[signal.SIGALRM](signal.SIGALRM, None)
        if self.bad_owner:
            self.container["Config"]["Labels"]["xt.prefill.profile"] = "other"
        if self.failure == "changed-production":
            self.production["Config"]["Env"].append("CHANGED=1")
        return {
            "timings": {"cache_n": self.cache_n, "prompt_n": self.tokens},
            "quality_pass": True,
        }

    def invoke(self):
        profile.main(
            [
                "--owned-drain",
                "prefill-owned",
                "--variants",
                "baseline,ub2048,mtp-off",
                "--targets",
                "8000,128000",
                "--repeats",
                "9",
                "--skip-cache",
            ]
        )

    def assert_restored(self):
        self.assertTrue(self.production["State"]["Running"])
        self.assertIsNone(self.container)
        self.assertEqual(self.events[-1]["event"], "restored")
        self.assertTrue(self.events[-1]["original_argv"])
        self.assertTrue(self.events[-1]["original_container"])
        self.watchdog.terminate.assert_called_once()
        self.assertIn(
            unittest.mock.call(profile.WORK_SECONDS), self.alarm.call_args_list
        )
        self.assertIn(
            unittest.mock.call(profile.RESTORE_SECONDS), self.alarm.call_args_list
        )

    def test_full_success_lifecycle_is_one_baseline_cold_request(self):
        self.invoke()
        self.assert_restored()
        self.measure_mock.assert_called_once()
        self.assertEqual(
            next(c for c in self.calls if c[0] == "calibrate"), ["calibrate", 64000]
        )
        launch = next(c for c in self.calls if "launch" in c)
        self.assertEqual(
            launch[-len(self.original[1:]) :],
            profile.probe.variant_args(self.original, "baseline"),
        )
        start = next(
            i
            for i, c in enumerate(self.calls)
            if c[:2] == ["docker", "exec"] and "start" in c
        )
        request = next(i for i, c in enumerate(self.calls) if c[0] == "request")
        stop = next(
            i
            for i, c in enumerate(self.calls)
            if c[:2] == ["docker", "exec"] and "stop" in c
        )
        self.assertLess(self.calls.index(["health", 18093]), start)
        self.assertLess(start, request)
        self.assertLess(request, stop)
        provenance = self.events[0]
        self.assertEqual(provenance["variants"], ["baseline"])
        self.assertEqual(provenance["targets"], [64000])
        self.assertEqual(provenance["repeats"], 1)
        self.assertTrue(self.events[-2]["capture_success"])
        self.assertTrue((self.directory / "transfer-api-summary.txt").is_file())
        self.assertTrue(
            any(
                "--report=cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum" in c
                for c in self.calls
            )
        )
        self.assertTrue(self.events[-2]["aggregate_transfer_time_is_not_critical_path"])
        self.assertTrue(
            json.loads((self.directory / "restoration.json").read_text())[
                "original_container"
            ]
        )

    def test_mounts_environment_and_shutdown_are_scoped(self):
        self.invoke()
        command = next(c for c in self.calls if c[:2] == ["docker", "run"])
        self.assertIn("production-id:ro", command)
        self.assertIn(
            f"type=bind,src={profile.NSIGHT_ROOT},dst={profile.NSIGHT_ROOT},readonly",
            command,
        )
        self.assertIn("sha256:immutable", command)
        for value in ("GGML_FLAG=1", "CUDA_FLAG=2", "NVIDIA_FLAG=3"):
            self.assertIn(value, command)
        flattened = " ".join(str(v) for c in self.calls for v in c)
        for forbidden in (
            "private",
            "--privileged",
            "--cap-add",
            "--pid=host",
            "pkill",
            "killall",
        ):
            self.assertNotIn(forbidden, flattened)
        shutdown = next(c for c in self.calls if "shutdown" in c)
        self.assertIn("--kill=sigterm", shutdown)
        self.assertTrue(any(v.startswith("--session=prefill") for v in shutdown))
        self.assertEqual(self.directory.parent, profile.REPORT_ROOT)
        self.assertEqual(self.directory.stat().st_mode & 0o777, 0o700)

    def test_failed_actions_and_interrupts_restore_without_success_claim(self):
        for failure in (
            "health",
            "voice-health",
            "start",
            "request",
            "flush",
            "interrupt",
            "deadline",
            "stop-lost-response",
            "run-lost-response",
            "transfer-stats",
        ):
            with self.subTest(failure=failure):
                self.failure = failure
                self.events.clear()
                self.watchdog.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "Profile failed"):
                    self.invoke()
                self.assert_restored()
                self.assertFalse(self.events[-2]["capture_success"])

    def test_incomplete_capture_is_not_success(self):
        for field, value in (
            ("missing_report", True),
            ("kernel_stats", "No CUDA events"),
            ("warning", "CUPTI_ERROR_INVALID_PARAMETER"),
            ("stop_warning", "Warning: not all CUDA events were collected"),
            ("cache_n", 1),
        ):
            with self.subTest(field=field):
                previous = getattr(self, field)
                setattr(self, field, value)
                self.events.clear()
                self.watchdog.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "Profile failed"):
                    self.invoke()
                self.assert_restored()
                self.assertFalse(self.events[-2]["capture_success"])
                setattr(self, field, previous)

    def test_calibration_failure_never_starts_capture_or_inference(self):
        self.tokens = 60000
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_restored()
        self.measure_mock.assert_not_called()

        self.assertFalse(
            any(c[:2] == ["docker", "exec"] and "start" in c for c in self.calls)
        )

    def test_unowned_probe_is_not_removed_and_watchdog_retained(self):
        self.bad_owner = True
        with self.assertRaisesRegex(RuntimeError, "unowned"):
            self.invoke()
        self.assertFalse(any(c[:2] == ["docker", "rm"] for c in self.calls))
        self.assertFalse(any(c[:2] == ["docker", "start"] for c in self.calls))
        self.watchdog.terminate.assert_not_called()
        self.assertEqual(self.events[-1]["event"], "restoration_failed")

    def test_changed_production_is_not_started(self):
        self.failure = "changed-production"
        with self.assertRaisesRegex(RuntimeError, "identity/configuration"):
            self.invoke()
        self.assertIsNone(self.container)
        self.assertFalse(any(c[:2] == ["docker", "start"] for c in self.calls))
        self.watchdog.terminate.assert_not_called()

    def test_running_screen_is_never_joined(self):
        self.production["State"]["Running"] = False
        with self.assertRaisesRegex(RuntimeError, "never join"):
            self.invoke()
        self.assertFalse(any(c[0] == "docker" for c in self.calls))
        profile.subprocess.Popen.assert_not_called()
        self.assertEqual(self.events[-1]["event"], "restoration_failed")

    def test_preflight_busy_port_emits_verified_untouched_receipt(self):
        profile.check_ports.side_effect = OSError(errno.EADDRINUSE, "Address in use")
        with self.assertRaises(OSError):
            self.invoke()
        self.assertEqual(self.events[-1]["event"], "restored")
        for key in ("original_untouched", "original_container", "original_argv"):
            self.assertTrue(self.events[-1][key])
        self.assertEqual(self.events[-1]["phase"], "preflight")
        self.assertFalse(self.events[-2]["capture_success"])
        self.assertFalse(any(c[0] == "docker" for c in self.calls))
        self.assertFalse(profile.REPORT_ROOT.exists())
        profile.subprocess.Popen.assert_not_called()
        self.measure_mock.assert_not_called()

        from prefill_fleet import require_restoration

        output = profile.REPORT_ROOT.parent / "preflight-output.jsonl"
        output.write_text("\n".join(json.dumps(event) for event in self.events) + "\n")
        require_restoration(output)

    def test_preflight_receipt_refuses_changed_snapshot(self):
        initial = json.dumps(self.production)
        for change in (
            "config",
            "pid",
            "started",
            "restart",
            "health",
            "argv",
            "extra-engine",
        ):
            with self.subTest(change=change):
                self.production = json.loads(initial)
                self.events.clear()
                profile.probe.engines.return_value = [self.original, self.voice]

                def busy_port():
                    if change == "config":
                        self.production["Config"]["Env"].append("CHANGED=1")
                    elif change == "pid":
                        self.production["State"]["Pid"] = 9
                    elif change == "started":
                        self.production["State"]["StartedAt"] = "new-start"
                    elif change == "restart":
                        self.production["RestartCount"] = 1
                    elif change == "health":
                        self.production["State"]["Health"]["Status"] = "unhealthy"
                    elif change == "argv":
                        profile.probe.engines.return_value = [self.original]
                    else:
                        profile.probe.engines.return_value = [
                            self.original,
                            self.voice,
                            ["other"],
                        ]
                    raise OSError(errno.EADDRINUSE, "Address in use")

                profile.check_ports.side_effect = busy_port
                with self.assertRaises(OSError):
                    self.invoke()
                self.assertEqual(self.events[-1]["event"], "restoration_failed")
                self.assertFalse(any(e["event"] == "restored" for e in self.events))
        self.assertFalse(any(c[0] == "docker" for c in self.calls))
        profile.subprocess.Popen.assert_not_called()

    def test_preflight_verification_error_stays_fail_closed(self):
        def busy_port():
            profile.subprocess.run.side_effect = TimeoutError("Docker unavailable")
            raise OSError(errno.EADDRINUSE, "Address in use")

        profile.check_ports.side_effect = busy_port
        with self.assertRaises(OSError):
            self.invoke()
        self.assertEqual(self.events[-1]["event"], "restoration_failed")
        self.assertEqual(self.events[-1]["error_type"], "TimeoutError")
        self.assertFalse(any(c[0] == "docker" for c in self.calls))

    def test_missing_nsight_can_certify_untouched_production(self):
        Path(profile.NSYS).unlink()
        with self.assertRaisesRegex(RuntimeError, "Complete Nsight"):
            self.invoke()
        self.assertTrue(self.events[-1]["original_untouched"])
        self.assertEqual(self.events[-1]["event"], "restored")
        profile.subprocess.Popen.assert_not_called()

    def test_inspect_daemon_error_is_not_treated_as_absent(self):
        with patch.object(
            profile.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                [], 1, "", "Cannot connect to Docker"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "inspect"):
                profile.inspect_container("probe", optional=True)

    def test_early_dead_watchdog_prevents_cutover(self):
        self.watchdog.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "Profile failed"):
            self.invoke()
        self.assertFalse(any(c[:2] == ["docker", "stop"] for c in self.calls))
        self.assertEqual(self.events[-1]["event"], "restored")

    def test_watchdog_standalone_recovers_after_controller_loss(self):
        self.invoke()
        state = json.loads(profile.subprocess.Popen.call_args.args[0][-1])
        self.production["State"]["Running"] = False
        self.container = {
            "Id": "owned-id",
            "Image": state["image"],
            "State": {"Running": True},
            "Config": {
                "Labels": {
                    "xt.prefill.owner": state["owner"],
                    "xt.prefill.profile": state["uuid"],
                }
            },
        }
        self.run_watchdog(state)
        receipt = json.loads((self.directory / "watchdog-restoration.json").read_text())
        self.assertTrue(receipt["original_container"])
        self.assertTrue(receipt["original_argv"])
        self.assertIsNone(self.container)

    def run_watchdog(self, state):
        def call(args, **kwargs):
            if args[:2] == ("docker", "inspect"):
                return self.inspect_command(list(args), **kwargs)
            self.command(args, **kwargs)
            return subprocess.CompletedProcess(args, 0, "", "")

        files = {
            "/proc/1/cmdline": b"\0".join(v.encode() for v in self.original),
            "/proc/2/cmdline": b"\0".join(v.encode() for v in self.voice),
        }
        with patch.object(profile.subprocess, "run", side_effect=call), patch(
            "sys.argv", ["watchdog", json.dumps(state)]
        ), patch("glob.glob", return_value=list(files)), patch(
            "builtins.open", side_effect=lambda path, mode: io.BytesIO(files[path])
        ):
            exec(compile(profile.WATCHDOG, "watchdog", "exec"), {})

    def test_watchdog_refuses_foreign_container(self):
        self.invoke()
        state = json.loads(profile.subprocess.Popen.call_args.args[0][-1])
        self.container = {
            "Id": "foreign",
            "Image": state["image"],
            "Config": {"Labels": {}},
            "State": {"Running": True},
        }
        self.calls.clear()
        self.run_watchdog(state)
        receipt = json.loads((self.directory / "watchdog-restoration.json").read_text())
        self.assertFalse(receipt["original_container"])
        self.assertIn("unowned", receipt["error"])
        self.assertFalse(
            any(c[:2] in (["docker", "rm"], ["docker", "start"]) for c in self.calls)
        )

    def test_restore_deadline_keeps_watchdog_and_failure_evidence(self):
        with patch.object(
            profile, "restore", side_effect=TimeoutError("recovery deadline")
        ):
            with self.assertRaises(TimeoutError):
                self.invoke()
        self.watchdog.terminate.assert_not_called()
        self.assertEqual(self.events[-1]["event"], "restoration_failed")
        self.assertTrue((self.directory / "profile-status.json").is_file())

    def test_signal_deadline_bounds_streaming_not_only_socket_reads(self):
        self.failure = "deadline"
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assert_restored()
        self.assertEqual(self.events[-2]["error_type"], "TimeoutError")

    def test_report_parser_requires_actual_kernel_instances(self):
        self.assertFalse(profile.has_kernels('"Instances","Name"\n0,kernel'))
        self.assertFalse(profile.has_kernels("No CUDA kernel data"))
        self.assertTrue(profile.has_kernels(self.kernel_stats))

    def test_real_controller_bundle_imports_without_file_or_runtime_dependency(self):
        from prefill_fleet import bundle_source

        source = bundle_source(Path(__file__).parent, "profile")
        namespace = {"__name__": "bundled_profile"}
        with patch.dict(sys.modules):
            exec(compile(source, "<profile-bundle>", "exec"), namespace)
            self.assertEqual(namespace["probe"].__name__, "prefill_probe")
            self.assertTrue(callable(namespace["main"]))
            self.assertEqual(namespace["TARGET"], 64000)
        profile.subprocess.Popen.assert_not_called()
        profile.subprocess.check_output.assert_not_called()
        profile.subprocess.run.assert_not_called()


class PortGuardTests(unittest.TestCase):
    def test_reuseaddr_is_set_before_each_bind_without_reuseport(self):
        guard = MagicMock()
        with patch.object(profile.socket, "socket") as factory:
            factory.return_value.__enter__.return_value = guard
            profile.check_ports()
        self.assertEqual(
            guard.method_calls,
            [
                unittest.mock.call.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                ),
                unittest.mock.call.bind(("127.0.0.1", 18092)),
                unittest.mock.call.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                ),
                unittest.mock.call.bind(("127.0.0.1", 18093)),
            ],
        )

    def test_reuseaddr_does_not_bypass_real_listener(self):
        # Use an ephemeral local port, never a production or experiment port.
        with socket.socket() as listener, socket.socket() as guard:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            proxy = MagicMock()
            proxy.setsockopt.side_effect = guard.setsockopt
            proxy.bind.side_effect = lambda address: guard.bind(listener.getsockname())
            with patch.object(profile.socket, "socket") as factory:
                factory.return_value.__enter__.return_value = proxy
                with self.assertRaises(OSError) as raised:
                    profile.check_ports()
            self.assertEqual(raised.exception.errno, errno.EADDRINUSE)


if __name__ == "__main__":
    unittest.main()
