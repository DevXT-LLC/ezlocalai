import csv
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import prefill_probe as probe
from prefill_probe import variant_args


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.original = [
            "llama-server",
            "-m",
            "/app/models/Qwen.gguf",
            "--port",
            "8092",
            "-c",
            "230000",
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

    def test_only_batch_changes(self):
        before = self.original.copy()
        result = variant_args(before, "ub2048")
        expected = before[1:].copy()
        expected[expected.index("--port") + 1] = "18092"
        expected[expected.index("-ub") + 1] = "2048"
        self.assertEqual(result, expected)
        self.assertEqual(before, self.original)

    def test_diagnostic_does_not_change_context_or_precision(self):
        result = variant_args(self.original, "mtp-off")
        self.assertEqual(result[result.index("--spec-type") + 1], "none")
        for key in ("-c", "-ctk", "-ctv", "-ub", "-b"):
            self.assertEqual(
                result[result.index(key) + 1],
                self.original[self.original.index(key) + 1],
            )

    def test_reject_unknown_and_oversized_batch(self):
        for variant in ("ub8192", "q8", "smaller-context"):
            with self.assertRaises(ValueError):
                variant_args(self.original, variant)
        self.original[self.original.index("-b") + 1] = "512"
        with self.assertRaises(ValueError):
            variant_args(self.original, "ub1024")

    def test_kernel_experiment_preserves_all_engine_arguments(self):
        expected = variant_args(self.original, "baseline")
        for name in ("gdn-on", "gdn-off"):
            self.assertEqual(variant_args(self.original, name), expected)

    def test_candidate_changes_only_model_and_physical_batch(self):
        expected = variant_args(self.original, "ub1024")
        expected[expected.index("-m") + 1] = "/prefill-candidate.gguf"
        self.assertEqual(variant_args(self.original, "nvfp4-ub1024"), expected)
        self.original[self.original.index("-b") + 1] = "512"
        with self.assertRaises(ValueError):
            variant_args(self.original, "nvfp4-ub1024")

    def test_ub3072_preserves_logical_batch_context_and_other_options(self):
        original = self.original.copy()
        expected = variant_args(original, "baseline")
        expected[expected.index("-ub") + 1] = "3072"
        self.assertEqual(variant_args(original, "ub3072"), expected)
        self.assertEqual(self.original, original)
        original[original.index("-b") + 1] = "2048"
        with self.assertRaises(ValueError):
            variant_args(original, "ub3072")


class OperatorTests(unittest.TestCase):
    # Exact current CSV printer order. It omits `passed`; test mode leaves
    # backend_reg_name empty (unlike performance mode).
    fields = [
        "backend_name",
        "op_name",
        "op_params",
        "test_mode",
        "supported",
        "error_message",
        "backend_reg_name",
    ]

    def rows(self):
        return [
            dict(
                zip(
                    self.fields,
                    [
                        "CUDA0",
                        "GATED_DELTA_NET",
                        f"type=f32,head_count=16,head_size=128,n_seq_tokens={tokens},"
                        f"n_seqs={seqs},v_repeat={repeat},permuted={permuted},kda=0,K=1",
                        "test",
                        "1",
                        "",
                        "",
                    ],
                )
            )
            for tokens, seqs, repeat, permuted in [
                (63, 1, 2, 0),
                (64, 1, 4, 0),
                (65, 1, 1, 0),
                (65, 3, 4, 0),
                (128, 1, 2, 0),
                (129, 1, 2, 0),
                (129, 2, 4, 1),
            ]
        ]

    def output(self, rows, fields=None):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields or self.fields)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()

    def invoke(self, output, codes=("0", "0"), fails=False):
        def command(*args):
            return output if args[:2] == ("docker", "logs") else ""

        with patch.object(probe, "run", side_effect=command) as commands, patch.object(
            probe.subprocess, "check_output", side_effect=codes
        ) as wait, patch.object(probe, "emit") as events:
            if fails:
                with self.assertRaises((RuntimeError, csv.Error)):
                    probe.check_operators(
                        {"Image": "pinned-image"}, "owned", "owner", Path("/stage")
                    )
            else:
                probe.check_operators(
                    {"Image": "pinned-image"}, "owned", "owner", Path("/stage")
                )
        return commands, wait, events

    def test_current_csv_without_passed_runs_both_modes(self):
        rows = self.rows()
        self.assertEqual({r["op_params"] for r in rows}, probe.GDN_OPERATOR_PARAMS)
        commands, wait, events = self.invoke(self.output(rows))
        launches = [
            c.args for c in commands.call_args_list if c.args[:2] == ("docker", "run")
        ]
        self.assertEqual(len(launches), 2)
        for enabled, launch in zip(("0", "1"), launches):
            self.assertIn("GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=" + enabled, launch)
            self.assertIn("type=bind,src=/stage,dst=/prefill-bin,readonly", launch)
            self.assertIn("LD_LIBRARY_PATH=/prefill-bin:/usr/local/cuda/lib64", launch)
            self.assertIn("/prefill-bin/test-backend-ops", launch)
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(
            [c.kwargs["passed_cases"] for c in events.call_args_list], [7, 7]
        )
        self.assertEqual(
            sum(c.args == ("docker", "rm", "owned") for c in commands.call_args_list), 2
        )

    def test_nonzero_exit_rejects_even_seven_success_rows(self):
        for codes in (("1",), ("124",), ("137",), ("0", "1")):
            with self.subTest(codes=codes):
                self.invoke(self.output(self.rows()), codes=codes, fails=True)

    def test_missing_duplicate_and_extra_cases_rejected(self):
        rows = self.rows()
        for bad in ([], rows[:-1], rows[:-1] + [rows[0]], rows + [rows[0]]):
            with self.subTest(count=len(bad)):
                self.invoke(self.output(bad), fails=True)

    def test_seven_distinct_wrong_cases_are_rejected(self):
        rows = self.rows()
        rows[-1]["op_params"] = rows[-1]["op_params"].replace("K=1", "K=2")
        self.assertEqual(len({r["op_params"] for r in rows}), 7)
        self.invoke(self.output(rows), fails=True)

    def test_case_status_mode_backend_and_error_are_checked(self):
        for field, value in (
            ("supported", "0"),
            ("error_message", "compare failed"),
            ("test_mode", "perf"),
            ("backend_name", "CUDA1"),
            ("op_name", "MUL_MAT"),
        ):
            with self.subTest(field=field):
                rows = self.rows()
                rows[0][field] = value
                self.invoke(self.output(rows), fails=True)

    def test_missing_required_column_and_polluted_csv_rejected(self):
        rows = self.rows()
        for row in rows:
            del row["error_message"]
        self.invoke(
            self.output(rows, [f for f in self.fields if f != "error_message"]),
            fails=True,
        )
        self.invoke("unexpected output\n" + self.output(self.rows()), fails=True)
        self.invoke(self.output(self.rows()) + "unexpected output\n", fails=True)

    def test_wait_timeout_propagates_without_starting_next_mode(self):
        with patch.object(probe, "run") as commands, patch.object(
            probe.subprocess,
            "check_output",
            side_effect=subprocess.TimeoutExpired("docker wait", 600),
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                probe.check_operators(
                    {"Image": "pinned"}, "owned", "owner", Path("/stage")
                )
        self.assertEqual(commands.call_count, 1)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "bin"
        self.root.mkdir()
        self.pins = {}
        for name in probe.GDN_BUILD_HASHES:
            path = self.root / name
            path.write_bytes(name.encode())
            path.chmod(0o755)
            self.pins[name] = hashlib.sha256(name.encode()).hexdigest()
            if ".so." in name:
                stem, version = name.split(".so.", 1)
                (self.root / (stem + ".so")).symlink_to(name)
                (self.root / (stem + ".so." + version.split(".")[0])).symlink_to(name)
        pinned = patch.object(probe, "GDN_BUILD_HASHES", self.pins)
        pinned.start()
        self.addCleanup(pinned.stop)

    def manifest(self, prefix="/validation/build/bin/", overrides=None):
        pins = {name: self.pins[name] for name in probe.GDN_ANCHORS}
        pins.update(overrides or {})
        path = self.root / "artifact-sha256.txt"
        path.write_text(
            "".join(
                f"{digest}  {prefix}{name}\n" for name, digest in sorted(pins.items())
            )
        )
        return path

    def test_hashes_all_files_and_aliases_with_relative_keys(self):
        (self.root / "extra-helper").write_bytes(b"helper")
        (self.root / "extra-helper").chmod(0o755)
        (self.root / "nested").mkdir()
        (self.root / "nested" / "libextra.so").write_bytes(b"nested")
        result = probe.engine_provenance(self.root, require_gdn=True)
        self.assertTrue(result["gdn_build_verified"])
        self.assertEqual(result["engine_sha256"], self.pins["llama-server"])
        hashes = result["engine_files_sha256"]
        self.assertEqual(
            set(hashes),
            {
                p.relative_to(self.root).as_posix()
                for p in self.root.rglob("*")
                if p.is_file()
            },
        )
        self.assertEqual(
            hashes["libggml-cuda.so.0"], self.pins["libggml-cuda.so.0.29.0"]
        )

    def test_missing_payload_and_soname_fail(self):
        for name in ("libllama-server-impl.so", "libggml-cuda.so.0"):
            with self.subTest(name=name):
                path = self.root / name
                saved = path.rename(self.root / (name + ".saved"))
                try:
                    with self.assertRaises(ValueError):
                        probe.engine_provenance(self.root, require_gdn=True)
                finally:
                    saved.rename(path)

    def test_unpatched_cuda_or_mixed_operator_and_libraries_rejected(self):
        for name in ("libggml-cuda.so.0.29.0", "test-backend-ops", "libllama.so.0.9.0"):
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(b"stale or unpatched")
                try:
                    with self.assertRaisesRegex(ValueError, "pinned GDN artifact"):
                        probe.engine_provenance(self.root, require_gdn=True)
                finally:
                    path.write_bytes(name.encode())

    def test_matching_optional_manifest_accepts_supported_path_forms(self):
        for prefix in ("", "/validation/build/bin/", str(self.root) + "/"):
            with self.subTest(prefix=prefix):
                self.manifest(prefix)
                self.assertTrue(
                    probe.engine_provenance(self.root, require_gdn=True)[
                        "gdn_build_verified"
                    ]
                )

    def test_stale_manifest_rejected_even_when_binaries_match(self):
        self.manifest(overrides={"libggml-cuda.so.0.29.0": "0" * 64})
        with self.assertRaisesRegex(ValueError, "Stale or invalid"):
            probe.engine_provenance(self.root, require_gdn=True)

    def test_empty_incomplete_malformed_and_escape_manifests_rejected(self):
        path = self.root / "artifact-sha256.txt"
        for content in (
            "",
            "not a checksum\n",
            f"{self.pins['llama-server']}  llama-server\n",
            f"{self.pins['llama-server']}  ../llama-server\n",
            f"{self.pins['llama-server']}  /other/llama-server\n",
        ):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    probe.engine_provenance(self.root, require_gdn=True)

    def test_duplicate_manifest_entry_rejected(self):
        path = self.manifest()
        path.write_text(path.read_text() * 2)
        with self.assertRaises(ValueError):
            probe.engine_provenance(self.root, require_gdn=True)

    def test_external_and_broken_symlinks_rejected(self):
        outside = self.root.parent / "outside.so"
        outside.write_bytes(b"outside")
        link = self.root / "extra.so"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            probe.engine_provenance(self.root, require_gdn=True)
        outside.unlink()
        with self.assertRaises(FileNotFoundError):
            probe.engine_provenance(self.root, require_gdn=True)

    def test_non_executable_binary_rejected(self):
        (self.root / "test-backend-ops").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "Missing executable"):
            probe.engine_provenance(self.root, require_gdn=True)

    def test_non_gdn_engine_does_not_require_pins_or_manifest(self):
        (self.root / "libllama.so.0.9.0").write_bytes(b"other engine")
        self.manifest(overrides={"llama-server": "0" * 64})
        result = probe.engine_provenance(self.root)
        self.assertNotIn("gdn_build_verified", result)


class ScreenSafetyTests(unittest.TestCase):
    def setUp(self):
        self.original = [
            "/usr/local/bin/llama-server",
            "-m",
            "Qwen3.8-27B.gguf",
            "--port",
            "8092",
            "--alias",
            "qwen",
            "-c",
            "262144",
            "-b",
            "4096",
            "-ub",
            "2048",
            "--spec-type",
            "draft-mtp",
        ]
        self.production = {
            "Id": "production-id",
            "Image": "sha256:original",
            "Config": {"Env": [], "Labels": {}},
            "State": {"Running": True, "Health": {"Status": "healthy"}},
        }
        self.container = self.owned_container()
        self.actions, self.events, self.handlers = [], [], {}
        self.inspect_error = None
        self.after_remove_error = False
        self.removed = False
        self.keep_container = False
        self.recovery_timeout = False
        self.watchdog = Mock()
        self.watchdog.poll.return_value = None
        self.watchdog.terminate.side_effect = lambda: self.actions.append("terminate")
        self.watchdog.wait.side_effect = lambda **kw: self.actions.append("reaped") or 0
        self.enterContext(
            patch.object(probe.subprocess, "run", side_effect=self.command)
        )
        self.enterContext(
            patch.object(probe.subprocess, "check_output", side_effect=self.output)
        )
        self.enterContext(
            patch.object(probe.subprocess, "Popen", return_value=self.watchdog)
        )
        self.enterContext(patch.object(probe, "engines", return_value=[self.original]))
        self.enterContext(patch.object(probe, "emit", side_effect=self.emit))
        self.enterContext(
            patch.object(probe.signal, "signal", side_effect=self.install_signal)
        )
        self.alarm = self.enterContext(patch.object(probe.signal, "alarm"))
        self.enterContext(patch.object(probe.time, "sleep"))

    def owned_container(self):
        return {
            "Id": "owned-id",
            "Image": "sha256:original",
            "Config": {"Labels": {"xt.prefill.owner": "owner"}},
        }

    def install_signal(self, sig, handler):
        previous = self.handlers.get(sig, signal.SIG_DFL)
        self.handlers[sig] = handler
        return previous

    def emit(self, event, **values):
        self.actions.append(event)
        self.events.append(dict(event=event, **values))

    def command(self, args, **kwargs):
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 120)
        args = list(args)
        if args[:2] == ["docker", "inspect"]:
            if args[-1] in ("ezlocalai", "production-id"):
                container = self.production
            else:
                if self.inspect_error or (self.after_remove_error and self.removed):
                    return subprocess.CompletedProcess(
                        args, 1, "", self.inspect_error or "Cannot connect to Docker"
                    )
                container = self.container
            return subprocess.CompletedProcess(
                args,
                0 if container else 1,
                json.dumps([container]) if container else "[]\n",
                "" if container else "error: no such object: owned",
            )
        self.actions.append(args)
        if args[:3] == ["docker", "rm", "-f"]:
            self.assertEqual(args[-1], "owned-id")
            self.removed = True
            if not self.keep_container:
                self.container = None
        elif args[:2] == ["docker", "start"]:
            self.assertEqual(args[-1], "production-id")
            if self.recovery_timeout:
                self.handlers[signal.SIGALRM](signal.SIGALRM, None)
            self.production["State"]["Running"] = True
        elif args[:2] == ["docker", "stop"]:
            self.production["State"]["Running"] = False
        elif args[:2] == ["docker", "run"]:
            self.container = self.owned_container()
        elif args[:2] == ["docker", "exec"] or args[0] == "nvidia-smi":
            pass
        else:
            self.fail("Unexpected subprocess: " + repr(args))
        return subprocess.CompletedProcess(args, 0, "hash-or-output", "")

    def output(self, args, **kwargs):
        result = self.command(args, **kwargs)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, args)
        return result.stdout

    def restore(self):
        probe.restore(
            self.production, "probe", "owner", self.original, [], self.watchdog
        )

    def watchdog_run(self):
        with patch(
            "sys.argv",
            [
                "watchdog",
                "probe",
                "owner",
                "production-id",
                "sha256:original",
                str(probe.WATCHDOG_SECONDS),
                str(probe.RESTORE_SECONDS),
            ],
        ):
            exec(compile(probe.WATCHDOG, "<watchdog>", "exec"), {})

    def assert_no_restart(self):
        self.assertNotIn(["docker", "start", "production-id"], self.actions)
        self.assertNotIn("restored", self.actions)

    def test_cleanup_and_watchdog_require_confirmed_absence(self):
        for runner in (self.restore, self.watchdog_run):
            for failure in (
                "inspect",
                "post-remove-inspect",
                "still-present",
                "owner",
                "image",
            ):
                with self.subTest(runner=runner.__name__, failure=failure):
                    self.container = self.owned_container()
                    self.actions.clear()
                    self.removed = False
                    self.inspect_error = (
                        "Cannot connect to Docker" if failure == "inspect" else None
                    )
                    self.after_remove_error = failure == "post-remove-inspect"
                    self.keep_container = failure == "still-present"
                    if failure == "owner":
                        self.container["Config"]["Labels"][
                            "xt.prefill.owner"
                        ] = "foreign"
                    if failure == "image":
                        self.container["Image"] = "sha256:foreign"
                    with self.assertRaises(RuntimeError):
                        runner()
                    self.assert_no_restart()
                    if failure in ("owner", "image", "inspect"):
                        self.assertFalse(
                            any(
                                isinstance(a, list) and a[:2] == ["docker", "rm"]
                                for a in self.actions
                            )
                        )

    def test_lowercase_absence_is_safe_for_both_recovery_paths(self):
        for runner in (self.restore, self.watchdog_run):
            with self.subTest(runner=runner.__name__):
                self.container = None
                self.actions.clear()
                runner()
                self.assertIn(["docker", "start", "production-id"], self.actions)

    def test_owned_probe_is_removed_by_id_and_watchdog_reaped_before_receipt(self):
        self.restore()
        self.assertIsNone(self.container)
        self.assertLess(
            self.actions.index(["docker", "rm", "-f", "owned-id"]),
            self.actions.index(["docker", "start", "production-id"]),
        )
        self.assertLess(self.actions.index("terminate"), self.actions.index("reaped"))
        self.assertLess(self.actions.index("reaped"), self.actions.index("restored"))

    def test_standalone_watchdog_removes_owned_probe_before_restart(self):
        self.watchdog_run()
        self.assertIsNone(self.container)
        self.assertLess(
            self.actions.index(["docker", "rm", "-f", "owned-id"]),
            self.actions.index(["docker", "start", "production-id"]),
        )

    def test_watchdog_reaping_failure_never_emits_restored(self):
        self.watchdog.wait.side_effect = subprocess.TimeoutExpired("watchdog", 5)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.restore()
        self.assertNotIn("restored", self.actions)

    def test_watchdog_rejects_changed_production_image(self):
        self.production["Image"] = "sha256:changed"
        self.container = None
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.watchdog_run()
        self.assert_no_restart()

    def invoke_main(self, measure):
        self.container = None
        argv = [
            "probe",
            "--owned-drain",
            "owner",
            "--variants",
            "baseline",
            "--targets",
            "2000",
            "--skip-cache",
        ]
        with patch("sys.argv", argv), patch.object(
            probe.socket, "socket"
        ), patch.object(probe, "wait_health"), patch.object(
            probe, "calibrate", return_value=({"messages": []}, 2000)
        ), patch.object(
            probe, "measure", side_effect=measure
        ):
            probe.main()

    def test_main_absolute_deadline_interrupts_work_and_restores(self):
        def expired(*args, **kwargs):
            self.assertIn("deadline", kwargs)
            self.handlers[signal.SIGALRM](signal.SIGALRM, None)

        with self.assertRaisesRegex(TimeoutError, "phase deadline"):
            self.invoke_main(expired)
        self.assertTrue(self.production["State"]["Running"])
        self.assertIsNone(self.container)
        self.assertEqual(self.events[-1]["event"], "restored")
        self.assertIn(unittest.mock.call(probe.WORK_SECONDS), self.alarm.call_args_list)
        self.assertIn(
            unittest.mock.call(probe.RESTORE_SECONDS), self.alarm.call_args_list
        )
        self.assertLess(
            probe.WORK_SECONDS + probe.RESTORE_SECONDS, probe.WATCHDOG_SECONDS
        )
        self.assertLess(probe.RESTORE_SECONDS, 300)
        self.assertEqual(self.alarm.call_args, unittest.mock.call(0))

    def test_main_success_restores_and_reaps_watchdog(self):
        self.invoke_main(lambda *args, **kwargs: {"quality_pass": True})
        self.assertEqual(self.events[-1]["event"], "restored")
        self.assertLess(self.actions.index("reaped"), self.actions.index("restored"))
        self.assertIsNone(self.container)

    def test_main_recovery_deadline_fails_closed_with_watchdog_retained(self):
        self.recovery_timeout = True
        with self.assertRaisesRegex(TimeoutError, "phase deadline"):
            self.invoke_main(lambda *args, **kwargs: {"quality_pass": True})
        self.assertEqual(self.events[-1]["event"], "restoration_failed")
        self.assertTrue(self.events[-1]["watchdog_retained"])
        self.watchdog.terminate.assert_not_called()
        self.assertNotIn("restored", self.actions)


class StreamDeadlineTests(unittest.TestCase):
    def test_expired_deadline_never_submits_request(self):
        with patch.object(probe.time, "monotonic", return_value=10), patch.object(
            probe, "http"
        ) as http:
            with self.assertRaisesRegex(TimeoutError, "work deadline"):
                probe.measure({"messages": []}, "model", deadline=10)
        http.assert_not_called()

    def test_keepalive_stream_cannot_extend_absolute_deadline(self):
        response = MagicMock()
        response.__enter__.return_value = iter([b": keepalive\n", b": keepalive\n"])
        with patch.object(
            probe.time, "monotonic", side_effect=[0, 0, 9, 10]
        ), patch.object(probe, "http", return_value=response) as http:
            with self.assertRaisesRegex(TimeoutError, "work deadline"):
                probe.measure({"messages": []}, "model", deadline=10)
        self.assertEqual(http.call_args.kwargs["timeout"], 10)
        response.__exit__.assert_called_once()


class PreflightTests(unittest.TestCase):
    def test_bad_gdn_staging_rejected_before_any_subprocess_or_watchdog(self):
        with tempfile.TemporaryDirectory() as root, patch(
            "sys.argv",
            [
                "probe",
                "--owned-drain",
                "owner",
                "--variants",
                "gdn-off,gdn-on",
                "--engine-root",
                root,
            ],
        ), patch.object(probe, "run") as command, patch.object(
            probe.subprocess, "Popen"
        ) as watchdog:
            with self.assertRaises(ValueError):
                probe.main()
        command.assert_not_called()
        watchdog.assert_not_called()

    def test_ordinary_screen_skips_engine_provenance(self):
        with patch("sys.argv", ["probe", "--owned-drain", "owner"]), patch.object(
            probe, "engine_provenance"
        ) as provenance, patch.object(
            probe, "run", side_effect=RuntimeError("read-only inspect reached")
        ) as command, patch.object(
            probe.subprocess, "Popen"
        ) as watchdog:
            with self.assertRaisesRegex(RuntimeError, "read-only inspect reached"):
                probe.main()
        provenance.assert_not_called()
        command.assert_called_once_with("docker", "inspect", "ezlocalai")
        watchdog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
