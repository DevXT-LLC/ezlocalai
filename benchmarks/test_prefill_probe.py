import csv
import hashlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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
