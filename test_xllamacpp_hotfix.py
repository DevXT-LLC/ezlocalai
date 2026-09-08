"""Build/provenance contracts; real GPU regression is in benchmark_vision.py."""

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from scripts import build_xllamacpp as builder
from ezlocalai.Speculative import dflash_hotfix_status


class XllamacppHotfixTests(unittest.TestCase):
    def test_cached_native_build_forces_relink_without_removing_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            output = source / "build/lib.linux/xllamacpp"
            output.mkdir(parents=True)
            binary = output / "xllamacpp.abi3.so"
            binary.write_bytes(b"old extension")
            preserved = output / "other.so"
            preserved.write_bytes(b"not our generated binding")
            builder.invalidate_extension(source)
            self.assertFalse(binary.exists())
            self.assertTrue(preserved.exists())

    def test_official_wheel_is_identified_as_unpatched(self):
        with patch.dict(sys.modules, {"xllamacpp._ezlocalai_hotfix": None}):
            self.assertEqual(dflash_hotfix_status(), "unpatched")

    def test_patched_wheel_reports_its_marker(self):
        module = types.ModuleType("xllamacpp._ezlocalai_hotfix")
        module.HOTFIX = builder.HOTFIX
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertEqual(dflash_hotfix_status(), builder.HOTFIX)

    def test_cuda_build_includes_patch_and_all_three_gpu_families(self):
        docker = Path("cuda.Dockerfile").read_text()
        self.assertIn("native/patches", docker)
        self.assertIn("build_xllamacpp.py --cuda --install", docker)
        self.assertIn("86-real;89-real;120-real", docker)
        self.assertIn("ARG XLLAMACPP_BUILD_JOBS=20", docker)

    def test_unknown_source_revision_is_rejected_without_patching(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(builder, "run", return_value=Mock(stdout="wrong-sha\n")):
                with self.assertRaisesRegex(RuntimeError, "Refusing to patch"):
                    builder.prepare_source(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_pinned_source_gets_patch_and_provenance_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "patches/llama.cpp").mkdir(parents=True)
            (source / "src/xllamacpp").mkdir(parents=True)
            with patch.object(
                builder,
                "run",
                side_effect=[
                    Mock(stdout=builder.XLLAMACPP_REVISION),
                    Mock(stdout=builder.LLAMA_REVISION),
                ],
            ):
                builder.prepare_source(source)
            self.assertIn(
                builder.HOTFIX,
                (source / "src/xllamacpp/_ezlocalai_hotfix.py").read_text(),
            )
            content = (
                source / "patches/llama.cpp/0003-dflash-pinned-image-positions.patch"
            ).read_text()
            self.assertIn("has_embeddings && n_rows > 1 && pos_pinned", content)
            self.assertNotIn("n_rows > 1 && is_mrope", content)

    def test_conflicting_patch_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            location = (
                source / "patches/llama.cpp/0003-dflash-pinned-image-positions.patch"
            )
            location.parent.mkdir(parents=True)
            location.write_text("user changes")
            with patch.object(
                builder,
                "run",
                side_effect=[
                    Mock(stdout=builder.XLLAMACPP_REVISION),
                    Mock(stdout=builder.LLAMA_REVISION),
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                    builder.prepare_source(source)
            self.assertEqual(location.read_text(), "user changes")


if __name__ == "__main__":
    unittest.main()
