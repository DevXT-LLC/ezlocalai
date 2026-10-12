"""Build/provenance contracts; real GPU regression is in benchmark_vision.py."""

import tempfile
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from scripts import build_xllamacpp as builder
from ezlocalai.Speculative import dflash_hotfix_status


class XllamacppHotfixTests(unittest.TestCase):
    def test_arm64_native_caches_are_isolated_from_x86_builds(self):
        def cache_mounts(filename):
            mounts = {}
            for spec in re.findall(r"--mount=(\S+)", Path(filename).read_text()):
                options = dict(part.split("=", 1) for part in spec.split(","))
                if options.get("type") == "cache":
                    mounts[options["target"]] = options
            return mounts

        cuda, gb10 = cache_mounts("cuda.Dockerfile"), cache_mounts("gb10.Dockerfile")
        self.assertTrue(gb10)
        for target, options in gb10.items():
            self.assertNotEqual(
                options.get("id", target), cuda[target].get("id", target), target
            )
            # A second GB10 build must not concurrently mutate the source tree.
            self.assertEqual(options.get("sharing"), "locked")

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

    def test_cuda_build_includes_patch_and_all_three_gpu_families(self):
        docker = Path("cuda.Dockerfile").read_text()
        self.assertIn("build_xllamacpp.py --cuda --install", docker)
        self.assertIn("86-real;89-real;120-real", docker)
        self.assertIn("ARG XLLAMACPP_BUILD_JOBS=20", docker)

    def test_unknown_source_revision_is_rejected_without_patching(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(builder, "run", return_value=Mock(stdout="wrong-sha\n")):
                with self.assertRaisesRegex(RuntimeError, "Refusing to patch"):
                    builder.prepare_source(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
