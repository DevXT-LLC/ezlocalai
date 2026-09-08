import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cli


class NativeTtsInstallTests(unittest.TestCase):
    def test_all_images_install_a_websocket_transport(self):
        root = Path(__file__).parent
        for name in (
            "requirements.txt",
            "cuda-requirements.txt",
            "rocm-requirements.txt",
            "rpi-requirements.txt",
        ):
            self.assertIn("websockets", (root / name).read_text().splitlines())

    def test_native_install_uses_checked_out_source_and_removes_old_package(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            state = Path(directory) / "state"
            with mock.patch.object(cli, "STATE_DIR", state), mock.patch.object(
                cli, "_get_pip_cmd", return_value=["uv", "pip", "uninstall"]
            ), mock.patch.object(cli.subprocess, "run") as run:
                run.return_value.returncode = 0
                cli._build_native_tts("python", "nvidia", source)
                self.assertEqual(run.call_args_list[0].args[0][-1], "qwen-tts")
                self.assertNotIn("-y", run.call_args_list[0].args[0])
                build = run.call_args_list[1].args[0]
                self.assertEqual(build[1], str(source / "scripts" / "build_tts.py"))
                self.assertIn("--cuda", build)
                self.assertIn(str(state / "tts-build"), build)

    def test_version_pins_match_all_prebuilt_images(self):
        root = Path(__file__).parent
        for name in (
            "Dockerfile",
            "cuda.Dockerfile",
            "rocm.Dockerfile",
            "rpi.Dockerfile",
        ):
            self.assertIn(
                f"xllamacpp=={cli.XLLAMACPP_VERSION}", (root / name).read_text()
            )


if __name__ == "__main__":
    unittest.main()
