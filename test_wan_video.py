import ast
import asyncio
from contextlib import chdir
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image
from fastapi import HTTPException
from ezlocalai.WAN import WanVideo, T2V_REPO, I2V_REPO, download_wan_models
from scripts.download_wan_models import activate_wan


class WanVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        binary = self.root / "sd-cli"
        binary.touch()
        self.env = mock.patch.dict(
            os.environ, {"SDCPP_BIN": str(binary), "VIDEO_GPU_RESIDENCY": "auto"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_selects_both_pinned_experts_for_each_mode(self):
        for i2v, repo in ((False, T2V_REPO), (True, I2V_REPO)):
            with mock.patch(
                "huggingface_hub.hf_hub_download", return_value="weight"
            ) as fetch:
                assets = download_wan_models(i2v, "Q4_K_M")
            self.assertEqual(set(assets), {"high", "low", "encoder", "vae"})
            self.assertEqual(fetch.call_args_list[0].args, (repo,))
            self.assertIn("HighNoise", fetch.call_args_list[0].kwargs["filename"])
            self.assertIn("LowNoise", fetch.call_args_list[1].kwargs["filename"])
            self.assertTrue(
                all(len(c.kwargs["revision"]) == 40 for c in fetch.call_args_list)
            )

    def test_command_uses_two_experts_and_preserves_total_steps(self):
        video = WanVideo(device="cuda:1")
        args = video._command(
            dict(low="low", high="high", encoder="te", vae="vae"),
            "frame_%06d.png",
            prompt="cup",
            negative_prompt="blur",
            width=512,
            height=320,
            frames=33,
            fps=24,
            steps=18,
            guidance_scale=3.5,
            image_path="input.png",
        )
        value = lambda option: args[args.index(option) + 1]
        self.assertEqual(value("--high-noise-diffusion-model"), "high")
        self.assertEqual(value("--diffusion-model"), "low")
        self.assertEqual(value("--backend"), "cuda1")
        self.assertEqual(value("-M"), "vid_gen")
        self.assertEqual(value("--steps"), "10")
        self.assertEqual(value("--high-noise-steps"), "8")
        self.assertNotIn("--mmap", args)
        self.assertIn("--offload-to-cpu", args)
        self.assertEqual(value("-i"), "input.png")

    def test_native_frames_are_encoded_to_mp4_for_both_modes(self):
        real_run = subprocess.run
        for initial_image in (None, Image.new("RGB", (100, 80), "blue")):
            commands = []

            def run(cmd, **kwargs):
                commands.append(cmd)
                if cmd[0] == "ffmpeg":
                    return real_run(cmd, **kwargs)
                self.assertEqual("-i" in cmd, initial_image is not None)
                pattern = cmd[cmd.index("-o") + 1]
                for frame in range(5):
                    Image.new("RGB", (64, 64), (frame * 40, 0, 255)).save(
                        pattern % frame
                    )
                return types.SimpleNamespace(returncode=0)

            with chdir(self.root), mock.patch(
                "ezlocalai.WAN.download_wan_models",
                return_value=dict.fromkeys(("low", "high", "encoder", "vae"), "weight"),
            ) as download, mock.patch("ezlocalai.WAN.subprocess.run", side_effect=run):
                result = WanVideo().generate(
                    "cup",
                    size="64x64",
                    num_frames=5,
                    num_inference_steps=4,
                    image=initial_image,
                )
                self.assertGreater(Path(result).stat().st_size, 100)
                self.assertEqual(
                    download.call_args.kwargs,
                    {"image_to_video": initial_image is not None},
                )
                self.assertEqual(commands[-1][0], "ffmpeg")
                self.assertFalse(
                    Path(commands[0][commands[0].index("-o") + 1]).parent.exists()
                )

    def test_failure_or_timeout_never_returns_partial_video(self):
        for side_effect in (None, subprocess.TimeoutExpired("sd-cli", 1)):
            with chdir(self.root), mock.patch(
                "ezlocalai.WAN.download_wan_models",
                return_value=dict.fromkeys(("low", "high", "encoder", "vae"), "weight"),
            ), mock.patch(
                "ezlocalai.WAN.subprocess.run",
                side_effect=side_effect,
                return_value=types.SimpleNamespace(returncode=1),
            ):
                with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                    WanVideo().generate("cup", size="64x64", num_frames=5)
                self.assertEqual(list(Path("outputs").glob("*.mp4")), [])

    def test_invalid_conditions_and_i2v_without_image_do_not_download(self):
        with mock.patch("ezlocalai.WAN.download_wan_models") as download:
            with self.assertRaisesRegex(ValueError, "first-image"):
                WanVideo().generate("cup", conditions=[{"index": 0}])
            with self.assertRaisesRegex(ValueError, "requires an image"):
                WanVideo(model=I2V_REPO).generate("cup")
            with self.assertRaises(ValueError):
                WanVideo().generate("cup", guidance_scale=float("nan"))
        download.assert_not_called()

    def test_activation_preserves_other_settings_and_private_original(self):
        env = self.root / ".env"
        original = "OTHER=keep\nVIDEO_MODEL=old\n"
        env.write_text(original)
        activate_wan(env)
        activate_wan(env)
        self.assertEqual(env.read_text(), f"OTHER=keep\nVIDEO_MODEL={T2V_REPO}\n")
        backup = self.root / ".env.pre-wan2.2"
        self.assertEqual(backup.read_text(), original)
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)


class WanEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_request_is_400_and_empty_output_is_500(self):
        tree = ast.parse(Path("app.py").read_text())
        endpoint = next(
            n
            for n in tree.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate_video"
        )
        endpoint.decorator_list = []
        endpoint.args.defaults = [ast.Constant(None)]
        for arg in endpoint.args.args:
            arg.annotation = None
        request = types.SimpleNamespace(
            prompt="cup",
            response_format="url",
            size="64x64",
            num_frames=5,
            num_inference_steps=4,
            guidance_scale=3.5,
            frame_rate=24,
            image=None,
            conditions=None,
            n=1,
        )
        pipes = types.SimpleNamespace(
            get_image_server_client=lambda: types.SimpleNamespace(is_configured=False),
            should_use_ezlocalai_fallback=lambda: (False, ""),
            get_fallback_client=lambda: None,
        )
        for error, output, status in (
            (ValueError("invalid conditions"), None, 400),
            (None, None, 500),
        ):
            namespace = {
                "pipe": types.SimpleNamespace(
                    generate_video=mock.AsyncMock(
                        side_effect=error, return_value=output
                    )
                ),
                "HTTPException": HTTPException,
                "_router_client_for_unserved_capability": mock.AsyncMock(
                    return_value=None
                ),
                "_has_local_capability": lambda _: True,
            }
            exec(
                compile(
                    ast.fix_missing_locations(
                        ast.Module(body=[endpoint], type_ignores=[])
                    ),
                    "app.py",
                    "exec",
                ),
                namespace,
            )
            with mock.patch.dict("sys.modules", {"Pipes": pipes}), self.assertRaises(
                HTTPException
            ) as raised:
                await namespace["generate_video"](request)
            self.assertEqual(raised.exception.status_code, status)
