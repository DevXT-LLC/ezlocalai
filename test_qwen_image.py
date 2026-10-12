import ast
import asyncio
import base64
import io
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types
import unittest
from typing import List, Optional, Literal
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator
from ezlocalai.IMG import IMG


def png(color="red"):
    with io.BytesIO() as buffer:
        Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
        return buffer.getvalue()


def extract(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class QwenImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.img = IMG.__new__(IMG)
        self.img._available = True
        self.img.local_uri = None
        self.img.device = "cpu"
        self.img.models_dir = self.temp.name
        self.img.sdcli_bin = "sd-cli"
        for attr in ("_diffusion_path", "_vae_path", "_llm_path", "_mmproj_path"):
            path = Path(self.temp.name, attr)
            path.touch()
            setattr(self.img, attr, str(path))
        self.ref = base64.b64encode(png()).decode()
        self.cmd = None
        self.tmp_dir = None

    def run_cli(self, cmd, **kwargs):
        self.cmd = cmd
        output = Path(cmd[cmd.index("-o") + 1])
        self.tmp_dir = output.parent
        output.write_bytes(png("blue"))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"generated")

    def generate(self, **kwargs):
        with patch("ezlocalai.IMG.subprocess.run", side_effect=self.run_cli):
            return self.img.generate("change color", size="64x64", **kwargs)

    def test_text_to_image_returns_png_and_honors_steps(self):
        result = self.generate(num_inference_steps=7)
        with Image.open(io.BytesIO(base64.b64decode(result))) as output:
            self.assertEqual(output.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(self.cmd[self.cmd.index("--steps") + 1], "7")
        self.assertNotIn("--flow-shift", self.cmd)
        self.assertNotIn("-r", self.cmd)
        self.assertFalse(self.tmp_dir.exists())

    def test_device_selection_is_explicit(self):
        self.generate()
        self.assertEqual(self.cmd[self.cmd.index("--backend") + 1], "cpu")
        self.img.device = "cuda:1"
        with patch.object(self.img, "_should_offload", return_value=True):
            self.generate()
        self.assertEqual(self.cmd[self.cmd.index("--backend") + 1], "cuda1")
        self.assertIn("--offload-to-cpu", self.cmd)

    def test_single_image_has_visual_conditioning_and_strength(self):
        self.generate(image=self.ref, strength=0.3)
        self.assertIn("--llm_vision", self.cmd)
        self.assertEqual(self.cmd.count("-r"), 1)
        self.assertEqual(self.cmd[self.cmd.index("--strength") + 1], "0.3")
        self.assertEqual(
            self.cmd[self.cmd.index("-r") + 1],
            self.cmd[self.cmd.index("--init-img") + 1],
        )

    def test_multiple_references_preserve_order_without_init(self):
        self.generate(images=[self.ref, "data:image/png;base64," + self.ref])
        refs = [self.cmd[i + 1] for i, value in enumerate(self.cmd) if value == "-r"]
        self.assertEqual([Path(p).name for p in refs], ["ref_0.png", "ref_1.png"])
        self.assertNotIn("--init-img", self.cmd)

    def test_primary_plus_ten_additional_references(self):
        self.generate(image=self.ref, images=[self.ref] * 10)
        self.assertEqual(self.cmd.count("-r"), 11)

    def test_invalid_input_never_invokes_cli_and_cleans_temporary_files(self):
        with patch(
            "ezlocalai.IMG.tempfile.TemporaryDirectory",
            wraps=tempfile.TemporaryDirectory,
        ) as temp:
            with patch("ezlocalai.IMG.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "index 1"):
                    self.img.generate("x", images=[self.ref, "invalid"], size="64x64")
                run.assert_not_called()
                self.assertEqual(temp.call_count, 1)

    def test_missing_output_cannot_return_input_png(self):
        def no_output(cmd, **kwargs):
            self.tmp_dir = Path(cmd[cmd.index("-o") + 1]).parent
            return subprocess.CompletedProcess(cmd, 0, stdout=b"no image")

        with patch("ezlocalai.IMG.subprocess.run", side_effect=no_output):
            with self.assertRaisesRegex(RuntimeError, "without an output"):
                self.img.generate("x", image=self.ref, size="64x64")
        self.assertFalse(self.tmp_dir.exists())

    def test_process_failure_and_timeout_are_errors(self):
        for result in (
            subprocess.CompletedProcess([], 2, stdout=b"bad model"),
            subprocess.TimeoutExpired([], 600),
        ):
            with self.subTest(result=result):
                with patch(
                    "ezlocalai.IMG.subprocess.run",
                    side_effect=result if isinstance(result, Exception) else None,
                    return_value=result,
                ):
                    with self.assertRaises(RuntimeError):
                        self.img.generate("x", size="64x64")

    def test_invalid_parameters(self):
        for args in (
            {"size": "0x512"},
            {"size": "513x512"},
            {"size": "abc"},
            {"strength": -0.1},
            {"strength": float("nan")},
            {"images": [self.ref] * 11},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.img.generate("x", **args)

    def test_missing_vision_weights_fails_edit(self):
        Path(self.img._mmproj_path).unlink()
        with self.assertRaisesRegex(RuntimeError, "vision weights"):
            self.generate(images=[self.ref])

    def test_load_http_image_closes_response(self):
        response = Mock()
        response.iter_content.return_value = [png()]
        context = Mock(__enter__=Mock(return_value=response), __exit__=Mock())
        with patch("ezlocalai.IMG.http_requests.get", return_value=context):
            with self.img._load_image("https://example.test/input.png") as image:
                self.assertEqual(image.size, (64, 64))
        context.__exit__.assert_called_once()

    def test_invalid_data_url_and_base64_are_rejected(self):
        for source in (
            "data:text/plain;base64," + self.ref,
            "data:image/png," + self.ref,
            self.ref + "!",
        ):
            self.assertIsNone(self.img._load_image(source))

    def test_concurrent_generations_serialize_cli(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading

        guard = threading.Lock()
        active = 0
        peak = 0

        def run(cmd, **kwargs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            result = self.run_cli(cmd, **kwargs)
            with guard:
                active -= 1
            return result

        with patch("ezlocalai.IMG.subprocess.run", side_effect=run):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(self.img.generate, "draw", size="64x64")
                    for _ in range(2)
                ]
                self.assertTrue(all(future.result() for future in futures))
        self.assertEqual(peak, 1)

    def test_initialization_downloads_missing_mmproj(self):
        files = [IMG.DIFFUSION_MODEL_FILE, IMG.VAE_FILE, IMG.LLM_FILE]
        for filename in files:
            path = Path(self.temp.name, filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with patch.dict(
            os.environ,
            {"SDCPP_BIN": self.img._llm_path, "SDCPP_MODELS_DIR": self.temp.name},
        ):
            with patch.object(IMG, "_download_models") as download:
                IMG()
                download.assert_called_once()

    def test_precache_downloads_exact_runtime_paths(self):
        namespace = dict(
            is_image_enabled=lambda: True,
            has_image_server_url=lambda: False,
            is_text_server_mode=lambda: False,
            logging=logging,
            os=os,
            time=time,
            getenv=lambda key, default=None: {
                "IMG_MODEL": "qwen-image-2.1",
                "SDCPP_MODELS_DIR": self.temp.name,
            }.get(key, default),
            download_with_progress=Mock(),
        )
        cache = extract("precache.py", "precache_image_model", namespace)
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": types.SimpleNamespace(hf_hub_download=Mock())},
        ):
            cache()
        calls = namespace["download_with_progress"].call_args_list
        self.assertEqual(len(calls), 4)
        self.assertIn((IMG.VAE_REPO,), [call.args for call in calls])
        self.assertIn(IMG.VAE_FILE, [call.kwargs["filename"] for call in calls])
        self.assertTrue(
            all(call.kwargs["local_dir"] == self.temp.name for call in calls)
        )


class EditContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        namespace = dict(
            BaseModel=BaseModel,
            Field=Field,
            model_validator=model_validator,
            List=List,
            Optional=Optional,
            Literal=Literal,
        )
        self.model = extract("app.py", "ImageEdit", namespace)

    def test_validation_rejects_empty_or_invalid_edits(self):
        for args in (
            {},
            {"images": []},
            {"images": [""]},
            {"image": "x", "strength": 2},
            {"image": "x", "n": 0},
            {"image": "x", "size": "bad"},
            {"images": ["x"] * 11},
        ):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                self.model(prompt="edit", **args)
        self.assertEqual(self.model(prompt="edit", images=["a", "b"]).strength, 0.75)

    async def test_all_forwarding_routes_preserve_references_and_strength(self):
        for route in ("image", "router", "fallback", "local"):
            with self.subTest(route=route):
                from fastapi import HTTPException, Depends

                clients = {
                    name: Mock(
                        is_configured=(route == name),
                        check_availability=AsyncMock(return_value=(True, "")),
                        forward_image_generation=AsyncMock(return_value={"data": []}),
                    )
                    for name in ("image", "router", "fallback")
                }
                pipe = Mock(generate_image=AsyncMock(return_value="png"))
                namespace = dict(
                    ImageEdit=self.model,
                    Depends=Depends,
                    verify_api_key=lambda: None,
                    logging=logging,
                    time=time,
                    HTTPException=HTTPException,
                    pipe=pipe,
                    _router_client_for_unserved_capability=AsyncMock(
                        return_value=clients["router"] if route == "router" else None
                    ),
                    _has_local_capability=lambda _: route == "local",
                )
                endpoint = extract("app.py", "edit_image", namespace)
                module = types.SimpleNamespace(
                    get_image_server_client=lambda: clients["image"],
                    get_fallback_client=lambda: clients["fallback"],
                )
                with patch.dict("sys.modules", {"Pipes": module}):
                    await endpoint(
                        self.model(prompt="edit", images=["a", "b"], strength=0.3)
                    )
                call = (
                    pipe.generate_image.call_args
                    if route == "local"
                    else clients[route].forward_image_generation.call_args
                )
                self.assertEqual(call.kwargs["images"], ["a", "b"])
                self.assertEqual(call.kwargs["strength"], 0.3)

    async def test_forwarding_clients_select_edits_for_images_only(self):
        tree = ast.parse(Path("Pipes.py").read_text())
        methods = [
            method
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            for method in node.body
            if getattr(method, "name", None) == "forward_image_generation"
        ]
        for method in methods:
            namespace = dict(Optional=Optional, logging=logging)
            exec(
                compile(ast.Module(body=[method], type_ignores=[]), "Pipes.py", "exec"),
                namespace,
            )
            response = Mock(status=200, json=AsyncMock(return_value={"data": []}))
            request = Mock(
                __aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock()
            )
            session = Mock(post=Mock(return_value=request))
            context = Mock(
                __aenter__=AsyncMock(return_value=session), __aexit__=AsyncMock()
            )
            client = Mock(
                is_configured=True,
                base_url="http://worker",
                _get_headers=Mock(return_value={}),
            )
            with patch("aiohttp.ClientSession", return_value=context):
                await namespace["forward_image_generation"](
                    client, "edit", images=["a", "b"], strength=0.3
                )
            self.assertEqual(
                session.post.call_args.args[0], "http://worker/v1/images/edits"
            )
            self.assertEqual(
                session.post.call_args.kwargs["json"]["images"], ["a", "b"]
            )
            self.assertEqual(session.post.call_args.kwargs["json"]["strength"], 0.3)


if __name__ == "__main__":
    unittest.main()
