"""Exercise the notebook's setup failure handling and feature configuration."""

import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock


NOTEBOOK = json.loads(Path(__file__).with_name("ezlocalai-colab.ipynb").read_text())
SETUP = "".join(NOTEBOOK["cells"][2]["source"])
START = "".join(NOTEBOOK["cells"][3]["source"])
GPU_DEFAULTS = {
    "gpu_name": "Tesla T4",
    "gpu_family": "T4",
    "vram_free_gib": 14.5,
    "vram_total_gib": 15,
    "settings": {"LLM_MAX_TOKENS": "8192", "LLM_PROMPT_CACHE_MIB": "0"},
}


class ColabNotebookTests(unittest.TestCase):
    def execute_setup(self, namespace, run):
        with (
            mock.patch("subprocess.run", run),
            mock.patch("subprocess.check_output", return_value="/venv/site-packages"),
            mock.patch("os.chdir"),
            mock.patch("os.environ", {}),
            mock.patch("pathlib.Path.exists", return_value=False),
        ):
            exec(compile(SETUP, "colab-setup", "exec"), namespace)

    def test_setup_checks_commands_and_uses_one_interpreter(self):
        run = mock.Mock()
        namespace = {}
        self.execute_setup(namespace, run)
        self.assertTrue(namespace["SETUP_COMPLETE"])
        for call in run.call_args_list:
            self.assertTrue(call.kwargs["check"])
            command = call.args[0]
            if command[1:3] in (["pip", "install"], ["pip", "uninstall"]):
                self.assertEqual(
                    command[command.index("--python") + 1], namespace["PYTHON"]
                )
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(
            any("3.12" in cmd and "--managed-python" in cmd for cmd in commands)
        )
        self.assertFalse(
            any(
                "git+https://github.com/huggingface/transformers" in cmd
                for cmd in commands
            )
        )
        self.assertEqual(commands[-1][0], namespace["PYTHON"])
        self.assertIn("import dotenv, aiohttp", commands[-1][-1])

    def test_failed_dependency_install_invalidates_previous_setup(self):
        def fail_requirements(command, **kwargs):
            if "cuda-requirements.txt" in command:
                raise subprocess.CalledProcessError(1, command)

        run = mock.Mock(side_effect=fail_requirements)
        namespace = {"SETUP_COMPLETE": True}
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute_setup(namespace, run)
        self.assertFalse(namespace["SETUP_COMPLETE"])
        self.assertIn("cuda-requirements.txt", run.call_args.args[0])

    def test_failed_import_check_does_not_mark_setup_complete(self):
        def fail_imports(command, **kwargs):
            if "-c" in command:
                raise subprocess.CalledProcessError(1, command)

        namespace = {}
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute_setup(namespace, mock.Mock(side_effect=fail_imports))
        self.assertFalse(namespace["SETUP_COMPLETE"])

    def test_start_requires_successful_setup(self):
        for namespace in ({}, {"SETUP_COMPLETE": False}):
            with self.subTest(namespace=namespace), mock.patch("subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "installation cell"):
                    exec(compile(START, "colab-start", "exec"), namespace)
                run.assert_not_called()

    def execute_start(self, source=START, **overrides):
        namespace = {
            "SETUP_COMPLETE": True,
            "verify_environment": mock.Mock(),
            "router_url": "https://router.example",
            "router_api_key": "test-key",
            "worker_name": "colab-worker",
            "ngrok_token": "",
            **overrides,
        }
        with (
            mock.patch("builtins.open", mock.mock_open()) as output,
            mock.patch("subprocess.run") as run,
            mock.patch(
                "subprocess.check_output", return_value=json.dumps(GPU_DEFAULTS)
            ) as probe,
            mock.patch("os.chdir"),
            mock.patch("os.environ", {}),
        ):
            exec(compile(source, "colab-start", "exec"), namespace)
            self.assertEqual(
                probe.call_args.args[0][0], "/content/ezlocalai-venv/bin/python"
            )
            self.assertEqual(probe.call_args.kwargs["cwd"], "/content/ezlocalai")
            namespace["verify_environment"].assert_called_once()
            run.assert_called_once_with(
                ["/content/ezlocalai-venv/bin/python", "start.py"], check=True
            )
            return dict(
                line.split("=", 1)
                for line in output().write.call_args.args[0].splitlines()
            )

    def test_default_is_requested_text_vision_model_only(self):
        config = self.execute_start()
        self.assertEqual(config["DEFAULT_MODEL"], "unsloth/Qwen3.8-27B-GGUF")
        self.assertEqual(config["QUANT_TYPE"], "Q3_K_XL")
        self.assertEqual(config["LLM_MAX_TOKENS"], "8192")
        self.assertEqual(config["LLM_PROMPT_CACHE_MIB"], "0")
        for name in ("TTS", "STT", "EMBEDDING", "IMAGE", "MUSIC", "VIDEO"):
            self.assertEqual(config[f"{name}_ENABLED"], "false")
        self.assertEqual(config["ROUTER_URL"], "https://router.example")
        self.assertEqual(config["WORKER_TUNNEL"], "true")

    def test_existing_feature_options(self):
        enabled = {
            1: {"TTS", "STT", "EMBEDDING"},
            2: {"IMAGE"},
            3: {"MUSIC"},
            4: {"VIDEO"},
            5: {"TTS", "STT", "EMBEDDING", "IMAGE", "MUSIC"},
        }
        for option, expected in enabled.items():
            with self.subTest(option=option):
                config = self.execute_start(
                    START.replace("features = 6", f"features = {option}")
                )
                self.assertEqual(
                    {
                        k.removesuffix("_ENABLED")
                        for k, v in config.items()
                        if k.endswith("_ENABLED") and v == "true"
                    },
                    expected,
                )
                self.assertEqual(config["DEFAULT_MODEL"] == "", option == 5)

    def test_ngrok_configuration(self):
        config = self.execute_start(router_url="", ngrok_token="test-ngrok-token")
        self.assertEqual(config["NGROK_TOKEN"], "test-ngrok-token")
        self.assertNotIn("ROUTER_URL", config)

    def test_explicit_inference_settings_override_detected_profile(self):
        config = self.execute_start(
            START.replace(
                "inference_overrides = {}",
                'inference_overrides = {"LLM_MAX_TOKENS": "16384", "MTP_SPEC_DRAFT_N_MAX_T4": "3"}',
            )
        )
        self.assertEqual(config["LLM_MAX_TOKENS"], "16384")
        self.assertEqual(config["MTP_SPEC_DRAFT_N_MAX_T4"], "3")
        self.assertEqual(config["LLM_PROMPT_CACHE_MIB"], "0")


if __name__ == "__main__":
    unittest.main()
