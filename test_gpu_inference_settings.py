import os
import unittest
from types import SimpleNamespace
from unittest import mock

from ezlocalai.InferenceSettings import (
    gpu_profile,
    resolve_kv_cache_type,
    draft_length_setting,
    colab_inference_defaults,
)

MODEL = "unsloth/Qwen3.8-27B-GGUF"


class GpuInferenceSettingsTests(unittest.TestCase):
    def setUp(self):
        # Keep GPU probes independent of both local hardware and other tests'
        # lightweight torch stubs.
        cuda = SimpleNamespace(
            is_available=mock.Mock(return_value=False),
            device_count=mock.Mock(return_value=0),
            get_device_properties=mock.Mock(),
            mem_get_info=mock.Mock(),
        )
        patcher = mock.patch.dict(
            "sys.modules", {"torch": SimpleNamespace(cuda=cuda), "torch.cuda": cuda}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_cache_per_card_and_model(self):
        for family, capacity, expected in (
            ("3090", 24, "q4_0"),
            ("4090", 24, "q4_0"),
            ("5090", 32, "q4_0"),
            ("5090", 24, "q4_0"),
            ("", 80, "q4_0"),
            ("T4", 15, "q4_0"),
            ("A100", 40, "q4_0"),
            ("A100", 80, "q4_0"),
            ("H100", 80, "q4_0"),
        ):
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "ezlocalai.InferenceSettings.gpu_profile",
                return_value=(family, capacity),
            ):
                self.assertEqual(resolve_kv_cache_type(0, MODEL), expected)
                self.assertEqual(resolve_kv_cache_type(0, "Qwen3.5-4B"), "q4_0")

    def test_explicit_global_and_card_cache_overrides(self):
        with mock.patch(
            "ezlocalai.InferenceSettings.gpu_profile", return_value=("5090", 32)
        ):
            with mock.patch.dict(os.environ, {"KV_CACHE_TYPE": "q4_0"}, clear=True):
                self.assertEqual(resolve_kv_cache_type(0, MODEL), "q4_0")
            with mock.patch.dict(
                os.environ,
                {"KV_CACHE_TYPE": "q4_0", "KV_CACHE_TYPE_5090": "q8_0"},
                clear=True,
            ):
                self.assertEqual(resolve_kv_cache_type(0, MODEL), "q8_0")
            with mock.patch.dict(os.environ, {"KV_CACHE_TYPE": "typo"}, clear=True):
                with self.assertRaises(ValueError):
                    resolve_kv_cache_type(0, MODEL)

    def test_card_draft_length_precedence(self):
        with mock.patch(
            "ezlocalai.InferenceSettings.gpu_profile", return_value=("3090", 24)
        ):
            for env, expected in (
                ({}, 3),
                ({"DFLASH_SPEC_DRAFT_N_MAX": "5"}, 5),
                (
                    {
                        "DFLASH_SPEC_DRAFT_N_MAX": "5",
                        "DFLASH_SPEC_DRAFT_N_MAX_3090": "3",
                    },
                    3,
                ),
            ):
                with mock.patch.dict(os.environ, env, clear=True):
                    self.assertEqual(draft_length_setting(0), expected)

    def test_device_selection_and_conservative_unassigned_planning(self):
        devices = [
            SimpleNamespace(
                name="NVIDIA GeForce RTX 3090 Ti", total_memory=24 * 1024**3
            ),
            SimpleNamespace(name="NVIDIA GeForce RTX 5090", total_memory=32 * 1024**3),
        ]
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.device_count", return_value=2
        ), mock.patch(
            "torch.cuda.get_device_properties", side_effect=lambda index: devices[index]
        ):
            self.assertEqual(gpu_profile(0), ("3090", 24))
            self.assertEqual(gpu_profile(1), ("5090", 32))
            self.assertEqual(gpu_profile(None), ("5090", 32))

    def test_colab_gpu_names_and_visible_memory(self):
        for name, capacity, family in (
            ("Tesla T4", 15, "T4"),
            ("NVIDIA T4", 15, "T4"),
            ("NVIDIA A100-SXM4-40GB", 40, "A100"),
            ("NVIDIA A100 80GB PCIe", 80, "A100"),
            ("NVIDIA H100 80GB HBM3", 80, "H100"),
            ("NVIDIA H100 PCIe", 80, "H100"),
            ("NVIDIA A100-SXM4-40GB MIG 1g.5gb", 5, "A100"),
            ("NVIDIA H100 80GB HBM3 MIG 1g.10gb", 10, "H100"),
            ("NVIDIA RTX A1000", 8, ""),
        ):
            with (
                self.subTest(name=name),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(
                        name=name, total_memory=capacity * 1024**3
                    ),
                ),
            ):
                self.assertEqual(gpu_profile(0), (family, capacity))

    def test_colab_card_overrides(self):
        for family, default in (("T4", 2), ("A100", 4), ("H100", 4)):
            with mock.patch(
                "ezlocalai.InferenceSettings.gpu_profile", return_value=(family, 40)
            ):
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(draft_length_setting(), default)
                with mock.patch.dict(
                    os.environ,
                    {
                        "KV_CACHE_TYPE": "q4_0",
                        f"KV_CACHE_TYPE_{family}": "q8_0",
                        "DFLASH_SPEC_DRAFT_N_MAX": "5",
                        f"DFLASH_SPEC_DRAFT_N_MAX_{family}": "2",
                    },
                    clear=True,
                ):
                    self.assertEqual(resolve_kv_cache_type(), "q8_0")
                    self.assertEqual(draft_length_setting(), 2)

    def test_colab_context_and_host_cache_use_available_memory(self):
        for name, total, free, ram, context, cache in (
            ("Tesla T4", 15, 14, 12, 8192, 0),
            ("NVIDIA A100-SXM4-40GB", 40, 38, 48, 262144, 6144),
            ("NVIDIA A100 80GB PCIe", 80, 78, 96, 262144, 8192),
            ("NVIDIA H100 80GB HBM3", 80, 78, 96, 262144, 2560),
            ("NVIDIA A100-SXM4-40GB MIG 1g.5gb", 5, 4, 48, 8192, 0),
            ("NVIDIA H100 80GB HBM3", 80, 22, 2, 230000, 0),
            ("NVIDIA H100 80GB HBM3", 80, 38, 16, 262144, 2048),
            ("NVIDIA GeForce RTX 3090", 24, 22, 32, 230000, 4096),
        ):
            with (
                self.subTest(name=name, free=free, ram=ram),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(
                        name=name, total_memory=total * 1024**3
                    ),
                ),
                mock.patch(
                    "torch.cuda.mem_get_info",
                    return_value=(free * 1024**3, total * 1024**3),
                ),
                mock.patch(
                    "psutil.virtual_memory",
                    return_value=SimpleNamespace(available=ram * 1024**3),
                ),
            ):
                profile = colab_inference_defaults()
                self.assertEqual(profile["gpu_name"], name)
                self.assertEqual(profile["vram_free_gib"], free)
                settings = profile["settings"]
                self.assertEqual(settings["LLM_MAX_TOKENS"], str(context))
                self.assertEqual(settings["LLM_PROMPT_CACHE_MIB"], str(cache))
                self.assertLess(int(settings["LLM_MAX_OUTPUT_TOKENS"]), context)
                self.assertEqual(
                    settings["N_PARALLEL"],
                    "3" if "H100" in name and free == 78 else "1",
                )

    def test_h100_replica_budget_is_model_quant_and_free_memory_specific(self):
        import torch

        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_properties.return_value = SimpleNamespace(
            name="NVIDIA H100 80GB HBM3", total_memory=80 * 1024**3
        )
        with mock.patch(
            "psutil.virtual_memory",
            return_value=SimpleNamespace(available=96 * 1024**3),
        ):
            for free, model, quant, slots in (
                (78, MODEL, "Q3_K_XL", 3),
                (70, MODEL, "Q3_K_XL", 3),
                (69, MODEL, "Q3_K_XL", 2),
                (48, MODEL, "Q3_K_XL", 2),
                (47, MODEL, "Q3_K_XL", 1),
                (10, MODEL, "Q3_K_XL", 1),
                (78, "unsloth/Qwen3.5-4B-GGUF", "Q3_K_XL", 1),
                (78, MODEL, "Q8_0", 1),
            ):
                with self.subTest(free=free, model=model, quant=quant):
                    torch.cuda.mem_get_info.return_value = (
                        free * 1024**3,
                        80 * 1024**3,
                    )
                    settings = colab_inference_defaults(
                        model_name=model, quant_type=quant
                    )["settings"]
                    self.assertEqual(int(settings["N_PARALLEL"]), slots)
                    self.assertLessEqual(
                        int(settings["LLM_PROMPT_CACHE_MIB"]) * slots, 8192
                    )

    def test_colab_requires_a_gpu_runtime(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "GPU runtime"):
                colab_inference_defaults()


if __name__ == "__main__":
    unittest.main()
