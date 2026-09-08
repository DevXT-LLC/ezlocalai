import os
import unittest
from types import SimpleNamespace
from unittest import mock

from ezlocalai.InferenceSettings import (
    gpu_profile,
    resolve_kv_cache_type,
    draft_length_setting,
)

MODEL = "unsloth/Qwen3.8-27B-GGUF"


class GpuInferenceSettingsTests(unittest.TestCase):
    def test_default_cache_per_card_and_model(self):
        for family, capacity, expected in (
            ("3090", 24, "q4_0"),
            ("4090", 24, "q4_0"),
            ("5090", 32, "q8_0"),
            ("5090", 24, "q4_0"),
            ("", 80, "q4_0"),
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


if __name__ == "__main__":
    unittest.main()
