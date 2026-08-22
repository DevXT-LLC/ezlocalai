import sys
import types
import unittest
from unittest import mock


sys.modules.setdefault("xllamacpp", types.SimpleNamespace())

from Pipes import (
    MODEL_CONFIG_OVERRIDES,
    Pipes,
    _is_qwen35_hybrid,
    _is_retryable_gpu_load_error,
    _pop_disable_fallback,
    _reduced_ubatch_candidates,
)


QWEN38_MODEL = "unsloth/Qwen3.8-27B-GGUF"


class FallbackControlTests(unittest.TestCase):
    def test_disable_fallback_is_consumed_before_inference(self):
        data = {"disable_fallback": True, "messages": []}

        self.assertTrue(_pop_disable_fallback(data))
        self.assertNotIn("disable_fallback", data)

    def test_disable_fallback_accepts_true_string_only(self):
        self.assertTrue(_pop_disable_fallback({"disable_fallback": "true"}))
        self.assertFalse(_pop_disable_fallback({"disable_fallback": "false"}))


class Qwen38ModelConfigTests(unittest.TestCase):
    def _apply(self, data):
        pipe = Pipes.__new__(Pipes)
        pipe.current_llm_name = QWEN38_MODEL
        return pipe._apply_model_config_overrides(dict(data))

    def test_qwen38_defaults_to_recommended_thinking_settings(self):
        configured = self._apply({})

        self.assertEqual(
            {
                key: configured[key]
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "repetition_penalty",
                )
            },
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
        )
        self.assertEqual(
            configured["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
        )

    def test_qwen38_top_level_reasoning_effort_overrides_thinking_default(self):
        configured = self._apply({"reasoning_effort": "medium"})

        self.assertNotIn("reasoning_effort", configured)
        self.assertEqual(
            configured["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "medium"},
        )

    def test_qwen38_standard_reasoning_object_controls_local_thinking(self):
        configured = self._apply(
            {"reasoning": {"enabled": False, "effort": "low", "exclude": True}}
        )

        self.assertEqual(configured["temperature"], 0.7)
        self.assertEqual(configured["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(
            configured["reasoning"],
            {"enabled": False, "effort": "low", "exclude": True},
        )

    def test_qwen38_string_false_disables_thinking(self):
        configured = self._apply({"reasoning": {"enabled": "false"}})

        self.assertFalse(configured["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(configured["temperature"], 0.7)

    def test_qwen38_uses_recommended_instruct_settings_when_thinking_is_off(self):
        configured = self._apply(
            {
                "reasoning_effort": "low",
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                },
            }
        )

        self.assertEqual(configured["temperature"], 0.7)
        self.assertEqual(configured["top_p"], 0.8)
        self.assertEqual(configured["top_k"], 20)
        self.assertEqual(configured["min_p"], 0.0)
        self.assertEqual(configured["presence_penalty"], 1.5)
        self.assertEqual(configured["repetition_penalty"], 1.0)
        self.assertNotIn("reasoning_effort", configured)
        self.assertEqual(
            configured["chat_template_kwargs"],
            {"enable_thinking": False, "preserve_thinking": False},
        )

    def test_qwen38_is_treated_as_qwen35_hybrid_for_vram_planning(self):
        self.assertTrue(_is_qwen35_hybrid(QWEN38_MODEL))
        self.assertTrue(_is_qwen35_hybrid("Qwen3.8-27B-Q4_K_M.gguf"))

    def test_static_qwen38_config_is_not_mutated_by_request_merging(self):
        self._apply({"chat_template_kwargs": {"enable_thinking": False}})

        self.assertEqual(
            MODEL_CONFIG_OVERRIDES[QWEN38_MODEL]["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
        )


class LlmLoadFallbackTests(unittest.TestCase):
    @staticmethod
    def _pipe():
        pipe = Pipes.__new__(Pipes)
        pipe._resolve_source_model = lambda model_name: model_name
        return pipe

    def test_xllamacpp_opaque_init_failure_is_retryable(self):
        self.assertTrue(
            _is_retryable_gpu_load_error(
                RuntimeError("Failed to init server, please check the input params.")
            )
        )
        self.assertFalse(
            _is_retryable_gpu_load_error(ValueError("invalid chat template"))
        )

    def test_ubatch_candidates_preserve_layers_before_offloading(self):
        self.assertEqual(_reduced_ubatch_candidates(1024), [512, 256, 128])
        self.assertEqual(_reduced_ubatch_candidates(128), [])

    def test_opaque_init_failure_retries_full_gpu_with_smaller_ubatch(self):
        attempts = []
        loaded = object()

        def fake_llm(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise RuntimeError("Failed to init server, please check input params")
            return loaded

        with (
            mock.patch(
                "ezlocalai.LLM.download_model",
                return_value=("/tmp/model.gguf", None),
            ),
            mock.patch("Pipes.LLM", side_effect=fake_llm),
            mock.patch("Pipes._configured_llm_ubatch", return_value=1024),
            mock.patch("Pipes._cleanup_failed_gpu_load"),
        ):
            result = self._pipe()._load_llm_resilient(
                QWEN38_MODEL,
                262_144,
                main_gpu=0,
                n_parallel=1,
                quant_type="Q4_K_XL",
            )

        self.assertIs(result, loaded)
        self.assertIsNone(attempts[1]["gpu_layers"])
        self.assertEqual(attempts[1]["ubatch_size"], 512)

    def test_layer_fallback_finds_highest_count_that_fits(self):
        attempts = []
        probed_layers = []

        def fake_llm(**kwargs):
            attempts.append(kwargs)
            layers = kwargs.get("gpu_layers")
            if layers is None or layers < 0:
                raise RuntimeError("Failed to init server, please check input params")
            return types.SimpleNamespace(gpu_layers=layers)

        def fake_probe(**kwargs):
            layers = kwargs["gpu_layers"]
            probed_layers.append(layers)
            return layers <= 58

        with (
            mock.patch(
                "ezlocalai.LLM.download_model",
                return_value=("/tmp/model.gguf", None),
            ),
            mock.patch("Pipes.LLM", side_effect=fake_llm),
            mock.patch("Pipes._configured_llm_ubatch", return_value=256),
            mock.patch("Pipes._reduced_ubatch_candidates", return_value=[128]),
            mock.patch("Pipes._model_layer_count", return_value=66),
            mock.patch("Pipes._probe_llm_gpu_layers", side_effect=fake_probe),
            mock.patch("Pipes._cleanup_failed_gpu_load"),
        ):
            result = self._pipe()._load_llm_resilient(
                QWEN38_MODEL,
                262_144,
                main_gpu=0,
                n_parallel=1,
                quant_type="Q4_K_XL",
            )

        self.assertEqual(result.gpu_layers, 58)
        self.assertIn(58, probed_layers)
        self.assertIn(59, probed_layers)
        self.assertEqual(attempts[-1]["gpu_layers"], 58)

    def test_non_resource_configuration_error_is_not_retried(self):
        with (
            mock.patch(
                "ezlocalai.LLM.download_model",
                return_value=("/tmp/model.gguf", None),
            ),
            mock.patch(
                "Pipes.LLM", side_effect=ValueError("invalid rope setting")
            ) as llm,
        ):
            with self.assertRaisesRegex(ValueError, "invalid rope"):
                self._pipe()._load_llm_resilient(
                    QWEN38_MODEL,
                    262_144,
                    main_gpu=0,
                    n_parallel=1,
                )

        llm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
