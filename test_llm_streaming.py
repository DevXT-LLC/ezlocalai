import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0


sys.modules.setdefault("xllamacpp", types.SimpleNamespace())
sys.modules.setdefault(
    "huggingface_hub",
    types.SimpleNamespace(
        hf_hub_download=lambda *args, **kwargs: None,
        list_repo_files=lambda *args, **kwargs: [],
    ),
)
sys.modules.setdefault("torch", types.SimpleNamespace(cuda=_FakeCuda()))

from ezlocalai.LLM import (
    LLM,
    calculate_auto_batch_sizes,
    get_mtp_spec_draft_n_max,
    get_mtp_spec_draft_p_min,
    get_model_image_min_tokens,
    is_mtp_model,
    normalize_stream_chunk_delta,
    resolve_prompt_cache_mib,
    stream_chunk_finish_reason,
    stream_chunk_has_assistant_text,
)


class _FakeStreamingServer:
    def __init__(self, chunks):
        self.chunks = chunks

    def handle_chat_completions(self, request, callback):
        for chunk in self.chunks:
            if callback(chunk):
                break
        return {}


class _FakeChatServer:
    def __init__(self):
        self.request = None

    def handle_chat_completions(self, request):
        self.request = request
        return {"choices": [{"message": {"content": "ok"}}]}


def _fake_llm(chunks):
    llm = LLM.__new__(LLM)
    llm.server = _FakeStreamingServer(chunks)
    llm.model_name = "test-model"
    return llm


class LlmStreamingTests(unittest.TestCase):
    def test_chat_explicitly_enables_prompt_cache_reuse(self):
        llm = LLM.__new__(LLM)
        llm.server = _FakeChatServer()
        llm.model_name = "test-model"
        llm.system_message = ""
        llm.params = {
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": [],
        }

        llm.chat([{"role": "user", "content": "hello"}])

        self.assertIs(llm.server.request["cache_prompt"], True)

    def test_qwen38_auto_ubatch_uses_benchmarked_hardware_caps(self):
        ampere_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            mem_get_info=lambda _index: (22 * 1024**3, 24 * 1024**3),
            get_device_capability=lambda _index: (8, 6),
        )
        blackwell_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            mem_get_info=lambda _index: (30 * 1024**3, 32 * 1024**3),
            get_device_capability=lambda _index: (12, 0),
        )
        with mock.patch("ezlocalai.LLM.torch.cuda", ampere_cuda):
            normal = calculate_auto_batch_sizes(0, 262_144, "Qwen3.6-27B")
            qwen38_180k = calculate_auto_batch_sizes(
                0, 180_000, "unsloth/Qwen3.8-27B-GGUF"
            )
            qwen38_262k = calculate_auto_batch_sizes(
                0, 262_144, "unsloth/Qwen3.8-27B-GGUF"
            )
            other_mtp = calculate_auto_batch_sizes(
                0, 262_144, "unsloth/Qwen3.6-27B-MTP-GGUF"
            )
        with mock.patch("ezlocalai.LLM.torch.cuda", blackwell_cuda):
            qwen38_5090 = calculate_auto_batch_sizes(
                0, 262_144, "unsloth/Qwen3.8-27B-GGUF"
            )

        self.assertEqual(normal[:2], (4096, 1024))
        self.assertEqual(qwen38_180k[:2], (4096, 1024))
        self.assertEqual(qwen38_262k[:2], (4096, 512))
        self.assertEqual(qwen38_5090[:2], (8192, 1024))
        self.assertEqual(other_mtp[:2], (4096, 256))
        self.assertIn("MTP long-context cap", qwen38_262k[2])
        self.assertIn("MTP long-context cap", qwen38_5090[2])

    def test_qwen38_uses_benchmarked_mtp_defaults(self):
        with (
            mock.patch("ezlocalai.LLM.get_total_vram_per_gpu", return_value=[24.0]),
            mock.patch("ezlocalai.LLM.getenv", return_value="auto"),
        ):
            self.assertEqual(
                get_mtp_spec_draft_n_max(0, "unsloth/Qwen3.8-27B-GGUF")[0], 3
            )
            self.assertEqual(
                get_mtp_spec_draft_n_max(0, "unsloth/Qwen3.6-27B-MTP-GGUF")[0],
                2,
            )
            self.assertEqual(get_mtp_spec_draft_p_min("unsloth/Qwen3.8-27B-GGUF"), 0.1)
            self.assertEqual(
                get_mtp_spec_draft_p_min("unsloth/Qwen3.6-27B-MTP-GGUF"), 0.25
            )

    def test_explicit_mtp_probability_still_overrides_family_default(self):
        with mock.patch("ezlocalai.LLM.getenv", return_value="0.4"):
            self.assertEqual(get_mtp_spec_draft_p_min("unsloth/Qwen3.8-27B-GGUF"), 0.4)

    def test_qwen38_standard_repo_is_recognized_as_built_in_mtp(self):
        self.assertTrue(is_mtp_model("unsloth/Qwen3.8-27B-GGUF"))
        self.assertTrue(is_mtp_model("models/Qwen3.8-27B-Q4_K_M.gguf"))
        self.assertTrue(is_mtp_model("unsloth/Qwen3.6-27B-MTP-GGUF"))
        self.assertFalse(is_mtp_model("unsloth/Qwen3.6-27B-GGUF"))

    def test_qwen38_reserves_enough_image_tokens_for_grounding(self):
        self.assertEqual(get_model_image_min_tokens("unsloth/Qwen3.8-27B-GGUF"), 1024)
        self.assertEqual(get_model_image_min_tokens("unsloth/Qwen3.6-27B-GGUF"), -1)

    def test_prompt_cache_auto_scales_with_context(self):
        self.assertEqual(resolve_prompt_cache_mib("auto", "qwen", 65_536)[0], 8192)
        self.assertEqual(resolve_prompt_cache_mib("auto", "qwen", 262_144)[0], 16384)
        self.assertEqual(resolve_prompt_cache_mib("auto", "qwen", 500_000)[0], 32768)

    def test_prompt_cache_can_be_disabled_or_explicit(self):
        self.assertEqual(resolve_prompt_cache_mib("off", "qwen", 262_144)[0], 0)
        self.assertEqual(resolve_prompt_cache_mib("0", "qwen", 262_144)[0], 0)
        self.assertEqual(resolve_prompt_cache_mib("12288", "qwen", 262_144)[0], 12288)

    def test_reasoning_delta_is_preserved_when_wrapping_chunk(self):
        delta = normalize_stream_chunk_delta(
            {"delta": {"reasoning_content": "I need to inspect this."}}
        )

        self.assertEqual(delta, {"reasoning_content": "I need to inspect this."})

    def test_content_delta_still_wins_for_answer_tokens(self):
        delta = normalize_stream_chunk_delta(
            {
                "delta": {
                    "content": "Done.",
                    "reasoning_content": "hidden planning",
                }
            }
        )

        self.assertEqual(delta, {"content": "Done."})

    def test_top_level_text_token_and_response_chunks_are_preserved(self):
        for chunk, expected in (
            ({"text": "Hello"}, "Hello"),
            ({"token": " world"}, " world"),
            ({"response": "!"}, "!"),
            ({"delta": {"text": " From delta."}}, " From delta."),
        ):
            with self.subTest(chunk=chunk):
                self.assertEqual(
                    normalize_stream_chunk_delta(chunk), {"content": expected}
                )
                self.assertTrue(stream_chunk_has_assistant_text(chunk))

    def test_structured_content_parts_are_preserved(self):
        delta = normalize_stream_chunk_delta(
            {
                "delta": {
                    "content": [
                        {"type": "text", "text": "Structured "},
                        {"type": "output_text", "text": "content"},
                    ]
                }
            }
        )

        self.assertEqual(delta, {"content": "Structured content"})

    def test_empty_keepalive_chunk_is_not_assistant_text(self):
        self.assertFalse(
            stream_chunk_has_assistant_text(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
            )
        )

    def test_choices_text_chunk_is_assistant_text(self):
        self.assertTrue(
            stream_chunk_has_assistant_text(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"text": "Visible text"},
                            "finish_reason": None,
                        }
                    ]
                }
            )
        )

    def test_finish_reason_is_detected_without_assistant_text(self):
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

        self.assertEqual(stream_chunk_finish_reason(chunk), "stop")
        self.assertFalse(stream_chunk_has_assistant_text(chunk))

    def test_stream_with_only_empty_final_chunk_returns_explicit_error(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        outputs = list(_fake_llm(chunks)._chat_stream({"messages": []}))

        self.assertEqual(outputs[0], chunks[0])
        self.assertEqual(outputs[-1]["error"]["type"], "empty_stream")

    def test_nested_stream_error_raises_real_context_message(self):
        chunks = [
            {
                "error": {
                    "code": 400,
                    "message": (
                        "request (262146 tokens) exceeds the available context size "
                        "(262144 tokens), try increasing it"
                    ),
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": 262146,
                    "n_ctx": 262144,
                }
            }
        ]

        with self.assertRaisesRegex(Exception, "262146.*262144"):
            list(_fake_llm(chunks)._chat_stream({"messages": []}))

    def test_stream_with_text_yields_deferred_final_once(self):
        final_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"content": "hello"}}]},
            final_chunk,
        ]

        outputs = list(_fake_llm(chunks)._chat_stream({"messages": []}))

        self.assertEqual(outputs[0], chunks[0])
        self.assertEqual(outputs[1], chunks[1])
        self.assertEqual(outputs[2], final_chunk)
        self.assertEqual(len(outputs), 3)


if __name__ == "__main__":
    unittest.main()
