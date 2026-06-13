import pathlib
import sys
import types
import unittest


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
    normalize_stream_chunk_delta,
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


def _fake_llm(chunks):
    llm = LLM.__new__(LLM)
    llm.server = _FakeStreamingServer(chunks)
    llm.model_name = "test-model"
    return llm


class LlmStreamingTests(unittest.TestCase):
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
