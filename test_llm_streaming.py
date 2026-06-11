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

from ezlocalai.LLM import normalize_stream_chunk_delta


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


if __name__ == "__main__":
    unittest.main()
