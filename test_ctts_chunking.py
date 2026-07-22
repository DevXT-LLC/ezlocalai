import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

qwen_tts_stub = types.ModuleType("qwen_tts")
qwen_tts_stub.Qwen3TTSModel = object
sys.modules.setdefault("qwen_tts", qwen_tts_stub)

from ezlocalai.CTTS import clean_text_for_tts, split_text_into_stream_chunks


class CTTSChunkingTests(unittest.TestCase):
    def test_stream_chunks_preserve_multilingual_sentences(self):
        text = clean_text_for_tts(
            "Hello. Привет, как дела? Повтори слово спасибо. Now say it slowly."
        )

        chunks = split_text_into_stream_chunks(text, target_chars=40)

        self.assertEqual(
            chunks,
            [
                "Hello.",
                "Привет, как дела?",
                "Повтори слово спасибо.",
                "Now say it slowly.",
            ],
        )
        self.assertEqual(" ".join(chunks), text)

    def test_stream_chunks_do_not_truncate_long_text(self):
        text = clean_text_for_tts(
            " ".join(
                [
                    "Это",
                    "длинный",
                    "пример",
                    "русского",
                    "текста",
                    "который",
                    "должен",
                    "разбиваться",
                    "на",
                    "несколько",
                    "частей",
                    "без",
                    "потери",
                    "слов.",
                ]
            )
        )

        chunks = split_text_into_stream_chunks(text, target_chars=25)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), text)


if __name__ == "__main__":
    unittest.main()
