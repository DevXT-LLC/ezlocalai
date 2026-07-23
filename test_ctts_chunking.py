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

        chunks = split_text_into_stream_chunks(text, target_chars=50)

        self.assertEqual(
            chunks,
            [
                "Hello. Привет, как дела? Повтори слово спасибо.",
                "Now say it slowly.",
            ],
        )
        self.assertEqual(" ".join(chunks), text)

    def test_stream_chunks_pair_short_sentences_for_natural_playback(self):
        text = clean_text_for_tts(
            "Hello. This is the second sentence. "
            "Here is Russian: Привет, как дела? Now this is the final sentence."
        )

        chunks = split_text_into_stream_chunks(text, target_chars=160)

        self.assertEqual(
            chunks,
            [
                "Hello. This is the second sentence. Here is Russian: Привет, как дела?",
                "Now this is the final sentence.",
            ],
        )
        self.assertEqual(" ".join(chunks), text)

    def test_stream_chunks_start_with_short_opening_unit_for_runway(self):
        text = clean_text_for_tts(
            "WorkConductor routed TTS probe. "
            "Here is Russian: Привет, как дела? Probe complete."
        )

        chunks = split_text_into_stream_chunks(text, target_chars=160)

        self.assertEqual(
            chunks,
            [
                "WorkConductor routed TTS probe. Here is Russian: Привет, как дела?",
                "Probe complete.",
            ],
        )
        self.assertEqual(" ".join(chunks), text)

    def test_stream_chunks_use_larger_followup_units_after_fast_start(self):
        text = clean_text_for_tts(
            "Start with a quick sentence. "
            "This follow up sentence has enough detail to be grouped with the next idea. "
            "The Russian practice line says: Привет, меня зовут Анна. "
            "Another sentence keeps the thought together for smoother audio. "
            "Final sentence closes it naturally."
        )

        chunks = split_text_into_stream_chunks(text, target_chars=280)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            chunks[0],
            "Start with a quick sentence. "
            "This follow up sentence has enough detail to be grouped with the next idea.",
        )
        self.assertIn("Привет, меня зовут Анна.", chunks[1])
        self.assertTrue(chunks[1].endswith("Final sentence closes it naturally."))
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
