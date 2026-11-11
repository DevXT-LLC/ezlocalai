import json
from pathlib import Path
import tempfile
import unittest

from summarize_prefill_fleet import summarize


class SummaryTests(unittest.TestCase):
    def test_requires_final_restoration_and_excludes_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            rows = [
                {"event": "provenance", "image": "sha256:test"},
                {
                    "event": "sample",
                    "timings": {"prompt_n": 8000},
                    "quality_pass": True,
                    "answer": "do not publish raw output",
                },
            ]
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaises(ValueError):
                summarize(path)
            rows.append({"event": "restored", "original_container": True})
            path.write_text("\n".join(map(json.dumps, rows)))
            result = summarize(path)
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(result["quality_passes"], 1)
            self.assertNotIn("answer", result["samples"][0])
            self.assertNotIn("do not publish", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
