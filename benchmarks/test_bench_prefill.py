import unittest
from unittest.mock import patch

from bench_prefill import measure, prefill_cases, require_cold


class PrefillBenchmarkTests(unittest.TestCase):
    def test_nonce_is_in_the_shared_prefix_for_every_continuation(self):
        cases = dict(prefill_cases("abc", 8, 2, "[nonce]\n"))
        self.assertEqual(cases["cold"][1], {
            "role": "user", "content": "[nonce]\nReference code:\nabcabcab\n"
            "What language is this? Reply in one word."})
        self.assertEqual(cases["repeat"], cases["cold"])
        for name in ("append", "extend"):
            self.assertEqual(cases[name][:2], cases["cold"])
        self.assertTrue(cases["extend"][-1]["content"].startswith("ab\n"))

    def test_legacy_prompt_layout_is_preserved_without_nonce(self):
        cases = dict(prefill_cases("abc", 3))
        self.assertEqual(cases["cold"], [
            {"role": "system", "content": "You review source code. Reply briefly."},
            {"role": "user", "content": "Reference code:\nabc\nWhat language is this? Reply in one word."}])

    def test_a_cached_extension_cannot_be_called_cold_prefill(self):
        require_cold({"timings": {"cache_n": 13}})
        with self.assertRaises(RuntimeError):
            require_cold({"timings": {"cache_n": 68073}})

    def test_http_success_without_inference_timings_is_rejected(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *args):
                return b'{"choices":[{"message":{"content":"Python"}}]}'

        with patch("bench_prefill.urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "timings"):
                measure("http://127.0.0.1:1", "", [])

    def test_invalid_inputs_are_rejected(self):
        for source, size, extension in [("", 3, 2), ("x", 0, 2), ("x", 3, -1)]:
            with self.assertRaises(ValueError):
                prefill_cases(source, size, extension)


if __name__ == "__main__":
    unittest.main()
