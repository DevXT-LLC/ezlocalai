import base64
import json
import struct
import unittest
import zlib

from bench_capacity import (prepare_capacity, qualify, red_png, request_body,
                            validate_completion, validate_properties)


def props(context=230144):
    return {"default_generation_settings": {"n_ctx": context, "params": {}},
            "total_slots": 1, "modalities": {"vision": True}, "is_sleeping": False}


METADATA = {"original_parameters": {"-c": "230000", "-ub": "512"}, "overrides": {"-ub": 1024}}


def completion(count, word="Python", cached=0):
    return {"choices": [{"message": {"content": word}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": count},
            "timings": {"prompt_n": count, "cache_n": cached, "prompt_ms": 20,
                        "prompt_per_second": count * 50, "predicted_n": 2,
                        "predicted_ms": 1, "predicted_per_second": 1000}}


class CapacityGuardsTests(unittest.TestCase):
    def test_declared_context_is_retained_with_only_upward_padding(self):
        result = validate_properties(props(), 230000, 1024, METADATA)
        self.assertEqual(result["effective_context"], 230144)
        self.assertEqual(result["expected_n_ubatch"], 1024)
        for value in [props(229888), props(262144), {**props(), "total_slots": 2},
                      {**props(), "modalities": {"vision": False}}]:
            with self.assertRaises(RuntimeError):
                validate_properties(value, 230000, 1024, METADATA)
        with self.assertRaisesRegex(RuntimeError, "launch metadata"):
            validate_properties(props(), 230000, 2048, METADATA)

    def test_tokenization_prepares_near_95_percent_without_context_reduction(self):
        def count(messages):
            return len(messages[-1]["content"]) // 4 + 30

        _, result = prepare_capacity("def example(): pass\n", 230000, count, "run")
        self.assertGreaterEqual(result["input_fraction"], .949)
        self.assertLessEqual(result["input_fraction"], .95)
        self.assertEqual(result["declared_context"], 230000)
        self.assertGreater(result["remaining_context_tokens"], 10000)
        self.assertLess(result["tokenizer_passes"], 35)
        with self.assertRaises(ValueError):
            prepare_capacity("x", 512, count, "run", .9999)

    def test_caching_truncation_wrong_answers_and_length_stops_fail(self):
        self.assertEqual(validate_completion(completion(100), 100, "Python", 1000)["answer"], "Python")
        bad = [completion(100, cached=1), completion(99), completion(100, "Java"),
               {**completion(100), "truncated": True},
               {**completion(100), "choices": [{"message": {"content": "Python"}, "finish_reason": "length"}]}]
        for value in bad:
            with self.assertRaises(RuntimeError):
                validate_completion(value, 100, "Python", 1000)
        with self.assertRaises(RuntimeError):
            validate_completion(completion(100), 100, "Python", 107)

    def test_png_contains_only_red_pixels_with_valid_chunk_checksums(self):
        png = red_png(2, 3)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        offset, compressed = 8, b""
        while offset < len(png):
            count = struct.unpack_from(">I", png, offset)[0]
            kind = png[offset + 4:offset + 8]
            value = png[offset + 8:offset + 8 + count]
            crc = struct.unpack_from(">I", png, offset + 8 + count)[0]
            self.assertEqual(zlib.crc32(kind + value) & 0xffffffff, crc)
            if kind == b"IDAT":
                compressed += value
            offset += count + 12
        self.assertEqual(zlib.decompress(compressed), (b"\0" + b"\xff\0\0" * 2) * 3)

    def test_http_qualification_evaluates_text_and_image_with_cache_disabled(self):
        calls, saved = [], {}

        def client(path, body=None):
            calls.append((path, body))
            if path == "/props":
                return props(4096)
            if path == "/apply-template":
                return {"prompt": body["messages"][-1]["content"]}
            if path == "/tokenize":
                return {"tokens": [1] * (len(body["content"]) // 4 + 30)}
            content = body["messages"][-1]["content"]
            count = len(content) // 4 + 30 if isinstance(content, str) else 300
            if path.endswith("input_tokens"):
                return {"input_tokens": count}
            if path == "/v1/chat/completions":
                self.assertFalse(body["cache_prompt"])
                self.assertEqual(body["temperature"], 0)
                if not isinstance(content, str):
                    url = content[1]["image_url"]["url"]
                    self.assertEqual(base64.b64decode(url.split(",", 1)[1]), red_png())
                return completion(count, "Python" if isinstance(content, str) else "red")
            raise AssertionError(path)

        qualify(client, "def f(): pass\n", 4096, 1024,
                {"n_ctx": 4096, "n_ubatch": 1024}, "test", lambda k, v: saved.update({k: v}))
        self.assertEqual(saved["capacity"]["answer"], "Python")
        self.assertEqual(saved["vision"]["answer"], "red")
        self.assertEqual(len([path for path, _ in calls if path == "/v1/chat/completions"]), 2)


if __name__ == "__main__":
    unittest.main()
