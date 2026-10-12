import unittest

from ezlocalai.context_retry import (
    classify_inference_capacity_error,
    context_reload_can_help,
    parse_context_error_limits,
)


class ContextRetryTests(unittest.TestCase):
    def test_classify_context_capacity_error(self):
        error = (
            "request (264165 tokens) exceeds the available context size "
            "(262144 tokens) [n_prompt_tokens=264165, n_ctx=262144]"
        )

        classified = classify_inference_capacity_error(error)

        self.assertEqual(classified["status_code"], 413)
        self.assertEqual(classified["detail"]["type"], "context_capacity_error")
        self.assertEqual(classified["detail"]["n_prompt_tokens"], 264165)
        self.assertEqual(classified["detail"]["n_ctx"], 262144)
        self.assertTrue(classified["detail"]["compact_and_retry"])

    def test_classify_gpu_memory_capacity_error(self):
        classified = classify_inference_capacity_error(
            "CUDA error: out of memory while cudaMalloc attempted 248 MiB"
        )

        self.assertEqual(classified["status_code"], 503)
        self.assertEqual(classified["detail"]["type"], "inference_capacity_error")
        self.assertTrue(classified["detail"]["compact_and_retry"])

    def test_generic_model_initialization_error_is_not_retryable_capacity(self):
        self.assertIsNone(
            classify_inference_capacity_error("failed to init server: invalid model")
        )

    def test_parse_context_error_limits_from_llama_cpp_error(self):
        error = (
            "request (264165 tokens) exceeds the available context size "
            "(262144 tokens), try increasing it [n_prompt_tokens=264165, "
            "n_ctx=262144]"
        )

        self.assertEqual(parse_context_error_limits(error), (264165, 262144))

    def test_skip_reload_when_backend_already_capped_slot(self):
        error = (
            "request (264165 tokens) exceeds the available context size "
            "(262144 tokens), try increasing it [n_prompt_tokens=264165, "
            "n_ctx=262144]"
        )

        self.assertFalse(
            context_reload_can_help(
                error_msg=error,
                current_context=300032,
                requested_context=272357,
            )
        )

    def test_allow_reload_when_model_was_loaded_at_smaller_context(self):
        error = (
            "request (21922 tokens) exceeds the available context size "
            "(16384 tokens), try increasing it [n_prompt_tokens=21922, "
            "n_ctx=16384]"
        )

        self.assertTrue(
            context_reload_can_help(
                error_msg=error,
                current_context=16384,
                requested_context=30114,
            )
        )


if __name__ == "__main__":
    unittest.main()
