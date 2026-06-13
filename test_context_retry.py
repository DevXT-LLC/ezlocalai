import unittest

from ezlocalai.context_retry import (
    context_reload_can_help,
    parse_context_error_limits,
)


class ContextRetryTests(unittest.TestCase):
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
