import re


_MEMORY_CAPACITY_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "failed to allocate",
    "alloc_buffer",
    "cudamalloc",
    "insufficient memory",
)


def parse_context_error_limits(error_msg: str) -> tuple[int, int]:
    """Extract prompt token count and actual n_ctx from llama.cpp errors."""
    text = str(error_msg or "")
    prompt_match = re.search(r"n_prompt_tokens=(\d+)", text)
    if not prompt_match:
        prompt_match = re.search(r"request\s*\((\d+)\s+tokens?\)", text, re.I)
    ctx_match = re.search(r"n_ctx=(\d+)", text)
    if not ctx_match:
        ctx_match = re.search(
            r"available context size\s*\((\d+)\s+tokens?\)", text, re.I
        )

    prompt_tokens = int(prompt_match.group(1)) if prompt_match else 0
    actual_context = int(ctx_match.group(1)) if ctx_match else 0
    return prompt_tokens, actual_context


def classify_inference_capacity_error(error_msg: str) -> dict | None:
    """Return structured recovery metadata for actionable capacity failures."""
    message = str(error_msg or "Inference failed")
    prompt_tokens, context_tokens = parse_context_error_limits(message)
    if prompt_tokens > 0 and context_tokens > 0:
        return {
            "status_code": 413,
            "detail": {
                "type": "context_capacity_error",
                "message": message,
                "n_prompt_tokens": prompt_tokens,
                "n_ctx": context_tokens,
                "compact_and_retry": True,
            },
        }

    lowered = message.lower()
    if any(marker in lowered for marker in _MEMORY_CAPACITY_MARKERS):
        return {
            "status_code": 503,
            "detail": {
                "type": "inference_capacity_error",
                "message": message,
                "compact_and_retry": True,
            },
        }

    return None


def context_reload_can_help(
    error_msg: str,
    current_context: int,
    requested_context: int,
) -> bool:
    """Return False when llama.cpp already reported the actual slot cap.

    Some server paths cap the opened slot to the model training context even if
    a larger context was requested. In that case retrying with an even larger
    context just fails server initialization and hides the real context error.
    """
    needed_tokens, actual_context = parse_context_error_limits(error_msg)
    if actual_context <= 0 or needed_tokens <= 0:
        return True
    if needed_tokens <= actual_context:
        return True
    # If we requested more context than llama.cpp actually opened, the backend
    # has already capped the slot. Asking for even more will not help and can
    # fail server initialization. If current_context is equal to n_ctx, this may
    # simply be a small model loaded too narrowly, so let the normal reload path
    # try a larger slot.
    if current_context > actual_context:
        return False
    return requested_context > current_context
