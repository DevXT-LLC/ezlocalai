"""Shared model-specific inference settings.

Keep request configuration that must behave identically on local workers and
managed router providers in this lightweight module.  It intentionally has no
runtime/model dependencies so router mode can import it safely.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


QWEN38_MODEL = "unsloth/Qwen3.8-27B-GGUF"
QWEN38_THINKING_SETTINGS: Dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    },
}
QWEN38_INSTRUCT_SETTINGS: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


def copy_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a flat settings mapping, including its nested option mappings."""
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in settings.items()
    }


def is_qwen38_model(model: Optional[str]) -> bool:
    """Return whether a local or hosted identifier names Qwen3.8-27B."""
    if not model:
        return False
    name = str(model).strip().rsplit("/", 1)[-1]
    while True:
        normalized = re.sub(r"(?i)(?:-gguf|\.gguf|-tee|-mtp)$", "", name)
        if normalized == name:
            break
        name = normalized
    return bool(re.fullmatch(r"(?i)qwen3\.8-27b", name))


def apply_qwen38_model_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ezlocalai's effective Qwen3.8 thinking or instruct profile.

    ``chat_template_kwargs`` remains the local/vLLM representation.  The
    standard ``reasoning`` object is also understood so callers can use the
    same request with local workers and OpenAI-compatible managed providers.
    Provider-specific serialization happens later in router mode.
    """
    configured = dict(data)
    template = dict(QWEN38_THINKING_SETTINGS["chat_template_kwargs"])
    user_template = configured.get("chat_template_kwargs")
    if isinstance(user_template, dict):
        template.update(user_template)

    top_level_effort = configured.pop("reasoning_effort", None)
    if top_level_effort is not None:
        template["reasoning_effort"] = top_level_effort

    reasoning = configured.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        enabled = reasoning.get("enabled")
        if str(effort or "").lower() == "none":
            enabled = False
        elif effort is not None and enabled is None:
            enabled = True
        if enabled is not None:
            if isinstance(enabled, str):
                template["enable_thinking"] = enabled.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                template["enable_thinking"] = bool(enabled)
        if effort is not None and str(effort).lower() != "none":
            template["reasoning_effort"] = effort

    thinking_enabled = template.get("enable_thinking") is not False
    profile = (
        QWEN38_THINKING_SETTINGS
        if thinking_enabled
        else {**QWEN38_THINKING_SETTINGS, **QWEN38_INSTRUCT_SETTINGS}
    )
    for key, value in profile.items():
        if key != "chat_template_kwargs":
            configured[key] = value

    if not thinking_enabled:
        template.pop("reasoning_effort", None)
    configured["chat_template_kwargs"] = template
    return configured
