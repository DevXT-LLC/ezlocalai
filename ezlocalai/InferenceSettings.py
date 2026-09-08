"""Shared GPU-aware inference settings for loading and memory planning."""

import os
import re


def gpu_profile(main_gpu=0):
    """Return a consumer-card family and capacity; unknown hardware stays generic.

    Before placement is known, prefer a 5090 profile if present so memory
    planning does not underestimate the cache later selected on that GPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "", 0.0
        indices = range(torch.cuda.device_count()) if main_gpu is None else [main_gpu]
        profiles = []
        for index in indices:
            info = torch.cuda.get_device_properties(index)
            match = re.search(r"\bRTX\s+(3090|4090|5090)\b", info.name.upper())
            profiles.append(
                (match.group(1) if match else "", info.total_memory / 1024**3)
            )
        return max(
            profiles, key=lambda item: (item[0] == "5090", item[1]), default=("", 0.0)
        )
    except Exception:
        return "", 0.0


def resolve_kv_cache_type(main_gpu=None, model_name=""):
    """Respect explicit overrides; use Q8 KV for the 27B on 32 GB RTX 5090s.

    This is a precision/headroom profile, not a promise of faster decoding.
    Context/model fit is still checked by the normal residency loader.
    """
    family, capacity = gpu_profile(main_gpu)
    raw = os.getenv(f"KV_CACHE_TYPE_{family}", "") if family else ""
    raw = (raw or os.getenv("KV_CACHE_TYPE", "auto")).strip().lower()
    if raw in ("", "auto"):
        qwen27 = re.search(r"qwen3\.8-27b(?:$|[-_./])", (model_name or "").lower())
        return "q8_0" if qwen27 and family == "5090" and capacity >= 30 else "q4_0"
    if raw not in {"q4_0", "q8_0", "f16", "f32"}:
        raise ValueError("KV_CACHE_TYPE must be auto, q4_0, q8_0, f16 or f32")
    return raw


def draft_length_setting(main_gpu=0):
    """Card override > global override > conservative four-token starting point."""
    family, _ = gpu_profile(main_gpu)
    card = os.getenv(f"DFLASH_SPEC_DRAFT_N_MAX_{family}", "") if family else ""
    return int(card or os.getenv("DFLASH_SPEC_DRAFT_N_MAX", "4"))
