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
    """Keep Q4 KV by default on all cards; Q8 remains an explicit precision opt-in."""
    family, _ = gpu_profile(main_gpu)
    raw = os.getenv(f"KV_CACHE_TYPE_{family}", "") if family else ""
    raw = (raw or os.getenv("KV_CACHE_TYPE", "auto")).strip().lower()
    if raw in ("", "auto"):
        return "q4_0"
    if raw not in {"q4_0", "q8_0", "f16", "f32"}:
        raise ValueError("KV_CACHE_TYPE must be auto, q4_0, q8_0, f16 or f32")
    return raw


def draft_length_setting(main_gpu=0):
    """Card override > global override > card-specific starting point."""
    family, _ = gpu_profile(main_gpu)
    card = os.getenv(f"DFLASH_SPEC_DRAFT_N_MAX_{family}", "") if family else ""
    value = (card or os.getenv("DFLASH_SPEC_DRAFT_N_MAX", "auto")).strip().lower()
    if value in ("", "auto"):
        return 3 if family == "3090" else 4
    return int(value)
