"""Shared GPU-aware inference settings for loading and memory planning."""

import os
import re


def gpu_profile(main_gpu=0):
    """Return a supported GPU family and visible capacity (including MIG slices).

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
            match = re.search(
                r"\b(?:RTX\s+)?(3090|4090|5090|T4|A100|H100)\b", info.name.upper()
            )
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
        return {"T4": 2, "3090": 3}.get(family, 4)
    return int(value)


def colab_inference_defaults(main_gpu=0):
    """Conservative single-model notebook settings, not throughput benchmarks.

    Probe inside the server venv, not the Colab kernel. Use visible/free memory
    rather than assuming an A100/H100 is an entire 40/80 GB device. Context and
    host prompt-cache limits only apply when the notebook requests this profile;
    normal server deployments retain their configured context/cache budgets.
    """
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before starting ezLocalai.")
    info = torch.cuda.get_device_properties(main_gpu)
    free_bytes, total_bytes = torch.cuda.mem_get_info(main_gpu)
    family, _ = gpu_profile(main_gpu)
    free_gib = min(free_bytes, total_bytes, info.total_memory) / 1024**3
    # Qwen3.8-27B Q3_K_XL runs at 230K on the user's 24 GB cards;
    # a 40 GB card has enough headroom for the full 262K context.
    if free_gib >= 32:
        context = 262144
    elif free_gib >= 20:
        context = 230000
    else:
        context = 8192

    # A T4 session often has little host RAM left for mmap'd model weights.
    # On larger runtimes budget at most 1/8 of available RAM, capped at 8 GiB.
    ram_available = psutil.virtual_memory().available
    cache_mib = min(8192, int(ram_available / (8 * 1024**2)) // 256 * 256)
    if family == "T4" or free_gib < 20 or cache_mib < 512:
        cache_mib = 0
    return {
        "gpu_name": info.name,
        "gpu_family": family,
        "vram_free_gib": round(free_gib, 1),
        "vram_total_gib": round(info.total_memory / 1024**3, 1),
        "settings": {
            "LLM_MAX_TOKENS": str(context),
            "LLM_MAX_OUTPUT_TOKENS": str(min(16384, context // 2)),
            "LLM_PROMPT_CACHE_MIB": str(cache_mib),
            "LLM_BATCH_SIZE": "auto",
            "LLM_UBATCH_SIZE": "auto",
            "KV_CACHE_TYPE": "auto",
            "MTP_SPEC_DRAFT_N_MAX": "auto",
            "LLM_SPECULATIVE_TYPE": "auto",
            "N_PARALLEL": "1",
        },
    }
