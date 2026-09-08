"""Speculative backend selection, shared by loading, precache and slot accounting."""

import logging
import math
import os
import re
from ezlocalai.InferenceSettings import draft_length_setting

DFLASH_REPO = "incoai/Qwen3.8-27B-DFlash2-GGUF"
DFLASH_FILE = "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
# Includes the corrected vision-compatible target layer mapping.
DFLASH_REVISION = "51962825493a48b846b40126d35c799ac4093ad0"


def speculative_backend(model_name):
    """Auto-select only a known compatible target; never guess a draft pairing."""
    name = (model_name or "").lower()
    qwen38 = bool(re.search(r"qwen3\.8-27b(?:$|[-_./])", name))
    mtp = qwen38 or "-mtp" in name
    backend = os.getenv("LLM_SPECULATIVE_TYPE", "auto").strip().lower()
    backend = {"draft-dflash": "dflash2", "draft-mtp": "mtp", "off": "none"}.get(
        backend, backend
    )
    if backend in ("", "auto"):
        return "dflash2" if qwen38 else "mtp" if mtp else "none"
    if backend not in ("dflash2", "mtp", "none"):
        raise ValueError("LLM_SPECULATIVE_TYPE must be auto, dflash2, mtp or none")
    # A global override must not attach the 27B draft to other configured models.
    if backend == "dflash2" and not qwen38:
        return "mtp" if mtp else "none"
    if backend == "mtp" and not mtp:
        return "none"
    return backend


def dflash_settings(main_gpu=0):
    """The published DFlash2 block is anchor + seven draft tokens."""
    n_max = draft_length_setting(main_gpu)
    p_min = float(os.getenv("DFLASH_SPEC_DRAFT_P_MIN", "0.0"))
    if not 1 <= n_max <= 7:
        raise ValueError("DFLASH_SPEC_DRAFT_N_MAX must be between 1 and 7")
    if not math.isfinite(p_min) or not 0 <= p_min <= 1:
        raise ValueError("DFLASH_SPEC_DRAFT_P_MIN must be between 0 and 1")
    return n_max, p_min


def download_dflash_model():
    """Use HF's revision-aware shared cache (also safe for concurrent workers)."""
    from huggingface_hub import hf_hub_download

    path = os.getenv("DFLASH_MODEL_PATH", "").strip()
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DFLASH_MODEL_PATH does not exist: {path}")
        return path
    filename = os.getenv("DFLASH_MODEL_FILE", DFLASH_FILE).strip() or DFLASH_FILE
    logging.info("[LLM] Resolving DFlash2 draft %s/%s", DFLASH_REPO, filename)
    return hf_hub_download(DFLASH_REPO, filename, revision=DFLASH_REVISION)
