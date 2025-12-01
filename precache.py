#!/usr/bin/env python3
"""
Pre-cache all models and resources before starting workers.

This script runs ONCE before uvicorn workers are spawned to:
1. Download all model files (LLM, TTS, STT, Vision, etc.)
2. Warm model caches (TTS voice models, etc.)
3. Pre-calculate GPU layer calibrations

After this runs, workers can start immediately without redundant downloads.
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path

# Suppress SyntaxWarnings from third-party packages
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Setup logging - minimal format for cleaner output
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
)

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from Globals import getenv

# Lock file to prevent multiple precache runs
PRECACHE_LOCK = Path("/tmp/ezlocalai_precache.lock")
PRECACHE_DONE = Path("/tmp/ezlocalai_precache.done")


def precache_llm_models():
    """Download and calibrate all configured LLM models."""
    from huggingface_hub import hf_hub_download, list_repo_files

    model_config = getenv("DEFAULT_MODEL")
    if model_config.lower() == "none":
        return

    quant_type = getenv("QUANT_TYPE")

    for model_entry in model_config.split(","):
        model_name = model_entry.strip()
        if "@" in model_name:
            model_name = model_name.rsplit("@", 1)[0]

        if not model_name or "/" not in model_name:
            continue

        start_time = time.time()

        try:
            # Get list of files in repo
            files = list_repo_files(model_name)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not gguf_files:
                logging.warning(f"[ezlocalai] No GGUF files in {model_name}")
                continue

            # Find best quantization
            best_file = None
            patterns = [quant_type, "Q4_K_M", "Q4_K_XL", "Q4_K", "Q5_K", "Q6_K", "Q8"]
            for pattern in patterns:
                for f in gguf_files:
                    if pattern in f:
                        best_file = f
                        break
                if best_file:
                    break

            if not best_file:
                best_file = gguf_files[0]

            # Download the model file
            model_path = hf_hub_download(model_name, best_file)
            elapsed = time.time() - start_time
            logging.info(f"  ✓ {model_name} ({elapsed:.1f}s)")

            # Also download vision projector if it exists
            mmproj_files = [
                f for f in files if "mmproj" in f.lower() and f.endswith(".gguf")
            ]
            if mmproj_files:
                proj_file = mmproj_files[0]
                hf_hub_download(model_name, proj_file)

        except Exception as e:
            logging.error(f"  ✗ {model_name}: {e}")


def precache_tts():
    """Download and warm TTS models."""
    if getenv("TTS_ENABLED").lower() != "true":
        return

    start_time = time.time()

    try:
        from CTTS import CTTS

        # Initialize TTS - this downloads models
        ctts = CTTS()
        elapsed = time.time() - start_time
        logging.info(f"  ✓ TTS models ({elapsed:.1f}s)")

        # Clean up
        del ctts

        # Force garbage collection to free memory
        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    except Exception as e:
        logging.debug(f"  - TTS: {e}")


def precache_stt():
    """Download STT/Whisper models."""
    if getenv("STT_ENABLED").lower() != "true":
        return

    whisper_model = getenv("WHISPER_MODEL")
    if not whisper_model:
        return

    start_time = time.time()

    try:
        from faster_whisper import WhisperModel

        # Download model (compute_type doesn't matter for download)
        model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        elapsed = time.time() - start_time
        logging.info(f"  ✓ Whisper/{whisper_model} ({elapsed:.1f}s)")

        # Clean up
        del model

        import gc

        gc.collect()

    except Exception as e:
        logging.error(f"  ✗ Whisper: {e}")


def precache_image_model():
    """Download image generation models if configured."""
    img_model = getenv("IMG_MODEL")
    if not img_model:
        return

    try:
        from huggingface_hub import snapshot_download

        # Download the model
        start_time = time.time()
        snapshot_download(img_model)
        elapsed = time.time() - start_time
        logging.info(f"  ✓ {img_model} ({elapsed:.1f}s)")

    except Exception as e:
        logging.error(f"  ✗ Image model: {e}")


def run_precache():
    """Run all precache operations."""
    # Check if already done
    if PRECACHE_DONE.exists():
        return True

    # Acquire lock to prevent concurrent runs
    try:
        # Create lock file atomically
        fd = os.open(str(PRECACHE_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        # Another process is running precache
        logging.info("[ezlocalai] Waiting for model cache...")
        while PRECACHE_LOCK.exists() and not PRECACHE_DONE.exists():
            time.sleep(1)
        if PRECACHE_DONE.exists():
            return True
        # Lock file exists but done file doesn't - stale lock?
        pass

    try:
        logging.info("[ezlocalai] Caching models...")
        total_start = time.time()

        # Run all precache operations
        precache_llm_models()
        precache_tts()
        precache_stt()
        precache_image_model()

        total_elapsed = time.time() - total_start
        logging.info(f"[ezlocalai] Models cached in {total_elapsed:.1f}s")

        # Mark as done
        PRECACHE_DONE.touch()
        return True

    except Exception as e:
        logging.error(f"[ezlocalai] Cache failed: {e}")
        return False

    finally:
        # Release lock
        try:
            PRECACHE_LOCK.unlink()
        except:
            pass


if __name__ == "__main__":
    success = run_precache()
    sys.exit(0 if success else 1)
