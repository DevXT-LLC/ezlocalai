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


def has_voice_server_url() -> bool:
    """Check if a voice server URL is configured (not 'true' mode, but actual URL).

    When a voice server URL is configured:
    - Voice requests (TTS/STT) are forwarded to the voice server
    - Local voice models should NOT be loaded/cached at all

    Returns:
        True if VOICE_SERVER is set to a URL (not empty, not 'true')
    """
    voice_server = getenv("VOICE_SERVER")
    if not voice_server:
        return False
    # If it's set to "true", this server IS the voice server, so we DO need local models
    if voice_server.lower() == "true":
        return False
    # Otherwise it's a URL to another voice server
    return True


def has_image_server_url() -> bool:
    """Check if an image server URL is configured (not 'true' mode, but actual URL).

    When an image server URL is configured:
    - Image/video requests are forwarded to the image server
    - Local image/video models should NOT be loaded/cached at all

    Returns:
        True if IMAGE_SERVER is set to a URL (not empty, not 'true')
    """
    image_server = getenv("IMAGE_SERVER")
    if not image_server:
        return False
    if image_server.lower() == "true":
        return False
    return True


def has_text_server_url() -> bool:
    """Check if a text server URL is configured (not 'true' mode, but actual URL).

    When a text server URL is configured:
    - Text completions are forwarded to the text server
    - Local LLM models should NOT be loaded/cached at all

    Returns:
        True if TEXT_SERVER is set to a URL (not empty, not 'true')
    """
    text_server = getenv("TEXT_SERVER")
    if not text_server:
        return False
    if text_server.lower() == "true":
        return False
    return True


def is_image_server_mode() -> bool:
    """Check if this server IS the image server (IMAGE_SERVER=true).

    Returns:
        True if IMAGE_SERVER env var is set to 'true'
    """
    return getenv("IMAGE_SERVER").lower() == "true"


def is_text_server_mode() -> bool:
    """Check if this server should act as a text server.

    A server acts as a text server when:
    - TEXT_SERVER is explicitly set to 'true', OR
    - Neither IMAGE_SERVER nor VOICE_SERVER is set to 'true' (default behavior)

    Returns:
        True if this server should load and serve LLM models
    """
    text_server = getenv("TEXT_SERVER")
    if text_server and text_server.lower() == "true":
        return True
    # Default: act as text server unless this is a dedicated image or voice server
    if is_image_server_mode():
        return False
    if getenv("VOICE_SERVER").lower() == "true":
        return False
    return True


# Lock file to prevent multiple precache runs
PRECACHE_LOCK = Path("/tmp/ezlocalai_precache.lock")
PRECACHE_DONE = Path("/tmp/ezlocalai_precache.done")


def precache_llm_models():
    """Download and calibrate all configured LLM models."""
    # Skip if text server URL is configured (text passthrough mode)
    if has_text_server_url():
        text_url = getenv("TEXT_SERVER")
        logging.info(f"  - LLM: Skipped (text server: {text_url})")
        return

    # Skip if this is a dedicated image or voice server (not a text server)
    if not is_text_server_mode():
        logging.info("  - LLM: Skipped (not a text server)")
        return

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
            patterns = [quant_type, "Q4_K", "Q5_K", "Q6_K", "Q8"]
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

    # Skip if voice server URL is configured (voice passthrough mode)
    if has_voice_server_url():
        voice_url = getenv("VOICE_SERVER")
        logging.info(f"  - TTS: Skipped (voice server: {voice_url})")
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

    # Skip if voice server URL is configured (voice passthrough mode)
    if has_voice_server_url():
        voice_url = getenv("VOICE_SERVER")
        logging.info(f"  - STT: Skipped (voice server: {voice_url})")
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
    """Download image generation GGUF transformer and pipeline components if configured.

    Downloads the GGUF file during precache. Pipeline components
    (text_encoder, vae, etc.) are downloaded on first inference by
    Flux2KleinPipeline.from_pretrained.
    """
    # Skip if image server URL is configured (image passthrough mode)
    if has_image_server_url():
        image_url = getenv("IMAGE_SERVER")
        logging.info(f"  - Image: Skipped (image server: {image_url})")
        return

    # Skip if this is a dedicated text or voice server (not loading image models)
    if is_text_server_mode() and not is_image_server_mode():
        # Text servers that aren't also image servers still load image models locally
        # unless they have an image server URL configured (handled above)
        pass

    img_model = getenv("IMG_MODEL")
    if not img_model or img_model.lower() == "none":
        return

    try:
        from huggingface_hub import hf_hub_download

        start_time = time.time()

        # For GGUF models, download just the quantized transformer file
        if "gguf" in img_model.lower() or "FLUX.2-klein" in img_model:
            gguf_filename = "flux-2-klein-4b-Q4_K_M.gguf"
            repo = (
                img_model
                if "gguf" in img_model.lower()
                else "unsloth/FLUX.2-klein-4B-GGUF"
            )
            hf_hub_download(repo, filename=gguf_filename, cache_dir="models")
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(img_model)

        elapsed = time.time() - start_time
        logging.info(f"  ✓ {img_model} ({elapsed:.1f}s)")

    except Exception as e:
        logging.error(f"  ✗ Image model: {e}")


def precache_video_model():
    """Download video generation GGUF transformer if configured.

    Only downloads the GGUF file during precache.  Pipeline components
    (text_encoder, vae, etc.) are downloaded on first inference by
    LTX2Pipeline.from_pretrained which is smarter about fetching only
    the files each component actually needs.
    """
    # Skip if image server URL is configured (image/video passthrough mode)
    if has_image_server_url():
        image_url = getenv("IMAGE_SERVER")
        logging.info(f"  - Video: Skipped (image server: {image_url})")
        return

    video_model = getenv("VIDEO_MODEL")
    if not video_model or video_model.lower() == "none":
        return

    try:
        from huggingface_hub import hf_hub_download

        start_time = time.time()

        # Download GGUF transformer file
        gguf_filename = "ltx-2.3-22b-dev-Q4_K_M.gguf"
        logging.info(f"  Downloading {video_model}/{gguf_filename}...")
        hf_hub_download(video_model, filename=gguf_filename, cache_dir="models")

        # Download matching connector text projections from unsloth
        connector_file = (
            "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors"
        )
        logging.info(f"  Downloading {video_model}/{connector_file}...")
        hf_hub_download(video_model, filename=connector_file, cache_dir="models")

        elapsed = time.time() - start_time
        logging.info(f"  ✓ {video_model} ({elapsed:.1f}s)")

    except Exception as e:
        logging.error(f"  ✗ Video model: {e}")


def run_precache():
    """Run all precache operations."""
    # Check if already done
    if PRECACHE_DONE.exists():
        return True

    # Ensure outputs directory exists for static file mounting
    os.makedirs("outputs", exist_ok=True)

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
        precache_video_model()

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
