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


def has_embedding_server_url() -> bool:
    """Check if an embedding server URL is configured."""
    embedding_server = getenv("EMBEDDING_SERVER")
    if not embedding_server:
        return False
    if embedding_server.lower() == "true":
        return False
    return True


def is_image_server_mode() -> bool:
    """Check if this server IS the image server (IMAGE_SERVER=true).

    Returns:
        True if IMAGE_SERVER env var is set to 'true'
    """
    return getenv("IMAGE_SERVER").lower() == "true"


def is_image_enabled() -> bool:
    """Check if this worker should cache and serve local image generation."""
    return (getenv("IMAGE_ENABLED") or "false").strip().lower() == "true"


def is_video_enabled() -> bool:
    """Check if this worker should cache and serve local video generation."""
    return (getenv("VIDEO_ENABLED") or "false").strip().lower() == "true"


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


def _format_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    if num_bytes is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f}TB"


def _get_remote_size(repo_id: str, filename: str, revision=None) -> int:
    """Best-effort lookup of a file's total size on the Hugging Face Hub."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        meta = get_hf_file_metadata(hf_hub_url(repo_id, filename, revision=revision))
        return int(meta.size) if meta and meta.size else 0
    except Exception:
        return 0


def _scan_downloaded_bytes(local_dir, cache_dir, repo_id: str, filename: str) -> int:
    """Return the current size of the in-progress or completed download.

    huggingface_hub stages downloads into different locations depending on
    whether ``local_dir`` is provided:
      * ``local_dir`` mode: writes to
        ``<local_dir>/.cache/huggingface/download/<hash>.incomplete``, then
        moves the final file to ``<local_dir>/<filename>``.
      * cache_dir mode: writes to
        ``<cache_dir>/models--<org>--<repo>/blobs/<hash>(.incomplete)``.
    We probe both and return the largest matching size so progress reflects
    whichever file is currently growing.
    """
    candidates = []
    if local_dir:
        candidates.append(os.path.join(local_dir, filename))
        staging = os.path.join(local_dir, ".cache", "huggingface", "download")
        if os.path.isdir(staging):
            for entry in os.listdir(staging):
                if entry.endswith(".incomplete"):
                    candidates.append(os.path.join(staging, entry))
    if cache_dir and repo_id:
        repo_cache = os.path.join(
            cache_dir, "models--" + repo_id.replace("/", "--"), "blobs"
        )
        if os.path.isdir(repo_cache):
            for entry in os.listdir(repo_cache):
                candidates.append(os.path.join(repo_cache, entry))

    largest = 0
    for path in candidates:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > largest:
            largest = size
    return largest


def download_with_progress(repo_id: str, filename: str, **kwargs):
    """Download a file from the Hugging Face Hub and log progress periodically.

    Wraps ``huggingface_hub.hf_hub_download`` so that long downloads (e.g. multi
    GB GGUF model files) emit periodic progress lines that survive Docker log
    aggregation (no carriage returns / TTY required). The underlying download
    is unchanged — caching, symlinks, and resume behavior are preserved.
    """
    import threading

    from huggingface_hub import hf_hub_download

    revision = kwargs.get("revision")
    total = _get_remote_size(repo_id, filename, revision=revision)

    cache_dir = kwargs.get("cache_dir") or os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )
    local_dir = kwargs.get("local_dir")

    label = f"{repo_id}/{filename}"
    total_str = _format_bytes(total) if total else "unknown"
    logging.info(f"  ⬇ {label} ({total_str})")

    result: dict = {}

    def _do_download():
        try:
            result["path"] = hf_hub_download(repo_id, filename, **kwargs)
        except BaseException as err:  # noqa: BLE001 - re-raised below
            result["error"] = err

    worker = threading.Thread(target=_do_download, daemon=True)
    start = time.time()
    worker.start()

    interval = float(os.getenv("EZLOCALAI_DOWNLOAD_PROGRESS_INTERVAL", "10"))
    while True:
        worker.join(timeout=interval)
        if not worker.is_alive():
            break
        downloaded = _scan_downloaded_bytes(local_dir, cache_dir, repo_id, filename)
        elapsed = max(time.time() - start, 1e-6)
        rate = downloaded / elapsed
        if total and downloaded:
            pct = min(downloaded * 100.0 / total, 99.9)
            remaining = max(total - downloaded, 0)
            eta = remaining / rate if rate > 0 else 0
            logging.info(
                f"    … {label}: {_format_bytes(downloaded)}/{_format_bytes(total)}"
                f" ({pct:.1f}%) {_format_bytes(rate)}/s ETA {eta:.0f}s"
            )
        else:
            logging.info(
                f"    … {label}: {_format_bytes(downloaded)} so far"
                f" ({_format_bytes(rate)}/s, {elapsed:.0f}s elapsed)"
            )

    if "error" in result:
        raise result["error"]
    return result.get("path")


def snapshot_download_with_progress(repo_id: str, **kwargs):
    """Run ``snapshot_download`` while logging cache growth periodically."""
    import threading

    from huggingface_hub import snapshot_download

    cache_dir = kwargs.get("cache_dir") or os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )
    logging.info(f"  ⬇ snapshot {repo_id}")

    result: dict = {}

    def _do_snapshot():
        try:
            result["path"] = snapshot_download(repo_id, **kwargs)
        except BaseException as err:  # noqa: BLE001
            result["error"] = err

    worker = threading.Thread(target=_do_snapshot, daemon=True)
    start = time.time()
    worker.start()

    interval = float(os.getenv("EZLOCALAI_DOWNLOAD_PROGRESS_INTERVAL", "10"))

    def _dir_size(path: str) -> int:
        total = 0
        if not path or not os.path.isdir(path):
            return 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    continue
        return total

    while True:
        worker.join(timeout=interval)
        if not worker.is_alive():
            break
        size = _dir_size(cache_dir)
        elapsed = max(time.time() - start, 1e-6)
        rate = size / elapsed
        logging.info(
            f"    … snapshot {repo_id}: {_format_bytes(size)} cached"
            f" ({_format_bytes(rate)}/s, {elapsed:.0f}s elapsed)"
        )

    if "error" in result:
        raise result["error"]
    return result.get("path")


def precache_llm_models():
    """Download and calibrate all configured LLM models."""
    # Skip if text server URL is configured (text passthrough mode)
    if has_text_server_url():
        text_url = getenv("TEXT_SERVER")
        logging.info(f"  - LLM: Skipped (text server: {text_url})")
        return

    # Skip if this is a dedicated image or voice server with no LLM configured
    if not is_text_server_mode():
        model_check = getenv("DEFAULT_MODEL")
        if model_check.lower() == "none":
            logging.info("  - LLM: Skipped (not a text server)")
            return

    from huggingface_hub import hf_hub_download, list_repo_files

    model_config = getenv("DEFAULT_MODEL")
    if model_config.lower() == "none":
        return

    quant_values = [v.strip() for v in getenv("QUANT_TYPE", "Q4_K_XL").split(",")]

    for i, model_entry in enumerate(model_config.split(",")):
        model_name = model_entry.strip()
        if "@" in model_name:
            model_name = model_name.rsplit("@", 1)[0]

        if not model_name or "/" not in model_name:
            continue

        quant_type = quant_values[i] if i < len(quant_values) else quant_values[-1]
        quant_type = quant_type if quant_type else None

        start_time = time.time()

        # Use the same local_dir as download_model() in LLM.py so that
        # the file is found on disk during model loading without re-downloading.
        model_short = model_name.split("/")[-1].split("-GGUF")[0]
        model_dir = os.path.join("models", model_short)
        os.makedirs(model_dir, exist_ok=True)

        try:
            files = None  # Will be populated if we need to query the repo

            # Check if a model GGUF already exists in the target directory
            existing_gguf = [
                f
                for f in os.listdir(model_dir)
                if f.endswith(".gguf") and "mmproj" not in f.lower()
            ]
            # Prefer an existing file that matches the requested QUANT_TYPE.
            # If none matches, we need to download the correct quant.
            matching_gguf = [f for f in existing_gguf if quant_type and quant_type in f]
            if matching_gguf:
                elapsed = time.time() - start_time
                logging.info(
                    f"  ✓ {model_name} (cached: {matching_gguf[0]}, {elapsed:.1f}s)"
                )
                # Still check for vision projector below
            elif existing_gguf and not quant_type:
                # No specific quant requested, use whatever exists
                elapsed = time.time() - start_time
                logging.info(
                    f"  ✓ {model_name} (cached: {existing_gguf[0]}, {elapsed:.1f}s)"
                )
            else:
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

                # Download the model file to the same local_dir that LLM.py expects
                model_path = download_with_progress(
                    model_name, best_file, local_dir=model_dir
                )
                elapsed = time.time() - start_time
                logging.info(f"  ✓ {model_name} ({elapsed:.1f}s)")

            # Also download vision projector if it exists
            mmproj_exists = any(
                f
                for f in os.listdir(model_dir)
                if "mmproj" in f.lower() and f.endswith(".gguf")
            )
            if not mmproj_exists:
                if files is None:
                    files = list_repo_files(model_name)
                mmproj_files = [
                    f for f in files if "mmproj" in f.lower() and f.endswith(".gguf")
                ]
                if mmproj_files:
                    proj_file = mmproj_files[0]
                    download_with_progress(model_name, proj_file, local_dir=model_dir)

        except Exception as e:
            logging.error(f"  ✗ {model_name}: {e}")


def precache_embedding_model():
    """Download the configured GGUF embedding model."""
    if getenv("EMBEDDING_ENABLED").lower() != "true":
        logging.info("  - Embedding: Skipped (disabled)")
        return

    if has_embedding_server_url():
        embedding_url = getenv("EMBEDDING_SERVER")
        logging.info(f"  - Embedding: Skipped (embedding server: {embedding_url})")
        return

    model_name = getenv("EMBEDDING_MODEL")
    if not model_name or model_name.lower() == "none":
        return

    if "/" not in model_name:
        model_name = "Qwen/" + model_name

    quant_type = getenv("EMBEDDING_QUANT_TYPE", "Q8_0") or "Q8_0"
    model_short = model_name.split("/")[-1].split("-GGUF")[0]
    model_dir = os.path.join("models", model_short)
    os.makedirs(model_dir, exist_ok=True)

    start_time = time.time()
    try:
        from huggingface_hub import list_repo_files

        existing_gguf = [
            f
            for f in os.listdir(model_dir)
            if f.endswith(".gguf") and "mmproj" not in f.lower()
        ]
        matching_gguf = [f for f in existing_gguf if quant_type and quant_type in f]
        if matching_gguf:
            elapsed = time.time() - start_time
            logging.info(
                f"  ✓ {model_name} embedding (cached: {matching_gguf[0]}, {elapsed:.1f}s)"
            )
            return

        files = list_repo_files(model_name)
        gguf_files = [
            f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()
        ]
        if not gguf_files:
            logging.warning(f"[ezlocalai] No GGUF files in {model_name}")
            return

        best_file = None
        for pattern in (quant_type, "Q8_0", "Q6_K", "Q5_K", "Q4_K"):
            for filename in gguf_files:
                if pattern and pattern in filename:
                    best_file = filename
                    break
            if best_file:
                break
        if not best_file:
            best_file = gguf_files[0]

        download_with_progress(model_name, best_file, local_dir=model_dir)
        elapsed = time.time() - start_time
        logging.info(f"  ✓ {model_name} embedding ({elapsed:.1f}s)")
    except Exception as e:
        logging.error(f"  ✗ Embedding model: {e}")


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
    if not is_image_enabled():
        logging.info("  - Image: Skipped (disabled)")
        return

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
            download_with_progress(repo, filename=gguf_filename, cache_dir="models")
        else:
            snapshot_download_with_progress(img_model)

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
    if not is_video_enabled():
        logging.info("  - Video: Skipped (disabled)")
        return

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
        download_with_progress(video_model, filename=gguf_filename, cache_dir="models")

        # Download matching connector text projections from unsloth
        connector_file = (
            "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors"
        )
        download_with_progress(video_model, filename=connector_file, cache_dir="models")

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
        precache_embedding_model()
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
