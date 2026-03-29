import os
from dotenv import load_dotenv

load_dotenv()


def getenv(var_name: str, default_value: str = None) -> str:
    default_values = {
        "EZLOCALAI_URL": "http://localhost:8091",
        "EZLOCALAI_API_KEY": "none",
        "ALLOWED_DOMAINS": "*",
        "DEFAULT_MODEL": "unsloth/Qwen3.5-4B-GGUF",
        "WHISPER_MODEL": "large-v3",
        "IMG_MODEL": "none",  # Set to "unsloth/FLUX.2-klein-4B-GGUF" for image generation + editing, "none" or empty string to disable
        "VIDEO_MODEL": "none",  # Set to "unsloth/LTX-2.3-GGUF" to enable video generation, "none" or empty string to disable
        "TTS_ENABLED": "true",
        "TTS_PROVIDER": "chatterbox",  # Chatterbox TTS
        "STT_ENABLED": "true",
        "NGROK_TOKEN": "",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "%(asctime)s | %(levelname)s | %(message)s",
        "UVICORN_WORKERS": 1,  # GPU inference: 1 worker to avoid duplicate model loads
        "MAIN_GPU": "0",
        "TENSOR_SPLIT": "",
        "QUANT_TYPE": "Q4_K_XL",
        "LLM_BATCH_SIZE": "2048",
        "LLM_MAX_TOKENS": "65536",
        "REASONING_BUDGET": "-1",  # Max thinking tokens per response (-1 = unlimited, 0 = disabled, N = limit)
        # Parallel inference slots — 1 = single slot (default),
        # N = fixed number of parallel slots.
        # Each slot gets n_ctx / n_parallel tokens of context. VRAM is constant.
        "N_PARALLEL": "1",
        "VLM_MAX_TOKENS": "8192",  # Vision models don't need large context
        # Queue system defaults - MAX_CONCURRENT_REQUESTS should match N_PARALLEL
        "MAX_CONCURRENT_REQUESTS": "1",
        "MAX_QUEUE_SIZE": "100",
        "REQUEST_TIMEOUT": "120",
        # How long (seconds) a request waits in queue before trying fallback server.
        # 0 = disabled (wait full REQUEST_TIMEOUT). Only applies when FALLBACK_SERVER is set.
        "QUEUE_WAIT_TIMEOUT": "10",
        # Fallback server for when local resources are exhausted (VRAM/RAM)
        # Can be another ezlocalai instance (e.g., "http://192.168.1.100:8091") for full feature parity
        # Or an OpenAI-compatible API (e.g., "https://api.openai.com/v1") for LLM-only fallback
        "FALLBACK_SERVER": "",  # e.g., "http://other-machine:8091" or "https://api.openai.com/v1"
        "FALLBACK_API_KEY": "",  # API key for fallback server
        "FALLBACK_MODEL": "",  # Model to use for non-ezlocalai fallback (e.g., "gpt-4o-mini")
        # Minimum combined free memory (VRAM + RAM) in GB before falling back to remote server
        # Models can offload to RAM, so combined memory is more accurate than VRAM alone
        "FALLBACK_MEMORY_THRESHOLD": "1.0",
        # Voice server URL for offloading TTS/STT requests
        # - Empty (default): Load voice models locally on demand (lazy loading)
        # - URL (e.g., "http://192.168.1.100:8091"): Forward voice requests to another ezlocalai server
        # - "true": This server IS a voice server - keep TTS/STT loaded, lazy-load LLMs instead
        "VOICE_SERVER": "",
        "VOICE_SERVER_API_KEY": "",  # API key for voice server (uses local key if not provided)
        # Image server URL for offloading image/video generation requests
        # - Empty (default): Load image/video models locally on demand (lazy loading)
        # - URL (e.g., "http://192.168.1.100:8091"): Forward image/video requests to another ezlocalai server
        # - "true": This server IS an image server - keep image/video models loaded, skip LLMs
        "IMAGE_SERVER": "",
        "IMAGE_SERVER_API_KEY": "",  # API key for image server (uses local key if not provided)
        # Text server URL for offloading LLM text completion requests
        # - Empty (default): Determined automatically - acts as text server unless IMAGE_SERVER or VOICE_SERVER is "true"
        # - URL (e.g., "http://192.168.1.100:8091"): Forward text requests to another ezlocalai server
        # - "true": This server IS a text server (explicit)
        "TEXT_SERVER": "",
        "TEXT_SERVER_API_KEY": "",  # API key for text server (uses local key if not provided)
        # Lazy load voice models (TTS/STT) - when false, preload them at startup for faster first response
        # Default true = lazy load (load on first request), false = preload at startup
        "LAZY_LOAD_VOICE": "true",
    }
    if not default_value:
        default_value = default_values[var_name] if var_name in default_values else ""
    return os.getenv(var_name, default_value)
