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
        # GPU assignment — comma-separated to match DEFAULT_MODEL order.
        # e.g., DEFAULT_MODEL="model_a,model_b" with MAIN_GPU="0,1" loads model_a on GPU 0, model_b on GPU 1.
        # Single value applies to all models.
        "MAIN_GPU": "0",
        "TENSOR_SPLIT": "",
        # Quantization preference — comma-separated to match DEFAULT_MODEL order.
        # e.g., QUANT_TYPE="Q4_K_XL,Q3_K_XL" picks per-model quants.
        # Single value applies to all models.
        "QUANT_TYPE": "Q4_K_XL",
        "LLM_BATCH_SIZE": "auto",
        # Context size — comma-separated to match DEFAULT_MODEL order.
        # e.g., LLM_MAX_TOKENS="65536,262144" gives model_a 65k and model_b 262k context.
        "LLM_MAX_TOKENS": "65536",
        "REASONING_BUDGET": "-1",  # Max thinking tokens per response (-1 = unlimited, 0 = disabled, N = limit)
        # Parallel inference slots — comma-separated to match DEFAULT_MODEL order.
        # 1 = single slot (default), N = fixed number of parallel slots.
        # Each slot gets n_ctx / n_parallel tokens of context. VRAM is constant.
        "N_PARALLEL": "1",
        # MTP speculative decoding: require draft tokens to have at least this probability.
        "MTP_SPEC_DRAFT_P_MIN": "0.25",
        # MTP draft length. auto = 2 on large models with 24GB cards, 3 on
        # 25-31GB, 4 on 32GB+ or <=4B models with at least 20GB VRAM.
        "MTP_SPEC_DRAFT_N_MAX": "auto",
        "VLM_MAX_TOKENS": "8192",  # Vision models don't need large context
        # Queue system defaults. Text queue concurrency is derived from loaded
        # model n_parallel slots.
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
        # =====================================================================
        # Router / Load Balancer mode
        # =====================================================================
        # When ROUTER_MODE=true, this instance becomes a pure router:
        #   - No models are loaded locally
        #   - Worker ezlocalai instances register themselves and send heartbeats
        #   - Inference requests are proxied to the best available worker
        # Use start.py to launch; it auto-selects router_app vs app based on this.
        "ROUTER_MODE": "false",
        # Shared secret that workers must present to register/heartbeat with the router.
        # If empty, EZLOCALAI_API_KEY is used as the shared secret.
        "ROUTER_REGISTER_KEY": "",
        # How long (seconds) since last heartbeat before a worker is pruned.
        "ROUTER_WORKER_TTL": "30",
        # How long (seconds) the router waits for a worker slot before erroring/queuing.
        "ROUTER_WAIT_TIMEOUT": "120",
        # =====================================================================
        # Worker registration (set on each ezlocalai worker instance)
        # =====================================================================
        # If set, this ezlocalai instance will register itself with a router and
        # heartbeat its current state (free VRAM, queue depth, loaded models).
        # Example: ROUTER_URL="http://192.168.1.50:8092"
        "ROUTER_URL": "",
        # API key/shared secret to authenticate with the router. Falls back to
        # EZLOCALAI_API_KEY if unset.
        "ROUTER_API_KEY": "",
        # Friendly name of this worker (shown in router admin). Defaults to hostname.
        "WORKER_LABEL": "",
        # Heartbeat interval in seconds.
        "WORKER_HEARTBEAT_INTERVAL": "10",
        # Open a reverse WebSocket tunnel to the router so the router can dispatch
        # inference requests back through it. Required when this worker has no
        # public IP (CGNAT, friend's home box, etc). When true, the router will
        # use the tunnel instead of trying to call EZLOCALAI_URL directly.
        "WORKER_TUNNEL": "false",
    }
    if not default_value:
        default_value = default_values[var_name] if var_name in default_values else ""
    return os.getenv(var_name, default_value)
