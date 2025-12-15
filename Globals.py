import os
from dotenv import load_dotenv

load_dotenv()


def getenv(var_name: str, default_value: str = None) -> str:
    default_values = {
        "EZLOCALAI_URL": "http://localhost:8091",
        "EZLOCALAI_API_KEY": "none",
        "ALLOWED_DOMAINS": "*",
        "DEFAULT_MODEL": "unsloth/Qwen3-VL-4B-Instruct-GGUF",
        "WHISPER_MODEL": "large-v3",
        "IMG_MODEL": "Tongyi-MAI/Z-Image-Turbo",  # Set to Z-Image-Turbo for image generation, empty string to disable
        "TTS_ENABLED": "true",
        "STT_ENABLED": "true",
        "NGROK_TOKEN": "",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "%(asctime)s | %(levelname)s | %(message)s",
        "UVICORN_WORKERS": 10,
        "MAIN_GPU": "0",
        "TENSOR_SPLIT": "",
        "QUANT_TYPE": "Q4_K_XL",
        "LLM_BATCH_SIZE": "2048",
        "LLM_MAX_TOKENS": "40000",
        "VLM_MAX_TOKENS": "8192",  # Vision models don't need large context
        # Queue system defaults - supports concurrent requests with resource fallback
        "MAX_CONCURRENT_REQUESTS": "5",
        "MAX_QUEUE_SIZE": "100",
        "REQUEST_TIMEOUT": "300",
        # Fallback server for when local resources are exhausted
        # Set these to use an external API when local model loading fails
        "FALLBACK_SERVER": "",  # e.g., "https://api.openai.com/v1"
        "FALLBACK_MODEL": "",  # e.g., "gpt-4o-mini"
        "FALLBACK_API_KEY": "",  # API key for fallback server
    }
    if not default_value:
        default_value = default_values[var_name] if var_name in default_values else ""
    return os.getenv(var_name, default_value)
