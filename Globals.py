import os
from dotenv import load_dotenv

load_dotenv()


def getenv(var_name: str, default_value: str = None) -> str:
    default_values = {
        "EZLOCALAI_URL": "http://localhost:8091",
        "EZLOCALAI_API_KEY": "none",
        "ALLOWED_DOMAINS": "*",
        "DEFAULT_MODEL": "unsloth/Qwen3-1.7B-GGUF",
        "WHISPER_MODEL": "base.en",
        "VISION_MODEL": "",
        "SD_MODEL": "",
        "EMBEDDING_ENABLED": "false",
        "IMG_ENABLED": "false",
        "TTS_ENABLED": "true",
        "STT_ENABLED": "true",
        "IMG_DEVICE": "cpu",
        "NGROK_TOKEN": "",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "%(asctime)s | %(levelname)s | %(message)s",
        "UVICORN_WORKERS": 10,
        "GPU_LAYERS": "0",
        "MAIN_GPU": "0",
        "TENSOR_SPLIT": "",
        "QUANT_TYPE": "Q4_K_M",
        "LLM_MAX_TOKENS": "2048",
        "LLM_BATCH_SIZE": "16",
        # Queue system defaults
        "MAX_CONCURRENT_REQUESTS": "1",
        "MAX_QUEUE_SIZE": "100",
        "REQUEST_TIMEOUT": "300",
    }
    if not default_value:
        default_value = default_values[var_name] if var_name in default_values else ""
    return os.getenv(var_name, default_value)
