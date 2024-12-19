import os
from dotenv import load_dotenv

load_dotenv()


def getenv(var_name: str):

# Define new environment variables for model functionalities
default_values.update({
    "LLM_ENABLED": "true",
    "VISION_ENABLED": "true",
    "IMG_ENABLED": "true",
    "TTS_ENABLED": "true",
    "STT_ENABLED": "true",
})

default_values = {
"EZLOCALAI_URL": "http://localhost:8091",
"ALLOWED_DOMAINS": "*",
"DEFAULT_MODEL": "TheBloke/phi-2-dpo-GGUF",
"WHISPER_MODEL": "base.en",
"VISION_MODEL": "",
"SD_MODEL": "",
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
}
default_value = default_values[var_name] if var_name in default_values else ""
return os.getenv(var_name, default_value)
    default_values = {
        "EZLOCALAI_URL": "http://localhost:8091",
        "ALLOWED_DOMAINS": "*",
        "DEFAULT_MODEL": "TheBloke/phi-2-dpo-GGUF",
        "WHISPER_MODEL": "base.en",
        "VISION_MODEL": "",
        "SD_MODEL": "",
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
    }
    default_value = default_values[var_name] if var_name in default_values else ""
    return os.getenv(var_name, default_value)
