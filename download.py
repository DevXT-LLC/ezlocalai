import os
import logging

try:
    import requests
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def download_whisper_model(model="base.en"):
    # https://huggingface.co/ggerganov/whisper.cpp
    if model not in [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
    ]:
        model = "base.en"
    os.makedirs(os.path.join(os.getcwd(), "whispercpp"), exist_ok=True)
    model_path = os.path.join(os.getcwd(), "whispercpp", f"ggml-{model}.bin")
    if not os.path.exists(model_path):
        logging.info(f"[STT] Downloading {model} for Whisper...")
        r = requests.get(
            f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin",
            allow_redirects=True,
        )
        open(model_path, "wb").write(r.content)
    return model_path


def download_xtts():
    files_to_download = {
        "LICENSE.txt": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true",
        "README.md": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true",
        "config.json": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true",
        "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true",
        "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/dvae.pth?download=true",
        "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/mel_stats.pth?download=true",
        "speakers_xtts.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/speakers_xtts.pth?download=true",
        "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true",
    }
    os.makedirs(os.path.join(os.getcwd(), "xttsv2_2.0.2"), exist_ok=True)
    for filename, url in files_to_download.items():
        logging.info(f"[CTTS] Downloading {filename} for XTTSv2...")
        destination = os.path.join(os.getcwd(), "xttsv2_2.0.2", filename)
        if not os.path.exists(destination):
            response = requests.get(url, stream=True)
            block_size = 1024  # 1 Kibibyte
            with open(destination, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)


download_whisper_model()
download_xtts()
