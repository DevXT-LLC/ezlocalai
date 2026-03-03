# Raspberry Pi 5 / ARM64 Dockerfile
# Runs ezlocalai with xllamacpp CPU inference for any GGUF model
FROM python:3.12-bookworm

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends \
        git build-essential gcc g++ cmake \
        ffmpeg libsndfile1 \
        python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip --no-cache-dir

WORKDIR /app

# Install PyTorch CPU (arm64 wheels available from official index)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install numpy
RUN pip install numpy --no-cache-dir

COPY rpi-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --no-cache-dir -r rpi-requirements.txt

# Install xllamacpp - prebuilt aarch64 wheels available on PyPI
RUN pip install xllamacpp --force-reinstall --no-cache-dir

COPY . .

# Pi 5 appropriate defaults
ENV HOST=0.0.0.0 \
    TOKENIZERS_PARALLELISM=false \
    UVICORN_WORKERS=2 \
    LLM_BATCH_SIZE=512 \
    LLM_MAX_TOKENS=8192 \
    TTS_ENABLED=false \
    STT_ENABLED=false \
    IMG_MODEL="" \
    DEFAULT_MODEL=unsloth/Qwen3.5-0.6B-GGUF

EXPOSE 8091
CMD ["python", "start.py"]
