# ROCm Dockerfile for AMD GPU support (RDNA 2/3/3.5 and CDNA architectures)
# Supports AMD Radeon RX 6000/7000/9000 series, Radeon PRO, and Ryzen APUs with integrated graphics
FROM rocm/dev-ubuntu-24.04:6.4.1
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm/hip \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:/opt/rocm/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/rocm/lib"
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
       git build-essential cmake gcc g++ ninja-build \
       portaudio19-dev ffmpeg libportaudio2 libasound-dev \
       python3 python3-pip python3-venv wget python3-dev unzip curl \
       rocm-hip-runtime rocm-hip-sdk hipblas-dev rocblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*
# Install uv and create venv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv /opt/venv
WORKDIR /app
# Install PyTorch with ROCm 6.4 support (stable version 2.9.1)
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
# Install numpy and Cython for pkuseg (required by chatterbox-tts)
# numpy>=1.26.0 required for Python 3.12 compatibility
RUN uv pip install "numpy>=1.26.0" Cython
# Install pkuseg separately (required by chatterbox-tts)
RUN uv pip install pkuseg==0.0.25 --no-build-isolation
COPY rocm-requirements.txt .
RUN uv pip install -r rocm-requirements.txt
# Install chatterbox-tts with --no-deps to bypass transformers==4.46.3 pin
# This allows us to use transformers>=4.53.0 for security fixes
RUN uv pip install chatterbox-tts --no-deps
ENV HOST=0.0.0.0 \
    ROCM_VER=6.4.1
# Install xllamacpp with ROCm 6.4.1 support
RUN uv pip install xllamacpp --reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-6.4.1
COPY . .
EXPOSE 8091
# Use start.py which runs precache once, then starts uvicorn workers
CMD ["python", "start.py"]
