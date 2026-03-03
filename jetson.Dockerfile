# Jetson (ARM64 + CUDA) Dockerfile - JetPack 5.x / 6.x
# Base: NVIDIA L4T JetPack container with CUDA, cuDNN, TensorRT pre-installed
# Build ON the Jetson itself: docker compose -f docker-compose-jetson.yml build
#
# Supports JetPack 5.1 (L4T R35) and 6.x (L4T R36) via build arg.
# Note: L4T base images ship Python 3.8 which is too old for xllamacpp (>=3.10).
# We build Python 3.10 from source (deadsnakes PPA lacks ARM64 packages for focal).
ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies + build deps for Python 3.10 from source
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential cmake gcc g++ \
        ffmpeg libsndfile1 libopenblas-dev \
        curl wget unzip \
        zlib1g-dev libffi-dev libssl-dev libbz2-dev \
        libreadline-dev libsqlite3-dev liblzma-dev \
        libncurses5-dev libncursesw5-dev uuid-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source (~5 min on Jetson Orin NX)
ARG PYTHON_VERSION=3.10.16
RUN wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local 2>&1 | tail -5 && \
    make -j$(nproc) 2>&1 | tail -5 && \
    make altinstall && \
    cd / && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3

# Create virtual environment with Python 3.10
# Use --system-site-packages to inherit JetPack's CUDA bindings
RUN python3.10 -m venv --system-site-packages /opt/venv && \
    pip install --upgrade pip setuptools wheel

# Install uv for fast package management
RUN pip install uv

WORKDIR /app

# Install PyTorch with CUDA support for Jetson
# JetPack 6 (R36): torch 2.5 with CUDA from NVIDIA's index
# JetPack 5 (R35): torch 2.0 with CUDA from NVIDIA's index
# We try the community Jetson AI Lab index first (broader Python version support),
# then fall back to NVIDIA's official index, then worst-case PyPI.
ARG TORCH_INDEX=https://pypi.jetson-ai-lab.dev/simple/
RUN pip install torch torchvision torchaudio \
        --extra-index-url ${TORCH_INDEX} \
        --no-cache-dir 2>/dev/null || \
    pip install torch torchaudio --no-cache-dir || true

# Verify CUDA is available (non-fatal)
RUN python3 -c "import torch; print('PyTorch', torch.__version__); \
    print('CUDA:', torch.cuda.is_available()); \
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" \
    || echo "WARNING: torch CUDA not available, will use CPU"

# Install numpy and Cython for pkuseg
RUN pip install numpy Cython --no-cache-dir

# Copy and install requirements (skip packages without ARM64 wheels)
COPY cuda-requirements.txt .
RUN pip install -r cuda-requirements.txt --no-cache-dir 2>/dev/null || \
    (echo "Batch install failed, installing individually..." && \
     while IFS= read -r line || [ -n "$line" ]; do \
         line=$(echo "$line" | sed 's/#.*//;s/[[:space:]]*$//'); \
         [ -z "$line" ] && continue; \
         pip install "$line" --no-cache-dir 2>/dev/null || \
         echo "SKIP: $line (no ARM64 wheel)"; \
     done < cuda-requirements.txt)

# Install chatterbox-tts with --no-deps to avoid transformers pin
RUN pip install chatterbox-tts --no-deps --no-cache-dir 2>/dev/null || \
    echo "SKIP: chatterbox-tts (optional)"

# Build xllamacpp from source with CUDA for Jetson
# This is a compile step (~5-15 min on Jetson Orin NX)
RUN pip install cython setuptools wheel rust-setuptools 2>/dev/null || true
ARG CUDA_ARCH=87
RUN git clone --recursive https://github.com/xorbitsai/xllamacpp.git /tmp/xllamacpp && \
    cd /tmp/xllamacpp && \
    XLLAMACPP_BUILD_CUDA=1 \
    CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    pip install . --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/xllamacpp || \
    (echo "WARNING: xllamacpp source build failed — GGUF inference will not be available" && \
     rm -rf /tmp/xllamacpp)

# Copy application code
COPY . .

# Create outputs directory
RUN mkdir -p /app/outputs

# Jetson-appropriate defaults
ENV HOST=0.0.0.0 \
    TOKENIZERS_PARALLELISM=false \
    UVICORN_WORKERS=1 \
    CUDA_DOCKER_ARCH=all

EXPOSE 8091
CMD ["python", "start.py"]
