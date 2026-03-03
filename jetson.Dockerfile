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
    PATH="/opt/venv/bin:/usr/local/cuda/bin:/root/.local/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies + build deps for Python 3.10 from source
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential gcc g++ \
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
    pip install --upgrade pip setuptools wheel && \
    pip install cmake --no-cache-dir && \
    cmake --version

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

# Install numpy, Cython, and build tools needed by xllamacpp
RUN pip install numpy Cython setuptools wheel --no-cache-dir

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
# xllamacpp is a HARD requirement (LLM.py imports it directly) — fail build if it can't be built
# Uses xllamacpp's own build system: scripts/setup.sh (cmake + copy_libs) → setup.py
# Requires Rust for llguidance static library
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . "$HOME/.cargo/env" && rustc --version
ENV PATH="/root/.cargo/bin:${PATH}"

ARG CUDA_ARCH=87
ENV XLLAMACPP_BUILD_CUDA=1 \
    CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    CUDA_ARCHITECTURES=${CUDA_ARCH} \
    CUDA_PATH=/usr/local/cuda \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LIBRARY_PATH}"

# Build xllamacpp: run cmake directly (setup.sh swallows errors) with FA disabled for CUDA 11.x compat
RUN set -e && \
    echo "=== Building xllamacpp from source (CUDA arch: ${CUDA_ARCH}) ===" && \
    echo "nvcc: $(nvcc --version 2>&1 | tail -1)" && \
    git clone --recursive https://github.com/xorbitsai/xllamacpp.git /tmp/xllamacpp && \
    cd /tmp/xllamacpp/thirdparty/llama.cpp && mkdir -p build && cd build && \
    echo "--- Configuring cmake ---" && \
    cmake .. \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DLLAMA_CURL=OFF \
        -DLLAMA_LLGUIDANCE=ON \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_BUILD_BORINGSSL=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} && \
    echo "--- Building cmake targets ---" && \
    cmake --build . --config Release -j$(nproc) \
        --target common llama ggml ggml-cpu ggml-cuda mtmd cpp-httplib server-context llama-server && \
    echo "--- Verifying generated headers ---" && \
    ls tools/server/index.html.gz.hpp && \
    echo "--- Copying libs ---" && \
    cd /tmp/xllamacpp && \
    python scripts/copy_libs.py && \
    echo "--- Verifying static libs ---" && \
    ls src/llama.cpp/lib/libllama.a src/llama.cpp/lib/libggml-cuda.a && \
    echo "--- Installing xllamacpp Python package ---" && \
    pip install . --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/xllamacpp && \
    python -c "import importlib.util; print('xllamacpp installed:', importlib.util.find_spec('xllamacpp') is not None)"

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
