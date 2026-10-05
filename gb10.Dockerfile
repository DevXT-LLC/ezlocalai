# GB10 (Grace Blackwell, ARM64) Dockerfile for ezLocalai
# Base: NVIDIA CUDA 13.0 on Ubuntu 24.04 (ARM64 variant auto-selected by Docker)
# Build ON the GB10 itself: docker compose -f docker-compose-gb10.yml build
#
# The GB10 has compute capability 12.1 (Blackwell), unified memory, CUDA 13.0 driver.
# Compile native sm_121 kernels to avoid first-request PTX compilation.
#
# Prerequisites on host:
#   - NVIDIA Container Toolkit installed and configured
#   - Docker with GPU support: docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH"

RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
       git build-essential cmake gcc g++ ninja-build \
       portaudio19-dev ffmpeg libportaudio2 libasound-dev \
       wget ocl-icd-opencl-dev opencl-headers sox libsox-dev \
       clinfo libclblast-dev libopenblas-dev unzip curl && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# Install uv and create venv with Python 3.12
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv /opt/venv --python 3.12

WORKDIR /app

# Build ACE Step for GB10 (Blackwell = compute capability 121)
ARG NATIVE_BUILD_JOBS=12
ARG ACESTEP_CPP_REF=master
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    git clone --depth 1 --recurse-submodules https://github.com/ServeurpersoCom/acestep.cpp.git /opt/acestep.cpp && \
    cmake -S /opt/acestep.cpp -B /opt/acestep.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="121-real" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" && \
    cmake --build /opt/acestep.cpp/build --target ace-server --parallel "${NATIVE_BUILD_JOBS}"

# Install a matched CUDA-enabled ARM64 torch/torchaudio pair. PyPI previously
# supplied torch transitively but no torchaudio, silently disabling Qwen TTS.
RUN uv pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Install base Python dependencies
COPY cuda-requirements.txt .
RUN uv pip install -r cuda-requirements.txt

# gTTS has a click<8.2 constraint that conflicts with huggingface-hub, install without deps
RUN uv pip install gTTS --no-deps

# esp-ppq has an onnx<1.18.0 pin conflict, install without deps
RUN uv pip install esp-ppq --no-deps

ENV HOST=0.0.0.0 \
    CUDA_DOCKER_ARCH=all \
    CUDAVER=13.0.2 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    ACE_STEP_BIN=/opt/acestep.cpp/build/ace-server \
    SDCPP_BIN=/opt/stable-diffusion.cpp/build/bin/sd-cli

# Build CUDA-enabled CTranslate2: the PyPI ARM64 wheel is CPU-only.
# Whisper needs cuDNN as well as CUDA. Pin both native and Python code together.
COPY native/patches/ctranslate2-gb10-cuda.patch /opt/ctranslate2-gb10-cuda.patch
ARG CTRANSLATE2_REF=d44d2d069eb88c7b7804da864c10c201501cb4a9
RUN git clone https://github.com/OpenNMT/CTranslate2.git /opt/ctranslate2-src && \
    cd /opt/ctranslate2-src && git checkout "${CTRANSLATE2_REF}" && \
    git submodule update --init --recursive --depth 1 && \
    git apply /opt/ctranslate2-gb10-cuda.patch && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP -DCUDA_ARCH_LIST="12.1" && \
    cmake --build build --parallel "${NATIVE_BUILD_JOBS}" && \
    cmake --install build && ldconfig && \
    CMAKE_BUILD_PARALLEL_LEVEL="${NATIVE_BUILD_JOBS}" \
        uv pip install ./python --no-deps --reinstall

# Build the pinned, patched CUDA binding for GB10. Do not accept an ARM64
# wheel with unknown GPU targets, or silently fall back to a CPU build.
RUN uv pip install pip
COPY scripts/build_xllamacpp.py /opt/ezlocalai-native/scripts/build_xllamacpp.py
COPY native/patches/dflash-pinned-image-positions.patch /opt/ezlocalai-native/native/patches/dflash-pinned-image-positions.patch
ARG XLLAMACPP_BUILD_JOBS=12
RUN --mount=type=cache,target=/opt/xllamacpp-build \
    --mount=type=cache,target=/root/.cargo \
    --mount=type=cache,target=/root/.rustup \
    export PATH="/root/.cargo/bin:$PATH" && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH" \
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    python /opt/ezlocalai-native/scripts/build_xllamacpp.py --cuda --install \
        --jobs "${XLLAMACPP_BUILD_JOBS}" --cuda-architectures "121-real" \
        --source-dir /opt/xllamacpp-build/source --wheel-dir /opt/xllamacpp-build/wheels

# Build TTS native component for GB10 (Blackwell = 121)
COPY native/tts /opt/ezlocalai-tts
ARG TTS_BUILD_JOBS=12
RUN cmake -S /opt/ezlocalai-tts -B /opt/ezlocalai-tts/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="121-real" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" && \
    cmake --build /opt/ezlocalai-tts/build --target ezlocalai-tts --parallel "${TTS_BUILD_JOBS}"

# The GB10 compose advertises image generation, so include its native backend.
ARG SDCPP_REF=c678dfe704a2230342376b46add9c8ca736a653d
RUN git clone --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git /opt/stable-diffusion.cpp && \
    cd /opt/stable-diffusion.cpp && git checkout "${SDCPP_REF}" && \
    git submodule update --init --recursive && \
    cmake -B build -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES="121-real" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" && \
    cmake --build build --parallel "${NATIVE_BUILD_JOBS}" --target sd-cli

# Catch missing/import-incompatible voice dependencies at build time.
RUN python -c "import torch, torchaudio, ctranslate2, xllamacpp; assert torch.version.cuda is not None" && \
    test -x /opt/ezlocalai-tts/build/bin/ezlocalai-tts && \
    test -x /opt/stable-diffusion.cpp/build/bin/sd-cli

COPY . .

EXPOSE 8091

CMD ["python", "start.py"]
