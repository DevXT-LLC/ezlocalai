# GB10 (Grace Blackwell, ARM64) Dockerfile for ezLocalai
# Base: NVIDIA CUDA 12.8 on Ubuntu 24.04 (ARM64 variant auto-selected by Docker)
# Build ON the GB10 itself: docker compose -f docker-compose-gb10.yml build
#
# The GB10 has compute capability 12.0 (Blackwell), unified memory, CUDA 13.0 driver.
# We use a 12.8 container since the driver is backward compatible.
#
# Prerequisites on host:
#   - NVIDIA Container Toolkit installed and configured
#   - Docker with GPU support: docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

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

# Build ACE Step for GB10 (Blackwell = compute capability 120)
ARG ACESTEP_CPP_REF=master
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    git clone --depth 1 --recurse-submodules https://github.com/ServeurpersoCom/acestep.cpp.git /opt/acestep.cpp && \
    cmake -S /opt/acestep.cpp -B /opt/acestep.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="120-real" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" && \
    cmake --build /opt/acestep.cpp/build --target ace-server --parallel $(nproc)

# Install base Python dependencies
COPY cuda-requirements.txt .
RUN uv pip install -r cuda-requirements.txt

# gTTS has a click<8.2 constraint that conflicts with huggingface-hub, install without deps
RUN uv pip install gTTS --no-deps

# esp-ppq has an onnx<1.18.0 pin conflict, install without deps
RUN uv pip install esp-ppq --no-deps

ENV HOST=0.0.0.0 \
    CUDA_DOCKER_ARCH=all \
    CUDAVER=12.8.1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    ACE_STEP_BIN=/opt/acestep.cpp/build/ace-server \
    SDCPP_BIN=/opt/stable-diffusion.cpp/build/bin/sd-cli

# Install xllamacpp with CUDA 12.8 support for ARM64 (Blackwell)
# Try pre-built wheel first; fall back to source build if no ARM64 wheel available
RUN uv pip install xllamacpp==2026.9.10809 --reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 2>/dev/null || \
    (echo "No pre-built ARM64 wheel, building from source..." && \
     uv pip install pip && \
     COPY scripts/build_xllamacpp.py /opt/ezlocalai-native/scripts/build_xllamacpp.py && \
     COPY native/patches/dflash-pinned-image-positions.patch /opt/ezlocalai-native/native/patches/dflash-pinned-image-positions.patch && \
     ARG XLLAMACPP_BUILD_JOBS=4 && \
     export PATH="/root/.cargo/bin:$PATH" && \
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path && \
     LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH" \
     CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
     python /opt/ezlocalai-native/scripts/build_xllamacpp.py --cuda --install \
         --jobs "${XLLAMACPP_BUILD_JOBS}" --cuda-architectures "120-real" \
         --source-dir /opt/xllamacpp-build/source --wheel-dir /opt/xllamacpp-build/wheels)

# Build TTS native component for GB10 (Blackwell = 120)
COPY native/tts /opt/ezlocalai-tts
ARG TTS_BUILD_JOBS=4
RUN cmake -S /opt/ezlocalai-tts -B /opt/ezlocalai-tts/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="120" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" && \
    cmake --build /opt/ezlocalai-tts/build --target ezlocalai-tts --parallel "${TTS_BUILD_JOBS}"

COPY . .

EXPOSE 8091

CMD ["python", "start.py"]