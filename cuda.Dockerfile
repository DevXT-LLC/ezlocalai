FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN --mount=type=cache,target=/var/cache/cuda/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential autotools-dev nfs-common pdsh cmake gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget libopenblas-dev ninja-build build-essential pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install pip --upgrade --no-cache-dir && \
    pip install cmake deepspeed --no-cache-dir
WORKDIR /app
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_VULKAN=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN CMAKE_ARGS="-DLLAMA_VULKAN=ON" FORCE_CMAKE=1 pip install llama-cpp-python --verbose --force-reinstall --no-cache-dir
COPY . .
RUN python3 download.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]
