FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN --mount=type=cache,target=/var/cache/cuda/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential autotools-dev nfs-common pdsh cmake gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget ninja-build build-essential pkg-config ocl-icd-opencl-dev opencl-headers clinfo libclblast-dev libopenblas-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install pip cmake --upgrade --no-cache-dir
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_VULKAN=1
RUN CMAKE_ARGS="-DLLAMA_VULKAN=ON" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir deepspeed
COPY . .
RUN python3 download.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]
