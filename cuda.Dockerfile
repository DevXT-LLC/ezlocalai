FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN --mount=type=cache,target=/var/cache/cuda/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && \
    apt-get install -y --fix-missing --no-install-recommends build-essential cmake gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip wget vulkan-sdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install pip cmake --upgrade --no-cache-dir
WORKDIR /app
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_VULKAN=1
COPY . .
RUN CMAKE_ARGS="-DLLAMA_VULKAN=ON" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt
RUN python3 download.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]
