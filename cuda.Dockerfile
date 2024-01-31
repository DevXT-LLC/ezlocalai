FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN --mount=type=cache,target=/var/cache/cuda/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential autotools-dev nfs-common pdsh cmake gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget libopenblas-dev ninja-build build-essential pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install pip cmake --upgrade --no-cache-dir
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3 download.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]
