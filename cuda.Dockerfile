FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN --mount=type=cache,target=/var/cache/cuda/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential autotools-dev nfs-common pdsh cmake gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget libopenblas-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -U pip setuptools
WORKDIR /app
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_VULKAN=1
RUN pip install --no-cache-dir fastapi uvicorn pydantic==2.5.3 requests==2.31.0 tiktoken==0.5.2 python-dotenv==1.0.1 beautifulsoup4==4.12.3 whisper-cpp-pybind==0.1.3 pydub==0.25.1 ffmpeg==1.4 transformers==4.37.1 TTS==0.22.0 sounddevice==0.4.6 torch==2.1.2 torchaudio==2.1.2 && \
    CMAKE_ARGS="-DLLAMA_VULKAN=1" pip install --no-cache-dir llama-cpp-python
COPY download.py .
RUN --mount=type=cache,target=/var/cache/models,sharing=shared \
    python3 download.py
RUN git clone https://github.com/microsoft/DeepSpeed.git DeepSpeed
RUN cd DeepSpeed && \
    git checkout . && \
    git checkout master && \
    ./install.sh --pip_sudo
RUN rm -rf DeepSpeed
RUN python -c "import deepspeed; print(deepspeed.__version__)"

COPY . .
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]
