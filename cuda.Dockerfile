FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget libopenblas-dev pkg-config ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_BLAS_VENDOR=OpenBLAS
ENV LLAMA_BLAS=ON
RUN pip install -U pip setuptools cmake
RUN pip install fastapi uvicorn pydantic==2.5.3 requests==2.31.0 tiktoken==0.5.2 python-dotenv==1.0.1 beautifulsoup4==4.12.3 whisper-cpp-pybind==0.1.3 pydub==0.25.1 ffmpeg==1.4 transformers==4.37.1 TTS==0.22.0 sounddevice==0.4.6 torch==2.1.2 torchaudio==2.1.2 deepspeed==0.13.1
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install -U llama-cpp-python --force-reinstall
RUN python3 local_llm/CTTS.py
RUN python3 local_llm/STT.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]
