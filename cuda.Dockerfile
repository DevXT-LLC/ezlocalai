FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
RUN apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget libopenblas-dev pkg-config ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install -U pip setuptools cmake scikit-build
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install deepspeed && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -U llama-cpp-python --force-reinstall
COPY . .
RUN python3 local_llm/CTTS.py
RUN python3 local_llm/STT.py
EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["sh", "-c", "./start.sh"]