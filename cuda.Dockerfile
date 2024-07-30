FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential cmake gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip wget ocl-icd-opencl-dev opencl-headers clinfo libclblast-dev libopenblas-dev ninja-build python3.10-dev && \
    mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/* && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade cmake scikit-build setuptools wheel --no-cache-dir
WORKDIR /app
ENV HOST=0.0.0.0 \
    CUDA_DOCKER_ARCH=all \
    LLAMA_CUBLAS=1
COPY cuda-requirements.txt .
RUN pip install https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2B45fff310c8-cp310-cp310-linux_x86_64.whl && \
    pip install https://download.pytorch.org/whl/nightly/cu121/torch-2.4.0.dev20240603%2Bcu121-cp310-cp310-linux_x86_64.whl && \
    pip install torchaudio && \
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123 && \
    pip install --no-cache-dir -r cuda-requirements.txt
RUN git clone https://github.com/Josh-XT/DeepSeek-VL deepseek && \
    cd deepseek && \
    pip install --no-cache-dir -e . && \
    cd ..
RUN pip install spacy==3.7.4 && \
    python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8091
CMD uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers
