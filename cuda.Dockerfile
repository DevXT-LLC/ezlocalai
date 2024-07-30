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
    LLAMA_CUBLAS=1 \ 
    GGML_CUDA=1
COPY cuda-requirements.txt .
RUN pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.83 --no-cache-dir \
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
