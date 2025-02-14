FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
       git build-essential cmake gcc g++ ninja-build \
       portaudio19-dev ffmpeg libportaudio2 libasound-dev \
       python3 python3-pip wget ocl-icd-opencl-dev opencl-headers \
       clinfo libclblast-dev libopenblas-dev python3.10-dev unzip && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade cmake scikit-build setuptools wheel --no-cache-dir
WORKDIR /app
RUN pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN git clone https://github.com/Josh-XT/DeepSeek-VL deepseek && \
    cd deepseek && \
    pip install --no-cache-dir -e . && \
    cd ..
COPY cuda-requirements.txt .
RUN pip install --no-cache-dir -r cuda-requirements.txt
RUN pip install spacy==3.7.4 && \
    python -m spacy download en_core_web_sm
ENV HOST=0.0.0.0 \
    CUDA_DOCKER_ARCH=all \
    LLAMA_CUBLAS=1 \
    GGML_CUDA=on \
    CUDAVER=12.4.1 \
    AVXVER=basic
RUN CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_MMQ=ON -DGGML_AVX2=off -DGGML_FMA=off -DGGML_F16C=off -DCMAKE_CUDA_ARCHITECTURES=86;89" \
    pip install llama-cpp-python==0.3.7 --no-cache-dir
COPY . .
EXPOSE 8091
EXPOSE 8502
CMD streamlit run ui.py & uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers