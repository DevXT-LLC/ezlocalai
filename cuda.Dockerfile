FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
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
RUN pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# Install numpy and Cython for pkuseg (required by chatterbox-tts)
RUN pip install numpy==1.25.2 Cython --no-cache-dir
# Install pkuseg separately (required by chatterbox-tts)
RUN pip install pkuseg==0.0.25 --no-build-isolation --no-cache-dir
COPY cuda-requirements.txt .
RUN pip install --no-cache-dir -r cuda-requirements.txt
# Install chatterbox-tts with --no-deps to bypass transformers==4.46.3 pin
# This allows us to use transformers>=4.53.0 for security fixes
RUN pip install chatterbox-tts --no-deps --no-cache-dir
ENV HOST=0.0.0.0 \
    CUDA_DOCKER_ARCH=all \
    CUDAVER=12.8.1
# Install xllamacpp with CUDA 12.8 support
RUN pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --no-cache-dir
COPY . .
EXPOSE 8091
CMD uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers