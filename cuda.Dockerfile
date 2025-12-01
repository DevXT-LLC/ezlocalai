FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
       git build-essential cmake gcc g++ ninja-build \
       portaudio19-dev ffmpeg libportaudio2 libasound-dev \
       python3 python3-pip python3-venv wget ocl-icd-opencl-dev opencl-headers \
       clinfo libclblast-dev libopenblas-dev python3-dev unzip curl && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*
# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
RUN uv pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# Install numpy and Cython for pkuseg (required by chatterbox-tts)
RUN uv pip install numpy==1.25.2 Cython
# Install pkuseg separately (required by chatterbox-tts)
RUN uv pip install pkuseg==0.0.25 --no-build-isolation
COPY cuda-requirements.txt .
RUN uv pip install -r cuda-requirements.txt
# Install chatterbox-tts with --no-deps to bypass transformers==4.46.3 pin
# This allows us to use transformers>=4.53.0 for security fixes
RUN uv pip install chatterbox-tts --no-deps
ENV HOST=0.0.0.0 \
    CUDA_DOCKER_ARCH=all \
    CUDAVER=12.8.1
# Install xllamacpp with CUDA 12.8 support
RUN uv pip install xllamacpp --reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128
COPY . .
EXPOSE 8091
CMD uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers