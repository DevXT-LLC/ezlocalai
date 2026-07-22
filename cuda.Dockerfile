FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH"
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
       git build-essential cmake gcc g++ ninja-build \
       portaudio19-dev ffmpeg libportaudio2 libasound-dev \
       wget ocl-icd-opencl-dev opencl-headers sox libsox-dev \
       clinfo libclblast-dev libopenblas-dev unzip curl && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*
# Install uv and create venv with Python 3.12
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv /opt/venv --python 3.12
WORKDIR /app
# Use PyTorch 2.9.1 which is built against cuDNN 9.10.2
# Install nvidia-cudnn-cu12==9.10.2.21 to get matching cuDNN libraries (overrides system cuDNN 9.8.0)
RUN uv pip install torch==2.9.1+cu128 torchaudio==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install nvidia-cudnn-cu12==9.10.2.21 && \
    uv pip uninstall torchcodec -y 2>/dev/null || true
# Install numpy and Cython for packages that build native extensions
# numpy>=1.26.0 required for Python 3.12 compatibility
RUN uv pip install "numpy>=1.26.0,<2.5" Cython "setuptools>=78.1.1"
COPY cuda-requirements.txt .
RUN uv pip install -r cuda-requirements.txt
RUN uv pip install qwen-tts==0.1.1 --no-deps
# ezlocalai imports gTTS as a library. Install it outside the main solve because
# its CLI click<8.2 constraint conflicts with the current Hugging Face stack.
RUN uv pip install "gTTS>=2.4.0" --no-deps
# Install esp-ppq with --no-deps to bypass onnx<1.18.0 pin
# This allows us to use onnx>=1.21.0 for security fixes
RUN uv pip install esp-ppq --no-deps
ENV HOST=0.0.0.0 \
    CUDA_DOCKER_ARCH=all \
    CUDAVER=12.8.1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models
# Install xllamacpp with CUDA 12.8 support (compatible with CUDA 12.9)
RUN uv pip install xllamacpp==2026.7.10068 --reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128
COPY . .
EXPOSE 8091
# Use start.py which runs precache once, then starts uvicorn workers
CMD ["python", "start.py"]
