FROM python:3.10-bookworm
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update --fix-missing  && apt-get upgrade -y && \
    apt-get install -y --fix-missing --no-install-recommends git build-essential cmake gcc g++ ninja-build libopenblas-dev portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip gcc wget curl sox libsox-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip --no-cache-dir
WORKDIR /app

ARG ACESTEP_CPP_REF=master
RUN git clone --depth 1 --recurse-submodules https://github.com/ServeurpersoCom/acestep.cpp.git /opt/acestep.cpp && \
    cd /opt/acestep.cpp && \
    git checkout "$ACESTEP_CPP_REF" && \
    git submodule update --init --recursive && \
    cmake -B build -DGGML_BLAS=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release --parallel "$(nproc)" --target ace-server

# Install PyTorch CPU version
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install numpy and Cython for packages that build native extensions
RUN pip install numpy==1.25.2 Cython "setuptools>=78.1.1" --no-cache-dir

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install qwen-tts==0.1.1 --no-deps --no-cache-dir
RUN python3 -m pip install "gTTS>=2.4.0" --no-deps --no-cache-dir

# Install esp-ppq with --no-deps to bypass onnx<1.18.0 pin
# This allows us to use onnx>=1.21.0 for security fixes
RUN pip install esp-ppq --no-deps --no-cache-dir

# Install xllamacpp CPU version
RUN pip install xllamacpp==2026.8.10566 --force-reinstall --no-cache-dir

COPY . .
ENV HOST=0.0.0.0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    ACE_STEP_BIN=/opt/acestep.cpp/build/ace-server
EXPOSE 8091
# Use start.py which runs precache once, then starts uvicorn workers
CMD ["python", "start.py"]
