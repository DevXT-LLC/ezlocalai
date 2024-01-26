ARG LLAMACPP_IMAGE="nvidia/cuda:12.1.1-devel-ubuntu22.04"
FROM ${LLAMACPP_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \ 
    && ln -s /usr/bin/python3 /usr/bin/python \
    && pip install --upgrade pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir deepspeed

RUN python3 local_llm/CTTS.py
RUN python3 local_llm/STT.py

EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["/app/start.sh"]
