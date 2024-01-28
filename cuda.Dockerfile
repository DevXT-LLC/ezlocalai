FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git build-essential gcc g++ portaudio19-dev ffmpeg libportaudio2 libasound-dev python3 python3-pip python3-venv gcc wget ocl-icd-opencl-dev opencl-headers clinfo libclblast-dev libopenblas-dev pkg-config ninja-build && \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \ 
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
RUN python3 -m pip install --upgrade pip --no-cache-dir
RUN python3 -m venv venv
RUN venv/bin/pip install --no-cache-dir -r requirements.txt
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" venv/bin/pip install llama-cpp-python --no-cache-dir && \
    venv/bin/pip install --no-cache-dir deepspeed
RUN venv/bin/python3 local_llm/CTTS.py
RUN venv/bin/python3 local_llm/STT.py

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y ffmpeg libportaudio2 libasound-dev python3 python3-pip python3-venv && \
    pip install --upgrade pip --no-cache-dir && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV HOST 0.0.0.0
ENV CUDA_DOCKER_ARCH=all
COPY --from=builder /app /app
EXPOSE 8091
CMD "/app/venv/bin/python3 server.py"