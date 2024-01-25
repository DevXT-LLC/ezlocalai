ARG LLAMACPP_IMAGE="nvidia/cuda:12.1.1-devel-ubuntu22.04"
FROM ${LLAMACPP_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential portaudio19-dev \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \ 
    && ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app
COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN python3 -m pip install --no-cache-dir --upgrade pip cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context deepspeed
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# If LLAMACPP_IMAGE contains "cuda", we need to install requirements-cuda.txt

EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["/app/start.sh"]
