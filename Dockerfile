ARG LLAMACPP_IMAGE="nvidia/cuda:12.1.1-devel-ubuntu22.04"
FROM ${LLAMACPP_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
WORKDIR /app
COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN python3 -m pip install --upgrade pip cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context
RUN python3 -m pip install -r requirements.txt

EXPOSE 8091
RUN chmod +x start.sh
ENTRYPOINT ["/app/start.sh"]
