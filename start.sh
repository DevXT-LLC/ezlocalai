#!/bin/sh
set -e
CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
if [ "$CUDA_DOCKER_ARCH" = "all" ]; then
    CUDA_DOCKER_ARCH=all
    LLAMA_CUBLAS=1
    CMAKE_ARGS="-DLLAMA_CUBLAS=on"
    export CUDA_DOCKER_ARCH
    export LLAMA_CUBLAS
    export CMAKE_ARGS
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
fi
uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers