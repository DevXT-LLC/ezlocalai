#!/bin/sh
set -e
CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
if [ "$CUDA_DOCKER_ARCH" = "all" ]; then
    CUDA_DOCKER_ARCH=all
    LLAMA_BLAS_VENDOR=OpenBLAS
    LLAMA_OPENBLAS=1
    export CUDA_DOCKER_ARCH
    export LLAMA_BLAS_VENDOR
    export LLAMA_VULKAN
    export CMAKE_ARGS
    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall
fi
uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers