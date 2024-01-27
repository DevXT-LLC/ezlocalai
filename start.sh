#!/bin/sh

set -e

GPU_LAYERS=${GPU_LAYERS:-0}
# if GPU_LAYERS are greater than 0, then we need to set CMAKES_ARGS to enable CUDA
if [ $GPU_LAYERS -gt 0 ]; then
    CUDA_DOCKER_ARCH=all
    LLAMA_CUBLAS=1
    CMAKE_ARGS="-DLLAMA_CUBLAS=on"
    export CUDA_DOCKER_ARCH
    export LLAMA_CUBLAS
    export CMAKE_ARGS
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
fi
uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers