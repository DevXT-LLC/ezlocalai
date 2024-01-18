#!/bin/sh

set -e

LOCAL_LLM_API_KEY=${LOCAL_LLM_API_KEY:-}
THREADS=${THREADS:-($(nproc) - 1)}
GPU_LAYERS=${GPU_LAYERS:-0}
MAIN_GPU=${MAIN_GPU:-0}
export LOCAL_LLM_API_KEY
export THREADS
export GPU_LAYERS
export MAIN_GPU
# if GPU_LAYERS are greater than 0, then we need to set CMAKES_ARGS to enable CUDA
if [ $GPU_LAYERS -gt 0 ]; then
    CUDA_DOCKER_ARCH=all
    LLAMA_CUBLAS=1
    CMAKE_ARGS="-DLLAMA_CUBLAS=on"
    export CUDA_DOCKER_ARCH
    export LLAMA_CUBLAS
    export CMAKE_ARGS
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
fi
uvicorn app:app --host 0.0.0.0 --port 8091 --workers 4 --proxy-headers