#!/bin/sh

set -e

LOCAL_LLM_API_KEY=${LOCAL_LLM_API_KEY:-}
THREADS=${THREADS:-($(nproc) - 1)}
GPU_LAYERS=${GPU_LAYERS:-0}
MAIN_GPU=${MAIN_GPU:-0}
make
uvicorn app:app --host 0.0.0.0 --port 8091 --workers 4 --proxy-headers