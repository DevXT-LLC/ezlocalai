#!/usr/bin/env bash
# Compile only the changed CUDA translation unit and syntax-check the operator cases.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source=${EXPERIMENTAL_PREFILL_SOURCE:-/home/josh/repos/xtsys/ezlocalai/build/validation/llama-v090-cuda-stage/source}
python=${PYTHON:-/home/josh/repos/xtsys/.venv/bin/python}
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}

EXPERIMENTAL_PREFILL_SOURCE="$source" "$python" "$repo/benchmarks/experimental-prefill-gdn-slice64-test.py"
work=$(mktemp -d /tmp/experimental-prefill-gdn-build.XXXXXX)
printf 'Isolated build and logs: %s\n' "$work"
mkdir -p "$work/source/ggml/src/ggml-cuda" "$work/source/tests"
cp "$source/ggml/src/ggml-cuda/gated-delta-net-mma.cu" "$work/source/ggml/src/ggml-cuda/"
cp "$source/tests/test-backend-ops.cpp" "$work/source/tests/"
(
    cd "$work/source"
    git apply --check "$repo/patches/experimental-prefill-gdn-slice64.patch"
    git apply "$repo/patches/experimental-prefill-gdn-slice64.patch"
    git apply --check "$repo/patches/experimental-prefill-gdn-slice64-tests.patch"
    git apply "$repo/patches/experimental-prefill-gdn-slice64-tests.patch"
)

"$nvcc" --version | tee "$work/compiler.log"
"$nvcc" -c "$work/source/ggml/src/ggml-cuda/gated-delta-net-mma.cu" \
    -o "$work/gdn.o" \
    -I "$source/ggml/src" -I "$source/ggml/src/ggml-cuda" -I "$source/ggml/include" \
    -std=c++17 -O3 -DNDEBUG -DGGML_CUDA_USE_GRAPHS -Xcompiler=-fPIC \
    -use_fast_math -extended-lambda \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120a,code=sm_120a \
    --ptxas-options=-v 2>&1 | tee "$work/compile.log"

g++ -std=c++17 -fsyntax-only \
    -I "$source/ggml/include" -I "$source/ggml/src" -I "$source/tests" \
    "$work/source/tests/test-backend-ops.cpp" 2>&1 | tee "$work/operator-syntax.log"
printf 'PASS: object compile and operator syntax only; no GPU execution. Artifacts: %s\n' "$work"
