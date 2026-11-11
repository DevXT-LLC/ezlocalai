#!/usr/bin/env bash
# Run inside the pinned compiler-only container; /validation is private scratch.
set -euo pipefail
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=/validation/source
export CMAKE_BUILD_PARALLEL_LEVEL=4

date -u +%FT%TZ | tee /validation/start-time.txt
/usr/local/cuda/bin/nvcc --version | tee /validation/compiler.log
g++ --version | tee -a /validation/compiler.log
cmake --version | tee -a /validation/compiler.log
if compgen -G '/dev/nvidia*' >/dev/null; then
    printf 'Refusing build: GPU device nodes are exposed.\n' >&2
    exit 1
fi
cp /app/build/llama.cpp/CMakeCache.txt /validation/production-CMakeCache.txt
cp /app/build/llama.cpp/ezlocalai-llama-cpp-version.txt /validation/production-version.txt
cp /app/build/llama.cpp/ezlocalai-llama-cpp-patches.txt /validation/production-patches.txt
cp /app/build/llama.cpp/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make /validation/production-cuda-flags.make
cp -a /app/build/llama.cpp/tools/ui/dist /validation/source/tools/ui/dist

cmake -S /validation/source -B /validation/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=OFF \
    -DGGML_CCACHE=OFF -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF \
    '-DCMAKE_CUDA_ARCHITECTURES=86-real;89-real;120-real' \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    2>&1 | tee /validation/configure.log

cmake --build /validation/build --target test-backend-ops llama-server --parallel 4 \
    2>&1 | tee /validation/build.log

test -x /validation/build/bin/test-backend-ops
test -x /validation/build/bin/llama-server
ldd /validation/build/bin/test-backend-ops | tee /validation/ldd-test-backend-ops.log
ldd /validation/build/bin/llama-server | tee /validation/ldd-llama-server.log
sha256sum /validation/build/bin/test-backend-ops /validation/build/bin/llama-server \
    /validation/build/bin/libggml-cuda.so* | tee /validation/artifact-sha256.txt
date -u +%FT%TZ | tee /validation/finish-time.txt
printf 'PASS: both targets built; no executable or GPU kernel was run.\n'
