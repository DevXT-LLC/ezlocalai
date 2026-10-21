# Jetson Orin / JetPack 6.1 (L4T R36.4, CUDA 12.6, sm_87).
# Build on Orin for practical compile times. Older Nano/Xavier/JetPack releases
# require different dependency sets and are not covered by this image.
# LibTorch is built for sm_87: generic ARM/SBSA CUDA wheels do not establish
# Tegra binary compatibility. Python is used only by upstream build tooling.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0@sha256:34ccf0f3b63c6da9eee45f2e79de9bf7fdf3beda9abfd72bbf285ae9d40bb673 AS jetson-base

ARG EZLOCALAI_LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG EZLOCALAI_LLAMA_CPP_REF=620d2225cc436ab0b001e3200e590e3b286ef9ab
# Keep CPU code portable when building under ARM emulation.
# Every Orin module (AGX, NX, Nano) uses Cortex-A78AE cores (Armv8.2-A with
# dot product and FP16). Name them explicitly: GGML_NATIVE under QEMU would
# detect the emulator's CPU, and the plain armv8-a baseline leaves them unused.
ARG EZLOCALAI_LLAMA_CPP_CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DCMAKE_CUDA_ARCHITECTURES=87-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
ARG EZLOCALAI_LLAMA_CPP_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    EZLOCALAI_NATIVE_BACKEND=llama.cpp \
    EZLOCALAI_LLAMA_CPP_REPO=${EZLOCALAI_LLAMA_CPP_REPO} \
    EZLOCALAI_LLAMA_CPP_REF=${EZLOCALAI_LLAMA_CPP_REF} \
    EZLOCALAI_LLAMA_CPP_CMAKE_ARGS=${EZLOCALAI_LLAMA_CPP_CMAKE_ARGS} \
    EZLOCALAI_LLAMA_CPP_JOBS=${EZLOCALAI_LLAMA_CPP_JOBS} \
    HOST=0.0.0.0 \
    PORT=8091 \
    MODELS_PATH=/app/models \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    LLAMA_CACHE=/app/models/.llama.cpp \
    LLAMA_CPP_FLASH_ATTENTION=true \
    KV_CACHE_TYPE=q4_0

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++ \
        gcc \
        git \
        libcurl4-openssl-dev \
        libssl-dev \
        zlib1g-dev \
        libbrotli-dev \
        libicu-dev \
        libpng-dev \
        libjpeg-dev \
        libwebp-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libopenblas-dev \
        libportaudio2 \
        make \
        ninja-build \
        poppler-utils \
        pkg-config \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# JetPack 6.1 ships an older CMake than LibTorch 2.9.1 requires.
RUN curl -fL https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-aarch64.tar.gz -o /tmp/cmake.tar.gz && \
    echo "609735983e3bdf24b6ab379d918458d64196fe72b98226f62dd5e9fe7b2997cc  /tmp/cmake.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/cmake.tar.gz --strip-components=1 -C /usr/local && \
    rm /tmp/cmake.tar.gz
RUN curl -fL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz -o /tmp/json.tar.xz && \
    echo "d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d  /tmp/json.tar.xz" | sha256sum -c - && \
    tar -xJf /tmp/json.tar.xz -C /tmp && \
    cmake -S /tmp/json -B /tmp/json/build -DJSON_BuildTests=OFF && \
    cmake --install /tmp/json/build && rm -rf /tmp/json /tmp/json.tar.xz
WORKDIR /app

FROM jetson-base AS engines-base
ARG EZLOCALAI_LLAMA_CPP_JOBS=2
# Ubuntu 22.04's FFmpeg 4.4 lacks the channel-layout/resampler API used by
# native Whisper. Build before OpenCV so both use the same media libraries.
# --disable-autodetect also drops zlib (the PNG codec, needed for video frame
# export and image decoding) and pthreads, so enable both explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libx264-dev libmp3lame-dev && rm -rf /var/lib/apt/lists/* && \
    curl -fL https://ffmpeg.org/releases/ffmpeg-6.1.2.tar.xz -o /tmp/ffmpeg.tar.xz && \
    echo "3b624649725ecdc565c903ca6643d41f33bd49239922e45c9b1442c63dca4e38  /tmp/ffmpeg.tar.xz" | sha256sum -c - && \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp && \
    cd /tmp/ffmpeg-6.1.2 && \
    ./configure --prefix=/usr/local --enable-shared --disable-static \
        --enable-gpl --enable-libx264 --enable-libmp3lame \
        --disable-doc --disable-debug --disable-autodetect \
        --enable-zlib --enable-pthreads && \
    make -j"${EZLOCALAI_LLAMA_CPP_JOBS}" && make install && ldconfig && \
    /usr/local/bin/ffmpeg -hide_banner -encoders | grep -qE '^ V[^ ]* png ' && \
    /usr/local/bin/ffmpeg -hide_banner -decoders | grep -qE '^ V[^ ]* png ' && \
    rm -rf /tmp/ffmpeg-6.1.2 /tmp/ffmpeg.tar.xz
RUN curl -fL https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz -o /tmp/opencv.tar.gz && \
    echo "1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d  /tmp/opencv.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/opencv.tar.gz -C /tmp && \
    cmake -S /tmp/opencv-4.13.0 -B /tmp/opencv-build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIST=core,imgproc,imgcodecs,videoio -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python2=OFF -DBUILD_JAVA=OFF -DWITH_IPP=OFF -DWITH_OPENCL=OFF \
        -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_ITT=OFF \
        -DWITH_OPENEXR=OFF -DWITH_OPENJPEG=OFF -DWITH_AVIF=OFF && \
    cmake --build /tmp/opencv-build --parallel "${EZLOCALAI_LLAMA_CPP_JOBS}" && \
    cmake --install /tmp/opencv-build && ldconfig && \
    rm -rf /tmp/opencv.tar.gz /tmp/opencv-4.13.0 /tmp/opencv-build

FROM jetson-base AS audio-dependencies
COPY cmake/CudaAudioDependencies.cmake cmake/jetson-audio-dependencies.json /tmp/audio-dependencies/
RUN --mount=type=cache,id=ezlocalai-audio-jetson,target=/var/cache/ezlocalai-audio \
    cmake -DDESTINATION=/opt/ezlocalai/deps -DDOWNLOAD_CACHE=/var/cache/ezlocalai-audio \
        -DMANIFEST=/tmp/audio-dependencies/jetson-audio-dependencies.json \
        -DCTRANSLATE2_INCLUDE_DIR=/opt/ezlocalai/ctranslate2/include \
        -DCTRANSLATE2_LIBRARY=/opt/ezlocalai/ctranslate2/lib/libctranslate2.so.4 \
        -P /tmp/audio-dependencies/CudaAudioDependencies.cmake

FROM jetson-base AS storyboard-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype-dev libharfbuzz-dev libfribidi-dev fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
COPY cmake/StoryboardDependencies.cmake cmake/gb10-storyboard-dependencies.json /tmp/storyboard-dependencies/
RUN --mount=type=cache,id=ezlocalai-storyboard-jetson,target=/var/cache/ezlocalai-storyboard \
    cmake -DDESTINATION=/opt/ezlocalai/storyboard -DDOWNLOAD_CACHE=/var/cache/ezlocalai-storyboard \
        -DMANIFEST=/tmp/storyboard-dependencies/gb10-storyboard-dependencies.json \
        -P /tmp/storyboard-dependencies/StoryboardDependencies.cmake

FROM jetson-base AS torch-libraries
ARG EZLOCALAI_TORCH_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-setuptools && \
    rm -rf /var/lib/apt/lists/*
# Ubuntu's typing_extensions lacks assert_never, used by pinned torchgen.
# These packages stay in the upstream compiler stage.
RUN python3 -m pip install --no-cache-dir \
        typing_extensions==4.15.0 PyYAML==6.0.3 sympy==1.13.3 \
        packaging==25.0 jinja2==3.1.6 numpy==2.2.6
# Match the version used by all platform policy captures. No Python bindings,
# distributed runtime or architecture-incompatible precompiled kernels are used.
RUN git init /tmp/pytorch && \
    git -C /tmp/pytorch remote add origin https://github.com/pytorch/pytorch.git && \
    git -C /tmp/pytorch fetch --depth 1 origin d38164a545b4a4e4e0cf73ce67173f70574890b6 && \
    git -C /tmp/pytorch checkout --detach FETCH_HEAD && \
    git -C /tmp/pytorch submodule update --init --recursive --depth 1
# Retain intermediate objects across retries; only installed native artifacts
# are copied into the worker image. Lock the cache against simultaneous builds.
RUN --mount=type=cache,id=ezlocalai-jetson-r36.4-torch-2.9.1-sm87,target=/tmp/pytorch/build,sharing=locked \
    PYTORCH_BUILD_VERSION=2.9.1 cmake -S /tmp/pytorch -B /tmp/pytorch/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ezlocalai/deps/torch \
        -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF -DBUILD_BINARY=OFF \
        -DUSE_CUDA=ON -DUSE_CUDNN=ON -DTORCH_CUDA_ARCH_LIST=8.7 \
        -DUSE_DISTRIBUTED=OFF -DUSE_KINETO=OFF -DUSE_CUSPARSELT=OFF \
        -DUSE_FLASH_ATTENTION=OFF -DUSE_MEM_EFF_ATTENTION=OFF \
        -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DBLAS=OpenBLAS \
        -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs && \
    cmake --build /tmp/pytorch/build --parallel "${EZLOCALAI_TORCH_JOBS}" && \
    cmake --install /tmp/pytorch/build && \
    mkdir -p /opt/ezlocalai/deps/torch/licenses && \
    cp /tmp/pytorch/LICENSE /opt/ezlocalai/deps/torch/licenses/

FROM engines-base AS ctranslate2
ARG EZLOCALAI_CTRANSLATE2_REF=d44d2d069eb88c7b7804da864c10c201501cb4a9
ARG EZLOCALAI_CTRANSLATE2_CUDA_ARCH=8.7
COPY patches/ctranslate2-gb10-cuda.patch /tmp/ctranslate2-gb10-cuda.patch
RUN git clone https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 && \
    git -C /tmp/ctranslate2 checkout "${EZLOCALAI_CTRANSLATE2_REF}" && \
    git -C /tmp/ctranslate2 submodule update --init --recursive --depth 1 && \
    git -C /tmp/ctranslate2 apply /tmp/ctranslate2-gb10-cuda.patch && \
    cmake -S /tmp/ctranslate2 -B /tmp/ctranslate2/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/ezlocalai/ctranslate2 -DBUILD_CLI=OFF \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
        -DCUDA_ARCH_LIST="${EZLOCALAI_CTRANSLATE2_CUDA_ARCH}" \
        -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs && \
    cmake --build /tmp/ctranslate2/build --parallel "${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}" && \
    cmake --install /tmp/ctranslate2/build && \
    mkdir -p /opt/ezlocalai/ctranslate2/licenses && cp /tmp/ctranslate2/LICENSE /opt/ezlocalai/ctranslate2/licenses/ && \
    rm -rf /tmp/ctranslate2

FROM engines-base AS diffusion-engine
ARG EZLOCALAI_SDCPP_REF=c678dfe704a2230342376b46add9c8ca736a653d
ARG EZLOCALAI_MEDIA_CUDA_ARCHITECTURES=87-real
ARG EZLOCALAI_MEDIA_CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"
COPY patches/sdcpp-ltx-conditions.patch patches/sdcpp-ltx-diffusers.patch /opt/ezlocalai-patches/
COPY patches/sdcpp-ltx-timesteps.patch /opt/ezlocalai-patches/
COPY patches/sdcpp-ltx-nf4.patch /opt/ezlocalai-patches/
COPY include/ezlocalai/nf4.hpp include/ezlocalai/torch_cuda_mean.hpp /opt/stable-diffusion.cpp/src/ezlocalai/
COPY patches/sdcpp-media-worker.patch /opt/ezlocalai-patches/
COPY engine/sd_cli_worker.hpp /opt/stable-diffusion.cpp/examples/cli/
COPY include/ezlocalai/media_worker_protocol.hpp /opt/stable-diffusion.cpp/examples/cli/ezlocalai/
COPY patches/sdcpp-nf4-storage.patch /opt/ezlocalai-patches/
COPY engine/nf4_ggml.hpp engine/nf4_cuda.cuh /opt/stable-diffusion.cpp/ggml/src/ggml-cuda/
COPY include/ezlocalai/nf4.hpp include/ezlocalai/nf4_storage.hpp include/ezlocalai/torch_cuda_mean.hpp /opt/stable-diffusion.cpp/ggml/src/ezlocalai/
COPY patches/sdcpp-ltx-nf4-packed.patch /opt/ezlocalai-patches/
COPY engine/nf4_ggml.hpp /opt/stable-diffusion.cpp/src/
COPY include/ezlocalai/nf4_storage.hpp /opt/stable-diffusion.cpp/src/ezlocalai/
COPY patches/sdcpp-ltx-gemma-residency.patch /opt/ezlocalai-patches/
COPY patches/sdcpp-ltx-offload-policy.patch /opt/ezlocalai-patches/
COPY patches/sdcpp-ltx-shared-cuts.patch /opt/ezlocalai-patches/
COPY patches/sdcpp-ltx-connector-pass.patch /opt/ezlocalai-patches/
COPY engine/ltx_condition_cache.hpp /opt/stable-diffusion.cpp/src/
RUN git init /opt/stable-diffusion.cpp && \
    git -C /opt/stable-diffusion.cpp remote add origin https://github.com/leejet/stable-diffusion.cpp.git && \
    git -C /opt/stable-diffusion.cpp fetch --depth 1 origin "${EZLOCALAI_SDCPP_REF}" && \
    git -C /opt/stable-diffusion.cpp checkout --detach FETCH_HEAD && \
    git -C /opt/stable-diffusion.cpp submodule update --init --recursive --depth 1 && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-conditions.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-diffusers.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-timesteps.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-nf4.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-media-worker.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-nf4-storage.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-nf4-packed.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-gemma-residency.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-offload-policy.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-shared-cuts.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-connector-pass.patch && \
    cmake -S /opt/stable-diffusion.cpp -B /opt/stable-diffusion.cpp/build \
        -DSD_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES="${EZLOCALAI_MEDIA_CUDA_ARCHITECTURES}" ${EZLOCALAI_MEDIA_CMAKE_ARGS} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs && \
    cmake --build /opt/stable-diffusion.cpp/build --target sd-cli --parallel "${EZLOCALAI_LLAMA_CPP_JOBS}"
FROM engines-base AS music-engine
ARG EZLOCALAI_MEDIA_CUDA_ARCHITECTURES=87-real
ARG EZLOCALAI_MEDIA_CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"
ARG EZLOCALAI_ACESTEP_REF=5b88ded7919285d5ad5b1e86ffe3378314f305db
RUN git init /opt/acestep.cpp && \
    git -C /opt/acestep.cpp remote add origin https://github.com/ServeurpersoCom/acestep.cpp.git && \
    git -C /opt/acestep.cpp fetch --depth 1 origin "${EZLOCALAI_ACESTEP_REF}" && \
    git -C /opt/acestep.cpp checkout --detach FETCH_HEAD && \
    git -C /opt/acestep.cpp submodule update --init --recursive --depth 1 && \
    cmake -S /opt/acestep.cpp -B /opt/acestep.cpp/build \
        -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES="${EZLOCALAI_MEDIA_CUDA_ARCHITECTURES}" ${EZLOCALAI_MEDIA_CMAKE_ARGS} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs && \
    cmake --build /opt/acestep.cpp/build --target ace-server --parallel "${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}"
ENV ACE_STEP_BIN=/opt/acestep.cpp/build/ace-server

FROM engines-base AS llama-engine
COPY cmake/BuildLlamaEngine.cmake /tmp/BuildLlamaEngine.cmake
RUN \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" \
    cmake -P /tmp/BuildLlamaEngine.cmake

FROM engines-base AS native-compiled
ARG EZLOCALAI_PLATFORM
ARG EZLOCALAI_CUDA_FLEET_PROFILE=jetson
ARG EZLOCALAI_NATIVE_JOBS=
COPY --from=diffusion-engine /opt/stable-diffusion.cpp /opt/stable-diffusion.cpp
COPY --from=diffusion-engine /opt/ezlocalai-patches /opt/ezlocalai-patches
COPY --from=music-engine /opt/acestep.cpp /opt/acestep.cpp
ENV SDCPP_BIN=/opt/stable-diffusion.cpp/build/bin/sd-cli \
    ACE_STEP_BIN=/opt/acestep.cpp/build/ace-server
COPY --from=llama-engine /app/third_party/llama.cpp /app/third_party/llama.cpp
COPY --from=llama-engine /app/build/llama.cpp /app/build/llama.cpp
COPY --from=audio-dependencies /opt/ezlocalai/deps /opt/ezlocalai/deps
COPY --from=torch-libraries /opt/ezlocalai/deps/torch /opt/ezlocalai/deps/torch
COPY --from=ctranslate2 /opt/ezlocalai/ctranslate2 /opt/ezlocalai/ctranslate2
COPY --from=storyboard-dependencies /opt/ezlocalai/storyboard /opt/ezlocalai/storyboard
COPY --from=storyboard-dependencies /opt/ezlocalai/storyboard/fonts/ /usr/share/fonts/truetype/dejavu/
ENV LD_LIBRARY_PATH=/opt/ezlocalai/deps/lib:/opt/ezlocalai/deps/torch/lib:/opt/ezlocalai/ctranslate2/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    QWEN_TTS_BIN=/usr/local/bin/ezlocalai-tts
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -C /opt/ezlocalai/deps/native-cache.cmake \
        -DEZLOCALAI_CUDA_FLEET_PROFILE="${EZLOCALAI_CUDA_FLEET_PROFILE}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
        -DEZLOCALAI_STORYBOARD_DEPS=/opt/ezlocalai/storyboard \
        -DEZLOCALAI_SDCPP_SOURCE=/opt/stable-diffusion.cpp \
        -DEZLOCALAI_SDCPP_BUILD=/opt/stable-diffusion.cpp/build \
        -DEZLOCALAI_TTS_LLAMA_SOURCE=/app/third_party/llama.cpp \
        -DEZLOCALAI_TTS_LLAMA_BUILD=/app/build/llama.cpp && \
    cmake --build build --target cuda_package --parallel "${EZLOCALAI_NATIVE_JOBS:-${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}}"

FROM native-compiled
# Emulated cross-architecture builds under QEMU can exclude tests that
# depend on native timing or syscalls QEMU user mode lacks.
ARG EZLOCALAI_PACKAGE_TEST_EXCLUDE=
RUN LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" ctest --test-dir build \
        ${EZLOCALAI_PACKAGE_TEST_EXCLUDE:+-E "$EZLOCALAI_PACKAGE_TEST_EXCLUDE"} -R '^((storyboard_.*|ltx_norm|ltx_timesteps|ltx_schedule|ltx_rng|ltx_nf4_loader|nf4.*|torch_checkpoint|wakeword_cache|speech_resources|speech_slots|worker_metadata|media_slots|service_clients|llm_residency|llm_headroom|llm_preload|live_capacity|llm_instances|startup|media_startup|cuda_allocator)_reference|wakeword_jobs|wakeword_qwen|torch_resources|media_engine|video_auxiliary|embedding_discovery|queue_history|queue_http|dotenv_reference|dotenv_startup|openapi_reference|file_response_reference|native_http|worker_heartbeat_reference|router_fallback|worker_heartbeat_http|worker_heartbeat_shutdown|worker_tunnel|stt_diarization_reference|runtime_version_reference|tunnel_idle_reference|tunnel_transport_reference|tunnel_connection_reference|tunnel_redirect_reference|http_cookies_reference|cuda_architecture_policy|cuda_package_architectures|tunnel_framing_reference|ltx_runtime|ltx_connectors|ltx_condition_cache|core)$' --output-on-failure && \
    cmake --install build && \
    ln -sf /app/build/ezlocalai-native /usr/local/bin/ezlocalai-native && \
    ln -sf /app/build/llama_cpp_tool /usr/local/bin/llama_cpp_tool && \
    ln -sf /app/build/ezlocalai-native-entrypoint /usr/local/bin/ezlocalai-native-entrypoint && \
    ln -sf /app/build/llama.cpp/bin/llama-server /usr/local/bin/llama-server && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" ldd build/ezlocalai-native > /tmp/native-linkage.txt && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" ldd /usr/local/bin/ezlocalai-tts >> /tmp/native-linkage.txt && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" ldd /opt/acestep.cpp/build/ace-server > /tmp/ace-linkage.txt && \
    grep -F 'libggml-base.so.0 => /opt/acestep.cpp/build/' /tmp/ace-linkage.txt && \
    cat /tmp/ace-linkage.txt >> /tmp/native-linkage.txt && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" ldd /usr/local/bin/llama-server >> /tmp/native-linkage.txt && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" ldd /opt/stable-diffusion.cpp/build/bin/sd-cli >> /tmp/native-linkage.txt && \
    /usr/local/bin/ffmpeg -hide_banner -encoders | grep -qE '^ V[^ ]* png ' && \
    ! grep -E 'not found|libpython' /tmp/native-linkage.txt

# The worker reports the checkout's short commit as its router version, as the
# Python image does. Copy only ref metadata; builds without .git keep the
# embedded source-hash version.
RUN --mount=type=bind,source=.,target=/tmp/build-context \
    if [ -f /tmp/build-context/.git/HEAD ]; then \
        mkdir -p /app/.git && cp /tmp/build-context/.git/HEAD /app/.git/ && \
        if [ -d /tmp/build-context/.git/refs ]; then cp -r /tmp/build-context/.git/refs /app/.git/; fi && \
        if [ -f /tmp/build-context/.git/packed-refs ]; then cp /tmp/build-context/.git/packed-refs /app/.git/; fi; \
    fi

EXPOSE 8091
ENTRYPOINT ["/usr/local/bin/ezlocalai-native-entrypoint"]
