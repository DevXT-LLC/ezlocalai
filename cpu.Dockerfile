ARG CPU_IMAGE=ubuntu:24.04
# CPU-only amd64/arm64 image: native engines and LibTorch, with no CUDA runtime.
FROM ubuntu:24.04 AS audio-dependencies
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates cmake && \
    rm -rf /var/lib/apt/lists/*
COPY cmake/cpu-audio-dependencies.json /tmp/audio-dependencies/cpu-amd64-audio-dependencies.json
COPY cmake/CudaAudioDependencies.cmake cmake/cpu-arm64-audio-dependencies.json /tmp/audio-dependencies/
RUN --mount=type=cache,id=ezlocalai-audio-cpu-${TARGETARCH},target=/var/cache/ezlocalai-audio \
    cmake -DDESTINATION=/opt/ezlocalai/deps -DDOWNLOAD_CACHE=/var/cache/ezlocalai-audio \
        -DMANIFEST=/tmp/audio-dependencies/cpu-${TARGETARCH}-audio-dependencies.json \
        -P /tmp/audio-dependencies/CudaAudioDependencies.cmake

FROM ubuntu:24.04 AS storyboard-dependencies
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake gcc curl make pkg-config libbrotli-dev libfreetype-dev libharfbuzz-dev libfribidi-dev fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
# The font dependency stage also embeds Brotli through Pillow's native wheel.
COPY docker/install-patched-brotli.sh /tmp/install-patched-brotli.sh
RUN sh /tmp/install-patched-brotli.sh && rm /tmp/install-patched-brotli.sh
COPY cmake/StoryboardDependencies.cmake cmake/storyboard-dependencies.json cmake/gb10-storyboard-dependencies.json /tmp/storyboard-dependencies/
# GB10's font/text libraries are ordinary ARM64 CPU libraries.
RUN --mount=type=cache,id=ezlocalai-storyboard-cpu-${TARGETARCH},target=/var/cache/ezlocalai-storyboard \
    manifest=storyboard-dependencies.json && \
    if [ "${TARGETARCH}" = arm64 ]; then manifest=gb10-storyboard-dependencies.json; fi && \
    cmake -DDESTINATION=/opt/ezlocalai/storyboard -DDOWNLOAD_CACHE=/var/cache/ezlocalai-storyboard \
        -DMANIFEST=/tmp/storyboard-dependencies/${manifest} \
        -P /tmp/storyboard-dependencies/StoryboardDependencies.cmake

FROM ${CPU_IMAGE} AS engines-base

ARG EZLOCALAI_LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG EZLOCALAI_LLAMA_CPP_REF=8642541b255bfb5dc39cd351a178be09013a55d6
# GGML_NATIVE=OFF: images run on other fleet CPUs. -march=native from the
# build host (e.g. Alder Lake AVX-VNNI) fails on Haswell or Zen hosts.
ARG EZLOCALAI_LLAMA_CPP_CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=ON -DGGML_NATIVE=OFF"
ARG EZLOCALAI_LLAMA_CPP_JOBS=

ENV DEBIAN_FRONTEND=noninteractive \
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
        nlohmann-json3-dev \
        make \
        ninja-build \
        poppler-utils \
        pkg-config \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# Ubuntu has not published a Brotli decoder backport for CVE-2026-111497.
COPY docker/install-patched-brotli.sh /tmp/install-patched-brotli.sh
RUN sh /tmp/install-patched-brotli.sh && rm /tmp/install-patched-brotli.sh

WORKDIR /app
RUN curl -fL https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz -o /tmp/opencv.tar.gz && \
    echo "1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d  /tmp/opencv.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/opencv.tar.gz -C /tmp && \
    cmake -S /tmp/opencv-4.13.0 -B /tmp/opencv-build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIST=core,imgproc,imgcodecs,videoio -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python2=OFF -DBUILD_JAVA=OFF -DWITH_IPP=OFF -DWITH_OPENCL=OFF \
        -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_ITT=OFF \
        -DWITH_OPENEXR=OFF -DWITH_OPENJPEG=OFF -DWITH_AVIF=OFF && \
    cmake --build /tmp/opencv-build --parallel "$(nproc)" && \
    cmake --install /tmp/opencv-build && ldconfig && \
    rm -rf /tmp/opencv.tar.gz /tmp/opencv-4.13.0 /tmp/opencv-build
FROM engines-base AS diffusion-engine-base
# Pin the same native media engine as the installed Python worker.
ARG EZLOCALAI_SDCPP_REF=c678dfe704a2230342376b46add9c8ca736a653d
ARG EZLOCALAI_MEDIA_CMAKE_ARGS=-DGGML_NATIVE=OFF
COPY patches/sdcpp-ltx-conditions.patch patches/sdcpp-ltx-diffusers.patch /opt/ezlocalai-patches/
RUN git init /opt/stable-diffusion.cpp && \
    git -C /opt/stable-diffusion.cpp remote add origin https://github.com/leejet/stable-diffusion.cpp.git && \
    git -C /opt/stable-diffusion.cpp fetch --depth 1 origin "${EZLOCALAI_SDCPP_REF}" && \
    git -C /opt/stable-diffusion.cpp checkout --detach FETCH_HEAD && \
    git -C /opt/stable-diffusion.cpp submodule update --init --recursive --depth 1 && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-conditions.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-diffusers.patch && \
    cmake -S /opt/stable-diffusion.cpp -B /opt/stable-diffusion.cpp/build \
        -DSD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
        ${EZLOCALAI_MEDIA_CMAKE_ARGS} && \
    cmake --build /opt/stable-diffusion.cpp/build --target sd-cli --parallel "${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}"
ENV SDCPP_BIN=/opt/stable-diffusion.cpp/build/bin/sd-cli

FROM diffusion-engine-base AS diffusion-engine-timesteps
COPY patches/sdcpp-ltx-timesteps.patch /opt/ezlocalai-patches/
RUN git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-timesteps.patch && \
    cmake --build /opt/stable-diffusion.cpp/build --target sd-cli --parallel "${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}"

FROM diffusion-engine-timesteps AS diffusion-engine-nf4
COPY patches/sdcpp-ltx-nf4.patch /opt/ezlocalai-patches/
COPY include/ezlocalai/nf4.hpp include/ezlocalai/torch_cuda_mean.hpp /opt/stable-diffusion.cpp/src/ezlocalai/
RUN git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-nf4.patch && \
    cmake --build /opt/stable-diffusion.cpp/build --target sd-cli --parallel "${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}"

# Apply the remaining engine patches before compiling once. Rebuilding after
# every patch repeats the same translation units under ARM64 emulation.
FROM diffusion-engine-nf4 AS diffusion-engine
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
RUN git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-media-worker.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-nf4-storage.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-nf4-packed.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-gemma-residency.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-offload-policy.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-shared-cuts.patch && \
    git -C /opt/stable-diffusion.cpp apply /opt/ezlocalai-patches/sdcpp-ltx-connector-pass.patch && \
    cmake --build /opt/stable-diffusion.cpp/build --target sd-cli --parallel "${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}"

FROM engines-base AS music-engine
ARG EZLOCALAI_MEDIA_CMAKE_ARGS=-DGGML_NATIVE=OFF
ARG EZLOCALAI_ACESTEP_REF=5b88ded7919285d5ad5b1e86ffe3378314f305db
RUN git init /opt/acestep.cpp && \
    git -C /opt/acestep.cpp remote add origin https://github.com/ServeurpersoCom/acestep.cpp.git && \
    git -C /opt/acestep.cpp fetch --depth 1 origin "${EZLOCALAI_ACESTEP_REF}" && \
    git -C /opt/acestep.cpp checkout --detach FETCH_HEAD && \
    git -C /opt/acestep.cpp submodule update --init --recursive --depth 1 && \
    cmake -S /opt/acestep.cpp -B /opt/acestep.cpp/build \
        -DGGML_CUDA=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DCMAKE_BUILD_TYPE=Release \
        ${EZLOCALAI_MEDIA_CMAKE_ARGS} && \
    cmake --build /opt/acestep.cpp/build --target ace-server --parallel "${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}"
ENV ACE_STEP_BIN=/opt/acestep.cpp/build/ace-server

FROM engines-base AS llama-engine
COPY cmake/BuildLlamaEngine.cmake cmake/ApplyLlamaPatches.cmake patches/llama-conversation-cache.patch patches/llama-cache-restore-atomic.patch patches/llama-cache-storage.patch patches/llama-mtp-resume.patch patches/llama-checkpoint-coverage.patch patches/llama-checkpoint-entry-window.patch patches/llama-cache-retention.patch patches/llama-cuda-q4-kv-fusion.patch patches/llama-prefix-anchor.patch /tmp/
RUN cmake -P /tmp/BuildLlamaEngine.cmake

FROM engines-base AS native-compiled
COPY --from=diffusion-engine /opt/stable-diffusion.cpp /opt/stable-diffusion.cpp
COPY --from=diffusion-engine /opt/ezlocalai-patches /opt/ezlocalai-patches
COPY --from=music-engine /opt/acestep.cpp /opt/acestep.cpp
ENV SDCPP_BIN=/opt/stable-diffusion.cpp/build/bin/sd-cli \
    ACE_STEP_BIN=/opt/acestep.cpp/build/ace-server
COPY --from=llama-engine /app/third_party/llama.cpp /app/third_party/llama.cpp
COPY --from=llama-engine /app/build/llama.cpp /app/build/llama.cpp
COPY --from=audio-dependencies /opt/ezlocalai/deps /opt/ezlocalai/deps
COPY --from=storyboard-dependencies /opt/ezlocalai/storyboard /opt/ezlocalai/storyboard
COPY --from=storyboard-dependencies /opt/ezlocalai/storyboard/fonts/ /usr/share/fonts/truetype/dejavu/
ENV LD_LIBRARY_PATH=/opt/ezlocalai/deps/lib:/opt/ezlocalai/deps/torch/lib \
    QWEN_TTS_BIN=/usr/local/bin/ezlocalai-tts
ARG EZLOCALAI_NATIVE_JOBS=
# Isolated portable CPU MeanVC2 engine; weights are provisioned only when opted in.
COPY cmake/BuildMeanVc2.cmake /tmp/meanvc2/BuildMeanVc2.cmake
COPY engine/meanvc2 /tmp/meanvc2/engine/meanvc2
COPY include/ezlocalai/voice_audio.hpp include/ezlocalai/voice_resample.hpp include/ezlocalai/process.hpp /tmp/meanvc2/include/ezlocalai/
RUN cmake -DEZLOCALAI_MEANVC2_WRAPPER=/tmp/meanvc2/engine/meanvc2 \
        -DEZLOCALAI_MEANVC2_PREFIX=/opt/ezlocalai/meanvc2 \
        -DEZLOCALAI_ARM_CPU_ARCH="${EZLOCALAI_ARM_CPU_ARCH:-armv8-a}" \
        -P /tmp/meanvc2/BuildMeanVc2.cmake && \
    ln -sf /opt/ezlocalai/meanvc2/bin/ezlocalai-meanvc2 /usr/local/bin/ezlocalai-meanvc2 && \
    rm -rf /tmp/meanvc2 /tmp/ezlocalai-meanvc2-source /tmp/ezlocalai-meanvc2-build
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -C /opt/ezlocalai/deps/native-cache.cmake \
        -DEZLOCALAI_REQUIRE_MEANVC2=ON \
        -DEZLOCALAI_STORYBOARD_DEPS=/opt/ezlocalai/storyboard \
        -DEZLOCALAI_SDCPP_SOURCE=/opt/stable-diffusion.cpp \
        -DEZLOCALAI_SDCPP_BUILD=/opt/stable-diffusion.cpp/build \
        -DEZLOCALAI_TTS_LLAMA_SOURCE=/app/third_party/llama.cpp \
        -DEZLOCALAI_TTS_LLAMA_BUILD=/app/build/llama.cpp && \
    cmake --build build --target cpu_package --parallel "${EZLOCALAI_NATIVE_JOBS:-${EZLOCALAI_LLAMA_CPP_JOBS:-$(nproc)}}"

FROM native-compiled
ARG EZLOCALAI_PACKAGE_TEST_EXCLUDE=
RUN ctest --test-dir build \
        ${EZLOCALAI_PACKAGE_TEST_EXCLUDE:+-E "$EZLOCALAI_PACKAGE_TEST_EXCLUDE"} \
        -R '^(cuda_mps|voice_conversion|voice_conversion_engine_version|realtime|(storyboard_.*|ltx_norm|ltx_timesteps|ltx_schedule|ltx_rng|ltx_nf4_loader|nf4.*|torch_checkpoint|wakeword_cache|speech_resources|speech_slots|worker_metadata|media_slots|service_clients|llm_residency|llm_headroom|llm_preload|live_capacity|llm_instances|startup|media_startup|cuda_allocator)_reference|wakeword_jobs|wakeword_qwen|torch_resources|media_engine|video_auxiliary|embedding_discovery|queue_history|queue_http|dotenv_reference|dotenv_startup|openapi_reference|file_response_reference|native_http|worker_heartbeat_reference|router_fallback|worker_heartbeat_http|worker_heartbeat_shutdown|worker_tunnel|stt_diarization_reference|runtime_version_reference|tunnel_idle_reference|tunnel_transport_reference|tunnel_connection_reference|tunnel_redirect_reference|http_cookies_reference|tunnel_framing_reference|ltx_runtime|ltx_connectors|ltx_condition_cache|llm_cache_package|llm_input_package|conversation_cache|cache_restore_atomic|cache_storage|llama_mtp_resume|llama_checkpoint_coverage|q4_kv_layout|managed_llama_patches|http_stream_socket|request_gzip_codec|request_gzip_http|stream_completion|tts_websocket_state|process_capture|core)$' --output-on-failure && \
    cmake --install build && \
    ln -sf /app/build/ezlocalai-native /usr/local/bin/ezlocalai-native && \
    ln -sf /app/build/llama_cpp_tool /usr/local/bin/llama_cpp_tool && \
    ln -sf /app/build/ezlocalai-native-entrypoint /usr/local/bin/ezlocalai-native-entrypoint && \
    ln -sf /app/build/llama.cpp/bin/llama-server /usr/local/bin/llama-server && \
    ldd build/ezlocalai-native > /tmp/native-linkage.txt && \
    ldd /usr/local/bin/ezlocalai-tts >> /tmp/native-linkage.txt && \
    ldd /usr/local/bin/ezlocalai-meanvc2 >> /tmp/native-linkage.txt && \
    ldd /opt/acestep.cpp/build/ace-server > /tmp/ace-linkage.txt && \
    grep -F 'libggml-base.so.0 => /opt/acestep.cpp/build/' /tmp/ace-linkage.txt && \
    cat /tmp/ace-linkage.txt >> /tmp/native-linkage.txt && \
    ldd /usr/local/bin/llama-server >> /tmp/native-linkage.txt && \
    ldd /opt/stable-diffusion.cpp/build/bin/sd-cli >> /tmp/native-linkage.txt && \
    ! grep -E 'not found|libpython|libcuda|libcudnn|libcublas|libnvrtc' /tmp/native-linkage.txt

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
