# ezlocalai

ezlocalai is a local AI inference server with an OpenAI-compatible HTTP API:
chat and vision, embeddings, speech-to-text, text-to-speech, image, music and
video generation, and wake-word training. It is written in C++17 and runs local
LLM and embedding inference through native llama.cpp. It replaces the Python
ezlocalai worker and registers with the Rust
[WorkerRouter](https://github.com/DevXT-LLC/WorkerRouter) for fleet routing.

The worker, request queue, management tools, container entrypoint and test
fixtures are all C++. `include/ezlocalai/ezlocalai_c.h` exposes a C-compatible
ABI implemented in C++. See [AGENTS.md](AGENTS.md) for the repository layout and
contributor workflow, [PARITY_PROGRESS.md](PARITY_PROGRESS.md) for parity status
against the Python implementation, and
[HARDWARE_TEST_PLAN.md](HARDWARE_TEST_PLAN.md) for hardware that still needs
physical validation.

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/parity_report parity_manifest.tsv
```

The strict parity gate is:

```bash
./build/parity_report --worker-only --enforce --source-root /path/to/ezlocalai parity_manifest.tsv
```

It fails while gaps remain or the reference source is unavailable. The tool scans the sibling `../ezlocalai` checkout automatically; use `--source-root /path/to/ezlocalai` to select another checkout. Manifest labels describe tracked coverage and do not replace runtime differential testing. See `PARITY_PROGRESS.md` for the current audit.

The replacement architecture uses this C++ worker with the separate Rust WorkerRouter service. `--worker-only` excludes the legacy router service routes while retaining worker registration, heartbeats, tunnels, model/resource reporting, and every worker API and packaging requirement. Omit it to inspect the historical combined checklist.

## Run

```bash
./build/ezlocalai-native --host 0.0.0.0 --port 8091
```

Native loads `.env` from the working directory without overriding process environment variables. Copy `.env.example` to `.env` locally before running if you want the CUDA compose defaults.

## CPU Docker

```bash
docker compose build
docker compose up -d
```

The default Compose file builds the x86-64 CPU image from
`cpu.Dockerfile`. It includes CPU llama.cpp, Qwen TTS, Whisper STT,
LibTorch wake-word training, and the native image/music/video engines.
It uses the same `.env` settings and seven persistent mounts as the CUDA
service, with no NVIDIA device requests or CUDA runtime libraries.
The portable x86 CPU baseline requires AVX2/FMA/F16C. CPU execution can be
substantially slower for larger models; inclusion of a media engine does not
establish output-quality acceptance for every preset.

To reuse an installation elsewhere, set `EZLOCALAI_ENV_FILE` and the
`EZLOCALAI_*_DIR` mount overrides in addition to passing its `.env` to Compose
with `--env-file`. CPU and CUDA services both use container name `ezlocalai`
and port 8091 by default; select one for deployment.

## CUDA Docker

```bash
docker compose -f docker-compose-cuda.yml build
docker compose -f docker-compose-cuda.yml up -d
```

The compose service builds `cuda.Dockerfile`, builds ezlocalai-managed llama.cpp with CUDA, and stores model/cache/output volumes under the repo root.

The x86-64 CUDA image also includes native Whisper STT and Qwen TTS dependencies
and the TTS executable, without a Python runtime. Their archives are pinned in
`cmake/cuda-audio-dependencies.json`. Speech models use the shared model
cache and remain controlled by the existing `.env` enable switches. See
`PARITY_PROGRESS.md` for actual CPU/GPU verification and remaining gaps.

Wake-word training uses Qwen TTS. Compose persists `wakeword_models`,
`wakeword_samples`, and `wakeword_jobs` beside the model cache. Their host paths
can be overridden with `EZLOCALAI_WAKEWORD_MODELS_DIR`,
`EZLOCALAI_WAKEWORD_SAMPLES_DIR`, and `EZLOCALAI_WAKEWORD_JOBS_DIR`.
Copy any existing container-local wake-word files into those host directories
before replacing that container. Persisted unfinished jobs remain visible for
cancellation; they are not automatically resumed after restart.
Job cancellation takes effect between synthesis calls, augmentation/noise
samples and training batches. An in-flight synthesis call, CUDA kernel or
export must finish before its resources are released. Exported files remain
private until the job is ready to complete; cancelled or failed exports are
not available as models.

The CUDA build also packages pinned stable-diffusion.cpp for local Qwen image
generation and editing. Enable it with `IMAGE_ENABLED=true` and
`IMG_MODEL=qwen-image-2.1`; it reuses `models/qwen-image` from the Python worker.
The original installation's enable flags remain authoritative.

## llama.cpp

```bash
./build/llama_cpp_tool update
./build/llama_cpp_tool build
./build/llama_cpp_tool status
./build/llama_cpp_tool model-status
```

Set `EZLOCALAI_LLAMA_CPP_REF` to pin an upstream llama.cpp branch, tag, or commit before building.

## Implementation details

This repository is the C++ implementation of ezlocalai. The replacement target is the Python worker's behavior with the separate Rust WorkerRouter. Historical Python router-service code is outside that target; use the worker-only parity gate. Application code, tools, entrypoint, and process fixtures are compiled as C++17; the C ABI is retained for callers.

The current native target ports the service boundary and router surface first:

- C ABI for shared environment defaults, timestamp formatting, and a bounded request queue. Completed queue history retains at most 1,000 results/status entries, including failures and cancellations; finished inputs are released immediately. Evicted IDs return not found, while already attached waiters retain their result.
- C++17 HTTP server with the ezlocalai/OpenAI-compatible API and router endpoints tracked in `parity_manifest.tsv`.
- A pluggable `InferenceBackend` interface. The native executable defaults to llama.cpp through `llama-server` for local LLM inference while keeping the ezlocalai/OpenAI endpoints unchanged; `EZLOCALAI_NATIVE_BACKEND=echo` is the test/development opt-out.

The CUDA replacement has real Rust-router acceptance for chat/SSE, vision,
embeddings, Whisper transcription, Qwen speech, image generation and wake-word
jobs. Direct and reverse-tunnel transport, streaming disconnects, resource
reporting and model handoff have separate regression and live acceptance records.
Default-size LTX and music-video acceptance is still being completed.
The x86 CPU and ROCm packages have CPU inference acceptance; this does not
establish AMD GPU support on physical hardware. GB10 and Pi validation uses
ARM emulation, and the Jetson Orin image is still a build candidate.

Remaining limits include physical multi-GPU and other fleet GPU execution,
broader model and checkpoint coverage, video quality/performance, and the
specific behavioral gaps recorded in `parity_manifest.tsv`. That manifest and
the detailed sections below retain historical milestones; later dated records
in `PARITY_PROGRESS.md` supersede earlier validation limits.

The CUDA image supports native wake-word prediction from existing Python CNN/GRU
`model.pt` ZIP checkpoints and `config.json` under `wakeword_models/<word>`.
JSON/base64 and multipart prediction use the same audio preprocessing and model
cache behavior as the Python worker. Host builds need the pinned LibTorch adapter
(`EZLOCALAI_LIBTORCH_ROOT`). The internal CPU/CUDA trainer, checkpoint writer and
ONNX exporter have separate reference coverage, as do PCM augmentation and
negative-noise generation. Qwen-backed background jobs run synthesis, training
and checkpoint/ONNX export, with live status, metrics, cancellation and persisted
records. Cancellation is cooperative at synthesis, sample/batch and export
boundaries. Working ESPDL export and trained-model quality remain unverified.
Historical TTS fallbacks and PDF-to-speech are outside the replacement target.
See `PARITY_PROGRESS.md` for current deployment evidence and remaining limits.

To make remaining gaps visible from the build output, use `parity_report`. Add `--enforce` when the build should fail until every manifest row is implemented.

## Build

Build dependencies include a C++17 compiler, CMake, pkg-config, OpenSSL, zlib, Brotli, ICU, libpng, libjpeg, libwebp (including demux), OpenBLAS, FFmpeg audio libraries, nlohmann-json 3.11+, and OpenCV 4.8+ with core/imgproc/videoio and FFmpeg support. On Debian/Ubuntu, the library packages include `libssl-dev zlib1g-dev libbrotli-dev libicu-dev libpng-dev libjpeg-dev libwebp-dev libopenblas-dev libavformat-dev libavcodec-dev libavutil-dev libswresample-dev nlohmann-json3-dev`. Native STT initialization also requires the PortAudio runtime (`libportaudio2`), included in the CUDA image; its C ABI is loaded directly without PyAudio or a Python runtime. Speaker diarization additionally uses `libsndfile1` and `libsoxr0`, also included in the CUDA image. Install a suitable OpenCV development package or build OpenCV and set `OpenCV_DIR` to its `lib/cmake/opencv4` directory. The CUDA Dockerfile builds OpenCV 4.13.0 from a SHA-256-verified archive. Video input uses `ffmpeg` on PATH; video integration tests also require `ffprobe` and `curl`.

```bash
cmake -S native -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/parity_report parity_manifest.tsv
./build/parity_report parity_manifest.tsv --enforce
# The flag order is flexible:
./build/parity_report --enforce parity_manifest.tsv
```

The `configuration_reference` test reads the sibling `../ezlocalai` checkout (override with `-DEZLOCALAI_REFERENCE_ROOT=/path/to/ezlocalai`). It checks all 147 built-in defaults, all 92 GPU tiers, and 536 capability/model-name cases captured from executing the reference functions. Fixture source hashes make reference changes fail visibly. The test executable itself is C++; the reference Python files are test data.

`openapi_reference` compares complete worker schemas and documentation responses
captured from the current Python app. `/openapi.json` captures `DEFAULT_MODEL`,
`EMBEDDING_MODEL` and `WHISPER_MODEL` when the worker initializes, matching Python
class/form defaults. Later environment changes do not alter those defaults.
Refresh the worker snapshot and `tests/fixtures/openapi_cases.json` together
when source routes or schemas change; the fixture records dependency versions
and source hashes. The legacy Python router snapshot is outside this check.

`test_speculative_model` is a manual live-model runner, excluded from CTest. Start
a worker with the selected `none`, `mtp` or `dflash2` backend and an isolated
WorkerRouter, then run:

```bash
./build/test_speculative_model \
  tests/fixtures/speculative_live_cases.json config.json /tmp/speculative-results
```

The JSON config supplies `backend`, `context`, `model`, worker `label`,
`worker_url`, `engine_url` and `router_url`; optional `worker_key`, `engine_key`
and `router_key` authenticate those endpoints. The engine URL must address the
worker's owned llama-server. The runner checks the selected backend/context,
four direct-engine greedy text cases and ten source vision cases in buffered
and streamed form through Rust. It records complete responses, text hashes,
draft/acceptance counters and queue cleanup in the output directory. The caller
owns model provisioning and service startup/restoration; this runner does not
download weights or replace a running service. Compare hashes separately:
passing semantic vision checks does not establish identical generated text.

Local music-video generation also needs the native storyboard libraries. The
CUDA image packages them automatically. For a host build, prepare them with
`cmake/StoryboardDependencies.cmake` and configure `EZLOCALAI_STORYBOARD_DEPS`
as described in [the storyboard verification notes](tests/fixtures/storyboard_README.md).
This enables both music-video routes, including default storyboards, theorem
overlays, ACE-Step audio, scene generation and FFmpeg stitching. Without these
dependencies, local music-video generation returns 501. Enable `MUSIC_ENABLED`
and `VIDEO_ENABLED` on the same worker and select its `VIDEO_MODEL`.

### Native Whisper verification

The x86-64 CUDA image packages native CTranslate2 4.8.2, ONNX Runtime 1.31.0,
LibTorch 2.9.1+cu128, cuDNN 9.10.2.21, Silero VAD assets and the exact libsndfile
binary used by the installed reference. It also builds the native Qwen TTS
worker from the pinned llama.cpp tree and sets `QWEN_TTS_BIN` to its installed
executable. No Python interpreter or binding library is required at runtime.
Downloads and licenses are managed by `cmake/CudaAudioDependencies.cmake` and
the SHA-256 lock file beside it. Models remain in the mounted model cache.

Current CPU/CUDA large-v3-turbo and Qwen inference comparisons are documented
in `tests/fixtures/audio_deployment_README.md`. Configure
`EZLOCALAI_WHISPER_CT2_TURBO_MODEL` to enable the current STT deployment tests;
`EZLOCALAI_STT_TORCH_GPU_TESTS=ON` additionally enables their GPU cases and
the Qwen GPU output check when its model directory is configured. The older
optional fixtures below retain their explicitly listed dependency versions.

`whisper_features_reference` checks native FFmpeg decoding and C++ log-Mel extraction against faster-whisper 1.2.0 (the reference CUDA dependency) and 1.0.3 (the reference CPU dependency). Its 32 feature cases cover 80/128 bands and empty, short and longer inputs; five audio format cases compare decoded samples exactly. The measured feature error is at most 1.19209e-7 with the tested dependency versions. This is numerical agreement within tolerance, not bitwise feature identity.

The optional C++ CTranslate2 adapter runs the same native generation engine as faster-whisper. Build CTranslate2 4.6.1 separately, then provide its headers and shared library. Supplying the pinned model also enables a real CPU int8 test:

```bash
cmake -S native -B build \
  -DEZLOCALAI_CTRANSLATE2_INCLUDE_DIR=/path/to/CTranslate2/include \
  -DEZLOCALAI_CTRANSLATE2_LIBRARY=/path/to/libctranslate2.so \
  -DEZLOCALAI_WHISPER_CT2_MODEL=/path/to/faster-whisper-tiny.en
cmake --build build
ctest --test-dir build -R '^whisper_' --output-on-failure
```

Use `Systran/faster-whisper-tiny.en` at revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba`; the test checks `model.bin` SHA-256 before loading it. The model directory must contain the upstream configuration and vocabulary as well as the weights. Fixture provenance is in `tests/fixtures/whisper_kernel_assets/README.md`. Eight decoder cases cover both feature profiles, beam search, timestamp suppression and an initial prompt. Each checks exact token IDs using both captured and C++-computed model inputs. The adapter and tests do not execute or link Python.

For a CTranslate2 binary built with the old libstdc++ ABI, add `-DEZLOCALAI_CTRANSLATE2_LEGACY_ABI=ON`. Only the adapter uses that ABI; its interface passes plain C types to the rest of the C++ application. Shared-library dependencies must be available alongside CTranslate2 or on the loader search path. The verified local setup uses the native library shipped with CTranslate2 4.6.1, with its matching source headers.

Native Silero VAD is also available through ONNX Runtime. Its C++ implementation supports the distinct faster-whisper 1.0.3 and 1.2.0 models, speech segmentation, silence removal, chunk metadata and timestamp mapping. `whisper_vad_reference` runs the source-derived policy fixtures without a model dependency. To enable actual model verification, add these options to the CMake configuration:

```bash
cmake -S native -B build \
  -DEZLOCALAI_ONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DEZLOCALAI_ONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.so \
  -DEZLOCALAI_WHISPER_VAD_MODERN_MODELS=/path/to/faster-whisper-1.2.0/assets \
  -DEZLOCALAI_WHISPER_VAD_LEGACY_MODELS=/path/to/faster-whisper-1.0.3/assets
```

The modern directory must contain `silero_encoder_v5.onnx` and `silero_decoder_v5.onnx`; the legacy directory must contain `silero_vad.onnx`. Tests check their SHA-256 values before execution. The verified runtime is ONNX Runtime 1.24.1 on CPU, matching the Python reference execution provider and session settings. No Python package is required to run the C++ tests or adapter; the native headers/library and model files are sufficient. With both ONNX Runtime and CTranslate2 configured, `whisper_speech_reference` verifies their combined VAD/filter/feature/decoder path. Model provenance and fixture details are in `tests/fixtures/whisper_vad_assets/README.md`.

`whisper_tokenizer_reference` verifies the C++ byte-level BPE tokenizer, English/multilingual special tokens, UTF-8 decoding, timestamp tokens, word splitting, prompt construction and both faster-whisper suppression profiles. Its pinned tokenizer files are bundled as compressed test data, so this test requires no downloaded model or Python runtime. Unicode classifications are captured from tokenizers 0.22.1 rather than inferred from the host Unicode library. The implementation reads standard Whisper ByteLevel/BPE tokenizer.json files; custom normalizers, stochastic BPE and nonstandard added-token flags are not implemented. The combined speech test also uses C++ tokenization, initial prompts and text decoding.

`whisper_transcription_reference` exercises the C++ standard transcription controller with 720 captured cases from both faster-whisper versions. It compares full encoder/decoder/alignment/language call traces, prompt context, fallback choices, segments, word timestamps and errors. With the native dependencies above configured, add `-DEZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL=/path/to/faster-whisper-tiny` to enable `whisper_transcription_model_reference`. Use `Systran/faster-whisper-tiny` revision `d90ca5fe260221311c53c58e660288d3deb8d356`; all model/configuration/tokenizer/vocabulary hashes are checked. Its 32 real-model cases cover VAD, long audio, context, word alignment, automatic language detection, translation and clips. Fixture provenance and numerical tolerances are in `tests/fixtures/whisper_transcription_assets/README.md`.

`whisper_batched_reference` covers the C++ faster-whisper 1.2.0 batch controller with 300 source cases and 342 invocations, including retained word-alignment state after an injected failure. With both Whisper models and the VAD directories above configured, `whisper_batched_model_threads_1_reference` and `whisper_batched_model_threads_8_reference` compare 25 actual-model transcriptions each, with matching one-/eight-thread OpenBLAS settings. Small preprocessing differences can change int8 model output; the fixture header and test enforce matching thread counts. The batch controller uses native CTranslate2 batch encoding, generation, language detection and alignment calls.

Configured CTranslate2/VAD builds now connect these components to the transcription, translation and subtitle APIs. The executable uses the modern profile; legacy-profile HTTP behavior is verified through the injectable native factory. Native subtitles preserve source segments, translate target languages concurrently through the existing LLM pipeline, and reproduce source word-count batching, numbered-line fallback and JSON/SRT/VTT formatting. Configured LLM audio/video input now uses the same native STT pool and guard, preserving the extra usage mark, audio failure lease retention, video finally cleanup and voice-server destruction rules. The separate speech-endpoint audio-to-audio/PDF-to-audio path remains on its legacy adapters. Complete request cancellation, failure-time scheduling of outstanding subtitle translations, live GPU API/handoff verification, arbitrary model/dependency configurations and production dependency packaging remain incomplete or unverified.

Native speaker diarization now supplies MFCC/average-linkage labels and persistent session speaker IDs through the shared STT runtime. Its deterministic feature/session suite and optional real-model test are documented in [stt_diarization_README.md](tests/fixtures/stt_diarization_README.md).

`stt_runtime_reference` verifies the C++ service wrapper's model selection, CPU fallback, call ordering, output formatting, exception handling and cleanup against 800 cases executing unchanged `STT.py` methods, plus nine source comparisons using real Linux filesystem operations. The optional `ezlocalai_stt_whisper` target connects that wrapper to native model inference with local-directory and model-alias resolution through a native Hub snapshot downloader; callers may override model resolution. `stt_runtime_model_reference` compares 30 complete base64/file/inference/result calls using local directories and offline alias snapshots of the pinned models, including word timestamps and the legacy batch-keyword error. Both native and reference runs initialize real PortAudio; GPU availability remains substituted. Fixture provenance is in `tests/fixtures/stt_runtime_assets/README.md`.

STT initializes PortAudio by default and terminates it on explicit close. `stt_audio_reference` compares 120 cases through the unchanged PyAudio 0.2.14 class and compiled extension, including failed initialization, repeated termination and multiple/dropped instances. `stt_audio_live_reference` checks real PortAudio initialization counts when the library is available. These operations initialize the audio library without opening a recording stream. Microphone listening and other stream/device operations remain incomplete; details are in `tests/fixtures/stt_audio_README.md`.

`SttPool` implements request-keyed resident/transient leases, FIFO reuse, resource registration and synchronous/queued destruction. `stt_whisper_pool_operations` connects it to the native STT engine and PortAudio. `stt_pool_reference` compares 540 source scenarios/12,480 actions and checks concurrent leases and the FIFO cleanup worker; optional `stt_pool_model_reference` compares 46 real-model lifecycle actions and 14 inference calls across both profiles. `SttService` connects the pool and guard to transcription/translation requests. Complete server lifecycle integration and GPU pool verification remain incomplete. See `tests/fixtures/stt_pool_README.md` for scope and provenance.

The native `tts_text` layer now matches current CTTS text preparation and both chunkers. The existing C ABI number, normalization and chunking functions share this implementation. It counts Unicode code points, uses pinned CPython Unicode 15.0 word/digit/whitespace/casing data, preserves current punctuation splitting and first-stream-chunk policy, and handles empty text and the minimum signed integer correctly. `ctts_text_reference` checks 44,831 source results plus 9,662 C ABI comparisons. Details and remaining type/integration scope are in `tests/fixtures/ctts_text_README.md`.

`LlamaTts` now provides a native persistent-process wrapper for the current Qwen TTS worker: source launch/download policy, child-only GPU settings, language inference, WAV readback, incremental float32 PCM, timeouts and cancellation cleanup. The optional `ezlocalai-tts` target uses the unchanged source C++ worker and an independently built pinned llama.cpp/libmtmd. Configure `EZLOCALAI_TTS_LLAMA_SOURCE` and `EZLOCALAI_TTS_LLAMA_BUILD`; add `EZLOCALAI_TTS_MODEL_DIRECTORY` for real CPU comparison. Policy/process tests cover 571 policy cases and 115 process scenarios/205 operations; model fixtures compare five requests and six exact audio buffers, including a 72-frame streaming window and final tail. The native CTTS runtime, cache, reference-audio policies, buffered/streamed generation and TTS pool now use this wrapper; local buffered/streamed HTTP now uses this stack, while the remaining routing and lifecycle integration stays partial. See `tests/fixtures/llama_tts_README.md` for provenance, build dependencies and remaining gaps.

`VoiceSlotGuard` provides native FIFO service admission, shared STT/TTS handoffs, cancellation and cross-thread lease release. `voice_slots_reference` compares 735 source scenarios/2,804 actions, including compound cleanup failures; `voice_slots_async_reference` compares 11 real-asyncio scenarios/84 settled states and handoff traces against native threads. Transcription/translation requests now connect STT admission and existing LLM handoff hooks; TTS pool/worker composition is also tested, while complete HTTP cancellation and public TTS integration remain open. See `tests/fixtures/voice_slot_README.md` for the tested concurrency scope.

`stt_api_reference` compares 392 multipart HTTP requests with source response formatting, kwargs, exceptions and resource/pool lifecycle. `stt_api_model_reference` adds 56 actual-model HTTP requests across both profiles; `stt_api_main_reference` checks the compiled executable over TCP without configured whisper.cpp executables. Set `WHISPER_VAD_DIRECTORY` to the modern ONNX asset directory; the executable defaults to `<configured install prefix>/share/ezlocalai/whisper-vad`. Supplying `EZLOCALAI_WHISPER_VAD_MODERN_MODELS` also installs those assets and notices through CMake. The current Docker build still needs CTranslate2/ONNX Runtime dependency packaging to enable this path. Details and remaining limitations are in `tests/fixtures/stt_api_README.md`. Native LLM input lifecycle has 640 source-block comparisons, managed HTTP checks and both real-model profiles in `tests/fixtures/stt_pipeline_README.md`.

Set `EZLOCALAI_LIBTORCH_ROOT` to the native headers/libraries of Torch 2.9.1 to enable `ezlocalai_stt_torch`. For CUDA builds, also supply `EZLOCALAI_CUDA_INCLUDE_DIRS` with the matching CUDA SDK include directories (a complete CUDA 12.8 SDK can use `/usr/local/cuda/include`). The native Whisper adapter then uses real Torch GPU availability, memory accounting, the source convolution probe and allocator/IPC cleanup by default, while preserving explicit callback overrides. The verified CUDA combination uses the C++ libraries shipped in Torch 2.9.1+cu128 with cuDNN 9.10.2.21; the standalone LibTorch archive bundles a different cuDNN version. No Python runtime is linked or executed. Production dependency packaging remains incomplete.

`stt_torch_unavailable_reference` checks native behavior with GPUs hidden. Enabling `EZLOCALAI_STT_TORCH_GPU_TESTS=ON` adds 18 real-GPU platform comparisons and 15 complete GPU float16 STT wrapper calls. Set `EZLOCALAI_STT_CUDA_LIBRARY_PATH` to a colon-separated path containing the pinned cuDNN/cuBLAS/CUDA runtime library directories, as in the reference CUDA image. The platform cases compare exact memory accounting, CPU/CUDA RNG states, restored cuDNN flags, constructor fallback and cleanup. Both real-model runners require the models to remain on CUDA. These optional GPU captures are pinned to the tested RTX 3090 Ti; other hardware, NVML availability mode, fork behavior, ROCm and CPU-only LibTorch builds remain unverified. Details and native library hashes are in `tests/fixtures/stt_torch_README.md`.

`hub_snapshot_reference` compares 37 snapshot and 27 single-file HTTP/cache scenarios against huggingface_hub 1.5.0, 504 faster-whisper alias/retry cases and 30 repository-validation cases. The native downloader handles mutable references, offline snapshots, shared blobs, range resume, redirects and concurrent file acquisition, and has downloaded the pinned tiny English model from the public Hub with matching hashes. Complete platform coverage, microphone streams, service pools/API integration and native diarization remain separate work. Xet and other unverified Hub behavior are listed in `tests/fixtures/hub_snapshot_README.md`; these checks do not establish full speech or service parity.

Missing or non-file `tokenizer.json` paths now select `openai/whisper-tiny` or `openai/whisper-tiny.en` from the engine's multilingual flag. The fallback uses the default Hub single-file cache independently of the model download root. `whisper_tokenizer_loader_reference` compares 216 constructor-path/download cases; `whisper_tokenizer_fallback_model_reference` compares eight real constructors and 32 wrapper calls using pinned OpenAI tokenizers. Native public downloads match both pinned hashes. Arbitrary tokenizer parsing/errors and Hub transport remain incomplete; see `tests/fixtures/whisper_tokenizer_fallback_assets/README.md`.

The native STT adapter reads `preprocessor_config.json` after loading the engine and tokenizer. Its sample rate, feature bands, FFT, hop and chunk settings drive resampling, feature extraction and controller timing. `whisper_preprocessor_reference` adds 64 file/constructor and 20 feature cases; the controller suites add 80 standard and 30 batch cases with configured timing. The optional `whisper_preprocessor_model_reference` compares 18 constructors and 48 real-model wrapper calls. It preserves invalid feature-shape error categories/messages across the CTranslate2 ABI boundary. Arbitrary malformed/numeric metadata and persistent direct-backend chunk overrides remain incomplete; provenance and scope are in `tests/fixtures/whisper_preprocessor_assets/README.md`.

`llm_output_reference` compares 2,632 cases from the reference's unchanged LLM response methods. Managed text/vision responses clean the first buffered choice, preserve the completion message-to-text fallback, and report the configured source model. Tests also exercise malformed-response fallback and outputs larger than the queue's C ABI buffer.

`llm_request_reference` compares 6,183 source-derived request and model-setting cases. The local wrapper filters inference fields, applies model-specific sampling/thinking profiles, and captures `LLM_MAX_OUTPUT_TOKENS` when each model loads. API requests retain their separate 8,192-token default. The pinned binding rejects local completion streaming without a callback; this enters the reference post-inference fallback path. Chat streaming uses a separate callback path. Full multimodal preparation, local stream/cancellation behavior and model-output equivalence across supported configurations remain open.

`llm_stream_reference` compares 4,720 cases from the unchanged stream helpers and complete chat generator, plus 96 managed API checks using the compiled pinned callback wrapper. The generator forces `stream=true` at the engine boundary, flattens callback batches, normalizes nested content/reasoning values, retains existing OpenAI chunks, delays text-free final chunks and reports empty streams with source diagnostics. Callback errors preserve preceding chunks and end without `[DONE]`; invalid binding JSON stops callbacks without delivering the binding's locally constructed error. Normalized output uses the API's JSON escaping and spacing. The same rules apply to successful and error-status engine responses.

Local chat now returns a lazy C++ stream through the request queue. Generation starts on first iteration, delivers chunks incrementally, and emits five-second keepalives during idle waits, including before engine response headers arrive. Model/replica leases stay with the generator through exhaustion or close. Closing waits up to five seconds; a single-slot lease remains held until native completion if cancellation is still pending. Discarding an unstarted response releases its lease without generating. `llm_stream_lifecycle_reference` checks executed source generator/cleanup traces and actual reference ASGI disconnect behavior, plus fragmented HTTP/SSE delivery, concurrent admission, keepalives, cancellation and owner destruction. With the tested Starlette 0.49.3/AnyIO 4.11.0/Uvicorn 0.40.0 stack, disconnect cleanup first waits for a pending generator read to yield, then starts its bounded close. Complete framework/version/backpressure behavior, transport codec/failure edges and real GPU cancellation remain unverified.

## Run

```bash
./build/ezlocalai-native --host 0.0.0.0 --port 8091
```

The default lookup table mirrors all current `Globals.py` defaults. Runtime use of every setting is still incomplete; see `parity_manifest.tsv`. Common settings include:

- `EZLOCALAI_API_KEY`
- `DEFAULT_MODEL`
- `QUANT_TYPE`
- `HOST`
- `PORT`
- `MAX_QUEUE_SIZE`
- `MAX_CONCURRENT_REQUESTS`
- `N_PARALLEL`
- `REQUEST_TIMEOUT`
- `EZLOCALAI_PORT`
- `MODELS_PATH`
- `ROUTER_URL`
- `ROUTER_API_KEY`
- `EZLOCALAI_ROUTER_REQUEST_GZIP`
- `WORKER_LABEL`
- `WORKER_HEARTBEAT_INTERVAL`
- `WORKER_TUNNEL`

`PORT` is the Python-compatible server port variable. `EZLOCALAI_PORT` is still accepted as a native compatibility override.
Native loads `.env` from the current working directory before reading config and does not override variables already present in the process environment. Set `EZLOCALAI_DOTENV_PATH` to point at a different dotenv file, or set it to an empty string to skip automatic dotenv loading.
File parsing follows the pinned python-dotenv 1.2.2 behavior for quoted/multiline values, escapes, comments, `${NAME}` and `${NAME:-default}` interpolation, and duplicate assignments. Process environment values take precedence; otherwise the last file assignment wins. `PYTHON_DOTENV_DISABLED` also disables loading. The CUDA entrypoint loads the file before supplying defaults. Invalid UTF-8 or unrepresentable environment values stop executable startup; diagnostics do not print configuration values. Native retains its explicit file/current-directory controls rather than Python stack-based file discovery.
`N_PARALLEL` remains the llama.cpp slot setting. If `N_PARALLEL` is unset, native uses `MAX_CONCURRENT_REQUESTS` as the queue concurrency fallback for compatibility with older/local `.env` files.

## Router Worker

Worker reports include the native runtime version. Overrides use
`EZLOCALAI_VERSION`, `EZLOCALAI_COMMIT`, then `GIT_COMMIT`; version files and Git
metadata are fallbacks. CUDA images without Git metadata identify the native
project version plus a runtime-source fingerprint. The result is cached for the
process lifetime. Loaded model files supply quantization metadata, local embedding
models and aliases report context/quantization, and loaded projectors add persistent
vision capability. See [worker_reporting_README.md](tests/fixtures/worker_reporting_README.md).

When `ROUTER_URL` is set and `ROUTER_MODE` is not true, native starts the same worker-side router client as the Python service: it registers on startup, sends resource/model/slot heartbeats, and deregisters during shutdown. `ROUTER_API_KEY` is used for router auth, falling back to `EZLOCALAI_API_KEY`; `WORKER_LABEL` controls the display label.

`EZLOCALAI_ROUTER_REQUEST_GZIP` defaults to `false`. Set it to `true` in a
worker's `.env` to let a supporting WorkerRouter compress large direct text
requests to that worker. Enable it where measured transfer savings outweigh
codec cost; fast LAN and loopback workers can benefit from leaving it off.
The worker reports the enabled encoding in every registration and heartbeat.
Compression preserves the decoded JSON bytes, uses bounded single-member gzip,
and covers `/v1/chat/completions` and `/v1/completions`; media and reverse tunnels
retain their existing transport. Authenticated gzip decoding remains available
independently of this automatic-router opt-in. Recreate the container to apply
an environment-file change. See the [measured tradeoffs](benchmarks/request-gzip-20261026.md).

Worker capabilities follow the reference enable switches and delegation rules, including embeddings, music, and music video. Model names now use whisper `large-v3-turbo`, Qwen TTS, and the Qwen3 embedding alias by default. Advertising a configured capability does not establish that its local inference implementation is complete; endpoint gaps remain in the manifest.

When `WORKER_TUNNEL=true`, native registers with a `tunnel://<worker_id>` URL and opens an outbound WebSocket to `<ROUTER_URL>/v1/router/tunnel`. The tunnel supports router ping/pong and HTTP-over-WebSocket request forwarding back to the local native HTTP server, including chunked local response forwarding.

Requests run concurrently while the WebSocket reader handles keepalives and cancellation. Disconnects and shutdown interrupt local HTTP requests; an outgoing queue bounds buffered response data. The worker forwards request/response headers, including authorization and audio format headers, and reports truncated responses as errors. Registration uses the resource manager's visible-device VRAM totals and live per-model context limits.

The tunnel's local socket timeout has a 600-second minimum and honors larger
`REQUEST_TIMEOUT` values. Video requests also honor `VIDEO_GENERATION_TIMEOUT`
(default 3,600 seconds); music requests honor `ACE_STEP_TIMEOUT` (default 1,800
seconds); music-video uses both. Configure the router's request budget to
cover the complete job as well. This corrects the Python tunnel's fixed
ten-minute limit, which could abandon generation while the router still waited.

The heartbeat interval is captured when registration starts, with a two-second
minimum. Initial registration failures back off from two seconds by a factor
of 1.5, capped at 60 seconds. After a heartbeat failure, the worker immediately
tries registration, then waits the captured interval before the next heartbeat,
even when registration fails. Router HTTP calls share a total request deadline
(10 seconds by default; `WORKER_HEARTBEAT_TIMEOUT` retains the native override),
redirect handling and TLS verification. As in Python, a successful status is
accepted even if its optional JSON response body is malformed or incomplete.
The router credential authenticates registration; the worker's own API key is
advertised for callbacks so distinct credentials continue to work.

To run the integration test against a built Rust WorkerRouter:

```bash
cmake -S native -B build \
  -DEZLOCALAI_REFERENCE_ROOT=/home/josh/repos/xtsys/ezlocalai \
  -DEZLOCALAI_WORKER_ROUTER_BIN=/opt/DevXT/WorkerRouter/target/release/worker-router
cmake --build build --target test_worker_router test_worker_tunnel
ctest --test-dir build -R '^(worker_router_integration|worker_tunnel)$' --output-on-failure
```

The integration test launches isolated worker/router processes with test credentials and temporary state. It checks direct and tunnel registration, model discovery, buffered and streamed chat, router restart recovery, completions and graceful deregistration. Echo inference isolates the wire protocol; real model validation is separate.

## Docker CUDA

The CUDA compose file runs the native ezlocalai server and builds ezlocalai-managed llama.cpp with CUDA support:

```bash
docker compose -f docker-compose-cuda.yml build
docker compose -f docker-compose-cuda.yml up -d
```

The service reads `.env` with the same ezlocalai names used by the Python container, including `DEFAULT_MODEL`, `QUANT_TYPE`, `MAIN_GPU`, `TENSOR_SPLIT`, `LLM_BATCH_SIZE`, `LLM_MAX_TOKENS`, `N_PARALLEL`, `REQUEST_TIMEOUT`, remote fallback server variables, and router worker variables. It also exposes native llama.cpp controls such as `LLAMA_CPP_MODEL_PATH`, `LLAMA_CPP_MODEL_PATHS`, `LLAMA_CPP_CONTEXT`, `LLAMA_CPP_MMPROJ_PATH`, `LLAMA_CPP_EXTRA_ARGS`, and the `LLAMA_CPP_EMBEDDING_*` variables.

To reuse an existing installation without copying its secrets or model cache, set both the Compose interpolation file and the container env file, plus the host storage paths:

```bash
export EZLOCALAI_ENV_FILE=/home/josh/repos/xtsys/ezlocalai/.env
export EZLOCALAI_MODELS_DIR=/home/josh/repos/xtsys/ezlocalai/models
export EZLOCALAI_OUTPUTS_DIR=/home/josh/repos/xtsys/ezlocalai/outputs
export EZLOCALAI_VOICES_DIR=/home/josh/repos/xtsys/ezlocalai/voices
export EZLOCALAI_WHISPERCPP_DIR=/home/josh/repos/xtsys/ezlocalai/whispercpp
docker compose --env-file "$EZLOCALAI_ENV_FILE" -f docker-compose-cuda.yml up -d --build
```

The existing worker must be stopped before starting its replacement on the same host port and GPU. These overrides only select existing files/directories; they do not edit the original `.env`.

For local transcription, native can call a whisper.cpp CLI through `WHISPER_CPP_BIN` or `WHISPER_CPP_CLI`; model resolution checks `WHISPER_MODEL_PATH`, `WHISPER_CPP_MODEL_PATH`, `WHISPER_MODEL`, `WHISPER_CPP_MODEL_DIR`, `whispercpp/models`, `/app/whispercpp/models`, and `MODELS_PATH`.
Audio data URI speech uses the native STT/TTS services described above. The CUDA and CPU images package their native dependencies. PDF-to-speech is outside the replacement scope.

Models and Hugging Face downloads persist under the existing host `./models` volume:

```bash
MODELS_PATH=/app/models
HF_HOME=/app/models
HF_HUB_CACHE=/app/models
LLAMA_CACHE=/app/models/.llama.cpp
```

If no explicit local GGUF path is configured, C++ acquires the model using the reference's per-model directory, quantization and optional projector selection, then starts managed `llama-server` with `-m`. Acquisition runs once for preliminary metadata and again for each construction or retry. Restoring an unloaded instance repeats selection with its captured quantization setting. `MODELS_PATH` can override the default `./models` root; explicit model and projector paths remain available.
The native executable and CUDA compose configuration preload configured LLMs by default before reporting ready. Voice-only mode and an explicit text-server URL skip local LLM preload; `VOICE_SERVER=true` together with `TEXT_SERVER=true` retains it. Set `EZLOCALAI_PRELOAD_MODEL_ON_START=false` to request lazy first-request startup. Startup initializes LLMs, Qwen TTS, STT and embeddings in that order, so embedding placement observes the resident speech pools. Enabled local Wan and Qwen-image wrappers are then prepared in that order when their handoff policies permit. Wan keeps no model weights resident; Qwen image preparation checks/downloads its four assets. Both start a fresh `sd-cli` for each generation, matching Python. LTX persistent pipeline preload remains unsupported and uses lazy generation.
`LLAMA_CPP_START_TIMEOUT` defaults to `3600` in CUDA compose and governs engine startup after native acquisition. Hub requests have separate downloader timeouts.
The default CUDA build targets compute capabilities 8.6, 8.9 and 12.0 and builds llama.cpp in parallel. llama.cpp, stable-diffusion.cpp and ACE-Step are built with `GGML_NATIVE=OFF`, a portable AVX2/FMA/F16C x86-64 baseline, so the image runs on any fleet CPU rather than only on CPUs like the build host. The GB10 image (`docker-compose-gb10.yml`) uses the same Dockerfile with `EZLOCALAI_PLATFORM=gb10`. Override `EZLOCALAI_LLAMA_CPP_CMAKE_ARGS` or `EZLOCALAI_LLAMA_CPP_JOBS` in `.env` for other architectures or job counts.

The default engine revision is llama.cpp release v0.7.0 (`620d2225cc436ab0b001e3200e590e3b286ef9ab`). It is newer than the Python reference’s xllamacpp engine, `aa94f2086147cd4bfbcff818ce998dffce5fcb39`. The C++ management tool, CUDA Dockerfile and Compose build use the same default; `EZLOCALAI_LLAMA_CPP_REF` remains an explicit override. To rebuild that revision in the CUDA image:

```bash
EZLOCALAI_LLAMA_CPP_REF=620d2225cc436ab0b001e3200e590e3b286ef9ab docker compose -f docker-compose-cuda.yml build
```

### CUDA package architecture gate

The CUDA, GB10, CPU, Raspberry Pi, ROCm and Jetson Compose profiles preserve the corresponding
Python profile's declared defaults and empty/override semantics, with the Pi
model correction documented below. The host-only
`compose_profiles` test checks 1,875 captured settings using a clean
environment and no `.env`. Configure the repository-root CMake project to run
this test; the native-only build does not need Docker Compose. Capture provenance
is in [compose_profiles_README.md](../tests/fixtures/compose_profiles_README.md).

Default RTX and GB10 builds inspect the CUDA code in the actual packaged
llama.cpp, stable-diffusion.cpp, ACE-Step, CTranslate2 and LibTorch binaries,
and check the ELF CPU architecture of those binaries and both worker
executables. Our engines must contain native targets for 8.6/8.9/12.0 (RTX)
or 12.1 (GB10). Third-party libraries may use compatible cubins or PTX.
The gate distinguishes architecture-specific `a` code from forward-compatible
code: an `sm_120a` target alone cannot satisfy GB10.

This prevents missing build targets; it does not establish inference quality
or GPU execution on hardware unavailable to the build host. NVIDIA documents
the [cubin/PTX and architecture-family compatibility rules](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/).

For a custom CUDA architecture outside these fleet profiles, retain the
existing engine-architecture overrides and explicitly set
`EZLOCALAI_CUDA_FLEET_PROFILE=` to disable only this fixed-fleet inventory
check. The other packaged tests still run. Such an image has no default-fleet
coverage claim.

`EZLOCALAI_NATIVE_JOBS` limits worker/test compilation independently of the
engine build jobs. This allows a lower memory budget for emulated builds
without invalidating cached engine builds.

## Prebuilt images from the LAN registry (cutover)

The registry host publishes the native images separately from the Python
`devxt/ezlocalai-*` images. Published images are `:latest` only:

| Host | Image | Compose file | Variable |
| --- | --- | --- | --- |
| x86_64 NVIDIA (RTX 3090/4090/5090; sm_86/89/120) | `192.168.1.243:5000/devxt/ezlocalai-native-cuda:latest` | `docker-compose-cuda.yml` | `EZLOCALAI_CUDA_IMAGE` |
| GB10 (aarch64, sm_121) | `192.168.1.243:5000/devxt/ezlocalai-native-gb10:latest` | `docker-compose-gb10.yml` | `EZLOCALAI_GB10_IMAGE` |

Before you start, the host needs:

- the Docker daemon to trust `192.168.1.243:5000` as an insecure registry;
- NVIDIA Container Toolkit;
- a driver that supports the image's CUDA runtime: 570 or newer for the
  CUDA 12.8 x86 image, and a CUDA 13 driver for GB10.

**Cut over an existing Python worker.** Check out this repository on the
branch being deployed, here at `/path/to/native-ezlocalai`. Keep the Python
installation in place; here it is at `/path/to/python-ezlocalai`. The native
worker reuses Python's `.env`, models and voices.

The repository's `voices/` folder holds the same 22 default voices as the
Python repository (`default.wav` and the named voices), byte for byte, and
the images include them. Compose mounts `EZLOCALAI_VOICES_DIR` (default
`./voices`) over `/app/voices`. Pointing it at Python's folder keeps any
uploaded voices; an empty folder hides the defaults, as in Python.

Both workers use port 8091, the same GPU and the container name `ezlocalai`.
Remove the Python container first with `down`; its image stays for rollback.

If Python's `.env` sets `EZLOCALAI_CUDA_IMAGE` or `EZLOCALAI_GB10_IMAGE` to the
Python image, the shell `export` below takes precedence for Compose
interpolation.

```bash
cd /path/to/native-ezlocalai
export EZLOCALAI_ENV_FILE=/path/to/python-ezlocalai/.env
export EZLOCALAI_MODELS_DIR=/path/to/python-ezlocalai/models
export EZLOCALAI_OUTPUTS_DIR=/path/to/python-ezlocalai/outputs
export EZLOCALAI_VOICES_DIR=/path/to/python-ezlocalai/voices
export EZLOCALAI_WHISPERCPP_DIR=/path/to/python-ezlocalai/whispercpp
export EZLOCALAI_CUDA_IMAGE=192.168.1.243:5000/devxt/ezlocalai-native-cuda:latest
# GB10: export EZLOCALAI_GB10_IMAGE=192.168.1.243:5000/devxt/ezlocalai-native-gb10:latest

(cd /path/to/python-ezlocalai && docker compose -f docker-compose-cuda.yml down)   # GB10: docker-compose-gb10.yml
docker compose --env-file "$EZLOCALAI_ENV_FILE" -f docker-compose-cuda.yml pull ezlocalai
docker compose --env-file "$EZLOCALAI_ENV_FILE" -f docker-compose-cuda.yml up -d --no-build ezlocalai
```

On GB10, use `docker-compose-gb10.yml` and `EZLOCALAI_GB10_IMAGE` in the same
commands. `--env-file` is required, because this repository has no `.env` of
its own for Compose to read.

**Verify the cutover:**

- `curl -s localhost:8091/health` returns `{"status":"healthy"}` once the model
  has loaded.
- The router's `/v1/router/workers` lists the host's `WORKER_LABEL` with a
  recent heartbeat, the deployed commit as `version`, and the same models,
  capabilities and context as before.
- A routed chat pinned with `"worker": "<label>"` returns 200.

**Roll back:**

```bash
cd /path/to/native-ezlocalai
docker compose --env-file "$EZLOCALAI_ENV_FILE" -f docker-compose-cuda.yml down
(cd /path/to/python-ezlocalai && docker compose -f docker-compose-cuda.yml up -d --no-build)
```

## GB10 (Grace Blackwell) image

`docker-compose-gb10.yml` builds the same Dockerfile for aarch64 with
`EZLOCALAI_PLATFORM=gb10`. That selects:

- a CUDA 13.0.2 cuDNN base image and `121-real` kernels;
- pinned aarch64 dependency manifests, including torch 2.9.1+cu130 and its
  NVIDIA cu13 libraries;
- CTranslate2 built from source with CUDA.

Its environment defaults match the Python `docker-compose-gb10.yml`, for
example two text, TTS, STT and embedding slots, `MEDIA_KEEP_MODELS_LOADED=true`
and a 120 s request timeout.

Build on the GB10 itself with the full packaged test gate:

```bash
docker compose -f docker-compose-gb10.yml build
```

For an x86 build using ARM64 emulation (QEMU binfmt), explicitly exclude the
four checks that need native timing or `PR_SET_CHILD_SUBREAPER`. This is an
emulation exception, not the default production build:

```bash
EZLOCALAI_PACKAGE_TEST_EXCLUDE='^(dotenv_startup|tunnel_framing_reference|ltx_runtime|media_engine)$' \
  docker compose -f docker-compose-gb10.yml build
```

To cut over, stop the Python container on the GB10 and point the native worker
at the existing installation, using the same overrides as the CUDA cutover:

```bash
export EZLOCALAI_ENV_FILE=/path/to/ezlocalai/.env
export EZLOCALAI_MODELS_DIR=/path/to/ezlocalai/models   # likewise OUTPUTS, VOICES, WHISPERCPP, WAKEWORD_* dirs
docker compose --env-file "$EZLOCALAI_ENV_FILE" -f docker-compose-gb10.yml up -d
```

## CPU and Raspberry Pi images

`docker-compose.yml` builds the x86-64 CPU worker.
`docker-compose-rpi.yml` selects Linux ARM64 for Raspberry Pi 5 and other
64-bit ARM hosts. Both use `cpu.Dockerfile`; the build target architecture
selects checksum-pinned native CTranslate2, ONNX Runtime, CPU LibTorch and
audio libraries. Qwen TTS, Whisper, wake-word training and the same native
LLM/media engines are included. Neither profile requests a GPU.

The Pi profile uses four build jobs and preserves its Python settings,
including 512-token batches, one request slot, lazy speech loading and a
600-second request timeout. Its default model is corrected to the published
[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF):
the source profile's 0.6B repository was unavailable during verification.
Explicit `DEFAULT_MODEL` values remain unchanged. The Compose reference test
records this exception separately from the captured source defaults.

Build on a 64-bit ARM host:

```bash
docker compose -f docker-compose-rpi.yml build
docker compose -f docker-compose-rpi.yml up -d
```

Use the same environment-file and persistent-directory overrides described
for CUDA when replacing an existing installation. The native profile also
persists wake-word jobs, samples and trained models. Engine builds disable
host-native CPU tuning, so an emulation host cannot select unsupported
instructions for the deployment host. An x86 build requires ARM64 binfmt/QEMU;
emulation does not establish physical Pi performance or memory suitability.
The full package test gate is the default.

## ROCm image

`docker-compose-rocm.yml` uses the Python ROCm profile's settings and AMD
device access with a native C++ worker. It uses pinned ROCm 6.4.1 and
Torch 2.9.1+rocm6.4 libraries. The native dependency extraction includes
the ROCm kernel assets; Python bindings are excluded.

The image passes all 60 package checks and actual CPU chat, vision, embeddings,
Whisper and Qwen speech through Rust, with direct and tunnel coverage. The
HIP-configured worker also passes simulated-device integration and CPU training
comparisons. Physical AMD GPU execution remains unverified.
The default `EZLOCALAI_HIP_ARCHITECTURES=gfx1100` matches the source
profile's `HSA_OVERRIDE_GFX_VERSION=11.0.0`. Set both for the actual AMD
hardware when building another target; the default is not a universal AMD
binary. `HIP_VISIBLE_DEVICES` retains the source default of `0`.

```bash
docker compose -f docker-compose-rocm.yml build
```

The profile reuses the same environment-file and persistent-directory
overrides as CUDA and CPU. Its package gate includes HIP device discovery,
worker reporting, speculative launch arguments, SDK ABI and LibTorch tests.
NVIDIA-captured policy tests retain their NVIDIA fixture adapter; separate
HIP tests exercise the AMD adapter. No AMD model inference is implied by
those simulated-device tests.

## Jetson Orin candidate image

`docker-compose-jetson.yml` targets Orin with JetPack 6.1 / L4T R36.4,
CUDA 12.6 and `sm_87`. It retains the Python profile's 79 settings and
NVIDIA container runtime. The native image is still being built and has
not passed package or hardware acceptance.

The image builds pinned LibTorch 2.9.1 and CTranslate2 for Orin. A generic
ARM/SBSA Torch wheel is insufficient: the inspected CUDA 12.6 wheel contains
`sm_80`/`sm_90` code, while NVIDIA excludes Tegra from desktop cubin compatibility.
The Jetson package gate therefore requires native `sm_87` in every checked
CUDA library, including LibTorch. See NVIDIA's
[architecture compatibility rules](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/).

Pinned CMake and JSON headers supply versions newer than Ubuntu 22.04's
packages. FFmpeg 6.1.2 supplies the resampler API used by native Whisper;
OpenCV builds against those same libraries. Python is used by upstream
Torch build tooling; the worker and its speech adapters are native C++.

```bash
docker compose -f docker-compose-jetson.yml build
```

Build on Orin for practical compilation times. `EZLOCALAI_TORCH_JOBS`,
`EZLOCALAI_LLAMA_CPP_JOBS` and `EZLOCALAI_NATIVE_JOBS` each default to two.
The default package gate has no exclusions. Nano, Xavier and JetPack 5.x
require separate dependency sets; changing only a CUDA target or base tag
does not establish support. Environment-file and persistent-directory
overrides match the other native profiles.

## llama.cpp Backend

The app is still ezlocalai. llama.cpp is the native local LLM runtime underneath ezlocalai's existing API surface. Native defaults to llama.cpp for local text and embedding inference; the echo backend is only an explicit test/development opt-out with `EZLOCALAI_NATIVE_BACKEND=echo`.

Point native at either an already-running `llama-server`:

```bash
LLAMA_CPP_SERVER_URL=http://127.0.0.1:8080 \
./build/ezlocalai-native --host 0.0.0.0 --port 8091
```

or let native start `llama-server` for local GGUF files:

```bash
DEFAULT_MODEL=unsloth/Qwen3.5-4B-GGUF \
LLAMA_CPP_MODEL_PATH=/data/models/qwen3.5-4b-q4_k_m.gguf \
./build/ezlocalai-native --host 0.0.0.0 --port 8091
```

For multiple comma-separated `DEFAULT_MODEL` values, set `LLAMA_CPP_MODEL_PATHS` to the matching comma-separated GGUF paths. Native starts one `llama-server` per model on consecutive ports beginning at `LLAMA_CPP_SERVER_PORT` (default `8092`) and passes `--alias` so llama.cpp reports the public ezlocalai model name. Without explicit paths, native acquires each model and optional projector using its captured quantization setting before starting the engine. `LLAMA_CPP_EXTRA_ARGS` is appended to the `llama-server` command for hardware flags such as `-ngl`, `-np`, cache settings, or chat-template flags.

When `GPU_LAYERS` is unset, native follows ezlocalai's Python policy explicitly: it starts managed llama.cpp with `-ngl -1` when NVIDIA VRAM is available and `-ngl 0` when no usable GPU is detected. `GPU_LAYERS=-2` is normalized to llama.cpp's all-layers value, `-1`.

Embeddings use a separate llama.cpp server configured by `EMBEDDING_MODEL` (default `Qwen/Qwen3-Embedding-0.6B-GGUF`), `EMBEDDING_MODEL_ALIAS` (default `Qwen3-Embedding-0.6B`), and `EMBEDDING_QUANT_TYPE` (default `Q8_0`). Default acquisition follows `LLM.download_model`: use `models/<repository basename before -GGUF>`, prefer an existing case-sensitive quant match, then list Hub files and try the requested quant followed by `Q4_K`, `Q5_K`, `Q6_K`, and `Q8`. Embeddings skip projector downloads. The C++ downloader acquires the file before memory estimation and starts llama-server with `-m`. Explicit `MODELS_PATH` discovery and `LLAMA_CPP_EMBEDDING_MODEL_PATH` overrides remain available. For an explicit local file:

```bash
LLAMA_CPP_EMBEDDING_MODEL_PATH=/data/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
./build/ezlocalai-native --host 0.0.0.0 --port 8091
```

When native starts the embedding server, it passes llama.cpp's `--embedding` flag. Use `LLAMA_CPP_EMBEDDING_SERVER_URL` for an already-running embedding server, `LLAMA_CPP_EMBEDDING_SERVER_PORT` for the managed port, `LLAMA_CPP_EMBEDDING_ALIAS` for the public model name reported through ezlocalai, `LLAMA_CPP_EMBEDDING_CONTEXT` for `-c`, `LLAMA_CPP_EMBEDDING_POOLING` for `--pooling`, `LLAMA_CPP_EMBEDDING_UBATCH` for `-ub`, and `LLAMA_CPP_EMBEDDING_EXTRA_ARGS` for embedding-only server flags.

Each embedding instance uses one full-context slot, LAST pooling, 10000 context tokens, 512 batch/ubatch, and f16 KV cache by default. `EMBEDDING_N_PARALLEL` controls the number of independent instances; owned processes use consecutive ports starting at `LLAMA_CPP_EMBEDDING_SERVER_PORT`. Resident instances are reused in FIFO order, and excess requests wait for an available instance. `EMBEDDING_KEEP_LOADED=false` creates and destroys a transient instance per request. Enabled local resident pools warm-load before the executable starts serving HTTP; a failed warm load leaves successful instances available and missing instances can load on demand.

`EMBEDDING_CONTEXT_LENGTH`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_UBATCH_SIZE`, `EMBEDDING_KV_CACHE_TYPE`, `EMBEDDING_GPU_LAYERS`, `EMBEDDING_MAIN_GPU`, and `EMBEDDING_NORMALIZE` configure each instance; legacy `LLAMA_CPP_EMBEDDING_*` overrides remain available. `embedding_runtime_reference` checks 569 source-derived cases with 4,167 lifecycle transitions, FIFO waiting, independent C++ processes, error cleanup, full-pool video handoff and executable startup.

For a resolved local GGUF, automatic GPU placement estimates weights, KV cache and compute buffers using the reference formula. A CUDA driver probe measures free memory on the selected visible ordinal; `EMBEDDING_VRAM_SAFETY_MARGIN` defaults to 0.75 GiB. Startup memory errors retry at 75%, 50% and 25% of the selected layer count, then CPU. CPU instances disable KV and operation offload. The engine disables automatic fitting and prompt/RAM caching. Resident GPU instances perform text and full-microbatch token inference before publication unless `EMBEDDING_WARMUP=false`; warmup failures discard the instance. Resource metadata aggregates estimated GPU memory across replicas. `embedding_placement_reference` checks 2,731 source-derived cases, 153 synthetic GGUF files, CUDA context handling and 15 process lifecycle scenarios. The probe was also exercised on the host GPU; model inference remains unverified.

Before loading an embedding instance, the C++ resource manager applies the reference's 2 GiB estimate and 1 GiB aggregate safety margin. With sufficient aggregate memory, it prefers the capability-ranked secondary GPU if that GPU has 2 GiB free, otherwise the GPU with most free memory or a tensor-split recommendation. A specific `cuda:N` recommendation overrides `EMBEDDING_MAIN_GPU`; a generic CUDA or CPU recommendation leaves the per-model estimator in control. Registration preserves active requests, reports the repository/model name, and updates aggregate residency before destroying instances. `/v1/resources` and fallback decisions use CUDA-visible memory. `torch_cuda` reports device memory and zero in-process PyTorch allocator counters because the native engines run in separate processes. Resource policy and state have 4,449 source-executed cases plus CUDA/API and concurrency checks. Other modalities' registration, full lifecycle/pool telemetry and worker-registration GPU reporting remain incomplete.

Owned text/vision engines now become visible as resident when startup succeeds, before their first inference response, and retain that state after ordinary inference errors. Shared GPU handoffs clear residency; successful restoration restores it, and failed restoration leaves it absent. `/v1/resources` includes `model_lifecycle` with configured/loaded/active models, captured context and supported engine placement/batch settings, load/unload summaries and the latest 100 events. The C++ recorder matches source rounding, ordering, failure counts and raw-duration averages in 818 source-executed cases. C++ process checks cover pending and concurrent first inference, text/vision failure and recovery, handoff cleanup, load order and repeated restoration. External servers become visible after a successful response; their configured URL alone does not establish residency. The lifecycle snapshot currently reports resident mode and an empty VRAM-estimate map. Full text/vision resource registration and VRAM estimates, automatic/forced swap policy, speculative engine execution, the complete resilient startup policy, all runtime metadata, image lifecycle hooks and model pool diagnostics remain incomplete.

Managed text/vision startup now performs the reference one-token completion (`{"prompt":"Hi","max_tokens":1}`) after HTTP readiness and before publishing residency. A dictionary containing `error` fails even when its value is null, false, zero or empty; non-dictionary results and top-level status dictionaries without that key retain the source constructor’s acceptance behavior. Failure diagnostics preserve the model prefix and Python value representation. Failed startup checks reap their child process; lifecycle accounting records one outcome for the complete resilient load. Resident reuse skips the check and each restoration runs it again. `llm_startup_reference` compares 741 source-constructor cases and exercises blocked startup, cleanup, retry, text/vision residency and restoration using C++ processes. The C++ resilient-load controller now follows the source retry sequence: halve physical micro-batches down to 128 while retaining requested GPU layers, then binary-search isolated fit probes and reload the maximum fitting placement once. Successful probes are disposed before the final load. Source resource/opaque-error classification and explicit CPU short-circuit behavior are preserved. Known GPU placement is reused only for matching context; physical micro-batches are recomputed on restoration. One unrounded total deadline covers each fit probe, including readiness and startup inference, using `LLM_LOAD_PROBE_TIMEOUT` with the source 30-second minimum. `llm_load_reference` compares 3,335 source-executed traces and checks managed process lifetimes, final-reload errors and timeout cleanup. Automatic GPU batch sizing now uses the source policy described below. Complete placement/acquisition, startup API error mapping and preload lifecycle behavior remain incomplete.

Text/vision batch sizing now reads free memory and compute capability on the selected CUDA-visible device, applies the source GPU-generation thresholds and T4/A100 profiles, and preserves the CPU and failed-probe defaults. Long-context MTP limits use the source model-name rules, including Qwen3.8’s capacity/context exception. `LLM_BATCH_SIZE` and `LLM_UBATCH_SIZE` accept Unicode decimal digits and separators, preserve signed values, apply distinct invalid-setting fallbacks and clamp the physical micro-batch to the logical batch. Retry setup refreshes automatic sizing after the first failed construction, then applies the source retry-helper context clamp. `llm_batch_reference` compares 7,242 executed reference cases, including probe order and reason strings, and verifies CUDA-visible ordering, probe failures and managed engine flags/runtime metadata. The tested policy uses the reference runtime’s default 4,300-digit decimal conversion limit. Exact errors at the xllamacpp/llama-server parameter boundary and real-model throughput/output equivalence remain unverified.

Managed text/vision loads resolve host prompt-cache size from `LLM_PROMPT_CACHE_MIB`. Automatic sizing retains the context-scaled 8–32 GiB baseline and raises the requested ceiling toward one quarter of RAM (at most 32 GiB) on hosts with at least 64 GiB. The effective ceiling still respects available RAM, `LLM_PROMPT_CACHE_RAM_MARGIN_MIB` (4096 MiB by default), and `LLM_PROMPT_CACHE_MAX_RAM_FRACTION` (0.25 by default). Explicit sizes and disabled caches retain their meaning. Each load/restoration recomputes the budget and reports `prompt_cache_requested_mib`, `prompt_cache_mib` and `prompt_cache_swap_allowance_mib` in `/resources` model lifecycle telemetry. These are ceilings, not allocated bytes or cache-hit rates.

The Linux probe includes psutil’s missing/zero `MemAvailable` and zone-watermark fallbacks, then constrains RAM and free swap by every visible cgroup-v2 ancestor. Reclaimable inactive file pages count toward container RAM headroom. Container swap limits follow the kernel’s [`memory.swap.max` accounting](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory-interface-files).

Swap supplementation is **off by default**. To permit up to 4 GiB of existing free swap for cold cache states, set `LLM_PROMPT_CACHE_SWAP_MIB=4096` in the worker `.env`. `LLM_PROMPT_CACHE_SWAP_MARGIN_MIB` reserves 1024 MiB by default; the allowance is bounded by free host swap and remaining cgroup swap capacity. This creates no swap files and changes no OS settings. Swap never rescues an exhausted RAM budget, and an explicit cache size remains the overall ceiling. Swap-backed states preserve their original precision, but restoring cold pages can take longer. `LLM_PROMPT_CACHE_ALLOW_UNSAFE=true` retains the legacy safety-cap bypass and does not opt into these automatic RAM/swap extensions.

The selected ceiling is passed to llama.cpp’s `--cache-ram`. Nonempty string `prompt_cache_key` values now reach the managed engine as conversation hints. The managed patch preserves outgoing named conversations even when their prompts share most of their prefix, restores the longest useful cached prefix, and protects the requested state during bounded eviction. When two states cannot fit, it skips the outgoing save instead of evicting the state being requested. No number of retained conversations is guaranteed beyond the byte budget. Keys do not rewrite prompts or bypass token-prefix matching. Custom llama.cpp revisions keep their own behavior; this patch is qualified against the default pinned revision.

The guarded 1024-token MTP checkpoint path also retains a state at an existing full logical-batch start when that batch jumps into the near-end window. This covers some shortened conversations that precede all tail checkpoints without adding any physical batch split. It requires at least five checkpoint slots, keeps the configured count and cache byte ceiling, and charges the additional state to that budget. See [checkpoint entry-window behavior](patches/llama-checkpoint-entry-window.md) for eligibility and memory costs.

Chat engine requests explicitly set `cache_prompt=true`, including when a caller supplies false. Completion and embedding startup retain their separate cache behavior.

KV precision follows `KV_CACHE_TYPE_<GPU family>` before `KV_CACHE_TYPE`, with the source distinction between empty and whitespace card overrides. Automatic selection is `q4_0`; accepted explicit types are `q4_0`, `q8_0`, `f16` and `f32`. Invalid values fail before launching the engine. `llm_cache_reference` compares 9,111 source-policy/Linux RAM cases and checks managed launch arguments, requested/effective telemetry, chat payloads, restoration, invalid-setting cleanup and matching engine build defaults. Real-model cache reuse, native parameter-range errors, the complete engine constructor and real checkpoint reuse remain unverified.

The managed CUDA engine automatically fuses Q4_0 KV expansion into the existing F16 attention tile loader for a narrowly guarded SM89 three/four-query shape (256 channels, 24 query heads and four KV heads). It preserves the F16 operands and original attention reduction partition. Other shapes and platforms keep their existing path. Set `GGML_CUDA_FATTN_Q4_KV_FUSION=0` in `.env` and restart to disable it; exact `1` enables the same path with diagnostics, while invalid values disable it. Automatic mode has no diagnostic counters or exit log. This targets MTP verification, rather than bulk prompt prefill, and retains the existing workspace allocation. See [the fusion contract](patches/llama-cuda-q4-kv-fusion.md) for eligibility and diagnostic limits.



Local text requests resolve public and explicit internal model names using the source precedence rules, including case-insensitive short names and preserved whitespace. Public requests select the least-busy replica with configured-order ties; explicit `#2` requests wait for that instance even when a sibling is idle. Per-model and per-replica counters reserve only available slots and release them after success or failure. The selected ID travels separately from the inference payload, and the engine receives the public source name. Configured capacity is captured at construction; ordinary `N_PARALLEL=0` uses one slot per 32,768 context tokens, bounded to 1–16. `llm_slots_reference` checks 1,084 source-executed cases plus concurrent API requests, independent model capacity and managed process/argument selection.

The C++ model catalog now expands speculative targets into separate single-slot instances according to `N_PARALLEL`. Selection follows the source Qwen3.8/MTP rules and `LLM_SPECULATIVE_TYPE` aliases. Added instances inherit their original CSV entry’s context, GPU, quantization and model-path index, and are appended after original entries. Explicit source mappings preserve literal numeric suffixes and source catalog collision behavior. `llm_catalog_reference` compares 2,216 source-executed cases covering expansion, ordering, aliases, Unicode, CSV fallback/empty fields, errors, GPU tensor configuration and resolver state; managed-process tests verify appended-instance selection and inherited launch arguments. Real-model speculative execution, preload failure tracking, swap residency, actual-runtime capacity updates, worker context/quant metadata, global queue sizing and streamed cancellation leases remain incomplete. Catalog integers currently use a signed 64-bit range; Python’s arbitrary-size integer behavior remains unimplemented.

Owned text/vision engines now enable the selected MTP or DFlash backend through the pinned llama.cpp speculative options. MTP uses GPU/model-aware draft lengths, card/global overrides, the 1–16 clamp, and model-specific probability defaults. DFlash validates lengths 1–7 and finite probabilities 0–1, resolves its pinned draft in the shared Hub cache (or `DFLASH_MODEL_PATH`), disables automatic memory fitting, and assigns the draft to the worker's visible CUDA ordinal. Draft GPU layers follow each target load/fit retry. Settings are read again on restoration. `llm_speculative_reference` compares 2,500 source-executed cases, including probe order and conversion errors; managed C++ process fixtures check launch flags, restoration, CPU/GPU selection, retry placement and float conversion. Hub tests exercise concurrent pinned draft resolution, blob reuse and offline behavior. These tests do not establish real-weight decoding/output equivalence, ROCm draft placement, large-image DFlash inference, or complete Hub authentication/transport parity.

Owned engines now receive the pinned defaults for continuous batching, warmup, 32 context checkpoints and an 8,192-token minimum checkpoint spacing. Runtime metadata includes both checkpoint values and honors existing extra-argument overrides. Source catalog assignments that isolate a worker use `--split-mode none --fit off`; runtime telemetry retains the original 128-element placement array. Qwen3.8-27B projectors receive a 1,024-image-token minimum.

Two reference binding quirks are preserved for exact behavior: nonnegative `REASONING_BUDGET` fails before startup because the pinned binding has no top-level `reasoning_budget` property, and the source's element-wise tensor-split assignments modify a returned copy, leaving the native ratios at zero. Negative reasoning budgets retain the unlimited default. `llm_parameters_reference` checks 2,794 cases generated from unchanged reference constructor statements and compiled selected binding properties, plus managed launch, restoration, validation-order and CUDA build-default checks. This does not complete automatic placement, native initialization error mapping or inference parity.

Managed text requests now load their captured per-model `LLM_MAX_TOKENS` ceiling. Long prompts and streaming estimates do not enlarge it, and context errors do not trigger an allocation above it. The source optimal-context fallback uses the maximum valid CSV value, or 40,000 after malformed/empty input; a zero configured ceiling uses that fallback or 16,384, while negative ceilings clamp to one for requests. Preload retains the separate constructor behavior for nonpositive values, including live global integer conversion and the pinned binding’s signed 32-bit context errors. Lifecycle context records the requested load rather than a stale catalog value. Inference memory retries are disabled for multiple configured models or more than one effective slot. Explicit native context overrides remain supported. `llm_context_reference` checks 1,066 source/compiled-binding cases and managed ceiling, preload, restoration, overflow and concurrency behavior. Complete fallback transport and startup/API capacity errors, image-description context transitions/reset timers, arbitrary-size catalog integers and actual runtime slot-capacity reporting remain open.

Managed inference now recognizes the pinned LLM wrapper’s buffered and chat-callback error shapes independently of HTTP status. Buffered falsey errors are ignored by the classifier; callback errors include every top-level `code` and falsey nested errors. Exception messages preserve Python value formatting and prompt/context counts. Memory errors carried with HTTP 200 can enter the guarded buffered retry loop; lazy chat errors do not reload. Typed lazy failures reach the API as an SSE `streaming_error` after preceding chunks, without `[DONE]`, with the source’s 1,000-character limit and ASCII JSON escaping. Raw truthy stream strings validated as false instead return the observed serialization failure. `llm_response_reference` checks 2,238 source wrapper cases, 12 source API event cases and managed process/API scenarios. Full fallback transport/dependency behavior, startup/API capacity errors and cancellation/slot lifetime remain incomplete.

Buffered local inference failures now enter the source post-inference fallback path, including when the request disables precheck fallback. An unconfigured fallback returns the canned HTTP-200 assistant response. Configured clients first probe ezlocalai availability, send only messages and the live default model, and preserve responses containing choices; other values become Python-style text. Failed or unavailable ezlocalai forwarding proceeds to the OpenAI-compatible path, which appends `chat/completions` to the configured base path and applies `FALLBACK_MODEL` only there. Native C++ code handles the tested SDK response extraction, error messages and retry statuses. Completion fallback reduces chat output to text/model; chat dictionary streaming iterates keys. Typed queue metadata keeps fallback dictionaries containing error fields at HTTP 200. Memory failures can use the final outer reduction after three inner attempts. `llm_post_fallback_reference` compares 252 source API/payload/request-trace cases using OpenAI 2.21.0 and HTTPX 0.28.1 for the SDK oracle; the reference leaves its OpenAI dependency unpinned. Native has no Python or SDK runtime dependency. Complete SDK HTTPX timeouts, headers, proxy/auth/TLS/redirect/cookie behavior, retries with HTTP-date headers, alternate dependency versions and numeric/Unicode extremes remain unverified or incomplete.

An optional `llm_engine_contract` test links an independently compiled pinned engine and verifies actual parser results, source hashes, speculative settings, subnormal probabilities, single-GPU/projector settings and checkpoint defaults. Configure `EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE` and `EZLOCALAI_LLAMA_CPP_REFERENCE_BUILD` to enable it; the latter must contain the engine's shared libraries in `bin`. The native build does not fetch or build that dependency implicitly. The reference build script confirms that its large-image DFlash fix is included in the pinned upstream revision, with no local patch. CUDA image defaults now match its `86-real;89-real;120-real` architecture list and disabled NCCL setting.

Also set `EZLOCALAI_LLAMA_CPP_SMOKE_MODEL` to a local copy of `ggml-org/models`'s `tinyllamas/stories260K.gguf` at revision `499bc8821c6b12b4e53c5bffcb21ec206f212d81` to enable `llm_real_engine_smoke`. The C++ test verifies SHA-256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`, serves those real weights through a loopback Hub fixture, and exercises native acquisition, CPU startup, deterministic buffered/streamed generation, unload/reload and process cleanup. It uses `LLM_MAX_TOKENS=512`, f16 KV and two CPU threads, with no native context override. This smoke test does not compare complete ezlocalai responses or establish model-output equivalence.

Fallback status checks remote availability before evaluating local memory, and uses the same 30-second availability cache as forwarding. Each application captures the fallback URL and API key on first client use, including an unconfigured client; later environment changes leave that client and its cache intact. Thresholds remain live. Explicitly empty or malformed `FALLBACK_MEMORY_THRESHOLD` values produce the reference's server error; absent values use 8 GiB. `FALLBACK_VRAM_THRESHOLD` defaults to 2 GiB, with invalid values marking the remote server unreachable. Both settings accept Python-style Unicode digits, whitespace and numeric separators. `fallback_reference` checks 936 source-executed API, cache, probe-order and JSON cases plus 134 client lifecycle scenarios over native HTTP. Complete HTTP transport behavior and numeric/codec extremes still need work; the fallback route remains partial.

Buffered fallback requests apply one monotonic deadline across DNS, connection, TLS, upload, headers and body. Timeouts match the client methods: health 5 seconds, model listing 10, embeddings 60, speech/transcription 120, images 660 and video 600. Text uses `REQUEST_TIMEOUT` (300 when absent); music and music-video use the larger of `REQUEST_TIMEOUT` and `ACE_STEP_TIMEOUT` (1800 defaults). Fractional values remain fractional; nonpositive values and NaN disable the total timer, while positive infinity fails before sending the request. Totals of at least five seconds round their absolute deadline up to the next whole monotonic second. `http_deadline_reference` covers 176 aiohttp timer fixtures, 68 real-HTTP source cases and native socket/DNS/TLS checks. Voice/router streaming paths and complete resolver/HTTP compatibility remain open.

Buffered fallback requests advertise `gzip, deflate, br` and decode those responses incrementally before JSON or audio handling. Deflate's raw-stream compatibility, truncated-stream behavior, extra gzip members, encoding header spelling/whitespace and duplicate headers follow the tested aiohttp 3.13.2 runtime with zlib 1.3 and Brotli 1.1.0. That runtime lacks Zstandard support, so a recognized `zstd` encoding fails at header parsing. `http_compression_reference` checks 1,407 decoder cases and 592 real-HTTP API cases, including binary audio, early corruption errors and deadlines that remain active after compressed-stream completion. The native build and CUDA image include zlib and Brotli dependencies; no Python codec runs in the service.

Buffered fallback clients follow 301/302/303/307/308 redirects using the reference's method/body rules, ten-redirect limit, first Location/URI header selection and one deadline for the whole chain. Authorization is removed when the scheme, canonical host or effective port changes; URL Basic credentials follow the source's Latin-1 and conflict rules. The C++ URL parser resolves relative paths, normalizes percent escapes and dot segments, handles IPv6 and uses ICU for Unicode hostnames. `http_redirect_reference` compares 1,102 Yarl 1.22.0 URL cases and 279 actual aiohttp request traces through native health, model, music, speech and transcription routes. Voice/router streaming clients, dependency/version alternatives, Happy Eyeballs connection scheduling and complete HTTP parser behavior remain open; this does not establish full transport parity.

Buffered fallback calls now keep a C++ cookie jar for the lifetime of each request/redirect chain. It matches the tested aiohttp parser and Python 3.12 cookie rules for DNS/IP hosts, domain/path/security filtering, duplicate names, quoted values, control-character rejection, expiry and replacement. Cookies are processed when response headers arrive and are discarded between client method calls. `http_cookies_reference` checks 1,159 source scenarios with 6,180 operations; another 217 real-HTTP cases in `http_redirect_reference` compare cookie headers and API outcomes across all five tested forwarding paths, including cross-origin hops, expiry during a chain and errors before a stalled body. This behavior is tied to the tested aiohttp 3.13.2/Python runtime; other dependency versions remain unverified.

Text fallback on resource precheck, queue timeout and during a shared GPU handoff now matches the pinned client/API response lifecycle. For a truthy raw `stream` value, the Python client returns an async generator after closing its HTTP response. The chat endpoint emits a `streaming_error` SSE event; the completion endpoint returns HTTP 500, and truthy string values validated as false also fail response serialization. Native preserves these observed outcomes and releases a successful upstream response after headers, retaining redirects, cookies, compression-header checks and the total deadline. Non-stream responses use the shared MIME/charset/JSON decoder. Typed forwarding results cross the native queue without losing status, media type or JSON values; a remote HTTP 200 dictionary containing `error` stays HTTP 200. `llm_fallback_reference` compares 178 source client/API cases through the native queue, and the music handoff tests cover typed streaming/error outcomes. Complete queue scheduling/error propagation and voice/router streaming transport still need work.

Before local LLM inference, the C++ queue now skips fallback when a text or vision model is resident, evaluates combined-memory thresholds, and then consults GPU admission. It forwards before reserving a local slot and retries locally when availability or forwarding fails. Per-request disabling skips those probes and is removed before local or remote inference. The shared resource tracker and the managed backend residency snapshot supply loaded-model state; full text/vision resource registration remains incomplete. Another 816 source-executed cases compare API responses and memory/admission/remote/local event order, including changing free-memory readings, idle models, malformed thresholds and probe failures. Managed-backend checks cover resident-model bypass and truthy raw stream strings validated as false: the engine receives boolean `true`, while the API serializes the yielded local chunks as a JSON array. This does not establish real-model output or complete local streaming parity.

Text, vision and embedding acquisition use C++ Hub listing and local-folder downloads through `curl`, including pagination, relative metadata redirects, signed CDN downloads without the Hub credential, commit/ETag metadata, global-cache copies, unchanged-file reuse and offline behavior. New and restored instances repeat reference file selection; matching top-level local files need no Hub request. `model_download_reference` checks 1,068 source-executed selection/call-order cases (including Unicode projector filenames), 16 `huggingface_hub` 1.28.0 HTTP traces, and download-to-placement/warmup integration using synthetic GGUFs. Twelve managed text scenarios cover captured per-model quantization, speculative replicas, wrong-quant local files, preliminary-download recovery, missing projectors, failures, layer-fit retries, offline reuse, explicit paths, custom roots and failed restoration after file removal. Complete Hub retry/timeout/authentication/Xet behavior, cross-modality resource accounting, lifecycle telemetry, cancellation/concurrent reconfiguration edges, oversized numeric metadata, engine/version compatibility and real model equivalence remain open.

`EMBEDDING_SERVER` delegates to another ezlocalai service with `EMBEDDING_SERVER_API_KEY`. The API preserves current dimensions/encoding fields on delegation, applies reference dimension trimming and normalization for local results, and keeps embedding work outside the LLM request queue. `embedding_reference` verifies 86 cases captured from executing the current reference schema and embedding methods. Real model output equivalence has not been established.

### Managed llama.cpp Checkout

ezlocalai can manage a local llama.cpp checkout without vendoring it into git:

```bash
./build/llama_cpp_tool update
./build/llama_cpp_tool build
./build/llama_cpp_tool status
./build/llama_cpp_tool model-status
```

The checkout goes to `third_party/llama.cpp` and build output goes to `build/llama.cpp`. Both stay outside the tracked ezlocalai source. The native runtime automatically looks for the managed `llama-server` binary before falling back to `llama-server` on `PATH`.

After `update` or `build`, ezlocalai writes `build/llama.cpp/ezlocalai-llama-cpp-version.txt` with the upstream repo, ref, commit SHA, and commit date. Print it with:

```bash
./build/llama_cpp_tool version
```

Use `status` when you need a quick audit of the configured upstream, checkout location, build location, current checkout SHA, version-file presence, and resolved managed `llama-server` path:

```bash
./build/llama_cpp_tool status
```

Use `model-status` to see how ezlocalai will resolve the configured models before the server starts them:

```bash
DEFAULT_MODEL=unsloth/Qwen3.5-4B-GGUF QUANT_TYPE=Q4_K_XL ./build/llama_cpp_tool model-status
```

The output is key/value lines showing each model id, a matching top-level GGUF in its per-model directory, `missing-local` for explicit missing paths, or `hf` when Hub selection is needed. The `hf` value describes the repository and requested quantization; managed engines receive the selected local file with `-m`. Status uses the shared catalog and local-selection policy without creating directories or contacting the Hub. Nested cached files still require Hub selection, as in the runtime.

To pin or update a specific upstream ref:

```bash
EZLOCALAI_LLAMA_CPP_REF=620d2225cc436ab0b001e3200e590e3b286ef9ab ./build/llama_cpp_tool update
./build/llama_cpp_tool build
```

CMake exposes the same workflow:

```bash
cmake --build build --target llama_cpp_update
cmake --build build --target llama_cpp_build
cmake --build build --target llama_cpp_print_bin
cmake --build build --target llama_cpp_version
cmake --build build --target llama_cpp_status
cmake --build build --target llama_cpp_model_status
```

Use `EZLOCALAI_LLAMA_CPP_CMAKE_ARGS` to pass backend build flags, for example:

```bash
EZLOCALAI_LLAMA_CPP_CMAKE_ARGS="-DGGML_CUDA=ON" ./build/llama_cpp_tool build
```

## Current native route status

Use `parity_manifest.tsv` as the source of truth. Routes marked `missing` are absent and rows marked `partial` are not 1:1 complete. The reference has changed beyond the inherited native implementation, including Qwen TTS, full embedding lifecycle, music, and managed router providers. Worker maintenance leases and request accounting now have reference-derived tests; their remaining edge cases are still tracked as partial. The `parity_report --enforce parity_manifest.tsv` intentionally fails until the remaining behavior gaps are closed.


The C++ music API supports `MUSIC_ENABLED=true` with either `ACE_STEP_SERVER_URL` or a local `ACE_STEP_BIN` and model files in `ACE_STEP_MODELS_DIR`. The CUDA image includes ACE-Step at the installed Python worker's pinned revision. Startup downloads missing selected weights from `ACE_STEP_MODEL_REPO` into `ACE_STEP_MODELS_DIR` (default `models/ace-step`), unless `ACE_STEP_PRECACHE=false` or an external server is configured. Existing paths are reused; preparation errors are logged without preventing startup. The local server starts lazily unless `ACE_STEP_AUTO_START=false`. `/v1/audio/music` and `/v1/audio/music/generations` support LM/synthesis jobs and URL or base64 audio results. By default, local ACE-Step waits for active text inference, unloads owned LLM engines, and restores their prior context and GPU layer count afterward. `MUSIC_UNLOAD_LLM_DURING_GENERATION`, `MUSIC_WAIT_FOR_LLM_IDLE_TIMEOUT`, `MUSIC_RELOAD_LLM_AFTER_GENERATION`, and `MEDIA_KEEP_MODELS_LOADED` follow the reference policies. `ACE_STEP_MAX_BATCH` controls the engine's maximum batch size; its default is one. Music-video routes support validation and forwarding; local scene generation and stitching remain unimplemented. See `PARITY_PROGRESS.md` and `tests/fixtures/music_deployment_README.md` for verification and remaining gaps.

Local `/v1/images/generations` and `/v1/images/edits` use the current Python
Qwen-Image-2.1 GGUF implementation. Set `IMAGE_ENABLED=true` and
`IMG_MODEL=qwen-image-2.1`; the CUDA image supplies `SDCPP_BIN`, and weights are
cached under `SDCPP_MODELS_DIR` (default `models/qwen-image`). Edits accept a
primary `image`, up to ten additional `images`, and `strength`. The primary is
both the init image and the first vision reference. URL outputs preserve RGBA
pixels and strip generation metadata; PNG compression can differ from Pillow.
`IMAGE_UNLOAD_LLM_DURING_GENERATION`, `IMAGE_MODEL_MIN_FREE_GB`,
`IMAGE_WAIT_FOR_LLM_IDLE_TIMEOUT`, `IMAGE_RELOAD_LLM_AFTER_GENERATION` and
`MEDIA_KEEP_MODELS_LOADED` control GPU handoff. The wrapper always uses Qwen's
four weights, as current Python does, regardless of stale FLUX comments.
See `tests/fixtures/qwen_image_README.md` and `PARITY_PROGRESS.md` for tested
scope, real CUDA pixel comparisons, and remaining lifecycle, download and transport gaps.

Local `/v1/videos/generations` now supports the reference Wan 2.2 GGUF model names and aliases. Set `VIDEO_ENABLED=true`, `VIDEO_MODEL=QuantStack/Wan2.2-T2V-A14B-GGUF`, and `SDCPP_BIN` to a compatible stable-diffusion.cpp `sd-cli`; `ffmpeg` and `curl` must be on PATH. The C++ backend resolves the pinned high/low experts, encoder and VAE in `HF_HUB_CACHE` (default `models`), downloads missing files, generates frames, and encodes MP4 results as URLs or base64. `HF_HUB_OFFLINE=true` uses existing snapshots only. No Python runtime is used. Image conditioning, resource management, download semantics and packaging still have parity gaps listed in `PARITY_PROGRESS.md`; actual Wan model inference has not yet been verified.

LTX-2.3 uses the same endpoint with `VIDEO_MODEL=unsloth/LTX-2.3-GGUF`.
The CUDA Dockerfile builds the required native engine extensions. Its Q4_K_M
transformer and Diffusers components reuse the Python Hugging Face snapshots;
`LTX2_CONFIG_REPO` selects the component repository (default
`diffusers/LTX-2.3-Diffusers`). First-image and positioned-condition inputs,
URL/base64 output, batching and LLM handoff are connected. `VIDEO_GPU_RESIDENCY`
and the existing LTX residency thresholds select the policy after model loading.
Model offload uses whole-graph execution; sequential offload evicts staged
weights after each graph segment. Gemma projections automatically use native NF4
conversion when the selected CUDA device has at least 12 GiB free. Compute
capability selects BF16/FP16, and actual device geometry controls nested
statistics. The loader preserves embeddings/norms and original cache files;
a conversion failure starts a fresh child with unquantized text weights.
The engine also uses Python's CPU noise, reference scheduler shift and masked
cross-attention timesteps. Real routed storyboard output now retains the
previously lost graphic. NF4 projections now retain packed nested storage and
expand during matrix execution. LTX preloads and retains its engine context
across requests. Full-matrix loader/CPU/CUDA tests and repeated real LTX
generation are recorded in `tests/fixtures/nf4_integration_validation.json`.
Quantized Gemma stays resident on the selected GPU. Connector preprocessing runs
separately and caches the positive/negative conditioning within one sampling
session, releasing its staged weights before transformer execution. The real
24 GB CUDA test completes repeated requests without the previous sequential
retry, with unchanged output bytes. See `tests/fixtures/ltx_connectors_validation.json`
for the measured scope. Quanto fallback, GGUF connector reconstruction, larger
clips/multiple GPUs and complete numerical parity remain incomplete.
See `tests/fixtures/nf4_README.md` for quantizer scope.
See `tests/fixtures/ltx_README.md` for model and runtime evidence.

Wan image conditioning uses C++ RGB decoding and Pillow's center-crop/Lanczos algorithm. The `image_processing_reference` test checks 979 decoded/resized pixel hashes from Pillow 11.3.0 and 181 base64/data-URL cases from the reference API. PNG (including palette/alpha/16-bit), JPEG and WebP have native decoders; other formats retain FFmpeg decoding pending further parity work. Invalid main images and condition frames fail before model handoff. The Wan integration test captures the conditioning PNG passed to `sd-cli` and checks its pixels against the reference.

Video requests use the current API defaults of 18 inference steps and 3.5 guidance. Forwarding tries `IMAGE_SERVER`, then `ROUTER_URL` when this worker lacks video capability, then the configured fallback under the reference availability/resource policy. Router dispatch preserves image and condition inputs; the image-server and fallback paths retain their respective reference field sets. `video_api_reference` replays 398 cases from the actual Python API and clients over native HTTP and checks positive/negative availability caching.

Local video selects the reference's capability-ranked secondary CUDA-visible GPU when present. The C++ driver adapter respects CUDA visibility and requires no toolkit at build time. Video LLM handoff also retries restoration if the initial attempt leaves all prior LLMs unloaded, including when `VIDEO_RELOAD_LLM_AFTER_GENERATION=false`, matching the reference. `video_runtime_reference` covers 2,769 source-derived ranking, placement, policy and restoration cases plus native driver/API integration. Auxiliary model eviction, complete residency/resource admission and real Wan inference remain unverified or incomplete.

Video handoff now unloads idle owned embedding pools and restores them according to `EMBEDDING_KEEP_LOADED`, enablement and delegation settings. Active embeddings are left running; failed restores can retry on the next embedding request. Shared auxiliary hooks and voice/image restoration policies have 3,057 source-derived cases, including API cleanup after generation errors. Persistent voice/image engines, complete resource admission and video retry-time VRAM cleanup remain incomplete.

`CttsRuntime` implements the current CTTS constructor settings, device/memory policy, generation and warmup kwargs, single-sample inference, CPU fallback and cleanup/close ordering. `ctts_runtime_reference` compares 787 source scenarios/1,268 operations, including two gated cross-thread close cases. The optional `ezlocalai_ctts_torch` adapter supplies native CUDA memory and cleanup hooks plus real LibTorch inference mode. `ctts_model_reference` compares three complete CPU audio hashes and an intervening invalid-text error/recovery through the actual worker. This is the constructor prefix through sample-rate initialization, not full CTTS filesystem/cache/warmup or API integration. See `tests/fixtures/ctts_runtime_README.md` for provenance and limits.

`ctts_audio` now implements CTTS mono float32 mixing, separate WAV/streamed-PCM conversion, in-memory audio decoding and tensor layout conversion. The optional LibTorch adapter provides torchaudio-compatible CPU resampling. `ctts_audio_reference` compares 2,122 source cases across strided arrays, quantization/non-finite boundaries, nine container encodings and rate conversion. For exact codec output, set `EZLOCALAI_SNDFILE_LIBRARY` to the matching native library from the source soundfile build; equal libsndfile version strings alone did not guarantee matching Vorbis samples. This library selection also applies to LlamaTts WAV reads. Without it, the system library is used. No Python runtime is invoked.

`ctts_paths` now provides current reference-voice lookup, default fallback, non-strict POSIX path containment and transcript reading, with 610 real-filesystem source comparisons including symlinks, loops and UTF-8 errors. Both components are composed into Ctts buffered and streamed generation; public API integration remains pending. Provenance, dependency hashes and numerical/interface limits are in `tests/fixtures/ctts_conversion_README.md`. The real-model CTTS comparison now also verifies complete WAV and PCM hashes for all three successful buffers.

`CttsAudioCache` now supplies source-shaped cache keys, persistent audio/metadata, expiry, size eviction, statistics, voice/full clearing, shortest/median sample selection and explicit warming. `ctts_cache_reference` compares 205 scenarios/703 operations with real files, complete metadata bytes, clock/generator calls and failure state. Native WAV duration parsing and ffprobe/ffmpeg fallback match the captured PyDub inputs; no Python runtime is used. It is composed into Ctts buffered generation; public API integration remains pending. Exact scope and remaining type/concurrency/dependency gaps are in `tests/fixtures/ctts_cache_README.md`.

`Ctts` composes full native construction, voice listing, nonfatal warmup and buffered generation with the cache, text/path helpers, model lifecycle and WAV conversion. `ctts_generate_reference` compares 140 scenarios/539 operations, including exact filesystem bytes and caller/worker call order. `ctts_generate_model_reference` checks complete actual CPU generation, cache hits and multi-chunk output against source result/file hashes and metadata. The synchronous interface leaves event-loop/transport cancellation and public endpoint integration open. See `tests/fixtures/ctts_generate_README.md`.

`CttsStreamBridge` adds a lazy capacity-two producer queue with explicit cancellation and source cleanup/error precedence. `TtsStreamIterator` owns native worker iteration and preserves the existing callback API. The bridge has 58 source comparisons plus 11 native process cancellation/lifetime cases; the actual Qwen stream comparison now passes through this bridge. CTTS framing is composed by Ctts::generate_stream; async transport and public endpoint integration remain open. See `tests/fixtures/audio_stream_bridge_README.md`.

`Ctts::generate_stream` composes native byte framing, PCM writes, inter-chunk silence, drain waits and yield-specific cancellation with the bounded bridge. It has 143 source comparisons, six actual CPU operations with 103 exact write hashes, and nine process cancellation/lifetime cases. Stream next/close use one consumer; cancellation may arrive from another thread. HTTP scheduling, disconnect handling and public endpoint integration remain unfinished. See `tests/fixtures/ctts_stream_README.md`.

`TtsPool` composes native CTTS factories with resident/transient leases, live model naming, CPU fallback admission, 4 GiB CUDA resource accounting, consumer lease transfer and cleanup. It matches 748 source scenarios/22,353 operations. Six native worker cases compose it with `VoiceSlotGuard` for buffered/streamed synthesis, cancellation, queued admission and cross-thread release. Public speech orchestration and transport wiring remain incomplete. See `tests/fixtures/tts_pool_README.md`.

`TtsService` now composes the source local buffered/streaming request lifetime with the native pool and voice guard. It handles base64 decoding, gTTS eligibility/invocation through a required transport callback, pre-header loading, consumer lease transfer and source exception/finally precedence. Tests cover 1,036 source scenarios/5,628 operations, twelve native worker cases, and 77 real-Qwen operations with 70 exact output hashes. Local buffered/streamed HTTP is wired to this service; WebSocket and remaining routing/lifecycle integration are still open. See `tests/fixtures/tts_service_README.md`.

`tts_gtts_operations` supplies the native gTTS fallback callback. The C++ implementation matches gTTS 2.5.3 language selection, tokenization, request packaging, audio extraction and error handling across 3,230 source cases. Native curl transport matches 14 loopback scenarios/55 HTTP requests, including redirects, cookies and per-part session isolation; native FFmpeg conversion matches exact WAV outputs and process failures. Both executables must be on PATH; no Python runtime or shell is used. The source dependencies permit other gTTS versions, and exhaustive external Google/proxy/TLS/cookie behavior is not verified. See `tests/fixtures/gtts_README.md` for provenance and remaining limits.

The native executable now serves local `/v1/audio/speech` and `/v1/audio/speech/stream` through the native TTS pool, CTTS worker and gTTS fallback. Streaming loads before sending HTTP 200, transfers the lease to the consumer, and cancels blocked synthesis on disconnect. Source HTTP tests cover 912 scenarios in buffered/deferred modes; four actual worker cases cover disconnect cleanup. The real executable matches two source WAV hashes and 65 stream packet hashes over TCP. Public voice/router/general provider selection now matches 2,124 source scenarios; 94 real source-client HTTP scenarios match 188 native buffered/deferred comparisons, including disabled-router forwarding and partial stream errors. Data-URI pipelines, local voices, model-management and complete cross-modality residency remain partial. See `tests/fixtures/tts_api_README.md` and `tests/fixtures/tts_routing_README.md`.

For an existing native worker and local Qwen model files, set `QWEN_TTS_BIN` to
the `ezlocalai-tts` executable, `QWEN_TTS_MODEL_FILE` to the talker GGUF, and
`QWEN_TTS_MMPROJ_FILE` to the tokenizer GGUF before starting `ezlocalai-native`.
Reference voices are read from the working directory's `voices` folder. These
are the same worker/model settings used by the native wrapper; automatic
deployment discovery and packaged voice assets remain incomplete.

`/v1/audio/speech/ws` now uses the native local TTS pool, including when a voice server is configured. It authenticates before upgrade, buffers text without reserving a model, bounds the synthesis queue to four segments and streams validated 24 kHz mono PCM. Source-backed tests cover 3,679 PCM/input cases, 352 lease scenarios, 127 API scenarios over 254 TCP comparisons, seven concurrent sessions and 43 Uvicorn wire captures. Worker tests cover HTTP/WebSocket disconnect and shutdown in four ownership modes; the actual executable also matches 67 real Qwen WebSocket frame hashes. See `tests/fixtures/tts_websocket_README.md` for provenance and remaining transport/concurrency limits. Speech parity remains partial.

Audio data URIs sent to `/v1/audio/speech` now use local native STT followed by local CTTS and return the source URL/base64 JSON string. Their parsing, sequential guard/resource ownership, errors and absence of gTTS fallback have source-backed comparisons. The source decodes the URI payload and then base64-decodes those bytes again inside STT; native code preserves that behavior. PDF data-URI extraction/routing and complete voice-management parity remain unfinished. See `tests/fixtures/tts_pipeline_README.md`.

Voice listing now returns sorted local WAV stems and shares the retained remote client and availability cache with public speech. Uploads preserve the supplied bytes, sanitize names, resolve destination-directory symlinks and choose unique filenames using source filesystem behavior. Reference captures cover 247 filesystem scenarios and 97 real-client HTTP scenarios, including JSON serialization failures, cache/configuration transitions and idempotent disconnect retries. Multipart edge cases and complete global voice-client lifetime remain partial; see `tests/fixtures/voice_management_README.md`.
