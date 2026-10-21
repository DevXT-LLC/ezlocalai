# Installed CUDA audio dependency captures

These captures execute the unchanged Python application's `STT` and `LlamaTTS`
classes inside the installed `ezlocalai-ezlocalai` image on 2026-10-18. They use
real cached models, PortAudio initialization and inference; no model boundary
is substituted. Capture scripts ran outside the repository. The repository
runners are C++.

`stt_deployment_cpu.json` and `stt_deployment_cuda.json` each contain four calls:
plain English transcription, word timestamps, batched beam-3 decoding, and
automatic language detection. The model is
`mobiuslabsgmbh/faster-whisper-large-v3-turbo`, revision
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`. The runner checks hashes of the source
wrapper, all five model files and the speech WAV before inference. It compares
text and JSON structure exactly, with a 1e-12 allowance for floating-point
timestamp arithmetic. Independent Python CPU runs selected adjacent 20 ms frames
for three word boundaries. The CPU fixture records both observed results, and
the test accepts only those recorded word times, without adding a general timing
tolerance. GPU results retain exact comparisons. It also checks the actual device, compute type, temporary
file cleanup and released native handles.

The installed versions are CTranslate2 4.8.2, faster-whisper 1.2.0, ONNX Runtime
1.31.0, PyTorch 2.9.1+cu128, NumPy 2.4.6 and PyAV 18.1.0. Native dependency
archives are pinned separately in `../../cmake/cuda-audio-dependencies.json`.
The CPU is an Intel Core i9-12900KS (eight Whisper threads); the GPU is an
RTX 3090 Ti (float16 inference). `OPENBLAS_NUM_THREADS=1` was set for both.
CPU int8 and GPU float16 have separate expectations because their word timings
differ. These cases do not establish equivalence on every device or codec.

`llama_tts_deployment_cuda.jsonl` repeats the five existing Qwen TTS requests on
that GPU, using seed 23 and four CPU threads. It checks exact SHA-256 values of
six float32 audio buffers, including a two-window stream. The header retains
the installed worker binary hash as provenance; native rebuilds need not have
the same executable hash. The same five requests were freshly captured on CPU
and all six buffers still match `llama_tts_model_cases.jsonl` exactly.

External capture script hashes:

- STT: `97c5bd606ad534e379eb1b754514c1a963bdc64ed6e2558e41aab2519635d363`
- TTS: `96ab62063ddf930b3cb4d81f12116407b3e76ced7e2c5a9f7ce8527f333ceabd`

Configure the optional native dependencies, `EZLOCALAI_REFERENCE_ROOT`,
`EZLOCALAI_WHISPER_CT2_TURBO_MODEL`, `EZLOCALAI_WHISPER_VAD_MODERN_MODELS`
and `EZLOCALAI_TTS_MODEL_DIRECTORY` to enable the CPU tests. Set
`EZLOCALAI_STT_TORCH_GPU_TESTS=ON` to add the hardware-specific GPU checks.
Run `stt_deployment_cpu_reference`, `stt_deployment_cuda_reference`,
`llama_tts_model_reference` and `llama_tts_deployment_cuda_reference` with CTest.
The CUDA image also requires the NVIDIA container runtime for the TTS binary's
driver library, including when selecting CPU inference.
