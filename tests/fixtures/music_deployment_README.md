# ACE-Step source and model comparisons

`music_precache_cases.json` records 45 executions of the unchanged Python
`precache_music_model` function. It captures filesystem/download calls for
enable switches, external servers, empty and duplicate file settings, cached
paths, directory settings, and failures. The fixture pins `precache.py`,
`MUSIC.py`, and `Globals.py` by SHA-256. `music_runtime_reference` replays these
calls through the native implementation, alongside its existing 681 policy
cases and real process/concurrency checks.

The real-model captures execute the installed Python worker's `MUSIC` class
with its actual ACE-Step server and cached models. No model or HTTP boundary is
substituted. The engine revision is
`5b88ded7919285d5ad5b1e86ffe3378314f305db`. Each capture records the wrapper
and capture-script hashes, four model-file hashes, request arguments, actual
LM and synthesis payloads, response metadata, and audio SHA-256 values.

Five requests cover seeded LM plus synthesis, the server's default maximum
batch size, direct synthesis from previously generated audio codes, WAV16,
WAV24, WAV32, persisted MP3, and an explicit maximum batch size of two. They
produce six stereo audio files. The default server returns one track even
when `n=2`; with `ACE_STEP_MAX_BATCH=2`, the last request returns two. The test
preserves these observed source behaviors. The requested five-second duration
does not imply that every generated file is exactly five seconds long.

`test_music_deployment` verifies all model hashes, starts a real native engine,
runs the worker HTTP handler, compares every audio byte and all response
metadata except timestamps and generated filenames, and cleans up its child
and temporary output directory. CUDA expectations are specific to the RTX
3090 Ti. CPU expectations were captured on an Intel Core i9-12900KS with 12 engine
threads. CPU and CUDA use separate captures rather than a cross-device numeric
tolerance.

To opt in, configure `EZLOCALAI_ACE_STEP_MODELS` and
`EZLOCALAI_ACE_STEP_BIN`. The CPU test disables CUDA visibility. Set
`EZLOCALAI_ACE_STEP_GPU_TESTS=ON` to add the CUDA test. Run
`music_deployment_cpu_reference` and `music_deployment_cuda_reference` through
CTest on the corresponding hardware. The CUDA-built engine needs the NVIDIA
driver library even when running the CPU case.

Keep ACE-Step's GGML libraries separate from llama.cpp's. They share library
names but have different ABIs, including the maximum tensor-name length.
The CUDA image uses engine-local runtime paths and gives the installed TTS
executable its own llama.cpp runtime path. A global llama.cpp library path can
make ACE-Step reject the DiT model before synthesis begins.

These checks do not establish full music-video, model placement, allocator
residency, MIME parsing, or transport parity. Those remain in the progress
document and partial manifest rows.
