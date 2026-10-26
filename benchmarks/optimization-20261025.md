# CUDA optimization measurements — 2026-10-25

Measured Qwen3.8-27B Q3_K_XL with q4 KV, 230K allocated context and the existing MTP settings. The current engine is llama.cpp `620d2225` (v0.7.0). Each cold depth started in a fresh isolated engine; three baseline cold samples per depth used nonces after the first exact historical prompt. Warmup was excluded. Both production workers were restored with their original images, environment and mounts, and their owned router drains were released.

| Worker | ~68K cold prefill | ~150K cold prefill |
| --- | ---: | ---: |
| DevXT4090 | 2529 tok/s | 1970 tok/s |
| DevXT3090V | 1246 tok/s | 939 tok/s |

The historical 4090 run on October 18 measured about 2350/1812 tok/s with the same frozen prompt hashes and model settings. Current medians are roughly 8–9% higher, but the engine changed from `aa94f20`/b11549 to v0.7.0, so this does **not** isolate the GPU unlock. No comparable saved pre-unlock 3090V result was found. 3090V is an RTX 3090 **Ti**. Both cards sustained P2 with 10251 MHz memory and a 450 W power ceiling; the test changed no host power or clock settings. Hardware thermal slowdown was not observed in the recorded samples.

The 4090 logical-batch 8192 trial was effectively unchanged; logical-batch 1024 was slower. On 3090V, physical batch 1024 measured about 1014 tok/s at ~150K, versus a 939 tok/s baseline median. This single trial is promising (~8%), but needs repetition plus max-context/vision memory validation before changing a default. Logical batch 1024 brought no benefit. Defaults were retained.

The separate same-host conversation replay used the full CUDA model on DevXT3090J, including MTP. It switched among three ~24K-token conversations. Stock returning turns re-evaluated ~23,800 tokens and took a median 19.04 s. The patched engine evaluated 4 tokens and took 0.916 s (~20.8× faster), with the same branch answers on every turn and the same 14336 MiB cache ceiling. This addresses cache retention; it does not increase cold GPU prefill throughput.

CPU request preparation for the 707 KB request improved from a median 9.65 ms to 3.53 ms (2.73×). All three measured long-prompt engine payloads were byte-identical. This saves milliseconds around inference; it is separate from token throughput.

Compact samples, configuration and restoration checks are recorded in `tests/fixtures/unlocked_gpus_20261025_validation.json`, `conversation_cache_20261025_validation.json` and `request_preparation_20261025_validation.json`. Bulk logs remain under ignored `build/validation/optimization-20261025/`. No optimization image has been deployed. RAM/cgroup/swap policy is covered by deterministic tests; actual swap-in latency and sustained larger-cache pressure still need physical validation.
