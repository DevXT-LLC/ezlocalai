# Complete CTTS construction and buffered generation

`ctts_generate_cases.jsonl.gz` contains 140 scenarios and 539 operations from
reference commit `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.
The external capture program `/tmp/ezlocalai_ctts_generate_oracle.py` executes
the complete unchanged CTTS module body and AudioCache class, supplying their
imports explicitly. The fixture header pins CTTS.py, AudioCache.py and
LlamaTTS.py by SHA-256. The model, hardware, inference context, environment,
clock and UUID boundary are controlled; filesystem operations, NumPy and
soundfile encoding are real. Only fixture data and C++ tests are included here.

Each comparison checks the exact result or exception, all environment/model/
hardware/inference/clock/UUID calls and whether each call runs on the caller or
a generation worker. It also checks voices, sample rate, model lifecycle state,
cache configuration/statistics and the complete filesystem tree, including all
file bytes. Only temporary-root prefixes in JSON strings are normalized;
metadata and audio bytes are not modified.

Coverage includes constructor directory ordering and conflicts, voice-list
snapshots, nonfatal warmup and fallback, empty text, chunking, language/voice
inputs, cache configuration and per-request overrides, live model-file/revision
environment keys, custom cache directories, cache hits without a reference WAV,
empty/corrupt/expired cached entries, voice-specific/full clearing, output
sanitization and write failures, changed/invalid sample rates, empty chunks,
multichannel mono conversion, mid-request failure, CPU fallback and requests
after close. Seeded cache entries use keys captured from the source.

The implementation retains the hard-coded `outputs/cache/audio` precheck even
with a custom cache directory. Cache hits return a cache URL, while fresh URI
results write a separate output file. Empty cached bytes fall through to
synthesis. Cache storage precedes output writing and may survive an output
failure. A UUID is evaluated for every fresh URI result, even with an explicit
output name. The last chunk's rate wins even if that chunk is empty; entirely
empty audio leaves the instance rate unchanged.

## Actual model comparison

`ctts_generate_model_cases.jsonl` captures complete source construction with
real default-voice warmup followed by seven operations: a fresh one-chunk URI
result, its cached URI, a fresh two-chunk URI result, reference-voice removal,
a cached base64 result, a missing-reference error on a new request and close.
The external program is `/tmp/ezlocalai_ctts_generate_model_oracle.py`.
The native runner compares result hashes/lengths, exact errors, every file's
SHA-256/size and cache/controller state after each operation. Metadata hashes
include exact source serialization. Warmup executes on the caller; all three
request synthesis chunks execute on a worker under real LibTorch inference mode.

This uses the same pinned Qwen 0.6B GGUF model, native worker and reference WAV
as `ctts_runtime_README.md`, with seed 23, four worker threads, context 4096,
eight generation tokens, four warmup tokens and a 12-character chunk limit.
The fixture pins both model files, source worker, speaker and audio library.
Its environment records the model-file paths because those strings contribute
to source cache keys. The runner retains those captured cache-identity strings while allowing the actual worker/model paths to be relocated.
Torch 2.9.1+cu128, NumPy 1.26.4 and soundfile 0.13.1 with the pinned native
libsndfile build are used. Synthesis is CPU-only. The worker's frame-limit
warnings are expected with the deliberately small token limits.

## Native interface and remaining scope

`Ctts` composes CttsRuntime, reference paths, CttsAudioCache and audio conversion.
The C++ request API is synchronous; the caller owns request scheduling and each
synthesis chunk runs on a native worker. No Python runtime is used. This does
not yet implement the full Python event-loop/cancellation contract,
TTS pool/voice guard or public
speech endpoint integration. Arbitrary Python objects, every malformed JSON
boundary, mutable attributes, concurrent requests/cache mutation, filesystem
races, GPU synthesis, logging and complete dependency packaging remain
incomplete or unverified. Passing these fixtures does not establish full 1:1
speech or application parity.

The standalone bounded bridge and lazy native iterator are now covered by
`audio_stream_bridge_README.md`; their CTTS framing/cancellation composition is
covered by `ctts_stream_README.md`.
