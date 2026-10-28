# Native LlamaTTS wrapper verification

The source is `ezlocalai/LlamaTTS.py` at ezlocalai revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Headers pin its SHA-256.
`tts/worker.cpp` is copied byte-for-byte from that revision; model tests
also verify both worker source hashes. The native wrapper, worker and test
programs are C++. Python is used only for external reference captures, retained
at `/tmp/ezlocalai_llama_tts_{policy,process,model}_oracle.py`.

`llama_tts_policy_cases.jsonl` contains 305 constructor and 266 language cases.
The capture imports the unchanged source module and substitutes filesystem,
Hub download, executable discovery and process-launch boundaries. It compares
arguments, child environment, timeout, exception category/message and boundary
order. Cases include model migration and revision defaults, local model files,
worker fallback, unsupported settings, Python integer conversions, CUDA index
selection (including negative indices), CUDA/HIP visibility, graph opt-out and
script/language precedence. Real Hub transfer behavior is independently covered
by the existing Hub tests.

`llama_tts_process_cases.jsonl` contains 115 scenarios and 205 operations. It runs
the complete unchanged Python class against the compiled C++ process fixture,
then compares the C++ wrapper against the resulting traces and audio samples.
The fixture receives real stdio JSON requests, writes real float32/PCM16 WAVs,
and emits PCM chunks or controlled failure replies. Checks cover persistent
reuse, mono/stereo WAVs, startup noise/strict readiness, worker errors/EOF,
per-response timeouts, invalid replies, missing WAV files, integer conversions,
strict base64 error precedence, request-construction errors, stream abandonment,
consumer exceptions, idempotent close and recovery attempts. Error categories,
messages, samples/rate/channel shape, requests and child arguments/environment
are compared. Temporary output/speaker paths are normalized; missing-WAV error
paths are normalized in exception strings. Rates are retained as decimal
integers in the native wrapper, including values beyond machine integer range.

Cancellation checks close a worker while buffered/streamed generation is waiting
under its inference lock. Both source and C++ must return in under one second,
without waiting for the configured synthesis timeout. Four gate-controlled
thread pairs verify buffered/buffered, buffered/stream, stream/buffered and
stream/stream serialization on one process. Every scenario checks that its
worker has been reaped and the parent's CUDA/HIP/graph settings are untouched.
The process fixture also checks the child receives the default SIGPIPE
behavior, as with Popen's restore_signals, while the parent may ignore it.
These are specific controlled interleavings, not an exhaustive scheduling proof.

`llama_tts_model_cases.jsonl` captures the complete source wrapper using the
unchanged compiled worker and actual pinned model files:

- Repository: `Mouserat/qwen3-tts-0.6b-base-gguf`
- Revision: `30954b7a9d99fad81064bf1dc35f881b029f9032`
- Files: `qwen-talker-0.6b-base-Q8_0.gguf`, `qwen-tokenizer-12hz-f16.gguf`
- Engine: llama.cpp `aa94f2086147cd4bfbcff818ce998dffce5fcb39`, CPU build
- Settings: seed 23, four threads, context 4096; other sampling settings are
  source defaults
- Reference audio: `stt_runtime_assets/speech.wav`, pinned by SHA-256
- WAV readback: soundfile 0.13.1 / libsndfile 1.2.2; native readback uses the
  installed libsndfile 1.2.2 C ABI, without Python

The five sequential requests share one worker and produce six audio buffers.
Both buffered and streamed short requests are tested. The final stream uses a
76-frame limit and emits a 72-frame window (138,240 samples) and four-frame tail
(7,680 samples), at 24 kHz. Each entire float32 little-endian buffer is compared
by SHA-256, including exact chunk boundaries. This establishes the captured
CPU behavior; it is not a GPU or arbitrary-model equivalence claim. The source
and native wrapper use the same unchanged worker and pinned engine, so these
checks isolate wrapper behavior rather than independently reimplementing the
upstream synthesis algorithm.

`llama_tts_model_v080_cases.jsonl` separately records those same five sequential
requests through the unchanged, hash-pinned Python wrapper with llama.cpp
v0.8.0 (`8642541b255bfb5dc39cd351a178be09013a55d6`). The historical fixture
above is unchanged. The new header distinguishes the reference worker source
from the adapted native worker. The default optional real CPU model test uses
this versioned fixture; `EZLOCALAI_TTS_MODEL_CASES` selects another record for
an explicitly matched engine.

Upstream v0.8.0 moves the codec generator RNG from the shared model context to
per-utterance state. The first buffered result matches the old engine exactly;
subsequent stochastic waveforms change. This is a seed-lifecycle correction,
not a bitwise audio-preservation claim. All six buffers retain their exact
lengths, rate and chunk boundaries, and the native wrapper matches the Python
wrapper's new hashes exactly. An independent direct-worker run produces the
same hashes. The existing composition/CUDA model fixtures remain historical
evidence for their recorded engines; they have not been regenerated for v0.8.0.
Model-free process, transport and package gates retain their own checks.

Configure `EZLOCALAI_TTS_LLAMA_SOURCE` and `EZLOCALAI_TTS_LLAMA_BUILD` to a pinned
source checkout and existing native shared-library build to enable the optional
`ezlocalai-tts` target. They default to the corresponding existing engine-test
paths. No dependency is downloaded implicitly. Set
`EZLOCALAI_TTS_MODEL_DIRECTORY` to the local GGUF directory to enable the real
model test. First-party CMake remains CXX-only. Policy/process tests need no
model weights; process WAV checks require `libsndfile.so.1`.

This layer now backs the CTTS configuration, model lifecycle, CPU fallback and
single-sample policies described in `ctts_runtime_README.md`, plus full buffered
CTTS composition in `ctts_generate_README.md`. The native bounded bridge and
lazy iterator are documented in `audio_stream_bridge_README.md`; the real-model
streaming test now uses that bridge with the same source fixture hashes.
CTTS framing/cancellation composition is covered by `ctts_stream_README.md`.
Async transport, TTS pool/voice guard and public speech
integration remain unfinished. Their manifest rows remain partial.
GPU execution, production engine/library packaging,
non-Linux process behavior, logging parity, malformed UTF-8/non-finite JSON
worker output, all Python arbitrary-value edges and exhaustive cancellation
interleavings remain unverified or incomplete. The worker itself emits valid
JSON with finite numeric rates. No end-to-end TTS completion is claimed.
