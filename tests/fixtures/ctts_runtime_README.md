# CTTS configuration and model lifecycle

`ctts_runtime_cases.jsonl` records 787 scenarios and 1,268 operations from
unchanged methods in `ezlocalai/CTTS.py`, pinned to source commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Its header checks SHA-256 hashes of
CTTS.py and LlamaTTS.py. The external capture program is
`/tmp/ezlocalai_ctts_runtime_oracle.py`; only captured data and the C++ runner
are shipped here.

The capture executes the original constructor prefix through
`self.sample_rate = 24000`, plus unchanged device selection, model loading,
generation settings, warmup kwargs, memory release, close, CPU reload,
single-sample generation and inference-context methods. It deliberately stops
before directory creation, reference audio, cache initialization and warmup
execution. Environment reads, Torch/GC operations and model construction and
generation are controlled boundaries. Comparisons include exact call order,
arguments, results/error categories/messages, device, closed state, backend
identity and closed flags for every constructed backend.

Coverage includes Unicode and unbounded integer configuration, chunk/drain
limits, model migration, generation kwargs and live warmup limits; automatic
and explicit device selection, memory-query fallback and thresholds; initial
load and CPU retry failures; duplicate/wrong generation kwargs, missing audio,
empty results and sample-rate conversion; inference-context entry/exit and
cleanup exception precedence; repeated close, failed close and failed reload.
Two gated, real-thread cases close the owner while generation is blocked, then
return audio or a CUDA RuntimeError. Close must finish while generation remains
blocked, and a closed owner must not reload on the later CUDA error. These are
specific interleavings, not exhaustive concurrency proof.

`CttsRuntime` composes these policies with the native `LlamaTts` process wrapper.
Without supplied hardware hooks it uses CPU-only defaults.
`ctts_torch_operations` supplies native LibTorch CUDA availability, memory
queries, cache/synchronization/IPC cleanup and a per-thread `c10::InferenceMode`
guard. Its CUDA memory query restores the previously selected device. The
optional adapter shares the configured STT LibTorch dependencies and links
CUDA runtime explicitly; it requires no Python runtime.

## Actual model comparison

`ctts_model_cases.jsonl` contains four sequential requests through the same
source constructor prefix and methods, with actual LlamaTTS, Torch 2.9.1+cu128,
NumPy 1.26.4 and soundfile 0.13.1. The external capture program is
`/tmp/ezlocalai_ctts_model_oracle.py`. The fixture pins both GGUF hashes, the
worker and source hashes and the reference speaker WAV. Model revision,
engine dependencies and build options are documented in `llama_tts_README.md`.
The model is `Mouserat/qwen3-tts-0.6b-base-gguf` revision
`30954b7a9d99fad81064bf1dc35f881b029f9032`, using the Q8_0 talker and f16 tokenizer.

With CPU execution, seed 23, four threads and context 4096, the requests are:

| Request | Generation limit | Result at 24 kHz |
| --- | --- | --- |
| Hello. | Warmup limit 4 | 7,680 samples |
| Good morning. | Configured limit 8 | 15,360 samples |
| Empty text | Configured limit 8 | Exact worker RuntimeError |
| Hello again. | Configured limit 8 | 15,360 samples after recovery |

The C++ runner compares complete little-endian float32 audio, WAV and PCM SHA-256 values,
rates, channels and the error message. With LibTorch configured, it additionally
checks actual CUDA memory-query bounds, inference mode during backend calls and
restoration after each success/error. This verifies CPU synthesis through CTTS;
it does not verify GPU TTS synthesis. Hardware checks ran on an RTX 3090 Ti with
the CUDA-enabled LibTorch build. The CPU-only LibTorch build is unverified.

Configure the worker/model options documented in `llama_tts_README.md` to enable
`ctts_model_reference`. `ctts_runtime_reference` needs neither models nor Torch.
Both tests and all runtime implementation in this repository are C++.

## Remaining scope

Reference-audio resolution/filesystem containment and audio conversion now have
standalone C++ implementations and differential fixtures documented in
`ctts_conversion_README.md`. Complete construction, cache/warmup and buffered
generation composition is covered by `ctts_generate_README.md`, and native byte
streaming by `ctts_stream_README.md`. Async transport, TTS pool/voice guard and
public endpoints remain unfinished. Logging, arbitrary Python
objects and malformed/non-finite JSON boundaries, all platform diagnostics,
production dependency packaging and exhaustive cancellation behavior are not
established by these fixtures. Speech manifest rows remain partial.

The native AudioCache component is covered by `ctts_cache_README.md` and is now composed into buffered Ctts generation.
