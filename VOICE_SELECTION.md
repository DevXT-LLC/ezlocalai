# Selectable voices for LFM3 calls

LFM3 Omni does not accept a reference speaker recording in its native speech
decoder. The pinned llama.cpp implementation explicitly ignores a speaker
reference for this model. The vendor's TTS interface documents male/female
speech prompting and language/locale selection; conversational realtime voice
control has not been independently verified. This is not arbitrary voice cloning. A
`session.audio.output.voice` field is not implemented by the pinned realtime
server and must not be presented as a working clone selector.

There are three different paths:

| Output path | Selection | Delivery and latency | Status |
| --- | --- | --- | --- |
| LFM native speech | Male/female/locale controls documented for TTS; realtime selection unverified | Original LFM delivery; current native streaming | Existing default; arbitrary speaker cloning unavailable |
| LFM text → Qwen TTS | Existing named reference WAVs or uploaded reference audio | Qwen generates new delivery; additional synthesis/buffering | Existing components demonstrated below; live-call adapter still required |
| LFM audio → voice converter | A converter's supported target voice/reference | Aims to retain source rhythm/prosody; adds conversion and possible artifacts | Research option; no converter shipped or validated here |

Changing voice cannot be promised to preserve every quality dimension.
Evaluate intelligibility, naturalness, speaker similarity, prosody, interruption
behavior and latency separately. No production output path changes automatically.

## Recommended direction for calls

Prioritize **streaming waveform voice conversion** for custom call voices.
The Qwen experiment establishes a working alternative using existing components,
but its measured extra turn delay makes it a weaker fit for natural conversation.
The preferred pipeline is `LFM native PCM → stateful voice converter → playback`.
This keeps LFM's speech generation and aims to preserve its timing and expression;
quality preservation still needs listening and content/prosody checks.

A native candidate is now available in
[audio.cpp's MeanVC2 implementation](https://github.com/0xShug0/audio.cpp/blob/cd770076152085af5b166521185bec76f9d47808/docs/models/meanvc2.md).
The source exposes a persistent streaming session, source-audio chunks and a
target-reference voice. Its default F32 package uses the **120 ms / 40 ms** path,
with **160 ms preferred input chunks at 16 kHz**. The upstream MeanVC2 paper's
**110 ms** result is a different reported configuration; do not use it as a
measurement of this native package or our complete LFM pipeline. Start evaluation
with F32 and precompute/cache target conditioning. The native runtime has not
been built or benchmarked here.

Other useful comparisons are
[RT-VC](https://arxiv.org/html/2506.10289v1), which reports **61.4 ms on an Apple M3
CPU** (32 ms lookahead + 15 ms input + 14.4 ms processing), and
[Seed-VC](https://github.com/Plachtaa/seed-vc#usage), whose realtime setup documents
roughly **300 ms algorithmic + 100 ms device delay**, with a **430 ms** example
on an RTX3060 Laptop GPU. These are upstream results under different conditions,
not fleet benchmarks. RVC-style changers are another option for a library of
trained voice models; that is a different workflow from selecting an arbitrary
reference recording without training.

RT-VC and the inspected native MeanVC2 output are **16 kHz**; LFM currently emits
24 kHz. Resampling back to 24 kHz does not restore the removed bandwidth. Their
lower latency is insufficient to establish unchanged quality. Compare source
and converted speech for intelligibility, speaker similarity, emotion, pitch
contours, pauses and audible seams, including cross-gender and multilingual cases.

Measure added first-audio delay, per-chunk p50/p95 processing, required lookahead,
underruns, memory and barge-in/recovery with the LFM engine running. Faster-than-
realtime throughput alone does not establish low latency. Source PCM generated
in bursts also means microphone latency reports cannot simply be added to LFM's
first-packet time. Use bounded queues, persistent per-call converter state and
immediate cancellation/discard on interruption. Native deployment should isolate
the converter's ggml libraries from the pinned llama.cpp runtime until ABI/linkage
compatibility is established. The inspected MeanVC2 session retains merged audio
until reset; use bounded per-utterance storage and verify stateful resampling
across input chunks. No converter runtime, weights, production settings
or public voice-selector protocol was added by this research update.

## Existing Qwen path

Our native Qwen worker already takes a reference WAV and streams new speech.
It uses Qwen3-TTS 0.6B Base Q8_0 and an F16 tokenizer/codec. Its upstream input
API provides a speaker embedding but no reference-transcript/ICL prompt. The
full Qwen model family supports richer reference-text cloning; that capability
is not implemented by this C++ path. Do not equate support for a reference
recording with every upstream voice-clone quality option.

The worker preserves the upstream **72-frame vocoder window**. At 12.5 codec
frames/second that covers **5.76 seconds of audio**, or the final shorter
utterance. This is an audio buffer length, not a wall-clock delay: GPU synthesis
may run faster than real time. Emitting partial windows earlier changes the
decoder's attention context and samples. We have not shortened that window or
claimed the upstream advertised first-packet latency for this implementation.

For a WorkConductor call adapter:

1. Keep LFM's incoming PCM, conversation context and tools. For a custom Qwen
   voice, request LFM **text output** and send stable spoken text to the existing
   streaming Qwen speech endpoint with the selected reference voice. There is
   no extra STT step between LFM's output and Qwen.
2. Speak only `response.output_text.delta` content. Do not synthesize tool
   argument JSON, control events or hidden reasoning. Prefer complete sentences;
   generating isolated words can damage continuity and prosody.
3. Correlate the Qwen stream with the LFM response. Interruption must cancel
   both generation paths, discard queued playback and prevent stale speech
   after the next turn. Original LFM timings/emotion/nonverbal sounds are not
   retained by text re-synthesis.
4. Read the TTS endpoint's actual format/rate metadata. LFM's PCM16 contract
   must not be assumed for every existing TTS transport.
5. An active LFM call owns its text slot. Default single-GPU speech handoff can
   try to unload that LLM, so use another available TTS worker or validate an
   explicitly resident layout before co-hosting. `TTS_UNLOAD_LLM_DURING_GENERATION=false`
   disables that handoff, and `MEDIA_KEEP_MODELS_LOADED=true` retains models;
   worker-level coexistence and cancellation still need validation. The raw
   engines fit together in the GPU demonstration below. These
   settings were **not** applied to production here.

   The native-only example in `REALTIME.md` sets `TTS_ENABLED=false`. Enable
   ordinary Qwen TTS on the selected speech worker when using this output path.

The existing `/v1/audio/voices` and speech routes provide voice selection for
Qwen. A unified call voice picker belongs to the client adapter in
[WorkConductor #429](https://github.com/Josh-XT/WorkConductor/issues/429).
No WorkConductor code was changed.

## Reproducible demonstration

`benchmarks/bench_reference_voices.py` runs only an owned, network-isolated,
CPU-only container using explicit immutable image/runtime/model inputs. It
never stops or replaces a production worker. It renders identical text with
`warm_american_narrator.wav`, `british_rp_female.wav`, then the first voice again.
The first and repeated voice produced identical float PCM; the alternate
reference produced different PCM. These checks establish controlled selection
and state reset, not perceptual voice similarity or quality equivalence.

Audio files, first-packet timing, frame/chunk counts, hashes and logs are under
ignored `build/validation/selectable-voice-20261028/cpu-reference-demo-final`.
CPU measurements must not be compared with the earlier GPU-native LFM timings.
See `tests/fixtures/selectable_voice_20261028_validation.json` for final evidence
and the independent limitations of the ASR spot check.

The same 14-word text was then tested on the local **RTX 3090 Ti**, with actual
CUDA allocations verified for both raw engines. LFM used F16 assets, an 8,192
context and one slot; Qwen used its existing Q8_0 talker/F16 codec. The loopback
experiment measured:

| Path | First audio | Audio duration |
| --- | --- | --- |
| Native LFM, first response | 193 ms | 5.12 s |
| Native LFM, second response | 68 ms | 6.80 s |
| LFM text → Qwen narrator, first render | 1.487 s | 4.96 s |
| LFM text → Qwen British voice | 1.187 s | 5.20 s |
| LFM text → Qwen narrator, repeat | 1.124 s | 4.96 s |

Qwen timings include 121 ms to finish LFM's text, followed by complete-utterance
synthesis; no sentence pipelining was used. The first native response was colder
than the second. These are small observations, not distributional latency
benchmarks. The two engine processes occupied **9,054 MiB total** after
speech (LFM 5,522 MiB, Qwen 3,532 MiB); this is an allocation snapshot, not a
memory ceiling. The LFM session stayed open during Qwen synthesis, but microphone
decoding and speech generation were not exercised simultaneously.

Both references produced nonempty speech and their ASR spot checks recovered all
14 words. The repeated narrator's float PCM was identical on GPU as well. CPU
outputs also passed finite/non-silent/unclipped amplitude checks. These results
do not establish human-rated speaker similarity, naturalness or preserved
emotion. GPU WAVs and exact metrics are in the ignored `gpu-compare-v1` directory.

The original J worker was drained while these unregistered raw engines ran,
then restored with the same image, environment, mounts, full engine argv and
two fresh matching router heartbeats. Rollback tag:
`ezlocalai:rollback-voice-selection-20261028-local3090ti`.
No permanent environment settings changed.

Human listening and a combined client interruption test are still required
before choosing Qwen as a default call voice. Native LFM remains the
lowest-complexity default; custom Qwen voices should initially be opt-in.

## Voice conversion research

A real voice converter consumes LFM's waveform and changes its speaker timbre.
[Seed-VC](https://github.com/Plachtaa/seed-vc) provides a reference implementation
with a realtime mode and documented additional algorithm/device delay. That is
not a measured delay on this fleet, and the repository is a Python deployment,
not a native C++ component we can silently add to these images. A production
converter needs its own native runtime/weights, bounded streaming, cancellation,
GPU accounting and word/speaker/prosody evaluation. Qwen TTS itself is not that
converter.

Sources: [LiquidAI model card](https://huggingface.co/LiquidAI/LFM3-Omni-2B),
[pinned llama.cpp TTS guide](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/tts/README.md),
[Qwen cloning interface](https://github.com/QwenLM/Qwen3-TTS#voice-clone).
