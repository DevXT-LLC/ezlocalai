# Selectable voices for LFM3 calls

Native LFM speech remains the default. Optional **MeanVC2** converts LFM's
streamed waveform to a reference voice; Qwen remains the ordinary TTS engine.
The converter is a native C++ executable, with no Python service or runtime.

## Enable MeanVC2

Images include the CPU engine, built from audio.cpp
[`cd770076152085af5b166521185bec76f9d47808`](https://github.com/0xShug0/audio.cpp/blob/cd770076152085af5b166521185bec76f9d47808/docs/models/meanvc2.md).
Weights are downloaded separately into the persistent models volume. Provision
before enabling the feature:

```sh
docker exec ezlocalai sh -lc 'ezlocalai-meanvc2 --download-model "${MEANVC2_MODEL_PATH:-${MODELS_PATH:-models}/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf}"'
```

The command downloads only the pinned **F32** 120/40 model and checks its size
and SHA-256 before atomic installation. An existing invalid file is rejected.
Model revision: `874fcb41ececda20bcbc0782d6b4ea7f7d069ea8` in
[`audio-cpp/audio.cpp-gguf`](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/874fcb41ececda20bcbc0782d6b4ea7f7d069ea8/MeanVC2-GGUF).
SHA-256: `340d5dd63cdd9d44045cdc4b394cccf677ae870a9ce4ed18f0007c8788e65245`.

Add these settings to the worker's existing `.env` and recreate its container:

```dotenv
MEANVC2_ENABLED=true
MEANVC2_BACKEND=cpu
MEANVC2_THREADS=4
```

The worker needs a configured LFM3 Omni model as described in [REALTIME.md](REALTIME.md).
No production environment setting is changed automatically. CPU processing keeps
VRAM available to LFM. Allow at least 3 GiB free RAM; admission also checks a
finite cgroup v2 memory limit and reclaimable file cache. There is one converted
call slot per worker, sharing its existing text capacity. The model and one reference-conditioning cache stay pinned
for the call and are released on disconnect; it is not an ordinary TTS pool.

| Setting | Default |
| --- | --- |
| `MEANVC2_ENABLED` | `false` |
| `MEANVC2_BIN` | `/usr/local/bin/ezlocalai-meanvc2` |
| `MEANVC2_MODEL_PATH` | `${MODELS_PATH:-models}/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf` |
| `MEANVC2_VOICES_DIR` | `voices` |
| `MEANVC2_BACKEND` | `cpu` (packaged engine); custom CUDA/HIP builds unvalidated |
| `MEANVC2_DEVICE` | `0` |
| `MEANVC2_THREADS` | `4`, range 1–64 |
| `MEANVC2_STARTUP_TIMEOUT` | `180` seconds, range 1–600 |
| `MEANVC2_CHUNK_TIMEOUT` | `15` seconds, range 1–120 |

## Select a call voice

Choose a WAV basename from `extra.realtime.voice_conversion.voices` in the
router's worker record. Existing bundled references and recordings placed in
`MEANVC2_VOICES_DIR` are supported. Paths, escaping symlinks, missing files and
references over 32 MiB are rejected. Decoded reference audio must be nonempty,
finite and no longer than 60 seconds. Use clean speech without music or noise.

```text
ws://router/v1/realtime?model=LiquidAI%2FLFM3-Omni-2B-GGUF&voice=british_rp_female
```

The companion WorkerRouter update must be deployed first: it forwards the voice
query through direct and reverse-tunnel transports, selects workers advertising
that reference and reserves converter capacity alongside text capacity. Older
routers drop this query. **Check `session.created.session.voice_conversion`**
before playing audio; it must identify `engine: "meanvc2"` and the selected
`voice`. Absence of this marker means conversion was not selected.

`voice=native` or an omitted voice keeps native speech. Selection lasts for the
call. A matching `session.update.session.audio.output.voice` is accepted; changing
it requires a new call. Text, tool calls, incoming audio and context updates retain
the existing realtime protocol. Transcription-only requests cannot select conversion.
Conversion failures produce an explicit error/closed socket, with no native-voice
fallback. Send `response.cancel`, clear local playback and wait for cancelled
`response.done` before starting the next response.

## Streaming and quality limits

MeanVC2 receives continuous 16 kHz audio in complete 160 ms inference windows,
independent of transport packet boundaries. Final partial windows are padded;
short acknowledgements get the minimum context needed by its frontend. Padding
cannot extend output beyond the source duration. Input and output resamplers
retain their state between chunks. Queues are bounded and utterances are capped
at 120 seconds, because the upstream session retains merged output until reset.
Cancellation discards queued and in-flight results by generation and resets the
converter for the next response. Audio already delivered to a client must be
removed from its playback queue there; ignore later deltas for the cancelled
response ID, including frames already in transit.

The wire remains mono PCM16 **24 kHz**. MeanVC2's actual bandwidth is **16 kHz**;
upsampling does not restore higher frequencies. F32 avoids model quantization,
but conversion can still change timbre, emotion, pronunciation or naturalness.
Do not claim unchanged quality without listening tests. Call setup loads and
conditions the reference before the WebSocket upgrade; it is separate from
per-turn first-audio latency. Published MeanVC2/RT-VC delays are not measurements
of this complete pipeline.

## Local validation (2026-10-28)

RTX 3090 Ti, LFM3 F16 at 8192 context, F32 MeanVC2 on six CPU threads; isolated
Rust router, both direct and reverse-tunnel transports. Native remains the default.
These are individual observations, not a percentile benchmark or network SLA.

| Observation | Result |
| --- | --- |
| Native later-turn first PCM packet | 61 ms |
| Converted later-turn first PCM packet | 88–143 ms |
| Converted later-turn first energy-bearing packet | 111–241 ms |
| Converted speech after cancellation | 96–122 ms to first PCM |
| Converted connection setup | About 3.0–3.3 seconds, separate from turn latency |
| Source-control CPU conversion | 6.16 seconds of speech in 0.60–0.63 seconds |
| Sampled converter RSS before conditioning memoization | 1.68 GiB |

The first implementation recomputed reference conditioning after every response:
later-turn packets took 1.36–1.60 seconds and post-interruption packets 2.84–2.88
seconds. A bounded, content-keyed cache in the pinned MeanVC2 session now retains
the exact embedding and projected speaker memory across resets. The real-model
A/A/B/B/A probe measured reference startup falling from 1.50–1.76 seconds to
0.19–0.25 milliseconds on cache hits. Cached A/A and B/B conversions are
bit-identical, and changing the reference invalidates the entry. Recomputing A
after B was not bit-identical to the original A; voice changes require new calls
in the product. No quantization, solver or vocoder-window change was used.

Both transports passed 26 protocol checks each, including incoming audio, tools,
context updates, text-only turns, cancellation/recovery and resource cleanup.
Tiny live transport packets initially exposed a frontend failure; complete 160 ms
windows fixed it. Real F32 output is bit-identical for 160 ms, 20 ms and mixed
transport fragments. Short/empty input also finalizes without extending duration.

Ten 14-word same-family ASR diagnostics yielded 0–7.14% word error. The exact
same-source native/converted controls both had 7.14% in both transports; other
samples included one-word differences. An earlier strict spelling check failed
on “working” → “worken”. **This is not evidence of unchanged intelligibility or
speaker quality.** Human listening, longer speech, emotion, cross-gender and
multilingual checks remain necessary. No ARM, ROCm or custom GPU converter was
physically validated. No production environment setting changed; the original
worker was restored with identical image, environment, mounts and full engine
arguments and two matching router heartbeats.

The optional real-model probe is built with
`-DEZLOCALAI_MEANVC2_VALIDATION=ON` in `engine/meanvc2`; invoke
`meanvc2-session-validation MODEL SOURCE_24K_MONO_WAV VOICE_A_WAV VOICE_B_WAV`.
Package tests use fixtures and `--version` without downloading weights. Detailed
machine-readable evidence is recorded in the MeanVC2 validation fixture.

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
