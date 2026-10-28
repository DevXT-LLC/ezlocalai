# LFM3 Omni voice calls

The supported release is [LiquidAI/LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF).
It uses llama.cpp v0.8.0's native audio/vision projector and speech codec.
Qwen remains the ordinary `/v1/audio/speech` engine; a realtime call does not
chain Whisper, a text model and Qwen TTS.

## Configuration

A dedicated voice worker can start with:

```dotenv
DEFAULT_MODEL=LiquidAI/LFM3-Omni-2B-GGUF
QUANT_TYPE=F16
LLM_MAX_TOKENS=32768
N_PARALLEL=1
KV_CACHE_TYPE=auto
TTS_ENABLED=false
STT_ENABLED=false
EMBEDDING_ENABLED=false
ACE_STEP_PRECACHE=false
```

Keep the normal worker URL, router registration and GPU settings. Acquisition
selects the backbone and the model-qualified `mmproj` together; cached files
work offline. Unsupported inherited Qwen quantization names select F16 for
this family. F16 weights/projector and automatic F16 KV are the quality baseline;
explicit cache/quantization overrides still apply, including per-card overrides.
Remove explicit per-card KV overrides to use this model's automatic precision.
The vendor's greedy text sampling and `--reasoning off` apply; the GGUF's audio
sampling metadata remains in effect. Explicit request sampling is respected.

LFM3 Omni's model card advertises **32K** supported context. Its config allows
128,000 positions, which does not establish quality at that length. The
quality-preserving profile uses **32,768** tokens. A larger inherited
context is capped only for this model, and automatic parallelism selects one
slot. **Qwen's 262,144-token configuration is unchanged.** `N_PARALLEL` divides
the engine context between slots as usual.

For an existing local GGUF, set `LLAMA_CPP_MODEL_PATH` and
`LLAMA_CPP_MMPROJ_PATH` to its matching backbone and complete audio/vision
projector. A missing projector fails the realtime upgrade.

## Connection and events

Connect to `/v1/realtime?model=LiquidAI%2FLFM3-Omni-2B-GGUF` using the worker's
`Authorization: Bearer` key. The Rust WorkerRouter exposes the same endpoint
with its normal client key and optional `worker=<label>` or `worker_id=<id>`.
Both direct workers and updated reverse tunnels support it. Unsupported workers
are excluded; failed or disconnected calls are never replayed on another model.
Authentication and backend preparation finish before HTTP 101.

Each call holds an ordinary text/model slot and an active model lease for its
whole connection. Heartbeats advertise `extra.realtime` with
`protocol=openai-realtime`, `shared_capacity=text`, the model list and tunnel
support. This does not add independent GPU capacity. Disconnect, failed startup
and shutdown release the lease. Admission waits are bounded to 30 seconds;
disconnected direct callers stop waiting for a slot. Keep connections alive
while idle; the worker's raw duplex relay has a 600-second inactivity budget.

1. Read `session.created` and its `session.audio.input.format.rate` and
   `session.audio.output.format.rate`. Audio is mono little-endian PCM16.
2. Send `session.update` before the first message/audio. Configure instructions,
   tools, output modalities and a finite `max_output_tokens` for the call.
3. Send user text with `conversation.item.create` and `input_text` content, or
   base64 PCM chunks using `input_audio_buffer.append`, followed by
   `input_audio_buffer.commit`. Send `response.create` to request the answer.
4. Consume `response.output_text.delta`, `response.output_audio.delta`, tool
   argument events and `response.done` incrementally. Run permitted tools in
   the client, then send `function_call_output` with the original `call_id`
   and request the next response.
5. On interruption send `response.cancel`, stop queued audio playback and
   finish cancellation before requesting the next turn. Microphone capture,
   voice activity detection, playback scheduling and tool authorization belong
   to the calling application.

Instructions and tool definitions are immutable after the first turn. Send
changing working-agent progress as additional conversation items between
responses; keep the working agent and its context separate from the voice
model. For native Omni transcription, set `session.type=transcription` in
the initial `session.update`. The optional `?intent=transcription` query is
forwarded upstream; it does not replace the required session type for a
conversational model.

For selectable or cloned output voices, see [voice paths and the measured
feasibility limits](VOICE_SELECTION.md).

## Images and video

The upstream realtime protocol currently accepts text and audio. Use
`/v1/chat/completions` for `image_url`, `input_audio` and `input_video` content
with the same model. Omni input bypasses the legacy STT/video-to-text adapter;
the native model sees frames and hears audio itself. Video decoding needs the
packaged `ffmpeg`. For video questions, the vendor recommends the system prompt
`Answer questions about the video using what you see and hear.` For standalone
ASR use `Perform ASR in English.` or the appropriate supported language.

A call occupies its text slot even between utterances. A concurrent visual
tool therefore needs another available slot or worker. Do not advertise images
or video as supported realtime WebSocket event types. Multiple slots, long
calls and complex audiovisual accuracy need separate workload validation.

See [the dated evidence](tests/fixtures/lfm3_realtime_20261028_validation.json)
and [upstream's protocol](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/server/README.md#websocket-v1realtime-realtime-transcription)
for the tested scope and event contract.
