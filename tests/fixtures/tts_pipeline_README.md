# Local audio data-URI speech pipelines

The public buffered speech route now runs `Pipes.audio_to_audio` behavior through
native STT and TTS services. Audio data URIs bypass voice/router/general remote
providers. The response is CTTS's JSON-encoded URL/base64 string; it is not the
raw WAV response used for ordinary text speech. STT/TTS guards are acquired in
sequence, with explicit resource marks. Normal inference failures skip the
later pool-destroy call, but still release the guard. Successful dedicated voice
mode retains the pool. This path does not invoke the public speech endpoint's
gTTS fallback.

The source first base64-decodes the URI payload and passes the resulting bytes
to STT. STT itself performs another base64 decode. Native code deliberately
preserves this behavior and its errors. The STT service, pool callback and
runtime now have a typed byte-input path, retaining the distinction between
bytes (whose nonalphabet octets are ignored) and Unicode input (which must be
ASCII). The same native decoding implementation serves both paths.

Source revision: `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.

- `Pipes.py`: `8238cef03d4a1db6e775d7f62234bcf998d8f08bd9fc15f1d8a36ea0d3280005`
- `app.py`: `fab365c963725d80c81063b544d5b18b4cf931263573bdcb236e537519c7d365`
- `STT.py`: `d0dd8340c243d7ff9ba84f4e3babe3c41b0b0ce7c7dddf2565eb886ba3f21fe2`

| Fixture | Evidence | SHA-256 (uncompressed for gzip) |
| --- | --- | --- |
| `tts_pipeline_lease_cases.jsonl.gz` | 816 source TTS lifecycle scenarios / 3,264 actions | `1b4cbb41bdecf95acf38a9c83dbe9368dfb4091d2f5be971833ad153a57d2050` |
| `stt_tts_pipeline_cases.jsonl` | 32 source STT pool scenarios / 128 calls | `12835998f3207ebd06cf58126964b525e80ed938dc47cfccc1c8c8c2d1d76ad5` |
| `stt_runtime_bytes_cases.jsonl` | 570 complete source runtime byte-input cases / 1,140 actions | `6d51db12deaa92a54945653517c02247ae8109032879bef9b34d1ed680919d62` |
| `tts_audio_api_cases.jsonl` | 630 full source audio data-URI API cases | `5155318c218e1851fb769d102dc01ce9a6e49fb0f781e7c5315c72c4ae376851` |

Reference capture scripts are external tools:
`/tmp/ezlocalai_tts_pipeline_lease_oracle.py`,
`/tmp/ezlocalai_stt_tts_pipeline_oracle.py`,
`/tmp/ezlocalai_stt_runtime_bytes_oracle.py`, and
`/tmp/ezlocalai_tts_audio_api_oracle.py`. No Python implementation or executable
test is added to this repository.

The TTS lease capture executes the unchanged TTS async-with blocks from both
`Pipes.audio_to_audio` and `Pipes.pdf_to_audio`, composed with the original
pool, voice guard and resource-manager methods. It preserves exceptions from
acquisition, loading, resource marks, generation, mode lookup, destruction,
scheduling and release, including BaseException families and cleanup precedence.
It tests transient, preloaded, dedicated and handoff modes. Native comparisons
check every action's result, full event sequence and pool/resource/guard state.
The initial capture incorrectly tagged the fixture with app.py's SHA because
the reused harness rebound the `source` variable. That metadata was corrected
and the fixture recaptured before its accepted comparison.

The STT fixture similarly executes the original STT async-with block, with
controlled incoming audio/model callbacks, four ownership modes, capacities
one/two, repeated requests, load/inference errors and recovery. It compares
state and active-use counts at inference. Actual raw bytes and decoding are
covered separately, rather than claiming this isolated lifecycle capture proves
the complete audio conversion path.

The byte fixture executes unchanged STT construction, `_transcribe_audio_sync`
and close with controlled inference and filesystem boundaries. Cases include
all 256 single-octet inputs, raw binary audio-like content, padding/junk suffixes,
non-ASCII octets, nulls/newlines and seeded randomized mixtures. The native
runtime suite compares outputs, exception categories/messages, file bytes,
operation order, model state and cleanup, alongside its previous 800 cases.

The API fixture executes the original Pydantic request class, FastAPI endpoint
and complete `Pipes.audio_to_audio`, with controlled local model boundaries.
Native comparisons verify HTTP status/content type, JSON or error response,
exact raw bytes delivered to STT, the STT result's type passed to CTTS,
voice/local_uri arguments, call order and released voice permits. They cover
malformed data URIs/base64, ignored suffix components, empty/non-ASCII input,
validation, null/typed results, disabled flags and whitespace/casing, ordinary
and fallback-eligible errors, local URI preservation, and configured/dedicated
voice servers. Pool cleanup is established by the separate lifecycle fixtures;
these endpoint mocks do not independently prove it. JSON object ordering and
content-length spelling are not compared.

The new source comparisons also correct STT's dedicated-voice check: it lowercases
VOICE_SERVER without stripping whitespace, matching the source client. Complete
client caching/configuration transitions remain a separate parity gap.

The older native HTTP test expected transcription followed by remote TTS for an
audio data URI. That assertion was removed; its server has no local model
factories and now checks failure when they are absent. The new source-backed
API and model tests exercise the actual local factories.

The TTS service lifecycle also supports PDF calls' output filename and raw
result, but the public PDF adapter and text extraction have not yet been ported
to full source parity. The existing PDF adapter still uses pdftotext and older
forwarding. The draft 553 audio and 413 PDF orchestration captures under `/tmp`
are additional preparation, not accepted native comparison suites. Remaining
scope includes complete PDF/local-voice routes, client cache transitions,
exhaustive cross-modality handoff and concurrency, GPU TTS, dependency variants,
model discovery and deployment packaging. Speech routes remain partial.

`tts_audio_model_cases.jsonl` captures the unchanged full audio-to-audio method,
STT async/sync methods and real CTTS using the pinned tiny.en Whisper and Qwen
models. Its metadata embeds the existing STT and TTS model, audio, environment
and dependency pins. The operation sequence matches the extended native API
test: two buffered Hello requests, the HTTP stream and WebSocket session, then
the audio pipeline. The successful input is the base64 encoding of an already
base64-encoded speech.wav. Source output is a 184,364-byte WAV, SHA-256
`f4cc267240aa04dbc17e17a10c20b3044ce516744dd642f437a78ed5816066e3`, returned as a
245,820-character base64 string. A normally encoded speech.wav data URI instead
raises ValueError with the source invalid-audio message; the public route maps
it to HTTP 500. This distinction is intentionally retained.

Capture script: `/tmp/ezlocalai_tts_audio_model_oracle.py`. Fixture SHA-256:
`c09296276cd1a62cedc4f2799a113306e3871a47a5c8293b2f5295b03edfffaa`.
The capture uses CPU inference and real PortAudio initialization. Pool/guard
boundaries are controlled to retain the preloaded models, while separate pool
fixtures compare their exact ownership rules. The native executable test is
extended when STT model/VAD test assets are configured; TTS-only builds keep the
existing HTTP/WebSocket comparison without requiring STT assets.


The initial combined model capture accidentally imported faster-whisper 1.2.1
from the user site and Silero VAD v6, despite copying the 1.2.0 metadata. The
first executable comparison correctly rejected its audio hash. An independent
trace located a punctuation difference in STT; varying four/eight threads did
not remove it. Explicitly loading the pinned 1.2.0 package and its encoder/
decoder v5 VAD assets made source/native transcripts identical and produced the
native-observed WAV hash above. That corrected source capture is the accepted
fixture. The test now checks profile, actual STT thread count, model and VAD
hashes; the capture asserts its package version and records package/VAD hashes.
No inference implementation was changed to fit the mistaken 1.2.1 expectation.
The current comparison covers the executable's 1.2.0 profile; it does not prove
1.2.1/VAD-v6 parity. The rejected capture is retained outside the repository at
/tmp/tts_audio_model_1_2_1_invalid_capture.jsonl for diagnosis.

The accepted 1.2.0 profile matches the pinned reference's
cuda-requirements.txt and rocm-requirements.txt. Its generic requirements.txt
pins 1.0.3; that legacy STT profile has separate existing native model suites.
The executable's current modern-profile comparison does not substitute for
matching each deployment's dependency selection and packaging.
