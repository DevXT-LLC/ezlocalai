# Native local speech WebSocket integration

`/v1/audio/speech/ws` now runs the local C++ TTS pool and CTTS stream. The source
endpoint authenticates before accepting, uses local synthesis even with a voice
server configured, and acquires a voice slot only when text is ready. Native
code implements these rules, a four-segment bounded queue, per-message 30-second
receive/send deadlines, cancellation, batch lease ownership and close handling.
The client can continue sending text while synthesis runs. Each session emits
one PCM header, each frame length and PCM bytes, one zero-length terminator and
the legacy empty binary notification.

All captures use unchanged source at revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Source SHA-256 values:

- `TTSWebSocket.py`: `4ab3efbea3f2c6bcb3b1cea8f641985522fe11866b6db48e52b8e8ec84f020d1`
- `app.py`: `fab365c963725d80c81063b544d5b18b4cf931263573bdcb236e537519c7d365`

The capture scripts are external reference tools under `/tmp`; first-party
implementation and executable tests in this repository remain C++.

| Fixture | Coverage | SHA-256 (uncompressed for gzip files) |
| --- | --- | --- |
| `tts_websocket_cases.jsonl.gz` | 1,788 PCM parser cases and 1,891 input sequences | `6faf50b3bab1e924597be30b3cf55faed1067838f97ac3f144dc4973fcd3b9d6` |
| `tts_ws_lease_cases.jsonl.gz` | 352 ownership scenarios, 1,408 operations | `860e708b5672d1ef5ccb16cda4d86aad6bcf8a9546c1090b326d9c3848ce8046` |
| `tts_ws_api_cases.jsonl` | 127 endpoint/session cases, 254 native TCP comparisons | `cdee22467e6fb9c6a315c18f560c6e794761c6fbe3b78ed378b2fb43c2d5d539` |
| `tts_ws_session_cases.jsonl` | Seven complete concurrent sessions | `fe5b8260973af17a6d10c3065a57657ced1110148d548f6fd7542e300f0e24c9` |
| `tts_ws_model_cases.jsonl` | 67 real Qwen WebSocket frame hashes | `8f3903f1325b586af0186268ad3812545dd3546e6049a2172f0928ba4d3f4ff5` |

The scripts are `/tmp/ezlocalai_tts_websocket_oracle.py`,
`/tmp/ezlocalai_tts_ws_lease_oracle.py`, `/tmp/ezlocalai_tts_ws_api_oracle.py`,
`/tmp/ezlocalai_tts_ws_session_oracle.py`,
`/tmp/ezlocalai_tts_ws_transport_oracle.py` and
`/tmp/ezlocalai_tts_ws_model_oracle.py`.

The PCM/input capture executes the unchanged `segment_events` and nested
`receive` definitions, using real asyncio scheduling and controlled streams and
sockets. It covers arbitrary fragmentation, partial emitted output before errors,
frame limits, malformed/truncated streams, Unicode whitespace and codepoint
limits, sentence/word boundaries, voice/language changes, Python JSON numeric
truthiness and input diagnostics. Numeric normalization does not narrow Python
integers to native integer widths. Lone-surrogate JSON strings remain unverified.

The lease capture executes the original nested synthesis coroutine with a
prefilled real asyncio queue and the existing unchanged Pipes pool/guard
reference harness. It compares model/guard/resource events and snapshots across
transient, preload, dedicated and handoff policies, including acquisition,
load, stream, close, destroy and release failures. Source cleanup precedence
is preserved. The first capture's model mock did not accept positional stream
arguments; its signature was corrected before generating the accepted fixture.

The API capture executes original authentication, endpoint and complete session
code with a controlled socket/model. Native comparisons use actual TCP, both
ordinary frames and JSON fragmented across frames coalesced with the upgrade
request. They compare status, exact frames, model calls and released ownership.
Each case has at most one ready segment to avoid imposing an incidental
cross-thread schedule on source asyncio batching. A deliberately unusable
VOICE_SERVER verifies that the native WebSocket still uses its local pool.

The complete-session capture tests queue backpressure, receive failure during
admission or blocked synthesis, stream-next failure, close-error precedence and
send timeout. At the blocked-acquisition barrier, six messages have been
received: one is held for synthesis, four await synthesis, and the sixth waits
for queue capacity. The successful queue case compares audio, total segments
and balanced ownership rather than the exact number of batches: thread
scheduling can change when the queue empties. Prefilled-batch ownership is
compared exactly in the separate lease fixture.

`tts_ws_transport_cases.jsonl` runs the actual source endpoint on Uvicorn 0.40.0,
websockets 10.4 and Starlette 0.49.3. Fixture SHA-256: `a4efaadff03aa90f43c38bdeaa47a5b99168ff284546d4f42d6094eb7bfb2f2f`. Its 43 TCP captures cover ping/pong, binary
application messages, UTF-8 including unfinished fragments, masking, reserved
bits/opcodes, continuation ordering, control-frame sizes, oversized messages,
extended length encodings and close codes/reasons. The capture found and fixed
native rejection timing for invalid UTF-8 in unfinished fragments. Compression
negotiation, keepalive pings and exhaustive handshake/timing behavior remain
outside this capture.

`tts_api_process` additionally exercises real C++ CTTS/LlamaTts blocked workers
over WebSocket TCP. Disconnect and server shutdown are tested in transient,
preloaded, dedicated and LLM-handoff modes, alongside the four existing HTTP
cases. It checks idle composition without a lease, worker cancellation/reaping,
queued request admission, permit/lease release and resource counts. A trace-file
creation race in the new test harness was corrected before accepting its run.

`tts_api_model_reference` extends its two buffered WAV/cache-hit requests and
65 HTTP packets with a real WebSocket session. The reference capture performs
that same operation sequence with the same native worker, CPU model, speaker,
seed, chunk and audio-library pins from `tts_service_model_cases.jsonl`.
It executes unchanged CTTS, LlamaTTS, AudioStreaming and full WebSocket session
code. The native executable compares all 67 binary frame sizes/SHA-256 hashes
and normal close, then exits normally on SIGTERM. This is CPU evidence; GPU
synthesis remains unverified.

Remaining scope includes complete data-URI and local-voices routes, all
cross-modality handoff/model-management paths, unsupported Python values,
exhaustive transport/cancellation schedules, compression and keepalive behavior,
GPU synthesis, deployment discovery and packaging. The speech manifest rows
remain partial and the overall 1:1 parity goal remains incomplete.

Verification: all six focused WebSocket/process targets passed in 30.81 seconds.
The complete CXX-only Debug build was warning-free and all 113/113 tests passed
in 818.84 seconds. The real native executable comparison passed in 66.37 seconds
and the existing native HTTP suite passed in 31.37 seconds. Full log:
`/tmp/ezlocalai-tts-ws-full-tests.log`. No implementation changed after that run.

The 2026-10-20 live CUDA acceptance additionally runs this route on the deployed
RTX 3090 Ti image with Qwen. Normal completion validates the 24 kHz, 16-bit mono
header, PCM frame lengths, terminator, empty notification and normal close.
Abrupt client disconnect during synthesis clears speech admission and loaded
state, restores the LLM on CUDA and permits a routed follow-up chat. GPU process
allocations are observed while synthesis is active. These are local WebSocket
checks; Rust WorkerRouter does not expose this route. See
`stream_cancellation_20261020_validation.json`. GPU synthesis and packaging are
therefore verified for this image and hardware; waveform equivalence and the
exhaustive transport/scheduling limits above remain unproven.
