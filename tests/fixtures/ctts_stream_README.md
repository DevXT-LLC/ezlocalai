# CTTS byte streaming composition

`ctts_stream_cases.jsonl.gz` contains 143 scenarios captured from the complete,
unchanged CTTS module and AudioStreaming bridge at reference commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Its header pins both source files.
The external capture program is `/tmp/ezlocalai_ctts_stream_oracle.py`.
Only fixture data and C++ tests are included in this repository.

The source capture uses actual CPython 3.12 asyncio/AnyIO, NumPy 1.26.4,
soundfile 0.13.1 and a real temporary filesystem. Model/hardware boundaries are
controlled; the bridge itself is unchanged. CTTS sleep calls are recorded and
yield to the real event loop without waiting for the requested positive delay.
Selected sleep calls raise configured exceptions. Native sleep hooks reproduce
that boundary. The production native default performs interruptible waits.

Each scenario compares every emitted write byte for byte, including write
boundaries, requested sleeps, producer iterator open/close calls, model close
calls and caller/worker placement, cleanup events, sample rate and final
controller/model state. Only temporary-root prefixes are normalized.

Coverage includes text/language/voice preparation, empty input, reference-path
failures, frame sizes, write-size limits, empty frames, channel mixing, clipped
and non-finite audio, changed/invalid rates, header overflow after buffered
sample-rate changes, inter-chunk silence, drain sleeps and errors, model/iterator/
close failures, kwargs collisions, and close/cancellation at individual yields.
The largest silence settings exercise source overflow errors without allocating
large audio buffers. Cancellation-at-yield cases inject CancelledError into the
source async generator and use the native stream's explicit cancellation request.

The native stream preserves these distinctions:

- The eight-byte little-endian rate/16-bit/mono header and terminal four-byte
  zero are outside the chunk exception handler. Closing there leaves the CTTS
  owner open. Empty prepared text still resolves the voice before emitting them.
- Frame lengths and bounded PCM payloads are separate writes. Each payload is
  followed by a zero-delay sleep when iteration resumes. Silence between text
  chunks is one combined length/payload write, followed by the configured drain.
- Errors or cancellation inside a chunk close the bridge and then the CTTS
  owner. Those are distinct model-close calls. Iterator/model cleanup errors
  retain source precedence. Streaming does not use buffered CPU fallback.
- Stream construction is lazy after argument binding and retains the CTTS state.
  Closing or exhausting the stream releases that state and its audio buffers.
  A failed preprocessing/header operation releases the stream without closing
  an otherwise retained CTTS owner.

## Actual model and process checks

`ctts_stream_model_cases.jsonl` records six operations through complete source
CTTS construction/warmup and actual CPU Qwen inference: close at the initial
header, empty text, a two-text-chunk stream with inserted silence, another
complete stream, close after a frame length, and buffered generation after
that close. It compares all 103 write lengths/SHA-256 values, exact errors and
controller state. The external capture is
`/tmp/ezlocalai_ctts_stream_model_oracle.py`. The fixture pins CTTS, the bridge,
LlamaTTS, the native worker, both GGUF model files and the reference WAV.

This uses the same model/worker dependencies as `ctts_runtime_README.md`, with
seed 23, four worker threads, eight generation tokens, four warmup tokens,
stream targets 12/8/1, 1024-byte writes and one millisecond of inter-chunk silence.
The native runner also verifies producer-thread iterator creation and the
absence of buffered Torch inference mode in streaming. CPU-only synthesis and
small-token frame-limit warnings are intentional.

`ctts_stream_process` checks nine additional cases with the native process
fixture: normal completion, blocked-worker cancellation, ordinary/very large/
infinite drain cancellation, payload abandonment, unstarted/header owner
retention and unstarted cancellation. Repeated cancellation requests must not
interrupt teardown. It checks worker reaping before release and prompt return
without waiting for the synthesis timeout or drain deadline. Long finite waits
use bounded clock intervals to avoid native duration overflow.

## Remaining scope

`Ctts::generate_stream` returns a synchronous pull stream. One consumer owns
next/close; request_cancel may be called concurrently. The shared controller is
retained until stream close/exhaustion. HTTP/event-loop scheduling, transport
backpressure/disconnect handling, TTS pools/voice guards and public speech
endpoints still need integration. These fixtures do not prove every scheduler
interleaving, mutable-attribute/concurrent-generation race, arbitrary Python
object/JSON boundary, logging behavior, GPU synthesis or dependency packaging.
All affected speech manifest rows remain partial; full 1:1 application parity
is not established by these checks.
