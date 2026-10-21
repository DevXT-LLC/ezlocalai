# Native bounded audio stream bridge

`audio_stream_bridge_cases.jsonl` contains 58 cases captured from the complete,
unchanged `AudioStreaming.stream_native_audio` function at reference commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The fixture header pins
AudioStreaming.py SHA-256. The external capture program is
`/tmp/ezlocalai_audio_stream_bridge_oracle.py`; only fixture data and C++ tests
are included in this repository. The capture uses actual CPython 3.12 asyncio,
threads and AnyIO, with controlled model/iterator boundaries and synchronization
gates. It does not replace the source queue or teardown implementation.

Comparisons cover complete and empty streams, iterator errors, iterator-close
errors, model-close errors and their precedence, unstarted close, abandonment
with a blocked or already-finished producer, cancellation while waiting,
producer-factory failures and BaseException propagation. Each case compares
all delivered audio/rates, producer open/next/close traces, whether callbacks
run on the consumer thread, model-close calls and producer completion. Three
slow-consumer cases verify four prefetched audio values: one held by the
consumer, two queued and one blocked in the producer's pending send.

The native `CttsStreamBridge` is lazy and uses a capacity-two queue. It owns a
producer thread, closes the iterator on that thread, and runs model.close on a
separate worker during incomplete teardown. Normal teardown joins the producer
before returning. A failed model.close has distinct source behavior: it returns
that error without awaiting the producer. The native bridge retains shared
state for this still-running producer. A gated case verifies that the producer
is still pending when the close error returns, then releases it explicitly.
A producer-factory error or uncaught BaseException does not enqueue an error;
as in the source, cancellation is needed to leave the waiting consumer and
observe that producer failure. Source unobserved-task logging after a failed
model.close is not reproduced.

`TtsStreamIterator` adds lazy native-worker iteration and explicit close.
Advancement and iterator close belong to one producer thread. Model.close may
interrupt it from another thread while native inference waits. The worker's
inference lock is retained across yields; callback-style streaming delegates
to this iterator. The iterator retains the native model state, including when
the original C++ model owner is released. Exhaustion or close releases that
reference. The native CTTS backend now exposes the iterator factory.

`audio_stream_bridge_process` checks 11 cases against the existing C++ worker
process fixture: complete streaming followed by buffered recovery, unstarted
close, abandonment, cancellation while the process hangs, consumer/worker/
reference errors, lazy iterator close and three iterator/owner lifetime cases.
It checks that incomplete streams reap the worker before returning and that
cancellation completes promptly without waiting for the inference timeout.
The existing 115-scenario/205-operation source process suite also continues to
exercise the callback API, native request/error behavior and cross-thread close.

The real-model `llama_tts_model_reference` now routes its streamed requests
through this bridge and native iterator. It retains the existing unchanged-source
fixture, pinned model/worker/speaker and complete float32 hashes, including the
72-frame request and final short window. See `llama_tts_README.md` for provenance.
The bridge forwards each native frame without audio conversion.

## Remaining integration

This is a synchronous pull interface with an explicit cross-thread cancellation
request. CTTS byte framing, silence, drain scheduling and chunk teardown now
compose with it as documented in `ctts_stream_README.md`. Asynchronous HTTP
consumption and TTS pool/voice-guard integration remain unfinished. The CTTS
composition preserves header-versus-chunk cancellation and the distinct bridge
model-close and controller-close calls.

The controlled cases do not prove every asyncio/AnyIO scheduler interleaving,
repeated cancellation scope, malformed arbitrary Python object/JSON boundary,
logging behavior or non-Linux platform behavior. There is one consumer for
next/close; request_cancel is the concurrent operation. Iterator operations
must remain on their producer thread. Full speech and application parity remain
incomplete; speech manifest rows remain partial.
