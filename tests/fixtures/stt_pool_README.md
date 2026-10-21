# Native STT model pool reference captures

The source is the unchanged `Pipes.py` at ezlocalai revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Each fixture header pins its SHA-256.
`stt_pool_cases.jsonl` contains 540 scenarios and 12,480 actions executed through
AST-extracted `_load_stt_instance`, `_register_stt_pool_resource`,
`_warm_load_stt_pool`, `_get_stt`, `_destroy_stt_sync` and `_destroy_stt` methods.
The real source ResourceManager registration, removal and usage-count methods
also execute. The capture substitutes model/system callbacks, lease keys and
cleanup scheduling. It records every callback, model identity, close count,
resident/transient/available list, active lease, queued cleanup and resource
entry after each action.

Coverage includes capacities -1/0/1/2/4, CPU/CUDA device labels, repeated request
keys, FIFO reuse, capacity fallback, transient models, force/handoff cleanup,
preload changes, stale available/lease entries, callback exceptions and partial
state changes. Resource admission returns false/CPU even when a model reports
CUDA: the source uses that result and force_cpu for logging, without overriding
the STT constructor's device selection. Deterministic randomized action sequences
use seed 7819. The external generator is `/tmp/ezlocalai_stt_pool_oracle.py`.

`stt_pool_model_{modern,legacy}.jsonl` each adds 23 pool actions and seven actual
wrapper inference calls. The same source pool methods invoke unchanged STT.py
constructor, transcribe and close methods with pinned tiny.en weights, real
PortAudio initialization and faster-whisper 1.2.0/1.0.3 respectively. Four CPU
inference threads and one BLAS thread are used. Headers pin the model/audio
hashes and dependency versions; audio fixtures are reused from
`stt_runtime_assets`. These cases exercise two resident instances, returned
instance reuse, ordinary/word/batched output, transcription after close,
invalid audio recovery, transient loading, queued close and synchronous handoff
cleanup. The generator is `/tmp/ezlocalai_stt_pool_model_oracle.py`.

The initial capture harness had an event-argument name collision. Those invalid
captures were replaced before verification. The corrected fixtures contain
successful loads/inference as well as intentional errors.

Native tests additionally check concurrent request-key reuse/capacity and the
process-wide cleanup worker's FIFO order, serialization and continuation after
an exception. The model test uses the native CTranslate2/STT/PortAudio factory.
Neither implementation nor test execution invokes Python. These captures do not
establish all concurrent warm-load interleavings, cancellation/voice-slot guard
behavior, GPU pool inference, model switching or public speech API integration.
The public speech API still uses the earlier whisper.cpp adapter.
