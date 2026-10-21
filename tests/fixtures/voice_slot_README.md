# Native voice-slot reference captures

The fixtures execute the unchanged `_VoiceSlotGuard` class in `Pipes.py` at
reference revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Each header pins
the complete source file's SHA-256. The external capture programs are
`/tmp/ezlocalai_voice_guard_oracle.py` and
`/tmp/ezlocalai_voice_guard_async_oracle.py`; neither is a runtime/test dependency.
All first-party implementations and test runners are C++.

`voice_slot_cases.jsonl` contains 735 scenarios and 2,804 actions. Substituted
acquisition, lifecycle and availability boundaries record exact callback order,
returned state, resource flags, held permits, per-key context stacks and errors.
Coverage includes STT/TTS dispatch, capacity clamping, nested context entry,
missing-key exit, cross-task release, repeated release, empty/truthy handoff
state, failures and cancellation at each boundary. Compound failures verify that
cleanup errors can replace an earlier error and stop the remaining cleanup,
while the lease remains marked released. These intentionally preserve source
behavior rather than assuming that every failed cleanup releases all resources.

`voice_slot_async_cases.jsonl` contains 11 scenarios and 84 actions using real
`asyncio.Lock` and `asyncio.Semaphore`. Controlled task scheduling covers FIFO
waiters, head/middle/tail cancellation, service capacity two, shared STT/TTS
handoffs, cancellation while waiting for LLM idle and cancellation after unload
while waiting for a service permit. The native runner uses actual threads and
FIFO queues. It compares settled queue/resource/lease state, exact callback order
within each request, global acquisition order and the global handoff callback
sequence. Independent completion notifications may interleave differently on
threads than on a single event loop, so those notifications are compared within
their request rather than by an artificial total order across requests.

Additional native checks exercise a 16-thread duplicate release and 100 races
between cancellation and permit wakeup. Cancellation subscriptions wake queues
without polling, and cancelled waiters leave the remaining FIFO order intact.

This establishes the covered guard/admission behavior, not complete ASGI/HTTP
cancellation equivalence. The native server still needs to connect request
lifetimes, LLM unload/restore operations and the STT pool to this guard. The public
speech endpoints still use the previous whisper.cpp adapter. GPU pool behavior,
all concurrent application interleavings and arbitrary service/capacity types
remain outside these fixtures.
