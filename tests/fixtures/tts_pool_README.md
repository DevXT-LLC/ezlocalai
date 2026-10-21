# Native TTS pool source comparisons

`tts_pool_cases.jsonl.gz` captures 748 scenarios and 22,353 operations from the
unchanged `Pipes.py` at reference revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Its header pins the complete source
and `ezlocalai/LlamaTTS.py` hashes. The external capture generator is
`/tmp/ezlocalai_tts_pool_oracle.py`; native test execution does not use Python.

The capture executes the nine original TTS creation, naming, load, registration,
warmup, acquisition, transfer and cleanup methods extracted by AST, along with
original ResourceManager registration/usage/removal methods and the original
model-ID resolver. Only model/system callbacks, environment, clock, request
identity and cleanup scheduling are substituted. Every operation compares its
return/error and pool, model, pending-cleanup and resource state. The test also
compares all callback events in order, including device/closed property reads,
clock calls, admission's explicit 4.0 GiB requirement, and environment reads.

Coverage includes capacities -1/0/1/2/4, preload/transient residency, CPU fallback
settings, forced CPU, missing dependencies and null model leases, live model-ID
migration/configuration, mixed device labels, missing device properties, closed
model release, repeated keys, FIFO reuse, resident capacity fallback, lease
transfer/overwrite/missing keys, forced global unload and stale available/lease
entries. Ordinary exceptions, AttributeError and selected BaseExceptions are
injected into callbacks. Deterministic randomized sequences use seed 8919.
Admission's boolean is deliberately ignored, matching the source. A forced
unload with no current model clears pool collections but retains other requests'
lease entries. Resource usage remains reference-counted even for repeated gets.

`tts_pool_reference` also checks 32 concurrent keys against an eight-instance
resident pool, reentrant transfer during resource marking, and the shared
STT/TTS cleanup queue's FIFO serialization and recovery after ordinary errors.
`tts_pool_process` adds six cases using actual native worker processes and the
complete Ctts factory, ResourceManager and VoiceSlotGuard. Resident, transient
and handoff modes each exercise buffered synthesis, streamed completion or
blocked-stream cancellation, lease transfer to another thread, a waiting voice
request, cross-thread slot release, resource removal and child reaping.
The process fixture generates deterministic audio; it is not a GGUF inference
comparison. Existing CTTS real-model tests separately verify exact CPU audio.

The native pool accepts integer request IDs and capacity and typed string/bool
model properties. Arbitrary Python objects/equality/key types, unbounded
integers, all reentrant/concurrent warmup interleavings, logging and GPU pool
inference are not established by this coverage. Cleanup uses the existing
process-wide FIFO worker. That worker continues after all C++ exceptions; the
Python worker can terminate on a BaseException outside Exception. Direct pool
cleanup preserves captured BaseException propagation, but that queue-level
edge remains open. The pool and slot guard are composable native components;
public speech routes, voice/router/fallback orchestration and asynchronous HTTP
disconnect integration still require porting and verification.
