# Retain shared conversation checkpoints within the actual byte budget

The seventh managed patch applies only to the qualified llama.cpp pin
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. It changes retention decisions;
model execution, checkpoint serialization and configured limits stay unchanged.

## Shared backing accounting

`common_state_blob` lets cached conversations share immutable target/draft
checkpoint vectors. Previously each owner charged the whole vector capacity to
the RAM ceiling. Two branches could therefore evict each other even when their
retained allocations fit. For example, two 192 KiB full snapshots sharing a
1 MiB checkpoint consume about 1.375 MiB, but were charged about 2.375 MiB.

Cache size and prospective admission now count each retained checkpoint backing
once, using the const data pointer as its identity. Distinct nonempty vectors
cannot share that address. Full snapshots, speculative CPU vectors and spare
capacities keep their separate charges. Copy-on-write detachment creates a new
allocation and is charged separately. With storage sharing disabled, incoming
checkpoint copies are charged separately too.

The protected restore check uses the union of the protected snapshot and the
incoming prompt. Eviction uses the union of all retained snapshots and that
prompt. Both the predicted allocation and the completed clone must fit. These
checks allocate no identity set, so they remain usable during allocation
failure. The scan is quadratic in checkpoint blob fields, and eviction can
repeat it; cache-depth benchmarks must check that cost. No GPU work or state
transfer is added.

This preserves the existing budget scope: cached snapshots and their spare
buffers. Checkpoints owned solely by the live slot remain runtime overhead;
the cache ceiling is not a process-wide RSS limit.

## Refreshing an existing boundary

A restored boundary can be captured again at the start of prefill. Previously
the checkpoint list made room before removing the entry at that same token
count. A full list could lose an unrelated oldest checkpoint and then finish
one slot below its limit. Remove superseded boundaries first, then apply the
existing minimum-step, current-task and capacity policies. Intentional
minimum-step cleanup can still remove nearby older checkpoints.

The production list policy is an inline helper exercised by the existing
checkpoint coverage test. Cache storage tests use the modified production
allocator, including shared and independent branches, protected restores,
copy-on-write, spare buffers, disabled sharing and failed saves/restores.
Managed-patch checks cover pristine installation, previous-prefix upgrades,
idempotence, hashes and complete removal.
