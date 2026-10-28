# Keep an early named-conversation checkpoint

The ninth managed patch preserves the earliest still-valid checkpoint when a
named prompt admits newer checkpoints. It applies after the eight existing
cache/state/MTP/CUDA patches at llama.cpp v0.8.0,
`8642541b255bfb5dc39cd351a178be09013a55d6`.

Hybrid/recurrent state cannot be rolled back to an arbitrary token prefix. If
all retained checkpoints are later than a changed user/runtime suffix, matching
system tokens alone cannot restore that prefix and the engine reprocesses it.
The previous oldest-first count eviction could eventually remove the early
user-start checkpoint. More idle time is not itself the cause: engine caches
have no default wall-clock expiry, but routing, count/byte eviction, prefix
changes, model unload and deployment restarts can all lose reuse opportunities.

For named prompts with at least two configured checkpoints, admission protects
one existing boundary with the smallest token count below the incoming boundary.
Same-boundary refresh replaces its payload first. The anchor is found by token
position, not insertion order, so refreshing it at the list tail stays safe.
A newly evaluated earlier boundary becomes the anchor for subsequent admissions.
Remaining entries retain their existing minimum-step/current-task and oldest
victim rules. A one-checkpoint configuration keeps the newest state, and unnamed
prompts keep the previous policy.

Exact `EZLOCALAI_LLAMA_PREFIX_ANCHOR=0` restores the previous eviction policy
for diagnostics or rollback. Unset enables the guarded named-prompt policy.

The server still invalidates checkpoints beyond the actual shared token prefix
before admission. This patch does not treat a cache key as proof of identical
content, invent checkpoint bytes, change batch boundaries, skip required tokens
or alter model/KV precision. It allocates no additional checkpoint and retains
the existing byte budget. Protecting one early checkpoint can evict a different
recent checkpoint; the count ceiling does not grow. It cannot guarantee 24-hour
KV survival under memory pressure or across engine restarts.

The actual-header regression uses 40 turns and 120 later admissions to reproduce
loss of the early state with the old policy and its preservation with the named
policy. It also covers refresh order, replacement by an earlier boundary,
invalidated state, two-slot and one-slot limits. Existing checkpoint, shared-byte
accounting, atomic save/restore and MTP tests remain required. Physical validation
and exact test counts are recorded in the dated parity entry and fixture.

Router affinity is separate: WorkerRouter PR #9 extends default user,
conversation and system-prefix hint lifetimes to 24 hours. Existing explicit
`ROUTER_USER_AFFINITY_TTL` / `ROUTER_PROMPT_AFFINITY_TTL` values must be updated
to `86400` in that deployment's environment to adopt the defaults. Neither repo
changes WorkConductor or omits volatile runtime context to manufacture a hit.
