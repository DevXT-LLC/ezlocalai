# Bounded host snapshot storage

Apply `llama-cache-storage.patch` after the conversation-cache patch and the
combined atomic restore/save patch, against llama.cpp `620d2225`. It changes
storage ownership and allocation, with no model arithmetic or precision change.

Target/draft checkpoint bytes use a copy-on-write blob. Checkpoint clones share
immutable payloads; a resize or mutable data access detaches first. Clear drops
only that owner's reference. Disk appendix helpers retain the same length and
byte format. CPU speculative state remains a vector for the speculative API.

The prompt cache retains at most one consumed main/draft snapshot buffer pair
after a complete successful restore. A later save can reuse initialized storage.
Cache charges include vector capacities, including the idle spare. An oversized
spare is discarded before its excess capacity can evict a useful state. Lowered
byte limits discard spare storage first. Shared checkpoint bytes are charged
conservatively per cached owner. Fresh allocation remains the fallback when
capacity is insufficient. Nothing is pinned, so host snapshots remain pageable.

`EZLOCALAI_LLAMA_CACHE_STORAGE=0` disables checkpoint sharing and snapshot reuse
for a controlled comparison in the same build. Default or `1` enables them. This
switch does not undo atomic save/restore validation or the separate MTP fix.

The slot must call the atomic `server_prompt_cache::save` API before this storage
change ships: partial writes into reused bytes must never remain selectable.
The combined MTP integration owns that slot wrapper and CPU-payload accounting.

Validation compiles the actual patched `common.cpp` and `server-task.cpp` with
state API fault injection. Ten scenarios cover copy ownership, production
checkpoint update/load, no-large-allocation clone/reuse, capacity accounting,
protected eviction, quota shrink, fresh allocation/restore/save failures and the
opt-out. Baseline controls reject both legacy deep copying and absent pooling.
The disk helper translation unit also passes a syntax check. CPU microbenchmarks
are isolated allocation/copy costs, not GPU or end-to-end latency measurements.

Integration must rebuild all common/server consumers because checkpoint field
layout changes. Add `test_cache_storage` to the native package target and all
four Docker package regexes; link it against the matching rebuilt common/server
libraries and compile actual `tools/server/server-task.cpp` as in the atomic
cache test. Do not mix new headers with an old common DSO. GPU snapshot, answer,
latency and sustained pressure validation remains the parent integration task.
