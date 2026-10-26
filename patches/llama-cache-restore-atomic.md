# Atomic prompt-cache restoration

Apply `llama-cache-restore-atomic.patch` after the managed conversation-cache
patch on llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab`.

The original loader released target bytes before it knew whether draft restore
would succeed. A failed draft left a selectable entry with an empty target;
the next target call could return zero and pass the `restored == size` check.
The patch rejects a missing target or required context, retains both payloads
until every restore succeeds, and evicts the selected entry on a restore
failure. Prompt/checkpoint ownership moves only after complete success.

This is atomic cache ownership and metadata commit. A backend can already have
mutated the target before a later draft failure. The existing caller must still
clear both active contexts when `load()` returns false before recomputing. Any
additional speculative CPU-state restoration must finish before the final
prompt move and entry removal as well.

`test_cache_restore_atomic` compiles the managed engine's actual
`tools/server/server-task.cpp` into its executable. Only the external
`llama_state_seq_set_data_ext` boundary is replaced with a fault injector. The
14 scenarios cover target/draft success, zero/partial/oversized return values,
empty targets, missing required contexts, survivor preservation, valid retry,
protected eviction and a miss without a backend call. During both backend
calls the test checks that neither cached payload nor its accounted budget has
been consumed. The same test built with the pre-fix production translation
unit fails at that ownership check.

Build the target with `tests/test_cache_restore_atomic.cpp` and the actual
`${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/server/server-task.cpp`, using the same
engine include directories and server/common/mtmd/llama/ggml libraries as
`test_conversation_cache`. It requires no Python checkout, model or GPU.
Register `cache_restore_atomic` as a mandatory package test in all four image
gates and the native package target. Evidence is in
`tests/fixtures/cache_restore_atomic_20261026_validation.json`.
