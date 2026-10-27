The sixth managed patch adds an optional checkpoint when an unchanged full
logical batch first enters the existing near-end coverage window. It applies
after the five cache/state/MTP/checkpoint patches at llama.cpp
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. It does not include the separate
sampler-clone/hidden-row host-overhead experiment.

The prior aligned policy has a token-phase gap. In the new public-code workload,
the seed has 25,758 tokens and the shortened prompt has 24,508. Starting from the
observed early user boundary at 77, the prior planner's near-end states are at
24,653, 24,730 and 25,754. They all follow the shortened prompt, so the worker
can reuse only the early state at 77. These checkpoint positions are derived
from source scheduling, not measured trace events. The observed native counters
show 77 reused and 24,431 evaluated rewind tokens.

When a full logical batch moves from before the window to inside it, the
candidate also saves the state at that batch's existing start. Here it is
20,557. The batch remains 4,096 tokens, and every target/draft physical group
is unchanged. The original two tail checkpoints and the prior aligned
checkpoint remain. The earlier qualified 24,577-token seed still has its exact
22,603, 23,549 and 24,573 near-end states.

The added checkpoint is eligible only when all the existing named-text,
MTP-only, single-slot/head, uninterrupted-user, matching-1024-batch and recurrent
rollback guards pass. It additionally needs at least five configured checkpoint
slots. In a fresh single-last-user prefill, early user state, both original tail
states, the prior aligned state and this entry state can then coexist. Existing
bounded eviction still applies to older states from other turns. Four-slot
configurations keep the prior planner policy. The default checkpoint count and hard cache byte budget are unchanged.

No new split is introduced. At most one checkpoint is added beyond the prior
aligned policy, or two beyond the original upstream scheduler. Existing
checkpoint eviction, transactional state save/restore and cache accounting still
apply. One additional partial target/draft state costs RAM (about 150 MiB for
the previously measured Qwen model); a tight hard budget may retain fewer
conversations. Hardware cache-pressure qualification must decide whether that
tradeoff is acceptable. This patch does not raise `.env` memory ceilings.

The extended `llama_checkpoint_coverage` guard checks all 4,096 logical-batch
residues at three starting positions and four recurrent tail sizes. It retains
every prior checkpoint and identical logical calls, with at most one new state.
A representative 672-case grid also compares complete ordered rows from the
actual production target and draft batch allocators, including logical sizes
2048, 2049, 3072, 4097 and 8192. The scoped test timeout is 300 seconds for slower ARM/emulated package runs;
that allowance does not claim hardware qualification. The prior parser, borrowed
MTP-context, media exclusion, tail-reservation and state-boundary tests remain.
The local model-free run passes 364,385 checks in 14.42 seconds; deliberately
disabling the new entry state fails its new public-workload regression. These
checks do not establish GPU acceleration, model quality or eight-conversation
retention. Hardware and package outcomes belong in the dated final report.
