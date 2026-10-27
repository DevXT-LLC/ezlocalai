The checkpoint entry-window candidate reduces reprocessing for a shortened
conversation whose usable prefix precedes every saved near-end state. On the
controlled 4090 workload, median rewind first output falls **8.522 → 1.785 s**
(79.1%) while every measured output hash agrees. Cold prefill and decode
throughput remain approximately unchanged. Finite-answer and cache-pressure
qualification is running; this proposal remains in draft pending that result.

## Behavior and controls

The public-source seed has 25,758 tokens and the rewind has 24,508. The previous
near-end states follow the shorter prompt, leaving only 77 reusable tokens.
The candidate captures an existing full logical batch's starting state too.
The rewind now reuses 20,557 tokens and evaluates 3,951 rather than 24,431.
Two subsequent conversations sharing the source prefix also reuse 20,557 tokens;
the first seed stays cold. Existing replay and append reuse is preserved.

The additional checkpoint introduces no split. Original target/draft token
groups, weights, precision, sampling, context and cache format are preserved.
Existing eligibility guards remain; the optional entry state also requires at
least five checkpoint slots. Configured count limits, bounded eviction and the
hard cache ceiling are unchanged. One extra partial state consumes RAM (about
150 MiB in earlier Qwen measurements), so conversation retention needs its own
pressure check. See [the scheduling rationale](../patches/llama-checkpoint-entry-window.md).

All four fresh-engine phases use worker revision `720ee94`, production image
`sha256:f78bbd2139979beb58859f79c4c0e5c233919f43908be720eb5e25d266a108f9`,
llama.cpp pin `620d2225cc436ab0b001e3200e590e3b286ef9ab`, and production
Qwen weights `8c2a45ff…` with projector `83ee4f4f…`. Actual controls remain
230,000 context, logical/physical batch 4096/1024, q4_0 K/V, one slot,
MTP maximum 3, 32 checkpoints and a 12,288 MiB cache ceiling.

Only `libllama-server-impl.so` changes: baseline `56435bfa…`, candidate
`2fbc02b3…`. Every phase verifies its loaded library and the unchanged original
common (`c9fd8582…`) and CUDA (`5fe5f7b9…`) libraries. The rejected speculative
host-work and cuBLAS variants are excluded. Original environment, full engine
argv, resource limits, model/projector hashes and immutable image are frozen.
The private loopback host/port is the only launch argument adjustment.

## Completed ABBA screen

A1/B1/B2/A2 each execute a cold 49,987-token finite-answer prompt, three short
first-output requests, three 256-token decode requests and twelve conversation
seed/replay/append/rewind requests. All **76 measured output hashes match**;
request/message hashes, generated counts and finishes agree. Both within-build
signature pairs repeat exactly. Only the expected five cache rows per phase
change evaluated/reused counts across builds. Native counters sum to each actual
prepared input; inference has no retry or context downgrade.

The table pools six samples per build/case. Seed medians include one cold seed
and two branches sharing its prefix in each phase.

| Case | Baseline first output | Candidate first output | Baseline evaluated / reused | Candidate evaluated / reused |
| --- | ---: | ---: | --- | --- |
| Seed (mixed cold/shared prefix) | 8.997 s | 2.319 s | 25,758/0 or 25,681/77 | 25,758/0 or 5,201/20,557 |
| Replay | 0.258 s | 0.258 s | 4/25,754 | 4/25,754 |
| Append | 0.533 s | 0.531 s | 91/26,013 | 91/26,013 |
| Rewind | 8.522 s | 1.785 s | 24,431/77 | 3,951/20,557 |

These are first nonempty content/reasoning times at an isolated engine on the
GPU host, with native token timings. They exclude router/worker transport and
are not production Wi-Fi measurements. The decode rate depends on prompt and
context; these requests are unsuitable as a full-day GPU average.

## Validation and remaining qualification

The actual scheduler/allocator guard passes **364,385 checks**, including all
4,096 logical-batch residues at three starting positions/four recurrent-tail
sizes and 672 representative actual target/draft splitter cases. It preserves
the earlier qualified prompt's coverage. A deliberate removal of the entry
checkpoint fails the new regression. Independent review found no runtime blocker.

All **seven focused engine checks** pass in 10.62 seconds, and all **72 exact
CUDA package checks** pass in 335.40 seconds with the Python reference absent.
The package builder has no GPU; physical inference evidence is the separate
4090 screen. Four platform recipes include the sixth managed patch, with
idempotence, five-patch upgrade, metadata and complete-removal checks. Fresh
images, other GPUs/platforms, maximum context and vision were not requalified.

The separate qualification runs finite latest-turn controls at 64/128 token
budgets, then eight independent named conversations with two return rounds under
the same 12 GiB ceiling. Its acceptance requires exact finite answers and no
observed OOM or retention regression. Until complete, RAM pressure and latest-turn
qualification remain pending; the screen is a bounded output check rather than
a general model-quality claim.

The screen restored the exact original container/image/environment/mounts/full
argv, confirmed health and two fresh stable router heartbeats, then released its
owned drain. Rollback tag: `ezlocalai:rollback-spec-overhead-4090-7a6e959d4a`.
Qualification uses its own lease and will verify restoration separately. No
permanent `.env`, precision, context, GPU power/clock or model changes are made.
Evidence: [CUDA validation](../tests/fixtures/checkpoint_entry_cuda_20261026_validation.json)
and [model-free validation](../tests/fixtures/checkpoint_entry_window_20261026_validation.json).

This round also rejects two alternatives: the
[speculative host-work screen](speculative-overhead-20261026.md) shows no useful
throughput gain, and the [cuBLAS prefill screen](cublas-prefill-screen-20261026.md)
is 15.93% slower on its fixed 4090 workload. Neither rejected implementation is
part of this runtime proposal. WorkerRouter PR #8 separately clarifies recent
cold/cached rates, context depth and optional native draft acceptance.
