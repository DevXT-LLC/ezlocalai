# CUDA throughput and latency measurements — 2026-10-26

This run starts after the completed rollout of worker `46d9ba8` and llama.cpp
`620d2225` (v0.7.0), including the existing conversation-cache patch. It measures
the deployed model on DevXT3090V, DevXT4090 and DevXT5090, then tests changes
against each machine's own baseline. No model weights, KV precision, context
limit, GPU clocks or power limits were changed for these experiments.

The code candidates in this branch have only run in isolated test containers.
They have not replaced the production images. Each machine was drained through
WorkerRouter while its original container was stopped; the same container was
then restarted with its original image, environment and mounts. The router
maintenance leases were released after restoration.

## Baseline and provenance

All workers use logical batch 4096, one slot, `q4_0` K/V cache, 32 context
checkpoints and checkpoint minimum step 8192. The model filename is
`Qwen3.8-27B-UD-Q3_K_XL.gguf`; the model API name is
`unsloth/Qwen3.8-27B-GGUF`.

| Worker | Actual GPU | Context budget | Physical batch | MTP maximum | Host prompt-cache ceiling | Engine CPU threads |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DevXT3090V | RTX 3090 Ti, 24 GB | 230000 | 512 | 2 | 7680 MiB | 6 |
| DevXT4090 | RTX 4090, 24 GB | 230000 | 1024 | 3 | 7680 MiB | 6 |
| DevXT5090 | RTX 5090, 32 GB | 262144 | 1024 | 3 | 32000 MiB | 4 |

The 5090 has a different actual GGUF despite its identical filename. Its full
SHA-256 is `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2`;
the 4090 and 3090V use
`8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`.
All 866 tensor names and shapes match, but 504 payloads differ, including 318
with different quantization types. These are mixed quantizations: their coarse
GGUF file-type labels do not describe every tensor. Cross-GPU numbers below
describe the deployed configurations and cannot isolate hardware performance.
The complete model and CPU audit is in
[`deployed_models_20261026_validation.json`](../tests/fixtures/deployed_models_20261026_validation.json).

The 3090V and 4090 production images are respectively `3e09d9905d59` and
`1197b0b18d41`; the 5090 also uses `1197b0b18d41`. Driver versions are
615.71.09, 580.178.04 and 590.48.01 respectively. Baseline telemetry recorded
450 W power limits on the 3090V/4090 and 575 W on the 5090. Loaded samples show
P2/10251 MHz memory on the 3090V/4090 and P1/13801 MHz on the 5090. Software
power limiting was observed; hardware thermal slowdown was not observed.

## Measurement method

`bench_progression.py` streams OpenAI-compatible chat completions, records TTFT
at the first nonempty content or reasoning delta, and uses native engine timing
fields for token rates. Stream chunk counts are never treated as token counts.
Warmup requests are excluded. Requests use greedy sampling, seed 42 and thinking
disabled. The long-prompt fixtures retain the historical message hashes:
`be2a9a92…9163de` at 68069 evaluated tokens and `7e2c9f31…068de` at 150026.
Each cold depth starts a fresh engine and rejects an unexpected cache hit.

The initial cold numbers below are **screening n=1 per depth**, not repeated
estimates of variance. Short suites contain seven latency requests, three
256-token decode requests and three independent conversations with a common
~24K-token prefix. Each returning conversation must recover its own branch
answer. Worker A/B suites use five conversation rounds, giving 12 cache-return
samples and 25 total comparisons per variant.

Normal model presets can override the public request's sampling fields. Direct
engine tests avoid that ambiguity. Full-worker A/B tests use a synthetic model
alias with identical explicit engine parameters and greedy sampling, and bypass
the router/fallback/media paths to isolate the HTTP worker changes.

## Current baseline and historical progression

| Worker | 68069-token cold prefill, tok/s | 150026-token cold prefill, tok/s | Short TTFT, ms | Decode, tok/s | Returning-conversation TTFT, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| DevXT3090V | 1260.6 | 945.8 | 149.88 | 86.60 | 759.63 |
| DevXT4090 | 2525.0 | 1976.3 | 99.43 | 113.06 | 452.35 |
| DevXT5090 | 2890.5 | 1774.8 | 71.95 | 157.66 | 313.44 |

The October 25 medians were 1246/939 tok/s on the 3090V and 2529/1970 on the
4090. The fresh exact-prompt samples are consistent with those results. The
October 18 4090 measurements were approximately 2350/1812 tok/s; current
prefill is about 7–9% higher. That comparison includes an engine change from
`aa94f20`/b11549 to v0.7.0, so it does not establish the GPU unlock's contribution.
No comparable saved pre-unlock 3090V result was found, and there is no equivalent
historical 5090 baseline. See the
[October 25 report](optimization-20261025.md) for those measurements.

The existing conversation-cache retention fix remains effective: returning
branches evaluate four tokens rather than re-evaluating their ~24K-token
histories. The separate October 25 3090J experiment measured 19.04 s → 0.916 s
for that change. It is a cache-retention improvement, separate from cold
prefill throughput and from the transfer work tested here.

## Prefill and decode screens

| Worker | Candidate | 150K cold prefill, tok/s | Baseline, tok/s | Sampled short outputs identical |
| --- | --- | ---: | ---: | ---: |
| DevXT3090V | Physical batch 1024, from 512 | 1018.6 | 945.8 | 16/16 |
| DevXT5090 | Physical batch 2048, from 1024 | 1880.7 | 1774.8 | 16/16 |

These single cold trials suggest about 7.7% and 6.0% improvements respectively.
The 5090 larger-batch screen also reused 21997 prefix tokens when conversations
B and C first arrived, evaluating 2052 tokens each; at batch 1024 those requests
evaluated 24033 tokens. Their TTFT fell from about 6.77 s to 0.89 s, with the
same branch answers. The first conversation stayed cold (~6.66 s), and returning
branches still evaluated four tokens. This is an observed checkpoint/cache
interaction in this workload, not a claim that every new conversation gains
that speedup.

The source suggests why this workload benefits: the pinned upstream server
places prompt-end checkpoints at offsets `4 + n_ubatch` and `4`, before decoding
the current batch. With this test's ~2K-token branch suffix, physical batch 2048
places an earlier checkpoint inside the shared prefix; the batch-1024 end
checkpoint lies after the branches diverge. That is a source-based explanation
consistent with the observed 21997 reused tokens, rather than a general cold
prefill speedup or a new checkpoint implementation.

MTP maximum 4/5 on the 4090 and 5090, and 4 on the 3090V, changed all three
freeform decode outputs (13/16 total outputs remained identical). Their modest
or negative throughput changes do not justify promoting them under the current
output-preservation criterion. MTP maximum 3 on the 3090V preserved all 16
outputs but gave no useful decode gain. The deployed MTP settings are retained.

The 5090 defaults to four inference threads on its 24-core hybrid CPU. Screens
at one, four and eight threads preserved all outputs and showed no useful gain;
no CPU-thread override is selected. CUDA MMQ/cuBLAS force switches are
compile-time options, not effective environment tuning knobs at this pin.

## State-transfer screen before pinned staging

The first state-transfer candidate grouped CUDA transfers without pinning the
pageable snapshot buffers. It used one candidate library with batching disabled
and enabled, keeping the engine arithmetic unchanged. All 16 requests and
outputs matched on every card. Each card's final ~1.17 GB serialized GPU slot
state was also byte-identical between the two modes.

| Worker | Synchronous cache-return TTFT, ms | Grouped pageable transfers, ms | Snapshot SHA-256 |
| --- | ---: | ---: | --- |
| DevXT3090V | 750.16 | 738.10 | `b8473e7af5715516c6ca68d6b238a07eab43de84788c0e3cb80f4e98166e509c` |
| DevXT4090 | 499.02 | 445.35 | `b1da7af69f42d4061340e3ead0f51e81adeb1a511ff3420924287979971b9e1a` |
| DevXT5090 | 313.23 | 311.80 | `ebbf5506e4e3ce915db00456e67d9d5df4027e1d5eb89252c90a67cc7a526b5b` |

This is only three return samples per mode. The 4090 stock baseline was already
452.35 ms, so its apparent synchronous-to-async difference is insufficient to
establish a repeatable improvement. This screen motivated a subsequent bounded
pinned-staging experiment.

## Rejected pinned-staging and CUDA graph screens

The follow-up state-transfer candidate used reusable pinned staging capped at
64 MiB per context/CUDA backend; the saved conversation buffers remained
pageable. A three-way screen compared the same library with state batching
disabled, pinned staging enabled, then pinned staging plus
`GGML_CUDA_GRAPH_OPT=1`. Each mode had seven latency requests, five decode
requests and 15 conversation requests, including 12 returning turns.

| Worker | Cache-return TTFT: synchronous → pinned → pinned + graph, ms | Decode: synchronous → pinned → pinned + graph, tok/s |
| --- | ---: | ---: |
| DevXT3090V | 746.59 → 792.50 → 798.04 | 87.21 → 86.19 → 85.71 |
| DevXT4090 | 453.71 → 503.90 → 503.96 | 113.10 → 113.05 → 112.93 |
| DevXT5090 | 312.02 → 359.97 → 376.53 | 157.05 → 155.10 → 155.27 |

All 27 request and output hashes matched between modes on each card. Each
card's final serialized GPU state was byte-identical across all three modes;
the complete hashes are in the compact evidence fixture. Correctness agreement
did not produce a performance benefit: pinned staging increased the measured
cache-return latency on every card, and graph optimization added no useful
decode gain in this configuration. The graph result is specifically a screen
on top of pinned staging, rather than an independent graph-only comparison.

Both state-transfer implementations were removed from the final change set,
and the graph environment switch is not selected. These rejected experiments
remain documented so their correctness evidence is not mistaken for a speedup.

## Full-worker transport and completion A/B

The new worker coalesces each HTTP chunk into one write, retries interrupted or
partial sends, requests TCP_NODELAY on streaming sockets, and wakes the stream
consumer promptly when the producer completes. The engine library is unchanged
in this A/B, so these numbers isolate the two worker changes.

| Worker | Short TTFT before → after, ms | Decode before → after, tok/s | Cache-return TTFT before → after, ms | Output matches |
| --- | ---: | ---: | ---: | ---: |
| DevXT3090V | 151.06 → 152.04 | 86.54 → 85.19 | 744.15 → 740.98 | 25/25 |
| DevXT4090 | 98.55 → 99.60 | 108.21 → 108.16 | 451.98 → 453.83 | 25/25 |
| DevXT5090 | 77.33 → 74.49 | 155.68 → 156.75 | 318.96 → 318.97 | 25/25 |

These live model runs show no consistent material throughput or TTFT change.
The controlled [socket experiment](stream-socket-20261026.md) does reproduce a
40.777 ms delayed-ACK stall falling to 0.0077 ms. A separate controlled
producer-completion experiment reduces stream exhaustion/admission-release delay
from 90.031 ms to 0.0526 ms. Those experiments isolate specific waiting
conditions; their reductions must not be added to, or substituted for, measured
model TTFT. The live A/B establishes output preservation and absence of an
obvious regression on the sampled workloads.

## Final qualification and applied environment settings

**Pending:** repeated positive prefill trials, near-limit context and vision
checks, and any resulting persistent environment changes. The benchmark
coordinator will replace this section with measured results and an explicit
old/new settings record.

## Follow-up cache work

A separate read-only audit of the pinned upstream MTP implementation found that
its CPU `pending_h`, `verify_h` and row bookkeeping are outside the target/draft
KV snapshots. The generic speculative-state hooks currently capture no MTP
payload, so restoring conversation A after B can retain B's predecessor row in
the draft path. This establishes a state-consistency gap in source; it does not
establish a measured acceptance-rate or answer-quality loss.

A production fix needs a position-aware resume design, including identical
prompt last-token replay, shorter shared prefixes, checkpoint rewind, full
reset and child-slot remapping. Copying only the saved final hidden row is
insufficient when replay begins at an earlier position. Tests should compare
uninterrupted A with A/B/restore-A at the same position, then measure MTP
acceptance and target outputs on a real model. Speculative payloads also need
cache-budget accounting.

The audit also found that a successful target restore followed by a failed
draft restore can leave a selectable entry whose main bytes have already been
cleared. A separate fix should preserve payloads until the whole restore
succeeds, reject an empty main payload and evict a failed entry. Failure-injection
tests should cover target success/draft failure and valid retry recovery. Both
upstream gaps remain follow-up work; no partial MTP fix ships in this change.

## Evidence and limits

Compact baseline, configuration, sample counts, file hashes, A/B comparisons and
restoration evidence are in
[`llm_throughput_20261026_validation.json`](../tests/fixtures/llm_throughput_20261026_validation.json).
Bulk JSON, engine logs and two-second telemetry remain under ignored
`build/validation/llm-throughput-20261026/`. The frozen benchmark driver SHA-256 is
`e54b07c9474a8145da088a2b93b68781ae3bd016ea5b8c6cfcc6feb9535909d6`;
its measurement/request behavior matches the checked-in driver, whose later
changes add provenance fields and explicit UTF-8 reads.

Exact sampled output and snapshot agreement does not prove quality equivalence
for all prompts. Sustained RAM/swap pressure, multi-GPU hardware, other hardware
platforms and video generation are outside these performance measurements.
