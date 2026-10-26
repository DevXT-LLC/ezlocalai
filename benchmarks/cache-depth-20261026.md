# Conversation cache and router admission measurements - 2026-10-26

Status: partial qualification report. The completed results below cover one RTX
4090, the four-patch cache/MTP candidate, an unchanged-engine router comparison,
baseline cache-capacity pressure, and an experimental fifth-patch checkpoint
comparison. The fifth patch changes sampled answers and remains unqualified for
quality. Other GPUs, final state roundtrips, swap, final images and any additional
live RAM settings remain explicitly pending. The earlier
[prefill tuning report](optimization-20261026.md)
is historical evidence and is not replaced by these numbers.

## Changes and fixed controls

The candidate adds atomic target/draft snapshot save and restore, bounded reuse
of initialized host snapshot buffers, shared immutable checkpoint payloads, and
position-aware MTP CPU carryover. A separate worker change publishes capacity
after actual request/admission release rather than waiting only for the next
periodic heartbeat.

The engine experiments use the deployed 4090 model and projector, with the same
230000 context, logical/physical batch 4096/1024, q4_0 K/V, one slot, MTP maximum
3 and probability threshold 0.1. The normal cache ceiling is 7680 MiB. Weights,
KV precision, presets, GPU clocks and power settings are not tuning variables
in this comparison. Direct-engine requests use greedy sampling, seed 42 and
thinking disabled; public-worker model presets are therefore not a confounder.

Upstream remains llama.cpp `620d2225`; the low-level `libllama` artifact is the
same `4c32861b...cc9f` in both engines. Common/server consumers are rebuilt
together because checkpoint field layout changed. The measured common/server
artifact pairs are:

| Engine | Server implementation SHA-256 prefix | Common library SHA-256 prefix |
| --- | --- | --- |
| Matched baseline, conversation patch only | `f95cf664d7f1` | `3d87cd42de02` |
| Four-patch candidate | `39c397e48e2b` | `c9fd858250b3` |

The full model SHA-256 is
`8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`.
The 5090's deployed weights differ despite the identical filename; future
cross-card results must retain that distinction or use verified identical
files. Current results compare each configuration on the same 4090.

## Resident conversation restore

`bench_cache_workload.py` alternates three approximately 24.6K-token
conversations sharing a long reference prefix. It records seed, identical
replay, append and shorter-prefix rewind requests, with three samples per case
and 256 generated tokens each. Append requests include their preceding model
answer; those answers and the resulting request hashes are identical across
the measured variants. The complete 12 request and answer hashes match in all
three variants.

The storage switch permits a comparison within the same fixed-MTP build:
`EZLOCALAI_LLAMA_CACHE_STORAGE=0` uses deep checkpoint copies and releases
consumed snapshot buffers; `1` shares immutable bytes and reuses one bounded
buffer pair. Capacities, including spare buffers and CPU speculative payloads,
remain charged against the same byte ceiling.

| Case | Baseline TTFT | Candidate, storage off | Candidate, storage on |
| --- | ---: | ---: | ---: |
| Seed | 8.632 s | 8.618 s | 8.547 s |
| Identical replay | 454.5 ms | 456.2 ms | 252.2 ms |
| Append | 685.1 ms | 684.9 ms | 516.9 ms |
| Shorter-prefix rewind | 8.296 s | 8.262 s | 8.071 s |

The same-build storage comparison reduces replay TTFT by **44.7%** and append
TTFT by **24.5%**. These are first-output latency reductions, not GPU token-rate
gains. Full 256-token turns improve less because generation still occupies most
of each request:

| Case | Full turn, storage off -> on | Native decode, storage off -> on |
| --- | ---: | ---: |
| Replay | 3.239 -> 3.098 s | 91.49 -> 91.51 tok/s |
| Append | 2.590 -> 2.417 s | 133.01 -> 132.91 tok/s |

Replay evaluates four tokens and reuses 24572 in every variant. Append evaluates
88 and reuses 24831. Shorter-prefix rewinds still evaluate 23252 tokens, reusing
only 74 template/system tokens: this workload has no useful earlier recurrent
checkpoint. The sampled identity guards and replay outputs agree at these
boundaries; these four-patch measurements do not show that arbitrary rewinds
became cache hits.

MTP-only replay acceptance changes from 477/856 accepted/proposed draft tokens
in the baseline to 479/850 in the candidate. That small change is not evidence
of a large decode speedup. Seed native prefill remains approximately 2930
tok/s in this workload. Three samples per case describe this run rather than a
variance estimate or a guarantee across prompts.

These requests test latency and branch isolation, not comprehensive instruction
following. All nine append answers in this comparison repeat their preceding
seed answer byte for byte, despite a new request for deletion, resizing and
concurrent-access examples. This happens in the matched baseline as well as both
candidate storage modes. Their input hashes agree, but continuation semantics
are not qualified by these results.

## Experimental shorter-prefix checkpoint coverage

A later, separate comparison uses identical five-patch engine artifacts and
switches only `EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE=0/1`, with storage enabled in
both. Its server implementation is `f7a59fd02dbf...`; the common library,
low-level engine, weights and other parameters remain as above. The added
checkpoint splits prefill to retain an earlier valid recurrent boundary for
text prompts, including text requests on an engine with a loaded projector.

| Case | Coverage off TTFT | Coverage on TTFT |
| --- | ---: | ---: |
| Seed | 8.497 s | 8.569 s |
| Identical replay | 254.2 ms | 251.3 ms |
| Append | 517.2 ms | 518.9 ms |
| Shorter-prefix rewind | 8.053 s | 621.2 ms |

All three rewinds change from 75 cached / 23252 evaluated tokens to 22525 cached
/ 802 evaluated. Median full rewind time changes from 10.786 to 3.298 seconds.
This is a measured cache-coverage benefit for these shortened requests; it is
not a faster matrix kernel or a higher native prefill token rate. The seed
median is about 0.85% slower with coverage enabled, and replay/append changes
are small in this three-sample run.

The output controls differ from the four-patch comparison: nine of twelve
request hashes match across modes, while **none of twelve answer hashes
match**. The three differing append inputs contain their mode's preceding model
answer. Added checkpoint boundaries change physical prefill partitioning even
though model precision and requested batch sizes are unchanged. Greedy sampling
does not establish byte-identical answers across these partitions.

All 24 outputs preserve the correct first-line branch marker and code with no
other branch identity; all six within-mode replay answers exactly match their
seed answer. Those guards do not establish factual or instruction-following
quality. Manual review found that coverage-on branch 0 incorrectly associates
Python dictionaries with separate chaining, a claim absent from the matching
coverage-off branch 0 output. Coverage-off branch 1 already makes the same error;
CPython instead probes table entries on collisions, as its
[dictionary implementation explains](https://github.com/python/cpython/blob/main/Objects/dictobject.c).
The errors shift between samples, so neither universal quality equivalence nor
a general quality regression follows from this small set.

All six append answers again repeat their mode's seed answer rather than
addressing the new continuation topics. Every output stops at the 256-token
cap, while the prompt asks for at least 800 words, a Python example and resource
management discussion. This workload cannot qualify those broader requirements.
Fifth-patch quality acceptance, a concise latest-turn instruction test and any
decision to enable coverage by default remain pending.

## Router admission with a real GPU worker

A separate experiment uses an actual private Rust WorkerRouter and full native
worker on the 4090. Baseline and candidate workers use the **same combined
engine artifacts and inference parameters**, isolating worker state publication.
The worker alias `perf-worker` avoids normal model sampling-profile overrides.
There are three paired trials, with a ten-second periodic heartbeat and 100 ms
state polling. Selection time comes from the router's streamed `worker_queue`
timing event; polling only corroborates worker occupancy.

The frozen worker binaries are `ea4deab9b3ae...` (the earlier PR build) and
`246b65420651...` (capacity-publication candidate). Those hashes, rather than a
registration version inherited from the image environment, identify the code
under comparison.

A cold request evaluates 68088 tokens and generates the answer `Python`; an
identical follow-up is queued while it runs. Every repeat evaluates four tokens
and reuses 68084. All twelve answers match, all three paired request hashes
match, and cold generation remains approximately 27 seconds in both variants.

| Aligned interval after first request completes | Baseline median | Candidate median |
| --- | ---: | ---: |
| Completion to follow-up worker selection | 2.758 s | 9.7 ms |
| Selection to follow-up first output | 166.8 ms | 167.2 ms |
| Completion to follow-up first output | 2.925 s | 177.0 ms |

The 2.748-second reduction removes stale reported-busy admission time. The
post-selection cached response path remains approximately 167 ms. This is not a GPU throughput
improvement, and the default heartbeat does not impose ten seconds of delay on
every request. The old delay depends on the heartbeat phase; these roughly
27-second generations leave about 2.76 seconds of stale admission wait.

The follow-up starts at the first observed busy heartbeat, approximately 9.8
seconds into baseline generation and 0.1 seconds into candidate generation.
Consequently, whole follow-up TTFT from its own arrival is not directly
comparable. The table deliberately aligns both variants at actual completion
of the first HTTP request. Stale reporting is counted only after that
completion, not while the first request is still legitimately running.

The physical measurement covers direct routed SSE, one model, one GPU and three
pairs. Separate controlled fixtures cover direct/tunnel and buffered/streaming
paths; their timings are not additional physical GPU evidence.

## Byte-budget pressure before code changes

The baseline engine was also tested with eight independent approximately
24.6K-token conversations and 128 generated tokens. Changing only its host
cache ceiling from 7680 to 12288 MiB changes retention:

| Replay configuration | Cache outcome | Median replay TTFT | Full replay turn | Native decode |
| --- | --- | ---: | ---: | ---: |
| 7680 MiB | 8/8 full reevaluations | 8.726 s | 10.162 s | 88.63 tok/s |
| 12288 MiB | 8/8 tail reuses | 499.4 ms | 1.930 s | 88.74 tok/s |

All sixteen request and answer hashes match between the two cache ceilings.
This is a capacity effect in the baseline, not a code or token-throughput
speedup. It establishes that insufficient cache space can dominate latency
even when a single retained conversation restores correctly. It does not
guarantee that every set of eight histories fits in 12 GiB; state size varies
with context, checkpoints, model and speculative state.

These standalone tests did not force operating-system memory pressure or
co-resident media workloads. Recorded swap counters and OOM counters were zero.
That does not measure swap-in latency or qualify a larger production RAM
fraction. Cgroup memory peaks include whatever page-cache charges the kernel
assigns to the container and are not a complete host-memory budget.

## Validation, provenance and limits

The completed four-patch development build passed 68 package checks with the
Python checkout absent, nine focused checks, and two real CPU cache/Rust-router
integrations. It is an incremental CUDA development build, not a complete new
production image. No physical GPU was present in that build container; the
separate 4090 experiments supply the hardware evidence above. Later call-site
coverage changes need their own updated test/artifact record.

All completed experiments restored the original production container, image,
environment, mounts and critical engine parameters. Their owned drains were
released, with no observed OOM kills. The pressure test's 12 GiB override was
isolated, not a persistent production setting.

Raw proof remains under ignored
`build/validation/llm-cache-depth-20261026/`, principally
`DevXT4090-baseline-depth-f9977f92`,
`DevXT4090-combined-resident-b269671d` and
`DevXT4090-router-gpu-54124ad3a6`, plus the experimental
`DevXT4090-coverage-fixed-resident-035908e1`. Exact artifact, harness, input and result
fingerprints are summarized in
[`cache_depth_20261026_validation.json`](../tests/fixtures/cache_depth_20261026_validation.json).

## Pending qualification

- **3090V and 5090:** equivalent cache/MTP and router comparisons, preserving
  their existing model files and selected batch sizes.
- **Coverage/checkpoint changes:** final source/build qualification and quality
  acceptance before enabling the experimental fifth patch. Its measured rewind
  gain above accompanies changed answers and does not establish no quality loss.
- **State roundtrips:** final text/vision slot-save, reload, continuation and
  legacy/corrupt-state evidence for the final patch stack.
- **Swap:** a controlled nonzero swap-in workload and host/container pressure
  measurements. Zero swap counters are not swap validation.
- **Live RAM fraction:** effective worker arguments, available-RAM reserves,
  co-resident-model pressure, routed checks, backups and rollback before any
  additional persistent cache setting is reported as applied.
- **Final images and deployment:** fresh package results and artifacts for the
  final source, then normal image automation and routed version/capacity checks.

These pending items prevent this partial report from being read as final fleet
qualification or universal quality equivalence. The sampled output agreement
and unchanged arithmetic do not replace broader model-quality evaluation.
