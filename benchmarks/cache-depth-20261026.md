# Conversation cache and router admission measurements - 2026-10-26

Status: completed measurements for the recorded CUDA scope. Results cover the 4090 four-patch
cache/MTP candidate and router comparison, 4090/3090V production-engine cache
capacity tests, applied RAM settings on those two workers, and baseline 4090
disk-state roundtrips. The latest release candidate also completes active
aligned-checkpoint, finite-instruction and eight-conversation pressure checks
on the 4090, plus resident/finite checks on the 3090V and 5090. Final candidate disk
roundtrips pass on 3090V, 4090 and 5090; its packaged CUDA image passes the
package gate and an isolated 5090 router canary. Earlier checkpoint results
remain historical. Near-capacity checks pass on all three, and an isolated
5090 experiment verifies retention during actual OS swapping. Broader quality and production rollout remain
outside these sampled checks. The earlier
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
3 and probability threshold 0.1. Those initial comparisons use a 7680 MiB cache
ceiling; the later live changes below raise it to 12288 MiB on 4090 and 3090V. Weights,
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
files. Each configuration comparison keeps its host's model files unchanged;
the 4090 and 3090V share the verified file above, while the 5090 retains its own.

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

## Historical unaligned checkpoint coverage

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
These older unaligned results are not release qualification. Intermediate
aligned attempts also had eligibility-guard problems, so their absence of an
effect cannot establish active checkpoint coverage. The release comparison
below verifies the actual runtime guard and checkpoint window separately.

## Active aligned checkpoints in the release candidate

The release candidate from source `448bf6b` uses server implementation
`56435bfa886d...`, with the same common library, low-level engine and model
files as above. Both freeform modes use the now-applied 12288 MiB cache ceiling
and identical parameters/artifacts; only
`EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE=0/1` changes. Runtime trace confirms an
eligible named text request with checkpoint window 2052, one active MTP head,
MTP-only drafting, and draft/target physical batches of 1024. This demonstrates
active coverage rather than inferring it from an environment flag.

| Case | Coverage off TTFT | Coverage on TTFT |
| --- | ---: | ---: |
| Seed | 8.530 s | 8.587 s |
| Identical replay | 251.9 ms | 255.9 ms |
| Append | 519.1 ms | 517.3 ms |
| Shorter-prefix rewind | 8.064 s | 590.2 ms |

All twelve freeform request hashes match across modes. All nine seed, replay
and append answer hashes also match exactly. All three rewinds change from
75 cached / 23252 evaluated tokens to 22603 cached / 724 evaluated. Median full
rewind time changes from 10.796 to 3.326 seconds, while native decode remains
93.33 versus 93.23 tok/s. The gain comes from avoiding prompt reevaluation,
with no measured decode-rate increase. Seed TTFT is about 0.68% slower in this
three-sample run; replay/append differences are small.

The three rewind answer hashes differ across modes. Two independent reviewers
read all six answers against the task and found coherent collision/chaining
explanations with no concrete new factual or identity regression in these
samples. Differences concern wording, emphasis and truncation. One enabled
answer uses an unsupported broad claim about the most common collision
strategy. Every answer is capped at 256 tokens and ends with `length`, so this
comparison still cannot qualify the requested 800-word explanation, Python
example or full resource-management discussion. Sampled semantic review does
not establish universal quality equivalence or byte-identical rewind answers.

A separate finite control directly tests the latest user instruction. Both
coverage modes return all twelve exact expected branch/turn answers and finish
with `stop`: seeds/replays use `TURN_ALPHA`, appended instructions change to
`TURN_BRAVO_r1`, and rewinds change to `TURN_CHARLIE`. All twelve paired request
and output hashes match. Enabled finite rewinds reuse 22610 tokens and evaluate
728, with median TTFT 590.5 ms. Finite-on additionally enables logging verbosity
4, so its timing comparison has that extra control difference. This test
qualifies the sampled latest-turn discriminator; it does not replace broad
freeform quality evaluation. The freeform append samples still repeat their
mode's seed answer rather than extending the requested topics, so their broader
continuation semantics remain unqualified.

The same release candidate also completes eight independent conversations for
three rounds at 12288 MiB. All sixteen replays reuse the tail, with median TTFT
252.4 ms, full turn 1.706 seconds and native decode 88.59 tok/s. All 24 request
and answer hashes match both prior production-engine pressure variants. At the
same 12288 MiB ceiling, the earlier production run has 461.8 ms TTFT and a
1.912-second turn. This is a comparison of the combined engine changes across
runs, not an isolated estimate for a single patch.

All five completed release-test phases record zero swap/OOM counters. The
pressure phase peaks at 13127340032 cgroup bytes, including active state and
reclaimable page-cache charges outside the conversation-cache ceiling. Host
memory pressure and concurrent media workloads are not forced. The original
production container, image, environment, mounts and critical arguments are
restored, and the owned drain is released. The 3090V and final disk/image checks
below add separate evidence, including final near-capacity checks.

## Resident release comparison on 3090V

The 3090V compares its matched conversation-cache baseline with the complete
release engine using defaults: storage and aligned checkpoints are enabled
without environment overrides. Both modes preserve context 230000, batches
4096/1024, q4_0 K/V, MTP maximum 2, one slot and the applied 12288 MiB ceiling.
The common/server libraries differ together; this comparison measures the
combined candidate rather than isolating checkpoint coverage alone.

| Case | Baseline TTFT | Release TTFT | Baseline -> release full turn |
| --- | ---: | ---: | ---: |
| Seed | 16.096 s | 16.070 s | 19.558 -> 19.556 s |
| Identical replay | 752.3 ms | 303.7 ms | 4.265 -> 3.836 s |
| Append | 1.183 s | 777.2 ms | 4.047 -> 3.653 s |
| Shorter-prefix rewind | 15.565 s | 918.2 ms | 19.142 -> 4.465 s |

All twelve input hashes and the first nine seed/replay/append answer hashes
match. Rewinds reuse 22603 tokens and evaluate 724 instead of reusing 75 and
evaluating 23252. Replay native decode remains 72.57 versus 72.34 tok/s; rewind
decode is 72.09 versus 71.75 tok/s. These are prompt/cache latency gains, with
no demonstrated decode throughput increase.

Two reviewers inspect the six differing rewind answers and find no concrete
new factual or identity regression in the sampled release text. Baseline
branches 0 and 1 incorrectly associate Python dictionaries with chaining;
those claims are absent in these release samples, while broad unsupported
statements about the most common strategy remain. All six stop at the
256-token cap, so this is not a complete programming-task quality evaluation.
Freeform appends still repeat seeds in both variants. The separate finite
release control returns all twelve exact latest-turn answers with `stop`.
The original production container/configuration is restored, its drain is
released, and observed swap/OOM counters remain zero.

## Resident release comparison on 5090

The 5090 preserves its distinct model file, context 262144, batches 4096/2048,
MTP maximum 3 and 32000 MiB cache. All twelve matched-baseline/release request
and answer hashes match exactly, including the three rewinds. Default release
settings are exercised without environment overrides.

| Case | Baseline TTFT | Release TTFT | Baseline -> release full turn |
| --- | ---: | ---: | ---: |
| Seed | 6.788 s | 6.734 s | 8.622 -> 8.602 s |
| Identical replay | 319.1 ms | 195.1 ms | 2.225 -> 2.146 s |
| Append | 483.3 ms | 372.4 ms | 1.858 -> 1.712 s |
| Shorter-prefix rewind | 607.7 ms | 470.8 ms | 2.466 -> 2.332 s |

The baseline already reuses 22525 tokens and evaluates 802 on these rewinds;
the release retains those exact counts. The extra aligned checkpoint has a
narrow eligibility scope and is skipped for this existing 2048-ubatch path.
These gains therefore reflect snapshot/restore handling rather than newly
avoided full reevaluation. Replay native decode is 133.83 versus 133.64 tok/s;
the measured improvements primarily reduce first-output latency.

All twelve finite release controls return the exact latest-turn answer with
`stop`. Exact agreement in the twelve freeform samples does not establish
universal model quality; the 256-token cap and repeated freeform append
behavior remain limitations of this workload. Production restoration and drain
release pass, with observed swap/OOM counters zero.

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

## Sustained cache capacity on production engines

A later qualification runs each worker's unchanged production image with no
engine overlays. Eight independent conversations run for three rounds, giving
eight cold seeds and sixteen replays per cache ceiling, with 128 generated
tokens per request. Only `--cache-ram` changes from 7680 to 12288 MiB. The 4090
uses MTP maximum 3 and the 3090V maximum 2, preserving their deployed settings;
this is not a controlled cross-card kernel benchmark.

| Worker | Replay TTFT, 7680 -> 12288 MiB | Full replay turn | Native decode |
| --- | ---: | ---: | ---: |
| 4090 | 8.726 -> 0.462 s | 10.164 -> 1.912 s | 88.62 -> 88.68 tok/s |
| 3090V | 16.451 -> 0.748 s | 18.297 -> 2.617 s | 69.00 -> 68.63 tok/s |

On each worker, all 24 paired request and answer hashes match. At 7680 MiB all
sixteen replays fully reevaluate; at 12288 MiB all sixteen reuse the tail,
evaluating four tokens and reusing 24575. No code optimization or higher decode
rate is needed for this capacity benefit. The isolated tests restore the
original containers, images, environments, mounts and engine parameters, then
release their drains.

Both runs record zero swap and OOM counters. Container memory is unbounded in
these tests, and no co-resident media load or forced host-memory pressure is
introduced. The results qualify retention for this workload, not swap latency,
arbitrary conversation counts or a general host-RAM stress limit.

## Applied RAM settings and rollback materials

After the sustained qualification, `LLM_PROMPT_CACHE_MAX_RAM_FRACTION` was
changed from `0.25` to `0.4` in the 4090 and 3090V installations. Each effective
engine command now has `--cache-ram 12288`; model, context, K/V precision,
batch sizes, MTP settings, immutable image, mounts, host configuration and
other engine arguments are preserved. The containers are intentionally
recreated for the environment change, so their IDs change.

Each worker passes pinned, fallback-disabled routed chat, streaming chat and
vision requests, plus two fresh router heartbeats with stable advertised
fields. Both final workers are healthy, show no OOM kill, and have their drains
released. The existing production images remain `1197b0b18d41...` on 4090 and
`3e09d9905d59...` on 3090V; these changes do not deploy the new engine candidate.

Rollback materials on the respective hosts are:

| Worker | Image tag | Private dotenv backup |
| --- | --- | --- |
| 4090 | `ezlocalai:rollback-env-cache-env-20261026-devxt4090-c16abb26` | `/tmp/ezlocalai-env-update-cache-env-20261026-devxt4090-c16abb26/dotenv.backup` |
| 3090V | `ezlocalai:rollback-env-cache-env-20261026-devxt3090v-c16abb26` | `/tmp/ezlocalai-env-update-cache-env-20261026-devxt3090v-c16abb26/dotenv.backup` |

The update helper creates backups with mode `0600`, in private run directories,
and retains the original inspection metadata. Successful finalization removes
the old stopped containers. The V1 helper used for these updates expects those
original IDs, which exposed a post-finalization recovery gap. The reviewed
[V2 recovery helper](operators/cache_env_recovery_20261026.py) now supports
recreation from the private saved configuration/environment and immutable
image, with 47 offline guards. It refuses unrelated current container IDs and
reports recreated IDs accurately. The source and workflow are in the PR; the
helper is not installed or run automatically. No live post-finalization fleet
rollback is claimed.

The 5090 installation is left unchanged in this work. Its isolated final-image
canary freshly verifies a 32000 MiB cache, zero cache-swap allowance, context
262144 and batches 4096/2048. The operator confirms `SWAP=0` remains unchanged
on the workers. The separate quota-controlled swap experiment below changes no
production setting.

## Disk state and benchmark-operator checks

The matched baseline 4090 engine completes normal text and vision slot-save,
state-overwrite, reload and continuation comparisons with identical request and
answer hashes. In-memory conversation caching is disabled for this test, and
the launch uses an isolated owned snapshot directory. Text saves/restores
24842 tokens in 1188510480 bytes; vision saves/restores 1091 tokens in 495318656
bytes. Both restored continuations reuse the saved prefix. These results do
not reproduce a legacy-state failure, validate corrupt/legacy payloads, or
qualify disk state for the final aligned candidate.

The final release engine subsequently passes a finite disk-state discriminator
on all three GPUs with in-memory conversation caching disabled. Text begins
with `TURN_ALPHA`, saves its prefix, executes an unrelated conversation, then
restores and correctly answers the changed instruction with `TURN_BRAVO_r1`.
Reference/restored requests and answers match, and the saved prefix is reused.
Vision also returns the expected red color before and after overwrite/reload,
with equal continuation requests, answers and reused prefix counts.

| Worker | Text saved tokens / bytes | Vision saved tokens / bytes |
| --- | ---: | ---: |
| 3090V | 24613 / 1341217444 | 1092 / 495914872 |
| 4090 | 24612 / 1341543092 | 1091 / 496056164 |
| 5090 | 24612 / 1184382360 | 1091 / 496056164 |

Each restore reports the same token and byte counts as its save. All three
owned runs restore production and release their drains. These tests qualify
normal current-format text/vision state and the sampled changed-turn behavior;
they do not establish compatibility with arbitrary legacy or corrupt files.

An earlier benchmark-operator attempt fails its private-port preflight before
preparing or stopping production. Version 2 addresses false busy detection for
`TIME_WAIT` sockets while retaining rejection of active listeners. Its 56
offline tests cover depth/state operations, provenance and real local sockets;
they are operator checks rather than additional GPU inference evidence. The
failed attempt releases its drain and verifies that production remains healthy.

## Cold requests near declared context capacity

One cold text request per card fully evaluates approximately 95% of declared
context, returns `Python`, and is followed by a small vision request returning
`Red`. Both answers stop normally. Usage and native prompt counts match every
input token, with no inference retry, prompt truncation or downward context
change.

| Worker | Declared context | Fully evaluated cold input | Fraction |
| --- | ---: | ---: | ---: |
| 4090 | 230000 | 218392 | 94.953% |
| 3090V | 230000 | 218392 | 94.953% |
| 5090 | 262144 | 248809 | 94.913% |

The first two engines allocate 230144 context positions because of flash
attention padding; this does not reduce the declared 230000-token capacity.
The harness targets 95% with a small tokenizer tolerance and retains response
headroom. It supplies a named cache key but explicitly sets `cache_prompt:false`.
These are two cold single-turn requests per card, not near-capacity cached
replays, rewinds, long generation or latest-turn tests. Native prefill rates at
these different deep contexts are not a paired speed comparison. All three
owned runs restore production and release their drains.

## Cache retention during actual OS swapping

On the 5090, seven independent conversations run for three rounds with a fixed
8192 MiB cache, 128 generated tokens, the release engine and unchanged model
parameters. The control inherits the production container's unbounded memory
configuration. The bounded variant uses 6 GiB RAM plus 8 GiB swap: Docker's
`MemorySwap` is the combined 14 GiB limit, not an extra 14 GiB of swap.

All 21 paired request and answer hashes match. All fourteen replays retain
their tails, evaluating four tokens and reusing 24580, while cgroup swap is
actually nonzero. Observed swap current reaches 5507969024 bytes and the kernel
swap peak reaches 5715623936 bytes, with zero OOM events or kills.

| Replay metric | Inherited memory limits | 6 GiB RAM + 8 GiB swap |
| --- | ---: | ---: |
| Median first-output time | 197.0 ms | 2.263 s |
| Median complete 128-token turn | 1.183 s | 3.240 s |

This shows output-equivalent retention under the measured OS RAM/swap pressure,
with substantially slower cached responses. Telemetry does not identify which
pages or buffers swapped and does not isolate pure swap-in time. The existing
backend is an approximately 96 GiB NVMe partition, `/dev/nvme0n1p3`, with
swappiness 60 and zswap disabled; host settings are unchanged. This result does
not establish another host's swap latency or justify unlimited cache growth.

A preceding experiment uses the same 8 GiB cache with only 6 GiB RAM plus
4 GiB swap. It fails during the seventh seed after reaching the full 4 GiB
swap limit and one isolated-container OOM kill. Six completed cold requests
match the control, but no bounded replay runs, so that attempt qualifies no
swapped-cache restore behavior. It demonstrates why a cache byte ceiling is
not a total process-memory budget: active state and other allocations also
need headroom.

Both experiments restore the original production container, image, environment,
mounts and critical parameters, and release their owned drains. The original
worker is healthy and not OOM-killed. No production `SWAP` value, cache ceiling,
host swap device or kernel setting is changed by these isolated experiments.

## Validation, provenance and limits

The completed four-patch development build passed 68 package checks with the
Python checkout absent, nine focused checks, and two real CPU cache/Rust-router
integrations. It is an incremental CUDA development build, not a complete new
production image. No physical GPU was present in that build container; the
separate 4090 experiments supply the hardware evidence above. Later call-site
coverage changes need their own updated test/artifact record.

A full x86 CUDA image from source `274eea0` also passes all 69 package checks
without the Python oracle checkout, plus linkage checks with ACE using its own
GGML. Its image ID is
`sha256:023c1df039ff54a9ca0185328d8d65c5c74bf8e4b4fd6f2a274851f42c438f56`.
It is historical, was neither pushed nor deployed, and is superseded by the
final `448bf6b` image. Intermediate held builds are not release evidence.

The final source `448bf6b` produces image
`sha256:462cd19b65435d0ff95153b8aed32d91e5a121b8e78df2fbe4d37c39ec28bb57`.
It passes all 69 package checks without the Python oracle checkout and clean
linkage, including ACE resolving its own GGML. Two final real CPU
cache/Rust-router integrations also pass in 29.88 seconds. These CPU checks
are separate from the physical CUDA evidence.

The final focused engine/source-contract set passes six checks in 4.01 seconds.
The frozen benchmark guard suite passes 87 tests, and the tracked recovery
helper passes 47 offline guards. These test sets overlap package coverage and
are recorded separately rather than added into one total.

An isolated 5090 canary runs that exact packaged image, with no worker/engine
overlays, against a real private Rust router. It preserves the public model
alias, application features and model presets. Buffered chat and SSE chat
return the requested nonce plus arithmetic answer 42; vision returns the
requested nonce plus red. The router advertises runtime `448bf6b`, the expected
262144-token context, positive text/vision capacity and populated RAM/VRAM/GPU
metadata. Presets are not disabled to obtain these functional results.

The canary never registers the candidate with the central router and does not
remain deployed. It restores the same original container, image, environment
and mounts, then releases its drain. Its rollback image tag is
`ezlocalai:rollback-canary-20261026-full-canary-fdb6add414`. This validates the
packaged text/vision router path; it does not validate STT, Qwen TTS, embeddings,
image/music/video/wake-word GPU generation or multi-GPU behavior in this canary.
The image is not pushed or rolled out to the fleet by these checks.

The final router snapshot shows all three production workers with fresh
heartbeats, no drain or open circuit, version `46d9ba8`, and unchanged declared
contexts of 230000, 230000 and 262144. This confirms the candidate is not left
deployed after testing. All benchmark operators are stopped and their original
workers are restored. The two verified 12288 MiB RAM settings remain applied;
the 5090's production cache-swap allowance remains zero.

All isolated experiments restore their original production configuration and
release owned drains. Successful final qualifications observe no OOM kills;
the deliberately undersized swap experiment records one isolated-container
OOM as described above, with production restored healthy. The separately
verified persistent RAM changes are the exception to environment restoration.

Raw proof remains under ignored
`build/validation/llm-cache-depth-20261026/`, principally
`DevXT4090-baseline-depth-f9977f92`,
`DevXT4090-combined-resident-b269671d` and
`DevXT4090-router-gpu-54124ad3a6`, plus the experimental
`DevXT4090-coverage-fixed-resident-035908e1`, the completed active release run
`DevXT4090-release-aligned-quality-ca9cf87e`,
`DevXT3090V-release-resident-quality-63c83517`,
`DevXT5090-release-resident-quality-03c6cc1f`,
`DevXT{4090,3090V}-ram-pressure-production-dc2ca20d`,
`cache-environment-qualification`, `disk-state`, `full-cuda-image-448bf6b`, and
`DevXT5090-full-canary-fdb6add414`, the three
`*-release-named-capacity-6e73606a` records, both bounded swap experiments, and
`final-fleet.json`.
Exact artifact, harness, input and result
fingerprints are summarized in
[`cache_depth_20261026_validation.json`](../tests/fixtures/cache_depth_20261026_validation.json).

## Limits and follow-up

- **Broader quality:** finite controls and sampled semantic checks pass;
  freeform rewind answers differ and long continuation behavior remains limited.
- **Legacy/corrupt state:** final normal current-format disk roundtrips pass on
  all three GPUs; arbitrary older or damaged files are not qualified here.
- **Swap generalization:** the measured 5090 workload uses actual OS swapping;
  other storage backends and per-buffer swap-in costs remain unmeasured.
- **RAM and rollback followups:** co-resident-model/host-memory pressure,
  and a live post-finalization recovery exercise. The V2 helper fix has 47
  offline guards; live settings, normal state and cold capacity checks pass.
- **Production deployment:** the packaged image and isolated canary pass, but
  no central candidate registration, image push or fleet rollout is performed.

These limits prevent the recorded tests from being read as universal quality
equivalence or proof for every fleet workload. Sampled output agreement and
unchanged arithmetic do not replace broader model-quality evaluation.
