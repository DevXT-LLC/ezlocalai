# Conversation cache and router admission measurements - 2026-10-26

Status: partial qualification report. Results cover the 4090 four-patch
cache/MTP candidate and router comparison, 4090/3090V production-engine cache
capacity tests, applied RAM settings on those two workers, and baseline 4090
disk-state roundtrips. The latest release candidate also completes active
aligned-checkpoint, finite-instruction and eight-conversation pressure checks
on the 4090. Earlier experimental checkpoint results remain historical. Final
state/quality qualification, other-GPU candidate checks, swap and release of a
final image remain pending. The earlier
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
restored, and the owned drain is released. Candidate-engine results on 3090V
and 5090, final disk-state checks and final-image release remain pending.

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
the old stopped containers. The helper version used for these updates expects
those original container IDs, so automatic rollback after finalization remains
a helper gap; retained files and image tags allow manual reconstruction. This
report does not claim that command has been validated after finalization.

The 5090 installation was left unchanged in this work. Its historical effective
cache is 32000 MiB, and the operator confirms `SWAP=0` remains unchanged on the
workers. Fresh 5090 state/capacity verification and any nonzero-swap testing are
pending; historical settings are not a new physical qualification.

## Disk state and benchmark-operator checks

The matched baseline 4090 engine completes normal text and vision slot-save,
state-overwrite, reload and continuation comparisons with identical request and
answer hashes. In-memory conversation caching is disabled for this test, and
the launch uses an isolated owned snapshot directory. Text saves/restores
24842 tokens in 1188510480 bytes; vision saves/restores 1091 tokens in 495318656
bytes. Both restored continuations reuse the saved prefix. These results do
not reproduce a legacy-state failure, validate corrupt/legacy payloads, or
qualify disk state for the final aligned candidate.

An earlier benchmark-operator attempt fails its private-port preflight before
preparing or stopping production. Version 2 addresses false busy detection for
`TIME_WAIT` sockets while retaining rejection of active listeners. Its 56
offline tests cover depth/state operations, provenance and real local sockets;
they are operator checks rather than additional GPU inference evidence. The
failed attempt releases its drain and verifies that production remains healthy.

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
It is under a quality hold, was neither pushed nor deployed, and is superseded
for release qualification by the aligned `56d6375` build. The latter build and
its final package/artifact results are pending here. Passing package tests on
the held image is not a production release decision.

All isolated experiments restore their original production configuration and
release owned drains, with no observed OOM kills. The subsequent, separately
verified persistent RAM changes are described above and are the exception to
environment restoration.

Raw proof remains under ignored
`build/validation/llm-cache-depth-20261026/`, principally
`DevXT4090-baseline-depth-f9977f92`,
`DevXT4090-combined-resident-b269671d` and
`DevXT4090-router-gpu-54124ad3a6`, plus the experimental
`DevXT4090-coverage-fixed-resident-035908e1`, the completed active release run
`DevXT4090-release-aligned-quality-ca9cf87e`,
`DevXT{4090,3090V}-ram-pressure-production-dc2ca20d`,
`cache-environment-qualification`, `disk-state`, and `full-cuda-image-274eea0`.
Exact artifact, harness, input and result
fingerprints are summarized in
[`cache_depth_20261026_validation.json`](../tests/fixtures/cache_depth_20261026_validation.json).

## Pending qualification

- **3090V and 5090 candidate engines:** equivalent cache/MTP and router
  comparisons, preserving their existing model files and selected batch sizes.
  The completed 3090V capacity result above uses its production engine.
- **Coverage/checkpoint changes:** final build/release qualification and broader
  quality acceptance. Active 4090 reuse and finite instruction checks pass;
  freeform rewind answers differ, with sampled semantic limits disclosed above.
- **State roundtrips:** final text/vision slot-save, reload, continuation and
  legacy/corrupt-state evidence for the final patch stack.
- **Swap:** a controlled nonzero swap-in workload and host/container pressure
  measurements. Zero swap counters are not swap validation.
- **RAM and rollback followups:** co-resident-model/host-memory pressure,
  post-finalization automatic rollback, and fresh 5090 settings/state/capacity
  evidence. The applied 4090/3090V `0.4` setting is already verified above.
- **Final images and deployment:** fresh package results and artifacts for the
  final source, then normal image automation and routed version/capacity checks.

These pending items prevent this partial report from being read as final fleet
qualification or universal quality equivalence. The sampled output agreement
and unchanged arithmetic do not replace broader model-quality evaluation.
