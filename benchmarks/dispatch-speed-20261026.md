# Long-conversation request preparation and dispatch — 2026-10-26

The worker now keeps the outer request parsed while preparing text, image and
audio input. It also reuses the normalized message list when transcription has
not changed it. This removes repeated parsing, copying and serialization of a
long conversation before the engine receives it. Model weights, presets, KV
precision, context limits and inference arguments are unchanged by this code.

These measurements cover CPU preparation and actual worker HTTP handling with
a deterministic loopback mock engine. They do **not** measure GPU prefill,
decode throughput, cache restoration or model quality. Separate 3090J physical-batch measurements and the applied setting are recorded
below; they use the existing production engine, not the new request code.

## Implemented behavior and controls

The ordered fields retain the last value of duplicate decoded keys and the
first key's output position. Model loading, input errors, media callbacks and
fallback preparation keep their original order. Every present `audio_format`,
including null or another falsey value, still invokes the existing audio helper.
The ordinary text path avoids that helper's no-op scan and serialization round
trip. Missing or invalid message access retains its existing error path.

`bench_input.cpp` calls the actual `llm_prepare_text_image_input` function with
fixed callbacks. Request, complete prepared-payload and helper-event SHA-256
values match across all seven workloads and all four baseline/candidate runs.
The 3,399 captured input cases and new escaped-key/null-audio guards also pass.

## CPU preparation

The same benchmark driver runs in baseline, candidate, candidate, baseline
order. Each invocation supplies seven timed batches per case, giving **14
sample batches per variant and case**. Each batch averages 100 calls for the
small input, ten for 100 KiB and three for the larger inputs. The table reports
medians in milliseconds per preparation call; construction of the benchmark
input and output hashing are outside the timed loop.

| Workload | Baseline | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| 1 KiB text | 0.0208 ms | 0.0088 ms | 57.8% |
| 100 KiB text | 0.894 ms | 0.324 ms | 63.8% |
| 1 MiB text | 13.242 ms | 3.753 ms | 71.7% |
| 256-message history | 13.000 ms | 2.760 ms | 78.8% |
| History with an image placeholder | 10.528 ms | 2.349 ms | 77.7% |
| History with `audio_format` and a fixed transcription callback | 13.781 ms | 9.492 ms | 31.1% |
| 1 MiB completion prompt | 3.681 ms | 0.719 ms | 80.5% |

The audio case retains the existing helper boundary, so it avoids less work.
These component timings do not predict the same percentage improvement in a
complete model request.

## Actual worker HTTP with a mock engine

`bench_worker_input.py` starts two owned CPU-only containers from the same
immutable CUDA image. Their worker executables differ; their planned settings
differ only in the worker port. Both workers and the mock engine bind to
loopback. No GPU, router registration, provider, fallback, media service or
production data directory is used. The harness verifies the image, CPU-only
runtime, environment and ownership labels, then removes its containers by
immutable ID after checking ownership. Generated worker and engine credentials
remain authenticated. Docker receives key-name-only `--env` arguments and an
explicit subprocess environment mapping; credential values are excluded from
command arguments, benchmark files and logs, and the parent environment is not
mutated. The original temporary dotenv transport was replaced after CodeQL
identified it; no authentication bypass or suppression was used.

Seven chat workloads run buffered and streamed. Text workload sizes below are
Unicode source-character budgets; encoded wire requests also include UTF-8 and
JSON escaping. Six ABBA blocks give **12 samples per variant and case/mode**,
or **336 measured requests**. An additional
28 warmups are excluded. Every request produces exactly one engine request;
the engine-bound raw bytes, length and path match across variants and repeats.
Actual answer content and the normal finish state also match. Volatile response
IDs and timestamps are not part of that output comparison. Streaming first
output means the first nonempty content delta, excluding role-only events.

| HTTP workload | Buffered full response, baseline → candidate | Streaming first content, baseline → candidate |
| --- | ---: | ---: |
| 1,024-character text | 1.262 → 1.243 ms | 1.035 → 1.120 ms |
| 102,400-character text | 4.180 → 3.887 ms | 3.713 → 3.013 ms |
| 1,048,576-character text | 32.011 → 26.145 ms | 30.723 → 24.168 ms |
| 16-message history | 5.168 → 4.120 ms | 4.189 → 3.405 ms |
| 256-message history | 32.494 → 24.905 ms | 32.044 → 23.086 ms |
| 256 messages, final content represented as text blocks | 33.151 → 26.620 ms | 30.599 → 23.549 ms |
| 256 messages, every content represented as text blocks | 39.982 → 34.273 ms | 42.318 → 34.382 ms |

The 256-message requests are about 1.13 MB on the wire. Their first content
arrives **28.0%** sooner; buffered completion improves **23.4%**. These timings
include HTTP parsing, validation, queueing, preparation, forwarding and a fixed
mock response. They show a worker overhead reduction rather than faster GPU
execution.

This table uses the complete rerun after the benchmark credential transport fix.
Worker binaries and all 14 request/engine-payload/output identities match the
previous run. Absolute HTTP times vary between runs on this shared host with
ambient load and scheduling; the lower absolute times are not attributed to the
launcher change. Medians are not selected across runs. Small-case gains are not
consistent: the current 1,024-character streaming median is slightly slower,
while the earlier 102,400-character streaming comparison was effectively neutral.
No universal short-request improvement or statistical confidence interval is
claimed.

## Regression and package checks

- All six strict reference suites pass in 29.58 seconds: output, request,
  input, audio, video and stream. They use an export of Python commit `0f7e9f4`
  with the recorded source hashes verified.
- All **70 CUDA-image package checks** pass in 307.93 seconds with the configured
  Python reference root absent. Candidate executables were rebuilt inside an
  owned CPU-only container from the existing full CUDA production image, using
  its unchanged engine and dependencies. This is not a fresh Docker image build
  or physical GPU inference validation.
- `llm_input_package` runs all 3,399 captures, the new duplicate-key/null-audio
  guards and managed preparation checks through explicit `--fixtures` mode.
  Only Python source-hash checks are omitted in that mode. Missing captures,
  corrupted expectations and an absent strict reference all fail as intended.
- All four platform Docker package regexes select `llm_input_package` and omit
  the strict external-source test. CPU-only, ROCm, Jetson and GB10 images were
  not rebuilt or physically validated in this pass.
- All 14 offline HTTP benchmark guards pass. They cover request counts and byte
  identity, first-content timing, isolation, loopback/redirect restrictions,
  refusal to remove an unowned container, and credential transport without
  temporary files, argument values or parent-environment mutation.

The first 70-check package run exposed a pre-existing media parent-death fixture
race. Its test executable was byte-identical to the existing production image's
test executable: SHA-256
`6b5644d99271f4e789798b9c3db3c51a15976767ce5ad20ba77ecfd7e74eba98`.
The fixture now waits for the active worker generation before abandoning its
parent. Production media runtime sources are unchanged. A fresh fixture build
passes 100 consecutive repetitions in 16.18 seconds; disabling the parent-death
signal still fails the negative control in 2.19 seconds. An intermediate build
with stale source freshness was discarded and is not qualification evidence.

## Provenance and reproduction

- Baseline source: `694137438e62895b89ed3d42b1d7790486e47c65`.
- Runtime preparation change: `8e1f9d607d69982cd6e49107fbf0af279646b153`.
- HTTP benchmark: `fb6b3afc483cf8921691f40c51b356d0c264d4cf`.
- Benchmark credential transport correction: `b26f0ce09951ae58c1c46b749bf8a93dd211a5ab`.
- Media fixture synchronization: `c36c9b71c670caae464a95ef8c8c5596e16f6ae0`.
- Existing image used by the owned build/benchmark containers:
  `sha256:845c0d4edad3ee960bb22e76a48de843560d4a197f6a430e799e3136b8a032f4`.
- Baseline worker SHA-256:
  `b411c47a897a4decc9d65ce0a5aca4f41ee8e54f3703833e23c4149e8ecab7fd`.
- Candidate worker SHA-256:
  `ae6634f1d0f32a05117c3ae79b1b8182a31a476ac614719ccec0e95b88ee5a52`.

Build the same `bench_llm_input` driver against the baseline and candidate native
libraries, retain both executables, and invoke each with a distinct output
directory in ABBA order. For HTTP reproduction, supply those workers to
`benchmarks/bench_worker_input.py` using its immutable `--image`,
`--baseline-binary`, `--candidate-binary`, `--output` and `--blocks 6` arguments.
The script owns its temporary containers and needs no production credentials.

Compact results, hashes and validation scope are recorded in
[`dispatch_speed_20261026_validation.json`](../tests/fixtures/dispatch_speed_20261026_validation.json).
Raw batches, payloads and full logs stay under ignored
`build/validation/dispatch-speed-20261026/`. The authoritative HTTP report is
`worker-http-key-only-env/worker-input.json`; the earlier HTTP report is
historical evidence, with matching workload/output identity but superseded timings. The native change is in
[ezlocalai PR #142](https://github.com/DevXT-LLC/ezlocalai/pull/142); separate
router scheduling work is in
[WorkerRouter PR #6](https://github.com/DevXT-LLC/WorkerRouter/pull/6).
The CPU-only dispatch checks did not replace production containers or images,
so they required no rollout rollback tag. No candidate image was published as
latest.

## 3090J physical batch: qualified and applied

The local DevXT3090J uses an RTX 3090 Ti, 128 GB RAM and the deployed model
SHA-256 `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2`.
Those weights match the recorded 5090 file and differ from V/4090 despite the
same filename. The comparison uses J's own unchanged image and weights, logical
batch 4096, one slot, `q4_0` K/V, MTP and its 31488 MiB host cache ceiling.
Only physical batch 512 versus 1024 varies. No GPU clock or power setting was
changed.

Four fresh-engine cold runs use ABBA order and identical 149992-token requests,
with zero cached tokens and identical normal-stop answers. Rates are
731.23/733.34 tok/s at 512 and 720.72/761.45 at 1024. The medians are
732.28 and 741.08 tok/s: the roughly 1.2% difference lies within the observed
spread and is **not a reliable cold-prefill gain**.

The returning-conversation result is stronger. All 24 finite controls pass;
all 12 paired requests and answers match exactly:

| Case | Physical batch 512 | Physical batch 1024 |
| --- | ---: | ---: |
| Replay first output | 648 ms | 470 ms |
| Append first output | 1.192 s | 0.891 s |
| Shorter-prefix rewind first output | 18.628 s | 1.184 s |

All three rewinds change from 23256 evaluated / 85 cached tokens to
728 evaluated / 22613 cached tokens. Batch 1024 enables the already-merged
aligned checkpoint path; this is cache coverage, not a claimed faster decode
kernel. These exact finite answers do not establish universal model-quality
equivalence for arbitrary rewritten prompts.

Both batches fully evaluate the same 208866-token near-context request with
zero cache reuse, zero inference retries and no context reduction. The declared
220000-token budget is retained (the engine pads internally to 220160). Both
return `Python`, then correctly recognize the red image. Capacity rates are
608.89 and 673.79 tok/s, **one screen each**, not a repeated rate estimate.
No OOM or test-container swap was observed.

The qualification retained and restored the exact original container, full
argv, configuration, environment and mounts, then verified fresh heartbeats
and released its owned drain. A separate reviewed rollout then applied only
`LLM_UBATCH_SIZE=1024` in `/home/josh/repos/xtsys/ezlocalai/.env` and the running
container. Previously the file had no such key and the container's `auto`
resolved to 512. Current container `79a536848ee1` uses the same immutable image
`845c0d4…`; every other engine argument, context, environment value, host limit,
mount and static network configuration is preserved.

Two fresh drained and two fresh resumed heartbeats pass, followed by routed
chat, streaming and red-image checks with fallback disabled. The original was
retained through those checks and then finalized. No owned drain remains.
The new C++ request-preparation code is still PR-only; this live setting uses
merged worker `6941374` and its existing engine. No other worker setting changed.

Rollback image: `ezlocalai:rollback-env-j-env-20261026-05d665ee`.
Private transaction directory:
`/tmp/ezlocalai-env-update-j-env-20261026-05d665ee`, including its `dotenv.backup`
at mode 0600 in a 0700 directory. The unchanged tracked recovery core supports
that finalized transaction; a rollback must use an owned idle lease and verify
full original argv/configuration and fresh heartbeats before resuming. The
rollout used 18 adapter guards, the core's 47 guards and independent review;
the GPU qualifier has 32 offline guards. These counts describe separate checks. No live
post-finalization rollback was performed; the retained image, private backup
and recovery guards are the rollback material and offline evidence.

The compact fixture contains the qualification and rollout proofs and hashes.
Raw operators/results remain under `build/validation/dispatch-speed-20261026/`
in `j-ubatch-live-ad88fa15/` and `j-env-promotion-plan/`. After merging the code
PRs, verify the deployed worker version and repeat real interleaved routed
conversations with stable keys and native cache telemetry. A separate agent
owns the model-matching additions to router PR #6; their runtime changes need
their own combined validation.
