# Long-conversation request preparation and dispatch — 2026-10-26

The worker now keeps the outer request parsed while preparing text, image and
audio input. It also reuses the normalized message list when transcription has
not changed it. This removes repeated parsing, copying and serialization of a
long conversation before the engine receives it. Model weights, presets, KV
precision, context limits and inference arguments are unchanged by this code.

These measurements cover CPU preparation and actual worker HTTP handling with
a deterministic loopback mock engine. They do **not** measure GPU prefill,
decode throughput, cache restoration or model quality. The separate live 3090J
physical-batch qualification remains pending below.

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
immutable ID after checking ownership.

Seven chat workloads run buffered and streamed. Text workload sizes below are
Unicode source-character budgets; encoded wire requests also include UTF-8 and
JSON escaping. Six ABBA blocks give **12
samples per variant and case/mode**, or **336 measured requests**. An additional
28 warmups are excluded. Every request produces exactly one engine request;
the engine-bound raw bytes, length and path match across variants and repeats.
Actual answer content and the normal finish state also match. Volatile response
IDs and timestamps are not part of that output comparison. Streaming first
output means the first nonempty content delta, excluding role-only events.

| HTTP workload | Buffered full response, baseline → candidate | Streaming first content, baseline → candidate |
| --- | ---: | ---: |
| 1,024-character text | 1.394 → 1.470 ms | 1.842 → 1.670 ms |
| 102,400-character text | 6.333 → 5.066 ms | 5.120 → 5.098 ms |
| 1,048,576-character text | 48.433 → 34.946 ms | 51.572 → 41.500 ms |
| 16-message history | 6.757 → 5.749 ms | 7.199 → 5.921 ms |
| 256-message history | 55.227 → 42.162 ms | 47.517 → 38.429 ms |
| 256 messages, final content represented as text blocks | 51.272 → 41.352 ms | 52.625 → 39.496 ms |
| 256 messages, every content represented as text blocks | 69.610 → 57.408 ms | 68.734 → 53.820 ms |

The 256-message requests are about 1.13 MB on the wire. Their first content
arrives **19.1%** sooner; buffered completion improves **23.7%**. These timings
include HTTP parsing, validation, queueing, preparation, forwarding and a fixed
mock response. They show a worker overhead reduction rather than faster GPU
execution. Small HTTP cases remain noisy: the 1,024-character buffered median is slightly
slower, and 102,400-character streaming latency is effectively unchanged. No universal
speedup or statistical confidence interval is claimed from this single-host run.

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
- All 12 offline HTTP benchmark guards pass. They cover request counts and byte
  identity, first-content timing, isolation, loopback/redirect restrictions and
  refusal to remove an unowned container.

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
`build/validation/dispatch-speed-20261026/`. The native change is in
[ezlocalai PR #142](https://github.com/DevXT-LLC/ezlocalai/pull/142); separate
router scheduling work is in
[WorkerRouter PR #6](https://github.com/DevXT-LLC/WorkerRouter/pull/6).
The CPU-only dispatch checks did not replace production containers or images,
so they required no rollout rollback tag. No candidate image was published as
latest.

## Separate 3090J physical-batch qualification — pending

A separate owned live operator is comparing physical batches 512 and 1024 on
DevXT3090J while retaining its 220000-token context, model files, precision,
presets and remaining engine arguments. Qualification includes matched cold
prefill, finite continuation/replay, near-context capacity and vision, followed
by exact original-container/argument restoration and fresh router heartbeats.
That run is still in progress. No GPU speedup, selected production setting or
permanent environment change is claimed here; the final results and restoration
evidence must be filled after the operator completes.
