# Long-Context Prefill Qualification

2026-11-11. The isolated screens below completed restoration. Subsequently,
the user authorized the 5090-only deployment documented at the end; historical
restoration statements describe the experiments, not that later rollout.
Passing retrieval and two-function checks is not proof of general quality or
mathematical equivalence. Global runtime defaults are unchanged.

## Method

- Reserve one idle worker through the router's owned maintenance lease. Keep at
  least two other fresh Qwen workers available. The router permits only one
  maintenance reservation at a time, so GPU runs are sequential.
- Retain the original container, image, arguments, context size, Q4 KV settings,
  projector, and any resident LFM voice engine. Mount models read-only into an
  unregistered diagnostic engine on loopback. Restore the original container and
  verify its exact engine arguments and health before releasing the drain.
- An independent, bounded watchdog restores the original container if the probe
  controller is lost. Preflight failures must not replace a healthy worker.
- Use the actual tokenizer and native timings. Cold requests must not reuse a
  meaningful prefix; repeated requests are measured separately. Report initial
  8K requests as initial-request measurements, not warmed graph throughput.
- Infrastructure cases require exact current paths, ports, health URLs and safe
  remediation IDs from early, middle and late inventory records. Conflicting
  obsolete records surround each target. All prompts are synthetic.
- Use identical prompts within each worker's bracketed sweep; keep weights fixed
  except for the explicitly paired Q3/NVFP4 artifact comparison. Greedy seed 42,
  thinking disabled, 1,024 output tokens for infrastructure-only runs and 2,048
  for the 64K infra/Rust/JavaScript comparison. This is a controlled screen,
  not a complete reproduction of the production thinking workload.
- Rust/JavaScript checks execute generated code only in network-disabled,
  read-only, resource-limited disposable containers. They do not execute it on
  the host. Their results are recorded separately from retrieval grades.

Commands and provenance are retained in the JSONL evidence under the ignored
`build/validation/prefill-fleet-20261111/` directory. The compact
[fixture](../tests/fixtures/prefill_fleet_20261111_validation.json) contains 11
files ending in confirmed restoration: 172 responses, including 160 passing
infrastructure grades and 12 externally graded code outputs (12/12 passed).
Experiment records use `summarize_prefill_fleet.py`, supplemented with hashed
code-grade provenance and three utilization-confirmed memory samples; prompt
and answer text are omitted. The intentionally partial A run is identified below.

## RTX 5090 Batch And MTP Screen

Context 262,144; logical batch 4,096; baseline physical batch 2,048;
MTP maximum 3, minimum probability 0.1; Q4 KV; LFM voice remains resident.
The original container and engine arguments were restored and the drain released.
All 40 cold/repeated retrieval responses passed. There are four distinct prompt
sizes, not 40 independent real-world tasks.

| Prompt | Baseline TTFT, before / after | Baseline prefill tokens/s, before / after | Batch 1,024 TTFT | Batch 4,096 TTFT | MTP off TTFT |
| --- | --- | --- | --- | --- | --- |
| 8K | 2.44 / 2.49 s | 3,306 / 3,244 | 2.36 s | 3.13 s | 1.97 s |
| 64K | 21.15 / 21.90 s | 3,047 / 2,944 | 22.41 s | 22.25 s | 20.46 s |
| 128K | 60.99 / 62.23 s | 2,111 / 2,069 | 66.38 s | 60.41 s | 58.33 s |
| 200K | 130.82 / 132.52 s | 1,536 / 1,519 | 141.02 s | 119.95 s | 123.26 s |

Actual evaluated lengths were 8,044, 64,183, 128,092 and approximately 200K;
use the native counts in the fixture for exact comparisons. Repeated requests
evaluated four new tokens and returned the first token in approximately 0.06-0.23 s.

Batch 4,096 improves the 200K case but is not a general-purpose promotion:
it leaves only 1,001 MiB free after loading both models, and a live under-load
sample reported 841 MiB. Batch 1,024 leaves 4,743 MiB free but slows the long
prompts. Baseline batch 2,048 leaves 3,381 MiB free. The separate batch-3,072
midpoint measurement is recorded below.

Disabling MTP does not improve total task time here. At 128K, baseline completes
in 65.12-66.38 s versus 76.57 s without MTP. At 200K it is 136.10-137.72 s versus
148.30 s. The MTP-off comparison bounds total MTP overhead; it does not attribute
that entire difference to host transfers or establish a device-copy speedup.

## RTX 5090 Midpoint Screen

`5090-ub3072-screen.jsonl` brackets batch 3,072 with production batch-2,048
controls, retaining logical batch 4,096, MTP max 3, context 262,144, Q4 KV and
resident LFM voice. All 24 responses passed; original-container/argument
restoration and drain release completed successfully.

| Prompt | Batch 2,048 TTFT, before / after | Batch 3,072 TTFT | Batch 2,048 prefill tokens/s, before / after | Batch 3,072 prefill tokens/s |
| --- | --- | --- | --- | --- |
| 8K | 2.804 / 2.596 s | 2.797 s | 2,875 / 3,107 | 2,883 |
| 64K | 22.52 / 22.29 s | 22.50 s | 2,877 / 2,891 | 2,865 |
| 128K | 63.62 / 63.02 s | 62.17 s | 2,045 / 2,044 | 2,072 |
| 200K | 133.64 / 133.07 s | 129.48 s | 1,509 / 1,513 | 1,566 |

TTFT improved 1.82% at 128K and 2.91% at 200K versus mean bracketed controls,
with no consistent 8K/64K benefit. Ready free VRAM fell from 3,381 to 2,019 MiB;
`5090-ub3072-memory.csv` records 1,899 MiB at **99% GPU utilization**, not a
worst-case guarantee. Given this tradeoff, batch 2,048 remains deployed.
No automatic batch cap changed.

## GDN Scheduling Screen

The CUDA 12.8 experimental engine was tested off / on / off at 64K with identical
prompts/Q3 weights, physical batch 2,048, MTP max 3 and resident LFM voice.
Engine SHA-256:
`b19b041625578c6f3c6c8d1294a361e3480c62d7403b896908397354eebc9db6`;
the JSONL also pins its shared-library hashes.

| GDN setting | Cold TTFT | Native prefill tokens/s | Total task time | Ready reserve |
| --- | --- | --- | --- | --- |
| Off, opening | 21.78 s | 2,951 | 25.15 s | 3,381 MiB |
| On | 21.95 s | 2,928 | 25.37 s | 3,381 MiB |
| Off, closing | 22.37 s | 2,874 | 25.82 s | 3,381 MiB |

All six responses passed and restoration is confirmed. The on result lies inside
the off-control spread, establishing no clear latency benefit or general
numerical equivalence; the experiment stays disabled. The profile retry captured
and analyzed valid Nsight data but required manual production recovery after
automatic cleanup failed. See [prefill-profile-20261111.md](prefill-profile-20261111.md).
Profile files are excluded from this automatic-restoration fixture; their capture
does not establish dispatch attribution for the experimental GDN toggle.

## Fleet Interpretation

The fleet has two different Q3 artifacts under the same filename. The larger
13,441,059,904-byte file hashes to
`00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2`.
Other workers use a 13,146,393,504-byte variant with a different tensor mix and
calibration metadata, SHA-256
`8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e`.
DevXT3090J uses the larger artifact; DevXT4090, DevXT3090V and DevXT3090A use
the smaller one. Cross-machine speed is therefore not a pure GPU comparison.
Every experiment records its own file hash and engine arguments.

The completed fleet screens below used logical batch 4,096, Q4 KV and MTP minimum
probability 0.1. Context was 230,000 on 4090/V/A and 220,000 on local 3090J.
V, A and J all identify as RTX 3090 Ti in their GPU provenance. MTP maxima and
baseline physical batches are not identical across workers; the table preserves
those differences rather than treating the configurations as interchangeable.

Cold TTFT is in seconds. Paired baseline values are the opening / closing runs,
not confidence intervals. The 8K rows are initial-request measurements. Reserve
means free VRAM in the recorded ready-state snapshot, not guaranteed peak-load
headroom. The short A comparison intentionally tested only 64K.

| Worker | Physical batch | MTP max | 8K TTFT | 64K TTFT | 128K TTFT | Ready reserve, MiB |
| --- | --- | --- | --- | --- | --- | --- |
| DevXT4090 | 1,024 baseline | 3 | 2.716 / 2.725 | 24.64 / 24.84 | 60.85 / 61.15 | 2,837 |
| DevXT4090 | 512 | 3 | 2.689 | 25.04 | 61.83 | 3,457 |
| DevXT4090 | 2,048 | 3 | 2.870 | 25.22 | 62.02 | 1,599 |
| DevXT3090V | 512 baseline | 2 | 5.067 / 5.229 | 49.72 / 50.87 | 127.53 / 128.46 | 3,745 |
| DevXT3090V | 1,024 | 2 | 5.134 | 47.71 | 119.48 | 3,125 |
| DevXT3090A | 1,024 baseline | 3 | Not retested | 52.06 / 50.22 | Not retested | 2,970 |
| DevXT3090A | 512 | 3 | Not retested | 54.73 | Not retested | 3,590 |
| DevXT3090J (local) | 1,024 baseline | 3 | 5.521 / 5.439 | 51.66 / 50.64 | 128.97 / 127.66 | 1,732 / 1,615 |
| DevXT3090J (local) | 512 | 3 | 5.694 | 54.86 | 139.37 | 2,174 |
| DevXT3090J (local) | 2,048 | 3 | 6.675 | 56.48 | 137.95 | 1,130 |

V's batch 1,024 reduced 128K TTFT by approximately 6.65% relative to the mean
of its two batch-512 baselines. Its 64K TTFT was 47.71 s versus 49.72 / 50.87 s.
On A, batch 512 was slower than both batch-1,024 baselines but left 620 MiB more
free at readiness. On local J, both alternate batches were slower at 64K and
128K; batch 2,048 left only 1,130 MiB free, below a 1.5 GiB reserve. On 4090,
batch 2,048 also slowed these sizes and left 1,599 MiB free. These are bracketed
synthetic retrieval observations, not fleet-wide configuration recommendations.

The original A queue was `baseline,ub1024,baseline`, but its provenance showed
that the baseline already used physical batch 1,024 and MTP max 3. It was stopped
intentionally after its first 64K cold/repeated pair, rather than running three
identical configurations through all sizes. Only the exact owner-verified remote
probe received SIGTERM; its controller and independent watchdog were left alone.
The partial file retains four passing responses: 8K cold TTFT 6.389 s and 64K
cold TTFT 49.60 s, plus their cached repeats. No 128K result is claimed. Its final
receipt confirmed restoration and the controller released the lease before the
separate `baseline,ub512,baseline` 64K-only retry started.

All 76 recorded cold/repeated responses passed across these five evidence files:

- [4090-screen.jsonl](../build/validation/prefill-fleet-20261111/4090-screen.jsonl): 24/24.
- [3090v-screen.jsonl](../build/validation/prefill-fleet-20261111/3090v-screen.jsonl): 18/18.
- [3090a-screen.jsonl](../build/validation/prefill-fleet-20261111/3090a-screen.jsonl): 4/4, intentionally partial.
- [3090a-batch-screen.jsonl](../build/validation/prefill-fleet-20261111/3090a-batch-screen.jsonl): 6/6, completed short retry.
- [3090ti-screen.jsonl](../build/validation/prefill-fleet-20261111/3090ti-screen.jsonl): 24/24, local DevXT3090J.

All five files end with `restored`, `original_container=true` and
`original_argv=true`. All recorded cold samples pass the uncached-prefix gate.
Post-run checks confirmed healthy original services and fresh router heartbeats;
all four workers were undrained with `drain_until=null`. The short A controller
exited successfully with `drain_released=true` and `renewal_failed=false`.

All deployed settings stayed original. No automatic cap change was made: the
code's `8192/1024` automatic policy is not the measured `4096` logical-batch
configuration, including the `4096/2048` pairings above. Its quantization-blind
selection cannot be generalized from these Q3 artifacts or synthetic passes.

The local DevXT3090J reports PCIe Gen4 x4 under 99-100% load, despite x16 maximum
capability. The other 24 GB cards report x16 width while idle. Idle Gen1 is a
power-saving state, not itself a fault. No PCIe/BIOS settings were changed, and
no claimed latency gain is attributed to changing the link.

## NVFP4 Model Audit (CPU Only)

Disk staging and the CPU metadata/payload audit are complete. The compact evidence
is [prefill_nvfp4_20261111_validation.json](../tests/fixtures/prefill_nvfp4_20261111_validation.json).
The [tiyuvta artifact](https://huggingface.co/tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF/blob/6c39753daccd7473d4e60735ec8ace1d30afdd10/README.md)
is pinned at revision `6c39753daccd7473d4e60735ec8ace1d30afdd10`:
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, 15,705,922,304 bytes, SHA-256
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
Full size and SHA-256 were verified before atomic rename on DevXT5090 under
`/home/josh/prefill-experiments/nvfp4-20261111/`, outside `/app/models`.

Against the actual larger Q3 identified above, all 866 tensor names/shapes and
architecture metadata match. Every tokenizer field matches, including all
248,320 tokens, 247,587 merges and the complete chat template; no paired
`--candidate-template` override is needed. All 360 shared F32 payloads are
byte-identical, and candidate/baseline/projector payload bounds pass without
overlap or trailing bytes. The embedded MTP block contains all 15 expected
tensors (eight NVFP4, seven F32), with full-vocabulary Q5_K embeddings/output
and no masked head. The existing BF16 projector's 5,120-dimensional output
matches structurally; vision inference has not been tested.

This is not a uniform precision upgrade: 17 baseline Q5_K attention-value
matrices use NVFP4 in the candidate, while token embeddings change from Q3_K
to Q5_K. These are comparisons of storage encodings, not a claimed conversion
from the deployed Q3. The baseline records imatrix calibration metadata; the
candidate does not, which does not prove calibration was absent. Neither
bitwidth nor these CPU checks provide general quality proof or establish which
artifact is better. The publisher declares BF16 ancestry, but **exact BF16
lineage is not independently pinned**, including ancestral checkpoint and
conversion-tool revisions; absence of an earlier FP8 step is not independently
proven. No GPU load, fit/throughput/quality test, queue change, voice eviction,
context reduction or runtime promotion was performed by this audit. Parent-run
fit and paired task screens remain separate evidence; a synthetic pass is not
production-quality certification.

## NVFP4 Paired Runtime Screen

The 64K Q3 / NVFP4 / Q3 screen used physical batch 1,024, logical batch 4,096,
context 262,144, Q4 KV, MTP max 3 and resident LFM voice. It used the audited
candidate without a template override; prompt hashes match across blocks within
each task. Evaluated lengths: 64,183 infra, 64,048 Rust and 64,017 JavaScript tokens.

| Task | Q3 TTFT, before / after | NVFP4 TTFT | Q3 prefill tokens/s, before / after | NVFP4 prefill tokens/s |
| --- | --- | --- | --- | --- |
| Infrastructure | 22.30 / 22.59 s | 16.12 s | 2,881 / 2,845 | 3,989 |
| Rust | 22.00 / 22.29 s | 15.77 s | 2,915 / 2,877 | 4,070 |
| JavaScript | 22.23 / 22.36 s | 15.84 s | 2,884 / 2,867 | 4,049 |

Infrastructure TTFT fell 28.2% versus mean **same-batch-1,024** Q3 controls.
Against the initial production-batch-2,048 controls (21.15 / 21.90 s), the reduction
is approximately 25.1%, a separate cross-run comparison. Different output lengths
prevent interpreting total task times as an equal-output decode-speed benchmark.

Q3 left 4,743 MiB free at readiness; NVFP4 left 2,895 MiB. The separate
`5090-nvfp4-memory.csv` sample records 2,847 MiB free at **99% GPU utilization**.
This is one under-load observation, not the minimum across all workloads.
All six infrastructure responses passed. The file ends with `restored`,
`original_container=true` and `original_argv=true`.

The CPU-only sandbox grader accepted **12/12 Rust/JavaScript responses**, with
no paired failures: opening Q3, candidate and closing Q3 each passed two Rust
and two JavaScript outputs (cold and repeated). Images were
`rust:1.93-slim-bookworm` and `node:22-alpine`, with no network, read-only root,
dropped capabilities, no new privileges, UID/GID 65534, 512 MiB memory, one CPU,
64 PIDs and a 128 MiB tmpfs. Generated code was not executed on the host.

Grading provenance:

- Input `5090-nvfp4-screen.jsonl` SHA-256:
  `4af0486c2879c5abe3bbee0910e65de4e419102d1dfe3e1c365383585b17d7e3`.
- Grade output `5090-nvfp4-code-grades.json` SHA-256:
  `84650b7a0525c02052b613e5d14c69098df479f65a4ef3bf04b9fa8aaef4afda`.
- Grader source snapshot `benchmarks/grade_prefill_code.py` SHA-256:
  `5b9e56940372310cf1376447452547c492eadaf880ab640b4af9cdf6accb9028`.

These are only `parse_port` and `mapLimit` on synthetic 64K prompts: four distinct
answer hashes, not 12 independent tasks. They establish no general coding/reasoning
quality, production thinking-mode parity, vision quality or exact BF16 lineage.

## NVFP4 Long-Context Follow-Up

`5090-nvfp4-long.jsonl` completed the separate infrastructure-only comparison
at physical batch 1,024, with cached repeats disabled. All six cold responses
passed, with identical prompt hashes/counts across blocks (128,092 and 200,035
evaluated tokens), unchanged context/KV/MTP settings and resident LFM voice.

| Prompt | Q3 batch-1,024 TTFT, before / after | NVFP4 batch-1,024 TTFT | Q3 prefill tokens/s, before / after | NVFP4 prefill tokens/s |
| --- | --- | --- | --- | --- |
| 128K | 66.30 / 66.36 s | 53.34 s | 1,934 / 1,932 | 2,404 |
| 200K | 141.10 / 142.14 s | 121.32 s | 1,424 / 1,421 | 1,661 |

TTFT reductions versus mean **same-batch** controls were 19.58% and 14.34%.
Separately, the initial production-batch-2,048 controls were 60.99 / 62.23 s
and 130.82 / 132.52 s: cross-run reductions are 13.42% and 7.86%, not the paired
same-batch gains. No Rust/JavaScript or vision quality is claimed at these lengths.

Ready reserve remained 2,895 MiB for NVFP4 versus 4,743 MiB for Q3. The parent
200K load sample in `5090-nvfp4-long-memory.csv` records 2,847 MiB free at **100%
GPU utilization** (2026-11-11 13:52:19.963, source-local timestamp), not a peak-memory
bound. The final receipt confirms original-container/argument restoration; the
controller exited successfully with drain released and no renewal failure.

## Experimental Paths

- A disabled-by-default GDN value-slice scheduling experiment is kept outside
  the production patch manifest. CUDA 12.8 builds for SM86/89/120a succeeded.
  The bounded 64K off/on/off screen above shows no clear latency benefit.
  Build and screen success are not general numerical or dispatch qualification.
  See the separate GDN experiment and build notes.
- Pinned NVFP4 disk staging and CPU checks are complete as recorded above.
  The 64K paired-task and 128K/200K retrieval screens establish only the bounded
  fit/quality results above, not general quality certification. Exact BF16 lineage
  and vision remain unqualified; no production model was replaced.
- Approximate cache/prefill methods are not enabled. Published short-context or
  different-quantization results do not establish a quality-preserving gain for
  these long-context infrastructure and coding workloads.

No permanent runtime default changed. Memory tradeoffs block large-batch promotion;
GDN has no clear measured benefit, and NVFP4's limited quality/provenance evidence
does not justify a general fleet-wide recommendation.

## Authorized 5090 Deployment

After reviewing the above evidence, the user requested the 5090 switch to NVFP4.
This is a per-worker configuration change, not a fleet-wide default. Native FP4
is a Blackwell feature ([NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-blackwell-nim-blueprints-pc/));
the older 3090/4090 have neither the same native FP4 acceleration nor the same
32 GB memory budget. We did not change their quantization.

The pinned artifact above was hardlinked into the 5090 model directory, not
redownloaded or requantized. Its deployment basename is
`Qwen3.8-27B-NVFP4-mtp.gguf`, while the canonical source remains
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`. SHA-256 was reverified after staging.
Only `QUANT_TYPE=NVFP4,Q4_K_M` and `LLM_UBATCH_SIZE=1024` changed in the
container and persistent dotenv. The microbatch setting also applies to LFM;
its Q4_K_M weights, F16 projector/KV and 32,768 context remain intact.
Qwen retains its alias, BF16 projector, 262,144 context, Q4 KV, logical batch
4,096, MTP max 3 / minimum probability 0.1 and 32,000 MiB cache setting.

An idle router reservation excluded the worker during replacement and direct
canaries. The original pinned image, mounts, HostConfig and static network
configuration were verified unchanged. The first attempt's canaries passed,
but the current filename quant parser recognized `Q5K` and advertised that
instead of NVFP4. The controller rejected this mismatch and automatically
restored the original healthy container and dotenv. No partial deployment was
accepted. The deployment basename avoids that auxiliary-quant token: the
parser skips the unrecognized filename, then falls back to configured NVFP4.
This label is configuration-derived, not a tensor audit; the artifact hash
establishes that the qualified bytes are loaded. There is only one NVFP4 match
in the model directory, and local projector discovery stays enabled.

The second attempt passed thinking-mode infrastructure retrieval, a simple
color/shape image, and actual native realtime audio including cancellation and
an after-cancel response. First PCM was 153 ms; first nonquiet packet 174 ms;
the 5.6-second response contained 540 ms of leading quiet audio. These packet
timings are not perceived speech latency or representative distributions.
After a fresh router heartbeat advertised NVFP4 and both original contexts,
the drain was released and a pinned routed chat completed successfully.
The old stopped container was then removed. Idle free VRAM was 3,323 MiB.
All five workers were fresh and undrained, with Q3 unchanged on the other four.

Image: `sha256:b7e7d157c93827b563cbda3f1e331343b17d0984afb370363881624e3198689d`.
Rollback tag: `ezlocalai:rollback-env-nvfp4-5090-v2-20261111`.
Private recovery directory on the 5090:
`/home/josh/prefill-experiments/nvfp4-rollout-v2-20261111/transaction`.
To roll back, first acquire an idle owned router drain and renew it, then run
the matching `updater.py rollback <transaction-directory>` on that worker.
Verify restored engine arguments, health and router metadata before resuming.
The updater can recreate the original container from retained configuration
after finalization; do not delete its backup or rollback image prematurely.

The operator's quant-list validation/preservation/rollback tests pass (58 total).
No native image rebuild was needed. Compact evidence, receipt hashes and bounded
claims are in [the rollout fixture](../tests/fixtures/nvfp4_rollout_20261111_validation.json).
This canary adds thinking/vision/audio compatibility evidence, not broad quality
parity, multi-day stability or proof of better answers.

## Older GPU Follow-Up

The user requested more speed screening on the 3090s and 4090 after the 5090
deployment. These follow-up screens leave production configuration unchanged.
PR #153 records the earlier 5090 MTP decision; PR #154 preserves the tools and
evidence, not a required activation step for the already-applied NVFP4 setting.

### Rebuilt 3090 V Thinking Batch Check

`3090v-thinking-batch.jsonl` repeated baseline 512 / candidate 1,024 / baseline
512 with thinking enabled, greedy sampling, seed 42 and a 4,096 output cap.
Logical batch 4,096, context 230,000, original Q3 weights, Q4 KV, MTP maximum 2
and probability 0.1 stayed fixed. All six cold infrastructure responses passed.
This is thinking-enabled greedy validation, not the stochastic production profile.

| Native prompt tokens | Baseline TTFT, before / after | Batch 1,024 TTFT | Reduction versus mean control |
| --- | --- | --- | --- |
| 64,219 | 49.62 / 51.01 s | 47.96 s | 4.67% |
| 128,128 | 127.41 / 128.57 s | 119.56 s | 6.59% |

Native decode rates stayed around 68-70 tokens/s at 64K and 53.6-53.8 at 128K.
Output lengths differ, so complete-answer time is not a matched-length decode
comparison. Candidate ready-state free VRAM was 3,125 MiB versus 3,745 MiB for
the controls; this is not a peak-load bound. The exact original container and
engine arguments were restored, health confirmed and the owned drain released.
This corroborates the earlier roughly 6.65% non-thinking 128K improvement.

The durable candidate is `LLM_UBATCH_SIZE=1024` in the deployment manager's
per-worker settings for `xt-3090/ezlocalai`, preserving the measured logical
batch 4,096. The rebuilt machine has no dotenv file at its managed Compose path
and its launch also used a generated temporary override. We have not established
that editing the host's generated Compose file would survive a Deploy update,
so neither that file nor an ephemeral live container setting was changed.
Do not imply merging this benchmark PR applies the candidate automatically.

### 3090 A And 4090 Production-Profile MTP

Each worker ran maxima 3 / 2 / 4 / 3 with one 64K cold prompt and its cache-hit
repeat per block. Weights, context 230,000, logical/physical batch 4,096/1,024,
Q4 KV, projector and minimum draft probability 0.1 stayed fixed. Requests used
the worker's native Qwen profile: temperature 1, top-p 0.95, top-k 20, min-p 0,
presence penalty 0, repeat penalty 1, thinking enabled and effort `xhigh`.
Seed 42 and an 8,192-token output cap bounded the comparison. All 16 responses
passed and stopped normally. A generated 1,320-1,677 tokens; the 4090 generated
1,016-1,587. Neither truncated. Final-answer correctness does not imply identical
reasoning, general task quality or equivalence across seeds.

Decode rates below are pooled generated tokens divided by native decode time,
not unweighted means of individual per-request rates. Controls combine the
opening and closing blocks, with four requests versus two per candidate.

| Worker | Maximum 3 decode | Maximum 2 decode | Maximum 4 decode | Draft acceptance, 3 / 2 / 4 |
| --- | --- | --- | --- | --- |
| 3090 A | 69.43 tokens/s | 61.23 tokens/s | 67.78 tokens/s | 80.22% / 83.74% / 66.92% |
| 4090 | 116.40 tokens/s | 100.46 tokens/s | 92.15 tokens/s | 79.95% / 83.91% / 65.58% |

Retain maximum 3 on both. A's cold TTFT was 50.99/52.11 s for the controls,
50.32 s at 2 and 51.65 s at 4. The 4090 controls were 24.73/24.89 s, versus
24.82 s at 2 and 24.85 s at 4. These are not meaningful prefill improvements.
Cache-hit TTFT remained about 0.39-0.41 s on A and 0.09 s on the 4090. Cached
requests reused 64,215 prompt tokens and evaluated four new tokens. Lower
complete-response times on some candidates reflect shorter reasoning outputs,
not faster decoding of equivalent work. No new coding quality claim is made.

The benchmark now validates explicit canonical MTP arguments before changing
only draft maximum and probe port. The production profile is opt-in and recorded
in provenance; existing greedy defaults are unchanged. New calibration forwards
matching thinking/template kwargs. V and A were already bundled before that
calibration correction, so their template-token estimate is not authoritative;
use recorded native `prompt_n`/`cache_n`. All three runs evaluated 64,219 tokens
for their 64K cold samples; V's 128K samples evaluated 128,128.

All 22 follow-up responses, actual arguments, model hashes and restoration
receipts are summarized in
[older_gpu_speed_20261111_validation.json](../tests/fixtures/older_gpu_speed_20261111_validation.json).
All 140 focused `test_prefill*.py` tests and formatting checks passed. There were
no permanent settings changes or native image builds in these follow-up screens.

### Hardware Observations

The [telemetry fixture](../tests/fixtures/older_gpu_telemetry_20261111_validation.json)
records four read-only samples per host. Local J's upstream PCIe port supports
only Gen4 x4 although the GPU supports x16; A's upstream port is limited to
Gen3 x16. V showed Gen4 x16 under 99-100% utilization and its normal 450 W
software power cap, with no observed thermal slowdown. The 4090 was idle during
these snapshots; its idle Gen1/P8 state does not establish a problem. A later
single active A snapshot showed Gen3 x16, 86% utilization, 73 C, 428 W and
2,010 MHz SM clock, consistent with the platform link ceiling.

A better slot for J is worth investigating during planned hardware maintenance,
but no link-width change or causal inference-speed gain has been measured.
No clocks, BIOS settings, power limits, model precision or context limits were
changed. The earlier per-GPU batch results still apply; larger batches are not
uniformly better and J has the tightest memory reserve.
