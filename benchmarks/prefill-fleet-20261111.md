# Long-Context Prefill Qualification

2026-11-11. These are isolated experiments, not a production configuration change.
The initial sweep is still being extended. Do not interpret a passing retrieval
screen as proof of general coding quality or mathematical equivalence.

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
- Use identical prompts and weights within each worker's bracketed sweep. Greedy
  seed 42, thinking disabled, 1,024 output tokens. This is a controlled screen,
  not a complete reproduction of the production thinking workload.
- Later Rust/JavaScript checks execute generated code only in network-disabled,
  read-only, resource-limited disposable containers. They do not execute it on
  the host. Their results are recorded separately from retrieval grades.

Commands and provenance are retained in the JSONL evidence under the ignored
`build/validation/prefill-fleet-20261111/` directory. The checked-in compact
fixture is generated with `summarize_prefill_fleet.py` and omits answer text.

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
prompts. Baseline batch 2,048 leaves 3,381 MiB free. An intermediate batch is a
separate candidate, not a measured result yet.

Disabling MTP does not improve total task time here. At 128K, baseline completes
in 65.12-66.38 s versus 76.57 s without MTP. At 200K it is 136.10-137.72 s versus
148.30 s. The MTP-off comparison bounds total MTP overhead; it does not attribute
that entire difference to host transfers or establish a device-copy speedup.

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

## Experimental Paths

- A disabled-by-default GDN value-slice scheduling experiment is kept outside
  the production patch manifest. CUDA 12.8 builds for SM86/89/120a succeeded.
  Build success is not a numerical, dispatch or throughput qualification.
  See the separate GDN experiment and build notes.
- Pinned NVFP4 disk staging and CPU checks are complete as recorded above.
  Runtime fit, long-context behavior and task quality remain unqualified by this
  audit. No production model was replaced.
- Approximate cache/prefill methods are not enabled. Published short-context or
  different-quantization results do not establish a quality-preserving gain for
  these long-context infrastructure and coding workloads.

No permanent runtime default is changed by this work. Pending experiments and
limitations must remain explicit until their device evidence is recorded.
