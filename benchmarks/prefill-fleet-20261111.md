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
calibration metadata. Cross-machine speed is therefore not a pure GPU comparison.
Every experiment records its own file hash and engine arguments.

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
