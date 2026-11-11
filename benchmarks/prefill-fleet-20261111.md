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

## Experimental Paths

- A disabled-by-default GDN value-slice scheduling experiment is kept outside
  the production patch manifest. CUDA 12.8 builds for SM86/89/120a succeeded.
  Build success is not a numerical, dispatch or throughput qualification.
  See the separate GDN experiment and build notes.
- A pinned NVFP4 artifact is being staged outside the production model directory.
  Its published size is 15,705,922,304 bytes. Tokenizer, full embedded MTP, tensor
  layout, real memory fit, long-context behavior and task quality must be checked
  before considering it for the 5090. No production model was replaced.
- Approximate cache/prefill methods are not enabled. Published short-context or
  different-quantization results do not establish a quality-preserving gain for
  these long-context infrastructure and coding workloads.

No permanent runtime default is changed by this work. Pending experiments and
limitations must remain explicit until their device evidence is recorded.
