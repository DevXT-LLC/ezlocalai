The October 26 RTX 4090 cuBLAS prefill candidate is **rejected**: mean native
cold-prefill throughput falls **2675.44 → 2249.13 tok/s (−15.93%)**. Short decode
is flat. No diagnostic environment setting or CUDA library was promoted.
The compact [validation record](../tests/fixtures/cublas_prefill_20261026_validation.json)
contains the full source, artifact, request and report fingerprints.

## What was compared

The diagnostic CUDA selector prefers the engine's existing cuBLAS path for
supported dense NVIDIA Turing-or-newer matrices with at least 128 input rows,
when `EZLOCALAI_CUDA_PREFILL_CUBLAS=1`. Baseline phases use the original CUDA
library without that key. Candidate phases use only the reviewed CUDA library
overlay and that transient child environment override. The actual loaded
`libggml-cuda.so.0` pathname and its bytes were verified before inference.

Weights, projector, K/V types, context, batches, MTP settings, other inference
arguments and resource settings remain fixed. The worker is `720ee94`, using
llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab` and image
`sha256:f78bbd2139979beb58859f79c4c0e5c233919f43908be720eb5e25d266a108f9`.
The Qwen3.8-27B UD-Q3_K_XL weights have SHA `8c2a45ff85e7…`. Declared context
is 230000 tokens (engine allocation 230144), logical/physical batches are
4096/1024, K/V are `q4_0`, and there is one slot. Draft-MTP maximum is three,
minimum probability 0.1, cache ceiling 12288 MiB, with a 32-checkpoint limit and
8192-token minimum-step configuration. Requested sampling is temperature zero, seed 42, thinking disabled.
GPU clock, power-limit and unlock settings were not changed.

This selector **changes arithmetic**. The original MMQ path uses its Q8
activation working format; the existing cuBLAS fallback normally dequantizes
quantized weights into F16 inputs/compute under its existing precision rules.
No precision override was added, but unchanged weights and K/V types do not
establish exact arithmetic or general model-quality equivalence.

## Matched fresh-engine screen

The order was baseline, candidate, candidate, baseline. Each phase first ran
three short latency cases and three 256-token decode cases, all below the
128-input-token selector threshold, then one cold finite prefill. Warmup was
excluded. Every cold prefill evaluated **49991 tokens**, reused zero tokens and
answered `Python`. There were no inference retries or context reductions.

| Phase | Cold native prefill, tok/s | Short native decode median, tok/s |
| --- | ---: | ---: |
| Baseline 1 | 2680.32 | 104.12 |
| Candidate 1 | 2249.42 | 104.17 |
| Candidate 2 | 2248.85 | 104.11 |
| Baseline 2 | 2670.56 | 104.06 |

The mean of phase values is **2675.44 → 2249.13 tok/s prefill (−15.93%)** and
**104.09 → 104.14 tok/s decode (+0.05%)**. Each decode phase value is the median
of three requests generating 256 tokens. Mean cold finite-request wall time
increases **18.807 → 22.349 seconds (+18.83%)**; this includes completion and is
not a separate streaming TTFT measurement. The prefill regression is much larger
than the spread between the two bracketed baseline phases.

All **28 request/output/token signatures match** across the seven distinct cases
and four phases. That verifies the finite answer and sampled short outputs only.
This screen uses a different nonce and request from the 49987-token host-overhead
experiment; its roughly 104 tok/s decode must not be compared as a paired change
with that experiment's roughly 115 tok/s. Maximum context, vision, grammar,
cache returns and broad quality were not qualified for the rejected candidate.

Before hardware execution, the actual selector bodies passed **53222400 host
seam checks** across five compile modes and seven startup values. Hardware probes
were simulated in those checks. The owned-run controller passed **16 offline
guards**. Neither count is a CUDA-kernel correctness or model-quality result.

## Restoration and limits

The original production container was retained, temporarily stopped inside an
owned renewable router drain, then restarted after the owned test children were
removed. Exact container/image, environment, mounts, configuration, resource
settings and full original engine arguments were restored. Health and two fresh
unchanged router heartbeats passed before releasing the owned drain; lease
renewal recorded no errors. The existing rollback tag is
`ezlocalai:rollback-spec-overhead-4090-7a6e959d4a`; no new tag was created for
this screen. No production environment settings were promoted.

All four test cgroups recorded zero OOM events and zero swap peaks; this does
not establish that the whole host never paged. These are single-card results
on the already unlocked 4090, with no locked/unlocked comparison or causal unlock
claim. Other GPUs were not screened after this negative result.
