# Qwen3.8 long-context prefill investigation, 2026-10-05

Local RTX 3090 Ti (24 GB), Qwen3.8-27B Q3_K_XL, 220,000 configured
context, Q4 KV, xllamacpp 2026.9.10809 with the existing DFlash image hotfix.
The frozen source is `Pipes.py` from `b5de518`, repeated/truncated to 720,000
characters. Native input is **158,616 tokens**. Full measurements and output
hashes are in [the JSON report](qwen38-prefill-20261005.json).

## GPU experiments

One isolated native process per configuration, full GPU offload, greedy sampling,
16 output tokens, unchanged weights/context/KV precision. These are single runs,
not statistical confidence intervals. Other media models were stopped.

| Backend | Batch / ubatch | Cold prefill | Cold tokens/s | 5,774-token cached extension |
|---|---:|---:|---:|---:|
| MTP | 4096 / 512 | 223.13 s | 711 | 13.11 s |
| MTP | 1024 / 1024 | 209.82 s | 756 | 14.49 s |
| MTP | 4096 / 1024 | 224.13 s | 708 | 13.55 s |
| DFlash2 | 4096 / 1024 | 209.15 s | 758 | 11.27 s |

The larger microbatch did not consistently improve MTP. GPU samples reached
about 23.9 GB used with the larger MTP graph, before adding the serving worker's
other resident models. Retain the existing automatic MTP/512 policy on the
24 GB, 220K setup. DFlash remains opt-in; these measurements do not justify a
fleet-wide backend switch. The unchanged-prompt baseline reused 158,612 tokens
and evaluated four, so caching is functioning. Q4/Q4 Flash Attention is supported
by the pinned native build; enabling all mixed-quant kernels is not the fix here.

## Avoidable repeated input

The retained 5090 history showed six consecutive transitions where the next
cached-token count was exactly the previous input count minus **6,287**.
That pattern is consistent with a repeated suffix after changing history;
request contents were not available to prove the identity of the suffix.
WorkConductor's coding continuation builder independently had this pattern:
seven fixed guidance paragraphs followed the growing execution evidence.

The companion WorkConductor change moves those complete instructions into its
stable prefix, preserving current steering, recovery feedback, requested-outcome
checks, repository observations, and response controls at the tail. A brief
tail reminder refers back to the full instructions. No context is discarded.

A real HTTP streaming comparison used the actual seven guidance constants
(8,588 bytes) and the same source prompt. The running worker selected batch
2048 / ubatch 512; both layouts used that same worker and configuration.

| Follow-up | Guidance after history | Guidance in prefix |
|---|---:|---:|
| Short follow-up: evaluated tokens | 1,531 | 30 |
| Short follow-up: prefill / wall | 4.32 / 5.13 s | 1.43 / 2.09 s |
| Large follow-up: evaluated tokens | 7,283 | 5,774 |
| Large follow-up: prefill / wall | 16.39 / 16.75 s | 13.40 / 13.69 s |

All eight HTTP responses ended with `[DONE]`, without stream errors. The prefix
layout's initial request reused the previous layout's base; its `cold` row in
the raw HTTP fixture is a **layout prime**, not a cold-prefill comparison.
The savings are in repeated continuations. Fully uncached 150K+ prompts still
require substantial GPU computation. Actual 4090/5090 speedups remain unmeasured;
those workers and the production router were inspected without modification.

Reproduce with `benchmark_prefill.py`, a frozen `--prompt-file`, and the same
instruction text supplied via `--guidance-file`. Compare `--guidance-layout
trailing` against `--guidance-layout prefix`. The JSON includes hashes of both
input fixtures. Keep the serving worker stopped for standalone native sweeps.

## Incomplete transfer errors

The four retained DevXT3090J incomplete-transfer errors belonged to old worker
IDs around previous rebuilds. None belonged to the worker active at the start
of this investigation. Docker's ten-second shutdown deadline can terminate
in-flight generation before Uvicorn finishes draining. The CUDA Compose service
now defaults to a ten-minute graceful-stop window. Forced termination, crashes,
and requests exceeding that window can still interrupt streams.

The live shutdown test sent SIGTERM during a 1,024-token response at 166K
context. Compose waited **29.01 seconds**; the client received `[DONE]`, the
request finished without a transfer error, and the container exited with code 0.
