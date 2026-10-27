The October 26 follow-up separates workload reporting from two engine host-work
optimizations. Model weights, K/V precision, context, sampling, batch sizes and
MTP limits remain fixed. Removing host work is a candidate speed improvement;
its hardware results must be assessed against the repeated baseline.

## Why the dashboard and short benchmarks differ

The router uses evaluated prompt tokens, excluding reused tokens, divided by
engine prompt time. Four newly evaluated tokens in 100 ms are 40 tok/s even
though the cache avoided evaluating thousands of tokens. Cache-hit tokens/s is
therefore unsuitable as a standalone cache-effectiveness measure. Prompt time,
reused/evaluated counts and actual first-output latency answer different questions.

The prior quiet 4090 benchmark measured 111.75 tok/s decode on short inputs.
A read-only snapshot of retained production history at 2026-10-27 01:29 UTC
shows 130 cache-hit requests averaging 130,577 input tokens and 59.50 tok/s
aggregate decode. Four cold requests average 109,657 input tokens and
63.34 tok/s aggregate decode; their aggregate prefill is 2144.18 tok/s.
These are mixed historical observations, not matched before/after measurements
or a full-day sample: the history contains only 500 retained requests and spans
multiple changes. They explain why the historical rate cannot be compared
directly with a short-context benchmark. No hardware acceleration is claimed
from reaggregating the existing samples.

WorkerRouter PR #8 adds separate cache miss/hit timing rows, evaluated/reused
counts, average engine prompt time and summed-token/summed-time throughput,
with recent windows and retention limitations. Existing history and accounting
remain intact.

## Engine changes

The sixth managed llama.cpp patch avoids cloning the sampler on verification
rounds where the largest possible rollback fits the existing partial/recurrent
rollback capacity. Cases needing checkpoint replay retain their original sampler
snapshot. It also reads the dense unmasked MTP hidden-output buffer once per
process hook instead of repeatedly synchronizing every backend through per-row
getters. The same row values, position metadata and snapshot format are used.
See [the patch rationale](../patches/llama-speculative-overhead.md).

The actual-source regression compiles the managed process method and sampler
snapshot selector against an instrumented backend. Its 21,051 checks cover call
counts, exact state bytes, multiple sequences, embedded/shared-KV/chained-head
paths and rollback boundaries. Reintroducing either removed behavior makes the
guard fail. This test establishes neither GPU speed nor broad model quality.

## Completed 4090 screen: no promotion

The fresh-engine order was baseline, candidate, candidate, baseline. Every cold
request evaluated all 49,987 tokens and returned `Python`. Each phase also ran
three short first-output cases, three 256-token decode cases and twelve
conversation seed/replay/append/rewind cases. All **76 request/output/token
signatures match**, with unchanged model/projector bytes and actual engine
arguments. Both overlaid libraries were verified in the engine's loaded mappings.

| Phase | Cold prefill, tok/s | Short decode median, tok/s |
| --- | ---: | ---: |
| Baseline 1 | 2669.77 | 115.47 |
| Candidate 1 | 2656.91 | 115.52 |
| Candidate 2 | 2653.46 | 115.55 |
| Baseline 2 | 2651.42 | 115.28 |

The mean of phase values is **2660.60 → 2655.19 tok/s prefill (−0.20%)** and
**115.37 → 115.54 tok/s decode (+0.14%)**. These changes do not establish a useful
speedup; the candidate is **not promoted**, and V confirmation was not run.
The earlier 49,993-token benchmark uses a different request and is not a paired
baseline for these measurements.

The worker ran revision `720ee94`, image
`sha256:f78bbd2139979beb58859f79c4c0e5c233919f43908be720eb5e25d266a108f9`.
The retained original container was restored with the exact image, environment,
mounts, resource configuration and complete engine argv. Health and two fresh
router heartbeats passed before the owned drain was released. Rollback tag:
`ezlocalai:rollback-spec-overhead-4090-7a6e959d4a`. No permanent settings changed.
The compact [validation record](../tests/fixtures/speculative_overhead_20261026_validation.json)
contains full artifact hashes and the comparison signature fingerprint.

A separate lead appeared in the rewind workload: it reused 77 tokens from a
25,758-token seed when rewinding to 24,508. The existing narrow aligned policy's
near-end checkpoints are after the shorter prompt, according to source-derived
scheduler positions. This is a coverage limitation rather than evidence of an
evicted conversation. Broader coverage requires its own quality and memory
qualification and is not part of the host-overhead measurements above.
