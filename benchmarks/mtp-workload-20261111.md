# RTX 5090 MTP workload screen

## Decision

Keep the current Qwen3.8-27B default: MTP draft length **3**, minimum draft
probability **0.1**. These tests did not establish a production speedup worth
promoting. No runtime code, fleet environment, quantization, context limit or
reasoning default changed. This is not a claim that length 3 is universally
optimal, or that MTP itself caused the observed answer failures.

The [validation record](../tests/fixtures/mtp_workload_20261111_validation.json)
contains the prompts, executable grading checks, per-response measurements,
failures and restoration receipt. All requests used synthetic data, not customer
conversations. Full local artifacts are retained in
`build/validation/mtp-workload-20261111/` (ignored).

## Conditions

- RTX 5090, driver 590.48.01, 575 W power limit; worker revision `485deb3`.
- Image `sha256:b7e7d157c93827b563cbda3f1e331343b17d0984afb370363881624e3198689d`.
- Same Qwen3.8-27B UD-Q3_K_XL weights; context 262144; Q4_0 K/V cache;
  batch 4096, microbatch 2048, flash attention on, one engine slot.
- Same minimum draft probability 0.1. Only draft length changed within each
  comparison; the off control used `--spec-type none`.
- Short inputs and approximately 58,300-token inputs containing archived,
  explicitly non-authoritative inventory paths. Prompt caching was disabled
  for the comparison; every measured response reports `cache_n=0`.
- Infrastructure diagnosis required the current checkout and published port,
  preservation of local edits, and no unnecessary Compose port changes.
- Rust required a checked port parser, including overflow and long leading-zero
  inputs. JavaScript required an ordered concurrency-limited mapper, including
  synchronous/asynchronous errors, empty inputs and invalid limits.
- Generated code ran only in network-disabled, read-only, non-root containers
  with no host mounts, dropped capabilities, CPU/memory/PID limits and a timeout.
  Known-good Rust and JavaScript fixtures passed the same grading harness.

The first screen used greedy sampling, no reasoning, a 1536-token output cap
and no resident voice engine. It is a microbenchmark, not the deployed request
profile. The second comparison loaded LFM3 alongside Qwen and used the worker's
actual Qwen reasoning profile from `llm_model_request_settings`: temperature 1,
top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1, and
`enable_thinking=true`, `reasoning_effort=xhigh`. Both sides used seed 42 and
an 8192-token output cap. These bounded requests are not full agent workflows.

## Greedy Screen

Throughput is total generated tokens divided by total engine decode time, not
an unweighted average of dashboard rates. Acceptance is total accepted draft
tokens divided by total proposed draft tokens.

| Draft length | Decode tokens/s | Draft acceptance | Task checks passed | Six-request elapsed |
| --- | ---: | ---: | ---: | ---: |
| 3, initial baseline | 161.25 | 84.37% | 5/6 | 70.23 s |
| Off | 55.86 | N/A | 5/6 | 88.67 s |
| 2 | 143.82 | 90.38% | 5/6 | 71.96 s |
| 4 | 177.87 | 80.98% | 5/6 | 69.23 s |
| 6 | 196.60 | 73.25% | 4/6 | 68.61 s |
| 3, final baseline | 162.05 | 84.37% | 5/6 | 70.63 s |

All settings failed the short JavaScript case. At length 6, the long JavaScript
answer also hung in an error-handling test. Its short answer additionally had an
unterminated Markdown fence. Complete JSON fences were accepted for functional
grading; strict JSON compliance is recorded separately. Each screen block had
4/6 strictly JSON-compliant responses.

Length 4 was about 10% faster at decoding than the repeated baseline, but its
total elapsed improvement was under 2% because cold prefill dominated. Length
2 had the highest acceptance rate and was slower than 3. Acceptance percentage
alone is not the optimization target. Different draft lengths can produce
different answers; the test does not establish bit-identical generation.

## Production-Profile Pair

| Draft length | Decode tokens/s | Draft acceptance | Task checks passed | Six-request elapsed |
| --- | ---: | ---: | ---: | ---: |
| 3 | 121.04 | 55.08% | 2/6 | 352.47 s |
| 4 | 118.85 | 45.20% | 3/6 | 344.46 s |

The geometric mean of the six matched per-task decode-rate ratios is **-0.54%**
for length 4. Aggregate decode throughput is **-1.81%**. The lower total elapsed
time is not evidence of faster execution of equivalent completed work: generated
lengths differ and most coding responses did not finish with usable answers.

Both infrastructure answers passed at both lengths. At length 3, three coding
responses hit the output cap and the long Rust response stopped without final
answer content. At length 4, short Rust completed and passed; the other three
coding responses hit the cap. This small, single-seed sample does not establish
that length 4 improves model quality. It cannot establish a quality-preserving
speedup, either. The planned second pair was not run: the operator stopped the
benchmark after the first pair because it failed the speed-improvement gate.
The next engine startup was interrupted, not a production user request.

Two earlier greedy/reasoning pilots with 4096 and 8192 output-token caps were
also stopped after truncation. Their five recorded responses remain in the
evidence as pilots and are excluded from both comparison tables. The final
pair above uses the actual production sampling profile, not those pilots.

## Interpretation And Limits

- MTP already provides a substantial decode benefit in the greedy control.
  Turning it off is not a justified fleet optimization from these results.
- A longer draft's apparent gain on simple outputs did not carry over to the
  actual reasoning profile. Neither length 4 nor 6 is promoted.
- MTP primarily targets decode work. It does not remove cold prefill or router
  queue time. The long cold prompts still took approximately 19 seconds to read.
- The incomplete reasoning answers warrant separate bounded-completion and
  task-quality evaluation. They do not justify silently lowering reasoning
  effort or claiming a production regression from this benchmark.
- No probability-threshold sweep, DFlash comparison, full-context stress test,
  cache-reuse qualification, other-GPU tuning or broad quality evaluation was
  performed. Results must not be generalized to the 4090 or 3090 workers.

## Restoration

The worker was acquired through an owned idle router drain with a renewable
300-second lease. Temporary engines listened only on host loopback and reused
the immutable image and read-only model mounts. An independent watchdog and
the driver's `finally` handler retained a path to restore the original container.

All four started runs restored that exact retained production container.
The final check verified its original image and complete Qwen engine argv,
healthy status, no remaining benchmark containers, and no active router drain.
A routed request pinned to DevXT5090 returned HTTP 200 and `OK` in 0.19 seconds.
Other workers were not restarted. No replacement image was built, so no new
rollback tag was needed; the original container/image remained the rollback.
