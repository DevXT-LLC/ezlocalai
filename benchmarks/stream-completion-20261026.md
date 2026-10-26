# Stream completion experiment — 2026-10-26

The native stream queue previously waited on callbacks or cancellation for up to 100 ms. Producer completion notified the condition variable, but the predicate ignored the `done` flag, so the consumer could continue waiting for the timeout. This delayed finalization and release of a single-slot admission lease.

The predicate now includes completion. Existing queued callbacks still drain first, followed by the held/implicit final event or the existing error path, and admission remains held until exhaustion. Cancellation and bounded close behavior are unchanged. This intentionally removes Python Queue.get's completion polling delay; a transport keepalive is no longer needed after native completion has already been observed.

A controlled replay emitted one assistant callback, held the producer until the consumer was waiting, then ended generation without another callback. Timing began immediately before the producer returned and ended in the admission-release callback. Both variants used the actual queue implementation and unchanged native normalizer/parsing code, with the same native/core archives and CPU-only container.

| Queue predicate | Median completion → exhaustion/admission release |
| --- | ---: |
| Previous callback/close predicate | 90.031 ms |
| Completion-aware predicate | 0.05263 ms |

There were 31 samples per variant. The producer gate waited 10 ms before allowing completion, leaving roughly 90 ms of the former 100 ms timeout. This measures completion/next-request admission overhead, **not** model TTFT, prefill or engine decode speed.

The model-free `stream_completion` package test checks final/callback/error ordering, exact-once admission release, empty diagnostics, owner abort and unstarted close. These contract checks pass for both variants; the latency improvement is measured separately, without a fragile CI timing threshold. The parent integration must also run the existing 4720 stream helper/generator cases, 96 managed API cases and lifecycle/ASGI traces.

After configuring the native project:

```sh
cmake --build build --target test_stream_completion bench_stream_completion
ctest --test-dir build -R '^stream_completion$' --output-on-failure
build/bench_stream_completion 31
```

Samples, source hashes and linkage provenance are in `tests/fixtures/stream_completion_20261026_validation.json`. No production API or GPU was used.
