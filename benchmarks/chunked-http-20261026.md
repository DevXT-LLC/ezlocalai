# Incremental chunked HTTP framing — 2026-10-26

Large chunked bodies previously caused the worker to copy and decode the entire
accumulated body after every socket read. An incremental framing probe now keeps
offsets across reads and invokes the existing decoder once at completion. It
covers incoming chunked requests and buffered local-engine responses. The legacy
size parser, final decoder, malformed-input behavior, EOF handling and receive
limit are preserved.

This does not change ordinary Content-Length or SSE decoding, model parameters,
weights, context or KV precision. The buffered voice path exercised here is chat
with `voice`; standalone `/v1/audio/speech` uses a different response decoder.

## Primary comparison: quiet 5090 host

The primary measurements are the quiet-host rerun at 00:56 UTC on 2026-10-27
(2026-10-26 locally). Baseline `c421488` and final combined worker `bff136f1`
ran in separate CPU-only containers from cached image `8688fdf0…`, with no GPU
devices, read-only root filesystems, 4 GiB memory each and the same CPU affinity
(4–7). Both used the Echo backend and a loopback fake voice service. Artifact
hashes were `562ee7c3…` and `09b3112c…`; the unchanged tracked driver was
`48b2d220…`. The final runtime-file hash was `20867326…`.

The driver alternated variant order, using seven samples through 1 MiB and
three at 8 MiB, after one warmup per case and variant: 218 timed requests across
19 cases. Wall time includes connect, prepared-byte send, receive and close.
Incoming bodies go to `GET /health`, which ignores decoded input, to isolate
receive framing from JSON parsing and inference. The separate `native_http`
tests verify decoded chat input. Buffered response timing includes voice-enabled
chat and writing its local audio asset; retrieving and comparing the exact
binary asset bytes happens outside the timed interval. Initial host load
averages were 0.03/0.67/0.98.

This compares the pre-change baseline with the final combined worker, which
also includes request-gzip handling. No gzip requests are sent by this driver,
and the comparison does not isolate every change in that worker.

| Path | Body | Framing | Baseline median, ms | Candidate median, ms |
| --- | ---: | --- | ---: | ---: |
| Incoming | 1 MiB | 4 KiB chunks | 9.517 | 0.819 |
| Buffered response | 1 MiB | 4 KiB chunks | 13.597 | 4.308 |
| Incoming | 8 MiB | 4 KiB chunks | 1363.542 | 14.995 |
| Buffered response | 8 MiB | 4 KiB chunks | 1440.573 | 14.716 |
| Incoming control | 8 MiB | Content-Length | 3.478 | 3.444 |
| Buffered response control | 8 MiB | Content-Length | 13.900 | 13.424 |
| SSE control | Short Echo | Streaming | 0.404 | 0.440 |

All 19 cases passed their response-byte, asset or SSE checks. Fragmented valid
trailers, malformed terminators and incomplete EOF cases also passed. These
numbers establish a large chunked-body transport improvement; they do not
establish GPU prefill or ordinary cached-chat speedups. Small control differences
are not treated as gains or regressions.

The controller made 32 read-only observations of the original production worker
across the reviewed run, including the companion gzip phases. Its identity,
PID/configuration and process snapshot stayed identical. Queue size, processing
count and active requests stayed zero; cumulative queued/processed/failed
counters stayed 7/7/0. Cleanup verified no owned containers, child processes or
listeners remained, with no production mutations or recorded production actions.

## Earlier J run: exploratory and contended

The earlier run on J is retained in the fixture as exploratory history. J was
shared with another agent building/testing a game, so background contention was
not controlled. That run used the framing-only artifact `6954c20b…`, image
`b2c1fd79…`, CPU affinity (4,5) and initial load averages 3.22/3.11/4.72.

| Earlier J path | Body | Framing | Baseline median, ms | Candidate median, ms |
| --- | ---: | --- | ---: | ---: |
| Incoming | 8 MiB | 4 KiB chunks | 1376.460 | 19.630 |
| Buffered response | 8 MiB | 4 KiB chunks | 1571.909 | 21.144 |

All 19 earlier cases and their samples remain recorded. They are not the
primary measurements, and differences between the two runs are not evidence of
relative machine or GPU performance.

## Compatibility and package checks

- All 65,519 prefix classification checks matched the legacy decoder's complete,
  incomplete or malformed result, using the actual framing candidate class. The
  parser/decoder/probe block is byte-identical in the final combined source.
- The expanded real-TCP `native_http` tests passed against both baseline and
  candidate. They cover decoded chat input, binary buffered responses before
  upstream EOF, permissive legacy size lines/extensions/trailers, fragmentation,
  malformed timing, trailing bytes and the strict raw receive-limit boundary.
- The original framing change passed the exact oracle-free CUDA package gate
  70/70 in 289.28 seconds. The final combined worker passed the expanded exact
  gate 72/72 in 307.84 seconds and four focused tests in 54.49 seconds.
  The original native, Qwen TTS and ACE linkage checks found no missing libraries
  or libpython; ACE retained its private ggml dependency.
- The final worker was built in `b2c1fd79…` and run in cached image `8688fdf0…`.
  Their native-cache fingerprint, 172 resolved DSOs, 169 ELF build IDs, three
  hashes for DSOs without build IDs and 30 ordered direct dependencies matched.
  This is SDK/runtime compatibility evidence, not a fresh image build or GPU
  inference validation.

For an A/B comparison, use the reusable [driver](bench_chunked_http.py) with two
distinct, explicitly identified loopback Echo workers configured for the same
loopback fake voice port. It checks process start identity, listener ownership,
synthetic model/auth configuration and absence of external forwarding before
measurement. Run
`python3 benchmarks/bench_chunked_http.py --help` for its URLs/PIDs/output options;
it does not create workers or modify deployments.

Primary and exploratory samples, hashes, scope and cleanup evidence are in
[`chunked_http_20261026_validation.json`](../tests/fixtures/chunked_http_20261026_validation.json).
The quiet-host raw evidence is under
`build/validation/request-gzip-20261026/http-opt-in-5090-0386e2fdd0c6e4a1-evidence`
in the gzip worktree. Earlier logs remain under
`build/validation/chunk-probe-20261026` in the framing worktree.
