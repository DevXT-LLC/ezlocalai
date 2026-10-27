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

## Actual worker comparison

Baseline `c421488` and candidate source `22a24b68…` ran in separate CPU-only
containers from the same immutable image `b2c1fd79…`, with no GPU devices, a
read-only root filesystem, 4 GiB memory each and the same CPU affinity (4,5).
Both used the Echo backend and a loopback fake voice service. Artifact hashes
were `562ee7c3…` and `6954c20b…`; the tracked driver was `48b2d220…`.

The driver alternated variant order, using seven samples through 1 MiB and
three at 8 MiB. Wall time includes connect, prepared-byte send, receive and
close. Incoming bodies go to `GET /health` to isolate receive framing from JSON
parsing and inference. Buffered responses use voice-enabled chat and verify the
generated binary audio asset outside the timed interval. Initial host load
averages were 3.22/3.11/4.72; these are local transport measurements.

| Path | Body | Framing | Baseline median, ms | Candidate median, ms |
| --- | ---: | --- | ---: | ---: |
| Incoming | 1 MiB | 4 KiB chunks | 14.333 | 0.991 |
| Buffered response | 1 MiB | 4 KiB chunks | 17.802 | 4.019 |
| Incoming | 8 MiB | 4 KiB chunks | 1376.460 | 19.630 |
| Buffered response | 8 MiB | 4 KiB chunks | 1571.909 | 21.144 |
| Incoming control | 8 MiB | Content-Length | 4.270 | 3.972 |
| Buffered response control | 8 MiB | Content-Length | 16.507 | 15.755 |
| SSE control | Short Echo | Streaming | 0.317 | 0.309 |

All 19 cases passed their response-byte, asset or SSE checks. Fragmented valid
trailers, malformed terminators and incomplete EOF cases also passed. These
numbers establish a large chunked-body transport improvement; they do not
establish GPU prefill or ordinary cached-chat speedups. Small control differences
are not treated as gains.

## Compatibility and package checks

- All 65,519 prefix classification checks matched the legacy decoder's complete,
  incomplete or malformed result, using the actual candidate class.
- The expanded real-TCP `native_http` tests passed against both baseline and
  candidate. They cover decoded chat input, binary buffered responses before
  upstream EOF, permissive legacy size lines/extensions/trailers, fragmentation,
  malformed timing, trailing bytes and the strict raw receive-limit boundary.
- The exact oracle-free CUDA package gate passed 70/70 in 289.28 seconds.
  Native, Qwen TTS and ACE linkage checks found no missing libraries or libpython;
  ACE retained its private ggml dependency.
- Both owned benchmark containers were removed and all three listeners released.
  No production worker was changed for this comparison.

For an A/B comparison, use the reusable [driver](bench_chunked_http.py) with two
distinct, explicitly identified loopback Echo workers configured for the same
loopback fake voice port. It checks process start identity, listener ownership,
synthetic model/auth
configuration and absence of external forwarding before measurement. Run
`python3 benchmarks/bench_chunked_http.py --help` for its URLs/PIDs/output options;
it does not create workers or modify deployments.

Full samples, hashes, scope and cleanup evidence are in
[`chunked_http_20261026_validation.json`](../tests/fixtures/chunked_http_20261026_validation.json).
Bulky logs remain under `build/validation/chunk-probe-20261026` in the source
worktree.
