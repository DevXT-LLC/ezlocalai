# Stream transport experiment — 2026-10-26

The worker wrote every HTTP chunk using three socket sends: its hexadecimal length, the SSE payload, then CRLF. The stream response socket retained the default Nagle setting. The new helper submits each complete chunk together, retries partial/interrupted sends and requests TCP_NODELAY for chunked response writers and upstream streaming connections. HTTP framing, SSE data, inference settings and chunk boundaries are unchanged.

A controlled loopback experiment requested delayed ACKs with `TCP_QUICKACK=0` after reading the response headers. It measured the interval from generating the first SSE event until receiving that complete event. Each variant had 31 samples.

| Chunk writes | TCP_NODELAY | Median first complete event |
| --- | --- | ---: |
| Three sends | Off | 40.777 ms |
| Coalesced | Off | 0.0082 ms |
| Three sends | On | 0.0159 ms |
| Coalesced | On | 0.0077 ms |

Every received frame was byte-identical. This experiment isolates a delayed-ACK transport stall; it does **not** measure model TTFT or claim a GPU throughput gain. Deployed worker/router measurements must establish the application benefit.

The standalone regression test covers UTF-8/NUL framing, empty chunks, partial/interrupted writes with real signals, disconnect handling and TCP_NODELAY read-back. Restoring the previous EINTR behavior makes the interrupted-write test fail; the fixed helper passed five repeated checks. No timing threshold is imposed by the package test.

Reproduce without inference dependencies:

```sh
g++ -std=c++17 -O2 -pthread -I include benchmarks/bench_stream_socket.cpp -o /tmp/bench_stream_socket
/tmp/bench_stream_socket 31
```

After configuring the native project, the optional `bench_stream_socket` CMake target builds the same benchmark. Compact samples and source hashes are in `tests/fixtures/stream_socket_20261026_validation.json`.
