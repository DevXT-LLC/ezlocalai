# Negotiated text-request gzip (2026-10-26)

Large conversation histories can spend substantial time crossing the worker
link even when the model reuses its prompt cache. The router can now send a
losslessly compressed request to a worker that explicitly advertises support.
The native worker decodes the same JSON bytes before ordinary request routing.
This changes transport only: model arguments, weights, precision, presets,
context limits and inference behavior are unchanged.

## Protocol and rolling deployment

Compression applies only to direct local POSTs to `/v1/chat/completions` and
`/v1/completions`, after the router has produced its final JSON, including
`disable_fallback`. Requests must be between 64 KiB and 16 MiB. Level-1 gzip is
used only if it saves bytes including its extra header; smaller and
incompressible requests retain their original bytes. External providers,
tunnels, media/vision requests and multipart forms keep their existing paths.

Router request gzip is **opt-in per worker** through
`EZLOCALAI_ROUTER_REQUEST_GZIP=true`; the default is `false`. Enable it only for
a worker whose constrained link benefits from the extra codec work. The native
decoder remains available to authenticated direct clients regardless of this
setting. An opted-in worker advertises
`extra.http_request_encodings: ["gzip"]`; otherwise it reports an empty array.
Both registration and every heartbeat report the current setting independently
of optional media-policy probes. The router
requires that **every live direct record for the same effective callback URL**
contains the literal array value `gzip`. It checks this before encoding and
again immediately before sending. URL comparison uses the existing URL parser,
retains path/query and does not change the actual callback URL. Missing or
invalid capability updates conservatively revoke only this new metadata key
from older records sharing that endpoint. An upgrade can therefore wait for an
old record to expire before compression activates. A rollback detected while
encoding selects the retained plain body for the first and only send. Compressed 400/413/415 responses do not trigger an uncompressed resend;
existing route error handling is unchanged. Normal drain/cutover procedures
still cover a deployment change after the final capability check.

The two-job router codec limit holds its permit inside the blocking closure.
Cancelling a request releases its request reservation but retains codec
capacity until that job ends. Compression starts after ordinary admission.

The native decoder runs after authentication on the two supported endpoints.
It accepts exactly one complete gzip member and validates its CRC and trailer,
rejecting malformed/truncated streams, concatenated members and trailing
bytes. Encoded and decoded lengths each have an independent 16 MiB limit.
Malformed gzip returns 400; an exceeded gzip limit returns 413. After decoding,
all `Content-Encoding` and compressed `Content-Length` header variants are
removed so fallback forwarding receives the decoded JSON and fresh framing.
Unsupported encodings, unrelated endpoints and plain requests retain existing
behavior. Original client authentication, routing/cache hints and the
`disable_fallback` guard remain covered by tests.

The legacy socket reader's raw-buffer limit includes headers and is unchanged.
The exact inclusive **encoded-body** 16 MiB boundary is tested at the codec
level; real-TCP tests do not claim acceptance at that wire edge. Real-TCP
coverage includes the declared encoded limit plus one and exact decoded
limit/limit-plus-one behavior.

## Measurement method

The fixture contains 160 complete MIT-licensed files (1,048,397 source bytes)
from the public native repository at
`c42148868a04d05171cdfc3bca7840f227b97216`. Every entry and the license were
independently matched against that commit's exact Git blobs. The archive and
manifest have bounded decoding and pinned SHA-256 fingerprints. No private
repository source, private conversation or repeated filler is included.

Four conversations select 32,768, 131,072, 524,288 and 1,000,000 source bytes,
split into user/assistant turns. The smallest case stays below the 64 KiB
request threshold and is a plain/plain control. The default script mode is explicitly a
**Python stdlib gzip/zlib diagnostic**, not an isolated Rust/C++ codec timing.
The route mode uses the actual candidate Rust router and native worker with a
deterministic loopback mock engine and a CPU-only owned container. The same
worker ID, explicitly opted-in worker environment and executable pair are used
for both variants; only the registered gzip capability changes outside each
timed request. The benchmark sets the flag once before startup, without
changing shared defaults or its parent environment.

Each link mode runs six ABBA blocks for buffered and streamed versions of all
four cases: 192 measured requests, 16 excluded warmups and 12 samples per
variant/case/mode. The shaped link paces request-body upload from its first byte
at 13,000,000 bytes/second. The unshaped loopback control has no artificial
sleeps; both modes retain the proxy and mock-engine overhead. This simulates a
link constraint and does **not** measure production Wi-Fi, GPU token rate or
model quality. Host scheduling/load can change absolute latency between runs.

Every sample verifies the exact final decoded router JSON, engine-bound raw
payload, expected answer and normal finish, with exactly one relay request and
one mock-engine request. Each sample checks `disable_fallback` and compares
final decoded/engine byte identity across variants; targeted router/native
tests cover the routing/cache hints, including hints consumed by the router.
Relay gzip/JSON/hash validation runs on the main thread only after the client
timer stops, avoiding relay-thread verification in the timed Python GIL window.
The mock engine still parses/hashes its identical input during each request.
Only sanitized lengths, hashes and metadata enter saved reports.

The harness binds only loopback listeners, proves ownership before readiness
probes, uses an immutable cached image with `--pull=never`, and supplies no GPU
or production-data mounts. Generated credentials pass through explicit
subprocess environments, with Docker key-name-only `--env` flags; values are
not written to dotenv files, argv or reports. Parent environments are
unchanged. Each run removes only its owned container and stops its owned
router. No worker fleet or production router is changed.

Final timing runs use the quiet 5090 host, with every owned benchmark process
and container pinned to CPUs 4–7 and the read-only production monitor on CPUs
0–1. All 32 before/during/after observations confirm the original worker remains
idle: cumulative queued/processed/failed counters stay **7/7/0**, all active/queue
counts stay zero, and original container/config/mounts, worker/engine PIDs,
start times and full argument fingerprints stay identical. Docker health is
checked where present. The original model stays resident and available: no
production inference, maintenance POST, lease, drain, restart or environment
change is performed by the benchmark. Arrival of real work invalidates and
stops the whole run; no samples are silently filtered or retried.

Earlier J timing runs are retained only as exploratory data after the user
identified concurrent game work there. **None of those timings is used below.**
The final benchmark worker, router and SDK fingerprints are fixed across both
link modes; no fresh image is pulled or built.

## Paired route results

Both final link modes pass all 192 measured requests and 16 excluded warmups:
**384 measured requests and 32 warmups total**. Every final decoded router
request, engine-bound payload and checked output hash matches across variants,
repetitions and link modes. The actual remote operator has stopped; zero owned
child processes or containers remain and every owned listener is released.

The largest buffered final request shrinks **1,029,190 → 380,814 bytes**
(63.0% fewer body bytes). Under the simulated 13 MB/s upload, its median full
response falls **105.453 → 59.532 ms** (43.5%); streamed first checked content
falls **105.487 → 60.444 ms**. On unshaped loopback, that buffered response
instead rises **22.183 → 28.519 ms**, and streamed first content rises
**21.786 → 28.378 ms**. This is a constrained-link tradeoff, so automatic router
compression remains **off by default** and is enabled per worker only.

All rows are medians in milliseconds from this final quiet-node run. Advertised
transport uses gzip only when eligible; the 32 KiB source control remains plain.

### Unshaped loopback control

| Source bytes | Mode | Final JSON bytes | Advertised wire bytes | Plain full | Advertised full | Plain → advertised first content |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 32768 | buffer | 33,533 | 33,533 | 1.074 | 1.081 | — |
| 32768 | stream | 33,532 | 33,532 | 2.238 | 2.193 | 2.177 → 2.135 |
| 131072 | buffer | 133,482 | 62,204 | 2.736 | 3.867 | — |
| 131072 | stream | 133,481 | 62,204 | 3.897 | 4.961 | 3.819 → 4.925 |
| 524288 | buffer | 538,407 | 202,805 | 10.494 | 14.098 | — |
| 524288 | stream | 538,406 | 202,805 | 10.857 | 14.491 | 10.808 → 14.441 |
| 1000000 | buffer | 1,029,190 | 380,814 | 22.183 | 28.519 | — |
| 1000000 | stream | 1,029,189 | 380,814 | 21.995 | 28.512 | 21.786 → 28.378 |

### Simulated 13,000,000 B/s body upload

| Source bytes | Mode | Final JSON bytes | Advertised wire bytes | Plain full | Advertised full | Plain → advertised first content |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 32768 | buffer | 33,533 | 33,533 | 4.665 | 4.476 | — |
| 32768 | stream | 33,532 | 33,532 | 5.690 | 5.674 | 5.641 → 5.537 |
| 131072 | buffer | 133,482 | 62,204 | 14.632 | 10.319 | — |
| 131072 | stream | 133,481 | 62,204 | 15.672 | 11.406 | 15.629 → 11.367 |
| 524288 | buffer | 538,407 | 202,805 | 56.339 | 30.547 | — |
| 524288 | stream | 538,406 | 202,805 | 54.614 | 30.548 | 54.582 → 30.515 |
| 1000000 | buffer | 1,029,190 | 380,814 | 105.453 | 59.532 | — |
| 1000000 | stream | 1,029,189 | 380,814 | 105.531 | 60.478 | 105.487 → 60.444 |

The actual Rust encoder produces 380,814 bytes for the largest case. The
Python diagnostic is a different encoder/input boundary and must not replace
actual wire sizes or be described as isolated Rust/C++ codec CPU cost.

## Qualification

The combined chunk-framing and default-off gzip worker at frozen runtime commit
`bff136f19aa86dc3d31bf02c113f0b327ece5ccc` passes four focused codec, gzip-TCP,
native-HTTP and metadata suites in **54.49 seconds**, then the exact updated
**72-test CUDA package gate in 307.84 seconds**. Reference environment variables
are unset and the configured Python reference path is absent. All 13 copied
source/package/document hashes match the frozen commit in both host and
builder. Metadata tests cover default off, explicit true/false, registration,
heartbeat and two live revocations, including media-probe failure. The real-TCP
gzip suite proves decoding still works while advertisement is off.

Both shipping gzip tests are in CMake package targets and all four platform
Dockerfile gates. This worker was rebuilt in an owned CPU-only container from
existing image `sha256:b2c1fd792f29205ab20fdea8a7c756549537a31a78b658f89d8a1b2922599be6`.
The quiet-node runner reuses cached image
`sha256:8688fdf01700b01b2d620988bb4652cfb38b529e793840b1de965f0fbe47e8b5`:
its SDK cache, all 172 resolved library paths, 169 GNU build IDs, three full
hashes for libraries without build IDs and 30 ordered DT_NEEDED entries match
the builder exactly. The mounted final worker resolves every dependency.
No fresh Docker or other-platform image was built for this qualification.

The paired router at `def8e4e7e8dd947bfd651670ddbb1fed81b87314` passes formatting,
strict Clippy, **78 Rust tests** (two existing manual benchmarks ignored) and
**573 local-fake Python integration tests in 418.874 seconds**. Its 14 new
raw-TCP compression cases also pass against the release binary in 2.596 seconds;
these are a repeated subset, not 14 additional unique integration tests.
Targeted cases cover auth/byte identity, external/media/tunnel exclusions,
threshold/incompressible controls, rolling old/new workers, stale affinity and
late positive heartbeat records, peer expiry, rollback during encoding, no
plain resend after error, and reservation/codec-permit cleanup on cancellation.

All **15 benchmark guards pass in 1.581 seconds**, including constant explicit
worker opt-in, public-corpus provenance/bounds, strict decoding, UTF-8/source
budgets, first-byte pacing, unshaped control, listener ownership, credential
handling, raw gzip/auth forwarding and verification after the client timer.
The ignored read-only controller and operator pass **16 and 14 offline guards**,
including contamination detection, PID-reuse protection and the atomic
startup/cleanup fence. Their exact reviewed source/plan hashes are recorded.

These benchmark runs make no production deployment or persistent environment
change. There is no deployment rollback tag for this PR-only qualification.
Old routers keep sending plain requests, and opted-out workers keep reporting
an empty encoding list. Normal deployment drain/rollback still applies if a
live worker changes after the final capability check. GPU inference throughput,
numerical model quality and production V Wi-Fi latency are not measured here.

## Separate staged setting on DevXT3090V

Only the comment and `EZLOCALAI_ROUTER_REQUEST_GZIP=true` were appended to
`/home/josh/ezlocalai/.env` on DevXT3090V for its **next deployment**. All 675
previous bytes, ownership and mode were preserved. The existing healthy c421
container environment, mounts, worker/engine PIDs and arguments are unchanged;
there was no restart, model operation or drain. This is not live gzip activation
or a measurement of production V throughput.

The exact backup is
`/home/josh/.local/state/ezlocalai-benchmarks-20261026-b8e1eddc923e410b/.env-router-gzip-backup-20261026-59ea36fc049d46e4/.env.before`
(0600 inside a 0700 archive outside the deployment checkout). Its mode, inode
and exact previous bytes were preserved when archived. Before full-file rollback, verify the current
file still matches staged SHA-256
`ca7cebc25d9ec1a50784e46cbc122d496d34b5f745de2285e0a2df1d88f544b1`.
If newer edits exist, review a fresh fingerprint and remove only the exact,
uniquely matching appended comment/setting block, preserving every other byte.
Stop if that block changed or became ambiguous. Compact evidence records the before/staged fingerprints and
sanitized operational proof separately from the benchmark.

## Reproduction and fingerprints

Run the 15 local guards with
`python3 -m unittest discover -s benchmarks -p test_bench_request_gzip.py`.
The default `python3 benchmarks/bench_request_gzip.py --output build/validation/request-gzip-codec`
runs only a Python codec diagnostic. Actual-route mode additionally requires
`--run-route`, an immutable `--image`, explicit `--worker-binary` and
`--router-binary`, `--blocks 6` and separate output directories. Run once with
`--bytes-per-second 0` and once with `--bytes-per-second 13000000`.
The final quiet-node runs additionally use the reviewed ignored wrapper for
fixed CPU sets, owned-resource cleanup and read-only contamination checks.

| Artifact | SHA-256 |
| --- | --- |
| Final native worker | `09b3112cf0825136495326ac6c0f2238396ad92643bf1bd99ab11d9091c65f26` |
| Router release | `db315055ad6b94bca125debd8e0576feda3cbca60e61fb27bcca1ea680559942` |
| Benchmark script | `8dc3a1f45e65002e8f0eed50e136b8b9038e16a3b17d418e71bf3101d7a7139f` |
| Public-source manifest | `d0c2e793c738b731eba7fb550edbfc609210d67d567abe980f89cde11d52efb4` |
| Public-source archive | `d0e8f46579d4d2d670a4abd7e75f8f5b1c5c1557b93976d237064a8d34537772` |
| Final unshaped raw report | `291655a3015580a7d639800855e6286dbbff59cca93f8f854ee8c3e5eebebed5` |
| Final shaped raw report | `dccf312ae5fe67b9014e2c6cc6b1156b21703e52e7d2002e8ecbe3e030e54826` |
| Final controller proof | `3ffbdf74b1237400060ed81e3d716788fde956e6736e2fc8a9e0ea56fbbff57b` |
| Final 72-test package log | `175ef2368943816b80d041facc2cf4971319b4bac356032de4173279490214e2` |

[Compact validation evidence](../tests/fixtures/request_gzip_20261026_validation.json)
records all 16 timing rows, eight exact request/engine/output identities,
read-only contamination/cleanup results, staged-setting rollback metadata and
raw artifact fingerprints. Final fetched reports remain ignored under
`build/validation/request-gzip-20261026/http-opt-in-5090-0386e2fdd0c6e4a1-evidence/`.
Final build manifests/logs are ignored under
`build/validation/http-request-gzip-20261026/combined-opt-in-bff136f19aa8/`
in the combined native qualification worktree. Remote audit artifacts remain
outside every deployment checkout in the exclusive 0700 temporary stage.
