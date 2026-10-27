# ezlocalai parity work

The goal remains behavior parity with ezlocalai within the scope below, with all first-party executable code implemented in C++. It is **not complete**.

## Benchmark relay response-header validation (2026-10-26)

CodeQL findings 59 and 60 identified unvalidated upstream header names/values
in the gzip benchmark's loopback relay. It now validates the entire response
header list before committing a status line or headers, rejects CR/LF and
invalid field syntax, and forwards valid values unchanged. The relay continues
to replace hop-by-hop/framing headers itself; no warning suppression is used.

All **18 benchmark guards** pass, including invalid names/values, valid field
preservation and a real folded-header response that must close without sending
any response bytes. Python compilation and whitespace checks pass. Native and
router runtime code, deployments and GPU settings are unchanged. Historical
timings keep the original measured script fingerprint; this follow-up does not
claim a fresh performance measurement. Compact evidence is in
`tests/fixtures/request_gzip_headers_20261026_validation.json`.

## Lossless HTTP transport and CUDA speed checks (2026-10-26)

Incoming chunked requests and buffered local responses now track framing
incrementally instead of repeatedly copying and decoding every accumulated
prefix. The final legacy decoder, permissive size parsing, malformed/EOF timing
and raw receive limit are retained. All **65,519 prefix classification checks**
match, with real-TCP decoded-chat, binary-response and fragmentation coverage.
Content-Length and ordinary SSE paths retain their existing decoding.

The worker also strictly decodes authenticated, single-member gzip JSON on the
two text completion endpoints, enforcing independent 16 MiB encoded/decoded
limits and validating CRC/trailers. Transport headers are removed before normal
routing/fallback. Automatic use by WorkerRouter is **opt-in per worker** through
`EZLOCALAI_ROUTER_REQUEST_GZIP` (default false): every registration/heartbeat
reports the current policy, including revocation and media-probe failures.
The paired router preserves final decoded bytes and existing admission,
fallback and cache hints; old workers, external providers, media and tunnels
retain their existing transport. Codec jobs have a bounded cancellation budget.

Final combined worker source `bff136f19aa86dc3d31bf02c113f0b327ece5ccc`
passes all **72 oracle-free CUDA package tests** in 307.84 seconds and all four
focused suites in 54.49 seconds. All 13 selected source/package hashes match the
builder; the final executable is `09b3112c…`. The configured Python checkout is
absent. The paired Rust implementation passes 78 Rust tests, strict Clippy and
formatting, and 573 local integration tests, including 14 direct wire cases.
Public benchmark inputs are complete files pinned to public native commit
`c421488`; no private repository corpus is published.

The transport reports separate controlled measurements from exploratory local
runs: [chunked HTTP](benchmarks/chunked-http-20261026.md) and
[request gzip](benchmarks/request-gzip-20261026.md). Gzip is deliberately off
by default because codec overhead can outweigh transfer savings on fast links.
Shaped upload measurements are simulations, not deployed Wi-Fi latency.

On the quiet 5090 CPU host, 8 MiB/4 KiB-chunk receive latency falls
**1363.542 → 14.995 ms**, and buffered responses **1440.573 → 14.716 ms**.
For the largest public-code JSON, the actual paired route under simulated
13 MB/s upload falls **105.453 → 59.532 ms**, while unshaped loopback increases
**22.183 → 28.519 ms**. This measured tradeoff motivates the opt-in default.
All 384 measured gzip payload/checked-output identities match; all 19 chunk
cases pass. Read-only monitoring records unchanged original production
counters/processes/configuration, and all owned benchmark resources are gone.

Fresh CUDA measurements use native token timings and preserve model precision,
sampling and context. With production weights, V's bracketed 49,993-token cold
prefill averages **1418.14 tok/s** at physical batch 1024; 2048 is flat and 4096
regresses, so no batch setting was promoted. The 4090 measures **2704.00 tok/s**
on that prompt and **111.75 tok/s** median decode. Common-model replay checks
reuse 49,989 tokens and evaluate four, with matching sampled answers. J results
are explicitly exploratory: the user confirmed concurrent game build/testing,
and desktop, driver, PCIe and paging differences prevent a clean unlock
comparison. J's dense diagnostic was skipped. V's standalone GEMM numerical
checks passed, but clock-plausibility acceptance failed and remains recorded as
such. No causal unlock percentage is claimed. See the
[CUDA report](benchmarks/unlocked-cuda-20261026.md) and its compact fixtures.

All GPU-test originals were restored and their owned drains released after
fresh router heartbeats. The 13.44 GB temporary V model copy was removed; private
operator records and the exact dotenv backup were archived outside the checkout.
After testing, only V's dotenv gained `EZLOCALAI_ROUTER_REQUEST_GZIP=true` for
its **next deployment**; its running c421 container stayed unchanged. Other
workers keep the default-off policy. No weights, KV precision, GPU clocks,
power limits or unlock settings changed. A remains API-only; its user-reported
1024 dotenv value is not an effective-process verification.

Post-merge validation still needs real routed V histories/cache returns after
both services deploy. A clean same-card before/after unlock experiment and a
quiet J baseline remain separate work; the contended samples cannot replace
them. Fresh platform images and non-CUDA hardware were not validated here.

## Long-conversation request preparation and dispatch (2026-10-26)

Queued input preparation now retains ordered outer fields and normalized
messages instead of repeatedly parsing, copying and serializing the entire
history. Duplicate-key values and output positions, input errors, model-load
order, media callbacks and the present-audio-format helper boundary remain
covered. Model weights, presets, KV precision, context limits and inference
arguments are unchanged by this code.

Four matched actual-function runs in ABBA order give 14 sample batches per
variant/case. One-MiB text preparation falls **13.242 → 3.753 ms** and a
256-message history falls **13.000 → 2.760 ms**. All seven complete prepared
payloads and helper-event hashes match baseline. In separate actual-worker HTTP
tests with a deterministic loopback mock engine, 336 measured requests and 28
excluded warmups preserve every engine-bound raw payload and the checked answer
content/finish state. A roughly 1.13 MB, 256-message request reaches first content
in **32.044 → 23.086 ms**; its buffered response falls **32.494 → 24.905 ms**.
The complete post-credential-fix rerun preserves all 14 request/engine/output
identities and the worker binaries from the earlier run. Absolute latency varies
with ambient host load; small-case gains are inconsistent, so timings are not
selected across runs or generalized to every short request. These are CPU/HTTP
overhead measurements, not GPU token-rate or model-quality measurements.

Validation: all **six strict Python-reference suites** pass, including 3,399
input captures and new escaped-key/null-audio guards; **14 HTTP benchmark guards**
pass; all **70 CUDA-image package checks** pass in 307.93 seconds with the
configured Python checkout absent. The new explicit `llm_input_package` keeps
all captured expectations and managed checks, while strict developer mode still
requires source hashes. Missing/corrupted captures and an absent strict reference
are negative controls. Four platform recipes select the oracle-free gate.
The benchmark now passes generated credentials through key-name-only Docker
`--env` flags and explicit subprocess environment mappings, preserving
authentication without temporary credential files, argv values, log exposure or
parent-environment mutation. Actual HTTP measurements were repeated after this
CodeQL correction.

The first package run found a pre-existing media parent-death fixture race, with
the old test executable identical to production. The fixture-only generation
handshake correction passes 100 fresh repetitions; disabling the parent-death
signal still fails. Production media runtime sources are unchanged. Candidate
executables were rebuilt inside an owned CPU-only container from existing CUDA
image `sha256:845c0d4edad3ee960bb22e76a48de843560d4a197f6a430e799e3136b8a032f4`,
with unchanged engine/dependencies. No fresh Docker or other-platform image,
physical GPU inference or multi-GPU behavior was qualified by these dispatch
checks, and no candidate was published as latest or deployed by them.

Separate matched 3090J physical-batch checks complete with all 24 finite
controls correct and all 12 paired request/answer hashes equal. Batch 1024
reduces rewind first output **18.628 → 1.184 s** by reusing 22613 tokens;
replay improves **648 → 470 ms** and append **1.192 → 0.891 s**. The roughly
1.2% cold-prefill median difference is within observed spread. Both batches
fully evaluate 208866 cold tokens at the unchanged 220000 context and pass
red-image recognition, with no retries, smaller-context path or observed OOM.

**Live environment change:** only DevXT3090J's `LLM_UBATCH_SIZE` is now `1024`
in its `.env` and running container, replacing absent/`auto` (effective 512).
Image, all remaining engine arguments, context, presets, precision, host limits,
environment and mounts are preserved. Fresh drained/resumed router reporting,
pinned chat, streaming and vision pass; no owned drain remains. The original was
retained through checks and then finalized. Rollback image is
`ezlocalai:rollback-env-j-env-20261026-05d665ee`; private backup is under
`/tmp/ezlocalai-env-update-j-env-20261026-05d665ee/` (0700/0600). New request
code is PR-only; the live setting uses existing merged image `845c0d4…`.

See [the dispatch report](benchmarks/dispatch-speed-20261026.md) and
`tests/fixtures/dispatch_speed_20261026_validation.json` for exact binaries,
sampled quality limits and remaining work. Native PR #142 contains this worker
optimization. WorkerRouter PR #6 has its separately validated performance
changes; another agent is adding model-matching fixes there.

## Cache state, RAM capacity and router admission (2026-10-26)

The optimization PR now includes transactional target/draft snapshot save and
restore, bounded reuse of initialized host buffers, shared immutable checkpoint
bytes, and position-aware MTP CPU carryover across cache, checkpoint and disk
boundaries. Missing or malformed CPU state invalidates carryover before target
prefill. Failed GPU restores evict the entry and clear the active contexts;
incomplete saves never publish a cache entry. Snapshot capacities, spare buffers
and CPU metadata remain charged against the configured hard cache budget.

Capacity notifications follow actual request/resource release, including TTS
websocket jobs, rather than relying only on periodic heartbeats. Linux capture
pipes use atomic close-on-exec. Three private real-Rust-router/full-CUDA-worker
pairs on the 4090 measured completion-to-next-output **2.925 s -> 177 ms**;
selection wait fell **2.758 s -> 9.7 ms**, while post-selection inference stayed
about 167 ms. This removes router waiting; it does not increase GPU token rate.

Same-build storage off/on on the 4090 reduced returning-turn first output
**456 -> 252 ms** and append **685 -> 517 ms**, with all 12 requests and outputs
identical. Native decode throughput was effectively unchanged. The final aligned
checkpoint candidate preserves the existing late checkpoints and adds at most
one earlier snapshot on a narrowly qualified single-slot, single-head MTP text
path with target/draft physical batch 1024. The active 4090 comparison preserves
all nine seed/replay/append answers and reduces shortened-prefix rewind first
output **8.064 -> 0.590 s** (23252 -> 724 evaluated tokens); full 256-token turns
fall **10.796 -> 3.326 s**. Rewind wording differs, so this is not byte equality
for rewritten prompts. All 24 finite turn checks across coverage off/on return
the exact requested branch and new turn and finish normally.

The earlier fixed-offset checkpoint trial changed cold answers and was rejected.
Projector-loaded text classification, borrowed MTP model ownership and the
parser's inactive NONE entry each needed actual-runtime regression coverage;
intermediate inactive trials are retained as diagnostics, not speed evidence.
The final planner is tested with actual target/draft splitters and actual CLI
parsing, rather than assuming the configured mode vector or microbatch layout.
Other physical batches, slots, model/head layouts, media and unsafe boundaries
retain the original checkpoint scheduler.

**Live environment change:** `/home/josh/ezlocalai/.env` on 4090 and 3090V now
contains `LLM_PROMPT_CACHE_MAX_RAM_FRACTION=0.4`, resolving to **12288 MiB**
instead of 7680. On unchanged production engines, eight independent ~24.6K-token
conversations over three rounds changed returning hits **0/16 -> 16/16**.
First output fell **8.726 -> 0.462 s** on 4090 and **16.451 -> 0.748 s** on
3090V, with all 24 request/output pairs identical on each host, no observed swap
or OOM, and verified original restoration before rollout. The rollout preserved
images, all other engine arguments and mounts, then passed fresh router metadata
and pinned chat/stream/vision checks. The existing UBATCH settings remain
1024/1024/2048 on V/4090/5090; weights, KV precision, contexts, presets and MTP
lengths are unchanged. No swap allowance has been enabled.

Rollback images are `ezlocalai:rollback-env-cache-env-20261026-devxt4090-c16abb26`
and `ezlocalai:rollback-env-cache-env-20261026-devxt3090v-c16abb26`. Private
0600 dotenv backups and saved original inspect data remain in the corresponding
0700 `/tmp/ezlocalai-env-update-cache-env-20261026-<label>-c16abb26/` directories.
The finalized stopped originals were removed after verification. The reviewed
[recovery operator](benchmarks/operators/README.md) supports recreating an owned
finalized transaction from its immutable image and private backup; its 47 guards
are offline evidence, not a live rollback test.

The 3090V release comparison reduces replay first output **752 -> 304 ms**
and rewind **15.565 -> 0.918 s**, preserving all nine seed/replay/append answers.
On 5090, replay improves **319 -> 195 ms**, with all 12 sampled answers identical;
its physical batch 2048 already has a useful earlier checkpoint. Across the
three cards, all 48 finite resident-turn controls pass. All three also pass disk
save/restore with RAM cache disabled, exact new-turn answers after stomping the
slot with another conversation, matching restored prefix/byte counts and vision.
Named cold capacity screens fully evaluate **218392/218392/248809 tokens** on
4090/V/5090, followed by correct red-image recognition, with no truncation,
smaller-context retry or cache reuse. These are single cold capacity screens,
not long-context replay or long-decode quality measurements.

An isolated 5090 paging study keeps all 14 returning cache hits and all 21
answers identical with an 8192 MiB cache, 6 GiB RAM and 8 GiB swap limits. It
uses about 5.32 GiB of swap on the existing NVMe partition; replay first output
rises **197 ms -> 2.263 s**, and full turns **1.183 -> 3.240 s**. This measures
end-to-end paging pressure, not page-attributed swap-in time. A preceding
6 GiB RAM + 4 GiB swap limit was too small for the cache plus runtime overhead
and OOM-killed only the isolated test process; its original worker restored
healthy. No host swap settings or production swap allowances changed.

Validation: **69/69 oracle-free CUDA package checks** on the complete final
image, **87 benchmark guards**, **47 recovery-operator guards**, **six focused
cache/planner/strict-source checks**, and **2/2 actual tiny-engine cache and
Rust-router integrations**. The complete image was built on 5090 from
`448bf6b` as `ezlocalai:cuda-candidate-20261026-448bf6b`, image
`sha256:462cd19b65435d0ff95153b8aed32d91e5a121b8e78df2fbe4d37c39ec28bb57`.
Fresh native/TTS/ACE/llama linkage passes, with ACE's own GGML and no missing
libraries or libpython. Its full-image CUDA canary uses the original public
model/presets and a private real Rust router, without worker/engine overlays.
Chat, streaming, vision, CUDA GPU metadata, RAM/VRAM, context and capacity
reporting pass; the exact original container/image/environment/mounts restore
before drain release. The retained rollback is
`ezlocalai:rollback-canary-20261026-full-canary-fdb6add414`. Earlier candidate
images remain historical held experiments. Later documentation and recovery-tool
commits do not change the tested production C++ or engine patch sources.

All three production workers are fresh/healthy on their existing `46d9ba8`
images with no owned drains. Only the documented RAM settings are live; the
new worker/engine code is PR-only and no candidate image was published as latest.
The 5090 still uses different deployed GGUF contents from V/4090; results compare
each card with its own baseline. Greedy rewritten-prompt wording can differ
through cache numerics; finite controls and sampled semantic checks do not
establish universal model-quality equivalence. Other hardware images, physical
multi-GPU, video and arbitrary deeper/longer multi-conversation cache workloads
were not qualified in this optimization pass.

Handoff: merge the optimization PR through the existing deployment automation,
then verify fresh deployed versions, model/context/capacity/GPU/RAM reporting,
selected RAM/UBATCH settings and interleaved routed conversations. Keep the
original weights, precision, contexts and presets. Broaden checkpoint eligibility
or optional swap only after matched workload and quality measurements. See
[the cache report](benchmarks/cache-depth-20261026.md) and
`tests/fixtures/cache_depth_20261026_validation.json`; older entries below are
historical and do not describe the current PR scope.

## CUDA prefill qualification and streaming latency (2026-10-26)

The worker now sends a complete HTTP chunk in one write, retries interrupted
writes, requests TCP_NODELAY for streaming sockets, and wakes its consumer when
generation completes. Response bytes, queued callback order, cancellation and
admission release remain covered by regression tests. Controlled delayed-ACK
and producer-completion stalls fell from 40.78/90.03 ms to below 0.1 ms; live
model A/B runs showed no consistent material TTFT or decode improvement, with
all 25 sampled outputs matching on each of the three GPUs.

Fresh measurements start from deployed worker `46d9ba8` and engine `620d2225`.
Paired 150K-token cold prefill improved from **952.8 to 1016.9 tok/s** on
3090V (RTX 3090 Ti), and **1781.4 to 1894.8 tok/s** on 5090. Their larger
physical batches also passed cold 218392/248809-token prompts and red-image
recognition without reducing their 230000/262144 context budgets. Weights,
KV precision, logical batch, MTP settings and model presets were preserved.
The 3090V's two capped summary probes changed wording without a concrete
semantic regression; 6/8 long outputs and all 16 short outputs matched exactly.
The 5090's eight long and 16 short outputs matched. These are sampled quality
checks, not universal numerical equivalence. The 5090 has different actual
GGUF contents despite the same filename, so comparisons use each card's own
baseline. Full progression and provenance are in
[`benchmarks/optimization-20261026.md`](benchmarks/optimization-20261026.md).

Applied only `LLM_UBATCH_SIZE` in `/home/josh/ezlocalai/.env`: 3090V's absent
setting (`auto`, effective 512) is now `1024`; 5090's `1024` is now `2048`.
4090 remains `1024`: its 2048 trial reduced 150K prefill from 1988.3 to
1820.1 tok/s, so further qualification was intentionally stopped and its
original container restored. State-transfer batching, pinned staging, graph
tuning and longer MTP drafts were not promoted; both experimental state-I/O
patches were removed from the final code.

Settings cutovers retained the existing images (`3e09d9905d59` on 3090V and
`1197b0b18d41` on 5090), mounts, ports and effective resource controls. Docker's
default `OomKillDisable` null/false representation was normalized for comparison.
Fresh router metadata and pinned chat, streaming and vision checks passed;
all three workers are healthy and no owned drains remain. Rollback tags are
`ezlocalai:rollback-env-env-20261026-devxt3090v-5903c2f8` and
`ezlocalai:rollback-env-env-20261026-devxt5090-5903c2f8`; private dotenv backups
remain beside each operator transaction on its host. Two earlier operator
verification attempts rolled back safely before the successful cutovers.

Validation: **62/62 CUDA package checks** without the Python checkout,
**2/2 real CPU cache/Rust-router integrations**, **23 benchmark guards**, and
per-card isolated CUDA measurements. Compact evidence is in
`tests/fixtures/llm_throughput_20261026_validation.json` and the linked model
and streaming fixtures. No complete new production image or other hardware
image was built in this pass; the new worker code ran in isolated A/B containers.

Handoff: merge the optimization PR, let image automation build/deploy it, then
repeat routed smoke and interleaved conversations on the complete image.
Position-aware MTP CPU carryover across cache/checkpoint rewinds and atomic
failed-draft cache restoration remain source-audited follow-ups; no partial
MTP state fix ships here. Sustained cache pressure and actual swap-in latency
also remain unmeasured. Preserve the current weights, precision and context
when benchmarking those changes.

## Oracle-free cache package gates (2026-10-26)

The optimization merge added `llm_cache_reference` to the image gate, but that
strict test reads the external Python checkout. Images do not contain it, so
production builds failed at source-hash validation. The earlier 60-test run
used a configured oracle and did not cover this packaging condition. The
ROCm speculative-decoding gate had the same checkout dependency.

Package gates now use explicit fixture-only modes while the strict developer
reference tests retain source-file and hash verification. All 9,111 recorded
cache/RAM cases, 2,500 speculative cases and their managed startup/restoration
checks still run. Missing fixtures, incorrect expected results, and missing or
modified source in strict mode fail explicitly. HIP builds select the HIP
fixture driver, and cache tests clear inherited GPU masks. Runtime inference,
cache retention, memory budgets and defaults are unchanged.

Validation: all 60 CUDA package checks passed with the configured reference
path absent (292.55 s). Both package modes passed in a Docker build on the
5090 with networking disabled and neither a reference checkout nor the local
oracle present. This test-only image extends the existing CUDA base with rebuilt
test binaries; it is not a complete production image build. Its image ID and
base digest are in `tests/fixtures/cache_package_gate_20261026_validation.json`.
Strict reference, missing/tampered fixture and missing/tampered source checks
passed their expected positive/negative outcomes. The HIP cache discovery
simulation passed with and without inherited GPU masks.

No production container was replaced and no production image was published.
Complete CUDA/GB10, CPU, ROCm and Jetson image builds and hardware deployment
remain unverified by this fix. Handoff: merge the build-gate correction, rerun
the existing image automation, and confirm the reported GB10 build advances
past the package gate before performing the optimization canary.

## Conversation-cache retention and CUDA optimization (2026-10-25)

The worker dropped WorkConductor's `prompt_cache_key`, and the pinned engine
could discard conversation tails when prompts shared most of their prefix.
The worker now forwards valid chat hints. A managed, SHA-recorded patch to
llama.cpp `620d2225` snapshots named conversation switches, restores the
longest useful prefix, and protects the requested state during bounded
eviction. It skips an outgoing save when two states cannot fit; cache limits
remain hard ceilings. Custom engine revisions retain their own behavior.

On DevXT3090J's RTX 3090 Ti, the same full Qwen3.8-27B/MTP three-conversation
replay improved returning turns from a median **19.04 s to 0.916 s**. Evaluated
tokens fell from roughly 23,800 to **4**, with identical branch answers and
the same 14336 MiB cache ceiling. A tiny real CPU engine and model-free
eviction/prefix tests also pass. The worker's long-request preparation now
parses fields once, improving the 707 KB CPU case from 9.65 to 3.53 ms while
preserving all three measured engine payloads byte for byte.

Automatic prompt-cache requests now grow toward the safe quarter-RAM ceiling
(at most 32 GiB) on hosts with at least 64 GiB. RAM and free swap respect all
visible cgroup-v2 ancestors. Optional `LLM_PROMPT_CACHE_SWAP_MIB` permits a
bounded allowance from existing free swap, with a separate reserve; it is off
by default. Explicit limits and disabled caches are preserved. All six compose
profiles pass the new settings through. Original Python policy helpers still
pass all 9,111 recorded cases; the native extensions have deterministic tests.

Unlocked-card cold-prefill medians at roughly 68K/150K tokens were
**2529/1970 tok/s on 4090**, and **1246/939 on 3090V** (an RTX 3090 Ti).
4090 is approximately 8–9% above the historical October 18 measurement,
but the engine changed, so the unlock's isolated benefit is unknown. No
comparable pre-unlock 3090V measurement was found. A single 3090V physical-batch
1024 trial reached 1014 tok/s at 150K; defaults remain unchanged pending
repeat/max-context/vision checks. See `benchmarks/optimization-20261025.md`.

Validation: the 60-test CUDA package gate passed, plus request/CLI tests,
the real CPU cache replay, and the Rust-router/native-worker integration.
The latter verifies buffered/streaming cache-key delivery on direct/tunnel
paths. Compose and benchmark guard checks passed. WorkerRouter PR #4 also
preserves the hint for local chat (its adapter previously stripped it) and
shows RAM beside VRAM. WorkConductor already sends the hint; no client issue
was confirmed.

All benchmarked production containers were restored with their original
image, environment and mounts; owned drains were released. No optimized
image was built or deployed, so no replacement/rollback tags were created.
The CUDA cache replay overlaid the patched engine implementation in an
isolated container from the existing production image. Actual swap-in latency,
sustained larger-cache pressure, other hardware images, and fleet rollout
remain unverified. Compact evidence is in the three dated optimization JSON
records under `tests/fixtures/`; bulk logs remain in ignored `build/validation/`.

Handoff: review/merge the paired router and worker changes, let the existing
image automation build CUDA/GB10, then canary the complete image through the
router before fleet rollout. Recheck interleaved long conversations at 68K and
150K and larger RAM ceilings; only opt into swap after measuring restore
latency. Repeat the 3090V ubatch candidate before considering a default change.
Video, PDF and unrelated parity work are outside this optimization change.

## Brotli decoder security packages (2026-10-24)

CPU, CUDA/GB10, ROCm, and Jetson image recipes now install checksum-pinned
upstream Brotli 1.2.1 after their distro dependencies. Ubuntu 22.04/24.04 still
supply affected Brotli versions while CVE-2026-111497 is under evaluation.
The font dependency stages also install the patched package, and the storyboard
installer overlays Pillow’s hashed Brotli library copies with the patched build.
Pinned FreeType, HarfBuzz, PNG, and font bytes remain unchanged.
The shared installer builds real replacement `libbrotli1` and `libbrotli-dev`
packages (1.2.1-0devxt1), preserving system library paths, SONAMEs, headers,
static archive names, and package ownership. It runs the upstream CTest gate
before installation and verifies the installed pkg-config version.

Validation: native amd64 library/package builds pass on Ubuntu 22.04 and 24.04,
plus an emulated Ubuntu 24.04 arm64 build, with 12 upstream tests each.
Dynamic/static encoder and decoder version checks, libcurl linkage, and
`dpkg --audit` pass. `tests/brotli_security_smoke.c` also checks the replaced
wheel decoder and renders a glyph with the pinned FreeType/font libraries.
Image IDs and source checksum are in
`tests/fixtures/brotli_security_2026-10-24_validation.json`.

Full worker/engine package gates, CUDA/GB10/ROCm/Jetson hardware execution, and
fleet rollout were not performed. No deployment or rollback tags were created.

## Default voices (2026-10-20)

The repository did not include Python's default voices: `voices/` was
excluded by both `.gitignore` and `.dockerignore`, so `GET /v1/audio/voices`
on a fresh checkout or image listed none, and requests for `default` or a named
voice failed unless a Python voices folder was mounted.

- Added the 22 WAVs from the Python repository's `voices/` (commit `0f7e9f4`).
  Git blob hashes match, so the files are byte-identical.
- They are force-added; `voices/` stays in `.gitignore`, so uploaded voices
  remain untracked.
- `voices/` is no longer in `.dockerignore`, so images ship the defaults at
  `/app/voices`, as Python's image does. A compose mount still takes
  precedence.

Not yet validated: a rebuilt image has not been run with these voices.

## Repository restructure and rename (2026-10-20)

The `native/` directory is gone. Its contents now live at the repository
root, and every `ezlocalcpp` reference is renamed to `ezlocalai`, because this
repository will replace the Python ezlocalai repository.

**Layout:**

- Sources, tests, CMake and Dockerfiles are at the root.
- Images copy the tree to `/app`, as the Python image does: the build is
  `/app/build` and llama.cpp is `/app/third_party/llama.cpp`.
- The CMake runtime root is the source directory, so `.git` version lookup
  reads `/app/.git` in images and the checkout root on hosts.
- The Python reference default stays the sibling `../ezlocalai`.
- The compose gate now has a dependency-free configure:
  `-DEZLOCALAI_COMPOSE_GATE_ONLY=ON`.

**Names:**

| Before | After |
| --- | --- |
| `EZLOCALCPP_*` build, CMake and runtime settings | `EZLOCALAI_*` |
| `ezlocalcpp:native-<platform>` image tags | `ezlocalai:<platform>` |
| container name `ezlocalcpp` | `ezlocalai` |
| LAN registry `devxt/ezlocalcpp-<platform>` | `devxt/ezlocalai-native-<platform>` |

None of the 76 settings collides with an existing `EZLOCALAI_*` name. Two now
share Python's names and meaning: `EZLOCALAI_CUDA_IMAGE` and
`EZLOCALAI_GB10_IMAGE`.

**Transition rules:**

- During cutover, a host whose Python `.env` points those variables at the
  Python image must override them with the native image, as the README's
  shell exports do.
- Both workers use container name `ezlocalai`, so cutover removes the Python
  container with `down`.
- The registry keeps a `-native-` infix so native images never overwrite the
  Python fleet's `devxt/ezlocalai-{cuda,gb10}:latest`. Those names are taken
  over only when the Python images are retired.

**Hashes re-pinned.** `tests/tts_process_fixture.cpp` and
`include/ezlocalai/tts_unicode.hpp` are hashed by their tests, and the rename
changed them. The recorded hashes were updated in `llama_tts_process_cases.jsonl`,
`gtts_cases.jsonl.gz` and `ctts_text_cases.jsonl.gz`. Historical evidence text
was renamed mechanically; image IDs, test counts and conclusions are unchanged.

**Validation:**

- Development container: all 163 tests configure and build. Every suite passes
  except four that already failed in that container for environmental reasons
  (offline `model_download`, three root-permission CTTS cases). Two tests that
  read compose defaults through the old parent path were corrected.
- The compose gate passes all 1,875 settings.
- See `AGENTS.md` for the contributor guide.

## Production readiness and session update (2026-10-20, after the checkpoint below)

**Readiness estimate vs Python,** weighted by what the fleet runs (six x86 CUDA
text/vision workers including one tunnel worker, plus the GB10
multi-capability worker):

| Area | Ready | Basis |
| --- | ---: | --- |
| Router protocol (registration, heartbeat, tunnel, failover, router fallback) | ~95% | Live fleet field comparison, ten real-Rust scenarios, production tunnel run, 802 s tunnel media request; no long soak |
| Text and vision LLM | ~95% | Python-identical response/SSE shapes, faster prefill and cache reuse on v0.7.0, context-overflow statuses |
| Embeddings, Qwen TTS, STT, image, music | ~85% | Python-equivalent I/O on x86 CUDA; GB10 hardware unverified |
| Wake word | ~85% | CPU/CUDA reference training, export and cancellation |
| Video and music-video | ~70% | Default-size and larger two-scene runs pass through Rust; quality, presets and multi-GPU unverified |
| x86 CUDA image | ~95% | Deployed; portable CPU code (no AVX-512/VNNI in any engine) |
| GB10 image | ~65% | 54/54 emulated gate, library resolution and endpoint smoke; no GB10 execution |
| CPU, ROCm, Pi, Jetson (not in fleet) | 20–85% | x86 CPU accepted, ROCm package only, Pi/Jetson in progress |

Overall: about 93% ready to replace the x86 CUDA text/vision workers with a
staged rollout, and about 80% for every Python deployment profile and modality.

**Done in this session:**

- **Worker VRAM.** The worker no longer holds a CUDA context. Two sources
  created one: free-memory probes through `cuMemGetInfo`, and torch
  allocator rows through `cudaMemGetInfo`. Both now read NVML, whose free value
  equals the CUDA value: 2138 MiB from both on the 3090 Ti. GB10 unified memory,
  HIP and missing NVML keep the context probe. Production free VRAM went from
  2267 to 2541 MiB, and `ezlocalai-native` is gone from `nvidia-smi` compute
  applications. Deployed image is `19d618b`; rollback is
  `ezlocalai:cuda-before-nvml`.
- **x86 portability check.** No `%zmm` (AVX-512) or VNNI instructions in
  `libggml-cpu`, the ACE-Step and stable-diffusion ggml libraries, the TTS
  engine, llama-server or the worker. The image is safe for the Haswell, Zen+
  and Zen 4 fleet hosts.
- **Handoff item 1: larger music-video retried.** On the corrected tunnel
  budget, the two-scene 768×512, 6 s, 18-step storyboard completed through
  Rust's reverse tunnel in 801.9 s; it previously failed at 600 s. Output is
  768×512 video and 24 kHz audio of 6.000 s each, two 3.04 s scenes with
  visible concert content and the formula overlay. The LLM was restored, there
  were no resource-poll timeouts, and production was restored healthy with chat
  passing and `.env` unchanged.
  - The driver's strict `memory.peak <= 28 GiB` assertion fired at 4096 bytes
    over the limit. `memory.peak` includes reclaimable page cache from model
    reads. There was no OOM kill: generation completed and the worker log has
    no OOM entries.
  - Evidence: `tests/fixtures/tunnel_media_timeout_live_20261020_validation.json`.
- **Handoff item 2: GB10 refreshed** at `19d618b` (image `3996802ecd96`).
  54/54 checks pass with the four QEMU-only exclusions, and health, models and
  resources respond.
  - The GB10 compose now defaults to the complete gate. Emulated builds must
    pass
    `EZLOCALAI_PACKAGE_TEST_EXCLUDE='^(dotenv_startup|tunnel_framing_reference|ltx_runtime|media_engine)$'`.
    Each of the four was confirmed to pass on ARM64 with native timing.

- **Handoff item 3: ARM64 CPU/Pi accepted under QEMU.**
  - `ezlocalai:rpi` (`b21eb78a9874`, 3.3 GB) passes 51/51 package
    checks with the four QEMU-only exclusions.
  - Real CPU inference through an isolated Rust router passes for chat, SSE,
    vision (Qwen3.5-0.8B with projector), Qwen3 embeddings and Whisper small
    STT.
  - The first SSE check returned "YES" instead of "READY". Its identical
    greedy prompt reused 15 of 19 cached tokens from the preceding non-stream
    request, and the tiny model flips. With per-request prefixes both return
    "READY". x86 27B stream and non-stream outputs are identical.
  - Qwen TTS works: 200, 24 kHz, 5.12 s of speech. It takes 890 s under QEMU,
    so it needs `QWEN_TTS_TIMEOUT` above 300 s there. At the default the worker
    returns 500 at 304 s, and through Rust the router's 300 s
    `REQUEST_TIMEOUT` returns 502. Real Pi TTS speed is unmeasured; slow CPUs may
    need both timeouts raised, as with Python.
  - Evidence: `tests/fixtures/arm64_cpu_acceptance_20261020_validation.json`.
    The superseded `native-cpu-arm64-compat-final` tag was removed.
- **Still open:**
  - Jetson Orin compilation (item 4). Not resumed: the torch build is at about
    237 of 5,318 actions with two jobs under QEMU, and Jetson is not in the
    fleet.
  - Physical GB10, AMD, Pi, Jetson and multi-GPU execution (item 5).

**Disk hygiene (user request, 2026-10-20).** The shared host disk was at 90%.
115 superseded `ezlocalai:*` tags were removed. Kept tags: production
`native-cuda`, rollbacks `native-cuda-before-nvml` and
`native-cuda-before-media-timeout`, current `native-gb10`, `native-rocm`,
`native-cpu`, `native-cpu-arm64-compat-final`, and
`native-cuda-nf4-integrated-candidate`, which the user's gemma-residency
container uses.

Removing the tags alone freed little, because build cache still referenced
the layers. 184 project cache entries for superseded worker compile, test and
final stages were pruned by ID. The engine stages and the Jetson cache were
kept. Re-downloadable binaries were deleted from
`build/validation/nf4` (text-encoder copies) and `hip-runtime-20261020`
(wheel, archive and extracted libraries), about 49 GB; logs and text records
remain.

Free space went from 389 GB to 434 GB. After each build cycle, repeat this for
superseded tags and cache. Other projects' images, volumes and cache are not
ours to prune.

## Agent handoff checkpoint (2026-10-20, branch `native-parity`)

The user requested a commit and handoff at the next safe stopping point.
All validation/build processes are stopped at this checkpoint.
Production `ezlocalai` is running the updated CUDA image, healthy, with
the original effective environment, `.env`, mounts, ports and GPU access
preserved. Real chat/SSE/vision passed through Rust in direct and reverse-tunnel
modes, then passed again after restoring production settings. Do not remove
the user's separate `ezlocalai-gemma-residency-build` container.

Current local images:

| Tag | Image SHA-256 prefix | Latest package acceptance | Scope |
| --- | --- | --- | --- |
| `ezlocalai:cuda` | `c42ad296d507` | 58/58, 282.62 s | Deployed on RTX 3090 Ti; includes both media fixes below |
| `ezlocalai:cpu` | `93b436302ed8` | 55/55, 273.20 s | x86 CPU; includes both media fixes |
| `ezlocalai:rocm` | `c3c915760860` | 60/60, 312.42 s | HIP package and earlier CPU-fallback inference; no AMD GPU execution |
| `ezlocalai:gb10` | `49ba38ab4749` | 54/54, 518.24 s | QEMU package/startup checks; includes audio fix, predates tunnel timeout fix |

Exact image/worker hashes, rollback tags and evidence are in
`tests/fixtures/tunnel_media_timeout_20261020_validation.json` and
`tests/fixtures/ltx_audio_timing_20261020_validation.json`.
CUDA rollback is `ezlocalai:cuda-before-media-timeout` (`db669f04…`).
These images were built before the checkpoint commit and retain the earlier
`216bbed` Git version metadata. Use the recorded image/worker hashes for their
provenance; rebuild from the checkpoint commit for release version stamping.
All earlier dated entries below are historical, not additional current images.

This checkpoint includes source-aligned Compose profiles and GPU architecture
gates, HIP runtime/metadata adapters, explicit-projector reporting, bounded
heartbeat shutdown, streaming/cancellation acceptance, subtitle and wake-word
job acceptance, ARM Torch dependency packaging, and the two media fixes below.
The six-profile Compose gate passed 1,875 comparisons. The latest worker tunnel
suite passed in 29.87 seconds, including 13 real-socket timeout cases; all ten
Rust integration scenarios passed in 33.72 seconds.

Next work, in priority order:

1. Retry the larger two-scene 768×512, six-second, 18-step music-video request
   through Rust with the corrected tunnel budget. The earlier run timed out
   at the inherited 600-second local limit; do not count it as successful
   video acceptance. Prepared local driver:
   `/tmp/ezlocalai-music-video-timeout-live.py`, using
   `ezlocalai:cuda-timeout-candidate`. It stops/restores production
   in `finally`, uses isolated router state, and checks scene outputs,
   restoration and a 28 GiB memory limit. Inspect the script before reuse;
   it is an external acceptance driver, not maintained first-party Python.
2. Refresh GB10 from this commit to include the tunnel timeout correction.
   QEMU-only exclusions remain explicit:
   `^(dotenv_startup|tunnel_framing_reference|ltx_runtime|media_engine)$`.
   Native GB10 builds default to the complete gate. The existing local
   incremental recipe is
   `build/validation/tunnel-media-timeout-20261020/gb10-refresh.Dockerfile`;
   the canonical recipe is `cuda.Dockerfile` via `docker-compose-gb10.yml`.
   Repeat health/resources/models acceptance; no physical GB10 is available.
3. Finish ARM64 CPU/Pi packaging and actual model acceptance. The missing
   Torch external libraries and indirect Fortran SONAME resolution are fixed,
   and actual ARM matrix/convolution probes pass under QEMU. The full worker
   compiled; the QEMU package gate passed 52/55 checks in 498.39 seconds.
   Failures are tunnel-framing timeout, LTX process-timeout cleanup and the
   media-engine subreaper fixture. Inspect
   `build/validation/arm64-cpu-20261020/build.log` and
   `tests/fixtures/arm64_cpu_20261020_validation.json`; do not claim an accepted
   Pi worker image. That build snapshot predates both final media fixes.
   `/tmp/ezlocalai-cpu-platform-acceptance.py arm64 <candidate-tag>` is prepared
   for isolated real CPU inference through Rust once packaging is resolved.
4. Resume Jetson Orin compilation. It was stopped for this handoff after
   successful Torch configuration and 237/5,318 build actions; completed
   objects remain in BuildKit cache
   `ezlocalai-jetson-r36.4-torch-2.9.1-sm87`. Resume with
   `docker build --platform linux/arm64 -f jetson.Dockerfile --target torch-libraries --build-arg EZLOCALAI_TORCH_JOBS=2 -t ezlocalai:jetson-torch-candidate .`
   from the repository root. Then build the complete image and run its gate.
   FFmpeg/OpenCV, engines and the final worker are not yet accepted on Jetson.
   The generic ARM CUDA Torch wheel lacks `sm_87` and must not replace this
   Orin-specific build. Scope is JetPack 6.1 / L4T R36.4, not Nano/Xavier.
5. Validate on physical 4090/5090/GB10, AMD, Pi/Jetson and multiple GPUs when
   available. Architecture inventories and simulated GPU policies do not
   establish physical execution. Broader video quality/presets, uncommon
   checkpoints and the remaining transport/lifecycle edges in the manifest
   also remain qualified. Keep video last behind worker/router correctness.

Qwen is the required TTS implementation. PDF-to-speech, historical TTS
fallbacks, the Python router port and separate experimental Gemma storage
variants are outside scope. Python reference checkout:
`/home/josh/repos/xtsys/ezlocalai` at
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`; Rust router:
`/opt/DevXT/WorkerRouter`. Keep production credentials and ignored build/model
artifacts out of Git. Detailed local logs are under `build/validation/`;
compact acceptance records under `tests/fixtures/` are checked in.

## Long media requests over the worker tunnel (2026-10-20)

The larger two-scene CUDA music-video acceptance exposed an inherited tunnel
limit: Python and the native worker gave local HTTP requests a fixed
600-second read budget. Rust allowed the longer request, but the worker
abandoned its local connection. The router then retried against a worker
still occupied by generation and returned 503 after 654.50 seconds.
The worker did not OOM; production was restored healthy, chat passed and
the original environment file remained unchanged.

The native correction retains the ordinary 600-second minimum and extends
it using `REQUEST_TIMEOUT`, plus `VIDEO_GENERATION_TIMEOUT` for video and
`ACE_STEP_TIMEOUT` for music. Music-video uses both generation budgets.
Invalid, nonfinite and nonpositive settings use the respective defaults;
positive fractional seconds round upward. This intentionally corrects the
inherited limit. A regression observes the actual local socket timeout while
exercising framed requests and responses; it failed before the fix and now
passes all 13 timeout cases alongside cancellation/reconnection coverage.
All ten Rust integration scenarios pass. CUDA/CPU/ROCm package gates pass
58/55/60 checks respectively. The corrected CUDA image is deployed and
direct/tunnel chat/SSE/vision plus production restoration pass. The larger
live music-video retry remains pending at the requested handoff.

Checked evidence: `tests/fixtures/tunnel_media_timeout_20261020_validation.json`.
Detailed evidence: `build/validation/tunnel-media-timeout-20261020/`.

## LTX audio timing correction in validation (2026-10-20)

The default 768×512, 121-frame CUDA generation completed through the Rust
reverse tunnel in 487.19 seconds, with responsive resources and successful
chat after model restoration. Inspecting the output exposed a source bug:
the 5.04-second picture had 10.02 seconds of audio. Python and the initial
native exporter labeled the native 48 kHz waveform as 24 kHz, doubling its
duration and halving its pitch.

The native exporter now resamples to 24 kHz, preserving duration and pitch.
This is an intentional compatibility correction. A real FFmpeg regression
failed before the fix and passes afterward, checking both the duration and
the frequency of a known tone. The complete LTX runtime suite passes in
19.56 seconds, including video-only and music-video paths. Actual corrected
CUDA generation completes through Rust in 460.86 seconds: audio is 5.01
seconds at 24 kHz alongside 5.041667 seconds of video. Decoded video frames
are byte-identical to the earlier output. Resource polling has no timeouts;
LLM restoration, follow-up chat and production restoration pass without OOM
under the 28 GiB limit. The subsequently deployed tunnel-timeout image retains
this exporter fix. Historical doubled-duration captures remain as evidence.

Checked evidence: `tests/fixtures/ltx_audio_timing_20261020_validation.json`.
Detailed evidence: `build/validation/ltx-audio-timing-20261020/`.

## Final x86 CPU package accepted (2026-10-20)

The CPU image now includes the shared runtime adapters and heartbeat shutdown
fix. All 55 mandatory package checks pass (280.66 seconds). Actual inference
through the Rust router passes chat/SSE, vision, two 1024-dimensional embeddings,
Whisper transcription, Qwen 24 kHz speech and chat after speech. Reverse-tunnel
chat/SSE pass separately. Reporting converges to matching worker/router slot
snapshots with zero VRAM.

Both isolated workers exit normally without OOM. Peak memory is
7,625,072,640 bytes in direct mode and 1,980,993,536 bytes in tunnel mode,
under the 8 GiB container limit. This uses the Pi profile's small model settings
on native x86 CPU; it is not ARM or physical Raspberry Pi validation.

The accepted `ezlocalai:cpu` image is
`fbc9d2b06615bcdb3823318a85c69f8dfac23bd2d68be816665ca20ecbf5edf7`,
worker SHA-256
`b0835eab4a13fc907000ab45065451f92091b7630c909fdbd4e1ad2ac889bf9e`.
Rollback is `ezlocalai:cpu-before-runtime-backends` (`8df94593…`).
The production CUDA container was not changed by these isolated checks.
Checked evidence: `tests/fixtures/cpu_package_20261020_validation.json`.
Detailed evidence: `build/validation/final-cpu-20261020/`.

## ROCm package and CPU fallback accepted (2026-10-20)

All 60 mandatory ROCm package checks pass (322.16 seconds), including HIP
ABI, device discovery, AMD worker metadata, speculative arguments and
LibTorch behavior with no GPU. The first gate passed 59/60: the speculative
fixture requires three unchanged Python source files to verify its capture.
The image now downloads those immutable files and checks their SHA-256 values
before testing. They are test inputs and are not executed.

The packaged worker passes actual Rust-routed CPU chat/SSE, vision, two
1024-dimensional embeddings, Whisper transcription, Qwen speech and chat after
speech. Reverse-tunnel chat/SSE pass separately. Both workers exit normally
without OOM. Peak memory is 8,030,134,272 bytes in direct mode and
2,043,006,976 bytes in tunnel mode, under an 8 GiB limit.

The accepted local tag `ezlocalai:rocm` points to
`c993297cb6b1fd9aa87f49c80ae732fc3a386be671bbc67f8427c729952ad57d`,
worker SHA-256
`e0efe67a31aca43674e2a31a2ca882e24408504184feda841a9e8468b226731d`.
There was no previous native ROCm tag. This is package and CPU-fallback
acceptance, **not physical AMD GPU execution**. The compiled gfx1100 target
and source HSA 11.0.0 override are not universal AMD support.
Checked evidence: `tests/fixtures/rocm_package_20261020_validation.json`.
Detailed evidence: `build/validation/rocm-package-20261020/`.

## Jetson Orin build candidate (2026-10-20)

The new Jetson Dockerfile and Compose profile target the source deployment's
JetPack 6.1 / L4T R36.4 base and Orin `sm_87`. All 237 source Compose
comparisons pass; the combined six-profile gate passes 1,875 comparisons
in 1.09 seconds, including the two documented Pi model-default corrections.
The original environment file is not read by this gate.

The generic ARM Torch 2.9.1 CUDA 12.6 wheel does not contain an Orin target.
The new package architecture check rejects that actual library. NVIDIA
excludes Tegra from desktop cubin compatibility, so this image builds pinned
Torch 2.9.1 and CTranslate2 for `sm_87`; it does not infer Orin support from
an `sm_80` library. The base image was pulled by digest and inspected:
Ubuntu 22.04, CUDA 12.6.68 and cuDNN 9.3.0.75.

Pinned CMake 3.31.8 and nlohmann_json 3.11.3 supply required build interfaces.
FFmpeg 6.1.2 supplies Whisper's newer channel-layout/resampler API, with OpenCV
built against the same libraries. Engine patches are applied before a single
diffusion build. The separate LibTorch build is now running under emulation.

This is not an accepted Jetson image: compilation, the full package gate,
runtime acceptance and physical GPU execution remain pending. Nano, Xavier
and JetPack 5 need other dependency sets. Checked evidence:
`tests/fixtures/jetson_package_20261020_validation.json`.
Detailed evidence: `build/validation/jetson-20261020/`.

## Heartbeat shutdown cancellation deployed on CUDA (2026-10-20)

A Rust integration shutdown timeout prompted a source audit. Python allows
an in-flight registration or heartbeat five seconds to finish, then cancels
it and separately deregisters any successfully registered worker. The C++
worker instead waited for the full HTTP timeout. A new source-derived
loopback regression failed before the fix at 10.28 seconds.

The worker now applies the five-second grace through its existing cancellable
HTTP deadline, preserves registration state across cancellation and retains
the separate normal timeout for deregistration. Four measured source/native
cases pass: stalled registration and heartbeat stop at about 5.01 seconds,
a released heartbeat completes at 0.25 seconds, and stalled deregistration
retains its roughly ten-second timeout. Request ordering matches the source.

The existing heartbeat policy and HTTP suites also pass, followed by all ten
actual Rust-router scenarios (four suites, 87.24 seconds). The integration
test's former five-second whole-process limit incorrectly included the
separate deregistration and engine cleanup. It now allows twenty seconds;
the focused shutdown test enforces the actual source timing boundaries.
No fixture-engine processes remained after the failed integration run.

The new test is mandatory in all Docker package gates. The final CUDA image
passes all 58 checks (288.03 seconds) and is deployed as
`ezlocalai:cuda`,
`db669f04add8660bccfeb3d89c6e7173eb1a8d912bf7c6c4073d79a73fd91a38`.
Its worker SHA-256 is
`3dfcd89807949a674895fc362988870d8b55356581f70587f12e155c7d2798ea`.
Actual CUDA chat, SSE and vision pass through Rust in direct and tunnel modes,
with 19,052 MiB allocated by the owned engine. All three pass again after
restoring the original production configuration. The original `.env`, effective
environment, mounts, ports and GPU access are preserved. Rollback is
`ezlocalai:cuda-before-runtime-backends` (`b46774a0…`).
ARM64 CPU and GB10 builds are still undergoing final validation.
Checked evidence: `tests/fixtures/heartbeat_shutdown_20261020_validation.json`.
Detailed evidence: `build/validation/heartbeat-shutdown-20261020/` and
`build/validation/final-cuda-20261020/`.

## ROCm LibTorch and router integration (2026-10-20)

The HIP-configured native worker now builds with the verified ROCm 2.9.1
Torch libraries. Speech, wake-word training and allocator diagnostics share
the correct HIP allocator/device guards while preserving PyTorch's public
CUDA device names. Speculative draft launches use llama's `ROCm` prefix.
Native dependency extraction retains the ROCm math-kernel assets and installs
the runtime SONAME link required by native executables.

Ten scenarios pass against the actual Rust WorkerRouter in 49.29 seconds,
including direct/reverse-tunnel requests, reporting, explicit projectors,
router restarts and managed-engine failures. These use controlled inference
engines. The actual HIP SDK/library ABI, unavailable-device Python reference
and allocator checks pass separately. Five native CPU training cases and
three cancellation boundaries match fresh captures from unchanged Python
classes using the matching ROCm Torch/torchaudio wheels.

The shared adapter also passes the existing 18 real NVIDIA GPU STT cases,
allocator checks and five CUDA training cases with cancellation. These ran
in an isolated container. The later CUDA acceptance above deploys the shared
adapter after the complete package gate and actual routed inference checks.
The subsequent cuDNN-availability correction changes only the HIP branch.
ROCm Python reports compiled cuDNN availability even though ATen's
`hasCuDNN()` is false; that distinction now matches the source.

ROCm model-engine builds, Compose packaging and physical AMD GPU execution
remain unverified. A candidate Dockerfile and Compose profile are now present;
its 252 environment-setting comparisons pass against the unchanged Python
profile (1,638 comparisons across all five profiles, 0.72 seconds).
The default gfx1100 target matches the source's HSA 11.0.0 override and is not
a universal AMD binary. The first image build exposed UTF-8 kernel asset
paths incompatible with the build stage's default locale; the explicit
UTF-8 locale fixes extraction, and engine compilation is in progress.

Five policy/reporting suites now pass in the HIP-configured build. The
NVIDIA-captured suites retain their NVIDIA fixture adapter. A separate HIP
suite checks 73 worker-extra and 23 model-reporting snapshots plus eight AMD
device-ordering, visibility and property-failure cases through registration
and heartbeats. The video-reference check initially used an incorrect source
directory; correcting the reference root made that suite pass.

Executing Python's actual GPU-reporting function then exposed a further
mismatch: AMD hardware metadata must use backend `rocm`, even though Torch
tensor devices use `cuda`. The new eight-case source capture fails before
the correction and passes afterward, including complete GPU entries, tiers
and `rocm:N` fallback names on property failures.

This supersedes the adapter/draft/router gaps listed in
the foundation milestone below; it does not establish an accepted AMD image.
Checked evidence: `tests/fixtures/hip_torch_20261020_validation.json`.
Detailed evidence: `build/validation/hip-runtime-20261020/`.

## AMD device adapter foundation (2026-10-20, not a complete ROCm image)

While the priority GB10 and ARM64 CPU builds continue, the hardware audit
confirmed that native discovery only loaded NVIDIA's driver. The opt-in
`EZLOCALAI_USE_HIP` build now uses HIP for device properties, visible logical
ordinals, memory queries and retained primary contexts. Public `cuda` naming
remains compatible with the Python worker's PyTorch interface. CUDA remains
the default and the accepted production image is unchanged.

The new C++ test covers two simulated AMD cards, reordered devices, memory and
context isolation across 200 concurrent queries, failures and recovery.
HIP visibility is delegated to the driver, including its CUDA-compatible
environment alias. The initial fixture assumption about that alias was
corrected from the pinned AMD source before the final run.
The HIP suite and existing CUDA worker-metadata/video-runtime suites all pass
(three suites, 42.86 seconds).

A separate C++ ABI gate compiles against the actual ROCm 6.4.1 SDK, checks
the twelve function signatures and two attribute values, then loads all twelve
symbols from its actual runtime library. It passes in the source Python
profile's ROCm base image without GPU access. This proves SDK/library
compatibility, not AMD GPU execution.

ROCm LibTorch/allocator wiring, engine builds, a source-aligned Compose profile
and full worker/router acceptance remain unfinished. The audit also identified
the current speculative draft argument's hard-coded `CUDA` device prefix;
the pinned llama HIP backend uses `ROCm`. No AMD image is claimed ready.
Checked evidence: `tests/fixtures/hip_runtime_20261020_validation.json`.
Detailed evidence: `build/validation/hip-runtime-20261020/`.

## Explicit projector reporting deployed; ARM64 CPU packaging in progress (2026-10-20)

The new ARM64 CPU/Raspberry Pi profile exposed a worker reporting bug during
real x86 CPU validation: `LLAMA_CPP_MMPROJ_PATH` was passed to llama.cpp, but
the owned server's projector state remained empty. A projector outside the
automatic model directory therefore failed to advertise vision to Rust.
The worker now retains the path it passes to the owned engine.

The real Rust-router regression fails before this fix and passes afterward
in direct and tunnel modes, including engine arguments and reporting after
router restart. The full ten-scenario integration passes in 34.19 seconds;
heartbeat source comparisons and LLM launch-parameter tests also pass
(three suites, 37.22 seconds). The CUDA image passes all 57 packaged checks
(278.91 seconds), and x86 CPU passes all 54 (259.82 seconds).

The fixed CUDA image passes actual chat, SSE and vision through Rust in both
direct and reverse-tunnel modes, using a synthetic model alias and explicit
model/projector paths outside its automatic download directory. Both modes
advertise vision and show 19,052 MiB allocated by the owned CUDA engine.
The image is deployed as `ezlocalai:cuda`,
`b46774a0d4218552afd2e58907115f9aef05421f4476c8be5e267062689712b5`;
worker SHA-256 is
`5ef49e4f1453bddc4603bc863dbea78ffc37d530081298c1ab7040168f9f9207`.
Original effective environment, mounts, ports and GPU requests are restored,
the original `.env` is unchanged, and production reporting/chat/SSE/vision
pass again. Rollback is retained as
`ezlocalai:cuda-before-explicit-projector` (`93cab6bf…`).
The first acceptance driver stopped before mutation because its Docker process
query lacked the required PID column; correcting that query required no worker
change.

The GB10 image now passes its 53-check package gate under ARM64 QEMU
(499.80 seconds) and starts the packaged worker successfully: health,
resources and model endpoints pass with inference disabled and CUDA stubs.
The four previously documented emulation exclusions remain explicit;
native GB10 builds still default to the complete gate. The accepted
`ezlocalai:gb10` image is
`93acd6008901c110e7f0049ffefe45f5d01e18d2a04c1ecec9d13e6e409e14e1`,
with rollback at `ezlocalai:gb10-before-explicit-projector`.
This snapshot contains the projector fix but predates the HIP/LibTorch work.
No physical GB10 GPU execution is claimed.

The fixed x86 CPU image also passes real Rust-routed chat/SSE, vision, two
1024-dimensional embeddings, small Whisper transcription, a four-second
24 kHz Qwen WAV and post-speech chat. Reverse-tunnel chat/SSE pass separately.
This uses the new Pi profile's small model and context settings on x86,
with no GPU access or external fallbacks. Peak cgroup memory is
6,274,179,072 bytes under an 8 GiB limit; both workers exit normally without
OOM and the temporary container is removed. This is not ARM execution.
The acceptance driver was corrected to accept null SSE content and to retain
the normal setting permitting native Qwen CPU inference.
The accepted `ezlocalai:cpu` image is
`8df9459391faad1285dc4dc41434aba39fdc8a85716f946a8041b2b5c44a8717`,
with rollback at `ezlocalai:cpu-before-explicit-projector`.

`docker-compose-rpi.yml` and the shared CPU Dockerfile now select ARM64 native
dependencies, including CPU LibTorch, CTranslate2, ONNX Runtime and the ARM64
font/text libraries already used by GB10. The first ARM build caught an x86
storyboard-library selection error; that selection and the architecture-specific
download caches are corrected. The ARM image has not yet passed its package
gate or runtime acceptance and is not an accepted release.

The host Compose test now checks 1,386 settings across four profiles. It keeps
the unchanged Python Pi capture and explicitly records two default corrections
(unset and empty `DEFAULT_MODEL`): the source's `Qwen3.5-0.6B-GGUF` repository
is unavailable without credentials; the new profile uses the published
`Qwen3.5-0.8B-GGUF`. Explicit model overrides still match the source.
The model/projector and default small Whisper weights are downloaded with
pinned revisions and recorded hashes for upcoming ARM runtime checks.

Checked evidence: `tests/fixtures/explicit_projector_20261020_validation.json`.
Detailed evidence: `build/validation/explicit-projector-20261020/` and
`build/validation/arm64-cpu-20261020/`. The ARM64 CPU build remains in progress;
its candidate image is not an accepted release.

## CUDA subtitle translation (2026-10-20)

The deployed RTX 3090 Ti image now passes the complete local subtitle pipeline:
Whisper produces English segments, the real Qwen3.8-27B LLM translates them
into Spanish and French, and both translations retain the exact source IDs and
timestamps. Separate fast-mode requests produce valid SRT and VTT output.
The Whisper `/v1/audio/translations` endpoint also passes on the English audio
fixture; this alone does not establish non-English Whisper translation quality.

Active STT and LLM CUDA allocations are observed. All four audio requests
unload and restore the LLM, release speech ownership, and leave the worker
ready for subsequent routed chat/SSE/vision. These endpoints are local worker
APIs; neither the Python router nor Rust WorkerRouter exposes them.
The existing 163 source HTTP comparisons and 70 helper cases pass again.

The first acceptance probe compared router slots immediately after local
completion and saw the previous busy heartbeat. The corrected probe waits,
with a 15-second limit, for reporting to converge before checking the final
idle snapshot. This was a validation timing issue; no runtime change was needed.
The original production configuration is restored and routed inference passes.

Checked evidence: `tests/fixtures/subtitles_cuda_20261020_validation.json`.
Detailed evidence: `build/validation/subtitles-cuda-20261020/`.
This closes the live weighted-LLM/CUDA subtitle handoff gap for this image and
fixture. Longer recordings, broader language/model quality, concurrent failure
schedules and other hardware remain outside this acceptance.

## CUDA streaming cancellation and recovery (2026-10-20)

The real Rust-router integration now cancels an active managed-engine chat
stream in both direct and reverse-tunnel modes. It checks reservations before
disconnect, observes cancellation at the owned engine, waits for both router
and worker admission to return to zero, then serves another chat without
restarting the engine. This is a permanent C++ regression test.
The final ten-scenario integration run passes in 32.47 seconds; embedding
runtime and chat-stream regressions also pass. Failure cleanup now gives the
worker a bounded graceful shutdown before resorting to SIGKILL, so an assertion
failure can reap owned engines. The earlier failed probe's orphaned fixture
was removed.

The deployed RTX 3090 Ti worker also passes six real CUDA chat interruptions:
after headers, after one content chunk and after five content chunks, each in
direct and tunnel mode. Worker admission returns to idle in 0.129–0.366 seconds
in these runs; the router also releases admission before each successful
follow-up chat. The same CUDA llama process survives every interruption.
Router timings in the raw artifact include the subsequent chat and must not
be interpreted as isolated cancellation latency.

Real Qwen speech passes routed PCM interruption in both modes, plus local
WebSocket completion and abrupt disconnect. Active Qwen CUDA process allocations
are observed before interruption. Speech admission and its loaded-model record
clear, the LLM reloads on CUDA, and a routed chat succeeds afterward.
Cancellation-to-restoration takes 4.555–4.604 seconds in the three interrupted
speech cases. The completed WebSocket has a valid 24 kHz, 16-bit mono PCM
header, complete frame lengths, a zero-length terminator, the legacy empty
notification and normal close. WorkerRouter does not expose the local speech
WebSocket route; its checks connect directly to the worker.

No runtime fix or image replacement was needed for this milestone. The
temporary speech/router configuration and voice are removed. Production uses
the same `93cab6bf…` CUDA image, original effective environment, ports, mounts
and GPU access; its `.env` is unchanged. Reporting and routed chat/SSE/vision
pass after restoration, and no Qwen child remains in the container.

Checked evidence: `tests/fixtures/stream_cancellation_20261020_validation.json`.
Detailed evidence: `build/validation/stream-cancellation-20261020/`.
This establishes the tested cancellation boundaries on one GPU, not exhaustive
scheduling, shutdown or remote-provider behavior. Speech waveform equivalence,
other GPU hardware execution and complete replacement parity are not claimed.

## NVIDIA fleet configuration and runtime acceptance (2026-10-20)

The native CUDA Compose profile now matches the Python profile's declared
defaults and empty-value interpolation. This adds 90 previously omitted
settings, including media handoff, speculative decoding, per-GPU overrides
and voice/cache controls. Five defaults differed, including quantization,
KV cache selection, lazy voice loading and request timeout. A host-only C++
test compares **1,161 settings** across CUDA, GB10 and CPU with unset, empty and
synthetic overridden inputs. The CPU profile also gains its 51 omitted
declared defaults, retaining its own Python quantization and timeout values.
The original `.env` is unchanged. The deployed
effective environment intentionally uses the corrected defaults; it is not
identical to the previous container environment.

Both NVIDIA profiles now gate the actual packaged CUDA binaries. The RTX
engines contain targets for 8.6, 8.9 and 12.0; GB10 requires 12.1. The gate
checks ELF architecture, llama.cpp, stable-diffusion.cpp, ACE-Step, CTranslate2
and LibTorch, including compatible PTX and architecture-specific `a` targets.
An `sm_120a` binary alone cannot satisfy GB10. This checks binary inventory,
not execution or every kernel on hardware unavailable here.

The RTX 3090 Ti CUDA worker passes real Rust-router requests in **both direct
and reverse-tunnel modes**: two 1024-dimensional embeddings, Whisper
transcription, Qwen WAV, Qwen streamed PCM, then chat/SSE/vision after actual
LLM unload/reload. Active model registration and GPU process allocations
confirm CUDA use. Reporting checks cover 18 router fields. The temporary
speech/embedding profile is removed and production direct-mode text/vision
requests pass again. Ports, mounts and GPU access are preserved.

The final CUDA image passes **57 packaged checks** (267.23 seconds) and is
deployed as `ezlocalai:cuda`, image
`93cab6bf6f0fb1b49ae26590c0d4dd8a705931f44f29c821663e79ff7443128c`.
Its worker binary is byte-identical to the modality-tested candidate
(`39091550f0404e39ddcd44c8fe9ea3e41c47acb5ff3166d3f561e406f9dd5c0d`).
The final package deployment preserves the already-corrected effective
environment and passes production chat/SSE/vision and reporting again.
The original pre-fleet image remains at
`ezlocalai:cuda-before-fleet-gate`; the intermediate fleet image is
also retained at `ezlocalai:cuda-before-fleet-http-fixture`.

An isolated CPU container also completes a real wake-word job: five Qwen
samples produce 30 augmented positives and 21 negatives, one training epoch,
and checkpoint/config/ONNX export. Both binary downloads match the saved
files and the exported checkpoint handles prediction. Cancellation during
synthesis publishes no checkpoint. Completed/cancelled status and checkpoint
prediction survive restart and container recreation. The test container is
removed and production data was never mounted writable. This establishes the
job pipeline, not detection accuracy. Two validation-driver errors were
corrected: the prediction field is `audio_base64`, and Docker can change an
automatically assigned published port on restart.

All eight host GPU-policy suites pass, covering resource accounting, embedding
placement, slots, batching, cache, speculation, launch parameters and allocation.
The allocation fixture now preloads before inference to match its captured
constructor boundary; lazy admission's separate headroom probe must not consume
its before/after samples. Actual multi-GPU execution remains unavailable.

The initial GB10 cross-build passed 52 of 53 enabled checks and exposed an HTTP-test
TLS fixture startup race: a fixed 50 ms delay can expire before its listener
is ready, leaving the test waiting in `join`. The fixture now signals
readiness and bounds socket waits. The final release-built HTTP suite passes
under emulation in 58.23 seconds without increasing its timeout or excluding it.
The complete GB10 cross-build passes **53 checks** (463.17 seconds), including
the actual binary architecture gate. Native GB10 builds default to the full
57-test gate; four documented QEMU-specific exclusions must be explicitly
requested when cross-building.

The accepted local `ezlocalai:gb10` image is
`c4e002b37f1de08b09e5964338127f17ecf93e1e0710828a793fec34b68b21c1`.
The packaged ARM64 executable also starts and serves health, resources and
models under QEMU with inference disabled and CUDA stubs; it correctly reports
zero visible VRAM. This is not physical GB10 GPU validation. The temporary
container is removed, and the previous image remains at
`ezlocalai:gb10-before-fleet-gate`.

Checked evidence: `tests/fixtures/fleet_hardware_20261020_validation.json`.
Detailed evidence: `build/validation/fleet-hardware-20261020/`.
Full replacement readiness remains incomplete. Real 4090/5090/GB10 and
multi-GPU execution is unavailable on this host; ROCm/Jetson/Raspberry Pi
packaging, broader video acceptance and remaining transport/lifecycle edges
are still open. PDF-to-speech and separate Gemma experiments remain excluded.

## Reverse-tunnel redirects (2026-10-20)

The tunnel worker previously required the initial router URL to return a 101
upgrade directly. It now follows HTTP redirects with the installed Python
CUDA image's **aiohttp 3.14.5** behavior: logical ws/wss origin changes remove
Authorization, relative redirects preserve permitted credentials, URL Basic
credentials follow source conflict rules, and worker_id is added only to the
initial request. Redirect cookies, the ten-redirect failure boundary and final
WebSocket challenge validation are included. Response fields larger than
8,190 bytes are rejected before cookie parsing.

**59 actual Python TCP/TLS traces** match the native connection, with a separate
cancellation test during the redirected upgrade. The first WebSocket frame is
preserved when coalesced with the HTTP response. The existing cookie profile
retains all 6,180 reference operations; fallback clients continue using their
separately captured aiohttp 3.13.2 profile. Modern response-cookie parsing is
bounded, but aggregate outgoing-header quotas and jar eviction are not yet
exact aiohttp 3.14 equivalents. See `tests/fixtures/tunnel_redirect_README.md`.

The four existing tunnel/Rust integration suites passed in 86.97 seconds.
After the response-field limit was added, all ten scenarios against the real
Rust WorkerRouter passed again on the final source (32.43 seconds). Final
release images pass **55 CUDA checks** (249.86 seconds) and **54 CPU checks**
(247.99 seconds), including both cookie profiles and the new network captures.

The deployed CUDA image is
`ea8cf96281896a6efd120962fc458879a13588ea7465f1255cfe57f066b3de2d`,
worker SHA-256
`eace06a9e063cae20d32982c88e7ab961d4b278dafef902591bc4458ad050c12`.
Routed chat, SSE and vision passed in direct mode, in the real production
reverse tunnel, and again after restoring the original direct setting. All
18 reporting fields were checked against the worker's models/resources.
The original environment file, effective environment, ports, seven mounts and
GPU access are preserved. Rollback is
`ezlocalai:cuda-before-tunnel-redirects` (`819d93f0…`).

The final CPU image is
`222fe7279ed1e2b397cdbae28ed433dc270f4ac76ab5e39d3508610147498a93`.
An isolated container with no GPU access passed real chat, SSE, two 1024-value
embeddings, Whisper transcription and Qwen speech, and reported zero VRAM.
The temporary CPU container was removed. Deployed runtime/test source hashes
match the checkout. These uncommitted changes still report base commit
`fefd3fc`; image and binary hashes identify the tested builds.

Checked evidence: `tests/fixtures/tunnel_redirects_20261020_validation.json`.
Detailed evidence: `build/validation/tunnel-redirects-20261020/`.
Full replacement readiness remains incomplete: other hardware images/runtime
validation, broader video/music-video acceptance and the tracked transport and
lifecycle edges remain. Real multi-GPU execution is unavailable on this host;
PDF-to-speech and separate Gemma experiments remain excluded.

## Download compatibility and CPU deployment (2026-10-20)

Wake-word checkpoint downloads and generated output files now share a native
file-response adapter checked against **Starlette 1.7.0 in the original Python
CUDA image**. Older multipart and 416 assertions described an older dependency
version. The updated behavior includes UTF-8 Content-Disposition, validators,
all three HTTP date formats, single/suffix/multipart ranges, sorted overlap
merging, malformed/unsatisfiable ranges and the source's 100-range limit.
StaticFiles applies conditional 304 responses; checkpoint FileResponse does not.

Responses stream through a 64 KiB read buffer with a declared Content-Length.
HEAD retains the length without sending a body. The opened inode remains valid
if the file is removed during transfer, and its descriptor is close-on-exec.
136 source responses match in buffered and deferred modes. Socket tests cover
checkpoint downloads and multipart/HEAD framing; a sparse file larger than
5 GiB verifies tail offsets and descriptor inheritance. All **70 live output
checks** pass on the deployed CUDA worker. See
`tests/fixtures/file_response_README.md` for capture provenance and limits.

The new `docker-compose.yml` / `cpu.Dockerfile` package an **x86-64 CPU
worker** with checksum-pinned CPU LibTorch 2.9.1, native Qwen TTS, CTranslate2
Whisper, ONNX Runtime and the same pinned llama.cpp, image/video and music
engines. There is no NVIDIA device request or CUDA runtime dependency. This is
an AVX2/FMA/F16C baseline; it is not an ARM or AMD GPU image.

Validation on the final images:

- CUDA: **53 packaged checks**, 251.51 seconds; CPU: **52**, 248.50 seconds.
- CPU container without GPU access: real Qwen3.8-27B chat and SSE, two
  1024-dimensional embeddings, Whisper transcription and Qwen speech. Local
  inference uses the native `b11669-620d2225c` engine. Resource totals report
  zero VRAM. These functional checks do not establish media output quality.
- CPU LibTorch: all five saved Python training cases, three cancellation
  boundaries, 147 checkpoint predictions, 290 JSON/multipart HTTP-handler
  predictions and 91 MFCC arrays (maximum absolute error zero) pass.
- Ten real C++/Rust integration scenarios pass. The final descriptor guard is
  additionally covered by both final image gates and live routed CUDA
  chat/SSE/vision after deployment.

The deployed CUDA image is
`819d93f03a10fb69cefdb4b356e8ec59f71ccdd1766ee46b00dfe0d2d3c6215c`,
worker SHA-256
`af0bce6524af70eb2aafd5c7056d52dada370cee15acd71cfc94fec0ea1111de`.
The CPU image is
`b50b0b29d48b5a407246266034c78369108969a111f7597b3235460a2d49fae8`.
Rollback to the preceding production-audit release remains available as
`ezlocalai:cuda-before-download-parity` (`db88214e…`).

The original environment, ports, GPU access and seven bind mounts are preserved.
This checkout now has ignored local symlinks for `.env` and those seven data
directories. A normal CUDA Compose configuration resolves to the deployed
settings with **zero environment differences**. No credentials or local links
are tracked. CPU testing used isolated containers; CUDA remains deployed.

Checked evidence: `tests/fixtures/downloads_cpu_20261020_validation.json`.
Detailed logs/artifacts: `build/validation/downloads-cpu-20261020/`.
The strict worker checklist still has **27 implemented / 38 partial rows**,
with all 45 source routes covered. Full replacement readiness is not declared:
video/music-video acceptance, other hardware images/runtime validation and the
remaining transport/lifecycle edge cases still need work. PDF-to-speech and
separate Gemma experiments remain excluded; video stays last.

## Production audit after the Claude handoff (2026-10-20)

The handoff started from clean commit `fefd3fc` and the healthy deployed CUDA
image `529dd8fb7c73ec3da41dedca6fc3a99fdfdc45f700b3e4b5262f288c411b994b`.
The sections below are a chronological work log; older statements that a
feature is absent are superseded by later implementation and deployment records.
The wake-word manifest entries have been reconciled with the implemented
background pipeline instead of continuing to call it pending-only.

Scope confirmed by the user: PDF-to-speech is excluded; Qwen is the required
TTS provider; complete LTX support remains in scope, but the separate Gemma
experiments do not. Video stays last. Multi-GPU behavior is reviewed and tested
with controlled device topologies because no real multi-GPU machine is available.

Two production issues were found in this pass:

- GPU reporting fell back to physical `nvidia-smi` cards when CUDA had no visible
  devices. This advertised unavailable hardware with `CUDA_VISIBLE_DEVICES=""`
  or `-1`. Reporting now enumerates only CUDA-visible ordinals. Like Python,
  a failed properties query retains a `cuda:<ordinal>` entry with zero memory,
  rather than preventing the whole registration/heartbeat. Eight isolated
  simulated-driver cases check registration and heartbeat with two cards,
  reordered cards, one card, both forms of hidden devices, and name/compute/
  memory property failures. The CPU entry and best tier are also checked.
- Wake-word cancellation waited until an epoch ended, and a cancellation or
  exception during export could leave a downloadable checkpoint behind.
  Cancellation now checks augmentation/noise samples and tensor training
  samples/batches. Exports are staged below the model directory; the checkpoint
  is published last, with cancellation and completion serialized under the job
  lock. Failed/cancelled exports are cleaned up and are not listed or downloadable.
  Status responses also preserve the full fractional progress value.

The wake-word tests cover live HTTP-handler status/list/metrics, repeat
cancellation, synthesis/training/export cancellation, failed partial exports,
persisted terminal state and successful retraining after cancellation.
Five CPU and five deterministic CUDA training reference cases still match saved
Python history, RNG, callbacks, checkpoint reload and predictions. Both devices
also pass cancellation checks before work and within the first epoch.
All 421 audio preparation WAV artifacts remain byte-identical.

The C++ worker passes ten actual Rust-router integration scenarios covering
direct/tunnel communication, metadata, embedding-only workers, restart and
deregistration. GPU selection, headroom/resource and launch-parameter
regressions also pass. These tests do **not** establish real multi-GPU execution.

Cancellation is cooperative: an active Qwen synthesis call, tensor operation
or export may still run to its next boundary before releasing resources.
The pipeline remains deliberately serialized, and persisted unfinished jobs
remain inspectable/cancellable rather than automatically restarting.
Model accuracy, uncommon checkpoint variants, full download `FileResponse`
semantics, hardware-specific images beyond CUDA/GB10, real GB10 execution and
video acceptance remain separate work. No PDF/Gemma experiment work is required.

Evidence for this pass is under
`build/validation/production-audit-20261020/`.
The compact checked record is
`tests/fixtures/production_audit_20261020_validation.json`.

**Deployment and production tunnel validation.** All 51 packaged CUDA checks
pass (221.47 seconds). The deployed image is
`db88214e09c8d231fc5d3e1488a1d9e96a308ef0efc159a37956cc2323c9352d`,
with worker SHA-256
`0089538a814e69c3cb68dc9691d6af9b4846e00a7bc28bc2047689340d0f35b8`.
The original `.env`, effective environment, seven mounts, ports and GPU access
are unchanged. Rollback is `ezlocalai:cuda-before-production-audit`.
The worker reports base commit `fefd3fc`; these uncommitted fixes are identified
by the image and binary hashes, not by that commit label alone.

Routed chat, SSE and vision passed after deployment. The same real CUDA worker
was then temporarily registered with `WORKER_TUNNEL=true`; its production Rust
registry URL was `tunnel://...`, and all three requests passed over that tunnel.
The original direct-connection setting was restored and all three passed again.
All nine responses identify the native `b11669-620d2225c` engine, with fallback
disabled in the requests. The final worker is healthy. Registry checks cover
18 report fields, GPU tier, model availability and idle slot/memory consistency.
This closes the “never exercised against the production router” tunnel gap;
it is not a claim of a long-duration tunnel soak.

## Current priority (2026-10-19)

The user placed video generation last. Prioritize complete worker reporting to
Rust WorkerRouter, then endpoint behavior against the current Python CUDA
worker. No Wan model downloads or real inference were started during its
preflight audit; that work is deferred until the reporting and endpoint checks.

## PDF speech scope correction (2026-10-19)

The user clarified that PDF-to-speech (`/v1/audio/speech` with PDF input,
extracted with `pdftotext`) was experimental and **is not a replacement
requirement for this worker layer**. Its remaining source-parity gaps below and
in `parity_manifest.tsv` do not block the replacement. Do not expand that work.

## llama.cpp v0.7.0, portable x86 build and GB10 image (2026-10-19)

**Engine.** The native engine is now llama.cpp release v0.7.0 (`620d222`). That
is ahead of the Python reference's xllamacpp engine (b11549). v0.7.0 adds only
optional launch flags. `llm_engine_contract` parses every flag the worker
passes, against the new source hashes.

**Portable x86 build.** The image used to compile llama.cpp's CPU backend,
stable-diffusion.cpp and ACE-Step with `-march=native` (`GGML_NATIVE=ON`) on
the build host, an Alder Lake i9. Those binaries can fail with illegal
instructions on other fleet CPUs (Haswell Xeon, Zen+, Zen 4). All three are now
built with `GGML_NATIVE=OFF`, a portable AVX2/FMA/F16C baseline.

**GB10.** `docker-compose-gb10.yml` builds the same Dockerfile with
`EZLOCALAI_PLATFORM=gb10`:

- CUDA 13.0.2 cuDNN base image, aarch64 (`platform: linux/arm64`);
- `121-real` kernels for every engine, and `GGML_NATIVE=OFF`;
- pinned aarch64 manifests: torch 2.9.1+cu130 with its exact NVIDIA cu13 wheels,
  ONNX Runtime aarch64, soundfile and Pillow aarch64;
- CTranslate2 4.8.2 built with CUDA/cuDNN for sm_121, using the Python GB10
  patch, because the aarch64 wheel is CPU-only.

Its environment applies every default from the Python `docker-compose-gb10.yml`.

GB10 validation status:

- The image (`ezlocalai:gb10`, arm64, 13.9 GB) was built on this x86
  host under QEMU, as the Python GB10 image is.
- Every engine compiled for sm_121, and 47 of the 51 packaged checks pass under
  emulation.
- The four excluded checks are `dotenv_startup`, `tunnel_framing_reference`,
  `ltx_runtime` and `media_engine`. Under QEMU they exceed native timing limits
  or need `PR_SET_CHILD_SUBREAPER`. Run individually with emulation-appropriate
  timing, all four pass on ARM64. `ltx_runtime` also exposed undefined behavior
  in the test itself (`back()` on an empty capture list), which is now fixed.
- Library resolution was checked inside the image: CTranslate2 with
  CUDA/cuDNN 9/cuBLAS 13, torch cu130 and ONNX Runtime all resolve, and the
  worker starts.
- **GPU execution on GB10 hardware is untested.** This host has no GB10 and no
  SSH access to the GB10 worker. The cutover steps are in `README.md`.

## Text fallback to ROUTER_URL (2026-10-19)

**New behavior, not in the source.** When `FALLBACK_SERVER` is unset and
`ROUTER_URL` is set, chat and completion requests that cannot be served locally
go to the router. This applies at the same points where the source uses
`FALLBACK_SERVER`:

- the low-memory pre-check;
- a GPU handoff (`temporarily_unavailable`);
- a queue wait past `QUEUE_WAIT_TIMEOUT`;
- a failed local inference.

Details:

- The router client uses `get_router_dispatch_client`'s key order:
  `ROUTER_CLIENT_API_KEY`, then `EZLOCALAI_API_KEY`, then `ROUTER_API_KEY`.
- It is used only while `/v1/router/health` reports live workers. That check
  is cached for 30 s. The router has no `/v1/resources`, so the existing
  fallback client could not be reused.
- Forwarded bodies carry `disable_fallback: true`, so the receiving worker
  answers locally and a request makes at most one router hop. Requests that
  already disable fallback are not forwarded.
- The local slot is released before a post-failure forward.
- Streaming chat is relayed incrementally through a passthrough
  `LlmInferenceStream`, without normalizing chunks a second time. The
  queue-wait path buffers streams.
- Router statuses pass through, and an unreachable router leaves the
  classified local failure.
- `FALLBACK_SERVER` still takes precedence, with the source behavior.

`router_fallback` uses a fake router speaking the WorkerRouter health and chat
protocol. It covers:

- forwarding after a failure, with the client key and `disable_fallback`;
- no forward for `disable_fallback` requests or when the router reports zero
  live workers;
- a router 413 passing through;
- an incremental stream relay from the pre-check;
- `FALLBACK_SERVER` taking precedence.

It is in the packaged CUDA checks (51 pass).

Live check: the worker was deployed with `MUSIC_ENABLED=true` set as a shell
override, leaving `.env` unchanged. During a 120 s music generation the text
slot reported capacity 0. A direct non-streaming chat and a direct streaming
chat to this worker were both forwarded to the production router. Both were
answered by a Python worker, identified by the xllamacpp `system_fingerprint`.
The stream arrived as 14 incremental events plus `[DONE]`, and the music
request completed with 200.

## Router failover statuses and media I/O against Python (2026-10-19)

**Intentional divergence from Python.** With no `FALLBACK_SERVER`, the source
answers a failed non-streaming chat or completion with an HTTP 200 "Local
resources exhausted and no fallback server configured." completion. WorkerRouter
treats that as success, so it neither fails over nor tells the client to
compact. Native now classifies the failure the way the source's
`classify_inference_capacity_error` does:

- context overflow returns 413 `context_capacity_error` with `n_prompt_tokens`,
  `n_ctx` and `compact_and_retry`;
- memory exhaustion returns 503 `inference_capacity_error`;
- llama.cpp 4xx errors keep their status;
- anything else returns 503 `inference_error`.

Forwarding to a configured fallback server is unchanged. That includes
`disable_fallback` requests, which the source also forwards after a local
failure (226 recorded cases). Non-streaming requests during a GPU handoff
return 503 with the source's `temporarily_unavailable` body, where the source
returned 200. Streaming paths already end without assistant text, which the
router retries, so they are unchanged.

Live checks against the deployed worker, using a 240052-token prompt against
220160 tokens of context:

- direct, and through the production router pinned to this worker: 413 with
  the source detail shape;
- streaming: the `streaming_error` SSE event, as before;
- a subsequent routed chat returns 200.

The tests that encoded the 200 response (`native_http`, `llm_output`,
`llm_residency`, `llm_post_fallback` no-fallback rows, `music_runtime`) now
assert the new statuses.

STT `text`, `srt` and `vtt` responses now include `; charset=utf-8`, which is
how Starlette renders the source's `Response(media_type=...)`. The STT API
fixture recorded the handler media type, so its comparison applies the same
rule.

**Media I/O.** With TTS, STT, image, embedding and music enabled in the
original `.env`, the same requests went directly to this worker and to the
Python GB10 worker, which serves the same model IDs. The worker re-registered
with the same `capabilities` and `cap_models`, and with the same embedding
context and quant, as GB10. Slot capacities differ only by hardware: this
24 GB card hands off TTS, STT, image and music with the 27B LLM, while GB10
runs them in parallel.

- **Voices:** identical lists.
- **Embeddings:** same shape, usage and `dimensions` handling, with
  cosine 0.9994–0.9998 to the Python vectors (Q8, different GPUs).
- **Speech:** both 24 kHz mono 16-bit WAV of about 4 s. The PCM stream headers
  match.
- **STT:** byte-identical JSON, verbose JSON and text transcripts, including
  segment timings.
- **Image:** 512×512 PNG in the same response shape, visually indistinguishable
  from the Python image (same default seed).
- **Music:** identical response metadata and a stereo 48 kHz WAV of about 8 s.

The same probe through the production router returned 200 for every request.
Image and music were served by this worker, and the router persisted and
re-hosted the assets from its `/outputs/`.

TTS unloads the LLM and restores it lazily, as the source's
`VOICE_RELOAD_LLM_AFTER_GENERATION=false` does. The image and music runs that
followed therefore had no LLM to restore, which also matches the source. The
next chat reloaded the 27B model at its full 220000 context in 22 s and
released the image model. Video was not enabled (lowest priority).

All 50 packaged CUDA checks pass in the deployed image. In the development
container, the CTTS and TTS-WebSocket suites needed the Python reference
package copied in, and `music_runtime` needed `precache.py`. Three CTTS
permission-failure cases cannot pass there because the container runs as root.
Rollbacks: `ezlocalai:cuda-before-failover`,
`ezlocalai:cuda-before-stt-charset`.

## Router contract audit against the live fleet (2026-10-19)

The worker-to-router contract was inventoried on three sides: the Python
worker (`Router.py:WorkerHeartbeatClient`, `Tunnel.py:TunnelClient`, app.py
forwarding), the Rust WorkerRouter (`src/main.rs`, `registry.rs`, `proxy.rs`,
`tunnel.rs`, `dashboard.rs`) and this worker. The C++ registration and
heartbeat payloads were then compared with the records the production router
held for five Python workers that run the same model on 3090 Ti/4090/5090 GPUs.

These fields matched the Python records: `capabilities`, `models`, `cap_slots`,
`model_slots`, `model_quant`, `cap_models`, `extra` (residency, loaded/active
LLM, unload-dependent capabilities), `queue_capacity`, `in_flight`,
`best_tier`, top-level VRAM/RAM totals, the `api_key` callback key and the URL.
`model_context` differed (220000 vs 230000), but only because this host's
`.env` sets `LLM_MAX_TOKENS=220000`.

These fields did not match, and are now fixed:

- `gpus[].total_vram_gb` came from nvidia-smi `memory.total` (23.988 GiB on a
  3090 Ti). Python uses torch `total_memory`, rounded to 2 dp (23.54). nvidia-smi
  also lists physical devices rather than the CUDA-visible ordinals. The list now
  uses `visible_cuda_devices()` (cuDeviceTotalMem/cuDeviceGetName), with
  nvidia-smi kept only as a fallback when there is no driver.
- `version` was empty in production, so the dashboard showed `—`. The deployed
  image predated `runtime_version.cpp`. The CUDA image also had no git metadata,
  so even now it would only report `ezlocalai-<ver>+<hash>`, which the
  dashboard does not link. The image now copies `.git/HEAD`, refs and
  packed-refs (not the object store) to `/app/.git`, so it reports the
  checkout's short commit, as the Python image does. The dashboard links hex
  versions to `DevXT-LLC/ezlocalai/commit/<sha>`; those links resolve once this
  repository replaces that one.
- If building the registration body threw, the heartbeat thread exited for
  good and the worker never registered. Python's `_gather_state` guards each
  probe, so in practice it keeps registering. A failed state collection now
  counts as a failed register attempt and is retried with the normal backoff.

Behavior that was checked and already matches:

- Register, then an immediate heartbeat. A failed heartbeat re-registers in
  the same cycle, and shutdown deregisters. The router's response body is not
  used.
- Tunnel framing. The router reads `Content-Type` or `content-type`, so
  lowercased `resp_start` headers are fine.
- Over-context chat with no `FALLBACK_SERVER`. Non-streaming returns Python's
  HTTP 200 `"Local resources exhausted"` body. Streaming returns a
  `streaming_error` SSE event with `exceeds the available context size ...
  n_ctx=`, which the router's `context_error` markers detect, so it fails over.
  Both were checked live against the deployed worker.

Validation: `worker_heartbeat_reference`, `worker_heartbeat_http`,
`worker_tunnel`, `worker_metadata_reference`, `worker_router_integration`
(the actual Rust router binary) and `runtime_version_reference` pass.
All **50 packaged CUDA checks** pass in image `1b4b58dcdb3b`.

That image is deployed with the original `.env`, effective environment, seven
mounts, ports and GPU requests. The only environment difference is the echoed
llama.cpp build arguments: the earlier image was built for sm_86 only, and this
one uses the Compose default of sm_86/89/120. The production router now records
`version: "a169f54"` and `gpus[0].total_vram_gb: 23.52`, and every other field
is unchanged. A non-streaming and a streaming chat sent through the production
router to this worker by label both return HTTP 200. The streaming response
carries 11 SSE events and `[DONE]`.

The C++ and Python (DevXT3090V) chat responses were compared field by field.
They have identical JSON shapes, including `usage.prompt_tokens_details` and
every `timings` field, and the same SSE event sequence. Python streams the GGUF
file path as `model`, while native streams the public model name; the router
does not read that field. Rollback: `ezlocalai:cuda-before-router-reporting`.

## LTX scope correction (2026-10-19)

The required feature is **LTX video generation as a complete pipeline**. The
user confirmed that the separate Gemma storage, quantization and residency
experiments in the Python implementation were unsuccessful speed experiments,
not behavior the replacement must reproduce. Reproducing every Gemma storage
variant, the Quanto CPU fallback, or the experimental module/offload call order
is **not an acceptance requirement**. Do not expand that work.

The native loader currently uses the LTX GGUF transformer with the pipeline's
text encoder, connectors, video/audio VAEs and vocoder. These are internal LTX
assets, not a separate Gemma product or user configuration requirement. Keep
the components needed for working inference; the scope correction does not
by itself remove functioning engine code or change the deployed model format.

Remaining LTX acceptance work concerns supported generation modes and presets,
usable video/audio output, original configuration, CUDA/device selection,
bounded memory use, model cleanup, LLM/auxiliary handoff and Rust router
communication. Successful native repeats do not establish output quality or
coverage of larger presets and multiple GPUs.

Earlier entries below are a chronological evidence archive. Their statements
that Gemma/NF4/Quanto variants or experimental internals "remain open" describe
the scope at the time and are superseded by this correction. They are not the
current replacement checklist. The existing tested format can continue to
serve LTX without implementing the abandoned alternatives.

## TTS scope correction (2026-10-19)

The user clarified that **Qwen TTS is the only required TTS engine**. gTTS,
Edge and other historical fallback engines are no longer replacement acceptance
requirements. Their earlier implementation and validation records below remain
historical evidence; their unresolved provider, transport and output differences
do not block the replacement. Stop expanding those fallback implementations.

Qwen remains in scope across buffered speech, streaming, WebSockets, voice
handling, router communication, configuration, CUDA device selection, resource
admission and model cleanup. The native Qwen implementation already has real
CPU/GPU synthesis and HTTP/WebSocket validation recorded below. This scope
change does not by itself change the deployed image or remove existing code.

The Python wake-word generator used gTTS/Edge. Native background jobs now use
the shared Qwen service for positive and similar-word samples, including its
admission and cleanup. Full CPU/CUDA training jobs and checkpoint round trips
are verified below. Source-exact fallback voice lists and sample counts are not
acceptance requirements; trained-model quality and sample diversity still need
separate evaluation.

## Native speaker diarization (2026-10-19)

The native STT runtime now supplies MFCC/average-linkage diarization instead
of rejecting `enable_diarization` when no callback was configured. It supports
automatic or explicit speaker counts and process-wide session speaker IDs,
including centroid updates and the source's lazy 24-hour expiration. The
session store survives model/pool reloads and synchronizes concurrent requests.

Validation against the unchanged Python implementation covers **33 feature
vectors, 72 clustering cases and 45 diarization/session cases**, plus concurrent
session access. Feature comparison allows 0.003 absolute floating-point error;
cluster labels and complete diarization JSON match exactly on these fixtures.
Six real CPU/int8 Whisper calls, two public multipart-handler calls and four
compiled-worker HTTP requests pass. The HTTP requests cover verbose JSON,
JSON and text, preserve session identity, and release the model and request
slot afterward. All four relevant regression suites pass. Two initially
lacked reference source files in the development container; their reruns pass
after those files were supplied, with the original failure logs retained.

All **49 packaged CUDA checks** pass. The worker and both new test binaries
match the tested build byte-for-byte. Image `4dc9a05ef394...` is deployed and
healthy with the original `.env`, effective settings, mounts, ports and GPU
requests preserved. Live Rust-routed chat, streaming and vision and eight
documentation responses pass. Production settings still disable STT and reverse
tunneling; diarization was exercised separately in an isolated CPU container.
This milestone does not claim new GPU-model diarization validation or measured
speaker-identification accuracy on a representative multi-speaker corpus.

See `tests/fixtures/stt_diarization_README.md` and
`tests/fixtures/stt_diarization_validation.json`. Overall replacement parity
remains partial. The LTX and Qwen-only scope corrections above still apply.

## Worker tunnel deadlines, TLS and backpressure (2026-10-19)

The worker now uses cancellable DNS/socket/TLS operations. TCP and TLS share
one source-compatible 15-second connection budget after DNS resolution. Upgrade
responses may take longer than the former one-second socket timeout. TLS verifies
trusted certificates and DNS/IP identities, with TLS 1.2 as the minimum version.
Shutdown interrupts pending setup and writes without closing a socket from a
second thread while OpenSSL is using it.

The 60-second receive-idle deadline now resets on complete text or binary
messages, including invalid JSON text. Partial fragments and automatically
handled WebSocket control pings/pongs do not reset it. This matches the executed
Python receive behavior. Outgoing frames retain their masked buffer across short
write slices, so a router that stops reading does not prevent reception of
cancellation messages. Response buffering and queued control messages remain bounded.

The new backpressure regression exposed a local-stream lifetime gap: queued
responses kept the request's interrupt closure and HTTP socket alive after its
I/O task ended. `shutdown()` alone left a large upstream upload blocked. Task
completion now releases the closure on the request thread, closing that stream
promptly even while outgoing WebSocket data is stalled. The regression cancels
a 64 MiB local upload and a second pending request before the router reads any
response, then verifies another request on the same connection.

Validation includes **nine actual Python/aiohttp network outcomes**, **six
executed source idle traces** and **three actual source shutdown traces**. The
source idle capture shortens only the outer wait for faster classification;
all six native integration cases run through full workers at the real 60-second
limit. Controls/partial frames expire at about 60.01 seconds; complete data
messages keep the connection alive for 65 seconds. A stalled upgrade stops in
0.57 seconds in the development integration test. Plain and TLS duplex tests
receive control traffic while an 8 MiB output frame is blocked, then finish the
complete frame. Delayed upgrades/TLS, certificate rejection, slow
readers and DNS/write cancellation also pass.

All **eight regression suites** pass, including broad native HTTP, existing
HTTP deadlines and the actual Rust router. All **48 packaged CUDA checks** pass.
The worker and three tunnel test binaries match the tested build byte-for-byte.
A duplicate CTest setting that overrode the intended 60-second worker test
limit with 30 seconds was removed; the initial timeout and subsequent lifetime
failure are retained as evidence rather than counted as successful runs.

Image `3e5c58baa864...` is deployed and healthy. The original `.env`, effective
environment, seven mounts, ports and GPU requests are unchanged. Rust
registration, real routed chat/streaming/vision and all eight documentation
GET/HEAD comparisons pass. The original environment leaves reverse tunneling
disabled, so these live inference checks use direct routing; separate actual
Rust/C++ integration tests cover the reverse tunnel. Entry-point and media-engine
hashes are unchanged.
Rollback: `ezlocalai:cuda-before-tunnel-deadlines`.

Evidence: `tests/fixtures/tunnel_deadlines_cases.json`,
`tests/fixtures/tunnel_stop_cases.json`,
`tests/fixtures/tunnel_deadlines_validation.json` and
`build/validation/tunnel-deadlines/`.

Remaining transport work includes upgrade redirects, complete HTTP parser
behavior and local callback URL/DNS/deadline handling. Address attempts remain
sequential rather than using aiohttp's Happy Eyeballs scheduling. Native shutdown
uses a best-effort close frame with a one-second write budget and does not
reproduce the source close-acknowledgment wait; observed source stop traces are
recorded separately. These tests do not establish exhaustive TLS policy or
shutdown equivalence. Overall worker and replacement parity remain partial.

## Worker tunnel upgrades, URLs and reconnect timing (2026-10-19)

The native worker now validates the WebSocket Upgrade/Connection headers and
server challenge before dispatching messages. Duplicate response headers use
the first value, matching aiohttp. An invalid upgrade closes its socket before
backoff. The earlier worker accepted any HTTP 101 response, and the HTTP test
fixture used a dummy challenge that concealed that gap; the fixture now sends
a valid challenge derived from the request key.

Router WebSocket URLs now use the existing source-tested URL parser and resolve both
IPv4 and IPv6. Host formatting, query preservation, Unicode/worker-ID escaping,
fragments, default ports and URL credentials follow the captured source cases.
Reconnect delays grow from two seconds by 1.5, capped at 15 seconds, across
short-lived connections. A successful connection lasting more than 30 seconds
resets the delay; a failed attempt does not reset it even when it lasted longer.
Previously every successful handshake reset the delay immediately.

All **47 actual aiohttp upgrade-response cases**, **20 request URL cases** and
**five executed TunnelClient reconnect traces** pass. Five integration suites
pass, including broad native HTTP, live IPv6, framing and actual Rust integration.
The live test rejects a bad challenge before dispatch and observes reconnect
waits of 3.003, 4.504 and 6.754 seconds after the initial two-second retry.
Executed source connection/request code with injected failures additionally
confirms that response-task write failures do not propagate out of the receive
coroutine, while application-pong failures and receive timeouts do. Native retry
classification preserves that distinction.

An intermittent packaged test failure reproduced as `EINTR` in the test reader.
The fixture now retries interrupted send/receive/accept calls and deliberately
injects SIGUSR1 to verify that an actually interrupted read resumes. Three
consecutive development runs and the controlled-signal check pass. CUDA builds
now compile the existing package targets together and retain a separate
`native-compiled` stage before validation. All **46 packaged checks** pass, and
both the worker and tunnel-test binaries match the tested build byte-for-byte.
The corrected packaged tunnel test then passes three consecutive runs
(81.37 seconds total).

Image `37dd5b530c1f...` is deployed and healthy with the original `.env`, effective
environment, seven mounts, ports and GPU requests unchanged. Rust registration
and real routed chat, streaming and vision pass, along with all eight public
documentation GET/HEAD comparisons. Entry-point and media-engine hashes are
unchanged. Rollback: `ezlocalai:cuda-before-tunnel-connection`.

Evidence: `tests/fixtures/tunnel_connection_cases.json`,
`tests/fixtures/tunnel_connection_outcomes.json`,
`tests/fixtures/tunnel_connection_validation.json` and
`build/validation/tunnel-connection/`. Local HTTP callbacks still use older URL
handling; these URL cases cover the router WebSocket connection. Redirects,
remaining HTTP parser behavior,
TLS, receive-idle timing and shutdown/deadline behavior remain open. The source
first awaits an existing WebSocket close and then applies its five-second task
wait, so five seconds is not a bound on the entire connected shutdown. The
overall worker protocol and replacement goal remain partial.

## Worker tunnel message framing (2026-10-19)

The worker tunnel now reassembles fragmented WebSocket text and binary messages,
including UTF-8 split across frames, and continues answering control pings
between fragments. Dispatch waits for the final continuation. The previous
reader ignored continuation frames and silently discarded frames above 16 MiB.
The new reader uses the Python tunnel's exclusive 64 MiB complete-message limit,
including cumulative fragments, and rejects oversized declarations before
allocating their payload. Protocol, UTF-8 and size errors send the corresponding
1002, 1007 or 1009 close code and reconnect with fresh reader state.

All **187 cases captured from the original image's compiled aiohttp 3.14.5
reader** pass. They include masked and extended-length frames, invalid control
frames/close codes, incomplete UTF-8, random fragment sequences, valid 17 MiB
messages and a valid message one byte below 64 MiB. A live worker socket test
also passes fragmented request dispatch, interleaved control frames, forwarding
a complete 13 MiB body (over 17 MiB after tunnel base64/JSON), protocol close
codes and reconnect, alongside existing concurrency/cancellation checks.

The actual Rust integration also passes large requests through direct and
tunnel workers, plus registration/authentication, JSON/SSE, restart recovery,
model switching and shutdown (31.11 seconds). Three framing/integration suites
and three broader HTTP/heartbeat/service-client regressions pass. All **45
packaged CUDA checks** pass, and the tested worker matches the candidate binary
byte-for-byte.

Image `1849b6932dd1...` is deployed and healthy with the original `.env`, effective
environment, seven mounts, ports and GPU requests unchanged. Rust registration
and real routed chat, streaming and vision pass. All eight public documentation
GET/HEAD responses still match the Python reference. Entry-point and media-engine
hashes are unchanged. Rollback: `ezlocalai:cuda-before-tunnel-framing`.

Evidence: `tests/fixtures/tunnel_framing_cases.json`,
`tests/fixtures/tunnel_framing_validation.json` and
`build/validation/tunnel-framing/`. Confirmed remaining transport gaps include
upgrade-response validation, backoff after short-lived connections, IPv6 URL
handling and receive-idle timing. TLS and complete shutdown parity also remain
unverified. The generic native server/TTS WebSocket readers are unchanged; the
overall router protocol row and replacement goal remain partial.

## Worker heartbeat recovery and HTTP deadlines (2026-10-19)

Native heartbeat recovery now follows the current Python client. Startup
registration still backs off from two seconds by 1.5, capped at 60 seconds.
After a heartbeat failure, it immediately attempts registration, then waits
the captured heartbeat interval before the next heartbeat, whether that
registration succeeds or fails. Previously it returned to startup backoff and
sent an extra immediate heartbeat after successful re-registration. Interval
configuration is now captured at startup rather than re-read on every loop;
invalid values disable registration without terminating the worker.

Router posts now use the existing HTTP client with a total request deadline,
redirect handling, TLS verification and source-compatible headers. The default
remains ten seconds; the existing native `WORKER_HEARTBEAT_TIMEOUT` override is
retained. A final successful status remains successful when the optional JSON
body is malformed, truncated or times out, matching the source `_post` exception
boundary. Non-success responses and failed redirects remain failures. The
native worker continues to authenticate registration with the router key while
advertising its own API key for callbacks. This intentional distinction from
the Python registration payload permits different router/worker credentials,
as verified by the actual Rust integration tests.

All **61 unchanged-source policy traces** pass, including startup backoff,
repeated failures, re-registration cadence and deregistration. **14 real Python
HTTP response captures** match native socket tests: success/created/no-content,
non-JSON and malformed bodies, gzip, truncation, redirects/loops and a held body.
A live C++ recovery scenario additionally verifies request ordering, retained
interval after environment mutation, stable worker identity and deregistration.
The Python capture used aiohttp 3.14.5 and its normal ten-second deadline; native
socket tests use the existing one-second override for bounded CI runtime, so
those cases assert response acceptance rather than identical elapsed time.

Seven regression suites passed, including the broad native HTTP suite, tunnel,
service clients, metadata and actual Rust integration. After matching the
captured User-Agent, the final heartbeat HTTP and Rust integration suites passed
again (12.83 and 28.56 seconds). Binary comparison caught stale incremental
objects caused by preserved copy timestamps; forced recompilation corrected
that before deployment. The tested worker and clean CUDA candidate now match
byte-for-byte. All **43 packaged CUDA checks** pass.

Image `155da064e55f...` is deployed and healthy with the original effective
environment, `.env`, mounts, ports and GPU requests unchanged. Entry-point and
media-engine hashes are unchanged. Rust registration and real routed chat,
SSE and vision pass, as do all eight public documentation GET/HEAD comparisons.
Rollback: `ezlocalai:cuda-before-heartbeat`.

Evidence: `tests/fixtures/worker_heartbeat_cases.json`,
`tests/fixtures/worker_heartbeat_http_cases.json`,
`tests/fixtures/worker_heartbeat_validation.json` and
`build/validation/worker-heartbeat/`. This covers the recorded retry and HTTP
response behavior. Full metadata/version auditing, cancellation of an in-flight
heartbeat at the source's five-second shutdown boundary, and remaining TLS,
framing and large/fragmented WebSocket cases are still open. The router protocol
row and overall replacement goal remain partial.

## Current worker OpenAPI and startup model defaults (2026-10-19)

The public schema was advertising obsolete request fields and omitted the four
music/music-video routes. It now matches the current source worker app: **31
paths, 34 operations and 22 schemas**, including current image editing,
embedding, music and video fields. The old gap note also mentioned drain/resume,
but those are not routes in the current Python worker app. The legacy Python
router snapshot is outside this refresh and the replacement scope.

`DEFAULT_MODEL`, `EMBEDDING_MODEL` and `WHISPER_MODEL` defaults are now captured
at worker initialization, matching Python class/form initialization. Four
complete schemas captured from the source app match byte-for-byte for defaults,
custom configuration, explicit empty values and Unicode/escaping. Changing the
environment after initialization does not change these schemas, including on
the first request. ReDoc's whitespace/response length now matches the source
FastAPI template as well.

Capture used the original Python image's FastAPI 0.143.0, Pydantic 2.14.1 and
Starlette 1.7.0, importing the complete current app with only `Pipes`
construction substituted to prevent model loading. Real declarations, routes,
OpenAPI generation and TestClient GET/HEAD handling executed unchanged. The
C++ tests contain no Python execution. Eight actual documentation GET/HEAD
responses match status, type, length and body in both an isolated candidate
container and the deployed CUDA service, using its original model settings.

All **41 packaged CUDA checks**, the full native HTTP suite and actual Rust
worker/router integration pass. The broad HTTP suite initially inherited the
image's `/app/models` while creating fixtures under `./models`; it now selects
its own temporary model directory and disables unintended Hub downloads. Two
embedding assertions now check file identity instead of rejecting equivalent
paths containing `/./`. These are test fixture fixes, not production model
selection changes. Full native HTTP passed in 29.46 seconds; Rust integration
passed in 29.69 seconds. Candidate/deployed worker hashes match the tested
binary. The entrypoint and media-engine binaries are unchanged.

CUDA image `60e6f4fe9917...` is deployed and healthy. Its original effective
environment, `.env`, seven mounts, ports and GPU requests are unchanged. Rust
registration and real routed chat, SSE and vision all pass after deployment.
Rollback image: `ezlocalai:cuda-before-openapi`.

Evidence: `tests/fixtures/openapi_cases.json`,
`tests/fixtures/openapi_validation.json`, and `build/validation/openapi/`.
The worker OpenAPI GET/HEAD rows are now implemented. This proves documentation
and schema serving for the current source; inference endpoints retain their
separate partial statuses, and the overall replacement goal remains incomplete.

## Real CUDA speculative decoding and Rust-routed vision (2026-10-19)

The deployed native CUDA image now has real-weight validation for target-only,
MTP and DFlash2 decoding with Qwen3.8-27B Q3_K_XL on the RTX 3090 Ti. Each
controlled worker used the original 220,000-token context, one slot, batch 1,024
and micro-batch 512. DFlash reused the pinned draft in the shared model cache;
its launch selected CUDA0 and matched the target's GPU layer setting.

The new manual C++ `test_speculative_model` runner exercises fixtures captured
from the source project's `benchmark_vision.py` and `benchmark_speculative.py`.
All **60 native vision cases** passed through an isolated actual Rust
WorkerRouter: ten cases, buffered and streamed, for each backend. Coverage
includes 336/1792-pixel images, cached repeats, long generation, text before and
after images, multiple-image ordering, OCR changes and text following vision.
Every stream completed with content and DONE; queues drained afterward.
MTP reported **1,223 drafted / 771 accepted** tokens and DFlash **1,110 / 805**
in these routed cases. A separate production MTP request also drafted 251
and accepted 171 tokens. These counters establish execution, not speed parity.

The unchanged source LLM wrapper, using the original Python container's
xllamacpp 2026.10.11549 binding and the same weights/settings, passed all
**30 reference vision cases**. All **12 direct-engine greedy text outputs**
match their corresponding Python backend exactly. Three of four greedy texts
change between target-only and each speculative mode in both implementations;
this does not represent a native/Python mismatch. Public worker requests use
the source Qwen preset overrides, so their vision checks assert source semantic
patterns rather than deterministic text equality.

The first completed target-only run hit a host file-ownership error while
saving extra metadata. The controller restored production, ownership handling
was corrected, and the completed baseline was reused. All later runs completed
successfully. Production was restored with its original image, configuration
and mounts; health, Rust registration, routed chat, SSE and vision were checked
again. No production implementation or image change was required for this
milestone. The live test runner is excluded from automatic CTest execution.

Evidence: `tests/fixtures/speculative_live_cases.json`,
`tests/fixtures/speculative_live_validation.json`, and full responses/logs under
`build/validation/speculative-live/`. This closes the earlier real-weight and
large-image DFlash verification gap for this configuration. Other model and
quantization combinations, multiple GPUs, ROCm and broader output/performance
parity remain unverified; affected manifest rows and the overall goal remain
partial. LTX/Gemma experiments and historical TTS fallbacks remain excluded
under the scope corrections above.

## Dotenv compatibility and startup ordering (2026-10-19)

The original line-based loader did not expand variables, dropped unknown
backslash escapes, truncated long lines and kept the first duplicate assignment.
Native file loading now follows the Python repo's pinned python-dotenv 1.2.2:
quoted/multiline values, known escapes, Unicode whitespace, comments, variable
and default interpolation, duplicate assignments and process precedence.
`PYTHON_DOTENV_DISABLED` is honored. Invalid UTF-8 or environment values stop
worker/entrypoint startup with a diagnostic that omits configuration values.

The CUDA entrypoint now loads the file before supplying defaults. Comparing
the actual old/new binaries shows that file values for host, port and model/cache
paths now take effect when the process environment does not already define them.
Existing process values, including Docker/Compose settings, still take priority.
Native keeps current-directory/explicit-path selection; Python stack-based
file discovery and warning-log text are not reproduced.

All 121 executed dependency cases pass, including sanitizer checks. An executable
regression verifies interpolated port/model/auth settings, duplicate/process
precedence, encoding failure and the disable switch. The original deployment
`.env` yields exactly the same 34-key environment in Python and C++; comparison
used digests without logging values or modifying the file.

The settings audit also passes 147 defaults, 92 GPU tier names, 536 capability/
model-name cases and 6,183 request/settings/binding cases, with managed engine
payload/reload checks. Eight actual Rust router scenarios and all 40 packaged
CUDA-image checks pass. Image `8a3409aa8ed3...` is healthy with the
original effective environment, mounts, ports and GPU settings preserved.
Live routed chat/SSE/vision and Python-derived metadata pass. Packaged worker
and entrypoint hashes match the tested binaries; the media engine is unchanged.
Rollback: `ezlocalai:cuda-before-dotenv`.

Evidence: `tests/fixtures/dotenv_cases.json`, `tests/fixtures/dotenv_validation.json`
and `build/validation/dotenv/`. Two initial harness issues (suppressed diagnostic
output and a baseline symlink) were corrected before recording the final checks.
This closes the file parsing/startup-order gap; complete runtime configuration
consumers and broader model/GPU replacement acceptance remain open.

## Bounded queue history and accurate status timestamps (2026-10-19)

The native queue now retains at most 1,000 terminal requests, including failures
and cancellations, and releases input payload storage as soon as processing
finishes. Pruning follows enqueue timestamp with a deterministic tie breaker;
active requests are preserved. A waiter already holding the request still
receives its result if the history entry is evicted. Recent results remain
replayable; expired IDs return the existing HTTP 404 response.

Executing Python `RequestQueue.py` confirmed its 1,000-entry target, but serial
successful requests retain 1,001 entries and failure-only traffic never prunes.
The native limit deliberately closes both gaps rather than copying that growth.
History is bounded by count, not by total result bytes. Full UUID/timeout/
cancellation/shutdown parity is not claimed.

A 1,105-request regression fails before the change. Success, failure, queued
cancellation, restart, old active work and attached-waiter checks pass, including
AddressSanitizer/UndefinedBehaviorSanitizer. With 256 KiB inputs and small
results, standalone peak RSS fell from 284.1 MiB to 4.4 MiB. The full HTTP worker
served another 1,105 requests, drained its queue and retained recent status;
its RSS grew 19.6 MiB.

That HTTP run also exposed timestamp rounding: the default output-stream
precision rounded epoch timestamps by thousands of seconds. Serialization
now preserves double precision. A C++ endpoint regression fails before this
fix and verifies the returned timestamp lies inside the request interval.

All five affected suites, eight scenarios against the real Rust WorkerRouter
and all 38 packaged CUDA-image checks pass. Deployed image
`4dbbc3e4b591...` is healthy, preserving the original `.env`, effective
environment, mounts, ports and GPU requests. Live routed chat/SSE/vision and
Python-derived worker metadata pass. The packaged binary matches the HTTP-
tested binary; the LTX engine is unchanged. Rollback:
`ezlocalai:cuda-before-queue-history`.

Evidence: `tests/fixtures/queue_history_cases.json`,
`tests/fixtures/queue_history_validation.json` and `build/validation/queue-history/`.
This closes the completed-request retention gap recorded below. Broader
replacement acceptance remains open under the Qwen-only TTS and complete-LTX
scope corrections above.

## Native auxiliary video handoff and embedding cache selection (2026-10-19)

Video handoff now reaches the actual native Qwen TTS and STT pools and image
wrapper, alongside the embedding backend. The prior backend-only path missed
those owners. Pool destruction rechecks active leases; image cleanup/restoration
uses a nonblocking generation lock to avoid an image/LLM lock inversion.
Restoration follows the existing source-derived preload policy and preserves
which owner each removed model belonged to. Partial cleanup state survives an
exception, and unused allocator memory is released before loading video.

Seven C++ scenarios cover idle cleanup, video failure, active TTS/STT/image,
retained-model mode and a failed TTS warm load while the other models restore.
The idle-pool regression fails before the fix. The 2,769 Python GPU/placement/
policy cases and 3,057 auxiliary cases pass, along with affected slot/headroom/
residency/embedding suites and eight actual Rust integration scenarios.
An older test now pins its temporary HF cache instead of inheriting `/app/models`.

Real startup exposed a second defect: recursive embedding discovery selected
ACE-Step's incompatible same-named GGUF even though the proper embedding model
was cached. Discovery now prefers the model's own directory and breaks fallback
ties deterministically. Explicit file overrides remain authoritative. The
reproduction fails before the change and passes afterward; the live worker
loads the proper cached embedding and produces embeddings successfully.

The CUDA run starts the original LLM model with an isolated 4,096-token context,
plus real Qwen TTS, STT, embeddings and the image wrapper. Speech synthesis,
transcription and embedding inference pass before and after video. Three LTX
requests through a Rust tunnel cover success, interrupted-engine failure and
recovery (client 200/502/200, with worker 500 wrapped by Rust). During video,
all auxiliary registrations disappear and TTS/embedding children are released;
after each request, pools return to one resident model, the image wrapper and
LLM restore, and queued/subsequent chat succeeds. First/recovered MP4, decoded
video and audio match. Resource responses stay below
1.817 seconds in the recorded samples.

All 36 packaged checks pass. Image `4a2695036442...` is healthy with the
original `.env`, effective environment, mounts, ports and GPU settings preserved.
Live routed chat/SSE/vision and Python-derived worker metadata pass. The packaged
worker matches the real-test binary; the LTX engine is unchanged. Rollback:
`ezlocalai:cuda-before-video-auxiliary`.

Evidence: `tests/fixtures/video_auxiliary_validation.json`,
`build/validation/video-auxiliary/` and `build/validation/video-auxiliary-live/`.
This closes the owner-wiring gap identified in the preceding status milestone.
Larger presets/output quality, multiple GPUs and broader simultaneous real-model
workloads remain unverified. Qwen-only TTS and the LTX scope correction still
apply; historical Gemma/Quanto experiments are not acceptance requirements.

This milestone identified unbounded completed-request retention as the next
memory-management gap. The later bounded-queue milestone above closes that gap;
broader queue lifecycle semantics remain under audit.

## Responsive router status during LLM restoration (2026-10-19)

Resource status and worker metadata previously waited on the LLM startup lock.
A real LTX handoff exposed an eight-second resource-request timeout and roughly
15 seconds between heartbeats while the LLM reloaded. Status observers now use
an immutable snapshot during startup. Admission and handoff decisions still
wait for current model state. Embedding-pool publication also avoids holding
its status lock while waiting on LLM startup.

Two controlled C++ regressions cover restoration with and without concurrent
embedding cleanup, and verify that action reads still wait. The test fails
before the fix. All four affected suites, eight scenarios against the actual
Rust WorkerRouter binary and all 34 packaged CUDA-image checks pass. The
packaged worker matches the binary used by the real GPU test; the LTX engine
is unchanged by this fix.

Three real CUDA requests across two isolated Rust-tunnel runs verify success,
interrupted-engine failure and subsequent recovery. Client statuses are 200/502/200 (the router wraps worker 500 as 502);
all three reap the video children, restore the original LLM and serve queued
and subsequent chat. The first and recovered MP4, decoded video and audio
match exactly. Maximum sampled resource latency is
0.073 seconds; maximum heartbeat gap is
2.936 seconds. The final test disables router retries to expose the
worker error; an earlier harness left retries enabled and correctly observed
worker 500 followed by successful router retry, which invalidated that
harness's expected-status assertion.

Image `710e4d7616d8...` is deployed healthy with the original `.env`,
effective environment, mounts, ports and GPU settings preserved. Live routed
chat/SSE/vision and Python-derived worker metadata pass. Rollback is
`ezlocalai:cuda-before-llm-observers`. Evidence:
`tests/fixtures/llm_observers_validation.json`, `build/validation/llm-observers/`
and `build/validation/ltx-handoff-final/`.

This covers small 64x64/nine-frame/two-step clips on one GPU. It does not prove
larger-preset quality or auxiliary handoff: speech, embeddings and images are
disabled in this real-model probe. Source inspection confirms an outstanding
auxiliary wiring gap: `unload_aux_models_for_video` delegates to the backend,
whose real implementation exposes/unloads embeddings but not the native Qwen
TTS, STT and image service owners. Their unload/restore wiring and concurrent
lease behavior need correction and validation. This is current acceptance work;
Gemma storage alternatives and Quanto remain excluded by the scope above.

<details>
<summary>Historical LTX/Gemma engine experiments and validation (superseded scope)</summary>

## Separate LTX connectors deployed (2026-10-19)

Connector preprocessing now runs in its own engine graph. Its staged weights
are released before transformer execution, and two bounded cache entries retain
positive/negative conditioning across denoising steps. Exact owned tensor bytes,
shape, audio mode and scale settings identify each entry. Mutable adapters bypass
reuse; failures and sampling-session cleanup invalidate it. CPU and CUDA graph
tests cover slicing, changed inputs, device-cache flushes and cleanup through the
base engine interface. Standalone identity tests cover bounded eviction and
non-reusable results. Docker builds run the CPU graph check with CUDA driver
stubs; actual GPU checks run separately.

The real 24 GB GPU test now retains the original engine process in model-offload
mode through both requests. Preload takes 48.07 seconds; generation takes 30.01
and 17.47 seconds, compared with 104.97 and 22.44 seconds before this change.
Both MP4s, decoded RGB and decoded PCM match the prior engine exactly. Sampled
combined wrapper/engine GPU usage peaks at 21,126 MiB and returns to 7,822 MiB
between requests. This whole-model path trades higher peak memory for avoiding
the failed attempt and reload. The container memory limit remains 28 GiB.

Five actual HTTP requests, four through a separate Rust WorkerRouter tunnel,
exercise changed prompts, dimensions, first-image and positioned conditioning,
then repeat the first request. One packed-NF4 engine context serves them all;
the repeated MP4/video/audio bytes match, active resource usage clears, and
teardown reaps the process. This dedicated media-worker run does not establish
quantized LLM/video handoff parity.

All 34 packaged tests, eight CUDA manager cases and the real CUDA connector
checks pass. The packaged engine and 18 recorded source hashes match the tested
build. Image `fef85f4523ee...` is deployed healthy with the original
`.env`, effective environment, seven mounts, ports and GPU settings preserved.
Routed chat/SSE/vision and Python-derived worker metadata pass. Rollback is
`ezlocalai:cuda-before-ltx-connectors`.

At this milestone, LLM handoff, larger presets and multiple GPUs remained
unverified. Quanto CPU INT8 and exact experimental module-call order are no
longer requirements under the scope correction above. Evidence:
`tests/fixtures/ltx_connectors_validation.json`, `build/validation/ltx-connectors/`
and `build/validation/ltx-connectors-http/`. The previous offload deployment below
is historical evidence.

## LTX post-load placement and sequential execution deployed (2026-10-19)

The CUDA worker now loads Gemma before selecting video placement. It queries
free memory on the selected GPU again and combines that observation with the
successful NF4 selection and the original generation hint. Model/sequential
policy changes reuse the retained context; changing full parameter placement
still rebuilds it. Quantized Gemma, including norms and embeddings, stays on
GPU while the other modules retain their CPU placement.

The engine now distinguishes whole-graph execution from forced graph segments.
Sequential mode disables prefetch and immediately evicts staged weights at each
segment boundary. Shared connector outputs and timestep adaptations are cached
at the graph prelude, avoiding repeated dependency execution in every layer.
Actual LTX plans execute 32,647 nodes against 32,631 original nodes across 50
segments. Cut caches are scoped to one graph, including failure cleanup.

Validation:

- Forty-five captured Python constructor cases validate post-load placement,
  hint handling, selected-device precision and placement fallback. These use
  synthetic model loading, so they do **not** prove Quanto tensor behavior.
  Two source cases that ignore failed sequential-hook installation are excluded.
- Eight real CUDA model-manager cases cover policy transitions, actual segment
  eviction, retained Gemma buffers/bytes, independent projector reload and
  cleanup. Six affected host suites pass, including eight scenarios against
  the actual Rust WorkerRouter binary. All 32 packaged CUDA checks pass, and
  the packaged engine and 14 recorded source hashes match the tested build.
- The real C++ wrapper selects model offload after preload. Its whole graph
  needs 17,723 MB against roughly 15,028 MB available, fails cleanly, and
  restarts with sequential execution. The successful replacement serves both
  requests and is reaped at teardown. A 28 GiB container limit remains in force.
- Two 64x64/nine-frame/two-step clips match the prior sequential baseline byte
  for byte as MP4, decoded RGB and decoded PCM. The shared-output fix reduces
  the repeat request from 256.10 to 22.44 seconds. The first request takes
  104.97 seconds including the failed attempt and model reload; preload takes
  45.61 seconds. This is a native implementation comparison, not a Python
  performance benchmark.
- Half-second process observations peak at 13,480 MiB, including the wrapper's
  CUDA context. Idle observations are 7,744 MiB after preload and 7,822 MiB after
  generation. These are sampled usage figures, not allocator-level peaks.

Image `0f9cfcac3897...` is deployed healthy with the original `.env`, effective
environment, seven mounts, ports and GPU requests preserved. Routed chat,
SSE and vision pass; one matching Rust worker advertises the Python-derived
metadata. Rollback is `ezlocalai:cuda-before-ltx-offload`.

Full parity remains open. Python runs connectors separately before denoising;
native model offload still combines connector and transformer work in one
graph, and connector reuse currently stops at the graph boundary. Quanto CPU
INT8, quantized HTTP/router handoff with retained auxiliary models, larger
clips/multiple GPUs and complete model numerical equivalence remain unverified.
Evidence: `tests/fixtures/ltx_offload_validation.json`, the separate slow-baseline
record `tests/fixtures/ltx_offload_baseline_validation.json`, and
`build/validation/ltx-shared-cuts/`.

## Separate Gemma residency: engine prototype verified (2026-10-19)

The next engine patch keeps the native quantized Gemma's parameters on its
compute device, including unquantized embeddings and norms. Other conditioner
parameters retain their configured CPU/disk placement. Graph eviction can
release and reload the projection module independently; unregistering the
conditioner releases all parameters. Memory reporting now accounts for each
parameter's backend instead of assigning the entire conditioner to one place.

Eight real CUDA `ModelManager` cases pass across quantized/unquantized weights,
direct/mmap loading and CPU/disk placement. They verify buffer placement and
identity, preserved bytes after eviction, independent projector reload,
complete unregister cleanup and unchanged source files. The private engine
protocol regression test also passes.

Real LTX-2.3 preloads in 72.19 seconds under a 28 GiB container limit, with
7,216.21 MB of registered parameters on GPU and 20,963.62 MB on CPU. Two
64x64/nine-frame/two-step generations reuse the context in 71.90 and 48.65
seconds. Decoded frames and PCM audio match each other and the prior packed-NF4
CPU-parameter test. Half-second observations show 7,482 MiB of GPU usage after
preload, 7,560 MiB during the first three-second post-request idle interval,
and a sampled maximum of 21,982 MiB. The final completed-request phase also
includes teardown; its minimum is not an idle-residency measurement.

This initial prototype was **not deployed**. At that point the wrapper still
selected residency before loading and used the same engine policy for
model/sequential modes; the subsequent offload-policy work addresses those gaps.
The full-model probe uses automatic engine graph segmentation; it does not
prove source-equivalent offload policy. Production image `ff99ef0255b0...` is
restored healthy, with routed chat/SSE/vision passing. Evidence:
`tests/fixtures/gemma_residency_validation.json`,
`tests/test_ltx_gemma_residency.cpp`,
`patches/sdcpp-ltx-gemma-residency.patch` and `build/validation/gemma-residency/`.

## Packed NF4 model integration deployed (2026-10-19)

The raw LTX Gemma loader and `Linear` blocks now use packed NF4 parameters.
Separate compute-precision metadata follows name conversion, type overrides,
lazy reader clones and model-source validation. Loading converts and quantizes
each projection into its packed destination without modifying checkpoint
mappings. Placement estimates account for packed sizes. Linear graphs expand
weights for computation; unrelated embeddings, norms and tensorwise INT8
weights retain their existing paths.

Validation so far:

- Thirty-two loader/Linear cases and two full Gemma projection matrices match
  Python-derived expanded weights and same-backend matrix products. Tests cover
  FP32 conversion, BF16/FP16, mmap, lazy reads, precision selection before/after
  discovery, source preservation and invalid destinations. Six affected host
  suites pass in 1.40 seconds.
- Real LTX-2.3 preloads in 66.99 seconds under a 28 GiB container limit while
  the production LLM remains loaded. Registered text-encoder parameters fall
  from 24,648.41 MB to 9,421.23 MB; total parameters are 28,179.82 MB. These are
  engine parameter accounting figures, distinct from process RSS or peak GPU
  usage. The retained context is successfully reaped.
- With production temporarily stopped to free the GPU, a second preload takes
  71.35 seconds and two 64x64/nine-frame/two-step generations take 25.77 and
  17.92 seconds. Both reuse the same context and produce matching decoded RGBA
  frames and PCM audio. Sampled GPU usage reaches 17,564 MiB; two-second polling
  is not an allocator-level peak measurement. Parameters remain on CPU under
  the existing offload policy, with transient CUDA execution.
- All 32 packaged CUDA checks pass in 76.94 seconds. The finished image passes
  the 34 loader/Linear cases and 146 CUDA operator cases, including both full
  Gemma matrices. Its engine binary exactly matches the binary used for the
  real generations above, and all recorded first-party/engine source hashes
  match. The image contains no Python executable.
- Deployed image:
  `sha256:ff99ef0255b045fc073230ef732561d152bd2b90d1c7d0fa92eca7a7b17a0b08`.
  Engine SHA-256:
  `22061ddae6dce05946f49fcb4cde587f9676e4505125799a9a97ec0e14d2831a`.
  The native worker executable is unchanged. The original `.env`, effective
  environment, mounts, ports and GPU requests are preserved. Health, live
  Rust-routed chat/SSE/vision, one-worker registration and Python-derived
  worker metadata pass. Rollback: `ezlocalai:cuda-before-nf4-packed`.

Evidence: `tests/fixtures/nf4_integration_validation.json`,
`tests/test_ltx_nf4_loader.cpp`, `patches/sdcpp-ltx-nf4-packed.patch` and
`build/validation/nf4-integration/`. This closes the loader/Linear selection
gap in the earlier operator milestone below. It does **not** establish full
placement parity: Python retains quantized Gemma separately while offloading
other pipeline components. Quanto, distinct offload behavior, full Python
output equivalence and actual quantized worker/router handoff remain open.

The next placement change is grounded in 47 newly executed Python constructor
cases in `tests/fixtures/ltx_placement_cases.json`. They execute `_load_pipeline`
and its residency helper with controlled CUDA observations and model/pipeline
stubs, covering the 12 GiB NF4 threshold, BF16/FP16 selection, NF4/Quanto/plain
fallbacks, post-load free-memory selection, full/model/sequential failures and
shared encoders/hooks. Quantized encoders are excluded from offload hooks;
Quanto is restored through the CPU bridge. These are source expectations,
**not native passing cases or real memory measurements**. The current native
constructor selects residency before loading and does not keep quantized
Gemma separate from other conditioner parameters. Capture script/log:
`build/validation/ltx-placement/`.

Installed Quanto source was also inspected (inventory and excerpts under
`build/validation/quanto/`). Its symmetric weight quantizer uses per-output
scales and rounded/clamped INT8 values. For BF16 CPU activations and input
widths divisible by four, its linear operator dispatches to PyTorch's
`_weight_int8pack_mm`; other cases use a dequantized floating-point path.
The pinned engine's existing tensorwise INT8 operation has no CPU
implementation. Reusing its name or storage type therefore does not establish
the historical Quanto CPU fallback. Implementing that experimental fallback
is no longer required under the scope correction above.

## Packed NF4 CPU/CUDA operator: initial validation (2026-10-19)

Added packed NF4 storage containing the value nibbles, nested statistics,
nested maxima and scalar offset. A private GGML operation expands this storage
to BF16/FP16 on CPU or CUDA when a projection is evaluated. Parameter buffers
retain the compressed representation; the operation does not retain expanded
scales. CUDA recognizes only this tagged custom operation and leaves unrelated
custom operations unsupported. Contiguous byte views may be unaligned.

CPU and CUDA tests compare expansion bytes against installed Python
bitsandbytes results and compare downstream matrix products with the same
backend using those source-expanded weights. They check independent output
buffers, preservation of compressed inputs, invalid shapes/storage and
unrelated custom operations. Coverage includes 136 existing nested cases,
eight newly captured cases spanning every value/statistics code, and two full
Gemma projection matrices: 146 cases per backend. The real query and gate
projections occupy 8,113,924 and 30,427,204 packed bytes respectively, versus
31,457,280 and 117,964,800 expanded BF16 bytes. These are parameter sizes,
not complete graph or pipeline peak-memory measurements.

Five affected host suites pass in 1.24 seconds. All 32 packaged checks pass
in 83.69 seconds. The final CUDA image also passes all 146 cases on the
RTX 3090 Ti under a 2 GiB host-memory container limit, including both full
projection matrices. Candidate image:
`sha256:aeca7dcda94912404ebebe232dcbe3dabdbe831857ec2b5bce47a11aa7130ed1`.
Its native executable is unchanged from production, its patched engine is
new, and it contains no Python executable. It has not been deployed.

**Model integration remains pending.** The LTX loader and Linear blocks still
use the earlier quantize/dequantize path with expanded resident weights. This
operator alone does not establish compressed full-model residency, Python
matrix-kernel equivalence, Quanto parity, distinct offload behavior or real
quantized LTX/LLM handoff. Production continues running the retained-LTX image
recorded below. Evidence: `tests/fixtures/nf4_storage_cases.json`,
`tests/fixtures/nf4_storage_validation.json`, `tests/test_nf4_storage.cpp`
and `build/validation/nf4-storage/`.

</details>

## Retained LTX startup and request lifecycle deployed (2026-10-19)

`LtxVideo` now owns the retained diffusion process validated below. Construction
prepares assets and a real model context before registration; subsequent
requests reuse that context. Placement and text-weight policy are captured at
load time, using the initial raw generation hint on lazy loads. Startup warms
enabled LTX after embeddings and before image preparation, subject to the
source forwarding and LLM-handoff gates. Music-video sessions load from the
first scene's hint and retain the context across scenes. Qwen-image and Wan
continue using their source per-request process lifetimes.

Forced handoff, idle reclamation and destruction release the LTX process.
Inference errors, timeouts and malformed private replies discard it and clear
its resource registration; a later request preloads a fresh process. Failed
startup also remains retryable. NF4-conversion and CUDA placement retries
start with clean processes. Registry entries use the source 12 GiB estimate,
which is not a measurement of GPU allocations. Resource readers use published
state and remain responsive while model loading/generation holds its own lock.

Validation:

- Twenty-four executed Python lifetime cases check preload gates, retained
  contexts, changed request sizes/frame counts, explicit/automatic handoff,
  keep-loaded configuration and usage cleanup. Native tests verify actual
  process/context counts and teardown, startup/inference/protocol failure
  recovery, text precision/NF4 policy, conditioning, exports, HTTP `n` and
  multi-scene music-video behavior. The final LTX suite passes in 18.97 seconds.
- Seven affected host regression suites pass in 74.28 seconds, including eight
  actual C++/Rust direct/tunnel scenarios. All 31 packaged CUDA checks pass in
  82.41 seconds; the LTX lifecycle suite passes there in 25.17 seconds.
- Real LTX-2.3 startup completes in 31.61 seconds with one
  context under a 28 GiB container memory limit. The existing LLM stays present
  during preload, leaving 1793 MiB of GPU memory; this
  selects unquantized BF16 text weights. After the LLM is stopped, the context
  retains its initial offload policy. Five two-step video requests cover direct
  HTTP, changed prompt/96x64 dimensions/17 frames, image input, positioned
  conditioning and a repeat of the original 64x64/nine-frame request. Four
  requests traverse an isolated Rust WorkerRouter WebSocket tunnel. Every
  request uses the same process/context, produces video plus 24 kHz audio,
  clears active usage, and keeps resource requests responsive. The repeated
  video's decoded RGBA frames and decoded PCM audio match after the intervening requests. The router
  observes one active video reservation; teardown reaps the engine.

Deployed image: `sha256:48264d550a35b0216795568751e47b2adda554940ac7708563e3de99314ed584`.
Native executable SHA-256: `61d576c6153be35c1e68de53e65c66bedffdf9805d32fd7f677befff54a064b5`.
The original `.env`, effective environment, seven mounts, port bindings and
GPU requests are preserved. Live routed chat, SSE and vision pass, and the
single production worker's extra metadata matches Python. Rollback image:
`ezlocalai:cuda-before-ltx-resident`.

Evidence: `tests/fixtures/ltx_lifetime_cases.json`,
`tests/fixtures/ltx_lifetime_validation.json`, `tests/test_ltx_runtime.cpp`,
`build/validation/ltx-resident/` and `build/validation/ltx-resident-gpu/`.
The integration gap described in the earlier retained-engine milestone is
closed. **Overall parity remains incomplete:** compressed NF4/Quanto GPU
storage, distinct offload modes, source-exact full-model output, real quantized
LTX/LLM handoff and broader model/preset/GPU coverage remain open. Matching
repeated native output does not establish Python output equivalence.

## Retained diffusion engine: initial validation (2026-10-19)

Added a C++ process layer and a private worker mode to the pinned diffusion
engine. The worker creates a real `sd_ctx_t`, retains it across generation
requests, and replaces it when model/context arguments change. Generation
arguments are refreshed for each request. Preload constructs the context
without generating output and requests eager parameter loading. The engine can
disable eager loading specifically for graph-cut layer splitting; this option
alone must not be treated as evidence of full physical residency in every mode.

The inherited Unix socket protocol has versioned, size-bounded frames and an
absolute request deadline. Logs remain file-backed. Model failures, protocol
errors and timeouts discard and reap the process; shutdown and parent death
also release it. A shared context marker makes actual load/reuse observable.
Normal one-shot `sd-cli` invocation remains supported.

Validation:

- The C++ process test passes (0.88 seconds), covering context reuse/replacement,
  preload, 2 MiB literal prompts, descriptor isolation, fragmented/oversized
  messages, timeouts, model errors, crashes, restart, teardown and parent death.
  All 30 packaged CUDA tests pass (54.10 seconds).
- Real Qwen-image weights preload in 3.63 seconds and serve two 256x256,
  two-step generations in 0.98 and 0.53 seconds on the same process/context.
  Their decoded RGBA pixels match. Observed engine GPU use is 9,224–9,270 MiB.
  Qwen production behavior remains one-shot, matching Python; this is a probe
  of the retained engine layer.
- The real LTX-2.3 checkpoint with an unquantized BF16 text encoder preloads
  in 30.35 seconds. The engine reports 43,407 MiB of parameter storage, with
  CPU offload and file mappings operating under a 28 GiB container limit.
  Two 64x64, nine-frame, two-step generations take 42.52 and 44.24 seconds on
  the same process/context. All nine corresponding RGBA frames and the
  63,404-byte stereo WAVs match between requests. The highest sampled GPU
  allocation is 22,028 MiB. Both real engine processes are reaped on teardown.

Candidate image (not deployed):
`sha256:820aac8bf6cc7d18af39e32541c14fe7b271e90ce5b7b763e8e63f05752e460c`.
Candidate `sd-cli` SHA-256:
`8ae0be1e2f3118397ee47d112955b8caff414242eeb37e007ce1fd2d3516b043`.
The original production image is restored healthy and unchanged. Live routed
chat, SSE and vision pass, and its single worker's extra metadata matches
Python.

Evidence: `tests/fixtures/media_engine_validation.json`,
`tests/test_media_engine.cpp`, `build/validation/media-engine/` and
`build/validation/media-engine-model/`. At this initial milestone the layer was not yet connected to `LtxVideo`;
startup and HTTP generation still used the older path. That integration and
varied-request validation are now recorded above. Compressed NF4/Quanto
storage, distinct offload modes and source-exact full-model outputs remain open. Successful repeated candidate outputs do not
establish Python output equivalence.

## Wan and Qwen image startup preparation deployed (2026-10-19)

Direct backend inspection corrected the previous residency audit. Python
`IMG.py` and `WAN.py` both launch a short-lived `sd-cli` for each generation.
Qwen image construction checks/downloads its four assets; Wan construction
validates the model and binary. Only `LTX_VIDEO.py` retains a diffusion pipeline.
Earlier references to persistent image/Wan engines as a prerequisite for parity
were incorrect.

The native executable now prepares enabled local Wan and Qwen-image wrappers,
in that order, after embeddings and before opening listeners. It applies the
source forwarding and LLM-handoff gates, registers Wan at 0 GiB and preserves
Python's 4 GiB image estimate. These are resource estimates, not measurements
of resident weights. Python registers even an unavailable image wrapper; native
startup now does the same. Failed Wan preparation can retry on a later request.
Wan lazy loading/restoration also registers the wrapper, active generation
marks its resource entry, and handoff/error cleanup removes it.

Validation:

- 86 cases execute the actual Python media startup block, handoff/client methods
  and Wan/IMG constructors with cached fixture assets. Native checks also verify
  embedding-before-media order, no startup engine launch, failed-Wan repair/retry
  and cleanup. Ten concurrent Wan sessions verify resource activity during
  generation and release afterward, including generation/restoration failures.
- All eight host suites pass (95.04 seconds), including all eight direct/tunnel
  scenarios against Rust WorkerRouter. All 29 packaged CUDA checks pass
  (55.20 seconds).
- A real dedicated-media CUDA worker prepares both wrappers without starting
  `sd-cli`. Two 256x256 Qwen-image requests complete in 8.08 and 7.86 seconds;
  each returns a valid 122,467-byte PNG and reaps its child before the next
  request. Worker GPU usage is 260 MiB before and after generation. The original
  service is restored healthy after the isolated test.

Deployed image:
`sha256:35b8ac0bcd30288f29db5970d91c92142a29cc1576151cf6bfa2557f97703766`.
Native executable SHA-256:
`e78aeb9a398e51c665f691e5d0f272d78a8171282963fa8e3f3174c7a3c35655`.
The original environment file, effective settings, seven mounts, ports and GPU
requests are unchanged; the container contains no Python interpreter. Live
routed chat, SSE and vision pass with the original Qwen3.8 configuration, and
one matching worker publishes extra metadata matching Python.
Rollback: `ezlocalai:cuda-before-media-startup`.

Evidence: `tests/fixtures/media_startup_cases.json`,
`tests/fixtures/media_startup_validation.json`, `tests/test_media_startup.cpp`
and local logs in `build/validation/media-startup/`. Full replacement parity is
still incomplete. LTX persistent pipeline preload, compressed GPU storage and
offload behavior, and broader video/model/GPU validation remain open. Startup
explicitly leaves LTX lazy rather than reporting its settings wrapper as a
preloaded pipeline.

## Text, speech and embedding startup order deployed (2026-10-19)

The executable now uses one tested initialization path: configured LLMs, Qwen
TTS, STT, then embeddings, before opening listeners. Previously embedding
warmup ran before the speech pools, changing the memory available for placement.
LLMs preload by default, matching Python; the explicit native
`EZLOCALAI_PRELOAD_MODEL_ON_START=false` option still requests lazy startup.
Dedicated voice-only mode and an explicit text-server URL suppress local LLM
preload. Explicit voice+text mode retains it. Image mode follows the actual
source condition, which still allows configured LLMs to preload.

Reinspection corrected an earlier audit note: the lazy Qwen cache-warm branch,
precache-marker check and partial auxiliary-pool failure handling already
existed. They were verified through the new startup path, not newly implemented.
Successful pool entries survive a later preload failure, and later startup
stages still run. An uncaught lazy-cache initialization failure stops startup.

Validation:

- 178 traces execute the Python constructor branches, real service clients,
  voice policies and pool warm-load methods with controlled loaders. They match
  load order, skipped stages, errors and resident pool counts. Three native
  option checks and five real executable mode checks also pass.
- All 11 regression suites pass, including speech lifetime, TTS/STT pools,
  embeddings, resource/client metadata, preload and all eight C++/Rust direct
  and tunnel integration scenarios. All 28 packaged CUDA checks pass.
- A real dedicated-voice CUDA worker leaves the configured LLM unloaded and
  preloads one Qwen TTS, one Whisper-base and one Qwen embedding instance on
  CUDA. Qwen's process starts before the embedding engine. GPU process memory
  is 3,516 MiB for Qwen, 2,344 MiB for embeddings and 470 MiB in the worker
  hosting Whisper. Real 32-dimensional embeddings, a 53,804-byte WAV and the
  expected JFK speech transcription complete successfully; all pools stay
  resident afterward.

Deployed image:
`sha256:c2063de651f001e9e96bf5344a66f3224a77147a0c8180bd7f9decfe55e55f71`.
Native executable SHA-256:
`9739ffa835d358cbff8f0d54f08140d4a0f36a9be48f6ea703c82dce23fee51a`.
The original environment file, effective settings, seven mounts, ports and GPU
requests are unchanged. The container has no Python interpreter. Live routed
chat, SSE and vision pass with the original Qwen3.8 configuration, and its
single matching worker's extra metadata matches Python.
Rollback: `ezlocalai:cuda-before-startup`.

Evidence: `tests/fixtures/startup_cases.json`,
`tests/fixtures/startup_validation.json`, `tests/test_startup.cpp` and local
logs in `build/validation/startup/`. At this milestone VIDEO then IMG startup
remained open. Subsequent source inspection corrected the residency assumption:
Python Qwen image and Wan also launch a fresh `sd-cli` for each request; only
LTX retains a diffusion pipeline. Their wrapper preparation is valid preload
parity, whereas LTX requires a persistent native context. Compressed GPU/video
and broader GPU behavior remain partial.

## Resident instance metadata deployed (2026-10-19)

The worker now publishes the number of resident copies of each public LLM,
independently of its parallel inference slots. Python emits `instances` only
when more than one copy is resident and the worker is outside swap/unavailable
states. Text inherits the count when there is one public model; vision inherits
it when there is one public vision model. WorkerRouter uses this metadata to
admit requests to otherwise idle replicas while another replica is busy.

The backend publishes residency alongside runtime capacity without adding the
model-loading lock to capacity reads. Failed preloads, unloads and restoration
update the count; detaching an external engine also clears local residency.
Video exclusivity preserves the raw instance count while setting capacity to
zero, matching the source snapshot. Actual model unavailability removes it.
Incoming router metadata retains its existing normalization rules.

Validation:

- 88 full snapshots execute Python's slot-reporting and resolved-parallel
  methods, covering duplicate groups, resident counts, failed replicas, vision
  groups and swap mode. Managed engines additionally verify explicit duplicates,
  automatic replicas, failed preload, process cleanup and unload/restore.
- Four additional video sessions check instance metadata through generation,
  exclusivity and restoration against captured Python transitions. The media
  suite retains its previous failure and concurrency checks.
- All seven final host suites pass. C++/Rust integration passes all eight direct
  and tunnel scenarios, including two concurrent requests routed to separate
  surviving engines, both router reservations, restart and deregistration.
  WorkerRouter's own heartbeat-normalization and idle-routing tests also pass.
- All 27 packaged CUDA checks pass. The deployed CUDA worker passes live routed
  chat, SSE and vision with the original Qwen3.8 configuration. Exactly one
  matching worker is registered, and its extra metadata matches Python.

Deployed image:
`sha256:8a3685ddf681a2fd3591b0439b5a5b3647cfaaea5926f7f19f488356de2d1ce0`.
Native executable SHA-256:
`02ffc40d48e2f0fd3bffc5e5665c6b76f5224ecac1baf59a0e5aa9ec9262fd4c`.
The original environment file, effective settings, seven mounts, ports and GPU
requests are unchanged, and the container has no Python interpreter.
Rollback: `ezlocalai:cuda-before-instances`.

Evidence: `tests/fixtures/llm_instance_cases.json`,
`tests/fixtures/llm_instance_validation.json`, `tests/test_llm_instances.cpp`,
the extended media/router tests and local logs in `build/validation/instances/`.
The next confirmed startup gaps are mode-based LLM preload suppression and moving
embedding warmup after Qwen TTS/STT initialization. Reinspection found that the
lazy Qwen cache-warm branch and precache-marker check already exist; their
constructor-order behavior still needs verification. Full video/image startup ordering depends on remaining persistent media
engine work. Compressed GPU/video and broader GPU behavior also remain partial.

## Runtime slots and dynamic queue capacity deployed (2026-10-19)

Managed engines now publish their resolved parallel count after readiness.
Admission, model/text slots and router per-slot context use that count, including
automatic sizing at the effective context and explicit engine argument overrides.
Unloading returns to configured capacity; restoring publishes the new runtime
count. The active model supplies the resource context fields, while external
endpoints retain configured context fallbacks when their runtime count is unknown.

The local queue refreshes at startup, enqueue and resource reads, matching
Python's call sites and response ordering. Shrinking preserves active requests
and holds queued work until capacity is available; growing starts additional
work. Worker heartbeats observe capacity without resizing the application queue.
Runtime single-slot engines retain cancellation leases until generation finishes,
and the memory-retry guard also uses the resolved count. Capacity reads remain
available during model startup, without taking the model-loading mutex.

Validation:

- Twelve transitions execute the unchanged Python request queue and application
  capacity-sync function. C++ checks cover growth, shrink with active work,
  zero normalization, shutdown, restart and failed allocation.
- Managed subprocess tests verify auto sizing, unload/restore, resource response
  ordering, enqueue refresh and cancellation when a four-slot configuration
  resolves to a single-slot engine. An external-endpoint regression found in
  the initial 20-suite run was fixed; all six affected suites pass after the
  correction. The other 19 suites passed the initial run.
- All 89 Rust wire tests pass. Actual C++/Rust integration passes eight scenarios
  across direct and tunnel transports, including resolved slot/context metadata,
  partial preload, swapping, router restart and deregistration.
- All 26 packaged CUDA checks pass. Real Qwen3.6-35B-A3B at a 65,536-token context
  resolves automatic parallelism to two slots. Two concurrent requests each
  generate 128 tokens in 1.93 seconds together; one engine uses 16,576 MiB and
  both slots return idle afterward.

Deployed image:
`sha256:ecf981fe1b0cc3e2d6a8df768bbde4a6f81ffd81f773890b1c52fe8159445550`.
Native executable SHA-256:
`8a232ba57108e6da455429321d7367920cd3d2e2d8dd6fa8c17a618d32b90488`.
The original environment file, effective settings, seven mounts, ports and GPU
requests are unchanged. The container has no Python interpreter. Live routed
chat, SSE and vision pass with the original Qwen3.8 configuration, and the
single registered worker's extra metadata matches the Python calculation.
Rollback: `ezlocalai:cuda-before-capacity`.

Evidence: `tests/fixtures/live_capacity_cases.json`,
`tests/fixtures/live_capacity_validation.json`, `tests/test_live_capacity.cpp`,
`tests/test_worker_router.cpp` and local logs in `build/validation/capacity/`.
Loaded instance-count metadata used by Rust idle routing is a confirmed next
gap. Full pipeline initialization/preload ordering, broader GPU behavior,
persistent media engines and remaining compressed GPU/video parity remain open.

## Failed replica preloads preserve worker startup (2026-10-19)

Managed LLM startup now catches each model's preload failure and continues with
the remaining models. Only automatically generated replicas that fail preload
become unavailable. Configured originals, including explicit duplicate model
entries, remain eligible for a later request to retry loading. Swap mode still
attempts only the first model and leaves deferred replicas unmarked.

The backend supplies actual preload outcomes to admission and resource/router
slot reporting. Failed automatic replicas are excluded from public model and
text capacity and from request selection. An explicit request for one fails
with the source error instead of waiting for an impossible lease. The local
queue starts with the surviving configured slot total, or one slot in swap
mode, instead of the unadjusted environment parallel count.

Validation:

- 138 cases execute the unchanged Python constructor preload branch and match
  attempts, successful registrations and failed automatic-replica sets.
- Managed subprocess tests verify continued preload, preserved original/explicit
  duplicate retries, failed-replica rejection, deferred swap instances, resource
  cleanup and three concurrent surviving slots with a fourth request queued.
- All 19 host regression suites pass, including loading, contexts, residency,
  streams, cancellation, embeddings and metadata. All 89 Rust wire tests pass.
  Real Rust integration additionally verifies partial preload over direct and
  tunnel connections, reduced queue/model/text capacity, routed inference,
  router restart and deregistration.
- All 24 packaged CUDA checks pass. A hardware run preloads Qwen3.6-35B-A3B
  with deliberately missing sibling models. Startup continues with one real
  engine using 16,066 MiB; the failed automatic replica is rejected with HTTP
  500, the configured original retains its retry slot, and real CUDA inference
  completes in 0.45 seconds at a 4,096-token context.

The deployed image is
`sha256:1391303baf5d9780c2297368a1490f65d7af9143b6c5b8819d2b1f1696839adf`;
its native executable SHA-256 is
`580afa1bd2f6f622033521a427108abc8039560145eadaed61a0061d581a47b3`.
The original environment file, effective settings, seven mounts, ports and GPU
requests are unchanged. The container has no Python interpreter. Live routed
chat, SSE and vision pass against the original Qwen3.8 configuration; exactly
one matching worker is registered and its metadata matches the Python
calculation. Rollback: `ezlocalai:cuda-before-preload`.

Evidence: `tests/fixtures/llm_preload_cases.json`,
`tests/fixtures/llm_preload_validation.json`, `tests/test_llm_preload.cpp`,
`tests/test_worker_router.cpp` and local logs in `build/validation/preload/`.
Full pipeline initialization/preload ordering, runtime-resolved parallel
capacity and queue refresh, broader GPU behavior, persistent media engines and
remaining compressed GPU/video parity remain open.

## LibTorch allocator diagnostics deployed (2026-10-19)

`/v1/resources` now reads actual allocated, reserved and peak CUDA memory from
the current process's LibTorch allocator when that optional runtime is present.
The previous unconditional zeros missed in-process allocations such as native
wake-word training. Builds without LibTorch retain zero allocator counters;
subprocess engine allocations continue to appear in driver free-memory readings
and the shared model registry. LibTorch builds return an empty allocator list
when CUDA is unavailable, including when device visibility hides the GPU.

Collection follows the Python availability/count/query order, isolates failures
per device and preserves its three-decimal rounding. Cache cleanup shares the
same runtime adapter and retains peak allocation history.

Validation:

- 330 cases execute unchanged `ResourceManager.get_torch_cuda_memory_stats`
  source, covering unavailable CUDA, multiple devices, rounding and failure at
  each query. The source hash is verified when the checkout is available.
- Resource, HTTP and cold-load reclamation regressions pass. All 89 Rust wire
  tests and all 23 packaged CUDA-image checks pass.
- A real 64 MiB CUDA allocation matches the original Python container's
  LibTorch 2.9.1 readings exactly: 0.062 GiB allocated/reserved, zero allocated
  after tensor release, zero reserved after cache cleanup, and a retained
  0.062 GiB peak. The native test also checks every visible device and passes
  with CUDA hidden. Hardware tests run without stopping the existing worker.

The deployed image is
`sha256:47b158de1c159216fa689d62fb57da5c2990d5d921595391ea9e96a785e241aa`;
its native executable SHA-256 is
`0705ee3b736241fcc42e6f821282729edb1007cefbd3c2d0ca6576aa020ce4d8`.
The original `.env`, effective environment, seven mounts, ports and GPU requests
are unchanged. No Python interpreter is installed. Live routed chat, SSE and
vision return HTTP 200; exactly one matching worker is registered and its
residency/dependency metadata matches Python. The live resource endpoint returns
valid allocator and driver counters. Rollback:
`ezlocalai:cuda-before-allocator`.

Evidence: `tests/fixtures/cuda_allocator_cases.json`,
`tests/fixtures/cuda_allocator_validation.json`, `tests/test_cuda_allocator.cpp`,
`tests/test_torch_resources.cpp` and local logs in `build/validation/allocator/`.
Broader GPU allocation/selection and cleanup equivalence remain open.

This audit identified that native warmup stopped at the first failed preload,
unlike Python's per-model failure handling. The later failed-replica milestone
above resolves that startup and capacity gap. Complete preload ordering,
runtime capacity refresh and remaining media-engine parity still prevent full
replacement acceptance.

## Cold LLM loads reclaim auxiliary memory (2026-10-19)

Cold text/vision loads now follow the Python target-specific headroom policy:
estimate the requested model/context, read free memory on its configured GPU,
apply the residency margin, and reclaim idle auxiliary models when needed.
Estimate failures retain the source 8 GiB fallback. Configured-context estimates
are cached and registration retains the larger estimated or observed footprint.
Resident reuse skips reclamation. Video restoration's final retry also uses the
source 16 GiB reclamation request.

Reclamation follows the existing stable LRU selection, including its primary-LLM
exclusion and treatment of selected but unsupported model types. TTS, STT and
embedding pools release idle instances synchronously; image/video cleanup
requires the corresponding generation lock. Active leases are rechecked before
destruction. LibTorch builds release unused allocator memory and the final live
free-memory probe preserves the source exception boundary. Cold loads serialize
while ordinary inference completion and embedding cleanup remain able to run;
the backend mutex is released around auxiliary destruction.

Validation:

- 512 headroom and 196 reclamation cases execute unchanged Python source and
  match native estimates, margins, GPU selection, destruction order and probes.
- Managed subprocess tests cover idle and active speech pools, embedding process
  cleanup, resident reuse, repeated cold loads and simultaneous cold loads.
  A concurrency regression holds cleanup blocked and proves an existing
  inference can finish and release its lease independently.
- All 15 broader host suites pass. Five focused suites pass after the final
  concurrency correction; all 89 original Rust wire tests and all 21 packaged
  CUDA-image checks pass.
- Two real Qwen TTS CUDA instances each occupy 3,516 MiB before a cold
  Qwen3.8-27B request. With 14.71 GiB free, both are reclaimed and their processes
  reaped; the 220,000-token-context LLM uses 18,746 MiB and completes the request
  in 6.80 seconds. The test restores the previous service before deployment.

The deployed image is
`sha256:2aa79fd5a4860f14d7aa24372520ca535fdfebe06329c89b64ddee9297f15acc`;
its native executable SHA-256 is
`918c7882362e3a515a777ec3f43d42ad1cf3594934b9cf2e2e541256e80cd2a4`.
The original environment file, effective settings, seven mounts, ports and GPU
requests are unchanged. No Python interpreter is installed. Live routed chat,
SSE and vision return HTTP 200; exactly one matching worker is registered and
its residency/dependency metadata matches the actual Python calculation.
Rollback image: `ezlocalai:cuda-before-headroom`.

Evidence: `tests/fixtures/llm_headroom_cases.json`,
`tests/fixtures/llm_headroom_validation.json`, `tests/test_llm_headroom.cpp` and
local logs in `build/validation/headroom/`. Complete startup/preload ordering,
failed automatic replicas, media registration/resident engines, allocator
statistics and broader GPU/media equivalence remain open. This milestone does
not establish full replacement parity.

## LLM residency and model swapping deployed (2026-10-19)

Native workers now capture `LLM_MODEL_RESIDENCY=auto|resident|swap` and
`LLM_MODEL_RESIDENCY_MARGIN_GB` using the Python decision policy. A single
configured instance stays resident. Forced modes skip estimation; automatic
mode sums configured model estimates by GPU and selects swapping when the
combined estimate exceeds usable capacity. Failed estimates preserve resident
mode. Capacity comes from the cached per-device totals used by the resource
manager, while free-memory reporting remains live.

The installed Python xllamacpp estimator returns `vram_size` and `total_size`,
while `Pipes.estimate_model_vram_requirement` reads `memory`. Successful
estimation therefore selects the empirical fallback. Native reproduces that
behavior and the separate file-size/projector estimate used after an estimator
exception. Hybrid Qwen3.5/3.6/3.8 KV sizing reads GGUF metadata, attention tensor
counts, cache precision and the source's metadata-free defaults and cache.
The reader checks descriptors and bounds without mapping the model weights.

Swap mode preloads the first instance, serializes HTTP text admission globally,
and unloads other idle text/vision processes before loading the selected model.
Model slots advertise capacity one and become occupied together during text
inference. Lifecycle state and worker registration/heartbeats report the actual
mode, loaded public identities and active model. Media handoff restores only
the previously resident engine. Failed target loads leave empty residency and
can be retried. Direct backend calls cannot evict another engine with active
inference. A cancelled running stream retains the exclusive lease beyond the
source's five-second close timeout until its producer finishes, including when
configured `N_PARALLEL` is greater than one.

Validation:

- 870 cases execute the actual Python residency method; 992 memory/KV cases
  execute its helpers with xllamacpp and GGUFReader copied from the original
  container. Native matches the captured results, including malformed GGUFs,
  metadata fallback, per-card allocation, failed estimates and forced modes.
- Actual Qwen3.8-27B, Qwen3.6-35B-A3B and Qwen embedding files match Python's
  estimates for FP16, Q4 and Q8 cache settings. The Qwen3.8 comparison uses the
  production 220,000-token context.
- Managed subprocess checks cover lazy startup, replica selection, active and
  cancelled streams, global queueing, truthful slot/lifecycle state, failed
  loads, retry, media restoration and resident-mode controls. HTTP, stream,
  loading, embedding, lifecycle, resource, service-client and metadata
  regressions pass.
- The actual Rust WorkerRouter integration now covers resident and swap modes
  over direct and tunnel connections, JSON/SSE, router restart, changed model
  identities in heartbeats and shutdown. All 89 original wire tests pass.
- The CUDA image passes 20 packaged checks. Real CUDA inference completes
  Qwen3.8 → Qwen3.6 → Qwen3.8 at a 4,096-token test context with automatic swap
  selection. Each switch leaves exactly one language engine and reaps the old
  process. Engine GPU allocations are 12,782 MiB and 16,066 MiB respectively;
  the two switches take approximately 14.1 and 14.5 seconds.

The running image is
`sha256:afa4ddcdbf17776dad391134844db5452dd456ba3ef5a5fdbb739f0193870feb`;
its native executable SHA-256 is
`20fd7e14a69270ab4e2e4292afaf8dad154d9dfa9fb63b5a8735228f62cc9b94`.
The original environment file, effective environment, seven bind mounts, ports
and GPU requests are unchanged. The container has no Python interpreter.
Rollback image: `ezlocalai:cuda-before-residency`.
Live routed chat, SSE and vision return HTTP 200. Exactly one matching worker
is registered, and its residency/dependency metadata matches the actual Python
worker-extra calculation against the observed live state. The production
single-model configuration correctly remains in resident mode.

Evidence: `tests/fixtures/llm_residency_cases.json`,
`tests/fixtures/llm_residency_gguf/`,
`tests/fixtures/llm_residency_validation.json`, and local artifacts under
`build/validation/residency/`.

The goal remains incomplete. Target-specific cold-load headroom and idle
auxiliary-model reclamation still need integration, as do complete startup /
preload ordering and failed automatic replica handling. Broader configuration,
GPU/allocator behavior, compressed GPU execution and remaining media-engine
parity retain their open status. This milestone supersedes older statements
that production residency is always resident; it does not close those broader
requirements. Qwen remains the only required TTS engine.

## Service-client configuration and forwarding deployed (2026-10-19)

Image, text and embedding clients now capture their raw service value, URL,
authentication and dedicated mode on first access. They preserve whitespace and
sentinel-looking values, remove trailing slashes, and recognize only the exact
case-insensitive word `true` as dedicated mode. Settings persist after environment
edits; a fresh application captures the changed values. Embedding/image/video
resource slots and image/video handoff policy use the captured client state.
Media policy evaluates the client predicate only when its decision requires it.

An important correction to the previous audit: `Router.detect_local_capabilities`
already matches native capability advertising. It intentionally uses an
HTTP(S)-only heuristic, while the Python service clients accept any nonempty
non-dedicated value. The source distinction is preserved. The actual defects
were in client parsing, lifetime, forwarding, and local image-load suppression.
An unreachable configured image client now leaves local image weights unloaded,
matching `Pipes._get_img`, including non-URL values such as `none`.

Service requests now reuse the existing native aiohttp-compatible transport and
JSON decoder. This includes URL normalization, redirects, compressed bodies,
source deadlines (5-second availability; 300-second text; 120-second embedding;
660-second image; 1800-second video), MIME/charset checks and response serialization.
Null, invalid/non-JSON and failed responses fall through to existing local/router/
fallback behavior. Other falsey JSON values are returned. Non-finite values and
invalid output Unicode produce the source serialization failure. Text-client
responses follow its JSON-only method rather than relaying arbitrary SSE bytes.

Validation:

- 252 cases execute the actual Python client constructors, configured predicates,
  headers and capability detector. Native settings/capability results match.
- 114 cases execute all six source forwarding methods (chat, completion,
  embedding, image generation, image editing and video) against a loopback HTTP
  server, then serialize results with installed Starlette. Native requests match
  response/fallback outcomes for scalar/empty values, JSON null, invalid JSON,
  MIME variants, Latin-1, gzip, HTTP errors, SSE and serialization failures.
  Actual native wire checks cover uppercase HTTP schemes and omitted `none`
  authentication. Reference versions are aiohttp 3.12.15 and Starlette 0.50.0.
- Persistent-client socket checks verify captured URL/key, cached availability,
  fresh-application changes, retained local clients, slot delegation, deferred
  policy predicates and suppressed local image loading. The new suite passes
  in 6.99 seconds. Seven broader native HTTP/router/media suites pass in 69.18
  seconds. All 89 Rust wire tests pass (40.796 seconds).
- All 19 packaged CUDA-image checks pass (18.79 seconds). The deployed worker
  registers once; routed chat, SSE and vision return HTTP 200, and live worker
  residency/dependency metadata still matches executed Python policies. The
  image has no Python interpreter.

Production is healthy on image
`sha256:34dbd4cd0489b08a0227c2dab574c223cac8f546725ebe586f1e6df0f573f86f`,
executable SHA-256
`09c108eed8256aa087949cd8d920946130cec0868f7093161381e47f2fca114e`.
Rollback is `ezlocalai:cuda-before-service-clients` (the prior `a2c3146e...`
image). The original env file, effective environment, seven mounts, ports and GPU
requests are unchanged. This supersedes prior deployment identities below.

Evidence: `tests/fixtures/service_client_cases.json`,
`tests/fixtures/service_response_cases.json`,
`tests/fixtures/service_clients_validation.json`, and local capture/build/live logs
under `build/validation/service-clients/`.

Full parity remains open. Lifetime checks establish behavior from first client
access; complete Python pipeline startup/preload ordering and all auxiliary/voice
policies that read service modes need further audit. Empty model defaults,
complete model residency/registry lifecycle, swap-mode execution, compressed GPU
execution and remaining transport/cancellation cases are also unresolved. These
client checks are not a new real-model media or TTS quality run.

## Image/video slot reservations deployed (2026-10-19)

Image slots now use the resource manager's activity count, including the source's
two active marks during image generation. Image and video slots reserve capacity
when their source-derived handoff policy applies and LLM inference is active.
Video uses an observable reservation tied to its serialization lock, including
GPU policy probing, waiting for the handoff lock, inference and restoration.
Both ordinary video and music-video scene sessions retain this reservation.
Observers do not block on the video mutex.

During the reservation, video and music-video report one occupied slot. Other
capabilities and model slots report zero capacity when the source requires an
exclusive worker, preserving existing activity and queue counts. Concurrent
media profiles retain other capacities unless explicit video handoff forces
exclusivity. Totals retain the source's exclusions for duplicate vision and
composite music-video slots. Reservation cleanup covers success and exceptions.

Validation:

- All 155 full Python slot snapshots match native resource responses, covering
  image residency and activity, LLM activity, free-memory thresholds, explicit
  handoff modes, concurrent-media settings and image forwarding values.
- Twenty transition snapshots were captured; 16 distinct reachable snapshots
  are exercised by six concurrent video requests. Checks cover busy LLMs,
  unloading, an active native fixture subprocess, restoration, and released
  reservations after success, generation failure and restoration failure.
  These checks use synthetic Wan weights/process output and real FFmpeg; they
  are not a new real-model CUDA video quality result.
- Eight focused native suites pass (84.83 seconds), including HTTP, tunnel,
  actual Rust direct/tunnel integration, metadata, image/video policy and speech
  slots. All 89 WorkerRouter wire tests pass (36.157 seconds).
- All 18 packaged CUDA-image checks pass (15.93 seconds). The first image build
  caught a fixture cache-isolation error: inherited `HF_HUB_CACHE=/app/models`
  hid temporary synthetic assets. The test now overrides that path, and passes
  both an explicit inherited-path host recheck and the image build.
- The deployed worker registers once; routed chat, SSE and vision return HTTP
  200. Its live residency/dependency metadata still matches executed Python
  policies. The image contains no Python interpreter.

Production is healthy on image
`sha256:a2c3146e55e70e3c8ed48f3674da98a6969a954986c4141c96fd05ce4c5d901a`,
executable SHA-256
`0fea096a9dadca0065e80651e45c2ff2e20d7c67dfbc488d5ad63e76a19ec652`.
Rollback is `ezlocalai:cuda-before-media-slots` (the prior `87effdc3...`
image). The original env file, all effective environment values, seven mounts,
ports and GPU requests are unchanged. This supersedes older deployment IDs
and the image/video exclusive-worker slot gap below.

Evidence: `tests/fixtures/media_slot_cases.json`,
`tests/fixtures/media_slots_validation.json`, and local capture/build/live logs
under `build/validation/media-slots/`.

Full parity remains open. A follow-up source audit confirms that native
capability detection already matches Python's HTTP(S)-only heuristic. Python's
client predicate differs from its capability detector; that source distinction
must be preserved, including for `IMAGE_SERVER=none`. Client configuration and
initialization, empty model defaults, complete video residency and
registry lifecycle, swap-mode execution, compressed GPU execution and remaining
worker transport cases need further work. Qwen remains the sole required TTS
engine; historical fallback engines are outside replacement acceptance.

## Router residency and media dependency metadata deployed (2026-10-19)

Worker registration and heartbeat now carry all four source `extra` fields:
`llm_model_residency`, `loaded_llms`, `active_llm` and
`llm_unload_dependent_capabilities`. Public model identities come from the
existing backend lifecycle snapshot, preserving duplicate replicas and loaded
order. Image/video dependencies now use the same source-derived policies as
local generation; `music_video` is appended when both video and music depend
on LLM unloading. Existing music and speech policy remains included in source
order. Image/video policy is evaluated even when those capabilities are not
advertised, as in the Python worker client.

Media policy helpers now support deferred GPU/memory probes, preserving source
short-circuit behavior. A failed probe during `extra` collection leaves that
object empty and the remaining worker state is still sent. This matches the
worker client's exception boundary; it does not claim recovery from every
possible earlier resource/heartbeat failure.

Production is healthy on image
`sha256:87effdc32efb58287b1014c167ebe3765da1ed8b91c1d22cd17fbf51b0a30949`,
executable SHA-256
`a21101516aa8151c5716075d3db0abaa3ee6ad2b9996f1b8b1079b34c45240e7`.
Rollback is `ezlocalai:cuda-before-worker-metadata`, the previous
`bfdc44bc...` image. This supersedes the deployment identities and pending
`extra` field/dependency gaps below. The original env file and all effective
environment values, seven mounts, ports and GPU requests are unchanged.

Validation:

- 73 cases execute the actual Python worker dependency/extra block, public
  name resolver and media/voice policies against controlled state. Native
  registration and heartbeat match both values and list order. Cases cover
  retained image models, residency, free-memory thresholds, explicit modes,
  forwarding, duplicate replicas and failed/skipped memory probes. A live
  worker instance also sends updated detached/restored model state without a
  configuration change or restart.
- Native HTTP, worker tunnel, real Rust direct/tunnel integration, image/video
  policy and managed LLM lifecycle/load-order checks pass. The Rust integration
  accepts the residency fields before and after router restart. All 89 Rust
  wire tests pass (35.060 seconds).
- The CUDA image passes 17 packaged checks (14.23 seconds), including the new
  metadata wire test, and contains no Python interpreter.
- Production registers once. Routed chat, SSE and vision return HTTP 200.
  A separate live check executes the same Python policies against production
  resource/configuration observations and exactly matches the router's `extra`:
  residency `resident`, loaded/active `unsloth/Qwen3.8-27B-GGUF`, dependencies
  `image`, `video`, `music`, `music_video`. Observed free VRAM was 1.536621 GiB.
  This checks the worker-to-router projection against live state; the Python
  server was not started for a second model load.

Evidence: `tests/fixtures/worker_extra_cases.json`,
`tests/fixtures/worker_metadata_validation.json` and
`build/validation/worker-metadata/`. The capture deliberately has no
secondary GPU; existing media policy tests cover secondary-device decisions.
Native residency execution is still `resident`; publishing the snapshot does
not implement swap mode. Image/video slot and exclusive-worker accounting,
complete allocator/lifecycle behavior, broader version/metadata and heartbeat
failure auditing, and other tracked worker replacement requirements remain.

## Speech capacity and router dependencies deployed (2026-10-19)

Managed speech `cap_slots` now use live `TTS_N_PARALLEL`/`STT_N_PARALLEL`
settings and resource-manager activity. A waiting HTTP request does not consume
a model slot. Shared speech capacity is one; the slot stays occupied while an
LLM, the other speech modality or either voice handoff is active. Handoff state
covers waiting for the LLM, unloading, synthesis and restoration, and is cleared
on restoration failure. Pool residency/configuration remains separately
reported through `model_pools`. Speech forwarding uses the captured voice client,
including Python's handling of values such as `false`, `none` and trailing
slashes. Aggregate resource slot totals now use 64-bit arithmetic.

Worker registration and heartbeat now include TTS/STT in
`extra.llm_unload_dependent_capabilities` when the capability is advertised and
its voice handoff policy applies. The real Rust router integration verifies
that two text slots plus shared TTS/STT capacity count as two total slots,
rather than four. It checks direct and tunnel connections, metadata retention
after router restart, JSON/SSE inference and deregistration. Socket tests also
check independent TTS capacity three alongside shared STT capacity one in both
registration and heartbeat payloads.

Production is healthy on image
`sha256:bfdc44bc8b457eef7eefef5e6899e5a384f9e75d5d576670234df4b73e0e085d`,
executable SHA-256
`e2b6666d62342ea5cbe13227152cc15360b6eb723ea70eb396bddb5c042d6a06`.
Rollback is `ezlocalai:cuda-before-speech-slots`, the prior `f7f14b8a...`
image. This supersedes the deployment identities and pending speech slot gap
below. The original `.env`, every effective environment value, seven mounts,
ports and GPU requests are unchanged. Production registers once; routed chat,
SSE and vision all return HTTP 200.

Validation:

- 84 snapshots executed from the complete Python slot method, voice handoff
  policy, voice-client configuration and parallel parsers match native HTTP.
  Other modalities are disabled in this capture. Cases include active counts,
  enable/forwarding settings, invalid values, Unicode digits and two maximum
  signed-32-bit capacities whose aggregate exceeds that range.
- Eight source transition snapshots support native concurrency checks for the
  full handoff sequence, restoration failure and three requests sharing two
  permits. The existing native HTTP and speech-resource regressions pass.
- Worker tunnel and real Rust integration checks pass in 2.82/5.49 seconds.
  Rust's `captured_dependency_slots` test passes 384 fixture cases; all 89
  WorkerRouter wire tests pass in 30.140 seconds.
- The CUDA image passes all 16 packaged checks (2.81 seconds) and has no Python
  interpreter. With CUDA hidden, its actual Qwen CPU pool preloads two models.
  Three HTTP requests return WAVs of 134,444/403,244/664,364 bytes. While two
  models synthesize and the third request waits, the API reports capacity two,
  active two, available zero; it returns to active zero/available two afterward.
  This test uses the identical packaged/deployed binary and the original env
  file with explicit isolated speech overrides. It does not claim a new GPU
  inference run; the previous milestone records real CUDA synthesis.

Evidence: `tests/fixtures/speech_slot_cases.json`,
`tests/fixtures/speech_slots_validation.json` and
`build/validation/speech-slots/`. The initial router metadata assertions
needed music handoff explicitly disabled in their fixture: the existing Python
policy can advertise that dependency even with music inference disabled.

Full parity remains open. Router slot fields retain their signed 32-bit bound,
so larger positive Python integers are clamped. Image/video/composite dependency
metadata, complete router residency fields, other modality slot policy and
allocator/lifecycle parity still need work.

## Speech resource reporting deployed (2026-10-19)

The resource API now exposes preloaded and active Qwen/STT models from the
shared registry, including idle residency, resource-counted activity, device
and estimated VRAM. It previously discarded these records and relied on HTTP
request counters. Registry entries now replace duplicate backend/approximate
model keys. `model_pools` reports the configured capacity, resident count and
resident devices of TTS, STT and embedding pools, matching Python's shape.
Transient models remain separate from resident counts.

Production is healthy on image
`sha256:f7f14b8a877d41f2eb878139df1dc0293ef9c9b708ae45ae1ae38a4b13a801c6`,
executable SHA-256
`3c43e04d579f3085b1dce2c5058c8cdffbb5235917ea836da086fdc970677c93`.
Rollback is `ezlocalai:cuda-before-speech-resources`, the previous
`2d244627...` image. This supersedes the deployment identity below; wake-word
jobs and persistent mounts remain included. The original `.env`, all effective
environment values, seven mounts, ports and GPU requests are unchanged. The
original modality flags still disable speech in production; isolated tests
explicitly enable it. Production exposes the new empty speech/embedding pool
snapshots and its loaded vision LLM.

Validation:

- Five snapshots executed from Python's actual registration/status/pool methods
  match the native HTTP response across mixed CUDA/CPU preload, active synthesis,
  idle residency and successive unloads. An overlapping image record verifies
  replacement without duplicate keys. Embedding's 569-case/4,167-step runtime
  test also checks two managed resident CPU devices through HTTP.
- Host HTTP, embedding, TTS API and speech-resource checks pass. CUDA-container
  speech and embedding checks pass; the latter needed its two reference source
  files supplied explicitly after the initial missing-reference failure.
- Real Qwen CPU and CUDA synthesis both verify idle/active/idle resource counts
  and retained pool residency. Outputs were 176,684 and 222,764 WAV bytes.
  NVIDIA independently reported the CUDA Qwen process at 3,516 MiB. The API's
  4 GiB value is Python's admission estimate, not a measured allocation.
- The packaged CUDA build passes 15 checks (1.77 seconds), contains no Python
  interpreter and has the identical executable used in both real speech runs.
  All 89 Rust WorkerRouter wire tests pass (31.561 seconds). Live registration
  has one matching worker; routed chat, SSE and vision return HTTP 200.

The smoke harness corrections are recorded in
`tests/fixtures/speech_resources_validation.json`: local speech requires an
empty `VOICE_SERVER` rather than the URL-like value `false`, Qwen's base model
needs the reference WAV, and the admission-selected device string is `cuda`.
Response snapshots are recorded in that fixture and under
`build/validation/wakeword-model/speech-resources-*`.

This closes the missing TTS/STT visibility and pool snapshot gap identified by
the previous CUDA job. It does **not** complete resource parity: speech
`cap_slots` still use legacy capacity/activity reporting rather than the full
resource-counted/shared-handoff policy. Allocator equivalence, broader lifecycle
work and other tracked replacement gaps remain open.

## CUDA wake-word jobs deployed with persistent storage (2026-10-19)

Production now runs image
`sha256:2d24462772d01e7065d7ec04563b6dcdd542bfcb3ab33fc1efffe797b2121746`,
with executable SHA-256
`3a9f1b8729b84e949512634ba76e47a344d2b07069cf60854437f03bea2b610e`.
This executable is identical to the tested background-job binary. The CUDA
image build now includes the standalone Qwen sample and background-job tests;
all 14 packaged checks passed (2.53 seconds). The image still has no Python
interpreter. Rollback is retained as
`ezlocalai:cuda-before-wakeword-jobs` (the prior `9961a2bb...` image).

A full CUDA HTTP job completed in **50.09 seconds**, including Qwen synthesis,
30 positive/21 negative training samples, two training epochs, PyTorch/config/
ONNX export and prediction. Qwen's actual NVIDIA process allocation was
3,516 MiB. The saved checkpoint's storage labels are `cuda:0`; Python loaded it
on CUDA and reproduced the native prediction exactly at `0.952076256275177`.
The checkpoint has 25 state tensors and passes weights-only loading; ONNX's full
checker passes. The resident production LLM was stopped for GPU headroom during
this isolated run and restored automatically afterwards.

The first CUDA harness attempted to read a preloaded TTS entry from
`/v1/resources.loaded_models`, but the current API omits that entry. The corrected
test used actual NVIDIA process allocation and checkpoint storage/device checks.
This is an identified **remaining TTS resource-reporting gap**, not evidence that
the initial telemetry assertion passed or that full memory parity is achieved.

Compose now mounts `wakeword_models`, `wakeword_samples` and `wakeword_jobs`
alongside the existing data mounts. Runtime directories are excluded from Git
and Docker build contexts. Both original Python and prior C++ containers were
checked: neither contained these directories, so no existing artifacts needed
migration. The deployment uses host directories under
`/home/josh/repos/xtsys/ezlocalai`; test files written through production mounts
were read unchanged by the host and a fresh disposable container, then removed.

The original `.env` hash and every effective environment value are unchanged.
Existing mounts, ports and GPU requests are preserved; exactly the three new
data mounts were added. Production is healthy, registers once with WorkerRouter,
and routed chat, SSE and vision all return HTTP 200. Original modality enable
flags remain authoritative. The broader replacement goal is still incomplete;
this deployment does not claim immediate in-flight cancellation, ESPDL support,
trained-model quality, complete resource reporting or remaining modality parity.

Evidence is in `tests/fixtures/wakeword_jobs_deployment_validation.json`,
`build/validation/wakeword-model/jobs-cuda`, adjacent deployment/mount
records and `/tmp/ezlocalai-wakeword-jobs-*.log`. Earlier not-deployed labels
below describe their individual implementation milestones and are superseded
by this deployment record.

## Background wake-word jobs and real training (2026-10-19, implementation milestone)

The worker HTTP training endpoint now starts a native background pipeline instead
of writing a permanently pending record. `WakewordJobs` connects Qwen synthesis,
augmentation, noise/similar-word negatives, native training and checkpoint/ONNX
export. The native executable supplies the LibTorch trainer in supported builds.
Configured reference voices are discovered from the worker's `voices` directory;
an empty directory selects Qwen's default voice. All synthesis uses the shared
local Qwen service with its existing admission and cleanup.

Job status, stage/progress, merged epoch/final metrics, duplicate-word lookup,
listing/filtering and cancellation are connected to the HTTP handlers. Records
are cached after initialization like Python's manager, and persisted active jobs
are not automatically resumed. Updates use atomic file replacement. A single
background worker serializes jobs and carries the augmentation RNG between them.
This scheduling choice avoids simultaneous training allocations; it does not
claim identical timing to Python's asynchronous sample generation. Cancellation
is checked at sample and epoch boundaries and around export, and cannot be
overwritten by a later completion/failure update. Buffered synthesis still needs
to return before its cancelled job releases the worker.

`wakeword_jobs` covers the complete pipeline with controlled synthesis/training,
twelve simultaneous case-insensitive duplicate requests, queued/active
cancellation, failure paths, persisted/cached records, restart without implicit
resume, negative list limits, HTTP model download and path containment. The
general HTTP fixture now restarts before expecting externally written job files
to be loaded and expects an explicit failure when its echo server has no trainer.
Those older pending-only/file-rescan assumptions were exposed by the first
broader regression run, rather than retained in production behavior.

A real HTTP run in the CUDA image reused the original `.env` with explicit
isolated test overrides. CPU Qwen synthesis generated five base samples; the
pipeline produced **30 positive and 21 negative samples**, trained two epochs,
and exported PyTorch/config/ONNX in **207.33 seconds**. The checkpoint is
1,708,625 bytes. Python loaded all 25 state tensors with `weights_only=True`,
ONNX's full checker passed, and Python/native predictions agreed exactly at
`0.8089866042137146` on the retained audio. This is functional integration
evidence, not wake-word accuracy or generalization validation.

After restart with the final executable, the completed job and downloaded
checkpoint were unchanged. Eight concurrent HTTP submissions shared one job;
cancellation remained terminal, produced no checkpoint, and released the worker
for a subsequent job after about five seconds. That subsequent zero-sample job
reported the expected failure. The final executable SHA-256 is
`3a9f1b8729b84e949512634ba76e47a344d2b07069cf60854437f03bea2b610e`.
The initial successful training run used
`ca36828b0457c8348b4727d0442266bb15c074507379e1fd46ea18db7d516d83`;
the subsequent production change adds the source HTTP 409 response for an
escaping sample-directory path. Final containment, restart and cancellation
checks use the rebuilt executable.

All **89 WorkerRouter wire tests pass** against the updated host executable
(29.52 seconds). The corrected full native HTTP suite passes (29.75 seconds),
as do the worker tunnel and Rust-router integration tests (2.63 and 5.42 seconds).
Wake-word routes themselves are worker API routes: neither the
current Rust router capability map nor the Python router exposes them. The
CUDA-container job/training/export regressions pass together (3.26 seconds).
Large evidence is under `build/validation/wakeword-model/jobs-live` and
the adjacent job validation records; the compact record is
`tests/fixtures/wakeword_jobs_validation.json`.

Production remains on the verified prediction image. A full CUDA inference/
training job, deployment packaging/persistence, immediate in-flight cancellation,
working ESPDL export and trained-model quality validation remain open. Earlier
sections describing pending-only HTTP jobs are superseded by this section.

## Qwen wake-word sample adapter (2026-10-19, not deployed)

`wakeword_qwen_tts_operations` now connects the sample generator to the existing
`TtsService`. Its local-only request path uses the shared pool, voice admission,
GPU handoff and cleanup without consulting a fallback provider. Training
requests explicitly disable the audio cache and request WAV bytes. The generator
normalizes those bytes to 16-kHz mono PCM, cycles the supplied Qwen reference
voices, and makes one synthesis attempt per requested sample. It no longer uses
the historical provider-list cap when Qwen is selected. Individual failures are
skipped and attempt progress is reported after each call. Qwen takes precedence
even if legacy callbacks are also present.

Per-job cancellation is linked to service shutdown during slot admission.
Cancellation after model loading prevents synthesis; cancellation during buffered
synthesis discards the result after the worker returns. Positive and similar-word
generation propagate cancellation. Dedicated voice-server mode also returns the
training lease to the pool. Interrupting an already-running buffered worker
mid-inference remains unfinished; this change does not claim immediate abort.

The standalone `wakeword_qwen` test covers fresh requests, conversion/metadata,
voice cycling, fallback isolation, partial failures, handoff/dedicated cleanup,
cancelled admission and cancellation during loading/synthesis. Host service,
pipeline and synchronous/asynchronous voice-slot regressions pass (five tests
across two runs). In the CUDA image, the Qwen adapter and existing sample/TTS
reference tests pass together (three tests, 0.78 seconds).

A real CPU run of the packaged Qwen model, with networking disabled, produced
two distinct samples through the adapter: 107,564 and 69,164 bytes after
normalization. Both complete normalized WAVs match installed Python pydub's
conversion byte for byte. The test checks lease release and resource removal
after explicit unload; no fallback was invoked. This proves the adapter's real
CPU path in the CUDA image, not new GPU inference or trained-model accuracy.
The CUDA executable needs the NVIDIA driver libraries even for CPU inference;
the successful run exposed those libraries while hiding CUDA devices.

Evidence is under `build/validation/wakeword-model/qwen`, with a compact
record in `tests/fixtures/wakeword_qwen_validation.json`. The source/container
file hashes were checked after rebuilding. Production remains on the verified
prediction image. HTTP background jobs, voice discovery for those jobs, complete
training integration and end-to-end cancellation/export remain unfinished.

## Native Edge TTS protocol and transport (2026-10-19, not deployed)

The wake-word native adapter factory now supplies both gTTS and Edge transports.
Edge's text cleanup/chunking, voice expansion/SSML, time-derived request token,
headers, speech configuration, audio frames and one-time HTTP-403 clock-skew
retry follow the installed `edge-tts` 7.2.8 package. The implementation preserves
the source's unusual full-length fallback for unbroken text above 4,096 bytes.
The native WebSocket client verifies TLS peers/hostnames and the upgrade
challenge, masks client frames, handles fragmentation and ping/pong, and supports
per-message deflate in both directions, including context-takeover settings and
negotiated client window sizes. Audio conversion remains serialized separately.

Direct Python captures validate ten text cases, seven SSML/voice cases, eight
request tokens, eleven stream success/error cases and an exact full request.
Native retry tests cover corrected tokens and missing server dates. Eleven
loopback WebSocket cases cover plain/compressed/fragmented messages, takeover
disabled, smaller client compression windows, large payloads, bad challenges,
empty/truncated streams, invalid compression, masked server frames, oversized
frames and invalid UTF-8. Seven combined Edge and wake-word CTest regressions
pass in the CUDA container (7.43 seconds).
The eleven transport cases also pass AddressSanitizer and
UndefinedBehaviorSanitizer with leak detection enabled.

**Live synthesis success is not verified.** The installed Python client and the
native client both received HTTP 403 twice, adjusted their clock using the first
response's Date, and ended with `WSServerHandshakeError`. Attempt/status/date and
clock adjustment are recorded for each. This is a matching live failure, not
evidence of synthesized Edge audio. Successful audio delivery is currently
covered by source parser captures and local network fixtures.

New files are `edge_tts.cpp`, `edge_tts_transport.cpp`, the two native Edge tests,
and the retained `edge-tts` license. Fixture metadata is in `edge_tts_cases.json`
and `edge_tts_validation.json`, with larger evidence under
`build/validation/wakeword-model/edge`. `edge_tts_network` is standalone;
`edge_tts_reference` uses `EZLOCALAI_WAKEWORD_REFERENCE_DIR`. An explicit
`test_edge_tts <root> live` performs a live attempt and records its result.
External capture scripts are `/tmp/ezlocalai-capture-edge-protocol.py` and
`/tmp/ezlocalai-edge-live.py`.

Background-job integration/cancellation and working ESPDL export remain
unfinished. Edge proxy/redirect/cookie variations, exact HTTP/TLS error details,
and successful live provider output need broader verification; native TLS uses
the system trust store whereas the source explicitly selects certifi. No new
training pipeline has been deployed, and production stays on the verified
prediction image.

## Wake-word TTS scheduling, gTTS and MP3 conversion (2026-10-19, not deployed)

The native sample generator now schedules the source's six gTTS languages and
24 Edge voices, preserves result order while skipping failed samples, and
reports progress only after all scheduled requests finish. Source floor division
and Python slicing are preserved for small and negative target counts; the
default target of 500 schedules at most 30 base samples. gTTS calls complete
sequentially before concurrent Edge adapter calls. Audio decoding is serialized
like Python's blocking pydub calls, preventing a burst of 24 FFmpeg processes.
Similar-word generation
preserves the source's 29 base words, Unicode code-point prefix/suffix variants,
selection order, two-sample requests and metadata.

The gTTS adapter reuses the native HTTP implementation with explicit English
language selection. All five configured regional codes fall back to English in
installed gTTS 2.5.4, including when the text contains Cyrillic. The ordinary
speech endpoint's language inference and 24-kHz conversion are not used.
The MP3 decoder follows pydub's seekable FFmpeg input and PCM16 conversion,
then uses the verified audioop-compatible 16-kHz/mono preparation. The source
and native containers both provide FFmpeg 6.1.1-3ubuntu5.

Source comparisons pass **32 generation cases, seven similar-word cases, 24
exact gTTS request comparisons and ten byte-exact MP3-to-WAV conversions**.
Scheduling tests inject controlled engine successes/failures and compare request
sets, returned sample order/metadata and callback cadence; they do not establish
Edge network behavior. Eight decode cases use generated mono/stereo MP3s at
16/24/44.1/48 kHz, including concatenated files. Two use actual live responses.

Live gTTS succeeds from both containers for `hello ezlocalai`, each producing
14,592 MP3 bytes and a 58,412-byte, 16-kHz mono WAV. Separate live requests do
not yield identical samples: repeated Python requests also differ from the
initial Python capture (maximum PCM difference 393; native-vs-initial difference
330). This is recorded as upstream variability, not exact synthesis equality.
Both captured responses decode byte-exactly across the two implementations.
All five sample/TTS/prediction/training/export CTest regressions pass together
in the native container (5.15 seconds).

Fixtures are `wakeword_tts_cases.json` and `wakeword_tts_validation.json`; large
audio evidence is under `build/validation/wakeword-model/tts`. The opt-in
`wakeword_tts_reference` test uses `EZLOCALAI_WAKEWORD_REFERENCE_DIR` and the
native executable accepts `<root> live-gtts` for an explicit network smoke test.
External captures are `/tmp/ezlocalai-capture-wakeword-tts.py` and
`/tmp/ezlocalai-wakeword-gtts-{live,repeat}.py`.

Edge's native network adapter is now implemented and covered above. The generator
is not connected to HTTP training jobs yet. This scheduling section alone does
not establish Edge transport parity.
Cancellation, live job integration, working ESPDL export and remaining decoder
error/format edges are also unfinished. Production remains on the prior verified
prediction image.

## Wake-word PCM sample preparation (2026-10-19, not deployed)

`wakeword_samples.cpp` now implements all sixteen source augmentation variants,
dataset selection/ordering/metadata/progress, and white/pink/brown/silence
negative samples. Its scalar-seeded MT19937, double draws and cached normal
generator reproduce NumPy RandomState. The same RNG instance carries through
augmentation and negative generation: even discarded augmentation variants
consume their normal draws. This state must remain shared when the background
pipeline is connected.

Integer WAV handling reproduces pydub's unsigned 8-bit conversion, 24-to-32-bit
promotion, audioop linear resampling, volume rounding/clipping, float32 reverb
and the source's int16 noise/reverb byte replacement while retaining original
sample-width metadata. A separate helper normalizes already-decoded TTS PCM by
resampling to 16 kHz before mono conversion, including pydub's different stereo
and multi-channel rounding rules. Pink noise follows SciPy's direct-form-II
transposed filter, and brown noise preserves cumulative-sum/normalization order.

Actual Python captures cover nine audio layouts (8/16/24/32-bit PCM, mono/stereo/
three channels, 8–48 kHz and empty/one/short clips), five noise configurations,
six dataset selections and four augmentation-to-negative sequences. **All 421
WAV artifacts match byte for byte**, including their headers. Metadata, order,
callback cadence and subsequent random draws also match. The host comparison
passes under AddressSanitizer and UndefinedBehaviorSanitizer with leak checking.
The CUDA-container build also passes all 421 comparisons and all four sample,
prediction, training and export regressions together (6.28 seconds). A fresh
CUDA export run still matches the eight source success/failure cases. Copied
source timestamps initially left an older sample-test object in the container;
the final evidence follows a forced rebuild and confirms the current case count.
The source environment is Python 3.12.14, NumPy 2.4.6, SciPy 1.18.1.

The native test is `test_wakeword_samples <wakeword-reference-root>`; configuring
`EZLOCALAI_WAKEWORD_REFERENCE_DIR` with `samples/reference.json` registers
`wakeword_samples_reference` independently of LibTorch. Fixture metadata lives in
`wakeword_sample_cases.json` and `wakeword_sample_validation.json`; large WAV
evidence remains under `build/validation/wakeword-model/samples`. The
external capture script is `/tmp/ezlocalai-capture-wakeword-samples.py`.

The algorithm references are [CPython audioop](https://github.com/python/cpython/blob/v3.12.12/Modules/audioop.c),
[NumPy legacy distributions](https://github.com/numpy/numpy/blob/v2.3.4/numpy/random/src/legacy/legacy-distributions.c)
and [SciPy lfilter](https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_lfilter.cc),
plus pydub source inspected in the installed Python image. These older reference
implementations were checked against the actual installed versions above.
Their licenses are retained under `licenses`.

The gTTS transport, MP3 decoding and similar-word scheduling are covered above;
Edge transport has separate coverage above; background-job integration remains unfinished.
The PCM helper does not claim pydub's float-WAV/compressed-input fallback or all
malformed-WAV exception behavior. Production still runs the verified prediction
image; the new preparation/training/export pipeline is not deployed.

## Native wake-word ONNX export (2026-10-19, not deployed)

The trainable LibTorch adapter now saves checkpoint/config and optional ONNX in
the source's artifact order. A native protobuf writer emits CNN and bidirectional
GRU graphs and their external `model.onnx.data` weights without Python or a
protobuf runtime. CNN batch-normalization folding, GRU gate order, final-time
selection, training-mode dropout and export RNG consumption match the installed
Torch 2.9.1 source. Mandatory checkpoint failures propagate; optional ONNX failures
retain the checkpoint and config.

Eight CPU and eight CUDA source captures cover trained/default/small CNNs,
one/two/three-layer GRUs, and untrained models with active dropout. Python's
weights-only loader verifies all sixteen native checkpoints: **344 parameter and
buffer tensors match exactly**. ONNX's full checker accepts all twelve successful
native exports. All **112 compared weight initializers** and **256 probabilities**
match exactly in ONNX Runtime's CPU provider; dropout comparisons use matching test-only seeds
and disabled graph optimization. Production exports retain unseeded dropout.
Separate C++ tests cover normal runtime optimization, format order, CPU RNG,
96 matching batch rejections and mandatory/optional write failures. Prediction,
training and export CTest regressions pass together (3 tests, 6.54 seconds).

The captures also establish limitations in the installed Python exporter that
the adapter preserves: CNN flatten specializes to batch one despite a dynamic
input annotation; output annotations remain batch one even for dynamic GRU
graphs; requested opset 11 remains opset 18 after failed conversion; CUDA GRU
export fails while tracing cuDNN and returns only checkpoint/config. CUDA CNN and
CPU CNN/GRU export succeed. These are compatibility choices against the pinned
reference, not limitations imposed by the ONNX format itself.

Fixtures are `wakeword_export_cases.json`, `wakeword_export_cuda_cases.json` and
`wakeword_export_validation.json`. Large artifacts are under
`build/validation/wakeword-model/exports*`. With LibTorch, ONNX Runtime and
`EZLOCALAI_WAKEWORD_REFERENCE_DIR` configured, `wakeword_export_reference` runs
the native test. The executable also accepts `<root> cuda exports_cuda` for
CUDA reference evidence. Source capture/checker scripts remain outside the repo
under `/tmp/ezlocalai-*wakeword-exports*.py`.

This completes the tested internal ONNX save path, not the HTTP training flow.
Live sample generation, job integration and working-installation ESPDL export
remain unfinished. The installed Python ESP-PPQ import fails on missing
`onnxsim` and skips ESPDL. Production remains on the verified prediction image;
the new training/export backend has not been deployed.

## Wake-word training backend and checkpoint writer (2026-10-19, not deployed)

The LibTorch adapter now exposes a native trainable CNN/GRU model. It reproduces
the source dataset split, DataLoader/RandomSampler seed consumption, batched MFCC
extraction, BCE, Adam updates, validation averages, relative-threshold learning
rate reduction, epoch history and progress callbacks. The source's purported
best-state copy shares storage with live parameters/buffers, so the native
implementation correctly retains final weights instead of restoring an earlier
epoch. A native ZIP/pickle writer saves `model.pt` and `config.json` that both
Python's weights-only loader and the native loader can read. ONNX export is
verified in the section above; ESPDL remains unfinished.

Five CPU and five deterministic-CUDA captures cover CNN, bidirectional GRU,
learning-rate reduction, zero epochs and the default 40-MFCC/64-hidden CNN with
batch size 32. Histories, progress and RNG state match. Python loaded all twenty
native initial/final checkpoints and verified **every parameter and buffer
exactly**, including key order, dtype, shape and config. The C++ Adam frontend
initially differed because it updates the first moment with multiply/add;
Python 2.9.1 uses `lerp`. The adapter now reproduces Python's individual CPU and
CUDA foreach operations. Prediction regression checks still pass.

Default CUDA training remains nondeterministic in the Python reference itself:
two identical-seed CNN runs differed by about 0.00044 in epoch-average training
loss. Default native CUDA runs are recorded separately without claiming exact
floating-point equality. Deterministic comparison enables matching test-only
Torch/cuDNN flags and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; production behavior is
unchanged. The native trainer does not set those flags.

New fixtures are `wakeword_training*_cases.json` and
`wakeword_training_validation.json`. Large evidence stays under
`build/validation/wakeword-model/training*`. The opt-in
`test_wakeword_training` accepts the evidence root and `cpu`,
`cuda-deterministic`, or `cuda-capture`; the last mode records nondeterministic
results and checks RNG/reload behavior, not numerical equality. External source
capture/verification scripts are `/tmp/ezlocalai-capture-wakeword-training*.py`
and `/tmp/ezlocalai-verify-wakeword-native*checkpoints.py`.

Actual Python CPU export was also tested. Both CNN and GRU produce `model.onnx` plus
`model.onnx.data`; Torch keeps opset 18 after failing the requested downgrade to
11. ESP-PPQ is installed but importing it fails because `onnxsim` is missing,
so the source catches ImportError and skips ESPDL. This does not establish ESPDL
parity on a working installation. The next work is live sample
generation and connecting the trainer to live jobs. The HTTP train
route still only creates pending jobs. Production continues running the verified
prediction image below; this internal training work has not been deployed.

## Native wake-word checkpoint prediction (2026-10-19)

The native worker now loads Python `WakeWordTrainer.save()` ZIP checkpoints and
executes both CNN and bidirectional GRU models through pinned LibTorch 2.9.1.
This includes strict parameter/shape loading, float32 soundfile decoding,
NumPy-compatible channel averaging before torchaudio resampling, waveform
normalization, HTK mel filters, decibel clipping and orthonormal MFCCs. Prediction
retains the source's fixed 150-frame crop/padding, CUDA-if-available device choice
and strict `confidence > 0.5` threshold. Both JSON/base64 and multipart routes
call the model. Cache keys use Unicode lowercase, preserve the source's distinct
space/underscore aliases, recheck model paths on each request, and invalidate
the corresponding key after successful deletion.

Standard LibTorch archive unpickling could not read Python's OrderedDict state
dictionaries. A restricted native metadata reader now handles the checkpoint's
tensor/storage records without executing Python globals. Captures cover 19
installed-Torch cases, 117 tensor layouts (including shared, strided and empty
storage), rejected globals/protocol operations and 1,445 truncated inputs.
AddressSanitizer and UndefinedBehaviorSanitizer pass, including cyclic metadata.
The reader is intentionally scoped to these model checkpoints, not arbitrary
Python serialization. Legacy non-ZIP checkpoints, uncommon tensor metadata and
old BatchNorm version migration remain unverified.

Actual Python-source captures exercise five CNN/GRU configurations and 29 audio
inputs: 8–48 kHz, mono/stereo/3/8/17/129 channels, PCM16/24 and float WAV, FLAC,
Vorbis, short/long/silent/invalid/nonfinite input. All 91 finite MFCC arrays match
exactly on this host. All 147 CPU prediction comparisons and 145 separate CUDA
oracle comparisons pass at a maximum permitted confidence difference of 2e-6;
CPU and CUDA are compared with their respective references. Another 290 real
model requests per device pass through native JSON/multipart HTTP handlers.
32 source manager/API steps verify cache lifetime, retry, aliases, decoding and
error behavior independently of neural inference. A CUDA-only test exposed an
`--as-needed` link omission, now fixed by retaining the CUDA implementation library.
The first packaged TCP run then exposed OpenBLAS symbol interposition: Whisper's
global dependency replaced Torch's MKL routines, changing MFCCs and confidence.
Whisper now resolves its SGEMM privately with local/deep symbol binding. The model
regression also invokes Whisper features before Torch to exercise coexistence.

Prediction oracle metadata is tracked in
`tests/fixtures/wakeword_prediction_cases.json` and
`tests/fixtures/wakeword_prediction_cuda_cases.json`; cache and pickle captures
are beside them. Large checkpoint/audio/feature artifacts remain in ignored
`build/validation/wakeword-model`, with hashes checked by the C++ runner.
External capture scripts are `/tmp/ezlocalai-capture-wakeword-model.py`,
`/tmp/ezlocalai-capture-wakeword-predictions.py`,
`/tmp/ezlocalai-capture-wakeword-cuda.py`,
`/tmp/ezlocalai-capture-checkpoint-pickle.py` and
`/tmp/ezlocalai-capture-wakeword-cache.py`. They execute actual source methods
using the original Python container; no Python runtime is linked or invoked by
the native worker. Configure `EZLOCALAI_WAKEWORD_REFERENCE_DIR` with the artifact
directory to register the optional `wakeword_model_reference` CTest; the model
test target also accepts an explicit `cuda` device argument.

The 118-test native suite passes in 406.70 seconds. After the final BLAS isolation
fix, seven affected host checks pass (30.73 seconds), and three packaged CPU
Whisper/model checks pass (3.07 seconds). The final image passes 12 build checks
(1.52 seconds), all CPU/CUDA model comparisons and 290 real TCP prediction
requests using automatic CUDA selection, with zero confidence difference from
the Python CUDA capture. WorkerRouter's 89 wire tests pass again against the
final native binary (27.412 seconds).

Production now runs `ezlocalai:cuda`, image
`sha256:9961a2bb539fbc5136c91cee0d1b6d7291f766f400d4193bc6909c44d55b8958`.
It is healthy, registered once with the existing Rust router, and routed chat,
terminal-complete SSE and vision pass. The original `.env`, effective environment,
mounts, ports and GPU requests are unchanged. The former image is retained as
`ezlocalai:cuda-before-wakeword-predict`. Detailed hashes, test scope and
deployment evidence are in `tests/fixtures/wakeword_prediction_validation.json`.

These seeded checkpoints establish numerical compatibility, not wake-word
recognition accuracy. Sample generation, augmentation, training,
checkpoint/ONNX/ESPDL export, live training-job lifecycle and persistence across
container recreation remain unfinished; the training endpoint still only
persists pending jobs. The original Python image has gTTS, Edge TTS, pydub,
ONNX/onnxscript and ESP-PPQ installed, so their behavior requires further audit.

## Wake-word storage, enablement and deployment (2026-10-19)

Native wake-word routes now follow current `WakeWordPaths.py`: Unicode lowercase
and ASCII-space normalization, rejection of empty/root/separator/control names,
non-strict realpath resolution and strict containment of model directories and
all four artifacts. Internal aliases retain source behavior; escaping/dangling
external aliases are rejected before reads or deletion and skipped by listing.
Config parse failures, Unicode download filenames and dependency/body-validation
ordering now match captured Python responses. Empty training names no longer
create a pending job. Enablement uses the shared cached voice-client setting and
dynamic `STT_ENABLED` without stripping whitespace, as Python does.

The native regression covers 146 source name cases, 262 HTTP exchanges through
actual Python/Uvicorn and C++ TCP listeners, and 676 enablement observations in
169 voice-client lifetime scenarios. It checks response bodies, disposition
headers, deletion effects and unchanged outside files. The first oracle used
FastAPI TestClient; a double-percent case exposed its extra URL decoding, so all
HTTP cases were recaptured over TCP before final comparison. Source/capture
hashes and framework versions are in `tests/fixtures/wakeword_paths_cases.json`.
The focused HTTP, wake-word, CTTS path and voice-lifetime checks pass. The full
116-test suite passes in 397.82 seconds. Ten packaged CUDA checks pass (1.59
seconds), and the new wake-word target also compiles and passes inside that
image using source files mounted read-only for hash verification.

Production now runs image
`sha256:4574087aa3c68e4b7cec724c9a17413542bce2c4eb82b52cf0356d3be5e708ff`
as `ezlocalai:cuda`. It is healthy and registered with the existing Rust
router; routed chat, terminal-complete SSE and vision pass. Live invalid-name
requests return the source's 400 response. Original `.env`, effective environment,
mounts, ports and GPU requests are unchanged. The prior NF4 image is retained as
`ezlocalai:cuda-before-wakeword-paths`. Binary/source hashes and validation
scope are in `tests/fixtures/wakeword_paths_validation.json`.

Prediction and background training remain incomplete. Real Python CNN and GRU
checkpoints and inference outputs have been captured under ignored
`build/validation/wakeword-model`. A direct LibTorch C++ reader rejects
their `OrderedDict` state dictionaries, establishing the need for a restricted
checkpoint reader compatible with the existing files before native prediction
can be connected. The probe uses no Python runtime in the C++ container and does
not convert or modify the captured checkpoint files.

## Automatic NF4 loading and CUDA deployment (2026-10-19)

The worker now selects native NF4 conversion when the selected CUDA device has
at least 12 GiB free. It selects BF16 on compute capability 8 or newer and FP16
on older GPUs, and queries the actual device's SM/thread geometry for nested
statistics. Geometry failures retain unquantized weights. A conversion failure
restarts a fresh engine child without NF4; Quanto fallback is still absent.

The engine converts only Gemma linear projections from raw LTX Diffusers
components. Embeddings/norms are preserved. Eligible tensors receive private
buffers even when source precision matches; file-reader clones retain the
quantizer settings. Thirty-two actual loader cases plus two rejected-dtype
cases cover BF16/FP16, float32 sources, mmap, lazy loading and unchanged source
bytes. The BF16 and FP16 quantizer suites each match 136 installed CUDA cases.
Ten worker-process scenarios cover the 12 GiB boundary, compute capability,
geometry failure, CPU mode and quantizer failure/retry. Driver tests verify
geometry follows CUDA-visible ordinal remapping.

This currently reconstructs NF4 values into BF16/FP16 parameter storage. It does
not yet provide compressed GPU storage or Python's complete memory/residency
behavior. The CUDA candidate passes ten image tests and two actual routed
requests: storyboard video in 168.70 seconds and ordinary video in 168.80 seconds.
Both use original cached components and restore the 220000-token LLM context;
invalid input returns 400, subsequent chat passes, and the source `.env` remains
unchanged. The storyboard MP4 matches the corrected derived-weight diagnostic
byte for byte, with the raw scene's graphic and performers still visible at four
seconds. Exactly two engine launches confirm no hidden retry. Cgroup memory peak
is 40,968,253,440 bytes including model mappings/page cache, not compressed storage.

The 89 unchanged WorkerRouter wire tests pass again (36.914 seconds). The first
full native run passed 114/115; its HTTP WebSocket read failed and then passed on
an isolated rerun. The test client could discard coalesced handshake/frame bytes;
it now consumes exact boundaries and has a deterministic two-frame regression.
Production transport is unchanged. The fresh full native suite passes all 115
tests in 415.26 seconds.

The validated image is now deployed as `ezlocalai:cuda`, image
`sha256:542fd39d1e26ea73e9a7864b8945788bf80d2547aad4f6d7462326ed8bcbee4e`.
Production is healthy and registered with the existing Rust router. Routed chat,
SSE (including its terminal marker) and vision pass. Effective environment,
original `.env`, mounts, ports and GPU requests are unchanged. The prior image
is retained as `ezlocalai:cuda-before-nf4`; the original Python container
remains stopped. Evidence, binary/source hashes, output hashes and explicit scope
limits are in `tests/fixtures/nf4_runtime_validation.json`. Full parity remains
incomplete; this deployment closes the automatic-conversion/visible-conditioning
milestone, not compressed memory or persistent residency.

## Music-video HTTP integration (2026-10-19, earlier candidates)

Both local music-video routes now connect native image decoding, ACE-Step,
scene planning, default storyboards, positioned conditions, video generation,
theorem overlays and FFmpeg stitching. The shared video-session helper retains
one video lock and one LLM unload/restore across the full scene loop, then
restores displaced models before final stitching. Ordinary video requests
retain their existing per-output handoff. The CUDA image enables the renderer;
host builds without `EZLOCALAI_STORYBOARD_DEPS` still return 501 locally.

The focused tests use actual HTTP handlers, synthetic engine processes and real
FFmpeg. They cover image precedence, scene audio, both aliases, URL/base64,
two-scene handoff and restoration after validation, scene and stitching failures.
All 111 native tests pass (410.92 seconds); 89 unchanged Rust WorkerRouter wire
tests pass through the C++ tunnel (36.226 seconds). Six numerical/rendering
checks also pass inside the rebuilt CUDA image.

The initial three real CUDA HTTP cases cover reused audio across two scenes,
new ACE-Step audio with a theorem overlay, and routed base64 output. All restore
the original 220000-token LLM context; invalid dimensions return 400 and restore
it too. The source audio is padded to five seconds, matching Python's stitcher.
Evidence is in `tests/fixtures/music_video_http_validation.json`.

Actual Python comparison exposed a visual regression despite pixel-identical
storyboard inputs. Three concrete discrepancies are corrected in the candidate:

- Raw Diffusers cross-attention receives scheduler timesteps (sigma times 1000)
  and preserves conditioned-video token masks. The actual GGML helper matches
  four inputs from each of 24 installed default-mode transformer traces, while
  preserving legacy standalone GGUF behavior.
- The installed pipelines calculate scheduler shift from configured maximum
  sequence length, independent of requested clip size. Equal base/max shifts
  reproduce the reference checkpoint's mu=2.05: 72 captured cases match within
  2.98023e-7. Arbitrary replacement scheduler configurations remain uncovered.
- Python uses CPU float32 noise with seed 42 even on CUDA. Native generation now
  selects that RNG explicitly; 88 consecutive draws across 44 captured cases
  match within 2.38419e-7.

CPU noise restores a recognizable concert scene. A remaining difference loses
the theorem graphic in later frames. Supplying identical Python inputs directly
to the native transformer produces first-step video predictions correlated at
0.999874 (positive) and 0.999841 (negative); the first two transformer blocks
correlate above 0.99997. Replaying only Python's prompt embeddings throughout the
native pipeline preserves the graphic and performers. This localizes the visible
difference to text conditioning, without separately isolating NF4 quantization
from text/connector arithmetic precision. Python selects nested NF4 Gemma weights
on this GPU; that candidate used unquantized BF16 weights. The automatic NF4
follow-up above now preserves the graphic.

The corrected CUDA candidate is retained separately as
`ezlocalai:cuda-music-video-corrected-candidate`; its deployment was held
for text-conditioning parity. The automatic NF4 candidate above supersedes this
hold. No Python runtime was introduced.

The packaged candidate also passes two final Rust-router tunnel requests: music
video (133.25 seconds) and ordinary text-to-video (122.14 seconds). Both restore
the 220000-token LLM context; invalid input and subsequent chat pass too. The
music MP4 is byte-identical to the corrected CPU-noise experiment. The ordinary
clip has 41 frames at 24 fps; its 3.38-second audio follows Python's 24 kHz label
convention, so container duration is longer than the video stream. An initial
harness assertion incorrectly compared container duration; it was corrected to
check both streams independently. The source `.env` is unchanged and production
was restored healthy. Full evidence: `tests/fixtures/ltx_alignment_validation.json`.

The native video engine still runs as a separate `sd-cli` process per scene,
so its weights reload even though the LLM stays displaced across the scene
loop. Persistent video weights, full model numerical equivalence, complete
image-decoder/URL error parity and the previously documented model/residency
gaps remain open.

## NF4 quantization groundwork (2026-10-19)

The standalone C++ `nf4.hpp` helper implements block-64 NF4, nested block-256
statistics and BF16 reconstruction, with the upstream MIT license retained.
It passes 136 actual installed CUDA bitsandbytes cases byte for byte, including
partial blocks, zero weights, nested statistics and signed-zero reconstruction.
Two real Gemma layer matrices also match exactly across 74,711,040 BF16 values,
packed weights and compressed statistics. The additional `nf4_reference` CTest
passes; this follows the 111-test suite recorded above.

The complete encoder comparison exposed a remaining rounding difference in the
initial CPU mean: 61,614 BF16 values across 76 of 336 matrices differ from Python,
out of 10,758,389,760 projection values. Host emulation of installed PyTorch's
CUDA reduction order now matches all 336 captured scale means using the actual
GPU geometry (84 SMs, 1536 threads per SM). The helper accepts that geometry
explicitly; four real scale arrays also pass a dedicated CTest. The full reconstructed-weight recheck reduces the mismatch to 404 BF16 values
across four matrices (maximum absolute error 0.0009765625). Complete numerical
equivalence is still not claimed. Both new NF4 CTests pass, following the
111-test suite above.

A separate real routed video using diagnostic NF4-derived weights preserves the
theorem graphic at four seconds (108.71 seconds). It restores the 220000-token
context and production health, with the original `.env` unchanged. This confirms
quantization is a major source of the visible difference; it does not establish
full numerical equivalence. The original model files are untouched. This first
experiment uses the earlier CPU mean; its remaining rounding differences are
retained in the record. A second routed run after CUDA mean-order correction
also preserves the graphic (113.43 seconds), restores context and production
health, and leaves `.env` unchanged. The 404 residual value differences are
localized to seven nested-statistic byte codes: packed NF4 weights, offsets
and nested block maxima match in all four affected matrices. Their remaining
code-selection difference is unresolved. The automatic-loading follow-up above
now integrates NF4 conversion. Compressed GPU storage/matmul, Quanto fallback
and complete model-placement policy remain open. See `tests/fixtures/nf4_README.md` and
`nf4_validation.json` for evidence and reproduction.

## Music-video planning and stitching groundwork (2026-10-19)

The native music-video core now implements frame-aligned scene planning,
environment defaults, lyric/theorem selection, music request construction,
audio reuse, scene prompt/image/condition precedence, storyboard decisions,
single-session scene orchestration, final metadata, output-path resolution
and FFmpeg stitching. Its model/graphics/lifecycle operations are callbacks
now connected by the server adapter described above. The initial core
checkpoint kept the HTTP route disabled until default rendering was available.

1,040 executions of the actual Python helpers/orchestrator match the C++
implementation. The orchestration oracle substitutes model, graphics, path
and stitch operations, so these checks establish decisions and call ordering.
Separate real FFmpeg tests verify scene order, duration, overlays and base64
output. Six direct executions of Python's actual stitcher and the C++ stitcher
produce byte-identical MP4s with the same host FFmpeg, including transparent
overlays and audio padding. The final music-video and manifest CTest targets
pass together in 1.04 seconds.
Evidence and limitations are in `tests/fixtures/music_video_README.md`.

The default storyboard and theorem-overlay renderer is now implemented in C++,
including Pillow drawing, FreeType/Raqm text shaping, superscripts, Unicode
case conversion, lyric wrapping, and all six scene phases. All 330 raster/text
cases and 300 complete storyboard/overlay images match actual installed Python
pixels exactly. The checks cover RGB/RGBA, strokes, multilingual and multiline
text, small/odd/invalid dimensions, negative scene indices and long/hyphenated
lyrics. A native-only dependency stage packages checksum-pinned Pillow 12.3.0
FreeType 2.14.3/HarfBuzz libraries, compiles bundled Raqm 0.10.5, carries licenses,
and verifies the DejaVu font hashes. Both image suites passed during the CUDA
build and again in a fresh container (1.02 seconds); native linkage has no
missing libraries or libpython. Candidate image `1a6adf6a4078a2cd12d31e5808717b59d03354818a0929e62e344e5e27a28165`
is retained as `ezlocalai:cuda-storyboard-candidate`, not deployed.
See `tests/fixtures/storyboard_README.md` for scope and reproduction.

API decoding and the shared video/LLM handoff are now integrated above;
real combined music/video/router verification is recorded above. No Python
runtime dependency has been introduced into the C++ worker. Production remains
on the healthy LTX CUDA image below, with its original effective configuration.

## LTX worker integration (2026-10-19)

The local video handler now dispatches LTX models through a native runtime,
using the same admission, auxiliary eviction, LLM unload/restore and resource
reporting flow as Wan. It supports text-to-video, first-image input and
positioned conditions, URL/base64 responses and repeated outputs. The existing
Hub snapshot implementation resolves Python cache refs and downloads only
the required components; every indexed shard is checked before engine startup.
The CUDA Dockerfile now applies both pinned LTX engine patches.

Installed source inspection established that first-image input uses Pillow
BILINEAR resizing, distinct from the float cover-resize/crop used for positioned
conditions. All 64 captured installed-processor cases match exactly. Runtime
tests exercise real child processes and FFmpeg encoding, condition precedence,
child-only pageable-memory settings, GPU ordinal selection, sequential/reduced
retries, timeout cleanup, HTTP batching and LLM restoration after success and
failure. Model tensor execution is substituted in these tests. The host suite
passed 102 tests, followed by six focused checks after the audio correction;
89 unchanged Rust WorkerRouter wire tests also passed.

Real CUDA HTTP tests exercised direct text-to-video, routed positioned
conditions, routed first-image input and a routed 512x320, 41-frame request.
All four produced decodable video and stereo audio, drained text admission,
and restored the original 220000-token LLM context. Invalid video dimensions
also restored the LLM. Resource polling still times out briefly during the
synchronous LLM reload. Music and TTS CPU inference from the repackaged image
retain their source-reference output hashes.

Visual review caught a defect these structural checks missed: the larger
native clip did not depict the requested red ball, while the actual Python
reference did. Deployment was held. Raw Hugging Face Gemma3 normalization
weights require a unit offset that normal GGUF conversion folds in. The
adapter now enables that offset only for raw Diffusers loading. Its RMS math
passes 36 installed-Transformers cases in both raw and folded-weight modes
(72 GGML graphs, maximum error 2.38419e-7). The corrected clip shows a rolling
red ball beneath a table, improving object correspondence but still differing
from the reference prompt's tabletop action. This does not establish numerical
or general prompt-fidelity parity.

That run also exposed a separate native VAE-loading SIGSEGV. The core backtrace
identified an unsafe lazy name-conversion table shared by loading threads.
Guarded static initialization fixes the race; a 32-thread regression reproduces
an incomplete mapping in the original source and passes 20 fresh executions
with the fix. The router retried the crashed request successfully, then the
original worker was restored healthy. The rebuilt image passes five fresh
CUDA cases without an engine failure or router retry: direct text-to-video,
routed conditions, routed first-image input, and two routed 512x320/41-frame
clips. The larger clips take about 99 seconds each including LLM restoration.
The red-ball MP4 is byte-identical to the successful normalization-only run;
the second clip visibly shows the requested yellow rubber duck on blue water.

CUDA image `bf51835f359370458db36aec93c2af5a6ae35c60e78510fa5fc4f143be8f6290`
is now deployed and healthy. The original `.env` hash is unchanged and no
effective environment values changed. The previous image remains tagged
`ezlocalai:cuda-before-ltx` for rollback. Run provenance, output hashes,
the rejected earlier candidate and remaining limits are recorded in
`tests/fixtures/ltx_http_validation.json`. Live verification finds exactly one
registered worker and HTTP 200 for routed chat, streaming and vision. The
worker-only parity report still correctly lists 36 partial rows; full parity
remains incomplete.

Remaining engine gaps include BNB/Quanto text encoders, distinct sequential
versus model-offload residency, persistent loaded model memory, GGUF connector
reconstruction, multi-frame internal conditions, complete CPU inference and
Python numerical/export parity. Follow-up source inspection traced the audio
length difference to native ceiling versus Python ties-to-even rounding of the
audio latent count. The engine now uses the installed Python calculation, with
196 captured cases. The worker labels the waveform at 24 kHz during export,
matching the source exporter without resampling. Real native output now has
the reference short clip's 0.660-second stereo audio duration at 24 kHz;
this verifies shape/timing, not waveform or video numerical equivalence.
Explicit `VIDEO_CONNECTORS_SOURCE=gguf` currently reports an unsupported error.
Earlier sections record historical milestones and their then-current limits.

## LTX native component loading and CUDA execution (2026-10-18)

An additional pinned stable-diffusion.cpp patch loads the existing Python
Diffusers component snapshot directly. It maps video/audio VAE, full published
connector, Gemma3 and vocoder names without creating duplicate model files.
The adapter handles the fourth decoder upsampling stage present in LTX-2.3
but absent from the installed legacy single-file conversion dictionary.
534 source/weight-metadata cases cover video/audio VAE and connector names;
real execution additionally exercises the encoder and vocoder mappings.
The CLI can export lossless PNG frames with an audio sidecar for final MP4
encoding. Both engine patches compile successfully with CUDA.

Native text-to-video now generates nine 64x64 frames and non-silent 48 kHz
stereo audio from the shared cached weights, using two Euler steps and seed
42. The PNG/WAV decode and mux into a valid MP4. A 17-frame conditioned request
also succeeds with first/middle/negative latent indices, separate strengths,
and a skipped out-of-range index. These are small execution checks, not
production-resolution quality or Python numerical-equivalence tests.

The conditioned request first timed out while loading transformer weights
under severe host-memory pressure. Repeating it with memory-mapped weights
and pageable host buffers (`--mmap`, `GGML_CUDA_NO_PINNED=1`) succeeded in about
56 seconds. Repeating text-to-video with that memory mode preserved every PNG
and WAV byte. The native BF16 encoder/offload path still differs from Python's
BNB/Quanto choices. The unquantized CPU probe reached its 180-second deadline
after text encoding/projection; complete CPU generation remains unverified.

The actual Python automatic residency path OOMed for the short request on
this machine. An oracle executing its real `Pipes.py` retry method with real
models also exhausted memory after sequential reload and the reduced-GPU
retry, returning no video. This oracle substitutes resource-manager callbacks
and model ownership, so it is not a complete Python server/LLM-handoff test.
A fresh Python process configured for sequential offload completes inference
in about 56 seconds. Its captured output contains nine frames and 15,840
audio samples per channel. The harness initially lacked an output directory;
replaying the original exporter with the captured tensors and a valid
directory produces a nine-frame MP4 with 24 kHz stereo audio lasting 0.660
seconds. Native audio instead contains 17,760 samples per channel at 48 kHz
(0.370 seconds). Audio shape/rate/duration and numerical parity are therefore
unresolved, rather than inferred from successful execution.
See `tests/fixtures/ltx_README.md` for scope, provenance and artifact locations.
`tests/fixtures/ltx_inference_validation.json` records the run results, hashes
and limitations. All four focused LTX/parity CTest targets pass. After the
isolated runs, the unchanged production worker is healthy and registered
once with context 220000; routed chat, streaming and vision return HTTP 200.

The production CUDA image and HTTP handler have not yet adopted either patch;
local LTX requests still return 501. Native HTTP/router integration, complete
configuration and residency/quantization behavior, multi-frame internal
conditions, export parity and normal-size inference remain outstanding.

## LTX policy and arbitrary-position conditioning groundwork (2026-10-18)

The native LTX settings, pipeline choice, GPU residency, device/dtype choice,
and OOM recovery policy pass 2,006 source-executed Python cases. The loader's
10 GiB short-clip threshold is preserved separately from VIDEO_UTILS's 12 GiB
standalone default. LTX failures can request a sequential reload or a reduced
GPU retry; the Python method named `_generate_cpu_fallback` does not actually
move execution to CPU. These are policy decisions, not verified model loading
or retry execution.

The packaged stable-diffusion.cpp supports LTX-2.3 but originally exposes only
start/end images. An experimental pinned-source patch now adds single-image
conditions at arbitrary latent indices with independent strengths. Auditing
the installed Diffusers pipeline established that these are latent indices,
not pixel-frame indices: negatives wrap and out-of-range positives are
skipped. Non-first conditions append separate keyframe tokens. Float bilinear
cover-resizing and center cropping preserve the installed preprocessing rule.
The patch builds successfully as a CUDA engine. Another 137 installed-source
cases verify preprocessing/index/coordinate helpers, with an additional
multiple-keyframe mask/position assembly check. Both focused CTest targets
pass. Provenance and numerical tolerances are in `tests/fixtures/ltx_README.md`.

This patch is not yet enabled in the Dockerfile or the HTTP backend. LTX HTTP
requests still return 501; the live CUDA worker remains on the previously
verified music image. Published Diffusers connector/encoder/VAE loading,
quantization, multi-frame internal conditions, audio export, lifecycle/retry
integration and actual LTX inference remain incomplete. The existing shared
cache contains the Python weights, but their names/layout and quantization
must be reconciled with the native engine rather than assumed compatible.
A read-only CPU loading probe confirms this: the native engine identifies
LTXAV and Gemma 3 but aborts in the VAE constructor before inference because
the unconverted Diffusers VAE layout is not recognized as version 2. The
probe changed neither the cache nor the production worker. Five malformed
condition CLI inputs also fail cleanly before model loading. Build hashes,
test counts and the failed probe are recorded in
`tests/fixtures/ltx_validation.json`. The video parity row remains partial.

## ACE-Step packaging, model preparation and real music verification (2026-10-18)

The CUDA image now builds ACE-Step at the installed Python worker's revision,
`5b88ded7919285d5ad5b1e86ffe3378314f305db`. Startup prepares its four selected
weights before the API starts, using the existing native Hub downloader.
Selection, deduplication, existing-path reuse, external-server bypass,
`ACE_STEP_PRECACHE`, and logged/nonfatal preparation failures match 45 fresh
executions of Python's `precache_music_model`. Real inference reused the
existing shared cache; this pass did not download another copy of those
multi-gigabyte weights.

Real testing exposed an ABI collision: the image-wide llama.cpp library path
caused ACE-Step to load llama.cpp's GGML and reject the DiT model's long tensor
names. The global llama.cpp path was removed. The installed TTS executable now
has an explicit runtime path to its own llama.cpp libraries; ACE-Step resolves
its own GGML libraries. The Docker build checks that resolution and missing
dependencies. llama.cpp now builds in a separate cached stage, independent of
first-party API changes, and the API is built once with its audio dependencies.
The pinned source revision and engine metadata are retained.

Five real requests produce six audio files on each of CPU and CUDA. Every
audio byte and response metadata field matches the installed Python wrapper,
apart from timestamps and generated filenames. These cases include seeded LM
generation, direct synthesis using actual generated audio codes, WAV16/24/32,
persisted MP3, the default batch limit, and an explicit batch limit of two.
The default ACE server returns one track for `n=2`; configuring
`ACE_STEP_MAX_BATCH=2` produces two, matching Python. CPU and GPU expectations
are captured separately. The C++ runner checks all four weight hashes, executes
the real worker handler and engine, and cleans up its process and outputs.
See `tests/fixtures/music_deployment_README.md` and the CPU/CUDA fixtures.

The newly packaged engine passes all six hashes on both devices. The installed
Qwen TTS executable also passes five requests/six exact float32 buffers on
each device after the library-path change. All 98 host CTest targets pass
(419.47 seconds), including the existing music protocol/startup/handoff tests
and new preparation cases. The earlier 89 unchanged WorkerRouter wire tests
remain recorded in the previous section; this pass adds real music through
the actual Rust router rather than another protocol-only run.

With the original Qwen3.8 27B model and context 220000, direct WAV generation,
tunneled two-track WAV generation, and tunneled MP3 generation/download all
match the Python audio hashes. Each request reserves text admission and
restores the original context, parallelism and measured 20.015381 GiB LLM
footprint. A failed ACE synthesis job returns HTTP 502 and also restores the
LLM. Chat returns READY before and after. Successful request times including
restore were 14.586, 15.605 and 12.346 seconds on this machine. Resource polling
hit four two-second deadlines per request during restoration; requests and
heartbeats recovered. Python's restoration is also synchronous, but exact
resource/heartbeat latency equivalence has not been established.

Deployed image:
`sha256:a72c50b630e18eea8b95e810fe17390ffc11df8d01ed07e350217e3bcda633bc`.
Its native executable SHA-256 is
`2884d6fb6b0059e2f5e775a1e0652f621835a3e665c83092c8fad89fadc65cd5`.
The original `.env` and model/output/voice mounts are retained: all 20 checked
settings match effectively (18 raw matches; two absent-versus-empty defaults).
The production media switches remain disabled as configured. The deployed
worker is healthy and registered once with the live Rust router; routed chat,
SSE and vision all return HTTP 200. The original Python container remains
stopped and available for rollback. Detailed artifact hashes, model tests,
handoff results and script provenance are in
`tests/fixtures/music_deployment_verification.json`.

Music rows remain partial. Local music-video scene generation/storyboarding/
stitching, complete MIME/JSON/deadline behavior, placement/residency accounting,
and lifecycle timing/concurrency equivalence remain open. The broader worker
goal still includes LTX video and other documented gaps. Worker-only parity
has 57 rows: 21 implemented and 36 partial, covering all 45 current worker
routes. Strict enforcement correctly fails; no Python router service port is
required or counted toward the replacement goal.

## Local Qwen image implementation and CUDA verification (2026-10-18)

Current Python `IMG.py` uses Qwen-Image-2.1 through stable-diffusion.cpp, even
though some `Pipes.py` comments still mention FLUX. The native implementation
now covers that engine wrapper, the four lazy-downloaded weights, explicit
CPU/secondary-GPU selection, memory-based CPU offload, cross-process inference
locking, primary init images plus ordered vision references, output validation,
base64/URL results, and temporary-file cleanup. Image requests now reach the
local engine instead of the modality 501 path.

The edit schema and forwarding paths were updated against current Python:
`images` alone is valid; empty references, dimensions, strength and count are
validated; image-server, Rust-router-client and fallback requests preserve
references/strength/count and use the edits endpoint. Generation fallback
sends the source client's default count of one. Existing tests expecting edits
to discard images or become generations were corrected against fresh captures.

The image pipeline uses source GPU handoff policy, registers image residency,
blocks LLM admission during a handoff, restores prior LLMs on success/failure,
and unloads/restores idle auxiliary services. New idle-only TTS/STT eviction
checks leases under the pool lock before closing anything. The native video
implementation still only supports Wan, so restoring other video model types
remains incomplete. Image lifecycle timing events and some download/transport
edge behavior also remain open, including image-server null/invalid JSON
fallback responses. URL PNG output preserves RGBA pixels and strips generation
metadata, but libpng compression differs from Pillow.

Source captures contain 56 engine-wrapper cases, 66 edit-schema cases, and 640
GPU handoff decisions. The Python project's own 18 Qwen-image tests pass.
Native HTTP tests cover local results, resource reporting, handoff restoration,
reference ordering and forwarding through isolated image/router/fallback peers.
Fixture provenance is in `tests/fixtures/qwen_image_README.md`.

The full host suite passed 98/98 targets (419.83 seconds), and the three
focused HTTP/image targets passed after the RGBA correction (34.64 seconds).
WorkerRouter's unchanged 89 wire tests passed through the native worker tunnel
(38.630 seconds). The full-suite binaries precede the final RGBA/resource-count
adjustment; the focused rerun covers those changes. No legacy Python router
service port was added or required.

The CUDA Dockerfile now builds `sd-cli` at the installed Python engine revision
`c678dfe704a2230342376b46add9c8ca736a653d`, before copying first-party sources to
retain its build cache. `EZLOCALAI_MEDIA_CUDA_ARCHITECTURES` configures its GPU
architectures (86/89/120 by default; 86 used on this machine).

The CUDA model verification is now complete for three 256x256/default-20-step
cases: text-to-image, primary-image editing, and two-reference editing. All
RGBA pixels match the installed Python engine with a test-only seed of 42.
A real multiple-reference edit also matched through the C++ WebSocket worker
tunnel and an isolated Rust WorkerRouter. URL outputs preserve RGBA and remove
generation metadata. Base64 PNG bytes differ only in the descriptive engine
version metadata (`unknown` in the shallow native build versus Python's
`master-889-c678dfe`); the commit matches. URL PNG compression differs from
Pillow. These byte differences remain recorded rather than called exact file
parity. See `tests/fixtures/qwen_image_deployment_cuda.json` for hashes.

The original Qwen3.8 27B LLM was also tested with image generation temporarily
enabled. The real handoff reserved text admission, released the resident LLM,
generated matching pixels, and restored context 220000 with one parallel slot
and the same 20.017639 GiB model footprint. Chat returned READY before and
after. The complete image request, including restore, took 12.385 seconds.

The tested image is now deployed as `ezlocalai:cuda`, ID
`sha256:0b593f7302bc4708977c7ff452ab9b6208a57a5f56516c7a32474abe44421c5f`.
The deployed native executable SHA256 is
`0ca04de5320263d2353b6083eec1185eeab8b227763ee26635aa4b72df81e552`.
It is healthy, registered with the existing Rust router, and passed routed chat,
SSE and vision checks. The original `.env` and model/output/voice mounts are
preserved; all original disabled media flags remain disabled. The original
Python container stays stopped and available for rollback. Temporary GPU test
containers and the isolated router were removed/stopped. This deployment
supersedes the older image IDs in the sections below.

Full replacement parity remains incomplete: local LTX video, music-video,
ACE-Step packaging/real inference, and the tracked lifecycle, config and
transport edges still need work. Native code remains C++; external Python
scripts were used only for validation.

## Native CUDA speech packaging and inference (2026-10-18)

The CUDA image now builds the existing native STT and CTTS adapters and Qwen
TTS executable. It packages checksum-pinned CTranslate2 4.8.2, ONNX Runtime
1.31.0, LibTorch 2.9.1+cu128, matching CUDA/cuDNN libraries, Silero VAD assets
and libsndfile. Native library hashes match the installed Python image. Only
headers, native libraries, model assets and licenses are extracted from the
upstream distributions; the image has no Python interpreter or Python binding
dependency. CUDA driver stubs are used for linking during the build, while
runtime resolves the real NVIDIA driver. The dependency lock and extraction
script are under `cmake/`.

Fresh captures execute the installed Python classes with cached real models.
Eight focused test targets have passing verification across the final runs:

- Four CPU int8 and four GPU float16 large-v3-turbo STT calls cover plain text,
  word timestamps, batched decoding and language detection. GPU output is exact.
  Independent eight-thread Python CPU runs selected adjacent 20 ms frames for
  three word boundaries; the CPU test accepts only those captured word-time
  variants. It retains exact text/structure checks and verifies device, compute
  type and cleanup.
- Five Qwen CPU requests and five GPU requests match six float32 audio buffers
  per device by SHA-256, including a two-window stream.
- The production executable passes real CPU HTTP WAV, cache-hit, HTTP streaming
  and WebSocket packet-hash comparisons. Separate GPU HTTP checks pass plain,
  word-level and automatic-language transcription, invalid-audio HTTP 400,
  buffered 24 kHz WAV and streamed PCM.
- All 2,122 CTTS audio conversion cases pass after independent recapture from
  the installed Python version. Forty-five resampling outputs and five NaN
  payloads differed from the old capture; no production algorithm was changed.
- All 18 Torch GPU allocation/RNG/flags/fallback/cleanup scenarios and the
  hidden-GPU test pass. The current source reports 25,258,688,512 device bytes
  and eight CPU fallback threads; the old hardware capture reported a different
  capacity and four threads. Those expectations were recaptured from the source.

Capture provenance is in `tests/fixtures/audio_deployment_README.md`,
`ctts_conversion_README.md` and `stt_torch_README.md`. Logs include
`/tmp/ezlocalai-audio-{cpu-validation,cpu-refreshed,production-cpu-api,gpu-validation-final}.log`
and `/tmp/ezlocalai-ctts-installed-validation.log`. Raw evidence remains under
ignored `build/validation/`. Initial failures and their causes remain in
the logs; this is not a claim that the first run passed.

The new image is `sha256:336a19cde395d077ccdeab9f69907bd5bd02dbb445f203426101c8b768853d87`.
The deployed production executable has SHA-256
`da544287961ac3bb22b60afebc83a82ef585ebf9cc3e64dbd287f25eae13e328`;
the HTTP checks use that exact artifact. The original `.env` and shared mounts
are retained, including its disabled speech switches. Audio tests use temporary
overrides in an isolated container.

After cutover the container is healthy, WorkerRouter registration advertises
the configured text/vision worker, and routed chat, SSE and vision return HTTP
200. Chat again returns `READY` with 18 prompt and two completion tokens.
The configured context remains 220,000 with one slot; llama.cpp reports its
internally padded context as 220,160. The current measured LLM allocation is
20.0267333984375 GiB. The original Python container remains stopped and available
for rollback. The worker gate still reports 57 rows, 21 implemented and 36
partial, with all 45 source worker routes covered.

This closes native x86-64 CUDA speech dependency packaging and the specific
inference cases above. It does not establish full speech parity: diarization,
microphone/wakeword behavior, exhaustive cancellation/resource handoff and
other platforms remain open. Native image/video/music engine packaging and
actual model inference, music-video execution, broader memory/residency policy
and remaining worker API/configuration differences still require work. The
worker-only strict gate remains intentionally failing.

## Allocation accounting and voice-fixture repair (2026-10-18)

Managed LLM startup now samples free memory on the selected main GPU before
creating the engine and after the startup inference probe. Negative changes
clamp to zero; unavailable telemetry does not fail startup. Successful loads
retain the maximum observed allocation for that model and publish it in
`vram_estimates_gb`. Shared ResourceManager records now include owned text and
vision models, resource activity during inference, and removal/restoration
during handoff and destruction. External engine URLs retain external metadata.
The Python convention of registering managed LLMs as `cuda`, including CPU
retries, is preserved; absent measurements use the reference type fallback.

Ten cases captured from unchanged Python allocation and lifecycle statements
pass for both text and vision, including selected-device, negative/zero delta,
missing-device and telemetry-error cases. Managed process tests also verify
active-resource eviction protection, retained estimates, unload/restore and
destruction. Provenance and limits are in
`tests/fixtures/llm_memory_README.md`. Pre-load GGUF footprint estimation,
automatic multi-GPU placement and full residency-policy equivalence remain
open; measured allocations alone do not complete memory-management parity.

The inherited voice-lifetime fixture has been independently recaptured using
the unchanged source constructor, voice guards/pools and data-URI audio route.
It now covers 843 responses (788 successful, 37 deliberate runtime failures,
18 disabled) and 1,631 inference events. All 219 scenarios and 2,073 state/event
snapshots match native behavior after fixing test environment leakage between
scenarios. The old capture contained no inference events; its exact capture
failure is unknown. No production voice behavior was changed to match either
fixture. See `tests/fixtures/voice_lifetime_README.md`.

The complete configured regression passed 96/97 in 158.10 seconds before the
fixture replacement; the repaired voice target then passed in 1.14 seconds.
Thus all 97 configured targets have passing verification across these runs,
not a single clean 97/97 full-run claim. The seven focused memory/lifecycle,
HTTP and router targets also passed together in 47.63 seconds. Logs are
`/tmp/ezlocalai-llm-memory-{build,focused,regression}.log` and
`/tmp/ezlocalai-voice-final-check.log`. Optional model/dependency targets from
older passes remain outside this configuration.

The updated CUDA image was rebuilt and deployed using the same original env
and cache mounts. Image ID:
`sha256:ce6b59cdebc620988462387f179794abaa6515220d4ddd00e401d65229f182de`.
It no longer injects `LLAMA_CPP_DEFAULT_CONTEXT`. The running Qwen vision
model reports device `cuda` and 19.86749267578125 GiB in both the shared
resource record and lifecycle estimate. A real 256-token request observed
`in_use: true, active_requests: 1` during inference and false/zero afterward.
The live router has one matching worker; targeted JSON chat, SSE and vision
all return HTTP 200 with the configured 220000 context. Artifacts are
`build/validation/cpp-memory-{resources,inference}.json`; routed
checks are in `/tmp/ezlocalai-memory-routed-inference.log`. The current
container is healthy, with the Python container still retained and stopped.

Next deployment work remains native STT/TTS dependency packaging and actual
CUDA audio inference, followed by the other supported media runtimes. The
current installation disables these modalities, so successful text/vision
cutover does not verify them. The older zero-VRAM and failing-voice observations
below describe the initial audit and are superseded by this follow-up.

## Current replacement scope (2026-10-18)

The replacement consists of this C++ worker and the Rust router at
`/opt/DevXT/WorkerRouter`. Porting the Python router service is outside the
requested scope. Its historical manifest rows and tests remain for reference;
use `parity_report --worker-only` to assess the replacement worker. Worker-side
registration, heartbeat, reverse tunnels and delegated requests remain required.
The worker gate currently includes 57 rows (21 implemented, 36 partial), with
all 45 Python worker routes represented. Strict enforcement must keep failing.

The source used for this pass is `/home/josh/repos/xtsys/ezlocalai`, commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Existing user changes to its README
and benchmark files were left untouched. Earlier test totals below describe
their own dependency configurations, not the currently configured test set.

The reverse-tunnel reader now dispatches local requests concurrently while
continuing to handle keepalives and cancellations. A bounded outgoing queue
keeps all WebSocket/TLS operations on one thread. Disconnect and shutdown
interrupt local HTTP requests and join their tasks; duplicate in-flight IDs
are suppressed. Request/response headers survive forwarding, fixed-length
bodies complete without waiting for EOF, truncated bodies report an error,
and WebSocket pong payloads echo the incoming ping. Heartbeat HTTP no longer
half-closes its socket, which caused intermittent registration loss against
the Rust server. Advertised VRAM comes from ResourceManager; model contexts
use each configured model and any live reduced context after memory fitting.

New native socket tests cover these paths. The optional
`EZLOCALAI_WORKER_ROUTER_BIN` integration test launches the actual Rust binary
and checks direct and tunnel registration, authenticated discovery, JSON/SSE,
router restart recovery and graceful deregistration. An external adapter
executed all 89 tests from the unmodified WorkerRouter `tests/test_worker_wire.py`
through the C++ tunnel (89/89 passed, 38.584 seconds); a separate 45-second
inference case survived the Rust router's keepalive interval. These establish
transport interoperability, not model-backed output parity. Logs are
`/tmp/ezlocalai-workerrouter-wire.log` and
`/tmp/ezlocalai-workerrouter-keepalive.log`. Fragmentation/large frames, full
TLS/deadline equivalence and all metadata/lifecycle transitions remain open.

CUDA Compose now accepts the existing installation's env file and model,
output, voice and whisper cache directories. Pass both `EZLOCALAI_ENV_FILE`
and Compose `--env-file` so explicit Compose environment entries use the same
values. The current machine's env disables STT, TTS, embedding, image, video
and music; testing that configuration cannot establish those modalities.
Their native dependency packaging and actual-model checks remain required.

The current host regression has 95/96 passing targets: 94/95 in the broad
run excluding `native_http` (131.79 seconds), plus `native_http` passing in
29.14 seconds. The failing `voice_lifetime_reference` was initially unable to
compile because it inherited from the final Echo backend; composition fixes
the build, but its fixture still disagrees with native behavior. Its 219
source scenarios contain no inference events and predominantly expect HTTP
500 before loading. Independently recapture the source errors and lifecycle
before changing production behavior or accepting this fixture as evidence.
It remains enabled and failing. Logs: `/tmp/ezlocalai-regression.log` and
`/tmp/ezlocalai-http-verified.log`.

Other inherited test problems corrected in this pass: the HTTP embedding
fixture no longer probes the live Rust router on port 8092; the queue fixture
explicitly represents a resident model and synchronizes its held request
instead of depending on host GPU availability and a short sleep; the
embedding executable fixture disables unrelated default voice precaching.
CTTS bit-exact fixtures use the pinned libsndfile binary with SHA-256
`c872fb84af2afc0e96b603e0000f0f407af86c49ad1126f2785e5722c7bb27d7`,
not the host library whose rounding differs. Optional CTranslate2, ONNX,
Torch and actual-model targets from older runs are not enabled in this host
configuration; the current total is not a replacement for those checks.

### CUDA cutover and actual inference

The CUDA image built successfully with the pinned llama.cpp commit and SM 86
for the RTX 3090 Ti. `/home/josh/repos/xtsys/ezlocalai/.env` and its existing
models/outputs/voices/whispercpp directories were used without changing that
installation. Python container `ezlocalai` is stopped and retained; C++
container `ezlocalai` is running and healthy on port 8091. The live router
has one worker with label `DevXT3090J`, text/vision capabilities, model
`unsloth/Qwen3.8-27B-GGUF`, quant `Q3_K_XL`, context 220000, and one available
text/vision/model slot. Direct and live-router-targeted chat, SSE and vision
requests all return HTTP 200. The text baseline returns `READY` in both
implementations with 18 prompt tokens and 2 completion tokens. Vision uses
1044 prompt tokens and returns 32 requested tokens in both, but generated
descriptions differ; this is not an exact-output equivalence claim.

The initial startup exposed a real deployment mismatch: the old Compose
`LLAMA_CPP_DEFAULT_CONTEXT=262144` silently overrode source `LLM_MAX_TOKENS`.
Compose now leaves that extension blank, and the sample env and Dockerfile
no longer inject native context overrides. The running container was
recreated with this Compose correction and its engine command line and
resource/registration responses all confirm 220000. The tested image was
built before the Dockerfile default removal; Compose explicitly overrides
the image's legacy value. A future image build must verify the raw-image
default removal as well.

Another confirmed gap: the native LLM backend's per-model resource record
still says device `llama.cpp` and `vram_gb: 0`, while Python reports `cuda`
and measured model VRAM. Aggregate free VRAM is real, and router capacity
works, but complete shared model-memory accounting remains required. The
disabled audio/media flags in this installation leave their CUDA packaging
and actual inference unverified. These gaps prevent calling this a full
replacement yet.

Comparison artifacts are under ignored `build/validation`; live
checks are logged in `/tmp/ezlocalai-{python,cpp}-inference.log` and
`/tmp/ezlocalai-live-router-inference.log`. To restore Python, stop
`ezlocalai` and then start the retained `ezlocalai` container. Do not run
both against the same port/GPU allocation.

Reference inspected: sibling `../ezlocalai`, commit `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544` (2026-10-14 audit). The native starting commit was `c7dda7e`. Reinspect the reference worktree on subsequent passes; commit identity alone does not account for uncommitted source changes.

## Completed in this pass

- Both CMake entrypoints enable only CXX. The former C implementation is now `src/core.cpp` plus `src/request_queue.cpp`. The C-compatible header retains the existing ABI, implemented by C++17.
- Request ownership, pending work, worker threads, mutexes, condition variables, and startup/shutdown are implemented with C++ standard-library objects. Queue restart no longer leaks worker arrays; callback exceptions become failed requests; waits use monotonic deadlines; cancellation state is read under the queue mutex.
- `llama_cpp_tool` replaces the Bash management tool for checkout/update, configure/build, binary discovery, version/status, model resolution, and build cleanup. It invokes argument vectors directly. Tests use a local Git remote and compile a real C++ fixture, including paths containing spaces and quoted CMake arguments.
- `ezlocalai-native-entrypoint` replaces the Bash container entrypoint. Docker wiring and documentation use the compiled tools.
- All executable HTTP test fixtures are C++ binaries with data sidecars. NVIDIA telemetry and TLS test certificate generation invoke external binaries directly without a shell.
- HTTP test servers read requests by HTTP framing. The inherited managed-llama test stalled because it waited for EOF while the client awaited a response.
- Remote TTS/image/video forwarding no longer advertises fictitious locally loaded models. The reference routes forward before acquiring local model resources. Regression assertions now reflect that behavior; local STT accounting remains tested.
- `parity_report` locates the sibling reference or accepts `--source-root`. Strict enforcement requires available sources and a nonempty manifest; malformed/unknown statuses and duplicate route rows fail validation. Regression tests exercise missing sources, partial/missing rows, unknown statuses, and new uncovered routes.

## Verification

- Debug build succeeds with GCC 13.3.
- `core`, `native_http`, `parity_manifest`, and `llama_cpp_tool` passed together. The complete HTTP suite took about 30 seconds.
- The subsequently added `parity_report_gate` test and updated `parity_manifest` both pass.
- A fresh CMake configuration succeeds with `CMAKE_C_COMPILER=/nonexistent-c-compiler`; CMake reports that the C compiler variable is unused. There are no first-party `.c`, `.py`, or `.sh` source files, embedded executable shell fixtures, `system()` calls, or `popen()` calls remaining under `native`.
- CUDA Compose configuration validates. The CUDA image itself has not been rebuilt or exercised in this pass, and real inference/model downloads were not tested.
- Route discovery finds 85 reference routes, with zero uncovered routes and zero extra manifest routes after adding missing entries. The manifest currently has 38 implemented, 58 partial, and 0 missing rows, including non-route concerns. `--enforce` fails as intended.

Passing these tests proves only their covered behavior. Inherited `implemented` labels are not a full current-reference equivalence proof. Existing HTTP tests primarily exercise native behavior with mock upstreams; they do not establish exact model output parity.

## Remaining work and next actions

1. Complete runtime consumption of the current defaults and model configuration. Default lookup and advertised capabilities now match source-derived fixtures; residency, speculation, full dotenv parsing, native model lifecycles and several consumer-specific policies remain open.
2. Complete music lifecycle and music-video execution. All music/music-video API/router route names and aliases are now present. Music executes ACE-Step LM/synthesis jobs with local process startup and LLM unload/restore. Model acquisition, packaged dependencies and remaining transport/runtime edge behavior still need work. Local music-video generation, storyboards and stitching remain absent and return 501 when selected. Drain/resume edge cases and OpenAPI refresh remain partial.
3. Audit existing router behavior against current provider fallback nodes, worker targeting, user/prefix affinity, MTP-aware model matching, capability retries, and dedicated-worker preferences. Earlier router coverage does not include these additions.
4. Implement current local modalities in C++: Qwen TTS using the reference `tts/worker.cpp` and the behavior in `ezlocalai/LlamaTTS.py`, whisper STT decoding/diarization/lifecycle, native image and video backends, ACE-Step music, music-video orchestration, and wakeword generation/training/prediction. Earlier Chatterbox and BGE notes in inherited manifest descriptions are historical coverage, not the current target.
5. Port model/resource lifecycle and GPU strategy, embedding configuration/token accounting, queue UUID/history/pruning/timeout/abandoned-stream cleanup, and the current wakeword name/path validation. The C++ queue conversion deliberately retains existing API behavior; it does not yet close all Python queue differences.
6. Refresh OpenAPI snapshots and audit every request/response schema and error branch against the current reference. Check the model-backed output and streaming behavior with differential fixtures and real supported models, not just route presence or test doubles.
7. Complete deployment/package parity, including CPU/CUDA/ROCm/RPi builds, model acquisition/cache/preload, native modality dependencies, and bundled assets/voices. Verify supported runtime configurations.

Retain the strict failure until every actual requirement is implemented and verified. The manifest is a gap tracker, not a substitute for a requirement-by-requirement completion audit.

## Configuration and capability alignment

- Replaced the outdated default table with all 147 current `Globals.py` values, and added explicit-fallback lookup with reference empty-value/precedence behavior. This includes whisper turbo, Qwen TTS, embedding/media switches, residency/speculation, and router settings. Lookup parity does not imply every runtime consumer implements the setting.
- Consolidated C ABI and worker capability detection into one C++ implementation. Registration and heartbeat now agree on enabled modalities, delegated services, embedding/music/music_video, disabled text, vision tags and local projector discovery. Capability model names follow the reference, including legacy Qwen TTS model migration and the reference Router-specific video fallback.
- Added Python-compatible Unicode whitespace stripping for configuration predicates and preserved case-sensitive projector path/extension behavior.
- Consolidated GPU scoring so worker metadata uses the full current 92-entry table, including GB10.
- Resource slots honor media enable switches and include local embedding capacity/active counts. Remote voice appending no longer marks a local TTS model as loaded. Image generation, image editing, and video disabled responses now respect current capability gating; unavailable image edits return the reference 503 detail.
- `KV_CACHE_TYPE=auto` resolves to q4_0 for the existing managed backend. Startup loads dotenv before selecting host/port. CUDA Compose exposes current media/embedding switches and current whisper/Qwen defaults.
- Added a C++ reference test that independently reads current default/GPU tables and verifies 536 capability/model-name cases produced by executing reference functions. SHA-256 headers bind fixtures to the inspected `Globals.py`, `Router.py`, and `ezlocalai/LlamaTTS.py`. Cases include all media switch combinations, delegation, whitespace, dedicated workers, disabled models, and projector discovery.
- A real native executable startup smoke check confirmed that host/port from a temporary `.env` are applied and `/health` responds healthy. CUDA Compose validates. The subsequent seven-test suite passed after the embedding changes below.

The defaults/capability portion of next action 1 above is done. Runtime consumers, full dotenv semantics, and model lifecycle remain open. Advertising music/music-video does not implement those endpoints or engines.


## Embedding request and engine configuration alignment

- The API now uses `EMBEDDING_SERVER` and `EMBEDDING_SERVER_API_KEY`, with cached model probes, 120-second forwarding, current optional fields, and null omission. `TEXT_SERVER` no longer handles embeddings. Fallback payloads retain dimensions/encoding fields and use the reference 60-second timeout.
- Disabled embeddings validate first, then can dispatch through `ROUTER_URL` with reference API-key priority and a live-worker health check; otherwise they return the current 404 detail.
- Local embeddings run outside the LLM queue, with capability activity accounting. Request and environment dimensions enforce the current 32..1024 range, truncate then L2-normalize long vectors, preserve shorter/zero vectors and native token usage, resolve response model names, and map reference client/context/runtime errors. Local encoding_format remains ignored, as in the reference.
- A source-executed fixture oracle covers 86 current EmbeddingModel/Embedding.get_embeddings cases. The C++ regression checks exact status/body and the backend request, along with disabled validation and LLM queue isolation. SHA-256 headers bind the fixtures to current Globals.py, app.py, and ezlocalai/Embedding.py.
- The fixture caught an inherited JSON lookup problem: a string value equal to a field name could be selected as a key. The new embedding path uses actual top-level object fields, including vector extraction from standard `{"object":"embedding",...}` entries.
- Managed embeddings and `llama_cpp_tool model-status` share Qwen3/Q8_0 model/alias/path/HF resolution. Embedding discovery requires a matching model and quantization, instead of using an unrelated text GGUF. Explicit native path/model overrides remain supported.
- Managed embedding startup uses independent context, one internal slot, LAST pooling, batch/ubatch caps, embedding KV types, embedding GPU layers/main GPU, and CPU no-KV-offload. A text-disabled service selects native embedding inference without starting a fictitious LLM.
- HTTP regressions cover dedicated credentials, null/default models, full forwarded optional fields, availability caching, fallback fields, disabled router dispatch, matching local Qwen discovery, HF arguments, startup flags, and context/batch limits.

Remaining embedding work: full parallel-instance pools and waiting semantics, resident/transient lifetime and warmup, reference VRAM estimates and progressive GPU retries, CPU operation offload, cache/download selection, complete schema edge cases (including large integers and Unicode), and comparison with real model outputs on the exact supported engine/version. No real model inference or downloads were performed in this pass. No exact-parity completion claim is justified.

Manifest after the embedding pass: 96 rows, 46 implemented, 40 partial, 10 missing. Reference route discovery still finds 85 routes with no uncovered or extra manifest routes. Strict enforcement continues to fail on the remaining gaps.

Final verification for this pass: the C++-only build succeeds and all seven CTest targets pass (34.22 seconds total). `git diff --check` and CUDA Compose validation pass. The reference checkout remains clean. Native source/tool/test scans find no first-party C, Python, or shell files and no system()/popen() calls. The CUDA image and real inference have not been exercised.

## Router maintenance and admission accounting

- Added worker-authenticated `/v1/router/drain` and `/v1/router/resume`. The router enforces one fleet-wide owner lease, validates the 30..300-second TTL and boolean idle requirement, checks live-label uniqueness, allows renewal while a worker is absent or re-registering, and preserves owner checks when resuming an expired lease. Integer string validation includes Unicode decimal digits and large values.
- Drain state follows stable worker labels across process IDs and router restarts. `ROUTER_DRAIN_FILE` defaults to `/data/router-drains.json`; an empty path disables persistence. Writes use a same-directory temporary file, fsync and atomic rename, with memory rollback and temporary-file cleanup on failure. Heartbeats cannot clear a maintenance lease.
- Dispatch reservations no longer mutate worker-reported counters. Locked LLM admission prevents simultaneous selection from oversubscribing a slot. Text/vision capacity reservations expire after `ROUTER_RESERVATION_TTL` (default 15 seconds); media/speech/embedding reservations last until completion. Separate active request lifetime remains visible to maintenance after a short LLM reservation expires. Direct and tunnel proxy streams retain their token through completion or response destruction.
- Public worker state includes reported and router counters, effective capability/model slots, instance counts, tunnel-adjusted priority, drain expiration, active request count and maintenance busy count. Worker listings advertise `maintenance_protocol: 1`. Re-registration preserves active tokens, errors and circuit state. Explicit registration `extra` metadata merges with unknown top-level fields as in the current reference.
- Slot accounting now includes model aliases, swap residency, music/video overlap and shared LLM unload dependencies. Total capacity does not double-count those shared slots. Full worker selection, managed provider pools and affinity policies remain separate open work.
- Added 219 fixture cases generated by executing the current `WorkerInfo`, `WorkerRegistry` and FastAPI maintenance/registration handlers. The C++ test compares HTTP status and complete ordered JSON after normalizing numeric spelling and four clock-dependent fields. SHA-256 headers bind the fixture to `Router.py`, `router_app.py`, and `Tunnel.py`.
- Additional C++ tests verify lease duration, restart recovery, expired ownership, persistence failure rollback, cleanup of temporary files, authentication, sixteen concurrent lease contenders, heartbeat/re-registration during active requests, LLM reservation expiry, speech reservations surviving that expiry interval, new-dispatch exclusion, deferred stream completion and discarded-response cleanup. Upstreams are local controlled C++ sockets; tests use ephemeral ports and bounded waits.

Verification: the full eight-test CTest suite passed in 34.77 seconds. After extending the maintenance test to cover non-expiring speech reservations and speech streaming, that target passed again in 1.88 seconds. The C++ build succeeds. No real model inference or container image rebuild was performed.

Manifest after the maintenance pass: 96 rows, 46 implemented, 42 partial, 8 missing. Drain/resume are deliberately **partial**, not declared exact: OpenAPI snapshots are stale, malformed persisted entries differ from Python's permissive load behavior, and nonfinite JSON/numeric/configuration edge cases still need reconciliation. Strict enforcement must continue to fail. Full 1:1 parity remains unfinished.

## Music API and ACE-Step client

- Added all eight API/router music and music-video route names and aliases in C++. API requests follow the current `MusicCreation`/`MusicVideoCreation` field order, required fields, optional defaults, Pydantic-style validation, exclusion of null fields, nested video conditions, and post-validation text/duration requirements. Integer music fields preserve large seeds instead of narrowing them to 32 bits.
- Unserved music/music-video requests try `ROUTER_URL` and then the configured fallback, using source availability checks, credentials, canonical target paths, normalized payloads and the maximum request/ACE timeout. Disabled responses preserve requested model/null values and reference error text.
- Local music now runs the ACE-Step HTTP job protocol in C++: LM request construction, optional audio-code bypass, queued/done/failed/cancelled polling, synthesis submission, batched LM results, direct audio and multipart audio, base64 transfer decoding, metadata, base64 output, and persisted downloadable audio. Model aliases report the configured available music model. The output metadata preserves the original output-format string, while the synthesis request uses its normalized format, as in the reference.
- Music calls serialize on a dedicated mutex and run outside the LLM request queue. Resource output includes music/composite music-video slots and retains the initialized external music client metadata after requests finish. Composite capacity is excluded from the machine total. LLM handoff support was added in the subsequent lifecycle pass below.
- Router aliases forward to canonical worker paths with request tokens, worker credentials and media timeouts. Successful response assets are downloaded and URLs rewritten to the router base URL. Music success counters are recorded; complete usage details and retry rotation remain open.
- Added 568 source-executed fixtures covering both API schemas, defaults/nulls, validation errors, required text/duration checks, ACE job payloads and error paths, batching, multipart/base64 audio, large seeds, output formats, model fallback and actual `EzlocalaiClient` router/fallback methods. HTTP is replaced by scripted transport only in the oracle; the C++ test replays those transcripts through real loopback sockets. SHA-256 checks cover `app.py`, `ezlocalai/MUSIC.py`, `ezlocalai/VIDEO_UTILS.py` and `Pipes.py`.
- Additional C++ regressions cover both router capabilities and aliases, canonical request paths, worker credentials, asset byte persistence and public URLs, token cleanup, authentication, POST-only behavior, generated audio downloads, retained resource metadata and LLM queue isolation.

Current limitations are substantive: ACE-Step model acquisition and packaged dependencies are incomplete. Enabled local music-video still returns 501 because scene generation, storyboards and stitching are not ported. Complete MIME/JSON edge behavior, total transport deadlines, current capability retries, per-model/output/timing usage, and real model inference still need work. OpenAPI snapshots are stale. These routes remain partial.

The current-source audit also found concrete gaps in previously inherited `implemented` router rows: the current non-LLM retry/admission policy and expanded usage/history fields were absent. Eight older rows were downgraded to partial rather than retaining an unsupported parity claim. The manifest now has 96 rows: 38 implemented, 58 partial, 0 missing. All 85 source routes have manifest coverage, but route presence is not behavior parity and strict enforcement still fails.

Verification for this pass: the C++ build and all nine CTest targets pass (38.22 seconds total). `git diff --check` passes. The reference checkout remains clean at the audited commit. First-party source scans still find no C, Python or shell files and no system()/popen() calls under `native`. Strict parity enforcement fails as required. No real ACE-Step model inference, model download or container rebuild was performed.

## ACE-Step startup and music LLM handoff

- Added a C++ ACE-Step process manager with lazy startup, external/manual/disabled bypass, health-first reuse, binary existence checks, missing-model warnings, startup polling, early-exit handling and restart of exited children. Startup failures and health timeouts do not prevent client construction, matching the reference. A health timeout leaves the child alive. The manager does not terminate ACE-Step when a NativeServer is destroyed; the Python module likewise retains its child across client lifetimes.
- Command construction preserves argument order, whitespace distinctions between URL and command settings, empty quoted arguments, literal shell characters, shlex-style escapes and malformed-quote errors. Execution uses `posix_spawnp` with an argument vector. Selected model files use the reference's trimming and deduplication rules.
- Music waits for active local LLM inference under the configured timeout. During the handoff, text/vision capacities are zero, new text requests use fallback or the reference temporary-unavailability response, and `disable_fallback` is consumed before backend inference. Music becomes unavailable while its dependent LLM slot is busy. Worker metadata advertises the music unload dependency.
- Owned llama-server processes are detached and restored with their prior context and GPU layer count. Previously unloaded models stay unloaded, embeddings are independent, and externally configured llama-server processes are never terminated. Reload-disabled requests leave models available for lazy loading. Generation failures still restore residency, and failed restores release text admission so later requests can retry.
- Added 681 cases executed against the reference configuration, command and unload/reload policy functions. The checked-in fixtures are data; the test runner and subprocess fixtures are C++. Additional integration tests cover native child processes, startup bypass/reuse/failure/timeout, lazy initialization through real HTTP, busy inference waits, concurrent handoff admission and fallback, error cleanup, retained embeddings, unchanged restoration arguments, lazy models and failed-restore retry.

Remaining lifecycle limits: native residency/pools and complete runtime placement remain partial, as do model acquisition and container packaging. Admission is rechecked after ACE-Step startup to avoid unloading inference that began while startup was waiting; exact behavior for that reference race still needs differential audit. Redirects and complete transport/timeout semantics remain part of the open HTTP parity work. No actual model inference or downloads were performed. Manifest music rows remain partial.

Verification: the C++ build and all 10 CTest targets pass (41.50 seconds total), including the 681 new reference cases and subprocess/concurrency integration checks. `git diff --check` passes. The reference checkout remains clean. First-party executable sources remain C++ only. The manifest remains at 38 implemented and 58 partial rows; strict parity enforcement still fails as intended.

## Wan native video backend

- Added C++ Wan 2.2 T2V/I2V backend selection, reference model aliases, pinned asset repositories/revisions/filenames and supported quantizations. Frame count alignment, dimension bounds and alignment, generation validation, high/low expert step distribution, fixed seed, sampling flags, CPU/CUDA selection arguments and offload flags are covered by reference-derived fixtures.
- Local `/v1/videos/generations` now invokes `sd-cli` directly and encodes its generated frames with FFmpeg. URL/base64 outputs, multiple requested items, retained video backend metadata, missing binary/no-output responses and child failures are implemented. The child owns model allocations, and generation/encoding failures clean temporary files and incomplete outputs. The container dependency list now includes FFmpeg; building and bundling `sd-cli` remains open.
- Added file-backed process execution with timeout, kill-and-wait cleanup and inherited-descriptor closure on the supported glibc runtime. Model logs and downloaded weights do not accumulate in memory. Command strings and model paths are passed as literal arguments.
- Added C++ Hub asset acquisition using the pinned snapshot/blob layout, existing snapshot reuse, offline mode, ETag and size metadata, cross-process file locking, concurrent download deduplication, temporary downloads and atomic publication. Transfers use the external `curl` binary; no Python or hf CLI runs in the product. Authentication is provided through a private temporary header file. The hf-cli skill was used only for read-only inspection of the existing cache.
- Video uses the existing LLM idle wait and unload/restore hooks, with restoration on both successful and failed generation. Music and video handoffs serialize their shared LLM state. Full auxiliary resident-model eviction/restoration is still absent.
- Added 818 cases executing `ezlocalai/WAN.py` settings, command and asset functions as the oracle. The C++ test verifies its source hash. Additional tests exercise real localhost downloads, authenticated transfers, concurrent cache calls, offline hit/miss behavior, failed-download retry, real C++ child processes, real FFmpeg encoding and FFprobe inspection, literal prompt arguments, missing frames, child failure, timeout cleanup, I2V asset selection, API output counts and video/LLM resource handoff.

Remaining gaps at this stage included I2V's FFmpeg Lanczos fitting and condition decode/error ordering, addressed in the subsequent image-processing pass below. Actual Wan weights and CUDA inference have not been run. Public image URL transport, arbitrary-size numeric and timeout exceptions require further differential coverage. Multi-GPU selection currently uses index 1 rather than the reference's capability-ranked secondary GPU; auxiliary embedding/voice/image handoff and full video residency accounting remain incomplete. The download implementation does not yet cover all huggingface_hub options, retries/resume, mutable refs, Xet transport or exact exceptions. The older forwarding defaults, router dispatch paths and LTX backend remain to be aligned. Music-video scene planning, storyboard rendering and stitching are still absent. These changes do not establish full video or deployment parity, and the manifest rows remain partial.

Verification: the C++ build and all 11 CTest targets pass (44.10 seconds total). The Wan target includes 818 source-derived cases and native process/real MP4 integration tests. `git diff --check` passes, the reference checkout is clean, and source scans find no first-party C/Python/shell files or system()/popen() calls. Strict enforcement continues to fail with 38 implemented and 58 partial manifest rows. No actual model weights were downloaded and no CUDA inference or container build was performed.

## Wan image conditioning

- Replaced FFmpeg scaling with C++ Pillow-compatible `ImageOps.fit(..., LANCZOS)` processing: fractional centered crop coordinates, float crop boundaries, normalized filter coefficients, 22-bit fixed-point rounding and intermediate channel clipping. The Pillow license is retained in `licenses/Pillow.txt`.
- Added native PNG, JPEG and WebP RGB decoding using libpng, libjpeg and libwebp/demux. Covered behavior includes dropping alpha without compositing, transparent palettes, low-bit grayscale, 16-bit grayscale clamping, 16-bit color truncation, CMYK JPEG conversion, progressive JPEG, ignoring EXIF orientation and first-frame animated images. Other formats retain the prior FFmpeg decoding fallback.
- Added default permissive base64 decoding for video inputs, including ignored ASCII punctuation, padding validation, non-ASCII rejection, and data-URL prefix handling. Both main images and condition frames are decoded before entering the model pipeline, so malformed images fail before dimension validation, Wan condition rejection or LLM unload.
- Added 979 pixel-level oracle cases generated with Pillow 11.3.0 across 64 image files, and 181 input cases executed against the reference API decoder. C++ tests compare exact RGB SHA-256 values before and after fitting and verify lossless PNG encoding. Fixture provenance records the reference source hashes and Pillow source/version.
- Extended Wan integration tests to capture the conditioning image actually consumed by the C++ `sd-cli` fixture, including the full API data-URL path, and compare its pixels with the Pillow oracle. Tests distinguish invalid condition images (500 before pipeline entry) from valid conditions rejected by Wan (400), and verify invalid inputs never unload the LLM.
- CMake and the CUDA dependency list now include the native image libraries. Product code, tests and process fixtures remain C++17; Python was used only outside the repository to execute the reference and generate data fixtures.

This closes the tested image-fitting gap, not all image-decoder parity. Other Pillow formats, unusual metadata/malformed streams, codec-version differences, decompression edge behavior, public image URL validation/redirects and all previously listed video/model/deployment gaps remain open. Compressed PNG bytes are not asserted identical; decoded conditioning pixels are. No actual model inference, weights download or container rebuild was performed.

Verification: the C++ build and all 12 CTest targets pass (52.83 seconds total), including the 1,160 new image/reference-input cases and updated Wan integration checks. `git diff --check` and CUDA Compose configuration validation pass. The reference checkout remains clean; first-party source scans find no C/Python/shell files or system()/popen() calls. Strict parity enforcement still fails as required: 38 implemented and 58 partial rows, with all 85 reference routes covered.

## Video API forwarding and validation

- Updated forwarded video defaults to the current API's 18 inference steps and 3.5 guidance. Payloads retain explicit nulls and normalized numeric types, preserve large integers, and use actual top-level fields so a prompt equal to a field name cannot shadow that field. The API continues to omit model/n from forwarding as the reference does.
- Added `ROUTER_URL` dispatch when local video is unavailable, after the image-server attempt and before fallback. Dispatch includes image and normalized condition frames (including explicit null index/strength), uses current client-key precedence, checks the router's live worker count, and caches positive/negative availability for 30 seconds. Worker count parsing accepts the reference's large integers, decimal Unicode strings and integer coercions without narrowing overflow.
- Preserved the distinct reference field sets: image-server forwarding includes the main image but omits conditions; fallback forwarding omits both. Image-server requests use 1800 seconds and router/fallback requests 600 seconds. The native HTTP implementation's complete total-deadline semantics still need work.
- Aligned status, JSON content-type and null handling. Image-server/router JSON null or empty bodies continue to the next service, while fallback null is returned as JSON null. Malformed JSON and wrong content types are rejected. Fallback health checks now also reject an incorrect content type before trusting resource data.
- Added C++ JSON syntax diagnostics for the video API, covering missing values/keys/delimiters, malformed numbers/escapes/control characters, truncated nesting, extra data and Unicode character offsets. A JSON-null request body receives the reference missing-body response. Other endpoint diagnostics and unusual/nonfinite JSON/Unicode cases remain under audit.
- Added 398 source-executed FastAPI/client/capability cases covering request schemas, defaults/nulls, large values, condition coercion, all forwarding destinations, chained failures, credentials, response types, resource pressure and malformed request bodies. The C++ runner replays captured requests/responses over real local sockets. Additional tests check repeated requests use cached positive and negative availability and create no local video slots. Source hashes cover `app.py`, `Pipes.py` and `Router.py`.

The broader video route remains partial. Capability-ranked multi-GPU selection was audited here and implemented in the subsequent runtime pass below; auxiliary model eviction/restoration remains open. Local numeric/lifecycle edges, complete HTTP transport/serialization behavior, LTX, real model inference, image-format edge cases and deployment remain open. No model weights were downloaded, no CUDA inference ran, and no container image was rebuilt.

Verification: the C++ build and all 13 CTest targets pass (54.14 seconds total), including all 398 new video API cases and positive/negative cache integration tests. `git diff --check` passes; the reference checkout remains clean. Source scans continue to find no first-party C/Python/shell files or system()/popen() calls. Strict enforcement still fails with 38 implemented and 58 partial rows, and all 85 reference routes have manifest coverage.

## Video GPU selection and restoration retry

- Video placement now uses CUDA-visible ordinals and the reference's capability-ranked secondary GPU, which can be ordinal zero. A C++ dynamic driver adapter queries device names, compute capability and total memory without a CUDA toolkit build dependency or model allocations. It honors driver visibility/reordering instead of inferring placement from `nvidia-smi` indices.
- Ported the actual `Pipes.py` generation/tier/variant scoring and stable descending ranking, including its tier substring behavior, whole-name TI detection and Quadro override. This ranking is distinct from the Router's 92-entry performance table. The placement helper covers force-CPU, secondary GPU priority and resource-manager advice; full resource admission wiring remains incomplete.
- Video unload policy follows explicit on/off settings, media residency preference, image-server configuration, presence of a secondary GPU and previous LLM residency. After generation, the initial restore preserves the saved runtime context and layers. If no LLM is restored, a second attempt uses the configured context, including when the initial reload flag is false. Matching contexts reuse prior placement. Music keeps its existing restore behavior.
- Added 2,769 source-executed cases for GPU name scores, ranked device sets, placement, unload/reload policy and restoration sequences. The 64 restoration cases replay successful, failed and throwing reloads, previously absent models, disabled initial reload and generation failure through the native video API. Fixture hashes bind the data to the current `Pipes.py`.
- Added a C++ CUDA driver fixture to exercise enumeration, visibility, reordered ordinals, availability failures and per-device query errors. API integration checks the device passed to `sd-cli` and reported in resources. Managed llama subprocess checks verify the first restore retains its runtime context and the second uses the configured context.
- Compared the native inventory and ranking against installed PyTorch 2.6.0+cu124 and the actual reference ranking function on this machine's RTX 3090 Ti, with default, hidden, ordinal-zero and invalid-prefix visibility masks. Device ordinals, names, compute capability, memory and scores matched exactly. Multiple physical GPUs were not available; multi-GPU ordering is covered by the fixture.

Auxiliary embedding/voice/image eviction and restoration, the reference's retry-time VRAM cleanup, complete persistent-model placement/pools, resource-manager admission and non-CUDA platform behavior still need work. These tests do not establish real Wan inference or model output equivalence. No model weights were downloaded, no CUDA inference ran, and no container was rebuilt. Video manifest rows remain partial.

Verification: the C++ build and all 14 CTest targets pass (66.99 seconds total), including the 2,769 video runtime cases and subprocess/API checks. `git diff --check` passes and the reference checkout remains clean. Source scans find no first-party C/Python/shell files or system()/popen() calls under `native`. Strict parity enforcement still fails as required: 38 implemented and 58 partial rows, with all 85 reference routes covered.

## Auxiliary model handoff around video

- Added C++ auxiliary lifecycle hooks in the reference's TTS, STT, embedding, image order. Video snapshots loaded models and activity, unloads idle residents after the LLM handoff, and applies the reference restoration policies after LLM restoration. Cleanup also runs after video generation fails. Media keep-loaded preferences bypass auxiliary unloading unless the explicit LLM handoff policy overrides them.
- Owned embedding llama-server processes now stop during this handoff and warm-load afterward when enabled, local and configured to remain resident. The backend rechecks active requests under its inference lock before unloading, preserving in-flight embeddings if they started after the resource snapshot. External llama-server processes are not terminated; only local residency metadata is detached.
- Restoring an already loaded embedding does not restart it. Failed warm loads leave no loaded-model metadata and later embedding requests retry. Auxiliary warm-load failures are logged without failing an otherwise completed video. This covers the existing single managed embedding engine; complete pools and transient instances remain open.
- Ported the auxiliary restoration predicates, including embedding residency/delegation, image enablement and existing-image checks, voice preload and handoff preferences, dedicated workers, media retention, and MiniCPM fast-voice model identities. Native voice and image commands currently retain no idle model engines; the shared hooks are ready for those backends, but the policy port does not implement their persistent engines.
- Added 3,057 cases executed from the actual reference auxiliary unload/restore and dependent policy methods. They cover main/pool/transient presence, activity counts, missing/default/empty/Unicode-whitespace settings, delegation, configured model identities and restore flags. Of these, 58 cases exercise the native video API with successful and failed generations, idle/busy models, forced LLM handoffs, retention bypass, and disabled/delegated restoration. Fixture hashes cover `Pipes.py`, `Globals.py` and `ModelSettings.py`.
- Real C++ managed-process tests check embedding shutdown/restart, identical configured arguments after warm restoration, active inference protection, failed startup metadata and lazy recovery. API lifecycle hooks are checked separately with backend doubles for voice/image states that lack persistent native engines.

Remaining gaps include persistent voice/image engines, complete embedding pools and transient lifetime, startup warmup, exact resource accounting and VRAM estimates, retry-time VRAM eviction, Unicode model-identity regex edges and real inference equivalence. No weights were downloaded and no model inference or container rebuild was performed. The goal and affected manifest rows remain incomplete.

Verification: the C++ build and all 14 CTest targets pass (71.26 seconds total), including all 3,057 new auxiliary cases. After adding failure-injection coverage, the updated video runtime target also passes (18.58 seconds): auxiliary warm-load exceptions preserve a completed video and do not prevent remaining restore attempts. The reference checkout remains clean, `git diff --check` passes, and first-party source scans still find no C/Python/shell files or system()/popen() calls. Strict enforcement correctly fails with 38 implemented and 58 partial rows, and all 85 reference routes covered.

## Embedding pools, transient lifetime and startup

- Added a C++ embedding pool with the reference's semaphore limit, FIFO available-instance queue, serialized resident loading and distinct transient-instance tracking. `EMBEDDING_N_PARALLEL` now controls independent full-context engines, rather than only advertised capacity. Requests exceeding the instance limit wait and reuse the next released instance. Failed acquisition releases admission for later requests.
- `EMBEDDING_KEEP_LOADED=false` loads an instance per request and destroys it after both successful and failed inference. A transient lease remains transient if the setting changes before release. Resident instances also leave the pool when keep-loaded is disabled before release. Mixed resident/transient changes and FIFO reuse match source-executed lifecycle traces.
- Owned instances use separate llama-server processes on consecutive ports, each with one full-context slot. Stable server storage keeps active text and embedding references valid as the pool grows. Text selection cannot fall through to an embedding engine. External native llama-server overrides retain operator ownership.
- The executable warm-loads enabled, local, resident embeddings before serving HTTP. Partial warm failures retain successful instances, stop the warmup loop and allow later loading to retry missing instances. Video handoff unloads an idle pool and restores its configured resident size. In-flight/loading leases protect the pool from unloading.
- Embedding resource slots count running inference, excluding requests waiting for a lease, matching the reference snapshot. Loaded-model activity aggregates across replicas. Full device/VRAM aggregation, per-instance lifecycle telemetry and the reference's complete resource manager remain open.
- Added 569 source-derived cases: 168 lifecycle scripts covering 4,167 transitions, 23 parallel-setting cases and 378 retention/preload combinations. The oracle executes the actual reference acquisition, release, warmup and destroy methods with synthetic instances; only the resulting TSV data is checked in. Hashes bind fixtures to `Pipes.py` and `Globals.py`.
- Added C++ process fixtures that serve HTTP themselves. Integration checks run two independent engines concurrently, queue a third request, reuse a released instance, preserve all contexts, unload/restore the full pool, destroy transient processes on success/error, recover from partial warmup and verify real executable startup in resident/transient/delegated/disabled modes. Separate concurrency checks verify FIFO admission and no unloading during an active lease.

Remaining embedding gaps include automatic memory estimation and placement/retry logic, full model acquisition/cache compatibility, lifecycle telemetry, cancellation and rare concurrent reconfiguration/cleanup behavior, oversized parallel values beyond native integer/port limits, schema numeric edges and engine/output equivalence with actual weights. No weights were downloaded and no model inference or container build was performed. The embedding and video routes remain partial.

Verification: all 15 CTest targets have passing results with the current changes. The full run passed 14 targets, including the new embedding runtime suite (5.86 seconds), and exposed an inherited timing assertion that matched the string prefix `0.0` even for positive sub-millisecond values. The assertion now parses the numeric duration and requires it to be positive; the rebuilt `native_http` target passed on rerun (31.29 seconds). `git diff --check` passes, the reference remains clean, and source scans find no first-party C/Python/shell files or system()/popen() calls under `native`. Strict parity enforcement still fails correctly with 38 implemented and 58 partial rows; all 85 reference routes are represented. No embedding fixture or native service processes remain after the tests.

## Embedding memory placement, startup retries and inference warmup

- Added native GGUF metadata reading with the reference architecture/fallback order, default fields, truthiness and partial-conversion behavior. The reader handles GGUF v2/v3 in both byte orders, scalar/array metadata, alignment, tensor type/row/bounds checks and duplicate keys/names. It reads descriptors without mapping model weights. The differential oracle uses `gguf` 0.19.0; the reference requirement itself is unpinned, so other package versions remain an explicit compatibility concern.
- Ported weight, KV and compute-buffer estimates, cache-byte aliases, safety margins, full/partial/CPU offload decisions, per-layer GPU estimates and the exact deduplicated 75%/50%/25%/CPU retry sequence. Explicit GPU layers bypass automatic selection when CUDA is available. The engine retries only reference-classified memory errors, and stops after CPU or non-memory failure.
- Added a CUDA Driver API free-memory probe using dynamically loaded primary-context and memory-query functions. Invalid visible ordinals fall back to zero, failures return zero free memory, and the calling thread's current context is restored after the query. Primary contexts are retained until process shutdown. Tests cover visibility reordering, absent CUDA, driver failures and simultaneous probes from threads with existing contexts. On the host's single GPU, both the C++ probe and a separate PyTorch query returned 23.268310546875 GiB free; this validates that host observation, not all driver/device combinations.
- Managed local GGUF engines now use those estimates before startup, disable automatic fitting and RAM/prompt caching, pass embedding normalization, and disable both KV and operation offload on CPU. GPU residents run the reference text input and a full microbatch of zero token IDs before becoming available in the pool. CPU/transient/disabled warmup paths skip inference. Warmup validates configured dimensions and discards failed instances without retrying smaller layer counts. Startup timeouts now terminate and reap the owned process.
- Loaded embedding metadata reports CPU/CUDA placement and sums estimated GPU memory across loaded instances. Full resource-manager admission, per-instance telemetry and remaining metadata/ordering details are still open.
- Added 2,731 source-executed cases: 153 synthetic GGUF files with hash checks, 420 memory estimates, 1,350 placement decisions, 315 fallback/memory cases, 13 error classifications and 480 warmup combinations. Additional C++ process tests cover full/partial/explicit/CPU placement, failed probes, retry sequences, non-memory errors, warmup and dimension failures, timeout cleanup, disabled warmup and transient lifetime. No Python implementation or test runner is checked in.

Remaining embedding gaps include native acquisition before estimation: unresolved `-hf` models still delegate download to llama-server and use the configured layer count (all layers for automatic mode on CUDA), with default metadata for retries and no weight-derived resource estimate. Full Hub/cache semantics, resource-manager placement/admission, rare concurrent reconfiguration/cleanup, cancellation, oversized integer/metadata bounds, complete malformed-GGUF/version behavior, engine flags across supported versions and real model output equivalence remain incomplete. No model weights were downloaded, no model inference was run and no container was rebuilt. The manifest and goal remain incomplete.

Verification: the C++ build succeeds and all 16 CTest targets have passing results. The full run passed 15 targets, including the 2,731 placement cases (7.65 seconds); an older HTTP argument fixture failed because it referenced a GGUF file it never created. It now creates the placeholder required by the file-size estimator, and the rebuilt HTTP suite passes (31.53 seconds). `git diff --check` and first-party C++ source scans pass. The reference checkout remains clean. Strict enforcement continues to fail with 38 implemented and 58 partial rows, with all 85 source routes represented. No fixture/native service processes remain after verification.

## Native GGUF acquisition before embedding placement

- Added a shared C++ port of `LLM.download_model`: short-name expansion to `unsloth/<name>-GGUF`, model-specific directories, first comma-separated quant selection, case-sensitive GGUF/quant matching, unsorted local/repository order, local projector discovery, the eight projector candidates, requested/Q4_K/Q5_K/Q6_K/Q8 preference, and the final missing-model error. Embeddings use its skip-projector path. The helper supports projector acquisition for later text integration; this pass does not replace text-model acquisition.
- Replaced embedding's delegated `llama-server -hf` download with native acquisition before GGUF metadata/file-size estimation. Default discovery uses the reference directory and case rules. Explicit native local-path/MODELS_PATH overrides remain. New pool instances and unloaded/restored instances repeat model selection, so an earlier fallback quant does not become permanently pinned by a reused process descriptor. Matching top-level local files remain usable without Hub access.
- Added C++ Hub tree listing with pagination and Unicode/escaped filename decoding, plus local-folder downloads using `curl`. The downloader follows relative metadata redirects, uses the resolved download location, omits the Hub credential on a different CDN origin, tracks commit/ETag/timestamps under `.cache/huggingface/download`, reuses valid local content, hashes preplaced SHA-256 files, copies existing global-cache snapshots and locks local downloads. Temporary downloads are removed on failure. Missing-entry/revision errors remain errors even if an old file exists; other metadata-access failures can reuse an existing local file. An outdated target is removed before a replacement network download, matching the reference's failure behavior.
- Added 1,028 cases executing the actual `LLM.download_model` function with controlled filesystem/Hub callbacks. C++ checks both results and call order, including failed projectors, failed lists/downloads, directory errors, case/quant mismatches, nested filenames and comma/whitespace behavior. Another 16 traces execute `huggingface_hub` 1.28.0 against local HTTP servers, covering pagination, escaped/Unicode paths, new/unchanged/updated files, commit fast paths, local hashes, global-cache copies, corrupt/stale metadata, denied/missing HEAD responses, download failure, size mismatch and relative/CDN redirects. Oracle scripts remain outside the repository; checked-in runners and HTTP fixtures are C++.
- An integration test acquires a synthetic GGUF through authenticated local HTTP, reads its actual metadata/size, chooses partial GPU offload through the CUDA fixture, launches the C++ HTTP engine fixture, warms inference and verifies cleanup. It also verifies acquisition failure prevents launch, nested cached files are revalidated without fetching weights again, restored instances reacquire, and offline mismatched quantization does not silently select another cached file.

Remaining acquisition work includes shared text/vision startup integration, source-version coverage beyond the installed Hub oracle, precise retry/backoff/timeout/error details, all authentication modes (including OAuth/OIDC), Xet transport/tree caching, all mutable global-cache operations, Unicode/path extremes and concurrent external filesystem changes. The public embedding/resource-manager, native engine/output, other modality and deployment requirements remain incomplete. Only synthetic GGUF data was transferred during these tests; no real model weights, model inference or container build was used as evidence.

Verification: the C++ build and all 17 CTest targets pass in one full run (85.51 seconds). The new model-download suite passes all 1,028 policy cases, 16 transport traces and managed acquisition/placement/restoration checks (2.08 seconds). `git diff --check` passes; source scans find no first-party C/Python/shell files or system()/popen() calls under `native`. The reference checkout remains clean. Strict enforcement still fails correctly with 38 implemented and 58 partial rows and all 85 source routes represented. No fixture/native service processes remain after verification.

## Shared resource policy, CUDA telemetry and embedding registration

- Added a C++ ResourceManager with all eight reference model types and default VRAM estimates. Registration preserves insertion order and active-request counts, uses case-sensitive CUDA device matching for memory estimates, and updates the usage timestamp. Activity is reference-counted; completion never advances the last-use timestamp. Missing-model activity and repeated unregister operations are harmless.
- Ported the aggregate free-memory gate and 1 GiB safety margin, single-GPU behavior, capability-ranked secondary preference for auxiliary models, most-free GPU selection, tensor-split recommendation, idle-model estimates, CPU/image/video offload choices and fallback reasons. Unload candidates exclude the primary LLM, active models and nonpositive estimates, then use stable least-recently-used order. Negative, NaN and infinite requirements/estimates are included in the policy fixtures.
- Added strict CUDA memory queries alongside the embedding estimator's existing forgiving probe. Resource admission propagates probe failures; allocator diagnostics return per-device errors. Resource capacity is captured at construction with each device's GiB floored, while free memory remains live. The resource API and local fallback decisions now use process-visible CUDA ordinals. `torch_cuda` includes rounded device free/total memory and zero PyTorch allocator counters for this C++ process; this does not establish allocator equivalence for modalities that still need a native replacement.
- Embedding loading now requests a resource recommendation before acquisition. Specific CUDA ordinals override the configured main GPU; generic CUDA/CPU recommendations preserve the embedding estimator's decision. The shared tracker registers resident instances before transient instances, reports the first model's repository name, sums estimates and preserves activity when the pool changes. Pool observers publish removals before process destruction and clear aggregate metadata before a full unload. The API and backend share the same tracker.
- Added 4,449 source-executed ResourceManager fixtures covering 48,655 operations, including repeated registration, nested activity, clock changes, deletion/reinsertion, changing memory readings, idle admission and exclusions. Results and probe call order are checked against the unchanged reference source hash. Only TSV data and C++ runners are checked in; the Python oracle is external to the repository.
- Added concurrent reference-count checks, pool publication/destruction ordering, native CUDA-driver and resource-API checks, rounded memory/error reporting, full-precision fallback threshold display and disabled-fallback probe avoidance. Embedding process coverage now has 15 scenarios, including secondary-GPU selection, a most-free recommendation overriding the environment, and partial GPU offload despite a coarse CPU recommendation. Existing acquisition, pool and API suites also pass with the shared tracker.

Remaining resource work includes registering the actual LLM/vision/voice/image/video/music lifetimes with the tracker, complete `model_lifecycle` and `model_pools` responses, worker-registration telemetry (which still uses NVIDIA-SMI), full modality allocator behavior, cleanup/cancellation races and real-model/device equivalence. The fallback-status row is now partial: the audit found malformed/empty local threshold handling and availability/local-memory evaluation order still differ from the current reference. Normal fallback memory selection now uses the shared tracker, but those remaining differences must not be hidden by the earlier implemented label. No real model inference or container build was performed.

Verification: the rebuilt C++ project passes all 18 CTest targets in one full run (30.32 seconds), including the 4,449 new resource cases (1.64 seconds). `git diff --check` and first-party source scans pass; both CMake projects enable only CXX and the reference checkout remains clean. The strict gate remains intentionally failing: 37 implemented and 59 partial rows, with all 85 reference routes represented. The goal is still active and incomplete.

## Fallback evaluation, availability cache and JSON behavior

- Consolidated fallback status and forwarding onto one availability implementation. Status now checks the remote server before reading local memory and evaluating the local threshold, then fetches models only when availability succeeded. Failed local probes or threshold parsing leave the already-recorded availability result cached. Positive/negative cache results, the exact 30-second boundary and wall-clock rollback follow the source.
- Local and remote thresholds distinguish missing settings from explicit empty strings. Native parsing now accepts Python float syntax, including Unicode decimal digits/whitespace, underscores, NaN and infinities, while rejecting malformed suffixes and ASCII separators that Python float rejects. Low-memory reasons retain full threshold precision. Invalid local settings/probes produce the reference's plain-text 500 response. Resource JSON rejects non-finite threshold values rather than emitting invalid JSON.
- Remote availability requires the response's JSON MIME type and a valid object. Duplicate fields use their final value; missing memory is zero; booleans remain numeric; strings, nulls and containers are unavailable. The comparison now follows the reference's less-than branch, including NaN behavior. A missing HTTP Content-Type is treated as octet-stream rather than silently assumed JSON.
- Ported the installed aiohttp JSON media-type prefix behavior and common charset decoding through native iconv, including UTF-8, Latin-1, UTF-16, UTF-32, UTF-8 BOM handling, invalid byte sequences and unknown charset fallback. Model listing validates the full response and preserves the source's empty-list error behavior. Returned model IDs retain arbitrary JSON types, duplicate-key order, Python float formatting and Unicode. Non-finite selected IDs and lone surrogate strings fail response serialization; values overwritten by later duplicate fields are discarded first.
- Fixed the shared native JSON string decoder to decode `\u` escapes and combine surrogate pairs. The JSON emitter now escapes all control characters instead of replacing uncommon controls with spaces. Fallback decoding also enforces the reference runtime's default 4,300-digit integer parsing limit across the whole response, including unused fields.
- Added 936 cases executing the unchanged `EzlocalaiClient`, `should_use_ezlocalai_fallback`, `Globals.getenv` and actual FastAPI status handler. The oracle uses aiohttp 3.13.2's real response JSON decoder with controlled transport, clock and memory probes. The C++ runner serves real local HTTP responses and checks response status/type/bytes plus the exact HTTP/clock/VRAM/RAM event sequence. Separate API checks cover resource threshold serialization errors. Checked-in tests and implementation remain C++; the source-execution script stays outside the repository.

The malformed-local-threshold and evaluation-order differences identified in the preceding pass are fixed. At this point, the fallback route stayed partial because source client URL/key settings were captured in a lazy singleton, while native consumers still reread them; the following pass addresses that lifetime difference. Whole-request timeout/deadline behavior, redirects/compression and full transport/codec compatibility also remain incomplete. Exact comparisons involving very large JSON integers and nondefault Python integer-limit settings still require work. Other resource/modality, lifecycle, model-output and deployment gaps remain unchanged. No real model inference or container build was performed.

Verification: the rebuilt C++ project passes all 19 CTest targets in one run (30.37 seconds), including the new fallback target (1.35 seconds). The shared JSON changes also pass all existing embedding, music, video, router, configuration, native HTTP and resource tests. `git diff --check` and first-party C++ source scans pass, and the reference checkout remains clean. Strict parity enforcement still correctly fails with 37 implemented and 59 partial rows; all 85 source routes are covered by the manifest. The full goal remains active and incomplete.

## Lazy fallback client settings and shared lifetime

- Native fallback consumers now share settings captured once, on the first client use. Capture uses the raw `FALLBACK_SERVER` with only trailing slashes removed, and the raw `FALLBACK_API_KEY` or `EZLOCALAI_API_KEY` fallback. Empty and literal `none` credentials omit authorization; other values retain their source spelling and whitespace. URL strings such as `none` and `true` remain configured invalid destinations, matching this client rather than the separate voice-server configuration rules.
- Unconfigured clients are captured too. Environment changes before first use are visible; changing URL or credentials afterward preserves the original destination, credentials and availability cache. Creating a fresh application captures the new settings. Memory and remote-VRAM thresholds remain live, and ResourceManager's separate resource-admission environment read is unchanged.
- All fallback status, resource-policy and forwarding consumers use the shared client. Queued text requests initialize it even when the request disables fallback, matching `_enqueue_with_fallback`. Initialization uses `std::call_once`; the availability cache retains its existing synchronized 30-second behavior.
- Added 134 scenarios executing the actual reference singleton getter, client, policy and FastAPI handler, with 382 status queries. C++ tests compare response bytes and HTTP destination, raw authorization, memory-probe and clock events across lazy initialization, URL/key replacement, disabled clients, cache expiry, application recreation and live threshold changes. The checked-in artifact is TSV data; the reference oracle remains outside the repository.
- Added checks for capture through 13 API entry points, including a health request that leaves the client uninitialized, resource/status requests, disabled-fallback chat/completion requests, embeddings, speech/transcription, image generation/editing, video and music/music-video. Eight simultaneous first-use requests and a subsequent configuration change verify shared initialization and retained credentials over real local HTTP.
- Updated existing HTTP, resource and music-handoff tests to configure separate application instances before first use. The music handoff still checks both fallback forwarding and per-request disabling with one captured client.

The source capture point is lazy `get_fallback_client()`, not construction or unconditional startup. This corrects the previous progress note's startup wording. Client lifetime is now covered, but whole-request deadlines, redirects/compression, complete transport/codec semantics, exact large-integer comparisons and configurable Python integer limits remain open. The fallback route stays partial. Full model/resource lifecycle, modality internals, real inference equivalence and deployment parity also remain incomplete. No real model inference or container build was performed.

Verification: the rebuilt C++ project passes all 19 CTest targets in one run (31.42 seconds), including the expanded fallback target (1.26 seconds) and the corrected music handoff tests. `git diff --check` and first-party source scans pass; both CMake projects enable only CXX, the reference checkout is clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`, and no fixture processes remain. Strict parity enforcement still fails as required: 96 rows, 37 implemented, 59 partial, with all 85 reference routes covered. The full goal remains active and incomplete.

## Total fallback HTTP deadlines and response completion

- Added a C++ monotonic HTTP deadline shared by name resolution, all connection attempts, TLS handshaking, upload, response headers and body reads. Nonblocking socket/TLS operations wait only for the remaining budget; receiving more bytes never resets it. Timed-out DNS requests return without waiting for the OS resolver, and late resolver results retain safe ownership. Numeric IP addresses bypass asynchronous resolution.
- Buffered fallback methods now apply their source timeout values: availability 5 seconds, model listing 10, embeddings 60, speech/transcription 120, images 660 and videos 600. Text forwarding parses live `REQUEST_TIMEOUT` with a 300-second absent default. Music and music-video use the source-ordered maximum of live `REQUEST_TIMEOUT` and `ACE_STEP_TIMEOUT`, both defaulting to 1800 seconds. Router music forwarding uses the same reference client policy.
- Fractional timeouts are preserved. Python float syntax includes Unicode digits/whitespace and separators. Malformed or explicitly empty settings fail forwarding; nonpositive values and NaN disable the total timer; positive infinity fails before the outgoing request. Totals of at least five seconds round the absolute monotonic deadline upward, matching aiohttp 3.13.2's `TimeoutHandle`.
- Health and model probes return immediately on non-200 status headers instead of waiting for an error body. Successful responses must supply their complete declared length. Chunked bodies now require the terminating empty line after the zero chunk and optional trailers; a valid JSON prefix, incomplete zero chunk or stalled trailer no longer counts as a complete response. Requests finish when their framed payload is complete, without waiting for connection closure.
- A subprocess disconnect test exposed SIGPIPE termination during TLS handshaking. Deadline-controlled TLS calls now block and consume their own newly generated SIGPIPE on the calling thread, preserving the process-wide signal handler and any prior signal mask/pending signal. Plain socket writes use `MSG_NOSIGNAL`.
- Added 176 fixtures executed against the installed aiohttp timer implementation and 68 cases executing the unchanged reference `EzlocalaiClient` over actual loopback HTTP. The latter cover music/music-video timeout selection, Unicode/empty/NaN/infinity settings, delayed and dribbled headers/bodies, length/chunked/EOF framing, truncated responses, complete/stalled trailers, keep-alive completion, the real five-second health deadline and early non-200 returns. Native tests compare outcomes and outgoing attempt counts. Separate socket, stalled-upload, cumulative-phase, TLS, disconnect-subprocess and delayed-DNS ABI checks verify cancellation and lifetime behavior. All checked-in runners and implementation remain C++; oracle scripts are external and fixtures are data.

This closes the buffered fallback per-operation timeout mismatch. Full HTTP equivalence remains incomplete: redirects, decompression, all URL/IPv6/framing/codec rules, resolver executor/concurrency behavior and separate streaming paths still require work. Very large JSON integer comparisons and configurable Python limits also remain open. Other modality engines, resource/model lifetimes, real inference and deployment parity remain incomplete, so the fallback manifest row and full goal stay partial. No model inference or container rebuild was performed.

Verification: the final rebuild passes all 20 CTest targets in one run (31.53 seconds), including `http_deadline_reference` with all 176 timer and 68 real-HTTP source cases (12.29 seconds). Existing TLS verification, native HTTP, fallback/cache/lifetime, music, router, image/video, embedding and resource suites pass after the shared chunk-framing change. `git diff --check` and first-party source scans pass; the reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`, and no fixture processes remain. Strict parity enforcement still exits 1 as required: 37 implemented, 59 partial and all 85 source routes represented. The full goal remains active and incomplete.

## Incremental fallback response decompression

- Added a C++ HTTP body decoder that processes payload bytes incrementally for declared-length, chunked and EOF-delimited responses. Buffered fallback clients now advertise the reference's `Accept: */*` and `Accept-Encoding: gzip, deflate, br` headers and decode gzip, deflate and Brotli before JSON or audio handling. Chunk payloads reach the content decoder as they arrive, so corrupt compressed data fails without waiting for the rest of a large or stalled chunk.
- Matched aiohttp's raw-deflate compatibility check on the first input byte, strict EOF requirement for nonempty deflate, permissive truncated gzip/Brotli behavior, ignored extra gzip members and unused zlib data, and Brotli's rejection of trailing data. Empty encoded bodies preserve the source behavior. Case-insensitive encoding recognition uses the last duplicate header and retains trailing whitespace, matching the installed C HTTP parser. Unknown or combined encoding names pass through unchanged.
- The reference runtime used here is aiohttp 3.13.2 with zlib 1.3 and Brotli 1.1.0, without Zstandard support. Native recognized `zstd` responses therefore fail during header parsing, even for empty bodies or failed-status probes. This is tied to the tested runtime; alternative dependency sets and parser implementations remain an explicit compatibility gap.
- Preserved the source's failed-probe distinction: a content feed error in an unread body does not replace the remote status, while a decoder EOF error encountered with already-received headers/body fails response parsing. Probes still return without waiting for additional error-body bytes. The total HTTP deadline remains active until framing is complete, even after a compressed stream has already ended and yielded a complete JSON object.
- Compression API cases exposed an existing music-forwarding difference for empty JSON responses. Music and music-video forwarding now use the shared aiohttp-compatible MIME/charset/JSON decoder, preserving `null` for an empty decoded response. Existing music API fixtures pass with this change.
- Added 1,407 cases executed against aiohttp's actual `DeflateBuffer`, checking failure outcomes and exact successful output lengths/hashes across empty, binary, Unicode, repetitive and random data; input splits; truncation; corruption; multiple streams; unknown encodings and large output buffers. Added 592 cases executing unchanged reference fallback client methods over actual loopback HTTP, then replaying those responses through native status/model/music/audio routes. Checks cover availability reasons, model IDs, forwarded JSON and binary audio, rejection of corrupt remote responses, content-encoding/header variants, real framing, early errors and deadlines after compressed-stream completion. These tests establish the forwarding behavior, not local model inference equivalence.
- Added explicit zlib/Brotli CMake dependencies and CUDA image build packages, and updated native build documentation. The implementation and test runners are C++17; checked-in fixtures are data and the Python reference oracles remain outside the repository.

Buffered fallback decompression is now covered for the tested runtime. Remaining HTTP work includes redirects, separate streaming paths, other dependency/parser versions, complete URL/IPv6/header/framing/charset behavior and DNS executor/concurrency equivalence. Very large JSON integer comparisons and configurable Python integer limits remain open. Full model/modality lifetimes, native inference engines and output equivalence, and deployment parity are still incomplete. The fallback row remains partial; no real model inference or container image rebuild was performed.

Verification: the final C++ rebuild passes all 21 CTest targets in one run (31.27 seconds), including all 1,407 decoder and 592 real-HTTP compression cases (1.76 seconds), the 176/68 deadline cases and existing fallback lifetime/cache/API suites. `git diff --check`, first-party source scans and CUDA Compose configuration validation pass. The linked zlib/Brotli versions match the reference codec versions, and the source checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Strict parity enforcement still exits 1: 37 implemented, 59 partial, with all 85 source routes represented. The full goal remains active and incomplete.

## Buffered fallback redirects and URL handling

- Added a C++ URL parser/resolver for buffered fallback requests, with Yarl-compatible scheme/authority splitting, relative and protocol-relative locations, dot-segment normalization, percent quoting, query/fragment inheritance, Unicode port syntax, default-port normalization and trailing-dot Host formatting. IPv6 addresses, mapped addresses and scope identifiers are handled separately from DNS names. ICU supplies Unicode normalization and IDNA conversion, including the source's legacy IDNA fallback; CMake, the CUDA image packages and native build documentation include the dependency.
- Buffered clients now follow 301/302/303/307/308, change POST to GET for 301/302 and non-HEAD requests to GET for 303, retain replayable payloads for 307/308, and preserve explicit JSON Content-Type while removing generated multipart Content-Type when a redirect drops the payload. Location/URI precedence, first-duplicate selection, empty/missing locations, unsupported schemes and the ten-redirect limit follow the tested aiohttp client.
- A single monotonic deadline covers DNS, connection, TLS, upload and response processing across the entire redirect chain. Redirect responses release their connection without waiting for an unread body; already-received compression errors retain the existing source-compatible behavior. Internal upstream header metadata is cleared before returning an API response.
- Bearer and Basic authorization are removed on an origin change, including scheme, canonical hostname or effective port, and do not reappear on a later return to the original origin. URL credentials follow source percent decoding, Latin-1 Basic encoding, username-colon validation and conflicts with an explicit Authorization header. Invalid UTF-8 percent escapes remain literal credentials. Raw malformed UTF-8 in a Location header is tested through both Yarl representation and aiohttp wire serialization: the URL requoter may retain surrogateescaped text when unchanged, while the request writer drops those bytes.
- Added 1,102 cases executed against Yarl 1.22.0 and aiohttp 3.13.2 URL/auth helpers. Added 279 actual loopback HTTP cases executing unchanged reference fallback methods, with matching native API response and request-trace assertions across health, models, music, speech and multipart transcription. Traces check method, target, authorization, content type, cookie absence for IP hosts and body preservation. Cases include IPv6, cross-origin and return redirects, Basic credentials, relative/query/fragment targets, malformed bytes, redirect limits, unread/corrupt bodies and cumulative deadlines. The implementation and test runners are C++17; checked-in fixtures are data and the reference Python oracles remain outside the repository.

This closes the tested buffered redirect and URL behavior, not all aiohttp transport parity. Redirect cookie storage/filtering, separate streaming clients, Happy Eyeballs and DNS concurrency, complete HTTP parsing, dependency/Unicode-version alternatives and untested URL/codec extremes remain open. The fallback manifest row remains partial. Local model/modality engines and lifetimes, actual inference/output equivalence and deployment parity remain incomplete. No real model inference or container image rebuild was performed in this increment.

Verification: the final rebuild passes all 22 CTest targets in one run (31.27 seconds), including the 1,102 URL cases and 279 real-HTTP redirect cases, plus the existing compression, timeout, fallback lifecycle/cache and native API suites. `git diff --check`, the first-party C++ source scan and CUDA Compose configuration validation pass. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Strict parity enforcement still exits 1: 37 implemented, 59 partial, all 85 source routes represented. The full goal remains active and incomplete.

## Buffered fallback cookie handling

- Added a C++ cookie jar scoped to each buffered fallback client method call. Redirect hops share that jar; availability probes, model fetches and later forwarding calls create separate jars, matching the reference's per-method `ClientSession` lifetime. Cookie processing runs when headers arrive, before waiting for the remaining response body, and errors become the existing fallback transport-failure outcome.
- Matched the tested aiohttp 3.13.2 parser's extended cookie names, missing-semicolon behavior, unmatched quoted values, duplicate headers/names, attribute handling, and exact preservation of quoted wire values. Unquoted values use Python-compatible cookie quoting and outgoing header ordering/whitespace. Cookie parsing rejects the control characters rejected by this installed Python 3.12.3 build, including characters decoded from octal escapes. The runtime's normal `Partitioned` attribute is ignored, while the dollar-prefixed form can raise because its `Morsel` lacks that attribute; this version-specific behavior is covered.
- Matched default rejection of cookies from IP-like hosts, case-sensitive domain checks, leading/trailing domain dots, host-only flags, decoded URL paths and default path derivation, path boundaries, Secure filtering, and duplicate-name precedence across matching domain/path scopes. Cross-origin authorization removal remains independent: cookies can follow a port change on the same hostname, are filtered out on an IP-host hop, and can return when a later redirect reaches their matching DNS host.
- Matched Max-Age parsing, expiration clamping, invalid/overflowing integer handling, Expires tokenization and calendar normalization, two-digit years, the special zero timestamp behavior, boundary expiry and replacement retaining an existing expiration. Host-only flag removal when an expiring sibling path is deleted follows the source. Expiry uses wall-clock time; an injectable clock makes state transitions deterministic in the C++ tests.
- Added 1,159 scenarios executing the actual reference cookie parser/jar/request-header methods, covering 6,180 update/filter operations. Added 217 loopback HTTP cases executing unchanged fallback client methods and comparing native health, models, music, speech and transcription API outcomes and outbound request traces. These include DNS/IP/IPv6 hosts, cross-port and host-return chains, duplicate path/name cookies, malformed/Unicode values, final/error response cookie failures, real expiry between hops and failure before a stalled body. Native probe fixtures also set cookies and check that they do not leak into another client method call. Existing URL and redirect cases still exercise the shared decoder and transport.
- The implementation and tests are C++17; fixture data is checked in and Python oracles remain outside the repository. Added the aiohttp attribution/license for the adapted cookie grammar and behavior. No additional runtime dependency was introduced.

The tested buffered cookie gap is now covered. Separate streaming paths, complete HTTP parsing and transport behavior, Happy Eyeballs/DNS concurrency, dependency/runtime variants, remaining numeric/codec extremes, local model/modality implementations and lifetime/output equivalence remain incomplete. The fallback row stays partial. No real model inference or container image rebuild was performed in this increment.

Verification: all 23 CTest targets pass in the final run (31.13 seconds), including all 1,159 cookie scenarios/6,180 operations, 217 new real-HTTP cookie cases, and the prior 1,102 URL/279 redirect cases. Compression, deadline, fallback lifecycle/cache and native API suites also pass. `git diff --check`, first-party C++ source scanning and CUDA Compose configuration validation pass. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Strict parity enforcement still exits 1: 37 implemented, 59 partial, all 85 source routes represented. The full goal remains active and incomplete.

## Text fallback response lifecycle and queue results

- Audited the actual fallback client and API functions together. For a truthy raw `stream` value, the pinned Python client returns an async generator after leaving its response/session contexts. Iterating that generator fails because the connection is closed. The chat API instead attempts synchronous iteration and emits one `streaming_error` SSE event without `[DONE]`; the completion API attempts to subscript the generator and returns HTTP 500. Truthy strings such as `"false"` that validate as false at the API also produce HTTP 500 during response serialization. Native queue-timeout and shared-GPU-handoff forwarding now preserve these observed errors. This is reference error-behavior parity, not evidence of working fallback text streaming.
- Successful streamed fallback responses release the upstream connection after headers without waiting for body bytes. Redirects, cookies, compression-header checks, already-received decoder/EOF errors and the total deadline still apply. Non-200 responses retain their body-reading behavior. Non-stream text fallback now uses the shared MIME/charset/JSON decoder and API serialization behavior, including non-object values and serialization errors after a successful client response.
- Added scoped, typed forwarding results alongside queued requests so the queue preserves the complete HTTP status, media type and body. Successful remote JSON containing an `error` field no longer becomes a local backend failure. Registrations are removed on every request exit, allowing late results after timeout/cancellation to be discarded without embedding internal markers in model-generated text. Local SSE media types now include the reference's UTF-8 charset.
- Added 178 fixtures executing the unchanged reference client methods and API endpoint bodies against actual loopback HTTP, with controlled queue/local-model substitutes. The C++ runner exercises native queue-timeout forwarding and compares status, media type, body, outbound request traces and local recovery. Cases cover raw/validated stream flags, early connection release, compressed/malformed/stalled bodies, redirects/cookies, shared deadlines, MIME/charset/JSON handling and serialization failures. The controlled reference queue does not establish full queue timing equivalence. Existing shared GPU handoff tests now also verify chat/completion stream failures and successful error-valued JSON crossing the typed queue path.
- All new implementation and test runners are C++17. Reference oracle scripts remain outside the repository; checked-in fixtures are data. No real model inference or container image rebuild was performed.

Normal pre-inference LLM fallback policy is still missing: the reference first checks resident LLM/vision models, memory thresholds and resource admission before deciding whether to try the remote server. Native forwarding currently covers queue wait and shared GPU handoff. Complete queue exception propagation, scheduling/cancellation, all model registrations/lifetimes, voice/router streaming transport, complete HTTP compatibility, native modality engines and real-output/deployment equivalence remain incomplete. Both text API rows remain partial.

Verification: the final C++ build passes all 24 CTest targets in one run (32.54 seconds), including the 178 text fallback cases (20.15 seconds) and expanded handoff checks (4.45 seconds). `git diff --check`, first-party C++ source scanning and CUDA Compose configuration validation pass. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Strict parity enforcement still exits 1: 37 implemented, 59 partial and all 85 source routes represented. The full goal remains active and incomplete.

## Pre-inference text fallback policy and local stream coercion

- Ported the missing normal `Pipes.get_response` fallback precheck into the C++ queue processor. Resident text or vision models bypass fallback first. Otherwise the queue checks combined free VRAM/RAM, then LLM resource admission, and tries the configured fallback before reserving a local inference slot. Resource admission retains its separate live memory reading, 8 GiB LLM estimate, 1 GiB margin and idle-model accounting. Availability or forwarding failure resumes local inference; successful forwarding preserves the typed HTTP response path added previously.
- Preserved per-request disabling and removal of `disable_fallback` before both local and remote inference. Generic precheck exceptions, including malformed memory thresholds and the tested memory-probe failures, now reach the API as HTTP 500 with `{"detail":"Internal server error"}`. The shared GPU handoff branch still precedes the normal precheck. Existing queue-wait fallback remains independent of resident-model preference.
- The precheck consults the shared resource tracker and the backend's existing text/vision residency snapshot. This makes the current managed engine's loaded state effective without claiming that complete text/vision resource registration, allocator accounting and lifecycle telemetry have been ported. Source resource diagnostics also count processing queue requests in aggregate text-slot usage; the existing native behavior is retained.
- Added 816 cases executing unchanged source ResourceManager, fallback policy, `Pipes.get_response` and API functions. Remote client results and the local allocation boundary are controlled substitutes; the native tests use the real native queue and loopback HTTP. They compare status, media type, body and memory/admission/remote/local event order across resident text/vision models, disabled/configured clients, current versus captured fallback configuration, CPU/single/multiple GPUs, threshold boundaries and malformed/non-finite/Unicode values, changing free-memory readings, idle/active models, failed probes, unavailable remotes, failed forwarding and stream flags. This establishes the tested decision path, not full source queue timing or model execution equivalence.
- Local recovery cases exposed another API mismatch: a truthy raw `stream` string such as `"false"` creates a source local generator, while Pydantic validates the flag as false and FastAPI serializes that generator as a JSON array. Native now serializes the buffered local SSE values into that array and sends boolean `true` to the managed engine for raw truthy stream flags, matching `LLM._chat_stream`'s request normalization. A LlamaCppInferenceBackend HTTP integration check loads a model locally, verifies that its residency bypasses invalid fallback memory settings, and checks all six false-string variants against two yielded chunks. Full local streaming, cancellation and error behavior remain incomplete.
- All implementation and checked-in test runners remain C++17. Reference oracle scripts are external; only their fixture data is checked in. No real model inference or container image rebuild was used as verification.

The normal pre-inference routing gap is now covered for these states. Complete text/vision and other-modality resource registration/lifetimes, shared-handoff exception propagation, queue scheduling/cancellation and capacity-error mapping, voice/router streaming transport, complete local inference semantics and real output/deployment equivalence remain open. Both text API rows and the overall goal remain incomplete.

Verification: the final C++ build passes all 24 CTest targets in one full run (33.27 seconds). The expanded LLM fallback suite passes its 178 existing cases, 816 new precheck cases and managed-backend checks (22.04 seconds); native HTTP, shared GPU handoff, fallback/cache and resource suites also pass. `git diff --check`, first-party source scanning and CUDA Compose configuration validation pass. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`, and no fixture service processes remain. Strict parity enforcement still exits 1: 37 implemented, 59 partial, all 85 source routes represented. The full goal remains active and incomplete.

## Text/vision readiness and lifecycle history

- Corrected owned text/vision engine residency to begin after successful startup, before the first inference response. An ordinary inference error no longer hides an already loaded engine. Concurrent requests can therefore use the resident-model fallback bypass while its first request is pending. Startup failure still leaves the engine absent, shared GPU handoff clears residency, and restoration publishes only successful loads. External server URLs keep their separate behavior: a successful response establishes observed residency; an unverified URL or failed response does not establish a local model load.
- Added a C++ lifecycle recorder matching `_record_model_lifecycle` and its summary projection: wall-clock event timestamps, negative/NaN duration clamping, direct binary-float rounding to four decimal places, raw-duration accumulation before averaging, insertion-preserving summaries, count/failure/min/max/last values and a 100-event deque. The source's colon-joined summary-key collision behavior is preserved. Recording and snapshots are synchronized; concurrent snapshots do not mutate totals.
- Owned text/vision load attempts now record one success or failure event around the complete startup attempt, including internal startup retries. Reusing a resident engine creates no new load event. Shared GPU unload records the same measured batch duration for each detached model, in first-load dictionary order; an empty handoff records the source's `LLM` placeholder. Restoration records failures and subsequent recovery. Ordinary inference failures and context teardown do not invent unload events.
- Added `model_lifecycle` to `/v1/resources`, with configured/loaded/active model names, supported runtime settings, recent events and summaries. Model list order survives unload/reload, model context is captured with configuration, and successful restoration restores the prior active model. Owned engine runtime details currently include context, GPU layers, main GPU, explicit tensor split, batch/microbatch sizes and the available KV-cache setting. No local load event is fabricated for an external server's successful response.
- Added 818 cases executing the unchanged reference recorder/snapshot functions, covering 64,195 recording operations. These exercise decimal halfway values and adjacent doubles, underflow/large/non-finite values, clock rollback, negative durations, Unicode/empty names, summary-key collisions, retention boundaries and repeated successes/failures. Native concurrent checks record 8,000 additional events while taking snapshots. Only TSV data and C++ runners are checked in; the Python oracle remains external.
- Added C++ process integration for text and vision: pending first inference, concurrent reuse, API publication, inference error retention, active-request unload rejection, successful/disabled/failed/retried restoration, child cleanup, captured context and engine flags. A multiple-model scenario checks load order, unload event order and active-model restoration. The shared C++ process fixture now responds to text requests as well as embedding requests. These are fixture processes, not real model inference.

Lifecycle parity remains partial. The published residency mode is currently `resident`, and `vram_estimates_gb` remains empty. Complete resource-manager registration and estimates, automatic/forced swapping, explicit replica selection and scheduling, the source's one-token startup inference verification, all runtime-shape fields, image lifecycle hooks, pool snapshots, cancellation and real model/output/deployment equivalence still need work. The audit found that explicit replica requests currently lose their numeric `#` suffix in native backend selection; that is a concrete follow-up gap. No real weights were loaded and no container image was rebuilt.

Verification: the final C++ build passes all 25 CTest targets in one full run (35.06 seconds), including the 818 lifecycle cases, concurrent recorder checks and text/vision/multiple-model process scenarios (7.70 seconds). The full run initially caught the external-server distinction; the implementation now preserves the existing assertion that an unverified external server's failed request does not mark a model loaded. Native HTTP, fallback, shared GPU, embedding, resource and transport suites all pass. `git diff --check`, first-party source scanning and CUDA Compose configuration validation pass. Strict parity enforcement still exits 1: 37 implemented, 59 partial, all 85 source routes represented. The reference remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The full goal remains active and incomplete.

## Explicit text replicas and native slot scheduling

- Added a C++ replica/slot policy matching the reference's public-source-first name resolution, full and short case-insensitive names, internal IDs, unknown-model fallback and preserved request whitespace. Unicode lowercasing uses the existing ICU dependency. Source order remains decisive for ambiguous short names, even if a later configured source is an exact full-name match.
- Public model requests now select the available replica with the smallest active count, breaking ties in configured order. Explicit requests containing `#` resolve and wait for that instance; an idle sibling does not replace it. Model aggregate capacity and individual instance capacity must both permit acquisition. Waiting does not reserve a model or replica slot, different models have independent counters, and success/error exits release both counts. Acquisition pauses if a shared GPU handoff begins while the request is waiting. Destruction wakes waiting queue workers before joining them.
- The queue passes its selected replica to the backend separately from the request JSON. Managed backend selection now honors that exact ID instead of stripping the numeric suffix and choosing the first sibling. The engine payload carries the public source model name. Unknown model requests retain first-model fallback. This applies to both chat and completion requests; ordinary mock backends retain the existing virtual-method fallback.
- Per-model resource capacities and managed engine parallel arguments now use captured configured parallelism. Zero parallelism resolves to one slot per 32,768 context tokens, bounded to 1–16, and this resolved value reaches the engine's `-np` argument. The integration check changes the environment after construction and verifies that this does not replace the captured value. Actual runtime parallelism after adaptive context changes remains a separate open issue.
- Added 1,084 cases executing the unchanged reference resolver, capacity, acquire and release functions. Cases cover ambiguous names, Unicode/case/whitespace, explicit and unknown IDs, zero/negative/positive configured capacities, actual-instance overrides, speculative capacity, failed replicas, swap conflicts, handoff waits, least-busy ties and seeded operation sequences with exact counter comparison. A blocked source acquisition is observed at its first 50 ms sleep; these fixtures establish policy/state transitions, not wall-clock fairness or cancellation equivalence.
- Native integration exercises two occupied sibling replicas, a waiting public request, independent capacity on another model, explicit busy-replica waiting, idle-sibling service, error cleanup, and managed child-process selection. The process fixture verifies only the requested replica starts, checks the public model payload and automatic parallel argument, and verifies child cleanup. The existing concurrent lifecycle fixture now explicitly declares two native slots. All implementation and checked-in runners are C++17; the external Python oracle contributes TSV data only.

The production integration covers configured ordinary replicas. Although the standalone policy tests speculative, failed-replica and swap states, automatic speculative replica expansion, failure tracking for automatically created instances and actual swap residency are not yet wired into the production catalog. Live instance-capacity refresh, global queue sizing, lazy stream/native cancellation leases, complete configuration conversion and catalog collision behavior, text/vision registration and estimates, the source startup inference probe, and real model/output/deployment equivalence remain incomplete. No real model inference or container image rebuild was performed; the text and resources manifest rows remain partial.

Verification: the final C++ rebuild passes all 26 CTest targets in one full run (54.72 seconds), including all 1,084 reference policy cases and concurrent/managed scheduling checks (2.50 seconds). Native HTTP, lifecycle, fallback, shared GPU handoff, embedding, resource and transport suites pass. `git diff --check`, first-party C++ source scanning and CUDA Compose configuration validation pass, with no fixture service processes remaining. Strict parity enforcement still exits 1: 37 implemented, 59 partial, all 85 source routes represented. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The full goal remains active and incomplete.

## Model catalog construction and automatic speculative replicas

- Ported the model-list construction and per-model CSV configuration into a C++ catalog shared by scheduling and managed backend setup. It preserves Unicode stripping, legacy `@tokens` removal, duplicate-entry IDs, explicit source mappings, source-group order and last-write behavior when an explicit name collides with a generated numeric suffix. Public model listing resolves the final source map before deduplicating, matching the source even for collisions. Numeric `#2` suffixes in genuine source names are no longer universally stripped by these consumers.
- Added source speculative-target selection: the Qwen3.8-27B boundary rule, `-mtp` names, automatic/default behavior, explicit `none`/`mtp`/`dflash2`, legacy aliases and invalid-setting errors. A global DFlash override only selects a compatible target. The Qwen boundary also preserves Python's end-before-final-newline regex behavior after legacy suffix removal.
- Configured speculative targets now expand `N_PARALLEL` into independent single-slot replicas. Expansion appends after all original entries, skips occupied numeric IDs, and copies the originating context, GPU, quantization and original CSV index. It retains the source's per-entry expansion order and collision behavior. CSV parsing preserves empty fields, repeats the final value, supports Python-style Unicode decimal digits/signs/underscores in the supported integer range, and validates GPU tensor indices only under the source's multi-GPU/multiple-instance condition. GPU count comes from the existing CUDA-visible probe rather than physical `nvidia-smi` enumeration.
- Both the queue and backend capture the expanded catalog. Explicit automatic IDs select their own managed process, retain the original model-path index and GPU setting, and launch with one native slot and the configured full speculative context. Engine requests receive the mapped source name, and `/v1/resources` aggregates configured replica capacities. Existing explicit `LLAMA_CPP_CONTEXT` overrides continue to apply by original CSV entry. Model acquisition quantization is captured with the catalog. This configures separate processes; it does not yet enable the source's native speculative decoding implementation.
- Added 2,216 cases executing the unchanged reference model-list loop, `_parse_per_model_configs`, resolver/slot-model methods, `get_models` and `speculative_backend`. Comparisons cover complete ordered available/group state, source/config maps, automatic IDs, tensor configuration, error categories, resolution and public model lists. Inputs cover target boundaries, aliases, Unicode, whitespace, legacy suffixes, explicit/duplicate collisions, empty/short/long CSVs, integer conversion and seeded combinations. Reference source hashes protect the fixture provenance; oracle scripts remain outside the repository.
- Expanded C++ integration checks to run three concurrent automatic replicas and block a fourth until one is released. Managed child-process checks verify appended process positions, original weight path/context/GPU inheritance, one-slot arguments, literal-suffix payloads, public listing, aggregate resource capacity and child cleanup. A separate API case checks colliding internal IDs resolve before public-list deduplication. Existing ordinary replica and lifecycle tests remain in place.

Production catalog expansion is now connected, while speculative engine execution, startup preloading and failed automatic preload tracking, swap residency, live instance capacity updates, full native placement/context behavior, worker context/quant metadata, global queue sizing and lazy stream/cancellation leases remain incomplete. Catalog integer values currently use signed 64-bit storage; arbitrary-size Python integers and all exact startup exception messages remain open. Complete collided-ID engine replacement lifetimes, actual model/output equivalence and deployment parity are still unverified. All first-party implementation/test code added here is C++17; no real weights were loaded and no container image was rebuilt. The goal and affected manifest rows remain incomplete.

Verification: the final C++ rebuild passes all 27 CTest targets in one run (54.28 seconds). The catalog target verifies all 2,216 source cases (1.13 seconds); expanded ordinary/automatic replica integration and the existing 1,084 slot cases pass (3.33 seconds). Native HTTP, lifecycle, fallback, shared GPU, embedding, resource and transport checks pass. `git diff --check`, the first-party C++ source scan and CUDA Compose configuration validation pass, and no fixture service processes remain. Strict parity enforcement continues to exit 1: 37 implemented, 59 partial, with all 85 source routes represented. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The full goal remains active and incomplete.

## Text/vision startup inference verification

- Owned text/vision startup now sends the source's minimal completion request, `{"prompt":"Hi","max_tokens":1}`, after the engine's readiness endpoint succeeds and before publishing residency or dispatching user generation. Embedding warmup retains its separate path. Reusing a resident instance does not repeat startup verification; restarting or restoring an owned text/vision engine does. External server configuration retains its separate observed-response behavior.
- Ported the actual `LLM.__init__` result check: any dictionary containing `error` fails, including null/false/zero/empty values. Lists, scalar results and dictionaries with only top-level `code`/`message` remain accepted by this constructor check. The managed HTTP adapter applies that rule to decoded result bodies; malformed JSON and transport failures fail initialization. This does not claim that HTTP status alone was part of the reference's direct native-call boundary.
- Constructor errors retain the `Failed to initialize LLM server for <source model>:` wrapper and `LLM server test failed:` result representation. The C++ formatter preserves Python dictionary order/replacement, nested values, quoting, non-finite/underflow numbers, integer text and Unicode printable/control/surrogate escaping for the tested runtime. It is scoped to startup diagnostics and leaves unrelated response formatting unchanged.
- A failed probe terminates and reaps the owned child before returning or retrying; it never invokes the waiting user generation or publishes a loaded model. Lifecycle timing covers readiness and verification together. The existing native memory-retry path also receives probe memory failures, and a recovered complete startup records one successful load event. That retry path is still the older coarse layer reduction, not the source's full micro-batch reduction and binary-search implementation.
- Added 741 cases executing the unchanged verification block from `LLM.__init__`, with controlled native method results/exceptions. They compare acceptance and complete error strings across ordinary/error shapes, falsey values, nested results, duplicate JSON keys, Unicode/control/private/unassigned/surrogate values, non-finite/extreme numeric values, exception messages and seeded combinations. Every source case also verifies the exact minimal request dictionary. Only TSV data and C++ runners are checked in; the Python oracle remains external.
- Added managed C++ process checks for text and vision: a blocked probe prevents user inference; falsey errors leave residency absent; recovery performs a fresh probe; ordinary reuse skips it; failed handoff restoration stays unloaded; and successful restoration republishes the model. Result-shape, malformed-JSON and dropped-connection checks cover adapter cleanup. A memory-error recovery check verifies failed probe processes are reaped before the surviving engine is published without asserting the currently divergent layer-search sequence. Existing managed HTTP fixtures now expect the startup completion on each launch, and the shared process fixture records startup and user requests separately.

The inner constructor verification boundary is now covered for the tested values. Full resilient loading remains incomplete: the source reduces micro-batches, classifies opaque initialization failures and searches the maximum viable GPU layer count in isolated probes before final reload. Native still uses its older coarse layer fallback. Complete initialization exception/API mapping, preload/failed-replica handling, native placement/context policy, swap residency, streamed cancellation leases, speculative execution and real model/output/deployment equivalence remain open. No real weights were loaded or container image rebuilt; affected manifest rows and the full goal remain incomplete.

Verification: the final C++ rebuild passes all 28 CTest targets in one run (53.70 seconds). The startup suite passes all 741 source cases plus managed text/vision lifetime, adapter-failure and recovery checks (7.41 seconds). Existing native HTTP launch/retry fixtures, model lifecycle, replica catalogs/scheduling, embedding warmup, fallback and transport suites all pass. The reference runtime is Python 3.12.3 with Unicode 15.0.0. `git diff --check`, first-party C++ source scanning and CUDA Compose configuration validation pass. Strict parity enforcement still exits 1: 37 implemented, 59 partial, all 85 source routes represented. The reference remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The full goal remains active and incomplete.


## Resilient text/vision load control and isolated fit probes

- Added a C++17 resilient-load controller matching the reference post-placement sequence. It first attempts normal construction, retries resource/opaque initialization errors at progressively halved physical micro-batches down to 128 while retaining the original GPU argument, then binary-searches the highest fitting layer count from zero through the requested layer ceiling minus one. The smallest tested micro-batch is retained for probes and the final load. Non-retryable errors stop immediately; explicit GPU argument zero short-circuits after the initial failure. An environment value of zero still enters the source retry policy when no explicit load argument was supplied.
- Successful fit probes are disposed, just like failed probes. The winning placement is loaded once after the search, and a failure of that final load propagates without restarting the search. Exhausted capacity preserves the source model/context prefix and the last full-residency error. The independent request-time coarse OOM layer-reduction policy remains separate.
- Connected the controller to owned text/vision engine startup. Each attempt keeps the requested context and performs the existing one-token native inference check. Startup process output is retained for error classification; child failures cannot abort the API process. Every probe child is terminated and reaped before another attempt. Probe attempts do not publish residency, dispatch user generation or add lifecycle events: the outer complete load records one success or failure. Embedding startup retains its separate source-derived fallback path.
- Known successful GPU placement is captured with its context and reused for matching-context reloads. Physical micro-batches are deliberately recalculated, matching the source wrapper. Retry micro-batch overrides are clamped against the selected logical batch. GGUF layer lookup now requires the source `.block_count` suffix rather than any substring match. Existing explicit native total-layer/context extensions remain supported.
- Fit probes use `LLM_LOAD_PROBE_TIMEOUT`, default 300 seconds and minimum 30. One unrounded monotonic deadline covers the whole candidate attempt, including engine readiness and the one-token HTTP exchange. The HTTP transport can accept that supplied deadline without changing the existing aiohttp fallback timer rounding. The ordinary startup timeout remains separate.
- Added 3,335 cases executing the unchanged reference `_load_llm_resilient_impl`, retry classifier and candidate reducer with controlled construction/probe outcomes. Fixtures compare the complete load/cleanup/probe trace, chosen parameters, error category and full message across initial successes, CPU arguments, odd/negative micro-batches, layer bounds, non-monotone outcomes, probe exceptions, final reload errors and seeded combinations. The full reference `Pipes.py` hash protects provenance; only TSV data and C++ runner code are checked in.
- Added managed C++ process checks for unchanged GPU residency across micro-batch reductions, restoration without micro-batch reuse, exact maximum-layer binary-search order, disposal of successful probes, one final reload, final-reload failure propagation, non-retryable failures and one-event lifecycle accounting. A real blocked-inference probe checks the 30-second minimum total deadline and process cleanup. Existing startup fixtures now distinguish environment CPU settings from known explicit CPU placement; the native HTTP startup OOM fixture checks halved micro-batches with retained GPU layers.

This closes the post-placement retry-controller gap for the tested typed inputs, not the entire LLM startup boundary. Automatic GPU-generation/free-memory batch sizing and all batch-environment parsing remain incomplete dependencies of production configuration. Smart placement/calibration, acquisition before probing, complete known main-GPU/tensor-split restoration, preload/failed-replica behavior, exact native-constructor/API error mapping, arbitrary-size Python integers, swap residency and real model/output/deployment equivalence remain open. First-party implementation and tests added here are C++17; no real weights were loaded or container image rebuilt. All affected manifest rows and the full goal remain incomplete.

Verification: the C++-only rebuild succeeds with `/nonexistent-c-compiler` still configured for C. All 29 CTest targets have passed: the initial run passed 27, then the two corrected test assertions passed in a focused rerun (`native_http` 31.37 seconds; `llm_load_reference` 38.91 seconds). The retry runner verifies all 3,335 source traces and managed process scenarios, including the actual 30-second blocked probe. The initial failures were assertions using the wrong runtime-reporting accessor and assuming a fixed initial micro-batch despite configured GPU information; neither required changing production behavior. Existing startup, lifecycle, embedding, catalog/slot, fallback and deadline suites pass. `git diff --check`, the first-party C++ source scan and CUDA Compose configuration validation pass; no fixture service processes remain. Strict parity enforcement still exits 1: 37 implemented, 59 partial, all 85 source routes represented. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The full goal remains active and incomplete.

## GPU-aware LLM batch sizing and batch-setting conversion

- Replaced the managed text/vision adapter's older free-memory heuristic with the reference automatic batch policy. It uses CUDA-visible ordinals, free and total memory, compute capability, and GPU family. Ada/newer, Ampere and older-device thresholds now match the source; A100 and T4 profiles retain their special limits. Context clamping occurs after successful GPU probing, while the source CPU `(1024, 256)` and probe-failure `(2048, 512)` early returns remain unclamped.
- Ported long-context MTP micro-batch limits, including the Qwen3.8 substring rule, the 131,072-token activation threshold, the 200,000-token/28-GiB exception, and the conservative cap for other MTP names. The batch helper uses `LLM.is_mtp_model` behavior, which differs from the catalog's stricter speculative-backend boundary rule. Diagnostic reason strings preserve source GPU index, one-decimal memory formatting, capability, family and cap suffix.
- Added a shared GPU inference-profile helper matching the source supported-family regex, Unicode word boundaries, visible capacities/MIG sizes, first-match behavior, and optional all-device selection that prefers a 5090 before comparing capacity. Strict C++ per-device property probes support the selected visible ordinal. Capability queries read the complete properties object, matching the inspected installed `torch.cuda.get_device_capability` implementation; property failures participate in the reference probe-failure fallback.
- Ported constructor batch/environment selection separately from the retry helper. Explicit overrides win; empty/auto/exact `0` use automatic values; invalid logical batches fall back to automatic while invalid physical micro-batches fall back to 256. Signed values, `-0`/`00`, Unicode decimal digits, numeric separators, whitespace and the reference runtime's 4,300-digit conversion limit are preserved. The physical micro-batch is always capped by the logical batch. Canonical decimal strings preserve arbitrary configuration precision through selection; exact native-binding/CLI range failures and API mapping remain open.
- Retry sizing now executes only after the first retryable construction failure, before cleanup/retry iteration, and re-reads current automatic sizing. It applies `_configured_llm_ubatch`'s distinct minimum-one and context clamp rules. The 3,335 resilient-load source fixtures now also compare the position of this sizing callback in the load/probe/cleanup trace.
- Added 7,242 source-executed cases: 3,482 automatic batch cases, 148 GPU-profile cases and 3,612 constructor/retry environment cases. The oracle executes the unchanged source functions and constructor statements, with controlled CUDA operations and a plain constructor parameter holder. Fixtures compare selected parameters, profile family/capacity, diagnostic text and complete probe order. They cover all threshold boundaries, invalid/fallback device indices, failure stages, CPU returns, context/MTP limits, Unicode/profile boundaries, override precedence, invalid strings, large integers and decimal conversion limits. Source hashes cover `LLM.py`, `InferenceSettings.py` and `Pipes.py`; only TSV data and C++ runners are checked in.
- Native CUDA-driver and managed-process checks verify T4, A100 and 5090 sizing, MTP caps, 5090 profile priority, reordered/hidden visible devices, property/memory failures, invalid-value fallbacks, Unicode settings, negative settings at the adapter boundary, engine arguments and runtime telemetry. A controlled failed first load changes the fake driver's free-memory reading; the next attempt must use the newly measured batch budget, proving production retry setup does not reuse an eager pre-failure measurement. Child-process cleanup is checked throughout.

The source batch/profile/configuration selection policy is now connected to production text/vision startup. Full smart placement/calibration, exact xllamacpp-to-llama-server parameter and exception behavior, native context policy, KV/cache and speculative settings, known main-GPU/tensor restoration, preload and failed automatic replica handling, swap residency, real model/output equivalence and deployment verification remain incomplete. The constructor fixtures verify Python selection statements with a parameter holder, not actual xllamacpp setters; xllamacpp is not installed in the host oracle environment. No real model weights were loaded or container image rebuilt. All first-party implementation/test code added here is C++17, and affected manifest rows remain partial.

Verification: the C++-only rebuild passes all 30 CTest targets in one full run (47.76 seconds), including the updated 3,335 retry traces and 7,242 batch/profile/environment cases. After adding the live-memory-change process check and aligning capability property-failure handling with the installed PyTorch source, the focused batch suite passes again (3.72 seconds). Native HTTP, startup/lifecycle, replica/catalog, embedding, resource, transport and modality suites passed in the full run. `CMAKE_C_COMPILER` remains `/nonexistent-c-compiler`; the first-party native source scan contains no Python, shell or C implementation files. `git diff --check` and CUDA Compose configuration validation pass, and no fixture service processes remain. Strict parity enforcement still exits 1: 37 implemented, 59 partial, with all 85 source routes covered. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The full goal remains active and incomplete.

## Prompt/RAM cache settings, GPU KV precision and engine revision

- Added C++ host prompt-cache resolution from `LLM_PROMPT_CACHE_MIB`: context-scaled automatic defaults from 8,192 to 32,768 MiB, explicit positive integers, disabled aliases/non-positive values, and the source invalid-setting fallback/reason text. Canonical decimal strings preserve configuration precision through policy selection, including Unicode digits/separators and the reference runtime's 4,300-digit conversion limit. Shared integer conversion now also serves the previously ported batch policy; the startup diagnostic string-representation helper is reused for cache diagnostics.
- Ported `cap_prompt_cache_mib_for_ram` with its exact ordered environment/probe behavior: disabled requests skip all work, unsafe bypass skips RAM probing, either missing memory argument triggers one replacement probe, selected probe exception classes fall back to a 4,096-MiB ceiling, invalid margins/fractions use source defaults, margins clamp at zero, fractions clamp to zero/one, and effective budgets round down to 256-MiB blocks. NaN retains the source conversion error instead of silently becoming a usable budget. Requested/effective sizes and reason strings preserve tested source behavior across ordinary/extreme integer values, signed zero, infinities and Unicode settings.
- Added the Linux total/available RAM subset used by this cache cap, verified against the installed psutil 6.0.0 implementation. Missing or zero `MemAvailable` uses the source free/cache or zone-watermark calculation, including active/inactive file pages and reclaimable slab. Negative availability clamps to zero; availability above total follows the source free-memory fallback. `/proc/zoneinfo` is read only when the fallback needs it. This is scoped to prompt-cache planning; other RAM consumers retain their existing implementation. The source requirements leave psutil unpinned, so other library/deployment versions remain unverified.
- Text/vision startup now computes the requested and effective cache size on every load and restoration, passes the selected `--cache-ram` value, and publishes `prompt_cache_requested_mib` and `prompt_cache_mib` in runtime telemetry. Existing explicit engine-argument overrides remain supported. Chat requests explicitly upsert `cache_prompt=true`, matching `LLM.chat` even when a caller supplied false. Completion requests and embedding startup retain their separate behavior.
- KV-cache selection now uses the selected GPU's source family profile. A truthy `KV_CACHE_TYPE_<family>` overrides the global setting before stripping, so an empty value falls through while a whitespace value selects the default. Automatic/empty resolves to `q4_0`; explicit `q4_0`, `q8_0`, `f16` and `f32` are accepted, and other values raise the source validation message before process launch. Both K/V engine arguments and runtime metadata use the selected type.
- Added 9,111 source-executed cases: 549 prompt-cache resolution cases, 7,052 RAM-cap cases, 550 KV selection cases and 960 Linux RAM cases. They compare values, full reason/error strings, error categories, and environment/probe order. Source hashes protect `LLM.py` and `InferenceSettings.py`; the RAM data records the tested psutil version/source hash. External Python oracles remain in `/tmp`; checked-in implementation and runners are C++17.
- Added managed C++ process checks for automatic/disabled/invalid/Unicode cache sizes, card-specific KV precision, whitespace overrides, requested/effective metadata, chat prefix caching, completion separation, restoration that refreshes its RAM cap, and pre-launch rejection of invalid KV/NaN fraction settings. Successful child cleanup and load lifecycle accounting are verified. Existing 7,242 batch cases and 741 startup diagnostic cases cover shared-helper regressions.
- The C++ engine manager, CUDA Dockerfile and Compose default now pin llama.cpp to `aa94f2086147cd4bfbcff818ce998dffce5fcb39`, the revision in the reference's `scripts/build_xllamacpp.py`. Explicit ref overrides remain available. Tests check the default tool status and consistency with the reference script, Docker and Compose. The cache option mapping was checked against that revision's [upstream argument implementation](https://github.com/ggml-org/llama.cpp/blob/aa94f2086147cd4bfbcff818ce998dffce5fcb39/common/arg.cpp#L1614); this does not verify a rebuilt engine image or real-model execution.

Cache configuration and selected runtime metadata are now connected to the managed adapter. Full checkpoint/context policy, reasoning/sampling and speculative engine settings, actual cached-prefix reuse/output equivalence, smart placement/calibration, complete known placement restoration, preload/swap behavior, native-binding/CLI range errors and API exception mapping remain incomplete. The unpinned reference dependency environment and full CPU/CUDA/ROCm/RPi deployment remain unverified. No real weights were loaded or container image rebuilt. All affected manifest rows and the full goal remain incomplete.

Verification: the C++ rebuild passes all 31 CTest targets in one run (43.42 seconds). `llm_cache_reference` verifies all 9,111 cases plus managed cache/lifecycle and revision checks (2.61 seconds). Existing batch, retry/startup, native HTTP, embedding, lifecycle, catalog/slot, transport and modality suites pass. `git diff --check`, first-party C++ source scanning and CUDA Compose configuration validation pass; no fixture service processes remain. Strict parity enforcement continues to exit 1: 37 implemented, 59 partial, all 85 source routes covered. The reference remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The goal remains active and incomplete.


## Managed MTP and DFlash speculative engine configuration

- Connected the existing speculative catalog selection to owned text/vision engine launches. Each load and restoration re-reads the reference backend selector, enables the pinned engine's `draft-mtp`, `draft-dflash`, or `none` mode, and forces a single inference slot for speculative backends. Embedding startup retains its separate configuration. Runtime lifecycle metadata now includes the selected speculative type.
- Ported MTP model-size parsing, total-visible-VRAM probes, GPU-family/card/global draft-length precedence, automatic Qwen/T4/small-model choices, the 1–16 override clamp, and model-specific probability defaults. NaN, signed zero, malformed values, Unicode digits, digit separators and Python's decimal digit limit follow the tested reference helpers. GPU property callbacks retain signed 64-bit ordinals until the native CUDA boundary.
- Ported DFlash's distinct override/whitespace rules and validation order. Zero draft length is invalid; valid lengths are 1–7 and probabilities must be finite in 0–1. The resolver honors explicit regular-file paths and custom filenames, otherwise downloading the corrected pinned draft revision through the native shared Hub cache. Cache location follows the existing HF_HOME/legacy/current cache selection helper.
- DFlash disables standalone automatic memory fitting, sets its own visible CUDA worker device, and receives the same GPU-layer argument as the target on every initial load, isolated fit probe and final reload. CLI probability formatting preserves the reference's float assignment for subnormal/underflow values using exact hexadecimal notation where decimal `std::stof` would fail. Native extra-argument overrides remain supported.
- Added 2,500 cases generated by executing the unchanged reference functions: 210 model-size cases, 1,100 MTP draft-length/probe traces, 168 MTP probability cases, 932 DFlash settings/error traces and 90 draft-file selection/download traces. All three source files are hash-checked. The oracle generator remains outside the repository; checked-in implementation and tests are C++17, with TSV fixtures.
- Added managed C++ process checks for MTP/default/disabled/incompatible target behavior, DFlash flags and explicit draft paths, restoration after backend/settings changes, validation before launch, clamped overrides, exact float conversion, CPU/worker GPU selection and target-matched final retry placement. The existing Hub transport suite now checks six concurrent draft workers sharing one pinned download, offline reuse and failure, custom filenames reusing blobs, explicit-path validation and cache-location precedence.

This implements speculative configuration and acquisition for the covered native CUDA/CPU paths. Real-weight speculative decoding and generated-output equivalence, ROCm device selection, the reference's separate native DFlash hotfix, full Hub authentication/redirect/Xet/error behavior, native engine parameter-range exceptions, live scheduler capacity refresh, complete tensor placement, reasoning limits and checkpoint behavior remain open. No real weights were loaded and no container image was rebuilt. All affected manifest rows remain partial and the full goal remains active.

Verification: the C++-only rebuild succeeds with `/nonexistent-c-compiler` still configured for C. All 32 CTest targets have passed: the full run passed the 31 existing targets (53.83 seconds), and after preserving the complete embedded-NUL exception message, the speculative and Hub suites passed together (7.41 seconds; speculative 7.41, Hub 2.38). Earlier build attempts corrected test-only restoration callback usage; the source oracle also identified the oversized-ordinal probe-order discrepancy, now fixed at the native probe boundary. All 2,500 speculative source cases and managed process scenarios pass. `git diff --check`, the first-party C++ source scan and CUDA Compose configuration validation pass. Strict enforcement still exits 1 with 96 rows: 37 implemented, 59 partial, all 85 source routes represented. The reference remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The full goal remains active and incomplete.


## Pinned engine parser and constructor binding behavior

- Audited the reference source-build path against its pinned xllamacpp binding and llama.cpp headers. The reference script explicitly states that the large-image DFlash cache fix is already upstream at `aa94f2086147cd4bfbcff818ce998dffce5fcb39`; it applies no separate patch. The earlier outstanding “native hotfix” note is therefore narrowed to verifying large-image DFlash execution with real weights. The binding's vendored `common.h` and pinned engine header have identical SHA-256 hashes.
- Built the actual pinned upstream CPU engine in an external temporary directory. Added an optional C++ contract test linking its shared libraries and parsing production-equivalent flags. It checks source hashes, context/batch/cache dimensions, KV precision, MTP/DFlash selection and draft arguments, exact subnormal float parsing, single-GPU/projector settings, warmup/continuous batching and checkpoint defaults/overrides. The optional target uses supplied source/build paths and does not download dependencies during normal native builds. This is parser/build verification, not a real-model inference test.
- Compiled unchanged selected `CommonParams` Cython property bodies externally against the pinned engine with CPython 3.12.3/Cython 3.1.3, then executed the unchanged reference constructor statements against those compiled properties. This confirmed two source quirks: nonnegative `REASONING_BUDGET` raises `AttributeError` because only `sampling.reasoning_budget_tokens` is exposed; element-wise tensor-split writes change a returned list copy and leave the native array at zero. Both behaviors are preserved instead of substituting intended behavior. No Cython or Python implementation is checked into ezlocalai.
- Added shared C++ parameter handling for Python integer conversion/errors, reasoning-budget validation, automatic/speculative slot counts and signed 32-bit slot bounds, tensor split mode/fitting effects, and Qwen's image-token minimum. Speculative draft validation reuses the shared integer converter. Validation now places reasoning before KV and live speculative selection. Negative reasoning budgets keep the unlimited default; malformed and nonnegative values fail before spawning an engine.
- Connected source catalog worker tensor arrays to `--split-mode none --fit off`, retaining the original full 128-element array in runtime telemetry. Explicit native `TENSOR_SPLIT` and extra-argument extensions remain supported. The broader automatic single-GPU selection and smart placement policies are still incomplete. Projectors for the source Qwen family receive the required 1,024 minimum image tokens.
- Explicitly passes the source warmup and continuous-batching defaults, 32 context checkpoints and 8,192-token checkpoint spacing. Lifecycle runtime shape now includes both checkpoint fields, including existing explicit native overrides and restoration. CUDA Docker/Compose defaults now use the reference's `86-real;89-real;120-real` architectures and `GGML_CUDA_NCCL=OFF`.
- Added 2,794 reference/compiled-binding cases: 47 reasoning conversions/errors, 468 parallel settings/overflow cases, 2,272 tensor/fitting cases and seven image-family cases. Managed process checks cover worker GPU isolation, full-array runtime metadata, defaults/overrides, restoration, projector selection, validation precedence and failed-startup cleanup. The fixture records both source and upstream binding provenance.

Open scope still includes automatic placement/calibration, full native parameter-range/error mapping and complete constructor ordering, exact source model acquisition, live scheduling/residency, real model outputs, GPU/ROCm engine execution, large-image DFlash behavior and deployment equivalence. The new source/build evidence does not establish full inference parity. All affected manifest rows remain partial; the full goal remains active.

Verification: the actual pinned CPU `llama-server` build completed successfully. The first-party C++-only build retains `/nonexistent-c-compiler` for C. All 34 enabled CTest targets have passed (33 standard targets plus the optional actual-engine contract): the full run passed 32 targets in 50.24 seconds, then the corrected native HTTP and parameter integration assertions passed together in 30.91 seconds (parameter suite 3.35 seconds). The actual-engine contract passed in 0.05 seconds. The two assertion updates account for source split-mode behavior and Python-style floating-point tensor metadata; no production fallback was added to satisfy them. An initial contract-test compile error was corrected by value-initializing the upstream parameter aggregate. All 2,794 source/compiled-binding cases and the existing 2,500 speculative cases pass. `git diff --check`, first-party C++ source scanning and CUDA Compose configuration validation pass. Strict parity enforcement remains intentionally failing: 96 rows, 37 implemented and 59 partial, all 85 source routes covered. The reference checkout is unchanged and clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. No real model weights were loaded and no CUDA container image was rebuilt. The full goal remains active and incomplete.

## Native text/vision acquisition and first real CPU inference smoke

- Connected the existing C++ `LLM.download_model` port to owned text/vision engines. Repository models no longer use the recursive “first GGUF anywhere” fallback or delegate normal acquisition to `llama-server -hf`. The source model and captured per-model quantization select a file under `models/<basename before -GGUF>`; native `MODELS_PATH` and explicit model/projector path extensions remain supported. The acquired optional projector reaches the engine and the existing Qwen image-token setting.
- Matched the reference's preliminary acquisition for placement metadata, its tolerated preliminary failure, and its repeated acquisition inside every constructor/fit attempt and restoration. The downloaded GGUF supplies block count plus the output layer for resilient fit bounds. A failed restoration after file removal cannot reuse a stale model path. This does not port automatic placement/calibration or the full metadata reader.
- Corrected Unicode lowercasing in projector candidates. The source-executed selection/call-order fixture grows from 1,028 to 1,068 cases, adding Kelvin sign, dotted I, Greek sigma, sharp S and mathematical letter names. The oracle generator remains outside the repository; checked-in implementation, test runners and transport fixtures remain C++17/TSV.
- Added 12 managed text scenarios using a loopback Hub and C++ engine fixture: per-model quant selection, restoration with captured configuration, inherited speculative replicas, wrong-quant/unrelated local files, preliminary failure recovery, final acquisition failure, missing projectors, metadata-bounded fit retries, offline local reuse, explicit overrides, a custom model root and failed restoration after deletion. Checks cover request/authentication paths, list/HEAD/GET counts, unchanged-file reuse, projector preference, one-token startup verification and child cleanup.
- Updated the C++ management tool's `model-status` to use the shared model catalog and the downloader's local selection policy with read-only hooks. It no longer advertises an unrelated/wrong-quant file as usable, supports captured replica configuration, and performs no Hub requests or directory creation. Its existing `hf` field describes pending repository/quant selection; the managed engine receives a local `-m` argument.
- Added optional `llm_real_engine_smoke`, enabled by a supplied model path alongside the pinned source/build paths. Downloaded the public 1,185,376-byte `ggml-org/models` asset `tinyllamas/stories260K.gguf` at revision `499bc8821c6b12b4e53c5bffcb21ec206f212d81` into an external temporary directory. The C++ test verifies SHA-256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`, serves those actual weights over a loopback Hub, then runs the actual pinned CPU engine through native acquisition/startup. Repeated greedy completions, SSE chunks and generation after media unload/reload produce identical text. Checks also cover byte preservation, residency, metadata rechecks without duplicate weight downloads, and closed engine listeners after destruction. The test uses explicit context 512, f16 KV and two CPU threads; ordinary builds do not fetch weights.
- The real-engine test exposed an existing ordinary-text context discrepancy: `LLM_MAX_TOKENS` is captured in the catalog but legacy adaptive context sizing can supersede it. The smoke test uses the explicit native context override to isolate acquisition and execution. Correcting source preload/lazy-load/context-growth/reset policy is still required; this limitation is now explicit in the README and manifest.

Verification: the C++-only build succeeds with `/nonexistent-c-compiler` retained for C. All 35 enabled CTest targets pass in a single final run (64.75 seconds): 33 standard targets, the pinned engine parser contract and the new real CPU smoke. The acquisition suite passes 1,068 source cases, 16 Hub HTTP traces and managed scenarios in 13.36 seconds; the actual-engine smoke passes in 1.92 seconds. Initial test failures were corrected for a fixture helper's empty parent path, an old music fixture relying on delegated fake-repository loading, and the text estimator's extra output layer. `git diff --check`, first-party C++ source scanning and CUDA Compose validation pass. No fixture or real-engine service processes remain. The reference checkout stays clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.

Strict enforcement still exits 1: 96 rows, 37 implemented, 59 partial, all 85 source routes represented. This is the first real-weight CPU inference smoke, not a differential comparison of full ezlocalai model outputs. Full Hub retry/timeout/authentication/Xet behavior, constructor errors, context policy, placement, residency, vision/speculative outputs, GPU/ROCm execution and deployment remain open. No CUDA image was rebuilt. The full parity goal remains active and incomplete.

## Captured context ceilings and inference reload guards

- Audited the current `Pipes._get_response_internal` request path. It selects `_configured_context_limit_for_model(target_model)` before `_get_llm`, uses that limit as the request context, clamps streaming estimates to it, and limits reloads to configurations with at most one model and one effective slot. Older comments describing automatic growth no longer describe the executed request path.
- Replaced native text startup's implicit 262,144-token floor and character-based prompt growth with the captured per-model ceiling. Long buffered/streamed prompts no longer enlarge the allocation, and context error text no longer causes an above-ceiling reload. Existing positive explicit native context overrides remain available. Multiple-model and multiple-slot configurations now suppress inference-time memory reloads, matching the source guard rather than relying only on momentary inactivity.
- Ported the source optimal-context CSV maximum/fallback and configured-ceiling rules to C++. Decimal strings preserve Python integer range inside the policy helpers. Empty/malformed optimal CSV input falls back to 40,000; zero configured context falls back through the captured optimum and 16,384; negative request ceilings clamp to one. Preload keeps its distinct constructor path: nonpositive arguments read the live global integer setting. Binding assignment distinguishes signed 32-bit overflow from values outside C `long`, using the messages observed from the compiled pinned property.
- Changed requested-context plumbing to optional signed 64-bit values so explicit zero is distinct from an absent override. Successful loads update lifecycle context with the selected load context, and known placement reuse compares that context. Media restoration preserves the saved runtime context; video retry uses the current explicit override or captured source configuration. An insufficient zero-context preload is replaced by the request's configured fallback allocation.
- Added 1,066 fixtures executing the unchanged source optimal-context block, configured-context method, reload guard, effective constructor context statement and compiled `CommonParams.n_ctx` setter. Managed C++ tests cover six scalar/CSV/Unicode/zero/negative settings, long buffered and streaming prompts, environment changes after configuration capture, media restoration, separate model contexts, zero preload, overflow-before-launch and inference reload suppression for explicit/automatic parallelism and multiple models. All children are checked for cleanup.
- Removed the real CPU smoke test's native context override. It now acquires and runs the pinned real GGUF with `LLM_MAX_TOKENS=512`, verifies runtime context directly, and retains deterministic buffered/streamed and unload/reload checks.

Verification: the first-party C++ build succeeds. All 36 enabled CTest targets have passed: the full run passed 34 targets in 55.08 seconds; after correcting the new C-long overflow distinction and an old OOM test's three-model configuration, the context suite, native HTTP suite and real CPU smoke passed together in 30.64 seconds (context 8.75 seconds; real smoke 2.05 seconds). The OOM retry test now explicitly uses one model; separate new scenarios verify that concurrent configurations do not reload. An initial test-fixture compile error was fixed by passing the cycle count explicitly through `std::thread`. `git diff --check`, CUDA Compose validation and the first-party C++ source scan pass. No test fixture or real engine remains running, and the reference checkout remains clean.

This closes the observed ordinary-request context ceiling discrepancy, not all lifecycle or error-response behavior. Complete post-inference fallback and API error propagation still differ from the reference; the native HTTP context test currently checks preserved engine errors, while the source may enter its separate inference fallback. Image-description context transitions/reset timers, arbitrary-size catalog/production context plumbing, runtime capacity refresh, source pointer/residency behavior and full model-output equivalence remain open. Strict enforcement still fails correctly with 96 rows, 37 implemented and 59 partial, all 85 source routes represented. The full goal remains active and incomplete.

## Buffered wrapper errors and lazy chat stream failures

- Executed the unchanged `LLM.py` buffered chat/completion error blocks and streaming callback against a shared input matrix. They deliberately differ: buffered falsey nested errors are ignored, while callback errors include falsey nested values and every top-level `code`. Buffered truthy scalar/list errors preserve the observed `AttributeError`; dictionary errors preserve Python string/repr formatting, default messages and prompt/context-count suffixes.
- Added a typed C++ inference result containing the original engine body, exception category/message, engine/startup provenance, lazy chat phase and preceding stream events. Legacy string backend methods still return the original body. The queue uses the typed method for text inference, allowing an engine error to be distinguished from an arbitrary JSON dictionary.
- Managed buffered memory errors now enter the existing guarded retry loop even when the engine reports HTTP 200. Lazy chat errors bypass that retry loop. Initial constructor failures remain distinguishable from inference failures; a failed reload after generation is marked as an inference failure. The source's additional outer memory reduction/attempt is still unimplemented.
- The native chat API now emits the source `streaming_error` SSE event after preceding chunks, with no subsequent `[DONE]`. Its error text matches Python's 1,000-character slicing, empty-message fallback, ASCII escaping and surrogate-pair encoding. Raw truthy stream strings validated as false instead produce the observed plain HTTP 500 serialization failure. Slot/process cancellation lifetime and callback-array flattening remain separate open work.
- Added `llm_response_reference`: 2,238 source-executed wrapper cases and 12 source-executed API error-event cases, with reference-file hash checks. Ten managed response/retry scenarios, one constructor-failure scenario and three managed queue/API scenarios check HTTP-200/non-200 errors, buffered versus lazy retries, context counts, falsey errors, preceding chunks, raw stream coercion, startup provenance and child cleanup. The source oracle is `/tmp/ezlocalai_llm_response_oracle.py`; only C++ tests and TSV fixtures are checked into the native project.
- The first differential run caught decimal underflow formatting (`1e-400` versus Python `0.0`). The C++ error formatter now applies Python double conversion recursively before rendering error values, without changing unrelated response serializers.

Verification: the C++-only build succeeds with `/nonexistent-c-compiler` still configured for C. All 37 enabled CTest targets have passed: the initial full run passed 36 targets in 63.79 seconds; after fixing the new underflow-format case, the response suite (7.30 seconds), native HTTP suite and real CPU smoke (2.28 seconds) passed together in 30.65 seconds. `git diff --check`, first-party C++ source scanning and CUDA Compose validation pass. No fixture or actual engine processes remain. The reference checkout is clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Strict enforcement correctly exits 1. No CUDA image or GPU inference was verified.

This work does not complete buffered post-inference fallback. The reference calls `fallback_inference(messages)` without the full request after buffered failures, returns a canned HTTP-200 response when unconfigured, first tries its ezlocalai client when available, then uses a separate OpenAI-compatible SDK path. Completion fallback converts the chat result into its reduced text/model shape. These paths, startup/API capacity-error mapping, successful response cleaning/model rewriting, complete local streaming and full real-model output equivalence remain open. Strict enforcement remains at 96 rows, 37 implemented and 59 partial, with all 85 source routes represented. The full parity goal remains active and incomplete.

## Buffered post-inference fallback and final memory retry

- Connected typed buffered inference failures to `Pipes.fallback_inference` behavior. An unconfigured client produces the reference assistant fallback dictionary with HTTP 200. Per-request `disable_fallback` still controls only precheck/queue forwarding; it does not suppress this post-error path. Initial constructor failures remain distinct and do not enter it.
- Implemented the separate post-error forwarding payload. Chat sends its messages; completion creates one user message from the prompt. Both omit the original requested model, stream and generation parameters, matching the source call without `data`. The captured fallback URL/key and existing availability cache are reused. Available ezlocalai clients receive only messages and the live default model; dictionaries containing `choices` survive intact, and other JSON values become Python-style string content.
- Implemented the tested OpenAI-compatible fallback policy in C++, including base-path-relative `chat/completions`, the live default model or `FALLBACK_MODEL` override, message extraction/defaults, malformed response errors and status-error formatting. The override is captured before availability, as in the source, and applies only to this SDK path. Source error responses can retry twice, including 408/409/429/5xx and explicit retry headers; native supports numeric Retry-After/Retry-After-ms and jittered default backoff. SDK authorization retains a nonempty literal `none` key, unlike the ezlocalai client.
- Added queue metadata identifying model/fallback response dictionaries. Successful dictionaries containing an `error` field no longer become generic backend HTTP errors. Completion fallback reduces chat output to choices/text and model, with source API failures for malformed shapes. Chat dictionary streaming iterates keys; completion streams the resulting text/iterable. JSON serialization still rejects non-finite final values.
- Included the final outer memory reduction/attempt after the source's three inner attempts. Managed tests now start with enough layers to reach all four attempts; exhaustion can still stop earlier. Lazy chat failures remain excluded from memory reloads.
- Added `llm_post_fallback_reference` with 252 cases executing the unchanged `Pipes.fallback_inference`, completion conversion and app endpoint functions. The oracle uses the installed real OpenAI 2.21.0/HTTPX 0.28.1 SDK with MockTransport for response construction and retry behavior. C++ tests replay real HTTP responses through the native queue and compare complete API bodies/status/media types plus method/path/auth/payload traces. Cases cover configured/unconfigured/available/unavailable routing, forwarding-to-SDK failure, model precedence, 201 success, status retries, error-bearing choice dictionaries, malformed choice/message/content values, stream coercion and completion iterable output. Only C++ and TSV fixtures are checked into the native project; the generation script is `/tmp/ezlocalai_post_fallback_oracle.py`.
- Updated earlier HTTP/lifecycle fixtures whose provisional expectations preserved raw engine error statuses. They now verify canned post-inference fallback and retained residency. The first full run also exposed a retry fixture that exhausted GPU layers before the fourth attempt and a fixture count compiled before oracle expansion; both were corrected and rebuilt.

Verification: the C++-only build succeeds with the nonexistent C compiler still configured. All 38 enabled CTest targets have passed: the first full run passed 34 in 56.33 seconds; the corrected native HTTP, lifecycle, response, post-fallback and actual CPU smoke suites passed together in 30.54 seconds. The 252 new cases passed in 1.53 seconds, managed response cases in 8.36 seconds and real CPU smoke in 2.10 seconds. `git diff --check`, the first-party C++ source scan and CUDA Compose validation pass. No fixture or actual engine remains running. The reference checkout is clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.

This establishes the tested post-inference response/routing policy, not full SDK transport parity. The reference OpenAI requirement is unpinned. Alternate SDK versions, HTTPX timeout phases and connection scheduling, default SDK headers, proxy/auth/TLS/redirect/cookie details, HTTP-date retry headers and numeric/Unicode extremes remain open. The native SDK path currently uses the shared C++ HTTP transport with a total deadline rather than HTTPX's complete timeout/session behavior. Startup/API capacity errors, no-local-model lifecycle branches, successful response cleaning/model rewriting, large queue response limits, callback arrays, cancellation/slot lifetime and complete real-model output parity also remain incomplete. Strict enforcement correctly exits 1 with 96 rows, 37 implemented and 59 partial, all 85 source routes represented. No CUDA image or GPU inference was verified. The full goal remains active and incomplete.

## Buffered response cleaning, source model names and large queue results

- Connected the reference `LLM.chat`/`LLM.completion` response transformation to managed text/vision inference. Only the first buffered choice is cleaned. Default stop tokens, Python Unicode whitespace stripping, conditional `[Insert ...]` removal, retained whitespace after placeholder removal and falsey content follow the source. Completion falls back from falsey `text` to message content; streaming completion skips cleaning. Per-request stop values do not change the wrapper's constructor stop list.
- Dictionary responses report the configured source model, including results without choices or with falsey error fields. Internal replica IDs and engine-provided labels no longer leak through this buffered wrapper. Observed resource names follow the source name. Two inherited HTTP/fallback expectations were updated after checking the reference behavior.
- Malformed choice/message/content values preserve source exception categories and messages, including missing-message `KeyError`, scalar/list attribute errors and nonfinite float types. These failures enter the existing post-inference fallback. Duplicate keys in modified objects collapse before assignment. Successful completion dictionaries with a true stream flag use `choices[0].text` through the API, as in the source.
- Typed local results retain an owned C++ string across the queue instead of relying on its bounded C ABI buffer. A managed API test sends more than 5 MiB with the buffer configured to 64 KiB and verifies the entire cleaned result. The C ABI remains bounded; full queue lifecycle/history/cancellation parity remains open.
- Added 2,632 cases executing unchanged `clean`, `LLM.chat`, `LLM.completion` and sampling helper methods with a controlled engine result holder. Cases cover both buffered methods, completion stream booleans/raw strings, every default stop token, Unicode whitespace, embedded NULs, placeholders, falsey/nonfinite/large-number values, malformed choices, preserved later choices, usage fields and source model names. Fixtures check the current `LLM.py` SHA-256. The external generator is `/tmp/ezlocalai_output_oracle.py`; checked-in implementation and runners are C++17, with TSV data.
- Eight managed process/API scenarios verify cleaning, wrapper stop behavior, fallback, stream-specific cleaning, source names, large results and child cleanup. The real CPU smoke compares cleaned buffered text with raw streaming text using the independently covered whitespace behavior, since the source preserves streaming whitespace.

The old build cache referred to missing temporary engine sources/libraries. A fresh build at `/tmp/ezlocalai-output-build` uses `CMAKE_C_COMPILER=/nonexistent-c-compiler`. The exact pinned llama.cpp source was reacquired and rebuilt externally. `hf download` restored the pinned 1,185,376-byte stories260K model, whose SHA-256 matches `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`. Optional actual-engine parser and CPU smoke checks are enabled in that build. No third-party engine or Python implementation was added to the repository.

Verification: all 39 enabled CTest targets have passed. The full run passed 37 in 92.07 seconds, including all 2,632 output cases and managed scenarios (4.88 seconds), the pinned parser (0.30 seconds), and real CPU acquisition/repeated/streamed/unload-reload generation (1.98 seconds). After correcting the two old model-name expectations, native HTTP and fallback passed together in 30.38 seconds. `git diff --check`, CUDA Compose validation and the first-party C++ source scan pass. No fixture or real-engine service remains running. The reference checkout is clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.

This verifies the covered response transformation and removes typed local result truncation. Full inference parity remains incomplete: the adapter still forwards fields the reference wrapper filters (including user stop/seed/model controls), and does not fully reproduce `Pipes._apply_model_config_overrides` or constructor sampling defaults. Local streaming callback normalization, cancellation/slot lifetime, startup capacity error mapping, no-local-model branches, residency/placement, modality engines, deployment and full differential model-output verification remain open. Strict enforcement still exits 1: 96 rows, 37 implemented, 59 partial, all 85 source routes represented. No CUDA image or GPU inference was verified. The full goal remains active.

## Local request filtering, model profiles and native callback guard

- Replaced direct forwarding of local inference requests with the current `LLM.chat`/`completion` payload construction. The wrapper selects input, output/sampling limits, stream, supported sampling extras and chat-template options. It explicitly enables chat prompt caching and implements repeat-penalty alias precedence. Controls the reference drops (including model, seed, stop, tools, user, n, echo and logit_bias) no longer reach local engines. Replica selection remains separate and response model names still identify the selected source. Remote forwarding retains its separate behavior.
- Ported all current `Pipes.MODEL_CONFIG_OVERRIDES` profiles and `ModelSettings` recognition/precedence. This includes MiniCPM5-2B variants, Qwen instruct/MTP profiles and Qwen3.8 thinking/instruct settings. User template options merge over defaults; the standard reasoning object and top-level effort follow source precedence; only literal false template thinking selects instruct mode. Model settings replace scalar user values as in the source, while the wrapper's explicit repeat_penalty still wins over repetition_penalty.
- Native model construction captures `LLM_MAX_OUTPUT_TOKENS` with Python integer parsing, before model acquisition in each constructor attempt. Resident instances retain it, and reconstruction/restoration refreshes it. Defaults sent by the wrapper are 1.31 temperature, 0.95 top_p and the captured output limit. API requests independently inject 8192 when max_tokens is absent. Nulls and raw stream types are preserved. Invalid output-limit strings fail before process launch.
- Inspected pinned xllamacpp `64fbe0fe7c407d5d6418bc794d89cf553736d9e3`. Its completion method requires a callback whenever stream is truthy, while `LLM.completion` supplies none. Added this exact ValueError before engine inference and connected it to existing post-inference fallback. Earlier wrapper fixtures used a result holder and did not verify this binding behavior. Chat uses its separate callback path; its stream value is no longer silently converted to a boolean.
- Added 6,183 source-executed cases: 44 model-recognition cases, 5,056 complete model-setting results and 1,083 wrapper payload/error/binding cases. The unchanged reference functions are executed against a recording engine. The actual dictionary/callback-guard code from the pinned binding is compiled externally with Cython 3.1.3 and exercised by the oracle. Fixtures hash `ModelSettings.py`, `Pipes.py` and `LLM.py`, and record the binding file hash/revision. The generator is `/tmp/ezlocalai_request_oracle.py`; only C++ implementation/tests and TSV data are checked in.
- Managed process checks verify ordinary/MiniCPM/Qwen/MTP payloads, source selection for an unknown requested alias, reasoning-driven profile changes, dropped controls, API versus constructor output defaults, resident capture, refresh after media restoration, invalid pre-launch limits, and callback rejection without a generation request. Existing replica tests now check selection through process identity and response source names rather than a model field the reference does not forward.
- Updated the real CPU smoke to verify buffered completion generation plus repeated buffered/streamed chat through an explicit ChatML template. It also checks completion's callback rejection, model acquisition, residency, and stable completion after unload/reload. Removing forwarded seed does not affect these deterministic temperature-zero checks.

Verification: the C++ rebuild and all 40 enabled CTest targets pass in one full run (64.58 seconds). The new request suite passes all 6,183 cases plus managed checks in 3.31 seconds. The pinned native parser passes, and the revised actual-engine CPU smoke passes in 1.82 seconds. The source checkout remains clean. No Python implementation or executable fixture was added to ezlocalai.

The full goal remains incomplete. Remaining LLM gaps include multimodal preprocessing/vision transitions, malformed API request branches, all native JSON/binding conversion errors, startup capacity-error mapping, no-local-model behavior, successful streaming callback normalization, cancellation and native slot lifetime, residency/placement and complete differential model-output checks. Router providers, local modality engines, model assets and all supported deployment configurations still need their separate completion audits. Covered model-profile policy does not establish provider or full inference parity. Strict parity enforcement must remain failing.

## Local chat callback normalization and API serialization

- Executed the complete unchanged `LLM._chat_stream` generator, extending the previous wrapper-only request audit. This revealed an additional boundary: `LLM.chat` preserves the caller's raw stream value, but the generator's inference thread copies the request and forces `stream=True`. Restored that engine-boundary behavior while retaining the independently verified wrapper policy. The previous section's statement about raw chat stream values applies only to wrapper entry, not the native call.
- Added a C++ stateful callback normalizer. It flattens callback lists, recursively selects source content/reasoning keys, prefers answer content over reasoning, preserves existing OpenAI-format chunks and parses string callbacks with the source raw-text fallback. Malformed non-text chunks are skipped. Text-free final chunks are held until completion, with the last such chunk winning; a stop chunk is synthesized only when the source would synthesize one.
- Empty streams now yield the source `empty_stream` object after any role-only chunks, followed by the API's DONE event. Diagnostics count dequeued chunks, retain only the source numeric usage/timing fields (including booleans and nonfinite floats), sort diagnostic keys and report source model/speculative metadata. The pinned binding's archive Versioneer metadata resolves to `2026.10.11549`, used in these compatibility diagnostics. Existing OpenAI chunk model/ID fields remain unchanged; wrapped chunks use the source model and generation-start timestamps.
- Callback errors retain preceding normalized chunks and emit the existing API streaming error without DONE. Errors nested inside callback lists follow ordinary chunk processing rather than top-level callback error detection. Python value formatting now also handles nonfinite error/reason values. API JSON spacing, ASCII escaping, raw-truthy/Pydantic-false list serialization and its failure cases follow the source.
- Compiled and exercised the unchanged pinned `callback_wrapper_dict` externally. Its JSON-decode failures construct an error locally but return true without invoking the callback. The native adapter now stops on those malformed records and Python's decimal integer conversion limit, then applies the generator's ordinary completion rules. A JSON string containing an oversized integer is instead skipped by the generator's outer ValueError handler. The pinned binding passes successful and error-status handler bodies through identical callbacks, so lazy chat normalization no longer substitutes HTTP-status errors for those callback results.
- Added 4,720 executed-reference cases: 2,200 helper cases and 2,520 full-generator transcripts with a controlled producer and frozen clock. Another 48 source API/binding cases run through managed C++ engines under both HTTP 200 and 500, giving 96 integration checks. They cover callback arrays, reasoning, final ordering, tool-only/empty output, falsey errors, JSON strings, malformed binding records, integer limits, duplicate keys, Unicode, diagnostics and serialization. Fixtures hash the source LLM/API files and record the binding revision/hash. The oracle and compiled extension remain outside the repository at `/tmp/ezlocalai_stream_oracle.py` and `/tmp/ezlocalai-stream-binding`; checked-in code/tests are C++ with TSV data.
- Updated prior response/fallback tests to expect normalized API serialization and actual assistant chunks. Unknown marker dictionaries are now correctly dropped instead of being passed through. A transient context-suite failure occurred in the first concurrent full run and passed on isolated retry; its failure output now includes the response/context and retains artifacts for diagnosis. The subsequent full run passes it without retry.

Verification: the C++ build and all 41 enabled CTest targets pass in one full run (65.05 seconds), including the new generator/managed API suite, pinned engine parser and actual-model CPU smoke. `git diff --check` and the first-party source/process-launch scans pass. The reference checkout remains clean at `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Strict enforcement still correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 source routes represented.

The full goal remains active and incomplete. This verifies covered stream contents, callback decisions and API serialization, not streaming transport parity. Local generation is still buffered before reaching the client. Incremental delivery, five-second keepalives, disconnect cancellation, bounded generator close, native generation completion and model/replica lease lifetime are the next connected gap. Complete native JSON/binding conversion and codec edges, multimodal preparation, startup capacity mapping, no-local-model branches, residency/placement, full differential model output, router providers, modality engines and deployment audits also remain open. No CUDA image or GPU inference was verified.

## Incremental local streaming, lazy ownership and cancellation leases

- Replaced the local chat buffering path with a C++ lazy generator and a separate native producer thread. Model preparation still occurs before returning the generator; native generation starts only on first iteration. The request queue passes the owned generator back immediately and transfers model/replica cleanup to it. Direct buffered callers collect the same generator, preserving the previous source-content oracle coverage without a second response implementation.
- The producer detects top-level callback errors before queueing, flattens batches into individual queued values and stops at the reference callback boundary when cancellation is requested. The consumer applies the existing source normalizer one queued value at a time. Its 100 ms empty-queue waits produce five-second keepalives, including before engine response headers arrive. Normal text does not reset the keepalive timer. Completion, error and pending-final behavior remain on the same path as the previously verified generator cases.
- Generator exhaustion or close releases accounting exactly once. Dropping an unstarted response releases its lease without invoking native inference. Close requests cancellation and waits up to five seconds. If a single-slot producer remains active, the lease survives generator destruction until native completion; multi-slot cleanup follows the source's immediate release after the bounded close. Background generation state is owned separately from the response object.
- Executed the unchanged `_chat_stream`, `_TrackedInferenceStream` and `finish_inference` with real threads and controlled native callback gates. Five captured lifecycle traces cover unstarted close, final-yield ownership, single/multiple-slot bounded cancellation and keepalive before any assistant text.
- Also ran the unchanged source API SSE generator through a real loopback ASGI server with its background close callback. The tested Starlette 0.49.3, AnyIO 4.11.0 and Uvicorn 0.40.0 stack does not abandon a pending synchronous generator read on disconnect. It first waits for that read to yield (the next keepalive in the controlled case), then starts the five-second close. Two captured concurrency traces exposed and corrected an initial native implementation that cancelled during the pending read. Native socket tests now verify both the intermediate busy state and distinct single/multiple-slot release after this full sequence.
- Added incremental local HTTP chunk framing, content-length completion and SSE record parsing across arbitrary read boundaries. Native API headers and first content arrive before generation finishes. Socket tests gate upstream headers/tail output, split HTTP and SSE boundaries independently, inspect busy-slot reporting, exercise concurrent admission and check upstream closure at the next callback after cancellation. The normalizer's existing 4,720 source cases and 96 managed API cases exercise this transport through both HTTP 200 and 500 responses.
- Owner shutdown can interrupt and join outstanding native work even after a generator's bounded close has returned and the response object has been discarded. Weak lifetime handles track the producer state, avoiding stale backend/server cleanup captures. Tests cover destruction during an upstream header wait, responses outliving their server, cancelled generators outliving their response, and disconnect before first iteration. Native HTTP shutdown now tracks and drains its client handlers; response writes tolerate peer closure without SIGPIPE.
- Added `llm_stream_lifecycle_reference` and source/ASGI TSV fixtures. External reference drivers are `/tmp/ezlocalai_stream_lifecycle_oracle.py` and `/tmp/ezlocalai_stream_asgi_oracle.py`; no Python implementation or executable fixture was added to the repository.
- An intermediate build mixed class layouts after a header edit occurred while compilation was running. GDB proved the test allocated 3,784 bytes while the linked constructor expected 3,920, explaining the stack corruption in three suites. Rebuilt every target with `--clean-first`; all three affected suites then passed. The stream timestamp test also exposed a real clock-source difference: C `time()` can lag Python's `time.time()` at a second rollover. A direct libc/Python experiment reproduced it; the test now uses the same system-clock seconds as the generator.

Verification: after the clean C++ rebuild and timestamp-test correction, all 42 enabled CTest targets pass in one full run (77.41 seconds). This includes both stream suites, the pinned engine parser and actual-model CPU generation. `git diff --check` and the first-party C++/process-launch scans pass. The reference checkout remains clean. Strict enforcement correctly exits 1 with 96 rows, 37 implemented, 59 partial and all 85 source routes represented.

The full goal remains active and incomplete. Covered local generator/lease behavior is now incremental and source-executed, but complete framework-version and backpressure behavior, external HTTP codec/framing/failure edges, native IPC cancellation scheduling and real GPU cancellation remain unverified. Native JSON/binding conversion extremes, multimodal preprocessing/vision transitions, startup capacity mapping, no-local-model branches, residency/placement, complete model-output comparisons, router providers, modality engines and deployment configurations still need completion audits. Strict enforcement must continue failing; no CUDA image or GPU inference was verified.


## Queued text/image input preparation and vision descriptions

- Added a separate C++ `pipeline_text_result` boundary for queued local requests. Direct `LLM.chat`/completion wrapper calls retain their existing behavior. The managed API now flattens content lists, concatenates string/text parts without separators, ignores unsupported scalar parts, and collects images from every message before loading the model. Images are placed on the final message when the selected engine supports vision. Completion requests also preprocess a supplied messages field, matching the source.
- Preserved input-error timing. Empty message lists, invalid text concatenation and missing final content usually fail before model load; a previously collected image can defer the final-content lookup until after load. These failures produce the API's generic HTTP 500 and bypass post-inference fallback. Managed tests verify both phases and successful reuse after failure. Existing inference/lifecycle tests that used empty API message lists now use a valid empty-content user message; direct wrapper fixtures remain unchanged.
- Implemented direct-image URL/MIME policy, data URL preservation, malformed URL errors, WebP/GIF data conversion, and remote WebP/GIF/BMP/TIFF/AVIF conversion. Remote HTTP failures omit the image. Direct conversion failures retain the original bytes and MIME, while description conversion failures omit the image. The two paths use their distinct source User-Agent strings. Base64 handling preserves the source's ignored characters, padding and non-ASCII behavior. Nested and escaped content-type keys/values are scoped correctly, and duplicate object keys are collapsed before mutation.
- Added cached vision-model discovery and the image-description path for non-vision requests. It loads the selected vision engine with the 16,384 context request, prepares images, sends the exact description prompt with max_tokens 2048 and temperature 0.3, and enriches the captured primary model's final message. Description calls use wrapper defaults without applying normal request model profiles. The covered managed test uses two local fixture models and checks both engine payloads, context, HTTP headers and child cleanup.
- Request preparation merges the constructor's source stop list when user stop is truthy. The wrapper still drops stop before local generation. On local inference failure, post-inference fallback now receives the prepared messages rather than the original content list. A managed loopback fallback test verifies this connected behavior.
- Captured 3,399 cases by executing the unchanged `_get_response_internal` prefix through image handling and the full `_describe_images_with_vision_model` method. The oracle controls model loading, HTTP and Pillow adapters and compares complete data, exception class/message and call order. Default stop tokens are extracted from the source LLM constructor. Fixtures hash both source files; inputs include nested fields, escaped type strings and duplicate JSON keys. `/tmp/ezlocalai_input_oracle.py` is the external generator. Only C++ implementation/tests and TSV data were added to the repository.

At this checkpoint, requests containing audio/video enrichment or audio_format retained the prior path, including audio_format dispatch before TEXT_SERVER/fallback. The following audio work replaces that part. Native video-input extraction and complete STT resource ownership remain incomplete. The image policy oracle uses controlled codec adapters: exact Pillow output bytes and all color modes, requests transport/timeouts, vision cache discovery details, swap/resident/context transitions and real vision-model output remain unverified. Strict parity labels remain partial, and no GPU or CUDA image validation is claimed.

Verification: all 43 enabled CTest targets pass in one final run (81.15 seconds), including the 3,399-case input suite, managed vision/fallback tests, stream lifecycle, pinned-engine contract and real-model CPU smoke test. The build remains C++-only with CMAKE_C_COMPILER=/nonexistent-c-compiler. git diff --check, first-party language/process-launch scans and child-process cleanup checks pass; the reference checkout remains clean. Strict enforcement still correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 source routes represented. The full parity goal remains active and incomplete.

## Queued audio input and nested STT admission

- Inline `audio_url` items now run through the queued Pipes preparation phase before LLM loading. They preserve source content-item precedence, Python membership/type failures, remote fetch behavior and the source's data-URL parsing quirk: the extension is taken from the payload after the comma. Remote audio uses the requests User-Agent and response body even on an HTTP error, with no total timeout. Transcripts prepend `Transcribed Audio: ...\n\n` to text already collected from that message.
- Top-level `audio_format` now executes after LLM loading and stop merging. It replaces the final chat content or completion prompt with the raw transcript. TEXT_SERVER dispatch and precheck fallback run first and receive the original audio fields. Preparation errors produce the generic API HTTP 500 before engine generation and post-inference fallback. Managed checks cover invalid base64, null content, invalid format types, empty messages and failing STT processes at both load phases.
- Added a C++ input adapter matching the unchanged STT format/base64 prefix: falsey-format default, MIME splitting, extension aliases, strict ASCII string conversion, Python's permissive base64 decoding/error messages and embedded-NUL timing. The adapter writes decoded bytes directly to a UUID-hex filename. Segment texts are joined without trimming or inserting separators. A C++ executable fixture inspects actual input bytes, filenames and process arguments and verifies that the exact joined transcript reaches llama-server.
- Captured STT concurrency at server construction and added queued local admission. Managed two-request checks cover capacities one and two, a later environment change, successful slot release and failure recovery. This is only the nested audio admission portion of the reference voice guard. The standalone audio routes, shared STT/TTS handoff state, persistent/transient STT model pool, ResourceManager registration/counts and source exception-leak behavior still need a full lifecycle port.
- Preserved the reference's nested handoff timeout. With ordinary models and the default voice policy, `get_response` already owns an LLM lease when STT tries to wait for LLM idleness. The source therefore times out before loading STT. C++ follows that ordering and clears its unavailable flag after failure. Source guard fixtures cover zero, positive, negative, NaN, negative infinity, invalid and empty timeout values and generic/service precedence. Managed requests verify the API error, load phase, 100 ms polling and a successful subsequent ordinary request. Tests that require successful nested audio explicitly disable STT handoff.
- `llm_audio_reference` compares 1,582 unchanged Pipes-prefix cases, 1,280 STT-prefix format/base64 cases, 9,720 voice-policy cases and 8 unchanged voice-guard cases. The TSV files hash Pipes, STT, LLM, ModelSettings and the capacity classifier. The STT prefix executes actual file I/O; model, HTTP and transcription boundaries are controlled. External generators are `/tmp/ezlocalai_audio_oracle.py` and `/tmp/ezlocalai_audio_guard_oracle.py`. The repository additions are C++ implementation/tests/process fixtures and TSV data.

This remains a partial audio port. The current native transcription adapter invokes whisper.cpp; the source uses faster-whisper/CTranslate2 with different inference, VAD, language/task, segment and diarization behavior. Passing these tests establishes preparation, dispatch and the covered admission behavior, not speech-model equivalence. Voice model residency/resource ownership, standalone voice handoff, video input, full requests transport semantics and real multimodal output parity remain open. No GPU or CUDA image validation is claimed, and strict parity enforcement must continue failing.

Verification: all 44 enabled CTest targets pass in one final run (87.95 seconds), including 12,590 source audio cases, managed audio process/HTTP checks, native HTTP regression, text/image preparation, streaming lifecycle, the pinned-engine contract and the real-model CPU smoke test. The separate audio suite also passes (13.00 seconds). The build remains C++-only with CMAKE_C_COMPILER=/nonexistent-c-compiler; first-party language/process-launch scans, git diff --check and fixture child cleanup checks pass. The reference checkout remains clean at 0f7e9f45ac0d67ea806ee19874f98bb8f47c7544. Strict enforcement correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 source routes represented. The full parity goal remains active and incomplete.

## Queued video input, frame sampling and transcript headers

- Removed the video bypass around queued input preparation. Video items now follow the source's text/image/video/audio precedence, video URL/object/fallback rules, falsey-input handling and environment parsing before model load. The source parses `VIDEO_FPS` but does not use it to control sampling; the native implementation preserves that behavior. Unicode digits and whitespace now follow Python float conversion for VIDEO_FPS and STT handoff timeouts. Two additional source guard fixtures cover Unicode zero and invalid ASCII separator whitespace, bringing audio coverage to 12,592 cases.
- Added a native OpenCV video reader and scene detector. It samples grayscale frame differences every half second, resizes scene probes to 160x90, uses the strict mean-difference threshold of 30, reserves half the frame budget for scenes, fills the remaining budget uniformly, rounds timestamps to two decimals and deduplicates decoded frame positions. The zero scene-budget division failure for long videos is preserved as an empty-frame result with any already-extracted audio still owned by the caller.
- Video data URLs are decoded into temporary files; HTTP(S) input uses the source Mozilla User-Agent, follows redirects and rejects error statuses. Local paths are rejected. Video audio extraction invokes the same FFmpeg PCM s16le/16 kHz/mono command with a 120-second timeout and omits files smaller than 1,000 bytes. Video/audio temporary files are removed at their corresponding source boundaries.
- Frames use RGB conversion, Pillow-compatible Lanczos resizing when the longest side exceeds 1,024 pixels, and JPEG quality 85. Added reusable C++ RGB resize/JPEG encoding operations. Extracted frames join the existing all-message image collection and vision/description path. Timestamp headers and optional segment/speaker transcript blocks prepend the message text before the LLM is loaded. Video transcript reading, STT and formatting errors retain the frames and header; audio cleanup errors still propagate. A source-default STT handoff timeout is therefore nonfatal for video requests.
- The nested STT adapter can now return segment data for video headers. This adapts the current whisper.cpp output; it does not establish equivalence with the reference faster-whisper/CTranslate2 model or diarization. Model pool ownership, shared voice residency, resource counts and exception-leak behavior remain incomplete as documented above.
- Added 776 source-prefix request cases, 492 unchanged transcript-formatter cases and 1,440 cases executing the unchanged frame-selection statements. The 25 real-media cases execute the full unchanged source extractor using OpenCV 4.13.0, Pillow 11.3.0 and FFmpeg. Native results compare exact JPEG and WAV SHA-256 values, frame timestamps and durations for generated lossless/H.264 clips, scene changes, resizing, audio, zero/one-frame budgets and disabled scene detection. The source and fixtures are hash checked; `/tmp/ezlocalai_video_oracle.py` is the external generator. The repository contains C++ tests and generated TSV/media fixtures, with no Python implementation or test runner.
- Managed C++ process and loopback HTTP checks verify that video STT starts before LLM loading, frames/transcripts reach the selected model, transcription/process/handoff failures remain nonfatal, malformed URL types fail before model load, redirects and source headers work, HTTP errors omit the video, and subsequent requests succeed. Tests also check child reaping and video/audio temporary-file cleanup.
- Added OpenCV 4.8+ as a native build dependency. The tested local build uses OpenCV 4.13.0 compiled from the official archive with SHA-256 `1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d`. The CUDA Dockerfile now builds that same release and installs FFmpeg development dependencies. This recipe has not been built or GPU tested in this verification pass.

The previous video-input bypass is closed, but full parity remains unproven. The media comparisons cover the captured clips and environments; arbitrary codecs, decoder/build differences, requests streaming/timeout behavior, damaged streams and every pixel mode still require broader comparison. Real vision-model output, vision residency/context transitions, STT engine/lifecycle and other partial manifest areas remain open. Strict labels stay partial.

Verification: all 45 enabled CTest targets pass in one final run (96.47 seconds), including 2,733 video reference/media cases, managed process/HTTP/cleanup checks, the expanded audio guard cases, native HTTP regression, image processing, stream lifecycle, the pinned-engine contract and the real-model CPU smoke test. The separate video run passed in 8.64 seconds. The OpenCV-linked first-party build remains C++-only with CMAKE_C_COMPILER=/nonexistent-c-compiler. git diff --check, first-party language/process-launch scans and child cleanup checks pass; the reference checkout remains clean at 0f7e9f45ac0d67ea806ee19874f98bb8f47c7544. Strict enforcement correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 source routes represented. The full parity goal remains active and incomplete.

## Native Whisper features and CTranslate2 generation kernel

- Added C++ audio decoding through libavformat/libavcodec/libswresample. It selects the first audio stream, decodes and resamples to signed 16-bit mono, then converts samples to float with the source's 32768 divisor. Five captured formats (mono PCM WAV, 44.1 kHz stereo 24-bit WAV, FLAC, 48 kHz stereo MP3 and AAC) match the actual faster-whisper/PyAV decoded samples exactly. The tested native FFmpeg library versions match PyAV 12.3.0: avutil 58.29.100, avcodec 60.31.102, avformat 60.16.100 and swresample 4.12.100. Damaged streams, changing stream parameters and all PyAV FIFO/error semantics remain unverified.
- Added native log-Mel extraction, centered reflection padding, periodic Hann windows, Slaney mel filters, OpenBLAS projection, logarithmic normalization and feature-window padding. The reference has two pinned preprocessing profiles: faster-whisper 1.0.3 in requirements.txt and 1.2.0 in cuda-requirements.txt. The implementation preserves their different waveform padding, Hann precision and mel-filter rounding. Sixteen cases per profile cover 80/128 mel bands and input lengths 0, 1, 159, 160, 401, 1600, 1999 and 32000. Tests compare complete float arrays and window layout, not only shape or aggregate statistics.
- Found an observable generation difference from using standard hypotf for NumPy's complex magnitude. Although feature differences were initially very small, one real-model result changed punctuation and the final timestamp token. Ported NumPy's magnitude arithmetic, including its fused operation on the tested FMA-capable host. The captured native and source inputs now produce identical decoder tokens. The final feature comparison still has a maximum absolute error of 1.19209e-7, with a 2e-7 test tolerance; complete bitwise feature identity is not established. FFT implementations, CPU dispatch and dependency versions require broader verification. Added the upstream faster-whisper and NumPy license notices.
- Added a C++ adapter for the actual CTranslate2 Whisper model and generation options. It accepts model inputs and token prompts, executes native generation, returns token IDs, scores and no-speech probability, and reports model/shape errors. Its plain C interface isolates the old libstdc++ ABI used by the tested library from the application's normal C++ ABI. No Python runtime is launched or linked. This is a generation kernel, not a replacement implementation of faster-whisper's complete transcription controller.
- Verified against CTranslate2 4.6.1 headers at commit 785c7e0d6299ae17dec66a25a9ea52b11219f11e and the matching distributed native shared library, SHA-256 0532bdc6f1181d1353e60c2aa365a44a2bcd0de1355eed4bf963e5e278615d25. The real CPU int8 model is Systran/faster-whisper-tiny.en at revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba; model.bin SHA-256 is 1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a and is checked by the test. This small verification model does not change the application's configured model default.
- Captured eight actual-model cases from the unchanged faster-whisper feature extractors and CTranslate2 binding: beam sizes 1/3, timestamp suppression and an initial prompt, for both preprocessing profiles. Every case runs twice in C++ with captured and natively computed features, comparing exact token IDs and bounded score/no-speech differences. The test also checks rejection of an incorrect mel-band count without a result callback. Compressed audio/features, prompts, outputs and provenance are stored in tests/fixtures/whisper_kernel_assets; model weights stay outside the repository. The external oracle is /tmp/ezlocalai_whisper_feature_oracle.py. Repository implementation and test runners are C++.
- Added OpenBLAS and the native FFmpeg audio libraries to build dependencies and the CUDA Dockerfile. CTranslate2 remains an explicit optional native dependency, with reproducible model/test configuration documented in README.md. The container recipe, a separately source-built CTranslate2 binary and GPU inference have not been validated in this pass.

The transcription API still uses whisper.cpp. Connecting the new kernel requires faster-whisper's VAD, tokenizer, language/task selection, timestamp segmentation, context/fallback controller, word alignment and model lifecycle. Diarization and complete deployment/version equivalence also remain open. These tests establish native feature and decoder progress on captured inputs; they do not establish complete STT or service parity. All existing partial manifest labels remain partial.

Verification: all 47 enabled CTest targets pass in one full run (111.28 seconds), including the 32 feature cases, five exact decoded-audio comparisons, eight real CTranslate2 cases with both source/native features, existing media and HTTP tests, streaming lifecycle, pinned llama.cpp contract and real-model CPU smoke. The first-party build remains C++-only with CMAKE_C_COMPILER=/nonexistent-c-compiler. The new kernel's shared-library dependencies resolve without libpython, and the new sources/tests contain no Python or subprocess invocation. git diff --check passes and the reference checkout remains clean at 0f7e9f45ac0d67ea806ee19874f98bb8f47c7544. Strict enforcement correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. The full goal remains active and incomplete.

## Native Silero VAD and combined speech decoder path

- Added C++ implementations of faster-whisper's speech/silence state machine, maximum-duration splitting, minimum-speech filtering, padding and gap sharing. The two reference deployment profiles retain their different defaults and comparison behavior: faster-whisper 1.0.3 uses a 250 ms minimum speech duration and an unclamped negative threshold, whereas 1.2.0 defaults to zero minimum speech and supports an explicit negative threshold with a 0.01 floor for its derived default. Scalar-versus-array NumPy threshold rounding is preserved at the model-output boundary. The source STT configuration of 500 ms minimum silence and 200 ms padding is exercised separately from library defaults.
- Added source-compatible audio chunk collection and timestamp mapping. The modern maximum-duration collector preserves the source's omission of the first segment from a newly split group's metadata, including its empty leading group for a first oversized chunk. Legacy collection concatenates slices directly. Mapping preserves right-biased boundary selection, the modern end-time exception at exact chunk ends, decimal rounding and explicit/negative chunk indices. Python slice clipping is retained when collecting audio. General non-default numeric extremes and timestamp precision outside the tested 0/2/3/6 settings are not established.
- Added an optional native ONNX Runtime adapter using C++ RAII. The modern profile runs the actual Silero v5 encoder in batches of at most 10,000 windows, then runs its recurrent decoder in order. The legacy profile runs the original combined ONNX model with explicit 16 kHz sample-rate input. Both preserve the reference single-thread session configuration; the modern profile disables the CPU memory arena. Recurrent state is local to each request and resets on every invocation.
- Preserved a subtle source preprocessing behavior: modern VAD always pads another 512-sample window when input is already aligned. Its NumPy context view also zeros the last 64 samples of the final padded window before assembling encoder inputs, which can remove real samples when the final partial window contains more than 448 samples. The C++ adapter reproduces that model input without changing the caller's audio. Legacy VAD uses only the windows necessary to cover actual samples and does not apply this mutation.
- Executed the unchanged upstream functions to capture 1,120 segmentation cases, 90 chunk-collection cases and 8,256 timestamp-map cases. These cover threshold edges, short speech, silence bridging, maximum-duration cuts, padding, empty input, negative/clipped slices, original-time boundaries and index errors. The model is replaced only at the probability boundary for these policy tests; source code remains unchanged. Source hashes and model identities are recorded in the fixtures.
- Added 29 real Silero model cases using NumPy 1.26.4 and ONNX Runtime 1.24.1 CPUExecutionProvider. Native probability arrays are bit-for-bit identical in these captures (maximum absolute difference zero). Native speech chunk boundaries also match both default and STT settings. Cases include generated short signals around 448/512-sample boundaries, silence, real JFK speech, inserted silence and a 10,001-window case crossing the encoder batch limit. Repeated calls verify state reset and unchanged caller input. Tests verify the three ONNX model hashes before loading; provenance is recorded in tests/fixtures/whisper_vad_assets/README.md.
- Added a reusable native filter operation and 12 combined VAD/filter/log-Mel/CTranslate2 cases. For both preprocessing profiles, speech, speech with inserted silence and all-silence audio are decoded with beam sizes 1 and 3. Exact filtered audio hashes, speech boundaries, feature frame counts and decoder token IDs match the captured source path; scores and no-speech probabilities are checked within tolerance. These cases use the same pinned tiny.en model as the earlier kernel test. All-silence raw generation still produces tokens, as in the source kernel; applying the transcription controller's no-speech policy remains necessary.
- ONNX Runtime headers and its native shared library are explicit optional CMake dependencies. The verified library is version 1.24.1 with SHA-256 94ac52f650e195b784587e1cb7d25b4bf074048b50fa9a7127795b2ed34db22e. The adapter and test executables do not launch or link Python. Models, dependency headers and reference capture programs remain outside the repository; first-party implementation/tests are C++, and compressed arrays/TSV files are test data. External captures are /tmp/ezlocalai_whisper_vad_oracle.py and /tmp/ezlocalai_whisper_speech_oracle.py.

This advances the actual faster-whisper replacement path, but does not replace the API's existing whisper.cpp adapter yet. Tokenization, complete standard/batched transcription controllers, no-speech/temperature fallback, timestamp segmentation, word alignment, language handling, diarization and model lifecycle must still be ported and integrated. Native model acquisition/packaging, GPU behavior and equivalence across supported dependency configurations remain unverified. Strict parity labels remain partial.

Verification: all 50 enabled CTest targets pass in one full run (108.70 seconds), including 9,466 VAD policy cases, 29 actual Silero model cases, 12 combined speech decoder cases and the prior HTTP/media/streaming/native-engine regressions. The isolated VAD suites passed in 7.50 seconds; the isolated combined decoder test passed in 17.16 seconds. The full run again measured zero maximum VAD probability difference. The CXX-only build retains CMAKE_C_COMPILER=/nonexistent-c-compiler; the new executables resolve their native dependencies without libpython, and first-party source/process-launch scans and git diff --check pass. The reference checkout remains clean. Strict enforcement still correctly exits 1 with 96 rows, 37 implemented, 59 partial and all 85 source routes represented. The full parity goal remains active and incomplete.

## Native Whisper tokenization, prompts and decoded speech

- Added a C++ ByteLevel/BPE tokenizer for the actual Whisper tokenizer.json model files. It loads vocabulary, ranked merges and added tokens, applies the source's two-stage special/normalized token matching, merges pairs by rank with the leftmost tie rule, decodes byte fragments with UTF-8 replacement semantics and skips special tokens at the source boundaries. The implementation has no Rust or Python runtime dependency. Nonstandard custom normalizers, stochastic BPE, unknown-token fallback and added-token strip/word flags remain unsupported; ordinary pinned Whisper model configurations are the verified scope.
- Implemented the exact ByteLevel splitting pattern in C++. Captured Unicode letter, number and whitespace classes from tokenizers 0.22.1 across every Unicode scalar value and stored the result as generated C++ tables: 677 letter ranges, 144 number ranges and 10 whitespace ranges. This avoids using a different system Unicode database. Splitting preserves lowercase contractions, the optional leading ASCII space and greedy-whitespace backtracking. Prompt stripping and word punctuation handling use the separate existing Python-compatible whitespace behavior.
- Added English/multilingual task and language configuration, start/end/previous-context tokens, timestamp-aware decoding and the faster-whisper Unicode/space/language word splitters. Preserved special-token removal, unknown timestamp IDs, incomplete UTF-8 grouping, replacement characters in the full decoded text, empty decoded special-token groups and language-specific splitting for zh/ja/th/lo/my/yue. Constructor validation follows the source's task-first order and accepted language-code list, including ignored task/language inputs for English-only models.
- Ported prompt construction with previous-context tails, hotwords, prefix insertion, whitespace handling, context limits and timestamp suppression. Preserved the profiles' distinct suppressed-token helpers: 1.0.3 passes through null, empty and lists containing -1; 1.2.0 expands -1 into tokenizer-derived non-speech symbols, removes other negative entries in that branch and appends the reserved tokens. The modern None input raises the source's membership TypeError rather than taking its unreachable later None branch. The C++ helper reports that failure through its exception boundary.
- Captured 15,055 non-metadata fixture rows: 1,741 pre-tokenizer cases, 3,482 full encodings, 800 multi-token decodes, 7,100 word splits, 1,260 task/language configurations, 640 prompts, 28 suppression results, two non-speech lists and two exhaustive decode arrays. The arrays compare ordinary/timestamp decoding for all 102,239 single-token inputs across both vocabularies, including IDs beyond each vocabulary. Cases include mixed scripts, controls/NUL, combining marks, emoji, whitespace, special tokens, long prompts and deterministic random text/token sequences. Both wrapper versions share the captured encode/decode/word behavior; prompt and suppression captures execute the unchanged methods from each version separately.
- Bundled compressed tokenizer JSON as test data, with SHA-256 verification before loading. English tokenizer: Systran/faster-whisper-tiny.en revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba, SHA-256 929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df. Multilingual tokenizer: Systran/faster-whisper-large-v3 revision edaa852ec7e145841d8ffdb056a99866b5f0a478, SHA-256 6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca. No multilingual model weights were required for tokenizer verification. Provenance is in tests/fixtures/whisper_tokenizer_assets/README.md; external captures are /tmp/ezlocalai_whisper_tokenizer_oracle.py and /tmp/ezlocalai_whisper_unicode_oracle.py.
- Expanded the combined native speech test from 12 to 16 cases, adding initial prompts and using the C++ tokenizer for prompt creation, suppression and final text. The previous legacy feature/VAD comparison supplied the modern explicit suppression list to both decoder profiles; the updated capture executes each version's actual helper, including legacy CTranslate2 default suppression. Exact filtered audio, token sequences and decoded text match the source for all 16 cases. Scores/no-speech probabilities retain the established tolerance. Both model weights and tokenizer bytes are checked before execution.
- Added nlohmann-json 3.11+ as a C++ header-only build dependency and the corresponding CUDA-image package. The verified local version is 3.11.3, official release archive SHA-256 d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d. Added the tokenizers Apache-2.0 notice alongside the existing faster-whisper notice. Container execution is still unverified.

The transcription API still uses the existing whisper.cpp adapter. Full standard/batched transcription control, no-speech and temperature fallback, timestamp segmentation, language detection, word alignment, diarization and model lifecycle/integration remain incomplete. The new tokenizer covers the pinned standard Whisper configurations; arbitrary tokenizer models and other dependency versions require further verification. These results do not establish whole-service parity, and strict manifest labels remain partial.

Verification: all 51 enabled CTest targets pass in one full run (111.69 seconds). The new tokenizer test passed separately in 8.55 seconds, and the expanded 16-case actual speech decoder test passed separately in 20.40 seconds. The full suite includes both tests, native VAD/feature/kernel comparisons and the existing HTTP/media/streaming/llama.cpp regressions. The build remains CXX-only with CMAKE_C_COMPILER=/nonexistent-c-compiler. Native dependency inspection shows no libpython or tokenizers runtime, first-party language/process-launch scans and git diff --check pass, and the reference checkout remains clean. Strict enforcement correctly remains failing: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. The full parity goal remains active and incomplete.

## Native standard Whisper transcription controller

- Added C++ standard transcription control for both faster-whisper deployment profiles: feature-window iteration, timestamp-token segment splitting, clipped intervals, initial string/token prompts, prefix/hotwords, previous-context accumulation/reset, temperature fallback, compression/log-probability thresholds and no-speech rejection. Calls preserve explicit empty suppression lists rather than silently restoring CTranslate2 defaults. Fallback selection retains the source's final attempted temperature even when it selects an earlier hypothesis.
- Extended the native CTranslate2 boundary with encoded-state ownership, model metadata, automatic language detection and alignment. Encoded states and model allocations remain on the same side of the libstdc++ ABI boundary. The C++ engine adapter owns them through shared handles, supports reuse across decoder attempts, and copies generated tokens, language probabilities and alignment results before callback data expires. No Python or tokenizers runtime is linked or invoked.
- Added native word alignment processing: token-to-word grouping, mean probabilities, alignment jumps, median-duration limits, sentence/segment boundary adjustments, punctuation merging, anomalous-word silence handling and restoration of original VAD timestamps. Automatic language detection handles threshold/majority selection; modern per-window multilingual decoding updates its prompts. The legacy path deliberately reuses the final language-detection encoder output for its first transcription window and reports the post-iteration seek position, matching observable source behavior.
- Added 720 captured controller cases executing unchanged faster-whisper methods with only the model boundary substituted. They compare complete encoder/decoder/alignment/language call traces, padded-feature hashes, prompt tokens, generation parameters, segment/word outputs and errors. This includes fallback thresholds/exhaustion, multiple windows/clips, context resets, forced anomalous alignments, token-limit failures and both deployment profiles. The external oracle is `/tmp/ezlocalai_whisper_transcription_oracle.py`; repository tests and the application implementation are C++.
- Added 32 complete actual-model transcriptions across both profiles, using the existing tiny.en model and the multilingual `Systran/faster-whisper-tiny` model at revision `d90ca5fe260221311c53c58e660288d3deb8d356`. The multilingual weight SHA-256 is `dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1`. Tests verify weights/configuration/tokenizer/vocabulary and VAD identities before loading. Cases cover silence, empty audio, VAD, audio beyond one window, beam sizes 1/3, initial prompts, clips, context switches, word timestamps, translation and automatic language detection. Exact tokens/text/segment IDs/seek positions/language decisions match; timestamps allow 1e-12 for floating arithmetic and model probabilities/scores retain the established 1e-4 tolerance. The legacy long-audio language-detection reuse and empty-input failure are captured explicitly.

The standard controller passes its captured policy and real-model cases, but the service still uses the earlier whisper.cpp adapter. Batched inference, service integration/model acquisition/lifecycle, GPU availability/error fallback, speaker diarization and production dependency packaging remain incomplete. Arbitrary model/configuration/dependency combinations, all invalid-option errors, CUDA execution and complete bitwise numerical identity remain unverified. The manifest stays partial; this is not a claim of whole-service parity.

Verification: all 53 enabled CTest targets pass in one full run (161.84 seconds). The 720-case controller suite passed separately in 5.30 seconds and the 32-case actual-model suite passed separately in 80.71 seconds. The full run also covers the existing feature/VAD/tokenizer/kernel checks, HTTP/media/streaming regressions, pinned llama.cpp engine contract and real-model smoke test. The build remains CXX-only with CMAKE_C_COMPILER=/nonexistent-c-compiler; the new native model test has no libpython/tokenizers runtime dependency or unresolved shared libraries. First-party language/process-launch scans, whitespace checks and fixture child cleanup checks pass. The reference checkout remains clean at 0f7e9f45ac0d67ea806ee19874f98bb8f47c7544. Strict enforcement correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. The full parity goal remains active and incomplete.

## Native batched Whisper transcription

- Added C++ faster-whisper 1.2.0 batched transcription with native CTranslate2 batch encoding, generation, per-item language detection and alignment. Model/encoded allocations remain owned across the libstdc++ ABI boundary; all outputs are copied before callbacks return. The implementation submits whole batches to the native engine rather than issuing independent single-item decoder calls. Version 1.0.3 does not provide this pipeline; its service fallback behavior remains part of the pending runtime integration.
- Added batch VAD defaults and dictionary overrides, maximum-duration chunk grouping, explicit clips, concatenated-feature language detection, fixed-width feature packing and original timestamp restoration. Batch decoding preserves source differences from standard decoding: initial prompts are encoded without added stripping/spacing; prefix/context/fallback thresholds are ignored; only the first temperature is used; empty text and zero-duration segments are retained; compression is computed from each unstripped segment; and word alignment carries the prior speech timestamp across batches. Successful exhaustion resets that state, while a mid-batch failure preserves it for the next call.
- Refactored standard word-alignment postprocessing for reuse by batched results. Corrected the earlier rejection of null suppression in both controllers: CTranslate2 4.6.1 accepts None and clears its suppression vector. Standard and batch policy fixtures now capture this native-boundary conversion, and two actual-model null-suppression cases exercise it. The modern standalone suppression helper still has its separately captured None membership error; the controllers call that helper only for a truthy list, as the source does.
- Added 300 unchanged-source batch policy cases, comprising 342 invocations, 321 recorded generation calls and 116 alignment calls. Tests compare complete batch input hashes, prompts, generation parameters, language/alignment events, output segments/errors and persistent state after an injected failure followed by recovery. The standard 720-case policy capture was refreshed to include null suppression. External capture programs remain under /tmp; repository code and runners are C++.
- Added 50 full actual-model batch transcriptions: 25 cases each with OpenBLAS thread counts 1 and 8. They cover the pinned English/multilingual tiny models, VAD defaults/overrides, six-/ten-/thirty-second grouping, multiple batch sizes, clips, prompts, language/translation, words, multilingual batches, silence, empty input, null suppression and errors. Weight/configuration/tokenizer/vocabulary/VAD identities and the native BLAS thread count are checked before running. Output tokens/text/IDs/seek/language decisions are exact, timestamps retain the 1e-12 floating-arithmetic allowance, and model metrics retain the existing 1e-4 tolerance.
- Found and isolated a reference/test environment mismatch: the initial capture used eight OpenBLAS threads while the test used one. Feature differences of at most 1.1920929e-7 changed the int8 model's first-case average log probability from -0.13064067363739013 to -0.12749824047088623. With matching eight-thread settings, diagnostic encoder features and scores were bitwise identical. Added separate one-thread source captures and tests rather than widening tolerances. Arbitrary hardware, BLAS builds and thread configurations remain unverified.

The new controller is not yet connected to the speech API, which still uses whisper.cpp. Model acquisition/lifecycle, GPU availability and error fallback, API integration/output/error behavior, speaker diarization and production packaging remain incomplete. Strict manifest labels remain partial, and whole-service 1:1 parity remains unproven.

Verification: all 56 enabled CTest targets pass in one full run (250.28 seconds), including the refreshed 720 standard policy cases, 300 batched cases/342 invocations, 32 standard actual-model transcriptions and 50 batched actual-model transcriptions across both BLAS thread settings. The full run also verifies the existing tokenizer/features/VAD/kernel paths, native HTTP/media/lifecycle regressions, pinned llama.cpp contract and real-model smoke test. The build remains CXX-only with CMAKE_C_COMPILER=/nonexistent-c-compiler. The batched model binary has no libpython/tokenizers runtime dependency or unresolved shared libraries. First-party language scans, whitespace checks, fixture child cleanup and reference-checkout cleanliness pass. Strict enforcement correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. The full parity goal remains active and incomplete.

## Native STT wrapper lifecycle, fallback and output

- Added a C++ STT runtime wrapper implementing the source's CPU-thread policy, 1,000 MB GPU threshold, cuDNN probe ordering, constructor fallback, inference fallback and close behavior through injectable native system/model boundaries. GPU inference failures discard the old handles and retry standard transcription on CPU, including when the failed request used batching. Failed reloads preserve the source's absent-versus-None handle state, so subsequent calls report the corresponding attribute error. Repeated close clears handles once while preserving the source's collection/cache operations and selective exception suppression.
- Split the existing source-compatible audio-format and base64 helpers so the wrapper preserves UUID/path construction before base64 decoding. The wrapper writes to the caller's outputs directory, forwards source transcription/VAD/beam/context/granularity arguments, and preserves each error/cleanup path. It does not silently remove files on paths where the source leaves them behind, including retry or postprocessing failures. Plain text retains leading/trailing whitespace; segmented output trims the aggregate and each segment/word, omits word probabilities, filters empty words and follows the source language fallback. A supplied diarization operation feeds source-compatible speaker-attributed formatting.
- Added `ezlocalai_stt_whisper`, which connects the wrapper to native FFmpeg decoding, the C++ tokenizer/controllers, CTranslate2 and ONNX Runtime through supplied model-resolution and platform/audio operations. Model loading receives the source `models` cache root. VAD sessions are initialized lazily and retained across wrapper instances, matching the source process cache. The legacy profile aliases the batch handle to the standard model and preserves the unsupported `batch_size` error. Invalid audio is normalized to the source ValueError. The adapter uses no Python runtime.
- Added 800 cases executing unchanged `STT.py` methods/helpers: 240 initialization/close cases and 560 transcription cases with subsequent calls. They compare results, exception category/message, state, remaining files, live model handles at collection points and complete boundary-call order. The capture contains 710 inference calls, 61 cases that load a second model, and 90 outcomes with source-retained files. Model/system/filesystem/audio/diarization boundaries are substituted. The source SHA-256 is checked before execution; external generation is `/tmp/ezlocalai_stt_runtime_oracle.py`.
- Added nine comparisons with real Linux filesystem operations, covering successful cleanup, missing/not-directory parents, directories at the audio path, `/dev/full`, dangling symlinks, a directory substituted during inference, NUL extensions and invalid base64. Corrected buffered write/close error reporting and used unlink for cleanup so unexpected directories produce the source error and remain intact. These compare exception category/message and remaining file type; temporary roots alone are normalized. External generation is `/tmp/ezlocalai_stt_runtime_io_oracle.py`.
- Added 30 actual-model wrapper calls, 15 per faster-whisper profile, covering base64 audio, real file writes/removal, native audio decoding/inference, text and JSON formatting, words, prompts, translation/language, long standard/batched audio, silence/empty WAV, invalid audio and invalid batching. Model/audio/VAD identities and thread settings are checked. Microphone initialization and CUDA availability remain substituted in these comparisons; CPU inference uses four threads and OpenBLAS uses one. The tests compare exact returned text/JSON, timestamp arithmetic within 1e-12, errors, output-file cleanup and explicit handle close. External generation is `/tmp/ezlocalai_stt_runtime_model_oracle.py`.

The speech API still uses its earlier whisper.cpp adapter. The new runtime currently requires supplied model resolution and platform/audio operations; native GPU/cuDNN probing, microphone initialization, Hub model acquisition, service pool/lease/resource integration, API behavior and native diarization remain incomplete. Arbitrary invalid backend options, malformed model configurations, codecs and CUDA execution require further verification. Source-only wrapper decisions and local-model CPU execution passing these cases do not establish whole-service 1:1 parity. Strict manifest labels remain partial.

Model-acquisition follow-up must use the reference requirements' `huggingface_hub==1.5.0` for both CPU and CUDA/ROCm profiles. The existing Hub local-file traces documented above use 1.28.0 and do not establish the pinned snapshot-download behavior. Source aliases also differ between faster-whisper 1.0.3 and 1.2.0; the legacy profile lacks the default `large-v3-turbo` alias. Mutable references, offline retries, missing-tokenizer fallback and preprocessor configuration loading remain to be ported and verified before connecting this wrapper to the service.

The first full run passed 57 of 58 targets; the LLM-parameter fixture failed at assigned-worker startup and then passed in isolation. Its free-port helper checked only the base port even though the assigned worker uses base + 1. The fixture now checks both adjacent ports and includes the response and port in startup failures. The corrected fixture passed five consecutive runs (18.32 seconds). The original failure did not preserve enough detail to establish its exact cause; the port-selection defect was found during investigation.

Verification after the final fixes: all 58 enabled CTest targets pass in one full run (285.87 seconds), including the 800 wrapper cases, nine real-filesystem cases, 30 actual-model wrapper calls and all earlier Whisper/HTTP/media/LLM regressions. The runtime policy/filesystem target passed in 1.75 seconds; the actual-model wrapper target passed in 159.19 seconds while the four inference suites ran concurrently. The CXX-only Debug build succeeds with CMAKE_C_COMPILER=/nonexistent-c-compiler. The runtime model executable has no unresolved shared libraries or libpython/tokenizers runtime linkage. First-party language scans, whitespace checks, fixture child cleanup and reference-checkout cleanliness pass. Strict enforcement still correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. Full 1:1 parity remains unproven and the goal remains active.

## Native STT model resolution and Hub snapshot acquisition

- Added native local-directory and faster-whisper alias resolution as the default STT model loader. Both dependency profiles preserve their source alias order and error messages; the legacy profile still rejects large-v3-turbo/turbo/distil-large-v3.5. The wrapper supplies the source models cache root and the five allowed filename patterns. Hub HTTP and requests connection errors trigger the source local-only retry; local cache failures, HTTPX connection errors and ordinary errors keep their distinct behavior. Hub exception category and OSError-family membership survive conversion to STT exceptions.
- Added a separate C++ snapshot-cache downloader following huggingface_hub 1.5.0, rather than treating the existing 1.28.0 local-directory tests as proof. It resolves repository metadata to a commit, records mutable references, uses immutable snapshot pointers and ETag blobs, reuses existing files/blobs, resumes incomplete downloads, resets an ignored range, verifies expected sizes, follows relative metadata and CDN redirects, and downloads files concurrently under shared blob locks. Cache state includes source-retained partial files, absent-file markers and references written before a failed tree listing. Offline snapshot lookup deliberately accepts an existing incomplete snapshot, as the source does.
- Added 37 comparisons against the unchanged pinned Hub implementation using independent real local HTTP servers. They check paths, exception category/message, methods/URLs/authentication/ranges, all cached bytes and symlink targets, and the complete directory tree. One case uses eight workers and three filenames sharing one blob; only its concurrent event order and interchangeable shared-blob GET filename are normalized. The remaining HTTP traces are exact after temporary-root/server-origin normalization. Captures use the HTTP path with Xet disabled and record filelock 3.29.0, HTTPX 0.28.1 and the reference requests 2.33.0 pin. Source hashes and the Hub wheel identity are recorded in fixture provenance.
- Added 504 unchanged-source alias/download-policy cases and 30 repository-validation cases. They cover all aliases in both profiles, invalid names, slash/newline behavior, Unicode and the distinctions among HTTP, cache, requests and HTTPX failures. The omitted/None snapshot revision is normalized to its main default; the supplied cache root and allow-list are compared exactly. All generators remain outside the repository, and the native code/runners remain C++.
- The existing 30 actual-model STT wrapper calls now use native model resolution, alternating local directories and offline alias snapshots between the English/multilingual models and dependency profiles. A separate native public-Hub check downloaded all four allowed assets of Systran/faster-whisper-tiny.en at revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba; model, configuration, tokenizer and vocabulary SHA-256 values exactly match the pinned inference assets. That isolated cache is /tmp/ezlocalai-hub-native-live-cache. No Python runtime participates in native acquisition or inference.

Remaining acquisition gaps include Xet transport, proxy/TLS exception fidelity, interrupted-stream retry details, arbitrary malformed metadata/filesystem errors, all optional Hub arguments and download scheduling during large-tree enumeration. Tokenizer fallback and preprocessor configuration loading remain to be ported. VAD assets still require an explicitly supplied directory; native asset packaging and default discovery remain incomplete. Native platform/audio operations, service pool/resource integration, speech API wiring and diarization also remain incomplete. The speech API still uses its earlier whisper.cpp adapter. These additions do not establish whole-service 1:1 parity, and strict manifest labels remain partial.

The first full run passed 58 targets but the native HTTP test hung: its client request had completed while a one-shot HTTPS fixture still waited for a connection, leaving the main test waiting in join. That confirmed live test was terminated; the failed run is not counted as a pass. The five users of that HTTPS helper now wait for a listener-ready signal instead of a fixed 50 ms sleep. The fixture has bounded accept/I/O waits, the previously hanging voice case deliberately delays startup by 100 ms, and native_http has a 180-second CTest timeout. Unexpected voice responses are reported before join. The original response was not preserved, so its exact failure cause remains unproven. The corrected HTTP test passed in isolation in 30.48 seconds and under the full parallel workload in 30.57 seconds.

Verification after the final fixes: all 59 enabled CTest targets pass in one full parallel run (230.23 seconds), including the new Hub snapshot/alias cases, native resolution in all 30 actual-model STT wrapper calls and the earlier Whisper/HTTP/media/LLM regressions. The build remains CXX-only with CMAKE_C_COMPILER=/nonexistent-c-compiler. Strict enforcement continues to report 96 rows, 37 implemented and 59 partial, with all 85 reference routes represented. Full 1:1 parity remains unproven and the goal remains active.

## Native STT PortAudio initialization and termination

- Added C++ PortAudio initialization as the STT runtime's default audio boundary. It loads the native PortAudio ABI directly, without PyAudio or Python. Initialization failure calls termination before retrieving the error text and preserves the source OSError message. Explicit termination ignores the PortAudio return value. Dropping an unclosed handle does not silently terminate it; the library remains resident across handle lifetimes, matching the imported source extension. Custom audio callbacks remain supported as a pair.
- Captured 120 cases through the unchanged PyAudio 0.2.14 class and compiled extension, substituting only three PortAudio ABI functions. Cases cover all 29 declared PortAudio errors, two unrecognized error codes, ignored termination failures, repeated termination, multiple instances, dropped objects, reopening and Unicode/multiline error messages. Native comparisons check exact call order and exception category/message. Additional wrapper checks preserve its single termination on repeated close and its explicit-close requirement.
- Captured and compared a separate real PortAudio lifecycle: two initializations, continued initialized state after the first termination, and uninitialized state after the last termination. Device inventory is normalized to initialized status; no recording stream is opened. The reference package/source hashes and host library identity are recorded in tests/fixtures/stt_audio_README.md, and the PyAudio MIT notice is included.
- Regenerated all 30 actual-model STT wrapper captures with real PyAudio initialization and termination, replacing their former audio substitute. Both profiles' complete outputs and errors are unchanged. Their native runner now uses default real PortAudio operations alongside native local/offline-alias model resolution. GPU availability remains substituted to test the pinned CPU inference paths. Added libportaudio2 to the native CUDA image; container execution remains unverified.

This implements the audio-library lifecycle required for uploaded-audio STT. Recording streams, microphone listening, wake detection, device selection and other PyAudio operations remain incomplete. Native GPU/cuDNN probing and allocator cleanup, tokenizer fallback, preprocessor configuration, VAD asset packaging, service pool/API integration and native diarization also remain outstanding. Speech API routes still use the earlier whisper.cpp adapter. The manifest remains partial and whole-service 1:1 parity is not established.

Verification: all 61 enabled CTest targets pass in one full parallel run (259.68 seconds). This includes the 120 new audio cases, real PortAudio lifetime check, 800 wrapper policy cases, nine filesystem cases, all 30 regenerated actual-model wrapper calls and all earlier Whisper/HTTP/media/LLM regressions. The actual-model wrapper target passed in 118.18 seconds under concurrent inference load. The new controlled audio test, live audio test and wrapper policy/filesystem target also passed separately (three targets, 1.81 seconds). The CXX-only full build passes with CMAKE_C_COMPILER=/nonexistent-c-compiler; the native model executable has no libpython/tokenizers runtime dependency or unresolved shared libraries. Whitespace checks pass and the reference checkout remains clean. Strict enforcement still correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. The full parity goal remains active and incomplete.

## Native Torch GPU initialization, cleanup and actual GPU STT

- Added an optional C++ LibTorch platform adapter for CUDA/cuDNN availability, device-0 total memory minus Torch allocations, the real source convolution probe, cache release, synchronization and IPC collection. Configured native Whisper adapters use these operations by default; explicit boundary overrides remain supported. Native shared ownership releases objects at the source collection points. The application neither links nor invokes Python.
- Changed the probe boundary to return an owned result that survives the following empty_cache call. Unchanged-source captures confirm that the convolution's autograd output keeps 2,048 bytes allocated through that call, then releases them when the probe returns while a 2 MiB cache segment remains. The C++ implementation preserves this lifetime, the source's CPU/CUDA RNG changes, and all six temporary/restored cuDNN flags.
- Added 18 real-GPU platform cases covering three external allocation sizes, two initial flag configurations, direct probing, normal initialization and GPU-constructor failure with CPU fallback. They compare exact allocated/reserved memory, RNG-state hashes, flags, available VRAM, loader arguments, handle/collection order and repeated close. Separate hidden-GPU captures compare compiled cuDNN availability, zero allocations, no-op cache release and exact memory/synchronization/IPC errors. Only model/audio boundaries are substituted in the platform cases.
- Added 15 complete GPU float16 STT wrapper comparisons using real Torch 2.9.1+cu128, cuDNN 9.10.2.21, faster-whisper 1.2.0, CTranslate2 4.6.1 and PortAudio. They use the existing checksum-pinned tiny English/multilingual models and audio/VAD assets. Both source and native runners require the models to stay on CUDA after every call. All text/structured output/errors match; timestamps retain the existing 1e-12 arithmetic allowance. The original 30 CPU wrapper comparisons remain separate.
- Inspected the standalone LibTorch 2.9.1+cu128 archive and found bundled cuDNN 9.8.0 rather than the reference's 9.10.2. The verified C++ target instead links the native implementation libraries shipped in the pinned Torch package, with recorded library hashes. Its version metadata is read without a Python interpreter. GPU model tests supply the cuDNN/cuBLAS/CUDA runtime search directories, matching the reference CUDA image. An initial reference capture without that environment aborted in cuDNN component loading; it was not retained as a successful capture.
- The first native GPU model comparison matched all 15 inference cases but segfaulted during temporary-directory cleanup. A debugger identified std::filesystem::remove_all interposed from the Torch package's libtorch.so compatibility shim. The target now links torch_cpu/torch_cuda/c10 implementation libraries directly, preserving the host C++ filesystem runtime. The corrected GPU test completes model close and filesystem cleanup successfully. Native dependency/relocation inspection reports no libpython, unresolved libraries or undefined symbols.

The verified GPU is an RTX 3090 Ti with 25,283,526,656 bytes of memory. The default availability path is covered; optional NVML availability mode, fork behavior, ROCm, CPU-only LibTorch builds, alternate allocators, arbitrary backend errors and other hardware remain unverified. Production dependency packaging, tokenizer fallback, preprocessor configuration, VAD asset packaging, service pools/speech API integration, microphone streams and native diarization remain incomplete. The speech API still uses whisper.cpp and strict manifest labels remain partial.

Verification: all 64 enabled CTest targets pass in one full parallel run (224.26 seconds), including all new GPU comparisons, the 30 CPU wrapper calls and the earlier model/HTTP/media/LLM regressions. The updated 800-case wrapper policy/filesystem target, hidden-GPU target, 18-case real-GPU platform target and 15-case real-GPU wrapper target also pass together separately (four targets, 20.40 seconds); the GPU model target completed that run in 17.19 seconds. The CXX-only full build succeeds with CMAKE_C_COMPILER=/nonexistent-c-compiler. Native library/relocation inspection, whitespace/language scans, fixture-child cleanup and reference-checkout cleanliness pass. Strict enforcement correctly remains failing: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. The full parity goal remains active and incomplete.

## Native Whisper preprocessor configuration and configured timing

- Added native loading of preprocessor_config.json after engine and tokenizer initialization. Missing files/directories and malformed JSON use source defaults; accepted keys are filtered against the source constructor, including its self-key invocation error. UTF-8 decoding errors propagate with source-shaped categories/messages. The native adapter now defaults to 80 feature bands when metadata is absent instead of deriving that setting from model weights.
- Configured sample rate now reaches FFmpeg decoding, feature extraction, durations, clips and timestamp restoration. Configured hop, FFT, bands and chunk size reach standard and batch controllers. Preserved source differences: modern feature padding and decoder width stay fixed, legacy feature padding/window width follow configuration, timestamp tokens retain 0.02-second precision, and batch chunk collection/VAD keep their source default rate where the reference omits an explicit rate.
- Added 64 unchanged-source file/constructor cases and 20 configured feature cases across faster-whisper 1.0.3/1.2.0. These cover 8/16/22.05 kHz, several hops/FFTs/chunk lengths/band counts, missing/malformed metadata, Unicode strings, UTF-8 failures and zero divisors. All captured feature arrays match exactly on the pinned environment; the established 2e-7 maximum tolerance remains unchanged. The legacy odd-FFT error is preserved.
- Added 80 configured standard-controller cases and 30 configured batch cases, comparing complete engine traces, feature hashes and segment/word output. The suites now contain 800 standard cases and 330 batch cases/372 invocations. New nondefault timing cases exposed NumPy scalar rounding versus Python float rounding; native word alignment now preserves the former while ordinary segment decimal rounding remains separate.
- Added 18 configured real-model constructors and 48 wrapper calls with actual tiny.en inference and PortAudio. They cover resampling, words, batch behavior, malformed-file defaults, incompatible feature bands and constructor errors. Native CTranslate2 exceptions now preserve their translated category through a thread-local C ABI accessor; shape validation and messages come from the engine. The tests match complete output, exception category/message and temporary-file cleanup, with the existing timestamp tolerance. Source capture programs remain outside the repository; implementations and test runners are C++.

Arbitrary metadata is not yet equivalent: float-valued timing fields, negative/oversized integers, malformed field types, nonstandard JSON numbers and all I/O errors remain incomplete or unverified. Persistent chunk overrides on a directly reused backend also remain open. Tokenizer fallback, VAD/dependency packaging, service pools/speech API integration, microphone streams and native diarization still require work. The speech API continues to use whisper.cpp; strict manifest labels remain partial. Fixture provenance is recorded in tests/fixtures/whisper_preprocessor_assets/README.md.

Targeted verification: all four affected reference suites pass (64.12 seconds), including all 48 configured real-model calls. The full parity goal remains active and incomplete.

Full verification: all 66 enabled CTest targets pass in one sequential run (702.92 seconds), including the configured-model suite, existing CPU/GPU STT comparisons, both BLAS-thread batch suites, streaming/lifecycle checks and real LLM smoke test. The CXX-only build still uses CMAKE_C_COMPILER=/nonexistent-c-compiler. The new model executable has no libpython/tokenizers runtime dependency or unresolved shared libraries. Whitespace/language/process-launch scans and fixture-child cleanup pass; the reference checkout remains clean at the pinned revision. Strict enforcement correctly exits 1: 96 rows, 37 implemented, 59 partial, all 85 reference routes represented. Full 1:1 parity remains incomplete.

## Native Whisper tokenizer fallback and single-file Hub fidelity

- Added the source constructor's tokenizer selection after engine initialization and before preprocessor loading. A regular local tokenizer file wins, including a symlink to one. Missing files, directories, dangling/looping symlinks and paths that fail isfile select openai/whisper-tiny or openai/whisper-tiny.en according to the loaded model's multilingual flag. The fallback uses main and the default Hub cache, independently of the model download root, and propagates acquisition failures without the model downloader's local-only retry.
- Verified that the Python binding of tokenizers 0.22.1 calls Python huggingface_hub.hf_hub_download; its Rust-only from_pretrained helper is not the relevant execution path. The C++ implementation reuses the native Hub 1.5.0 single-file path and does not link or invoke tokenizers/Python at runtime.
- Added 216 unchanged-source constructor/selection cases across both faster-whisper profiles. Real tokenizer parsing and substituted model/download boundaries cover both language modes, nine filesystem shapes and six success/error outcomes. Native tests compare selected paths, complete download kwargs/defaults and propagated category/message. Fixture headers pin both constructor sources and dependency versions.
- Added 27 direct hf_hub_download HTTP/cache captures, comparing exact request traces, results/errors, files, symlinks and directories. These exposed a real distinction from snapshot downloads: single-file reference updates propagate I/O errors and avoid rewriting an unchanged commit. The native paths are now separate. The consistency error abbreviates URLs; only its retained random local-server port is additionally normalized. The established 37 snapshot, 504 alias/retry and 30 repository-validation cases continue to pass.
- Added eight real-model fallback constructors and 32 STT wrapper calls using missing tokenizer paths and directories at that filename. Pinned OpenAI English/multilingual tokenizers are stored as compressed fixtures. Both profiles match ordinary/word/batched output, language detection, prompt/beam settings, legacy batch errors, explicit close and audio-file cleanup. Four CPU inference threads and one BLAS thread are enforced. The model download root remains separate from the default tokenizer cache and local tokenizer paths stay absent.
- The C++ downloader independently fetched both tokenizer files from the public Hub at pinned revisions; their SHA-256 values match the source assets. This live check remains outside mandatory CTest execution. Asset revisions, hashes, licenses, capture programs and scope are documented in tests/fixtures/whisper_tokenizer_fallback_assets/README.md.

Arbitrary tokenizer-file parsing errors/configurations, all Hub transport/filesystem behavior and import-time environment freezing remain unverified or incomplete. Preprocessor numeric/type edges, dependency/VAD packaging, native diarization, microphone streams and speech API/pool integration still require work. The public speech API still uses whisper.cpp. The manifest remains partial and the full parity goal remains active.

Targeted verification: all 216 tokenizer-selection cases and 32 real-model calls pass. After correcting the single-file reference behavior and ephemeral-port normalization, the complete Hub target passes (4.47 seconds). The build remains CXX-only; the new model test has no libpython/tokenizers runtime dependency or unresolved shared libraries. The source checkout remains clean. Strict enforcement correctly fails with 37 implemented and 59 partial rows, covering all 85 source routes.

The first full parallel run passed 67/68 targets; llm_context_reference failed during its second-model fixture launch. Retained arguments showed that worker using base + 1, while the test had checked only the base port; a retained TCP connection also referenced the unchecked port. Updated the C++ test allocator to bind/check all three adjacent worker ports together before releasing them. This changes fixture setup, not production model-context behavior. The failed run is retained at /tmp/ezlocalai-tokenizer-fallback-full-tests.log; it is not counted as a full pass.

A second full parallel run passed 66/68, including the context test, but intermittently failed model-download fixture startup and the remote-video request-count assertion. All speech and Hub tests passed. Both failing targets then passed ten repeated concurrent runs with added bind/request diagnostics. Their worker ports were released ephemeral ports, leaving a race during acquisition against other HTTP clients/listeners. Added a shared C++ test allocator that checks all adjacent ports outside Linux's ephemeral range and retains per-port advisory locks across fixture processes; the download, video, context and parameter fixtures now use it. Diagnostic messages are retained. These fixture changes do not alter production API/inference behavior. Both failed full-run logs remain recorded; neither is counted as a full pass.

Final verification: all 68 enabled CTest targets pass in a fresh full parallel run (288.21 seconds), recorded at /tmp/ezlocalai-tokenizer-fallback-full-tests-verified.log. The four affected worker-fixture targets also passed three consecutive concurrent runs (43.30 seconds) after port isolation. The full run includes all new tokenizer fallback/Hub comparisons, earlier configured CPU/GPU inference suites, HTTP/media/lifecycle regressions and real LLM smoke test. CXX-only configuration, whitespace/language/process-launch checks, native runtime dependency inspection, reference-checkout cleanliness and fixture-child cleanup pass. Strict manifest enforcement remains correctly failing with 96 rows, 37 implemented, 59 partial and all 85 source routes represented. The complete 1:1 parity goal is not achieved and remains active.

## Native request-keyed STT model pool

- Added a C++ STT pool with explicit request/task lease keys, resident/transient instances, FIFO available-instance reuse and the source's at-capacity fallback. Repeated acquisition of the same key returns the retained handle and increments resource usage as the source does. Resource registration aggregates CUDA-labelled instances at 2 GiB each and preserves reference active-count behavior.
- Preserved source cleanup distinctions: preload returns a leased handle to the queue; handoff forces synchronous destruction; force-clearing without a matching lease retains other lease entries; removal of a leased model does not scrub stale available references. Admission/force-CPU hints do not override the STT constructor's own device selection. A process-wide single C++ worker handles queued cleanup in FIFO order and survives job exceptions.
- Added 540 unchanged-source scenarios covering 12,480 actions, complete state snapshots, callback order, resource counts and injected failures. Deterministic randomized sequences cover mixed warm/get/destroy/register/configuration actions. The first capture harness had a name collision in tracing; it was corrected and the fixtures regenerated before the passing comparisons. Native concurrency checks cover same-key reuse, resident capacity and cleanup-worker serialization/failure recovery.
- Added a pool factory backed by the existing C++ STT runtime, CTranslate2 engine and PortAudio lifecycle. Two pinned real-model captures cover 46 actions and 14 inference calls across faster-whisper 1.2.0/1.0.3. They match resident reuse, word/batch output, closed-model/invalid-audio errors, transient reloads, queued cleanup, synchronous handoff and resource/temporary-file state. All first-party implementations and test runners are C++; source capture programs remain outside the repository. Provenance and limitations are recorded in tests/fixtures/stt_pool_README.md.

Targeted verification: both new targets pass together (30.14 seconds), including all policy actions and actual-model calls. The CXX-only full build succeeds. Concurrent warm-load interleavings, cancellation/voice-slot guards, GPU pool inference and complete server lifecycle integration remain unverified or incomplete. The public speech API still uses whisper.cpp. Native diarization, microphone streams, dependency/VAD packaging and earlier configuration/Hub edge cases also remain open. Strict manifest labels remain partial; the full parity goal is active and incomplete.

The first full run passed 69/70 targets (275.31 seconds), including both pool suites and every other speech test. llm_input_reference failed its vision-description assertion. Retained fixture output showed a vision child launch without a readiness probe; socket inspection showed that child's intended port 44802 already used as an unrelated HTTP client's source port (TIME_WAIT to port 11964). The input fixture still used bind(0)/close and checked only the base of its text/vision pair. Updated it to use the existing shared port allocator, reserving both adjacent worker ports outside Linux's ephemeral range. This changes fixture setup only. Failed-run log: /tmp/ezlocalai-stt-pool-full-tests.log; retained fixture: /tmp/ezlocalai-input-hcxeJc. This run is not counted as a full pass.

After port isolation, llm_input_reference, llm_parameters_reference and model_download_reference each passed three consecutive concurrent runs (43.02 seconds). The complete suite is being rerun against the final code and fixture setup.

Final verification: all 70 enabled CTest targets pass in a fresh full parallel run (275.29 seconds), recorded at /tmp/ezlocalai-stt-pool-full-tests-verified.log. This includes the corrected vision-input fixture, both pool suites, all earlier CPU/GPU inference comparisons, streaming/HTTP/media regressions and the real LLM smoke test. The full build remains CXX-only with CMAKE_C_COMPILER=/nonexistent-c-compiler. Native dependency inspection found no libpython/tokenizers runtime dependency or unresolved shared libraries; source/language/whitespace checks and reference-checkout cleanliness pass. Strict enforcement correctly remains failing with 96 rows, 37 implemented, 59 partial and all 85 source routes represented. Full 1:1 parity is still incomplete.

## Native voice-slot admission and handoff guard

- Added C++ FIFO permit queues with shared cancellation requests and cross-thread release. A shared binary queue serializes STT/TTS GPU handoffs; each guard has its own capacity-limited service queue. Cancellation wakes blocked admissions through registered callbacks, removes the cancelled waiter and preserves the remaining FIFO order.
- Added the source guard's acquisition, release and per-request context-stack lifecycle. Handoff marks LLM work unavailable before waiting for idle/unloading, then takes the service permit. Release marks the lease released before cleanup, synchronously destroys voice models before restoration, and clears availability/releases permits in source order. Acquisition failures and cancellation run the corresponding source cleanup. Cleanup failures replace prior failures and can interrupt later cleanup, matching the reference rather than silently continuing. A lease can be released by another thread; duplicate concurrent release runs cleanup once.
- Added 735 unchanged-source policy scenarios/2,804 actions, comparing exact callback order, complete state and exception category/message. Coverage includes STT/TTS dispatch, negative/zero capacities, nested/missing-key context operations, cross-task/repeated release, falsey/truthy handoff state, cancellation, individual boundary failures and compound original/cleanup failures.
- Added 11 traces/84 actions with real asyncio locks/semaphores, replayed using native threads and queues. They compare settled state, per-request event order, global acquisition order and serialized handoff callbacks. Covered cancellations include head/middle/tail waiters, waiting for LLM idle and waiting for a service permit after unload. Independent cross-thread completion notifications are not given an artificial global ordering. Native checks additionally cover 16-thread duplicate release and 100 cancellation/wakeup races per run.

Both new targets passed ten consecutive concurrent runs (6.66 seconds). Source capture programs remain external; all new runtime and test code is C++. Scope and provenance are documented in tests/fixtures/voice_slot_README.md. End-to-end request cancellation, native speech API/STT pool wiring, actual LLM handoff integration, GPU pool inference and earlier modality/configuration/deployment gaps remain incomplete. The public speech endpoints still use whisper.cpp and strict manifest rows remain partial; the full parity goal remains active.

Targeted memory/undefined-behavior verification: a standalone C++ build of voice_slots.cpp and its policy/concurrency runner passed all 735 scenarios plus FIFO/duplicate-release/cancellation-race checks with AddressSanitizer, leak detection and UndefinedBehaviorSanitizer enabled. Logs are /tmp/ezlocalai-voice-slots-asan-build.log and /tmp/ezlocalai-voice-slots-asan-test.log. The normal CXX-only build also succeeds with CMAKE_C_COMPILER=/nonexistent-c-compiler. The async runner has no libpython/tokenizers runtime dependency or unresolved shared libraries. Strict enforcement still correctly exits 1 with 37 implemented and 59 partial rows, covering all 85 source routes.

Full verification: all 72 enabled CTest targets pass in one fresh parallel run (276.02 seconds), recorded at /tmp/ezlocalai-voice-slots-full-tests.log. This includes both new guard targets, the STT pool/real-model suites, CPU/GPU speech regressions, HTTP/media/streaming/lifecycle checks and the real LLM smoke test. Whitespace/language/runtime-dependency checks and reference-checkout cleanliness pass. Full 1:1 parity remains incomplete: the guard and pool are not yet wired into the public speech endpoints, which still use whisper.cpp; strict enforcement retains 37 implemented and 59 partial rows.

## Native STT transcription and translation endpoint integration

- Added a C++ STT service joining the request-keyed pool, FIFO voice-slot guard and ResourceManager. Local transcription/translation now use it when a native model factory is configured. The service preserves endpoint usage decrements, conditional destruction, voice-server retained leases, preload reuse, synchronous handoff destruction before LLM restoration and failure precedence. The existing backend handoff hooks now connect STT requests to LLM unloading/lazy or explicitly requested restoration.
- Configured CTranslate2/VAD builds of ezlocalai-native select the native service automatically for POST /v1/audio/transcriptions and /v1/audio/translations. The executable uses the modern profile and reads WHISPER_VAD_DIRECTORY, defaulting to the configured install prefix's share/ezlocalai/whisper-vad directory. CMake installs supplied modern ONNX assets with their Silero/faster-whisper notices. Builds without this adapter retain their earlier CLI path; complete Docker/dependency packaging is still pending.
- Added 194 unchanged-source scenarios/392 HTTP requests using actual FastAPI multipart parsing. The C++ server matches response bodies/status/MIME, model kwargs, lifecycle order, pool identities, queued cleanup and resource counts across transient/preload/voice/handoff modes and load/inference errors. Transcription maps inference ValueErrors to HTTP 400; translation and constructor failures remain HTTP 500. The request's model field does not replace the configured local model. Bracketed timestamp fields take precedence and SRT/VTT preserve source formatting, including signed timestamps and speaker labels.
- The new cases found and corrected an existing audio form-validation mismatch: empty optional scalar fields use their FastAPI defaults. Decimal-zero num_speakers values now reach the model as integers. Initial API capture attempts had a tracing-argument collision; those invalid captures were replaced and normal-request success assertions added before the passing comparisons.
- Added 56 real-model HTTP requests across both faster-whisper profiles, using pinned tiny.en weights and real PortAudio. Complete JSON/text/word/SRT/VTT/translation output, invalid-audio HTTP 400, model lifecycle and temporary-file cleanup match the source. A separate C++ process test launches the actual executable and sends real TCP requests with whisper.cpp executables unconfigured; transcription, invalid-audio response and clean shutdown pass.

Targeted verification: all 392 final policy HTTP comparisons and the compiled-server TCP test pass together (3.95 seconds). The real-model API target passed all 56 requests (96.18 seconds) in the preceding targeted run; that run's policy target exposed the now-corrected empty-temperature mismatch and is not counted as a complete pass. Source provenance and scope are in tests/fixtures/stt_api_README.md. The subtitle endpoint and LLM/audio input paths still use whisper.cpp; native diarization, microphone streams, complete request cancellation, GPU API/handoff coverage, arbitrary form/configuration edges and container/dependency packaging remain incomplete. Strict manifest rows remain partial and the full parity goal is active.

Full verification: all 75 enabled CTest targets pass in one fresh parallel run (314.75 seconds), recorded at /tmp/ezlocalai-stt-api-full-tests.log. This includes the 392 policy API comparisons, all 56 real-model API requests, the actual executable TCP test, CPU/GPU speech regressions and existing HTTP/media/LLM lifecycle suites. The CXX-only build succeeds with CMAKE_C_COMPILER=/nonexistent-c-compiler; the executable has no Python/tokenizers runtime dependency or unresolved shared libraries. The install check copies both modern VAD ONNX files and both license notices with matching SHA-256 hashes; CMake excludes the upstream asset directory's empty __pycache__ directory. Whitespace checks and reference-checkout cleanliness pass. Strict enforcement remains 37 implemented and 59 partial rows across all 85 source routes; full 1:1 parity is not yet achieved.

## Native subtitle endpoint and batch translation

- Connected POST /v1/audio/subtitles to the native STT service in configured CTranslate2/VAD builds. It shares the tested pool/voice-guard lifecycle, passes source_language/prompt/temperature plus beam_size and condition_on_previous_text, and preserves fast_mode's beam-size override. Subtitle inference errors remain HTTP 500, including ValueError, as in app.py.
- Replaced the configured path's whisper.cpp segment conversion with source-shaped output: original-language JSON retains all segment fields; translated segments retain only id/start/end/text. SRT/VTT use the native source formatters. Duplicate target languages still do duplicate work and overwrite the same output key; unknown formats still perform translations before returning an empty subtitles object.
- Added C++ translation helpers matching Python word splitting, 4000-word batches (including empty/oversized segments), numbering reset, exact prompt/model inputs, Unicode decimal line numbers, duplicate-number replacement and original-text fallback. Malformed choices/message/content and translator exceptions propagate rather than being suppressed.
- Target languages run concurrently. Calls go directly through the existing LLM processing path, including fallback/slot acquisition/release, without HTTP chat defaults, queue admission/timeouts or API response conversion. Internal results use owned strings rather than the C queue's bounded output buffer.
- Added 81 unchanged-source scenarios/163 real multipart HTTP requests and 70 direct translation-helper cases. All pass, together with the earlier 392 transcription/translation policy comparisons (1.68 seconds). The helper runner additionally requires two target-language callbacks to enter before either can complete. Sequential per-language batching is compared exactly; independent language call ordering is compared as a multiset of complete request bodies.
- Captured 32 real-model subtitle HTTP requests across modern/legacy faster-whisper profiles and all four pool modes; all source/native comparisons pass (59.03 seconds). LLM translation is substituted in those cases. The compiled-server TCP smoke now also verifies full source-language subtitle JSON without a whisper.cpp executable, and passes (5.93 seconds). Both targets pass together in 59.04 seconds, recorded at /tmp/ezlocalai-subtitles-model-tests.log.

Provenance and precise scope are documented in tests/fixtures/subtitles_README.md. The row remains partial: Python asyncio.gather's error-time background scheduling differs from joining outstanding native futures; arbitrary Python object/form edges, cancellation, live weighted-LLM translation/GPU handoff, LLM audio input, diarization and deployment packaging remain incomplete. The existing non-adapter build still retains its CLI fallback. Full 1:1 parity remains unachieved.

Next integration evidence: Pipes.get_response's audio_url and audio_format branches (Pipes.py:10070, 10222, 10313) perform an extra STT mark-in-use after pool acquisition, decrement in finally, and call _destroy_stt only after successful inference, regardless of voice-server mode. Both video branches (10014, 10169) put _destroy_stt inside finally. audio_to_audio (8409) instead conditionally destroys after success outside finally. A future native LLM/audio integration must preserve these distinctions rather than reuse the public API's cleanup policy unchanged; the current CLI implementation does not yet do so.

Full verification: all 77 enabled CTest targets pass in one fresh parallel run (348.02 seconds), recorded at /tmp/ezlocalai-subtitles-full-tests.log. This includes subtitle policy/model/concurrency/TCP coverage, the earlier transcription/translation tests, CPU/GPU speech and LLM streaming/lifecycle/HTTP regressions. The complete CXX-only build succeeds with CMAKE_C_COMPILER=/nonexistent-c-compiler. Runtime dependency checks find no libpython/libtokenizers or unresolved libraries; native first-party implementation/tests contain no Python, C, JavaScript or shell source files. Whitespace checks pass and the reference checkout remains clean at the pinned revision. Strict enforcement still correctly exits 1: 37 implemented/59 partial rows, all 85 source routes covered. Full 1:1 parity remains incomplete.

## Native STT in LLM audio/video preparation

- Configured CTranslate2/VAD servers now send LLM audio/video transcription through the same native STT service used by public speech APIs. Raw JSON input types reach the existing STT wrapper, and source error categories are retained when crossing into the LLM preparation error path. Builds without the adapter keep their earlier CLI fallback.
- Added explicit endpoint, LLM-audio and LLM-video service usage modes. Both LLM modes add the source's extra resource mark after pool acquisition. Audio destroys after successful inference only, including voice-server mode; video destroys in finally. Audio failures deliberately retain the pool lease/resource state until subsequent source-defined cleanup. Guard handoff failure/exit precedence remains shared with the earlier implementation.
- Added 160 unchanged-source scenarios/640 calls covering all five STT blocks in Pipes._get_response_internal, capacities one/two, transient/preload/voice/handoff modes, constructor/inference errors, recovery and repeated task keys. Return/error, kwargs, in-inference usage count, lifecycle order and complete exposed pool/resource state match. The standalone target passes in 0.38 seconds.
- Added 48 real-model calls across faster-whisper 1.2.0/1.0.3 and both audio/video cleanup modes. Pinned tiny.en, real PortAudio, invalid audio and recovery match the source with four inference threads and one BLAS thread. The target passes (79.12 seconds). The actual executable TCP test now verifies speech-to-LLM audio_format transcription with whisper.cpp unavailable (8.65 seconds); both targets pass together in 79.13 seconds.
- Added managed LLM HTTP fixtures with native process-backed LLM lifecycle and injected STT: 24 scenarios/96 request state/event comparisons plus recovery, ignored-input and handoff checks. Initial test assumptions were corrected before verification: the first actual STT inference can occur after an earlier video failure already loaded the LLM, and the two legacy source branches cannot be reached by ordinary JSON after list-content flattening. Those legacy blocks remain covered directly; additional unchanged full-prefix captures verify ignored top-level message fields. The final managed HTTP runner passes all 128 requests (96 state/event comparisons, 24 text recoveries, four ignored-field requests and four handoff/recovery requests). It passes with the 640-call policy target in 17.70 seconds; /tmp/ezlocalai-stt-pipeline-http-tests.log records the final run. Full-suite verification is recorded below.

Source provenance and limitations are in tests/fixtures/stt_pipeline_README.md. This does not complete speech parity: the speech endpoint's separate audio-to-audio/PDF handler is not migrated, nor are full cancellation/interleaving behavior, native diarization/microphone streams, live weighted-LLM/GPU handoffs or complete container/dependency packaging established. Subtitle failure-time background scheduling remains partial. All affected manifest rows remain partial and the original full parity goal remains active.

Next speech evidence: the current reference's TTS backend is ezlocalai/LlamaTTS.py (302 lines) plus tts/worker.cpp (168 lines), with tts/CMakeLists.txt pinning llama.cpp aa94f2086147cd4bfbcff818ce998dffce5fcb39. CTTS.py wraps that subprocess protocol, normalization/chunking, reference audio, CPU fallback/cache and streaming behavior. A C++ port can reuse the source native worker while replacing those Python wrappers; the old Chatterbox-era manifest notes are not the current TTS completion target. Separately, Pipes.audio_to_audio decodes the data URI before passing bytes to STT, whose wrapper base64-decodes again; its error/cleanup semantics need a source comparison rather than reusing the current CLI shortcut.

Full verification: all 80 enabled CTest targets pass in one fresh parallel run (382.79 seconds), recorded at /tmp/ezlocalai-stt-pipeline-full-tests.log. This includes the 640-call service comparison, all 128 managed HTTP checks, 48 real-model pipeline calls, executable speech-to-LLM TCP verification, earlier STT/subtitle CPU/GPU tests and LLM/video/streaming/lifecycle regressions. The complete CXX-only build succeeds with CMAKE_C_COMPILER=/nonexistent-c-compiler. Whitespace checks and reference-checkout cleanliness pass; runtime dependencies include no libpython/libtokenizers or unresolved libraries. Strict enforcement remains 37 implemented/59 partial rows covering all 85 source routes, and correctly fails while those gaps remain. The original full parity goal is not complete.

## Native Qwen TTS process wrapper

- Added a C++ replacement for the current LlamaTTS.py process wrapper and copied the current private C++ worker byte-for-byte. The optional worker target links to the independently built pinned llama.cpp/libmtmd without enabling first-party C or fetching dependencies implicitly. The existing native Hub downloader resolves default/local/overridden model files and the legacy Qwen model-name migration.
- Implemented source constructor ordering, launch settings, integer conversions, child-only CUDA/HIP visibility and graph opt-out, language/script selection, strict worker readiness, persistent request serialization, float32/PCM16 WAV readback through libsndfile, incremental little-endian float32 PCM and source error/cleanup distinctions. Rates retain unbounded decimal integer conversion semantics. Timeout, explicit close and abandoned/failed streaming stop and reap the child without waiting on its inference lock. Child signal defaults match Popen restoration; parent signal/environment state remains unchanged.
- Added 305 constructor and 266 language cases from the unchanged source. Added 115 real-process scenarios/205 operations using a native fixture worker, including startup noise/errors, malformed replies, request errors, missing files, base64 validation/error precedence, timeout/EOF, aborted streams and consumer errors. Four gate-controlled threaded combinations verify buffered/streamed serialization; separate tests close during blocked inference. Every scenario verifies worker cleanup and unchanged parent environment. Initial comparisons exposed five error-message/precedence differences; those were corrected before the passing run.
- Downloaded the two pinned Qwen GGUF files through hf and captured five calls using the unchanged source wrapper and worker on CPU (seed 23, four threads). The native wrapper exactly matches all six float32 buffer SHA-256 values, rate and shape. The final stream crosses the 72-frame vocoder boundary and emits 138,240 samples plus a 7,680-sample tail. The initial three-target policy/process/model run passes in 156.81 seconds; model hashing was subsequently changed to bounded-memory chunked reads and child SIGPIPE restoration received a source comparison before full regression verification.

Provenance, optional build switches and scope are in tests/fixtures/llama_tts_README.md. This is a standalone backend layer, not complete local TTS: CTTS cleanup/normalization/chunking, reference audio, caching, CPU fallback, streaming transport, TTS pool/voice guard and endpoint integration remain pending. The previous Chatterbox-era helpers need current-source verification as they are integrated. Real GPU TTS, production engine packaging, arbitrary malformed worker/value edges and full cancellation interleavings remain incomplete or unverified. Affected endpoint rows remain partial; the full 1:1 parity goal remains active.

Full verification: the complete CXX-only Debug build succeeds with CMAKE_C_COMPILER=/nonexistent-c-compiler. All 83 enabled CTest targets pass in one fresh parallel run (463.14 seconds), recorded at /tmp/ezlocalai-tts-full-tests.log. This includes the final signal-restoration process comparison (2.89 seconds), exact real-model TTS comparison (208.20 seconds under concurrent model-test load), policy cases and all previous CPU/GPU STT, HTTP, LLM, media, streaming and lifecycle regressions. The wrapper test, native executable and worker have no unresolved libraries, libpython or libtokenizers dependencies. Source checkout remains clean at the pinned revision, whitespace checks pass, and no first-party .c/.py/.sh sources were added. Strict enforcement still reports 37 implemented and 59 partial rows covering all 85 reference routes and correctly fails until remaining gaps are closed. The full parity goal is not complete.


## Current CTTS text preparation and chunking

- Replaced the inherited C ABI TTS normalization/chunking internals with one native C++ text implementation shared by future CTTS generation. Inspection found real current-source gaps: UTF-8 byte counts instead of Unicode code points, missing CJK sentence/delimiter punctuation, an older comma split policy, empty-string/zero-limit differences, mixed-case AM/PM acceptance, time-pattern whitespace handling and invalid-date scanning. The signed minimum integer also no longer overflows during negation.
- Implemented current CTTS multilingual cleanup, language aliases/title fallback/Cyrillic override, English-only preparation, filename-stem rules and streaming first-chunk/target/minimum policy. The implementation preserves source quirks rather than imposing cleaner spelling, punctuation or whitespace. Python Unicode 15.0 properties and full case mappings are frozen as C++ data, including contextual Greek final sigma and title expansions, without adding a Python dependency.
- Executed unchanged source helper AST nodes and class methods to capture 44,831 cases: 2,963 integer, 4,974 normalization, 4,974 cleanup, 12,960 buffered chunk, 5,400 streaming chunk, 11,715 language, 585 preparation and 1,260 filename cases. Inputs include all decimal blocks, word/whitespace range boundaries, every nonidentity casing mapping, case-ignorable context, random multilingual input, embedded NUL, extreme integer boundaries and fractional chunk thresholds. Another 9,662 representable cases compare the existing C ABI directly. The initial source and core tests both pass (3.92 seconds). The independently captured current-source comma output replaces the obsolete core-test expectation.
- Compressed the 12 MB capture into a roughly 445 KB gzip fixture read by native zlib. Source and Unicode table hashes prevent silently using another reference. Provenance and typed-interface limitations are documented in tests/fixtures/ctts_text_README.md; the CPython capture-runtime notice accompanies the generated Unicode data.

The native text layer is implemented, but CTTS model policy/reference audio/audio conversion/cache/streaming transport, TTS pool/guard and endpoint integration remain pending. Existing helper manifest descriptions now reflect tested current-source scope; speech endpoints remain partial. Full parity is not complete and the original goal remains active.

Full verification: all 84 enabled CTest targets pass in one fresh parallel run (463.05 seconds), recorded at /tmp/ezlocalai-ctts-full-tests.log. The compressed CTTS source fixture passes in 3.59 seconds and reports all 44,831 source cases plus 9,662 C ABI comparisons. The complete CXX-only Debug rebuild is clean with no compiler warnings; CMAKE_C_COMPILER remains /nonexistent-c-compiler. Earlier exact Qwen CPU audio, CPU/GPU STT, HTTP, LLM, media and lifecycle suites also pass. Native executable/text-test dependencies contain no unresolved libraries, libpython or libtokenizers. Whitespace checks pass and the source checkout is unchanged. Strict enforcement remains 37 implemented/59 partial rows, all 85 source routes covered, and correctly fails while the remaining gaps are open. The full parity goal remains active.


## Native CTTS configuration and model lifecycle

- Added CttsRuntime for the unchanged CTTS constructor prefix through sample-rate initialization: current model migration, chunk/stream settings, generation kwargs, warmup limits, device selection and initial model construction. Integer settings retain decimal precision; device selection preserves memory-query fallback, explicit-device behavior and source call order.
- Added single-sample generation with inference-context ordering, source kwargs/error precedence, first-audio/rate conversion, RuntimeError-specific CPU fallback and cleanup in success/failure paths. Close detaches the backend before invoking its close hook; CPU reload preserves source ordering and failed-load state. State access is synchronized without holding the controller mutex across backend generation or close.
- Added an optional native LibTorch adapter for CUDA availability, free/total/allocated memory, cleanup and per-thread inference mode. CUDA memory queries restore the caller's selected device. The adapter shares the existing configured LibTorch runtime and explicitly links CUDA runtime; no Python dependency is introduced.
- Captured 787 unchanged-source scenarios/1,268 operations, comparing exact environment/hardware/model/context traces, errors, settings and backend state. Coverage includes arbitrary-size configuration integers, fallback and cleanup errors, repeated close/reload, and two gated cross-thread close cases while generation waits for a result or CUDA failure. Source captures stop before filesystem/cache/warmup construction and do not establish full CTTS behavior.
- Captured four actual CPU requests through the source controller prefix and native worker: warmup-limited Hello., configured Good morning., invalid empty text, and Hello again. after recovery. Native output matches all three complete float32 audio hashes (7,680/15,360/15,360 samples at 24 kHz) and the exact intervening error. The targeted model comparison passed in 30.88 seconds. The same runner verifies real CUDA memory-query bounds and actual LibTorch inference-mode entry/restoration; synthesis remains CPU-only.

Source/model provenance, dependencies and limits are documented in tests/fixtures/ctts_runtime_README.md. Reference-audio resolution/filesystem containment, audio conversion, cache/warmup execution, full buffered/streamed generation, TTS pool/voice guard and public speech integration remain unfinished. Logging, arbitrary Python/malformed JSON boundaries, exhaustive cancellation interleavings, GPU TTS synthesis and production packaging remain incomplete or unverified. All affected speech manifest rows remain partial and the full parity goal remains active.

Next integration evidence: CTTS creates outputs/voices and lists sorted .wav stems only after model construction, then constructs AudioCache and performs nonfatal default.wav warmup. Buffered generation checks a hard-coded outputs/cache/audio path before consulting AudioCache, even when a custom cache directory is configured; a cache hit can return a cache URL while a newly generated result writes a separate output file. Preserve this source behavior during integration. Streaming emits the sample-rate/16-bit/mono header even for empty prepared text, sends each PCM length separately from bounded payload writes, optionally inserts silence between text chunks, and closes the entire CTTS owner on any BaseException inside a chunk; it does not use single-sample CPU fallback. PCM conversion uses float32 nan-to-number/clipping, multiplication by 32767 and little-endian int16 conversion. These behaviors still require native implementation and differential fixtures.

Full verification: all 86 enabled CTest targets pass in one fresh parallel run (495.02 seconds), recorded at /tmp/ezlocalai-ctts-runtime-full-tests.log. The final 787-scenario/1,268-operation CTTS policy target passes in 1.42 seconds and the actual CTTS model target in 74.33 seconds under concurrent model load. Earlier exact Qwen CPU audio, CPU/GPU STT, HTTP, LLM, media, stream and lifecycle regressions also pass. The complete CXX-only Debug rebuild succeeds without compiler warnings, with CMAKE_C_COMPILER=/nonexistent-c-compiler. The native executable, Qwen worker and both new test binaries have no unresolved libraries, libpython or libtokenizers dependencies. Whitespace checks pass, no first-party .c/.py/.sh/.js files were introduced, and the reference checkout remains clean at the pinned revision. Strict enforcement correctly fails with 37 implemented/59 partial rows covering all 85 source routes. The full parity goal remains active and incomplete.


## CTTS audio conversion and reference-voice paths

- Added C++ CTTS audio arrays with explicit shape/strides/offset, NumPy-compatible mono mixing and reduction order, separate streaming PCM16 and soundfile WAV conversion, in-memory libsndfile read/write and tensor layout conversions. WAV retains the source's caught-error empty-byte result, while tensor/byte helpers retain their distinct optional failure results. Streaming PCM preserves float32 scaling by 32767 and non-finite/clipping rules.
- Added native LibTorch resampling matching torchaudio 2.9.1 default Hann sinc kernels, float64 kernel construction/float32 cache, convolution/padding and target-length rounding. Decoding follows soundfile float64 reads before float32 conversion and channel-first layout. No Python runtime or Python implementation is included.
- Captured 2,122 unchanged-source audio cases: 420 mono, 426 PCM, 435 WAV, 399 tensor-to-bytes and 442 decode cases. They cover C/Fortran/transposed/reversed/stepped arrays, empty/scalar/high-dimensional inputs, reduction boundaries, signed zero, denormals, clipping, NaN/infinity, nine audio container encodings and CPU rate conversion. Finite samples compare bit for bit; intermediate NaN payload/sign bits are not asserted, while all encoded byte output remains exact. The complete targeted audio comparison passes in 3.33 seconds.
- The audio differential checks found and corrected reduction-order differences for reversed/stepped views. They also exposed a one-ULP Vorbis decoding difference between two libsndfile builds reporting the same version (1.2.2). Added explicit EZLOCALAI_SNDFILE_LIBRARY selection shared with LlamaTts, and pinned the source soundfile native library by SHA-256. With that build configured, all captured decoding/resampling results match. Default system-library codec equivalence is not claimed.
- Added current CTTS contained paths, voice lookup/default fallback and transcript reading. Non-strict realpath follows CPython behavior for missing components and symlink loops, checks containment after resolution, and preserves source existence semantics. Transcript reads apply universal newlines/Unicode stripping, suppress OS errors and propagate exact UTF-8 decoding failures. All 610 real-filesystem source cases pass (0.43 seconds), including outside/dangling/looping symlinks, directory-valued voices, root containment, permissions, invalid extensions and malformed UTF-8.
- Extended the actual CTTS CPU model comparison to verify complete WAV and streamed-PCM hashes for all three audio buffers, retaining original float32 hashes, invalid-request error and recovery. The targeted model comparison passes in 31.76 seconds. Initial audio capture construction was corrected to copy negative-stride single-element views before creating Torch tensors; errors from that test setup were not treated as CTTS behavior.

Provenance and limits are in tests/fixtures/ctts_conversion_README.md. These components still need composition with full CTTS construction/cache/warmup, buffered generation and stream framing/cancellation, followed by TTS pool/voice guard and public API integration. Arbitrary Python object/dtype inputs, NaN payload bits, GPU resampling, Windows/surrogate filenames, filesystem mutation races, logging and complete native dependency packaging remain incomplete or unverified. All speech endpoint rows remain partial and the full parity goal remains active.

Next cache evidence: AudioCache creates directories and loads statistics even when disabled; enabled construction then removes expired entries and enforces size. A cache hit updates metadata/access_count before reading audio, and a store writes audio before metadata, so failures can leave observable partial state. Size enforcement counts all WAV bytes but only evicts entries whose metadata can be read; metadata.get("last_accessed", metadata["created_at"]) still requires created_at because the default expression is evaluated eagerly. Statistics persist every ten hit/miss/store/error events; expired counts are excluded from that total. Preserve these ordering/error details rather than replacing the cache with a conventional atomic cache. Full CTTS buffering and streaming composition remains the next integration work after caching.

Full verification: all 88 enabled CTest targets pass in one fresh parallel run (477.24 seconds), recorded at /tmp/ezlocalai-ctts-conversion-full-tests.log. This includes the extended real-model CTTS target (45.12 seconds), all 2,122 audio cases (4.00 seconds) and 610 filesystem cases (0.59 seconds), plus prior CPU/GPU STT, Qwen worker, HTTP, LLM, media and lifecycle suites. The rebuild revealed that CTTS's CUDA runtime lookup could select host CUDA 13 ahead of the explicitly configured LibTorch CUDA 12 runtime. CMake now searches configured runtime directories first; after clearing that cached library selection, the complete build succeeds without warnings and both CTTS binaries load only libcudart.so.12. All five affected CTTS targets then pass together in 40.71 seconds, recorded at /tmp/ezlocalai-ctts-runtime-link-tests.log; this is the final configuration. The CXX-only Debug build retains CMAKE_C_COMPILER=/nonexistent-c-compiler. Native executable/worker/new-test dependencies contain no unresolved libraries, libpython or libtokenizers. No first-party C/Python/shell/JavaScript sources were added, whitespace checks pass, and the reference checkout remains clean at the pinned revision. Strict enforcement still correctly fails with 37 implemented/59 partial rows and all 85 routes covered. Full 1:1 parity remains incomplete and the original goal remains active.


## Native AudioCache lifecycle and persistence

- Added CttsAudioCache with source configuration/defaults, directory initialization, persistent statistics, cache-key normalization/JSON hashing, reads/stores, TTL cleanup, size eviction, removal and voice/full clearing. It preserves metadata mutation before writes, audio-before-metadata storage, metadata-before-audio read updates and their observable failure states. UTF-8 text normalization shares the pinned CTTS Unicode implementation.
- Preserved source details that a conventional cache would change: disabled construction still creates directories and loads stats; orphan WAV bytes count toward limits without necessarily being evictable; last_accessed lookup eagerly requires created_at; expiry is excluded from the periodic event total but still runs the modulo save check; clearing all saves reset stats even when recording is disabled; symlinked cache directories are not removed by the source's ignored-error rmtree operation.
- Added shortest/statistical sample selection and explicit warming. Native duration analysis follows PyDub's PCM WAV parser/rounding and uses ffprobe/ffmpeg for other formats. Successful samples remain in the candidate list when subsequent temp-file deletion fails. Warming preserves audio-only existence skips, repeated normalized phrases, base-parameter overrides and source metadata order. No Python/NumPy/PyDub runtime is invoked.
- Captured the complete unchanged AudioCache class in 205 scenarios/703 operations. The C++ runner compares exact returns/errors, generator parameters, each clock call, configuration/raw statistics, metadata mutation and the complete filesystem tree with byte-for-byte file contents. Coverage includes disabled/error states, fractional/negative/large TTLs, aware dates, corrupt stats/metadata, periodic-save failures, cache limits/orphans, symlinks, selection/warming and native duration comparisons for nine container encodings and PCM widths/channels.
- Differential checks corrected C++ evaluation order when reading malformed statistics and the clock-call order before rejecting aware/naive datetime subtraction. The final targeted cache comparison passes in 3.05 seconds. Source and PyDub provenance, supported typed inputs and limits are documented in tests/fixtures/ctts_cache_README.md; NumPy/PyDub notices are included.

The cache still needs composition with complete CTTS construction/warmup, buffered generation and streaming bridge/framing/cancellation, then TTS pool/voice guard and public endpoints. Arbitrary Python objects/mapping keys, non-finite/unbounded JSON numbers, every ISO datetime representation, concurrency/filesystem races, logging, converter discovery and complete native dependency packaging remain incomplete or unverified. Speech manifest rows remain partial; the original full parity goal remains active.

Next integration evidence: AudioStreaming.stream_native_audio bridges the synchronous model iterator through a queue bounded to two items, polling pending sends at 0.1 seconds. It closes the iterator in the producer's finally block and closes the model on any incomplete consumer exit, then awaits the producer under cancellation shielding before releasing the caller's GPU lease. Native streaming must retain this bounded backpressure and teardown ordering when composed with CTTS's header/chunk/error handling. CTTS itself still does separate default.wav warmup after directory/voice listing and AudioCache initialization.

Full verification: all 89 enabled CTest targets pass in one fresh parallel run (476.10 seconds), recorded at /tmp/ezlocalai-ctts-cache-full-tests.log. The new cache comparison passes in 3.24 seconds with all 205 scenarios/703 operations, including real ffprobe/ffmpeg duration fallback. Existing CTTS text/runtime/path/audio/model tests, exact Qwen CPU output, CPU/GPU STT, HTTP, LLM, media and lifecycle suites also pass. The complete CXX-only Debug rebuild succeeds without warnings, with CMAKE_C_COMPILER=/nonexistent-c-compiler and the corrected CUDA 12 runtime selection retained. Native executable/cache/text/model-test dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. No first-party C/Python/shell/JavaScript sources were added; whitespace checks pass and the reference checkout remains clean at the pinned revision. Strict parity enforcement still correctly fails with 37 implemented/59 partial rows covering all 85 reference routes. The full 1:1 parity goal remains active and incomplete.


## Complete CTTS construction and buffered generation

- Added Ctts to compose native settings/model lifecycle, output/voice directories, sorted voice snapshots, AudioCache and nonfatal default-voice warmup in source order. Buffered generation now combines text/language preparation, live voice lookup, cache identity, chunked worker inference, mono/WAV conversion, cache persistence and base64/output-URI responses.
- Preserved the hard-coded cache precheck with custom cache directories, cache hits after reference removal, distinct cached/fresh output URLs, empty-cache fallthrough, last-chunk sample-rate behavior, UUID evaluation with explicit filenames and cache storage before possible output failure. Requests after close retain source cache-hit versus uncached-error behavior.
- Captured complete unchanged CTTS construction/generation and AudioCache in 140 scenarios/539 operations. Comparisons include exact errors/results, caller-versus-worker execution, environment/hardware/model/context traces, deterministic clock/UUID calls, voices/model/rate/cache state and every filesystem entry/file byte. The final policy comparison passes in 3.73 seconds. An omitted os.sep in the initial external capture harness was corrected before accepting fixtures; harness errors are not treated as source behavior.
- Added a complete real-model source capture with default-voice warmup and seven subsequent operations: fresh/cached URI results, two-chunk concatenation, reference removal, cached base64, uncached missing-reference failure and close. Its native runner compares result/file hashes and lengths, exact metadata/state and actual worker inference contexts. Provenance and scope are in tests/fixtures/ctts_generate_README.md.

Ctts is a synchronous native interface whose individual synthesis chunks run on workers. Full event-loop scheduling/cancellation, bounded streaming bridge/framing/error teardown, TTS pool/voice guard and public speech integration remain unfinished. Arbitrary Python object/malformed JSON boundaries, concurrent mutation/filesystem races, GPU synthesis, logging and full native dependency packaging remain incomplete or unverified. Speech manifest rows stay partial and the original full parity goal remains active.

Next streaming integration evidence: the current LlamaTts native wrapper already offers a synchronous stream callback, but CttsBackend exposes only buffered generate/close. A native bounded bridge needs a stream hook and capacity-two handoff with explicit cancellation. Source AudioStreaming obtains the iterator in its producer thread, sends audio/done/error events, closes the iterator in finally, and on incomplete consumer exit closes the model before joining the producer under cancellation shielding. CTTS's outer generator yields the eight-byte header before entering the chunk try/catch; cancellation at that first yield differs from cancellation within a chunk. Each PCM length and bounded payload write is a separate yield, silence between chunks is one combined length/payload yield, and a final four-byte zero terminator is yielded after all chunks. A changed sample rate or any BaseException inside a chunk closes the CTTS owner; streaming does not use the buffered sample CPU-fallback helper. These ordering and cancellation boundaries still require native implementation and differential fixtures.

Targeted actual-model verification: all seven buffered operations pass in 42.26 seconds, including exact generated WAV/metadata hashes, two-chunk concatenation, cached base64 after reference removal, missing-reference error and close. The constructor and each operation compare independent snapshots; an initial source harness snapshot alias and temporary-root normalization omission were corrected before accepting this fixture. The final native runner preserves captured model-path strings for cache identity while allowing actual worker/model paths to relocate. The policy target also passes after the final rebuild (3.65 seconds).

Full verification: all 91 enabled CTest targets pass in one fresh parallel run (520.49 seconds), recorded at /tmp/ezlocalai-ctts-generate-full-tests.log. The complete real-model buffered CTTS comparison passes in 92.28 seconds under concurrent model load. Existing CTTS text/runtime/cache/path/audio/model, exact Qwen output, CPU/GPU STT, HTTP, LLM, media and lifecycle suites also pass. The complete CXX-only Debug rebuild succeeds without compiler warnings, with CMAKE_C_COMPILER=/nonexistent-c-compiler and the pinned CUDA 12/libsndfile selection retained. Native executable/worker/new-test dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. No first-party C/Python/shell/JavaScript sources were introduced; whitespace checks pass and the reference checkout remains clean at the pinned revision. Strict parity enforcement still correctly fails with 37 implemented/59 partial rows and all 85 source routes covered. Full 1:1 parity remains incomplete and the original goal remains active.


## Native bounded audio stream bridge and iterator ownership

- Added lazy TtsStreamIterator to the native LlamaTts wrapper. It retains native model state and the inference lock across yields, preserves pre-request versus active-stream error/close boundaries, and releases its model reference on completion/close. Callback streaming delegates to the iterator. The native CttsBackend now exposes the iterator factory.
- Added CttsStreamBridge with a capacity-two queue, producer-thread iterator ownership, cross-thread cancellation request, separate-worker model close and producer join. Complete streams retain the model. Incomplete streams close it before joining. Source model.close failures return without waiting for the producer; shared native state retains that producer safely rather than changing this observable failure behavior.
- Captured 58 cases from the complete unchanged AudioStreaming bridge with actual asyncio/AnyIO threads and deterministic synchronization gates. They compare frames/rates, producer traces, callback thread placement, cleanup calls, exception precedence and producer completion. Three cases prove bounded prefetch; another proves a failed model.close returns while the producer remains pending. Factory failures and uncaught BaseExceptions retain the source's waiting-until-cancellation behavior.
- Added 11 native process cases for completion/recovery, abandonment, blocked inference cancellation, consumer/worker/reference errors, unstarted close and iterator/model ownership. The existing source process suite continues to cover callback compatibility. The actual Qwen model comparison now routes its streamed requests through this bridge while retaining the existing exact source float32 hashes and 72-frame/tail coverage.

Provenance and interface limits are in tests/fixtures/audio_stream_bridge_README.md. CTTS outer stream framing, silence/drain scheduling, async HTTP cancellation, TTS pool/voice guard and endpoint integration remain unfinished. Every scheduler interleaving, arbitrary Python/JSON boundary, logging and platform behavior remain incomplete or unverified. The original full parity goal remains active; speech manifest rows stay partial.

Targeted final verification: the 58 source bridge cases pass in 0.44 seconds, all 11 native process cases pass in 0.28 seconds, and the existing 115-scenario/205-operation worker-process comparison passes in 2.91 seconds. The complete rebuild is warning-free. No source edits were made during either active build. New bridge/process/model test binaries have no unresolved libraries or Python dependencies.

Next composition detail: CTTS yields its header before entering the chunk try/catch and its terminal zero after leaving it; closing at either yield must not mark the CTTS owner closed. Closing at a frame length, payload or inter-chunk silence yield is inside that try/catch and must close the owner, even if the underlying bridge producer already finished. The bridge's incomplete model close and CTTS's subsequent owner close are separate source calls. Payload writes yield separately and execute sleep(0) when resumed; inter-chunk silence uses one combined length/payload yield. Generation kwargs can fail during stream_native_audio argument binding before a bridge is created. Empty prepared text still resolves the voice and emits header/terminator. These outer-generator boundaries are the next differential target.

Real-model integration verification: the existing five-request/six-buffer Qwen CPU comparison passes through the new bridge in 207.39 seconds under concurrent suite load, preserving exact float32 hashes and the 72-frame stream/tail boundaries. The full buffered CTTS model comparison also passes after the iterator ownership refactor (95.23 seconds). These results establish the captured worker/bridge integration, not CTTS outer streaming or endpoint completion.

Full verification: all 93 enabled CTest targets pass in one fresh parallel run (533.30 seconds), recorded at /tmp/ezlocalai-stream-bridge-full-tests.log. The bridge source target passes in 0.44 seconds and its 11-process-case target in 0.29 seconds. The real Qwen comparison now exercises the bounded bridge and retains all exact source audio hashes; buffered CTTS, CPU/GPU STT, HTTP, LLM, media, cache and lifecycle suites also pass. The complete CXX-only Debug rebuild is warning-free, with CMAKE_C_COMPILER=/nonexistent-c-compiler and the pinned CUDA 12 runtime retained. Native executable/worker/bridge/process/model-test dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. No first-party C/Python/shell/JavaScript sources were added, whitespace checks pass, and the reference checkout is unchanged at the pinned revision. Strict parity enforcement correctly remains failed with 37 implemented/59 partial rows covering all 85 source routes. The full 1:1 goal remains active and incomplete.


## Complete native CTTS byte stream composition

- Added Ctts::generate_stream with lazy byte iteration over the native bounded bridge: source text/language/voice preparation, stream chunking, little-endian header/length/terminator writes, PCM conversion, bounded payload writes, inter-chunk silence and drain scheduling. The stream retains shared CTTS state and releases its state/audio buffers on close or exhaustion.
- Preserved close/cancellation boundaries: the header and terminal zero are outside chunk teardown; length/payload/silence yields are inside it. Incomplete bridge close and CTTS owner close remain distinct calls with source error precedence. Streaming never invokes buffered CPU fallback. Sleep(0) occurs after each payload when iteration resumes. Large silence settings preserve decimal division/overflow behavior; finite/infinite drain waits remain interruptible without native clock overflow.
- Captured 143 unchanged-source scenarios, comparing exact write bytes/boundaries, sleep requests, producer calls, caller/worker close placement, cleanup events and final model/rate/controller state. This includes cancellation injected at every yield position, kwargs collisions, empty/invalid audio and paths, sample-rate/header errors, silence and cleanup failures. The initial complete comparison passes in 0.65 seconds.
- Captured six full CTTS operations using actual CPU Qwen inference and compared all 103 emitted write hashes/lengths, exact errors and state. Coverage includes header-only close, empty text, two-chunk streaming with silence, subsequent complete generation, abandonment after a frame length and the resulting closed-owner buffered error. The targeted model comparison passes in 96.16 seconds while a rebuild runs concurrently.
- Added nine native process cases for completion, blocked synthesis cancellation, regular/huge/infinite drain cancellation, payload abandonment, stream-held owner lifetime and unstarted cancellation. Repeated cancellation requests exercise teardown shielding and worker reaping. Provenance and limits are documented in tests/fixtures/ctts_stream_README.md.

The native pull stream still needs asynchronous transport/disconnect integration, TTS pools/voice guards and public speech endpoint wiring. Every scheduler/concurrent-mutation interleaving, arbitrary Python/JSON boundary, logging, GPU synthesis and complete deployment packaging remain incomplete or unverified. Speech manifest rows remain partial and the original full parity goal remains active.

Final targeted checks after the long-drain fix: all 143 source stream cases pass in 0.65 seconds and all nine native process cases in 0.43 seconds. The finite timer implementation waits in bounded intervals instead of converting arbitrarily large seconds into an overflowing native deadline; both 1e300-second and infinite drain waits are exercised by prompt repeated cancellation. Decimal silence conversion is locale-independent and closed streams release retained PCM/text buffers.

Next pool evidence: Pipes._get_tts first reuses the current request lease, then an available instance, then resident/transient load according to voice preload policy. A full resident pool can reuse its first model. Resource admission is queried with 4.0 GiB; force_cpu overrides the recommended device, while CPU fallback policy can reject an unforced CPU recommendation. Registration includes resident and transient models and counts 4.0 GiB for every device containing "cuda". Unlike the STT pool, TTS release checks the controller's _closed flag before returning a resident model to the available deque. _transfer_tts_lease moves the HTTP task's model lease to StreamingResponse's consumer task. Forced/no-lease destruction and voice handoff have their own cleanup ordering. These rules need native TTS pool composition with the existing shared VoiceSlotGuard before public speech integration.

Full verification: all 96 enabled CTest targets pass in one fresh parallel run (589.97 seconds), recorded at /tmp/ezlocalai-ctts-stream-full-tests.log. The new 143-scenario source stream comparison passes in 0.70 seconds, and the complete six-operation/103-write actual CPU stream comparison passes in 117.72 seconds under concurrent model load. The nine native process cancellation/lifetime cases also pass, including huge/infinite drain waits. Buffered CTTS, bridge/worker ownership, exact Qwen output, CPU/GPU STT, HTTP, LLM, media, cache and lifecycle regressions remain green. The full CXX-only Debug rebuild is warning-free with CMAKE_C_COMPILER=/nonexistent-c-compiler and the pinned CUDA 12 runtime retained. Native executable/worker/new-test dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. No first-party C/Python/shell/JavaScript sources were introduced, whitespace checks pass, and the reference checkout remains unchanged at the pinned revision. Strict parity enforcement correctly remains failed with 37 implemented/59 partial rows covering all 85 source routes. The original full 1:1 parity goal remains active and incomplete.


## Native TTS pool, resource accounting and consumer lease ownership

- Added TtsPool with native Ctts factories, explicit request leases and resident/transient/available collections. Loading preserves the ignored admission boolean, explicit 4.0 GiB requirement, force-CPU override and CPU-fallback rejection. Resource registration counts each CUDA-labelled model, preserves active counts and reads the model ID live. Model-ID normalization is shared with the existing worker/runtime policy.
- Preserved current-key reuse, FIFO availability, full-pool first-instance reuse, null models when dependencies are unavailable, ordered transfer/overwrite, missing-owner errors and source partial state after failures. Release checks closed models before returning them to the pool. Forced global cleanup deliberately retains other active lease entries and ordinary releases preserve source stale available entries.
- Added source-ordered synchronous cleanup and shared FIFO scheduling, with handoff forcing synchronous destruction. Native callbacks retain Ctts ownership, expose buffered/streamed synthesis and metadata, and compose with existing VoiceSlotGuard permits and cross-thread handles.
- Captured 748 unchanged-source scenarios/22,353 operations. Each checks exact state, errors and callback order, including clocks and device/closed getters. Source ResourceManager methods and model-ID resolver execute in the capture. Ordinary exceptions, AttributeError, selected BaseExceptions, live configuration changes and deterministic random sequences are covered. The initial targeted comparison passed in 7.46 seconds.
- Six native worker-process cases cover resident/transient/handoff buffering, stream completion and blocked-stream cancellation, explicit consumer lease transfer, waiting voice requests, cross-thread slot release, resource removal and child reaping. These passed in 0.29 seconds. Concurrent key reuse/capacity, shared FIFO cleanup and reentrant resource-mark transfer are also checked; the latter retains the resolved model before invoking the callback to avoid invalidated lease iterators.

Provenance and limitations are in tests/fixtures/tts_pool_README.md. Public speech API routing, remote/gTTS fallback, full shared-GPU handoff orchestration and asynchronous transport/disconnect integration remain incomplete. Arbitrary Python object/key boundaries, all concurrency schedules, logging, queue-level BaseException termination, GPU pool inference and native deployment packaging are not established by these tests. Speech rows remain partial and the original 1:1 parity goal remains active.

Next endpoint evidence: buffered app.py speech holds the voice guard around get/generate/base64 decode and gTTS fallback, then performs an explicit resource decrement before optional pool destruction; dedicated voice mode skips that destruction. Streaming acquires the voice slot and loads before committing HTTP 200, transfers the model lease into the consumer task, closes the stream in aclosing, decrements resource use, synchronously destroys only for non-handoff leases, and always releases the slot. Its remote disabled/voice/router branches have distinct exception and availability behavior; general fallback is present for buffered speech but absent from this streaming endpoint. NativeServer currently preserves older forwarding paths and reaches 501 for local speech. A TtsService/endpoint layer should capture those unchanged source branches and cleanup precedence before wiring deferred transport and WebSocket behavior.

Verification: the complete CXX-only Debug rebuild passed without compiler warnings, with CMAKE_C_COMPILER=/nonexistent-c-compiler and the pinned CUDA 12/libsndfile selection retained. The full 98-target parallel run completed in 881.56 seconds with 96 passes and two failures, recorded at /tmp/ezlocalai-tts-pool-full-tests.log. Both new TTS pool targets passed (13.06 seconds for all 748 scenarios/22,353 operations and concurrency checks; 0.45 seconds for six process/guard cases). Complete CTTS streamed and buffered real-model comparisons passed with exact hashes, as did CTTS single-sample/runtime, CPU/GPU STT, media and other lifecycle checks.

The existing llama_tts_model_reference reached its 300-second limit under parallel model load without reporting an output mismatch. Its unchanged exact-output comparison passed sequentially in 123.94 seconds (/tmp/ezlocalai-tts-pool-isolated-tests.log). The existing LLM fallback fixture returned local output in one parallel case; its fixed 110 ms queue-holder window depended on scheduling. The fixture now synchronizes release to the remote request for recovery cases and to response completion for successful fallback cases, retaining the original response/deadline/local-execution assertions. An initial request-only synchronization attempt exposed premature local execution after successful fallback and was corrected. The final rebuilt target passed all 178 source API cases and 816 precheck cases in 5.46 seconds (/tmp/ezlocalai-tts-pool-fallback-final-tests.log). Thus all 98 enabled targets have passing verification across the full run and targeted follow-up; this is not a clean 98/98 parallel-run claim.

Native executable/worker/new-test dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. No first-party C/Python/shell/JavaScript sources were introduced; whitespace checks pass and the source checkout remains clean at the pinned revision. No TTS worker processes remain. Strict parity enforcement still correctly fails with 37 implemented/59 partial rows covering all 85 source routes. Full 1:1 parity remains incomplete and the original goal remains active.


## Native local speech request service and source cleanup precedence

- Added TtsService to compose native TtsPool, VoiceSlotGuard and ResourceManager for local buffered and streamed speech. Buffered requests decode source-compatible base64 and apply the exact gTTS eligibility predicate with live configuration. Fallback generation is an explicit required transport callback; this does not implement gTTS networking or wire public routes yet.
- Buffered cleanup performs the source explicit usage decrement, optional destruction outside dedicated voice mode, and guard exit with finally exception precedence. Streaming acquires and loads before returning its handle, transfers the model lease on first consumer iteration, closes the inner stream, then applies source resource/destroy/slot release order. Handoff cleanup runs through the guard. Unstarted close, pre-header BaseExceptions, and resource-mark failures preserve the source's observable retained leases/permits.
- Native stream state retains the service/model and permits after the public service handle is dropped, while explicit cancellation can interrupt a blocked Ctts stream. Service cancellation removes waiting admission; an active stream retains its permit until its own cleanup. Integer request identities can be transferred between threads without thread-owned semaphore release.
- Captured 1,036 unchanged-source scenarios/5,628 operations spanning endpoint local blocks, original pool/resource/voice guard methods, exact results/errors, callback order and state after each operation. Coverage includes four residency modes, gTTS policy, bad base64, null models, cancellation, BaseExceptions and compound cleanup failures. The targeted source comparison passed in 3.15 seconds; regenerating it after extending the capture harness produced identical uncompressed bytes.
- Twelve native worker-process cases cover buffered WAV decoding, streamed completion and blocked-stream cancellation in preload, dedicated, transient and handoff modes, including a queued following request and complete child reaping. Additional native checks cover queued admission cancellation, retained stream ownership, idempotent close and failed construction. Initial twelve-case verification passed in 0.60 seconds; the additional checks are included in the full rebuild.
- Added an actual-source/real-Qwen capture of 77 service operations and 70 exact raw-WAV/packet hashes: buffered output, a cache hit, two-chunk streaming, abandonment after a length packet, replacement of the closed model and final unload. The native service comparison passed in 100.22 seconds, including pool/slot/resource/model snapshots and event order. The external capture initially stopped at the wrong nested script delimiter; this harness error was corrected before accepting model fixtures. No model/source behavior was changed to fit it.

Provenance and limits are in tests/fixtures/tts_service_README.md. Public HTTP/WebSocket routes, native gTTS plus ffmpeg fallback, remote provider routing, complete shared-GPU residency handoff and asynchronous transport/disconnect integration remain unfinished. Arbitrary Python object/type/key boundaries, exhaustive scheduling, logging, GPU synthesis and deployment packaging remain incomplete or unverified. Speech rows stay partial, and the full 1:1 parity goal remains active.

Next fallback evidence: the pinned source app calls _gtts_language, which accepts English/Russian aliases and otherwise chooses Russian only for U+0400..U+04FF text, then constructs gTTS(slow=False), writes all MP3 parts and runs ffmpeg with pipe input/output, 24000 Hz, one channel and WAV format. The installed reference dependency is gTTS 2.5.3; source requirements install gTTS separately. Its default preparation strips/preprocesses text, splits/minimizes to 100 Unicode characters, packages nested JSON in the jQ1olc batchexecute form body, sends separate requests sessions using environment proxies and no timeout, and extracts base64 MP3 from response lines. A native fallback must capture these transformations and response/error behavior plus actual ffmpeg conversion before production TtsService/HTTP wiring can claim the same fallback semantics. No external Google request was made during this inspection.

Full verification: all 101 enabled CTest targets passed in one two-job run (949.86 seconds), recorded at /tmp/ezlocalai-tts-service-full-tests.log. The new source service comparison passed in 2.97 seconds, twelve process cases plus ownership/admission checks in 0.48 seconds, and the 77-operation/70-hash actual-model service comparison in 96.82 seconds. Existing buffered/streamed CTTS, Qwen wrapper, CPU/GPU STT, HTTP, LLM, media, cache and lifecycle suites also passed. The prior Qwen timeout did not recur (91.36 seconds), and the corrected LLM fallback fixture passed in 4.98 seconds.

Final review found that explicitly selecting automatic consumer identity more than once could allocate a different ID after lease transfer. The setter now preserves its previously selected identity when called with zero. A native regression verifies the same lease is removed and its permit released after repeated automatic selection. The complete final rebuild passed; both source and process service tests then passed again (3.50 and 0.56 seconds; /tmp/ezlocalai-tts-service-final-tests.log). The full-suite result above precedes this isolated identity correction; final targeted checks cover the changed behavior. Active-stream cancellation remains per-stream; service cancel stops waiting/new admission.

Both full CXX-only Debug rebuilds are warning-free, with CMAKE_C_COMPILER=/nonexistent-c-compiler and the pinned CUDA 12/libsndfile dependencies retained. Native executable/worker/service-test dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. Versioned/unignored first-party executable sources remain C++; the only C file encountered in an unrestricted directory scan was an ignored, generated CMake compiler-identification artifact in the old build directory. Whitespace checks pass, the reference checkout remains clean at the pinned revision, and no TTS worker processes remain. Strict parity enforcement still correctly fails with 37 implemented/59 partial rows covering all 85 source routes. The original full 1:1 goal remains active and incomplete.

## Native gTTS fallback preparation, transport and conversion

- Added C++ gTTS 2.5.3 language selection, Unicode preprocessing/tokenization, nested JSON/form RPC request packaging, response-line audio extraction and source error messages. Preparation finishes before network activity, and MP3 parts are concatenated before conversion. `tts_gtts_operations` supplies the required TtsService fallback callback while preserving an explicitly provided callback.
- Captured 3,230 unchanged-source/dependency cases: 180 language choices, 2,580 exact request preparations and 470 response/error cases. Coverage includes all ASCII characters, Unicode and 100-character boundaries, abbreviations, randomized text, malformed UTF-8/ASCII/base64 and HTTP errors. Actual FFmpeg output matches complete WAV hashes and sizes for one and two MP3 buffers; empty input matches the real converter exception.
- Native curl transport uses direct argument vectors, private temporary files, no shell and no Python. Fourteen reference loopback scenarios capture 55 actual HTTP requests, including POST-to-GET conversion and content-header removal on redirects, 307/308 body retention, redirect cookies, fresh sessions per text part, gzip, redirect limits and failures. Curl's user configuration is disabled. No external Google request was made.
- Twelve source converter-process cases cover four exit statuses and eight terminating signals; native error formatting preserves CalledProcessError categories/messages. Missing executables and refused connections are checked separately. Final review identified Python's restoration of ignored SIGPIPE/SIGXFZ/SIGXFSZ handlers before launching FFmpeg; the process helper now offers that opt-in behavior, and the regression runs with SIGPIPE ignored in the parent. Existing process-helper callers retain their previous defaults.
- Reference-only capture scripts remain under /tmp; shipped executable code and tests are C++. The initial mocked response capture lacked requests metadata and consumed-content state; those harness errors were corrected before fixtures were accepted. Details, hashes and limits are in tests/fixtures/gtts_README.md.

The first complete new reference/HTTP run passed (13.51 and 2.18 seconds); converter exit/signal cases then passed in 4.46 seconds before the final ignored-SIGPIPE regression. Public HTTP/WebSocket wiring, complete handoff orchestration and endpoint integration remain pending. Exhaustive requests-versus-curl proxy/TLS/cookie/auth/URL behavior, external Google behavior, other allowed gTTS/FFmpeg versions and resource-limit differences remain unverified. Speech rows remain partial and the full goal remains active.

Final verification: the complete CXX-only Debug rebuild passed without warnings (/tmp/ezlocalai-gtts-verified-build.log). All 25 selected regression targets passed in 138.10 seconds (/tmp/ezlocalai-gtts-regressions.log), including both new gTTS targets, actual-model TtsService, service/pool/voice lifetimes, streamed/buffered CTTS, audio bridge process cleanup, native HTTP, model/hub downloads, image/video/music subprocess users and parity gates. The 3,230-case gTTS comparison plus real FFmpeg conversions passed in 13.28 seconds; the HTTP/process target, including ignored-parent-SIGPIPE restoration and all twelve child failure cases, passed in 3.67 seconds. This is a targeted 25/25 run, not a new full 103-target run; the preceding complete 101-target result is recorded above.

The invalid C compiler setting and pinned CUDA 12/libsndfile remain in effect. Native executable and new test dependencies have no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. The reference checkout remains clean at the pinned revision, first-party executable source files remain C++, and whitespace checks pass. Strict parity enforcement continues to fail as intended: 37 implemented/59 partial rows, all 85 source routes covered. No native gTTS workspaces or TTS worker processes remain after verification.

## Public local buffered and streamed speech

- NativeServer now accepts a native TTS pool factory and request hooks, composes TtsService with shared voice admission/resource management, applies TTS capacity/preload/handoff policy, and supplies the native gTTS fallback by default. The actual main executable provides the native CTTS/LlamaTts factory and optional LibTorch hardware hooks. Existing injectable servers may omit a factory.
- Local POST /v1/audio/speech returns decoded WAV bytes. Local POST /v1/audio/speech/stream acquires its slot/model before HTTP 200, transfers the lease to the consumer, emits source binary metadata and HTTP chunks, and closes/releases the service on completion. A disconnect observer cancels a blocked underlying worker. Post-header failures truncate the transfer instead of sending a successful final chunk.
- Captured 912 scenarios from unchanged full source TextToSpeech/endpoint definitions in a real FastAPI ASGI application with controlled models/providers/authentication. The native test makes 1,824 buffered/deferred comparisons of validation, response metadata/body, transfer completion, generation calls, admission and resource cleanup. The capture found and corrected source-null-body handling and raw (untrimmed) TTS_ENABLED comparison; malformed JSON now reports its actual decoder position/reason.
- Four actual native-worker tests block generation, disconnect the consumer, confirm worker reaping and admit a queued empty stream in transient/preloaded/dedicated/handoff modes. The real native executable also matches two exact source WAV hashes (including a cache hit) and all 65 source stream packet hashes over TCP using the pinned CPU Qwen model. All three initial targets passed in 70.57 seconds (source API 1.83 seconds, process 0.48 seconds, actual TCP/model 70.09 seconds).

Evidence and boundaries are in tests/fixtures/tts_api_README.md. These captures control remote providers and do not certify router/unserved-capability or data-URI routing. WebSocket/local voices, complete model-management and cross-modality residency/handoff, all shutdown/cancellation schedules, GPU synthesis and deployment discovery remain incomplete. Existing remote/data-URI adapters retain earlier limitations. Speech rows remain partial; the full goal remains active.

Full verification: the complete CXX-only Debug rebuild passed without warnings (/tmp/ezlocalai-tts-api-full-build.log), followed by all 106 enabled CTest targets passing in one two-job run in 802.89 seconds (/tmp/ezlocalai-tts-api-full-tests.log). New source HTTP cases passed in 1.50 seconds, four disconnect/worker cases in 0.40 seconds, and the real native executable's two WAV/65-packet comparison in 48.49 seconds. The existing 77-operation real TtsService capture passed in 87.00 seconds and the existing main-process STT comparison in 7.72 seconds. Existing Qwen, buffered/streamed CTTS, CPU/GPU STT, HTTP, LLM, fallback, media, embedding, cache and lifecycle tests all passed.

The source checkout remains clean at its pinned revision. Whitespace checks pass and versioned/unignored first-party executable sources remain C++. CMAKE_C_COMPILER=/nonexistent-c-compiler and the pinned CUDA 12/libsndfile selections remain in effect. Native executable dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. No test TTS workers remain. Strict parity enforcement still fails correctly with 37 implemented/59 partial rows and all 85 source routes covered; this full test run does not establish full 1:1 parity.

Next routing evidence: /tmp/ezlocalai_tts_routing_oracle.py captures 2,010 unchanged-source provider-selection scenarios to /tmp/ezlocalai_tts_routing_cases.jsonl (SHA-256 9c414f38358041e15ec82554273e26c11cf5d9a169d10218915bc48fc0b0739a). This is a reference-only draft, not an accepted native comparison. Source disabled/unserved TTS may dispatch to ROUTER_URL; voice-server empty buffered output falls through whereas empty router/fallback output returns HTTP 200. Voice stream errors are swallowed into a completed empty stream, while router stream errors truncate after HTTP 200. General fallback is buffered-only. Unrecognized data: text still follows normal provider routing, unlike the existing native blanket bypass. Source remote streaming begins its upstream POST after committing downstream headers; the older native forwarding helper opens upstream first. Client configuration caching, router availability/auth policy and non-JSON request content types also need alignment. These findings define the next work without changing the completion target.

## Public TTS provider selection and remote streaming

- Added a C++ routing layer used by the public buffered/streaming speech endpoints. It preserves disabled-TTS router forwarding, local-capability gating, dedicated voice mode, voice/router/general priority, empty-result differences, buffered-only general fallback and lazy streaming POST execution. Data-URI branch selection follows the source; the older PDF/audio pipeline bodies remain partial.
- Captured 2,124 unchanged-source ASGI routing scenarios, comparing response metadata/bytes/completion and the entire provider/model operation sequence. Added 94 loopback scenarios using the actual unchanged VoiceServerClient/EzlocalaiClient implementations: 188 native buffered/deferred comparisons include exact upstream methods, paths, header maps and JSON payload bytes. Default language is now auto, with source JSON whitespace/ASCII escaping and the pinned source-client HTTP headers.
- Extended the shared HTTP body decoder with incremental draining while retaining compression/framing state. Remote TTS now uses the existing native deadline/redirect/cookie/content-decoding transport. Voice stream failures end normally; router failures abort the transfer after any emitted bytes. An interruptible duplicate socket descriptor avoids late cancellation touching a reused descriptor. Eight additional native cases cover blocked remote requests cancelled by client disconnect or application destruction.
- Integrated source client settings capture and router health checks; voice health is shared with existing voice routes. The older native HTTP fixtures now create fresh applications when voice settings change, expect source JSON packaging, and keep streaming request connections open while consuming output. Their first failures exposed the separate health cache and outdated fixture assumptions; the final native_http run passed in 31.79 seconds.
- The first source redirect fixture closed connections without advertising Connection: close, exposing aiohttp connection reuse races. Explicit close responses were captured before acceptance; reuse races remain documented as unverified. See tests/fixtures/tts_routing_README.md for both fixture hashes, source/package pins, capture scripts and boundaries. All shipped implementation and tests are C++.

The routing reference, 188 remote comparisons/eight cancellations, local API reference and shared HTTP deadline/compression/redirect/LLM-fallback targets pass. The complete CXX-only Debug build passed without warnings, retaining CMAKE_C_COMPILER=/nonexistent-c-compiler. Strict enforcement still correctly reports 37 implemented/59 partial, 96 rows, all 85 source routes covered. Complete data-URI pipelines, WebSocket/local voices, client cache/configuration transitions across all routes, DNS/connect/TLS cancellation, keep-alive reuse, GPU synthesis and packaging remain pending; the full 1:1 goal remains active.

Final verification: the complete 108/108 suite passed in 801.97 seconds (/tmp/ezlocalai-tts-routing-full-tests.log). The real native speech executable again matched both WAV hashes and all 65 source stream packet hashes; GPU STT, real TTS/STT service/pool models and shared transport regressions passed. A final test-only refinement requires body-phase cancellation to follow a complete delivered audio chunk and verifies that application destruction omits the HTTP success terminator even for voice-server streams; its rebuild was warning-free and all 188 remote comparisons/eight cancellation cases passed again in 3.07 seconds. No implementation changed after the full-suite run. Final whitespace checks pass, the pinned reference checkout remains clean, no test/server/TTS worker processes remain, and versioned/unignored first-party executable sources are C++ only. Native executable/worker dependencies contain no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime. The goal remains incomplete.


## Native local TTS WebSocket sessions

The public speech WebSocket now synthesizes through the native local TTS pool,
including with VOICE_SERVER configured. Authentication precedes upgrade.
Composition does not reserve or unload models; ready text enters a queue bounded
to four segments. A receiver runs concurrently with synthesis; queued batches
share one source-shaped lease. Stream/lease cleanup precedence, cancellation,
per-message deadlines, one session PCM header, per-frame lengths/audio, final
zero and legacy empty binary notification are implemented in C++.

Unchanged-source evidence now includes 3,679 PCM/input cases (1,788/1,891), 352
lease scenarios/1,408 operations, 127 full endpoint/session cases compared over
254 native TCP connections, and seven concurrent sessions covering backpressure,
blocked admission/model cancellation, stream errors, close precedence and send
timeout. Thread scheduling can change successful queue batch counts; audio,
segment totals and balanced ownership are compared there, with exact prefilled
batch ownership covered separately. A further 43 actual source Uvicorn TCP
captures cover control/data frames, malformed framing, lengths, UTF-8 and close
codes. They found a native mismatch in invalid UTF-8 handling before final
continuation; incremental validation now rejects it at the source's boundary.
The former handler's extra descriptor close was removed because handle_client
already owns cleanup, avoiding closure of a concurrently reused descriptor.

The process suite extends four HTTP cases with eight WebSocket disconnect/server
shutdown cases across transient/preload/dedicated/handoff ownership. An initial
new harness run read the worker trace before file creation; the harness now
waits for it. The actual native executable comparison has already passed with
two source WAV hashes, 65 HTTP packet hashes and 67 real CPU Qwen WebSocket
frame hashes (63.46 seconds). The source capture executes the same prior HTTP
operation sequence to preserve model/cache state. Provenance, fixture hashes,
dependency pins, capture corrections and remaining limits are recorded in
`tests/fixtures/tts_websocket_README.md`.

Speech rows remain partial. Complete data-URI/local-voice routes, exhaustive
transport/scheduling/cancellation, WebSocket compression and keepalive,
model-management/cross-modality handoff, GPU synthesis and packaging remain
incomplete. Broad verification of this integration follows below; the full
1:1 parity goal remains active.

Next source evidence prepared during regression testing (not yet native parity
claims): `/tmp/ezlocalai_tts_data_audio_oracle.py` captures the unchanged
`Pipes.audio_to_audio` with controlled guards/models/resources and real source
base64/type parsing into `/tmp/tts_data_audio_cases.jsonl` (553 cases, SHA-256
`f6673b37cbbfa7485d5ac1c363c2ae7121db6e2cce611d93245ed045548a69b3`).
`/tmp/ezlocalai_tts_data_pdf_oracle.py` captures unchanged `Pipes.pdf_to_audio`
with real temporary output files and controlled pdfplumber page text into
`/tmp/tts_data_pdf_cases.jsonl` (413 cases, SHA-256
`242a794d2374c66301c952edc6cd75095a668d0ca0fa709cdd9ece171819a5aa`).
These external draft captures are not added to the native test suite yet.
Both source pipelines use local pools and explicit true/false resource marks;
ordinary generation/transcription failures skip the subsequent destroy call,
while the guard still exits. Successful dedicated mode skips destroy. TTS
pipeline results remain strings (local URL or base64), unlike the HTTP speech
service's decoded WAV. PDF title sanitization allows ASCII spaces, writes
outputs/title.pdf, joins extracted page text with newlines, and passes
output_file_name=title.wav to CTTS. The chunk_size argument is unused. The PDF
capture does not establish text extraction equivalence. Source voice uploads
write raw uploaded bytes, with no conversion; local voice listing sorts stems
of globbed *.wav paths. Existing native listing/order, flag whitespace,
client caching and filesystem edge behavior still need direct source coverage.

Final WebSocket verification: the complete CXX-only Debug build passed without
warnings, retaining CMAKE_C_COMPILER=/nonexistent-c-compiler. All 113/113 tests
passed in 818.84 seconds (`/tmp/ezlocalai-tts-ws-full-tests.log`), including the
new protocol/lease/API/session/transport cases, all 12 real worker disconnect
and shutdown cases, the extended native executable model test (66.37 seconds),
GPU STT, and native_http (31.37 seconds). The earlier focused six-target run
passed in 30.81 seconds; the complete session test passed independently too.
Strict enforcement still correctly exits 1 with 37 implemented and 59 partial
entries, 96 total rows and all 85 source routes represented. First-party
versioned/unignored executable sources are C++ only, native executable/worker
linkage has no unresolved libraries, libpython, libtokenizers or CUDA 13 runtime,
and the reference checkout remains clean. No implementation changed after the
full-suite run. The overall goal remains active and incomplete.

## Native local audio data-URI speech

The public buffered speech route now dispatches audio data URIs through native
local STT followed by CTTS. It returns the source URL/base64 JSON string,
preserves voice/local_uri values and result types, bypasses remote providers,
and does not invoke gTTS fallback. TtsService.generate_pipeline implements the
source's separate lifecycle for audio/PDF calls: explicit true/false resource
marks, raw CTTS result, destroy only after successful generation and outside
dedicated voice mode, with guard release and cleanup precedence. SttService has
the corresponding pipeline use mode and typed byte input through its pool and
real runtime. Source audio_to_audio decodes the URI, then STT base64-decodes those
bytes again; the native code preserves this quirk rather than changing it.
STT's dedicated-voice/preload checks no longer strip VOICE_SERVER whitespace.

New reference evidence: 816 TTS pipeline lifecycle scenarios/3,264 actions,
32 STT pool scenarios/128 calls, 570 complete raw-byte STT runtime cases/1,140
actions, and 630 source FastAPI endpoint/full audio-pipeline cases. These compare
results, error families, cleanup precedence, resource/pool/guard states, file
bytes, call order, model input bytes/types, response status/content type and
bodies. The STT runtime now runs 1,370 wrapper scenarios plus its existing
filesystem cases. The isolated STT pipeline and runtime comparisons passed;
the TTS pipeline, HTTP API and existing service comparisons passed together in
6.50 seconds. A source-header metadata mistake in the first TTS pipeline fixture
was corrected and recaptured before acceptance. The updated content-type API
comparison passed in 0.77 seconds, and native_http passed in 31.52 seconds.

The source real-model capture uses the same preceding buffered/HTTP/WebSocket
sequence as the executable test, then real STT and CTTS. A doubly encoded input
produces a 184,364-byte WAV with SHA-256
f4cc267240aa04dbc17e17a10c20b3044ce516744dd642f437a78ed5816066e3; a singly encoded
input fails with the source invalid-audio error. The native executable test has
been extended to verify both outcomes when STT assets are configured.
Provenance and limits are in tests/fixtures/tts_pipeline_README.md.

The public PDF adapter still uses older forwarding and pdftotext extraction;
only its shared TTS lifecycle now has source-backed implementation. Complete
PDF/voice-management routes, client caching transitions, cross-modality
handoff/concurrency, GPU TTS, model discovery and packaging remain incomplete.
Speech rows remain partial and the overall goal remains active.


The first combined executable model comparison failed at its audio hash
(230.13 seconds). Investigation found a capture dependency mismatch: faster-
whisper 1.2.1 and VAD v6 were imported from the user site while the copied
metadata declared 1.2.0. A source/native trace differed at one comma. Four/eight
thread source checks retained the difference, ruling out that initial suspicion.
Pinning the actual 1.2.0 package and matching v5 VAD assets produced exactly the
native transcript and native-observed WAV hash. The corrected source fixture
records actual package/VAD hashes and eight STT threads; the executable test
asserts those pins. No inference code changed in this investigation. The full
116-test regression run uses that corrected evidence. The rejected capture
is preserved only under /tmp for diagnosis. Support for other dependency
profiles is not inferred from these results.

The corrected executable comparison passed in the complete regression run
(146.45 seconds): two buffered WAV/cache-hit results, 65 HTTP stream packets,
67 WebSocket frames, the exact 184,364-byte STT-to-TTS WAV hash and the singly
encoded audio HTTP 500 all match. The dependency profile now explicitly matches
reference cuda-requirements.txt/rocm-requirements.txt (faster-whisper 1.2.0),
with matching v5 VAD assets and actual STT thread count. Generic requirements.txt
uses 1.0.3, covered separately by existing legacy STT model tests; deployment
profile selection remains part of the outstanding packaging audit.

Final verification for this batch: the complete C++ Debug build and final model
test rebuild were warning-free, with CMAKE_C_COMPILER=/nonexistent-c-compiler.
All 116/116 tests passed in 876.19 seconds
(`/tmp/ezlocalai-tts-pipeline-full-tests.log`), including the corrected real
audio-pipeline comparison, GPU STT, all new reference fixtures and native_http
(31.41 seconds). No implementation changed during the full regression run.
The test servers and native TTS workers exited. Strict parity enforcement still
correctly exits 1: 37 implemented, 59 partial, 96 rows, all 85 source routes
represented. The reference checkout is clean, git diff --check passes, and no
remaining versioned/unignored first-party .c, .py or .sh files exist. Native
executable/worker linkage has no unresolved libraries, libpython, libtokenizers
or CUDA 13 runtime. This batch is verified; exact overall parity remains
incomplete and the goal stays active.

## Public voice listing and uploads

The preceding audio-pipeline turn made verified progress (116/116 tests).
This batch corrects the public voice routes against the current source:
sorted Path.glob stems including hidden entries, `.wav`, directories and
symlinks; exact disabled flags; raw-byte uploads with ASCII sanitization and
100-character names; non-strict POSIX realpath for the destination directory;
following-stat collision selection; selected duplicate/missing/typed multipart
validation; and plain source HTTP 500 on filesystem failures.

Voice listing now uses the same retained client and availability cache as public
speech, rather than rereading its URL/key for each request. It decodes the
upstream JSON and reserializes the selected value, with source falsey fallback,
non-finite-number and lone-surrogate errors. A source comparison identified
the glibc/Python `latin-1` alias difference. The shared aiohttp transport also
now retries an idempotent request once on pre-response disconnect, preserving
the original deadline and redirect state, with no body-error retry.

New original-source fixtures cover 247 real-filesystem scenarios/886 API calls
and 97 real-client loopback scenarios/110 API calls/207 upstream requests. The
filesystem test compares all resulting files, bytes, directories and links.
The client test compares response status/content type/JSON or bytes, cache
request counts and complete upstream wire requests, including expiry boundaries,
backward time, retained configuration and shared speech/listing state. Native
comparisons exposed and corrected dangling-link resolution, a missing response
body read, surrogate serialization, the charset alias and idempotent retry.
The existing HTTP test's unsorted-list expectation was corrected to match the
current reference. Provenance and remaining limits are recorded in
tests/fixtures/voice_management_README.md.

The four focused targets passed in 35.88 seconds: native_http (31.58), the new
filesystem suite (0.73), the expanded remote suite (0.49) and the existing
speech remote-client suite (3.07). A complete rebuild/regression is next because
the retry and charset fixes affect shared transport. Voice rows remain partial:
arbitrary malformed multipart/charset/filename escapes, spilled UploadFile error
serialization, permissions/races, exact JSON byte formatting and complete global
startup/STT client transitions are not proven. PDF, GPU TTS, discovery and
packaging gaps remain; the overall parity goal is active and incomplete.

Final verification: the complete C++ Debug build passed without warnings,
retaining CMAKE_C_COMPILER=/nonexistent-c-compiler. All 118/118 tests passed in
863.35 seconds (`/tmp/ezlocalai-voice-full-tests.log`), including both new voice
suites, shared HTTP deadline/compression/redirect tests, GPU STT, the executable
speech-model comparison (130.58 seconds) and native_http (31.50 seconds).
No implementation changed during that run. Strict enforcement still correctly
fails on 37 implemented and 59 partial entries, with all 85 source routes
represented among 96 rows. The reference checkout is clean; first-party
versioned/unignored executable sources remain C++ only. Native executable and
worker linkage has no missing libraries, libpython, libtokenizers or CUDA 13
runtime. This batch is verified; the full parity goal remains active.
