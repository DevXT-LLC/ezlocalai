# Experimental NVIDIA GDN Value-Slice64

2026-11-11. Isolated, disabled by default, NOT qualified for deployment.
The deployed patch stack and audit source remain unchanged. The authorized 5090
controller screen below temporarily replaced the workload, then restored the
original container and Qwen/voice arguments. No runtime promotion was made.
Additional device steps below remain pending and require separate authorization.

Follow-up: the [full clean CUDA 12.8 build](experimental-prefill-gdn-slice64-cuda128.md)
now provides both `test-backend-ops` and `llama-server` for SM86/89/120a, with
completion evidence and a bounded seven-case operator gate. The CUDA 13 object
compile recorded below is the earlier screen, not the final qualification build.

## 5090 Bounded Screen

2026-11-11: completed through the parent controller after explicit confirmation
of prior-owner restoration and watchdog cleanup. Raw evidence:
[`5090-gdn-screen.jsonl`](../build/validation/prefill-fleet-20261111/5090-gdn-screen.jsonl).
Only the requested gates and 64k off/on/off screen ran; no trace, perplexity run,
additional target, or follow-up experiment was launched.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/josh/repos/xtsys/.venv/bin/python benchmarks/prefill_fleet.py \
  --host josh@192.168.1.135 --label DevXT5090 \
  --router-key-ssh josh@192.168.1.214 --wait-seconds 3600 \
  --engine-root /home/josh/prefill-experiments/gdn-cuda128-20261111/bin \
  --variants gdn-off,gdn-on,gdn-off --targets 64000 \
  --output build/validation/prefill-fleet-20261111/5090-gdn-screen.jsonl
```

RTX 5090, driver 590.48.01, CUDA 12.8 build. All 24 staged executable/library
file hashes matched the pinned build (`gdn_build_verified=true`), including
server `b19b0416...`, operator binary `6bc78f74...`, and CUDA library `32a64b12...`.
Full hashes are in the raw provenance and linked CUDA 12.8 build document.
Same Q3_K_XL model, logical batch 4096, physical batch 2048, MTP enabled,
and original voice engine throughout; ready-state free VRAM was 3381 MiB.

Compact operator fixture: both flag modes `0` and `1` returned exit `0` and
exactly seven expected supported CUDA0 test rows with empty `error_message`.
The current CSV printer has no `passed` field. Each expected `op_params` string
uses this exact order (substitute the tuples below):

```text
type=f32,head_count=16,head_size=128,n_seq_tokens=T,n_seqs=S,v_repeat=R,permuted=P,kda=0,K=1
(T,S,R,P) = (63,1,2,0), (64,1,4,0), (65,1,1,0), (65,3,4,0),
            (128,1,2,0), (129,1,2,0), (129,2,4,1)
```

All 14 checks passed without changing tolerance. This is bounded operator
coverage, not broad model-quality or state-continuation qualification.

| Block / flag | Sample | TTFT (s) | Prompt (ms) | Prompt / cached tokens |
| --- | --- | ---: | ---: | ---: |
| 0 / off | cold | 21.778362 | 21748.264 | 64183 / 0 |
| 1 / on | cold | 21.947789 | 21917.967 | 64183 / 0 |
| 2 / off | cold | 22.365190 | 22334.293 | 64183 / 0 |
| 0 / off | cache repeat | 0.103297 | 73.590 | 4 / 64179 |
| 1 / on | cache repeat | 0.101571 | 71.620 | 4 / 64179 |
| 2 / off | cache repeat | 0.107434 | 75.364 | 4 / 64179 |

All six responses for the single infra fixture passed with identical prompt and output hashes and
483 predicted tokens. Cold TTFT improved only 0.562% versus the disabled mean
(22.071776 s), while the disabled brackets drifted 2.695%. Enabled was 0.778%
slower than the first disabled run and 1.866% faster than the second. Prompt
latency likewise improved only 0.559% versus the bracket mean. Cache repeats
are not prefill evidence. There is one cold sample per block, not a repeatability
study. **No clear >=3% gain: stop here; do not expand or promote.**

This measures the requested flag off/on behavior only. Without a fresh candidate
kernel trace, actual `gdn_single<16,64,1>` dispatch remains unverified. The parent
reported GDN at roughly 4.5% of profiled prefill kernel time, suggesting limited
upside; that is not a measured end-to-end bound or a trace of this candidate.

The final evidence row is `restored`, with `original_argv=true`,
`original_container=true`, and the original immutable image
`sha256:b7e7d157c93827b563cbda3f1e331343b17d0984afb370363881624e3198689d`.
The controller exited 0 and reported `restored=true`, `drain_released=true`,
`renewal_failed=false`, after its original-service health check. Its remote child
finished, owned probe containers were removed, and the owned watchdog was
terminated and waited for. No child from this run remains pending. The main
fleet report and other owners' queues were not modified.

## Candidate

Apply `patches/experimental-prefill-gdn-slice64.patch` only to a private copy of
the audited v0.9.0 tree. Set `GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=1` before starting
the process. Only the exact string `1` enables the experiment; unset, `0`, empty,
and unrecognized values preserve the existing dispatch. The value is cached on
the first eligible NVIDIA dispatch, so restart between modes, including when
using CUDA graphs. Failure to initialize slice64 falls back to the existing
selector, not an unguarded launch. A profiler must confirm that slice64 ran.

This forces the existing `gdn_single<16,64,1>` template for already-eligible NVIDIA
operations, using two CTAs per value head instead of four (V32) or one (V128).
It does not introduce a new automatic occupancy heuristic. The experiment is
most interesting near the existing V32/two-block versus whole-head boundary;
force mode also allows a controlled screen on either side of that boundary.

Unchanged admission: scalar gate, 128 key/value dimensions, H_k=16,
H_v in {16,32,48,64}, >=64 tokens, valid sequence ratio/grid bounds, and the
existing SM86/89/120/121 allowlist. No MMQ, KDA, decode, pre-Ampere, HIP, or MUSA
policy is changed. The kernel body and initialization helper are byte-identical.
FP32 recurrent state, BF16 high/residual packing, `lo*hi + hi*lo + hi*hi`
products, accumulation order within each product, token chunk size 16, gate
scan, triangular solve, state writeback, and input/output layouts are preserved.
No residual term is removed. V64 uses the existing generic projection path,
not V32's prefetch/split-projection specialization. Different ownership and
compiler register allocation can still change rounding; device quality is NOT
proven by source equivalence or compilation.

SM120 ordinary quant MMQ was not changed: `mmq-config-blackwell.cuh` specializes
MXFP4/NVFP4 and falls through to Ampere configs for ordinary quants. Altering those
tables has a wider dispatch/instantiation surface than this one GDN specialization.

## Source Provenance

Read-only audit root:
`/home/josh/repos/xtsys/ezlocalai/build/validation/llama-v090-cuda-stage/source`.
Git HEAD: `cd959bae99ff4bed22bef2604f45241c09bbb8c8`.
The audit tree includes local patches; HEAD alone is not a sufficient pin.

| File | SHA-256 before experiment |
| --- | --- |
| `ggml/src/ggml-cuda/gated-delta-net-mma.cu` | `bda5c4530c74a62ca1993d317de852635520dde7904cb199088a5dab1e9ae751` |
| `tests/test-backend-ops.cpp` | `9e258a11b819e947570b5a08e0ae5786dc29b19ea5198ac7bb8ef4a7da51d9d4` |

The host checks enforce these hashes and apply/reverse-check both patches in a
temporary directory. They do not modify or build in the audit source directory.
These patches are for that exact audit tree, not an arbitrary vanilla v0.9.0.

## Checks Completed

- 10 CPU-only tests passed: real selector compiled with mocked device/init APIs
  for NVIDIA, HIP, and MUSA; strict opt-in; stable per-process setting; default
  equivalence across occupancy boundaries; all admission guards; init failures;
  unchanged math; launch geometry; exact state-column ownership; test integrity;
  and audit source hashes. These are structural/dispatch tests, not GPU numerics.
- Patched `test-backend-ops.cpp` passes `g++ -std=c++17 -fsyntax-only`.
- Only the affected CUDA translation unit compiled with host NVCC 13.0.88,
  C++17, O3, original fast-math policy, graphs enabled, and targets
  SM86, SM89, SM120a. No link, backend initialization, GPU context, or GPU execution.
  This is not a CUDA 12.8 toolchain qualification or a native GB10 build.

Recorded replay: `/tmp/experimental-prefill-gdn-build.rVZCYK` (1.5 MB total).
`compile.log` contains the PTXAS output; `operator-syntax.log` is empty on success.
The replay passed all 10 host tests in 1.286 seconds. Compiler/syntax logs contain
no errors or warnings; shell syntax and Black formatting checks also pass.

PTXAS resource evidence from that compile (registers; spill store/load bytes):

| Specialization | SM86 | SM89 | SM120a |
| --- | --- | --- | --- |
| V32, BLOCKS=1 | 190; 0/0 | 190; 0/0 | 253; 0/0 |
| V32, BLOCKS=2 | 128; 24/24 | 128; 24/24 | 126; 0/0 |
| **V64, BLOCKS=1** | **188; 0/0** | **188; 0/0** | **247; 0/0** |
| V128, BLOCKS=1 | 255; 0/0 | 255; 0/0 | 255; 40/40 |

Dynamic shared memory from the unchanged layout is 38,016 / 56,448 / 93,312
bytes for V32 / V64 / V128. V64 halves duplicated Gram/inverse work versus V32
and doubles head-level parallelism versus V128, but uses more resources than V32
and loses its specialized overlap. These are hypotheses, not measured speedups.
No numerical, sanitizer, model-quality, or throughput claim is made.

Reproduce the small checks from this repository, without GPU access:

```bash
/home/josh/repos/xtsys/.venv/bin/python benchmarks/experimental-prefill-gdn-slice64-test.py
bash benchmarks/experimental-prefill-gdn-slice64-build.sh
```

The build script retains its private source copies, object, and logs under the
printed `/tmp/experimental-prefill-gdn-build.*` path. It compiles one `.cu`, not
the full CUDA backend or worker. Headers are read from the audited tree.

## Important Audit Findings

1. **The stock performance cases miss MMA.** Their key-head count is 32 or 4,
   while the selector requires H_k=16. An unfiltered stock `perf -o GATED_DELTA_NET`
   is not evidence about this experiment. The separate
   `patches/experimental-prefill-gdn-slice64-tests.patch` adds 32 eligible perf
   cases: H_v=16/32/48/64, sequences=1/2, tokens=64/128/512/2048. It also adds
   seven correctness cases covering 63-token fallback, exact chunks, tails,
   permuted inputs, KDA fallback, and K==tokens snapshot fallback. Existing
   normalization, random inputs, CPU comparison, and error limits are untouched.
2. **Inherited snapshot risk; accept only K=1.** In audited
   `gated_delta_net.cu:276`, MMA eligibility tests `K < n_tokens`, not `K == 1`.
   In `gated-delta-net-mma.cu:357`, writeback stores only the final state, whereas
   `ggml.c:6502` allocates K state slots. Thus an otherwise eligible K=2, T=64
   operation appears able to enter MMA without producing its older snapshot.
   This predates the experiment and is unchanged here. Do not infer MTP snapshot
   correctness from K=1 prefill results. Reproduce K=2,T=64 with H_k=H_v=16,D=128
   on the unmodified baseline first; if reachable, fix/qualify that separately
   before any production promotion. This is a static finding, not a GPU repro.

## Later Operator Build

Do not reuse the audit build directory or relink its deployed libraries. After
the parent's measurement window, copy the source (not its old CMake cache) to a
new temporary build root and apply both patches there. For a first RTX 5090
operator screen, compile only SM120 and the `test-backend-ops` target:

```bash
repo=/home/josh/repos/xtsys/ezlocalai-prefill
audit=/home/josh/repos/xtsys/ezlocalai/build/validation/llama-v090-cuda-stage/source
work=$(mktemp -d /tmp/experimental-prefill-gdn-ops.XXXXXX)
cp -a "$audit" "$work/source"
cd "$work/source"
git apply --check "$repo/patches/experimental-prefill-gdn-slice64.patch"
git apply "$repo/patches/experimental-prefill-gdn-slice64.patch"
git apply --check "$repo/patches/experimental-prefill-gdn-slice64-tests.patch"
git apply "$repo/patches/experimental-prefill-gdn-slice64-tests.patch"
cmake -S "$work/source" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF \
  -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build "$work/build" --target test-backend-ops -j2
```

The last step still compiles the CUDA backend's dependencies and is materially
larger than the completed `.cu` check. Avoid it during baseline measurements.
Use a matching isolated prebuilt engine stage if available; do not transplant
this standalone object into a deployed library. Use 86-real or 89-real on those
devices; qualify SM121 with the native GB10 toolchain separately.

## Later Device Gates

Select only a reserved idle GPU by UUID (`GPU_UUID` below). Use separate processes
so the cached environment setting is unambiguous. Same binary with knob 0/1 is
the clean schedule A/B; also verify knob 0 against an unpatched build before
promotion. Archive complete logs, device/toolchain IDs, model hashes, flags,
baseline dispatch, clocks, and timings. Do not silently accept zero matched tests.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=0 \
  "$work/build/bin/test-backend-ops" test -b CUDA0 -o GATED_DELTA_NET
CUDA_VISIBLE_DEVICES="$GPU_UUID" GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=1 \
  "$work/build/bin/test-backend-ops" test -b CUDA0 -o GATED_DELTA_NET
CUDA_VISIBLE_DEVICES="$GPU_UUID" GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=1 \
  compute-sanitizer --tool memcheck --error-exitcode 1 \
  "$work/build/bin/test-backend-ops" test -b CUDA0 -o GATED_DELTA_NET \
  -p 'head_count=16,head_size=128,n_seq_tokens=129,n_seqs=1'
CUDA_VISIBLE_DEVICES="$GPU_UUID" GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=0 \
  "$work/build/bin/test-backend-ops" perf -b CUDA0 -o GATED_DELTA_NET \
  -p 'head_count=16,head_size=128' --output csv
CUDA_VISIBLE_DEVICES="$GPU_UUID" GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=1 \
  "$work/build/bin/test-backend-ops" perf -b CUDA0 -o GATED_DELTA_NET \
  -p 'head_count=16,head_size=128' --output csv
```

Require all CPU comparisons to pass at the existing 1e-7 NMSE limit, including
the returned state tail. Add same-input attention and state comparisons between
V32/V64/V128 and CPU before promotion; the stock operator initializer uses
`random_device`, so two process runs are NOT same-input A/B numerical comparisons.
Check nonzero initial state, mild gates/long recurrent accumulation, strided
inputs, sequence broadcast, multi-chunk continuation, fused-cache continuation,
and sanitizer racecheck/synccheck as well as memcheck. Those additional device
checks are pending, not covered by the host ownership model.

For profiling, the parent has `/usr/local/bin/nsys` on the host (not in the
container). Profile the isolated host operator executable or use a separately
prepared diagnostic container; do not install into or restart production.
Confirm `gdn_single<16,64,1>` actually executes and record its share of prefill
GPU time. A trace showing only AR, MMQ, or attention is a dispatch/workload
mismatch, not evidence for or against slice64. Keep profiled and unprofiled
timings separate. Run randomized alternating repetitions across actual H/sequence
occupancy boundaries, then use the parent's batch/MTP workload and quality cases
with identical settings, cold/warm state controlled, and K=1 prefill separated
from snapshot/decode work. Do not change batch size or MTP policy to manufacture
a gain. If GDN is a small fraction f of runtime, even eliminating it caps total
speedup at 1/(1-f); prioritize the measured dominant kernels instead.

Promotion requires numerical/state-continuation checks, no model-quality
regression, and repeatable end-to-end improvement beyond noise on the intended
device/workload. Until then, leave both patches outside all build/deploy lists.
