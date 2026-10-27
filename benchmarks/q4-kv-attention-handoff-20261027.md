# Q4 KV attention: profiling-backed implementation handoff

This records the profiling and initial design behind the separate
[Q4 KV fusion implementation and validation](q4-kv-fusion-20261027.md).
The default-off prototype below preceded its guarded automatic activation.
The source audited here is
ggml-org/llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab` (v0.7.0).
Preserve the deployed model, Q4 KV bytes, query precision, context and sampling.

## Evidence and attribution

One physical RTX 4090 trace of a cached roughly 130K-token finite decode contains
864 `dequantize_block_q4_0<half>` calls totaling 316.367 ms, or 24.5% of the
1.291535 s summed recorded kernel durations. Its grids are 520192 and 521216
blocks of 32 threads. The original converter emits 256 elements per block:
these are 133169152 and 133431296 values, consistent with 130048/130304 padded
positions times 4 KV heads times 256 channels. The counts also fit 27 verification
rounds times 16 target attention layers times separate K/V conversions. This
source-and-shape attribution identifies repeated KV conversion; static model
weight dequantization would not explain those growing context-sized buffers.

The model metadata records 24 query heads and four KV heads (`n_gqa=6`). The
kernel template's `ncols2=8` is a padded tile width. The initial handoff inferred
eight actual grouped query heads from that width; the prototype audit corrected
this before GPU execution. Eligibility and padded-head tests must use GQA six.

The trace also records 459 `flash_attn_ext_f16<256,256,4,8,...>` calls (279.951 ms)
and 78 `<256,256,1,8,...>` calls (43.811 ms). These observations identify a target
for investigation; the profile alone establishes no speedup. Nsight warns
that not all CUDA events may have been collected, and the deep trace reports
150 GetGraphId and 150 GetGraphNodeId `CUPTI_INVALID_PARAMETER` errors. Kernel
timings are present; complete graph association is unproven. An uninstrumented
before-control exists, but there is no matched after-control for profiler overhead.
Raw captures are under ignored `build/validation/inference-round5-20261027/profiler-v2/`.

## Exact source path and the tempting unsafe shortcut

The pinned [dispatcher](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/fattn.cu#L637)
uses the direct quantized vector kernel for at most two query rows on Ada,
and one on Ampere; three/four-row MTP verification selects F16 MMA.
[`launch_fattn`](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/fattn-common.cuh#L1031)
then converts the full active K/V buffers to F16 and substitutes F16 strides.

Do not merely raise the vector threshold. The existing
[quantized vector path](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/fattn-vec.cuh#L140)
quantizes queries to Q8_1. That changes arithmetic relative to F16 MMA despite
unchanged model/KV files. Its two-query blocks also have different GQA reuse and
reductions, so avoiding standalone conversion does not guarantee better latency.

## Smallest plausible precision-preserving prototype

This requires substantial CUDA kernel work. Add a packed-Q4 storage policy to
the existing F16 MMA tile loader, preserving the actual F16 operands consumed
by the original MMA instructions. Start with an experimental, default-off path
limited to **SM89, D_K=D_V=256, three/four query rows, GQA ratio six (24 query heads / four KV heads), one
sequence, contiguous padded Q4_0 K and V, ordinary mask, no ALiBi, softcap,
sparse gather or shared-K/V view**. All other inputs retain the original path.
This eligibility must be checked from actual tensor metadata, not model names.

The relevant [MMA implementation](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/fattn-mma-f16.cuh)
has these boundaries:

- `flash_attn_ext_f16` (1797+) derives sequence/head bases from original byte
  strides; `process_tile` (1190+) and `iter` (575+) feed `load_tile` (384+).
- The traced four-query tile uses 128 threads, 32 KV rows, 256 channels and
  two pipeline stages. Each K/V row has eight 18-byte Q4 blocks (144 bytes).
  Retain packed byte strides and map each existing 16-byte shared-memory output
  chunk to its low/high nibbles, then write the identical swizzled half2 slots.
- Match the [original converter](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/convert.cu#L85):
  `d=__half2float(scale); dm=-8*d; half(d*nibble+dm)`, with identical compilation
  and rounding behavior. Replacing it with half arithmetic or `d*(nibble-8)`
  requires proof: signed zeros, nonfinite values and fast-math/FMA can differ.
- Preserve Q loading/scaling (1277+), MMA/softmax order, masks, sinks and fixups.
  Preserve the two-stage barriers around initial K preload (1334+), V loading
  (636+) and next-K preload (994+). `cp.async` cannot perform dequantization;
  the new synchronous tile stores require an explicit race/synchronization audit.
- Register pressure affects the occupancy-derived stream-K grid. The traced
  main kernel uses 231 registers and 256 blocks. Keep the original work
  partition/reduction order; identical MMA code alone does not establish that.

The first prototype can retain the old reserved F16 workspace while bypassing
its use. That avoids changing allocation contracts during numerical validation;
it claims no VRAM saving. Any later reservation reduction needs identical,
stable eligibility in both allocation-size calculation and execution. A
persistent full F16 mirror would consume substantial extra VRAM and is not the
proposed approach.

## Required acceptance before promotion

1. CPU indexing tests must cover Q4 nibble layout, byte/head/sequence strides,
   offsets, bounds and every excluded shape. GPU loader tests must compare each
   resulting **16-bit operand** with original CUDA conversion plus loading:
   all nibble values and scale bit patterns, signed zeros, subnormals, infinities,
   NaNs with explicit handling, unaligned 18-byte blocks and guarded canaries.
2. Compare actual FA output bytes against stock MMA on randomized and adversarial
   inputs, three/four queries, short/130K/near-context padded lengths and masks.
   Include stream-K boundaries/fixups, unchanged query operands and partitioning,
   CUDA memory checking and race checking. A failure blocks a bitwise claim.
3. Only after that, run isolated same-card ABBA operator and full-engine tests,
   preserving all launch/model/KV/context/sampling controls. Include finite and
   long freeform answers, cache/rewind/disk continuations, vision and package
   gates. Sample agreement does not establish universal model-quality equality.
4. Profile the candidate to verify conversion removal, then measure uninstrumented
   end-to-end prefill/decode/TTFT with variance and resource limits. Per-tile
   unpacking/register costs may offset the saved traffic. Qualify 3090V and 5090
   separately before extending the SM89 guard; do not extrapolate this trace.
