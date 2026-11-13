# CUDA Q4 KV loading for F16 attention

## v1.0.0 Compatibility (2026-11-13)

The active pin is `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, using
`llama-cuda-q4-kv-fusion-v100.patch`. Its dense SM89 eligibility, indirect-row
fallback, operand expression and occupancy/partition guards are retained.
Upstream's bounded K/V tile loads are preserved, and `<mutex>` is included
explicitly for our one-time shared-memory configuration after upstream removed
the transitive include. FP8 and other stock attention paths keep their dispatch.

The dispatcher, fused 4x8 MMA instance, ordinary 8x4 MMA instance and 256x256
tile control compile with CUDA 12.8 for SM86, SM89 and SM120a. Symbol inspection
confirms the Q4 fused specialization. This is focused compilation, without a
GPU device or complete CUDA backend/server link. Real-model numeric/launch/speed
qualification remains pending. Evidence:
`tests/fixtures/llama_v100_upgrade_20261113_validation.json`.


## v0.9.0 Compatibility (2026-11-07)

The active pin `cd959bae99ff4bed22bef2604f45241c09bbb8c8` uses
`llama-cuda-q4-kv-fusion-v090.patch`. It retains the qualified dense SM89 path,
operand expression, occupancy fallback and stock stream-K partition. Upstream
now supports indirect KV row lists; those layouts use stock attention rather
than the dense-only fused loader, with an explicit CPU policy regression check.
All 395 layout-policy checks and full patch lifecycle checks pass.

Focused CUDA compilation passes with NVCC 12.8.93 and GNU 13.3.0 for SM86,
SM89 and SM120a (upstream normalizes the requested 120-real to 120a-real).
The production dispatcher, 4x8 MMA fused instance, 8x4 MMA control and 256x256
tile control compile unchanged. All four objects contain all three cubins;
symbol inspection confirms the q4_kv=true 256x256/4x8 specialization exists.
The compile log has no warnings or errors; configure reports the existing,
deprecated and ignored LLAMA_CURL option. No GPU devices were exposed.

The complete CUDA backend/server link, image/package gates and GPU runtime
have **not been validated**. Historical GPU evidence below applies to the
earlier pin only; SM89 stock/fused numeric and launch comparisons must be
repeated before claiming v0.9.0 GPU qualification. ROCm and native ARM builds
also remain unvalidated. The original patch is retained for removing old
installed stacks safely. Reproduction scripts and logs are under
`build/validation/llama-v090-cuda-stage/`; compact results are recorded in
`tests/fixtures/llama_v090_upgrade_20261107_validation.json`.

## Original Implementation

This managed patch targets llama.cpp pin
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. Four-query MTP verification can
expand the entire active Q4 KV cache to F16 before executing attention. The
alternative decodes each required Q4 tile directly into the existing F16 shared
memory layout, retaining the original query conversion and MMA computation.

The deployed model has **24 query heads and four KV heads: GQA six**. Its
`ncols2=8` kernel dimension is a padded tile width. Existing padding of the two
unused head columns, output bounds and fixups remain intact.

## Operand and execution contract

For each nibble, match the original CUDA converter's operation sequence:

```text
d = float32(half_scale)
dm = -8*d
operand = half(d*nibble + dm)
```

Do not substitute half arithmetic or rearrange the expression without proving
operand equality, including signed zero and nonfinite behavior. Q4 blocks have
18-byte strides; their packed bytes cannot assume wider alignment. The loader
uses scalar byte reads and writes the same eight-half, 16-byte swizzled chunks.

The original F16 query operands, Tensor Core MMA, softmax, masking, accumulation
and fixup order are preserved. The two-stage pipeline retains its waits and
block barriers; synchronous KV tile stores replace only the corresponding
global-F16 tile copies. K and V use separate shared tiles.

## Eligibility and partitioning

Activation requires actual tensor metadata satisfying all of these conditions:

- NVIDIA SM89; head dimension 256; three or four query rows; 24 query heads,
  four KV heads and one sequence.
- F32 queries, Q4_0 K and V, and an ordinary F16 mask. K/V shapes and packed
  layouts agree; context rows are a positive multiple of 256.
- Recognized dense head-major or token-major layouts, checked logical spans,
  aligned base addresses and representable kernel strides. Production KV
  token-major views use 576-byte token strides and 144-byte head offsets.
  Single-sequence views may retain a larger allocated sequence stride.
- Enough mask rows, no input-span overlap or shared K/V alias, no sinks,
  ALiBi, softcap or requested/selected sparse gather.

The candidate must support at least the stock kernel's occupancy. Otherwise
execution retains stock conversion and attention. When fusion is selected,
launch planning uses the **stock occupancy**, preserving its block partition,
stream-K grid and reduction order even if candidate occupancy is higher.
The fused kernel body is compiled for SM89; other architectures keep stock
execution. The guard is independent of model names and worker labels.

## Activation and diagnostics

`GGML_CUDA_FATTN_Q4_KV_FUSION` is inherited by the engine through the normal
[runtime environment](../README.md):

| Value | Behavior |
| --- | --- |
| Unset | Automatic activation within the guard; no diagnostic atomics or log |
| Exact `0` | Disable fusion; collect stock-path diagnostic counters |
| Exact `1` | Enable guarded fusion; collect diagnostic counters |
| Anything else | Disable fusion silently |

Explicit diagnostic modes provide a private getter/reset interface and, after
an eligible dispatch, one sanitized exit summary. Counters measure host dispatch
selections, including
CUDA graph capture, **not graph replay executions**. Fields are individually
relaxed atomic snapshots; concurrent contexts need not produce a coherent
combined snapshot. They contain no tensor values, pointers or request content.
Default automatic execution does not update these counters or register the
exit summary. All activation modes retain the same shape and occupancy guards.

The old F16 workspace remains reserved. Eligible execution bypasses its use;
this patch makes no VRAM-saving claim and does not change the allocation
contract, model/KV precision, context size or sampling.

## Validation boundaries

[CPU policy checks](../tests/test_q4_kv_layout.cpp) cover parsing, layouts,
bounds and excluded shapes. The [CUDA loader oracle](../tests/test_q4_kv_loader.cu)
compares operand and shared-tile bits against the original converter and loader.
The [actual attention oracle](../tests/test_q4_kv_attention_operator.cpp) checks
output bytes, padding canaries, partitioning and fallback behavior. Physical
sanitizer, full-engine quality and performance evidence belongs in the dated
validation report and fixtures; this contract does not establish a speedup.
