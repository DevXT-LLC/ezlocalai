# CUDA Q4 KV loading for F16 attention

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
