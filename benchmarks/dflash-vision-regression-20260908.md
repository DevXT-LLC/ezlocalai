# DFlash large-image regression — RTX 3090 Ti

Configuration: Qwen3.8-27B UD-Q3_K_XL, target Q4_0 KV, 220,000 allocated
context, batch 1024 / ubatch 512, DFlash Q4_K_M drafter, n-max 3 / p-min 0.
CUDA 12.8; xllamacpp 2026.9.10809; the native patch and build revisions are
documented in [native/patches](../native/patches/README.md).

## Reproduction and correction

The deployed official wheel returned `failed to process mtmd chunk` for a
1792×1792 synthetic image, both alone and after a ~12K-character text prefix,
and for a 1008×1008 image after that prefix. A small 336×336 image passed.
Native logs identified a failed **draft** decode at offset 512 or 1024, not a
CUDA allocation failure. A full-size draft SWA-cache experiment, even with
Q4_0 draft KV, subsequently ran out of VRAM on the large image; it was rejected.
Neither experiment changed the serving worker's configuration.

The patch keeps target vision input intact and skips only pinned-position
embedding injection into the speculative draft cache. Results from
`benchmark_vision.py`, run directly against the actual Python bindings:

| Check | Target-only reference | Patched DFlash |
| --- | --- | --- |
| Small image, left/right colors | 2/2 | 2/2 |
| Large image, left/right colors | 2/2 | 2/2 |
| Immediate cached large-image repeat | 2/2 | 2/2 |
| 512-token generation after a large image | 2/2 | 2/2 |
| Long text before image | 2/2 | 2/2 |
| Long text after image | 2/2 | 2/2 |
| Two large images, order-sensitive question | 2/2 | 2/2 |
| Large-image OCR, number 42 | 2/2 | 2/2 |
| Changed-image OCR, number 73 | 2/2 | 2/2 |
| Text arithmetic after vision | 2/2 | 2/2 |
| Total | 20/20 | 20/20 |

The large-image prompt contains 3,168 tokens, versus 1,056 for the small image;
the long-prefix prompts contain 7,726. The cached large-image repeat reused
3,164 prompt tokens. No native decode failures occurred in the patched suite.
These are focused regression checks, not a general visual-quality benchmark.

## Text-generation control

Two 256-token greedy code completions after vision had identical output SHA-256
(`23efeb1a45966372a37cb42163035e83706d0f9d420e74e766f901aa9857ec89`)
and identical draft acceptance (174/242) on the official and patched bindings.
Observed decode was 79.5 tokens/s before and 75.6 after; these short measurements
ran under different concurrent native-build loads, so they are not a reliable
performance comparison. They establish unchanged output/acceptance on this
control, not performance parity or a speed improvement.

Only the RTX 3090 Ti was physically tested. The CUDA build includes SM86, SM89,
and SM120 code; 4090/5090 deployment still requires the updated worker image.
