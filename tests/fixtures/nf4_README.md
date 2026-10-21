# NF4 weight quantization and runtime loading

`nf4.hpp` is a portable C++ implementation of bitsandbytes block-64 NF4 with
optional nested block-256 statistics and BF16/FP16 reconstruction. The decision trees
come from [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/0.50.2).
The upstream MIT license is retained in the header and `licenses/bitsandbytes.txt`.
The exact dynamic codebook comes from the installed `create_dynamic_map()`.

The native LTX loader now converts Gemma linear projections automatically when
the selected CUDA device has at least 12 GiB free. It uses BF16 on compute
capability 8 or newer, FP16 otherwise, and the selected device’s actual SM/thread
geometry. Embeddings and norms are unchanged. Projection parameters now retain
packed NF4 values and nested statistics; the GGML CPU/CUDA operation expands
them into a temporary BF16/FP16 matrix for multiplication. Source mappings are
unchanged. Separate quantized-Gemma GPU residency and Quanto fallback remain
incomplete. No Python runtime or converted cache is required.

Current packed-loader deployment and repeated real LTX generation evidence is
in `nf4_integration_validation.json`; standalone storage/operator checks are in
`nf4_storage_validation.json`. These include two full Gemma projection matrices
and packaged CUDA tests. Earlier runtime and diagnostic records below describe
their own historical implementations and limitations.

`nf4_runtime_validation.json` records the deployed CUDA image, binary/source
hashes, 115 passing native tests, 10 image tests and 89 unchanged Rust wire tests.
Two actual routed requests use the original cached components with no hidden
engine retry. Storyboard output matches the corrected diagnostic MP4 byte for
byte; ordinary video passes stream/shape checks. Both restore the original
220000-token context, and invalid input plus subsequent chat pass. Production
router registration, chat, SSE and vision pass after deployment. The `.env`,
effective environment, mounts, ports and GPU access remain unchanged.

`nf4_cases.json` records 136 actual CUDA quantization/dequantization executions
from bitsandbytes 0.50.2 and PyTorch 2.9.1+cu128. Seventeen sizes from 1 to 16448
cover partial/odd blocks, nested-statistic block boundaries, two random seeds,
zeros and spikes, with and without nested statistics. Packed bytes, nested
statistics and reconstructed BF16 bytes match exactly, including signed zero.
BF16 buffers are stored as little-endian hexadecimal strings; no Python runtime
is required to run the test:

```sh
cmake --build build --target test_nf4
ctest --test-dir build -R '^nf4(_f16)?_reference$' --output-on-failure
```

`nf4_f16_cases.json` adds 136 installed CUDA FP16 cases using the same matrix
of sizes, seeds and nested-statistics settings. Both dtype fixtures run with
host mean accumulation and the captured CUDA geometry. `ltx_nf4_loader_reference`
compiles the actual patched loader: 32 combinations cover dtype, float32/native
source precision, mmap, lazy file readers and NF4 selection; two further cases
verify unsupported destination failure. Source files remain unchanged. Ten
worker-process cases verify threshold/precision selection, geometry errors,
CPU mode and a fresh unquantized child after an NF4 conversion failure.

Two real Gemma layer-zero matrices, attention query projection and MLP gate
projection, also match exactly: 74,711,040 BF16 values plus packed NF4 and nested
statistics. They are cast to BF16 before quantization, matching the worker's
`torch_dtype`; the original safetensors weights are float32. Full matrix buffers
remain ignored local artifacts under `build/validation/nf4`.
`nf4_validation.json` retains their hashes, shapes, versions and comparison counts.

The nested mean rounds the accumulated sum to float32, then multiplies by a
float32 reciprocal. Dividing in double precision caused a one-ULP offset error
and 268 BF16 differences in the query matrix. The corrected implementation matches
these cases, but its CPU double accumulator does not reproduce CUDA's reduction
order for every possible tensor/device. The full 336-matrix check subsequently found 61,614 BF16 differences across
76 matrices (10,758,389,760 values total). CUDA reduction-order emulation, selected
by an explicit `Nf4CudaDevice`, matches all 336 captured mean offsets; the
corrected full-model recheck leaves 404 BF16 differences across four matrices
(maximum absolute error 0.0009765625). The residual is retained in the evidence;
complete quantizer equivalence is not claimed.
Other unverified scope includes extreme/subnormal weights, GPU NF4 matmul,
Quanto fallback, residency selection and complete text-encoder equivalence.

Capture scripts: `/tmp/ezlocalai-capture-nf4-extended.py` and
`/tmp/ezlocalai-capture-nf4-weights.py`. The native real-matrix comparison is
`build/validation/nf4/test_native_weights.cpp`; capture/source hashes are
embedded in the fixtures. The initial large-matrix diagnostic incorrectly left
the source weights in float32; it was corrected and recaptured before recording
these BF16 comparisons.


`nf4_mean_cases.json` retains four actual Gemma block-scale arrays, compressed
as zlib/hex float32 buffers, and their installed CUDA mean results. Run
`ctest --test-dir build -R '^nf4_cuda_mean_reference$' --output-on-failure`.
The host emulation follows the installed aligned contiguous float32 reduction,
including vector accumulation, block/warp addition order and global reduction.
Actual driver attributes (84 multiprocessors, 1536 threads per multiprocessor)
select this GPU's grid. No CUDA/PyTorch library is linked by the helper. Other
GPU geometries still need live verification. PyTorch's license is retained in
`torch_cuda_mean.hpp` and the existing `licenses/pytorch-LICENSE`.

A real routed I2V experiment with diagnostic NF4-derived weights retains the
theorem graphic at four seconds and restores the LLM context/production health.
The original cache is untouched; derived safetensors remain ignored under
`build/validation/nf4/components`. This first experiment uses the initial
CPU mean, before reduction-order correction. It demonstrates the visual effect
of weight quantization, not full text-encoder equivalence or runtime integration.
The C++ converter and engine wrapper are diagnostic tools only.

The final corrected-weight routed experiment also preserves the raw scene's
graphic at four seconds (113.43 seconds) and restores the original 220000-token
LLM context and production health. Its `.env` remains unchanged. Capturing the
four residual matrices isolates the 404 value differences to seven nested
statistic byte codes; packed NF4 weights, offsets and nested block maxima match.
The cause of those seven code-selection differences remains unresolved.
