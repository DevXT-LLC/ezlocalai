# Native engine extensions

`sdcpp-ltx-conditions.patch` applies to stable-diffusion.cpp
`c678dfe704a2230342376b46add9c8ca736a653d` using `git apply` from that checkout.
It is applied by `cuda.Dockerfile`, followed by the component adapter below.

It adds `sd_ltx_condition_t` and the CLI `--ltx-conditions` option, accepting
a JSON array of local image paths, latent-frame indices, and strengths:

```json
[{"image":"first.png","index":0,"strength":1.0},
 {"image":"middle.png","index":4,"strength":0.7},
 {"image":"last.png","index":-1,"strength":1.0}]
```

Index semantics come from the installed Python LTX condition pipeline. Index
zero replaces the first latent frame. Other valid indices append keyframe
tokens at pixel position `(index - 1) * 8 + 1`; negative indices wrap in the
generated latent-frame count, and positive out-of-range indices are skipped.
Each condition retains its own strength, mask and positional coordinates.
Images use float bilinear cover-resizing without an antialias prefilter and
then a center crop. The extension rejects use with a non-LTX model.

The current extension covers single-image conditions. Multi-frame conditions
are not implemented; a hires second pass with conditions is explicitly
rejected. It passes tensor-helper tests and, together with the component
adapter below, a real CUDA generation with multiple positioned conditions.
See `tests/fixtures/ltx_README.md` for the source-derived cases, execution
evidence and remaining limits.

`sdcpp-ltx-diffusers.patch` applies after the conditioning patch. It adds
`--ltx-diffusers-root` for an existing LTX-2.3 Diffusers snapshot containing
`text_encoder`, `connectors`, `vae`, `audio_vae`, and `vocoder` components.
Indexed shards and single safetensors files are supported. Tensor names are
mapped in memory; no duplicate or rewritten model files are created. Normal
runtime dtype conversion still occurs. Published connector tensors replace the corresponding GGUF connector
entries, preserving the Python worker's default connector source.

The adapter includes the fourth decoder upsampling stage present in the
published 2.3 checkpoint, which the installed legacy single-file conversion
dictionary does not describe. Its ordering was checked against the installed
decoder's execution. Gemma3 normalization names and vocoder/filter names are
mapped to the native engine's names. The new root string is owned by the
engine configuration, like its other model paths.

Raw Diffusers Gemma3 RMS weights require the transform `normalized * (1 +
weight)`. The standard GGUF conversion folds that offset into the weights;
loading raw safetensors must enable it explicitly. The adapter enables the
engine's existing unit-offset RMS operation only for the Diffusers Gemma3
encoder. Converted GGUF encoders retain their previous behavior.
`ltx_norm_reference` executes 72 GGML graphs against 36 cases captured from
the installed Transformers implementation, covering raw and folded weights.
Enable it with both `EZLOCALAI_SDCPP_SOURCE` and `EZLOCALAI_SDCPP_BUILD`
pointing to the patched engine source and its completed build.

The engine's VAE name-conversion table also uses guarded static initialization.
Its previous `empty()`/`emplace_back()` initialization raced when model-loader
threads converted names concurrently, producing a real SIGSEGV during VAE
loading. `ltx_name_concurrency` exercises 32 simultaneous first-use threads;
the original implementation reproduced an incomplete mapping in this test.

Image-sequence output now also saves generated audio to a WAV sidecar. For
example, `frame_%06d.png` produces `frame_%06d.wav`, allowing the worker to
encode an MP4 from lossless frames without an intermediate lossy video.
The CUDA Dockerfile applies both patches before compiling the engine.

The adapter also replaces LTX's audio latent ceiling with the installed
Diffusers double-precision, ties-to-even rounding. `ltx_audio.hpp` is exercised
against 196 source-executed counts, including halfway cases. This changes
joint video/audio sampling when the old ceiling produced an extra audio
latent; older native output hashes are historical evidence, not expectations
for the corrected engine. The WAV retains the vocoder's native sample rate;
the worker's MP4 export follows the Python wrapper's 24 kHz interpretation.

`sdcpp-media-worker.patch` applies after the NF4 patch. The accompanying
`engine/sd_cli_worker.hpp` and `media_worker_protocol.hpp` add a private
`--ezlocalai-worker-fd FD --ezlocalai-parent-pid PID` mode. Ordinary CLI calls
keep their one-shot behavior. The private mode retains `sd_ctx_t` across
requests, requests eager loading, and replaces the context when non-generation
arguments change. It clears callbacks before their request-local owners expire.

The inherited Unix socket carries a four-byte big-endian length followed by a
version-1 JSON request (`args`, `preload`, `log`) or response (`status`). Frames
are capped at 16 MiB. The native `MediaEngineProcess` owns the child, serializes
requests, applies one absolute communication deadline, redirects diagnostics to
files, and kills/reaps failed or timed-out engines. The child also dies when
its parent disappears. No HTTP listener or additional router protocol is added.

The C++ test exercises this same worker/context implementation with controlled
model constructors. Its `--model-case FILE` mode invokes a real patched engine;
real Qwen-image and LTX-2.3 repeat-generation evidence is recorded in
`tests/fixtures/media_engine_validation.json`. `LtxVideo` now uses this layer
for startup, generation, recovery and unload. Source lifecycle and real HTTP/
Rust tunnel evidence is recorded in `tests/fixtures/ltx_lifetime_validation.json`.

`sdcpp-nf4-storage.patch` adds CUDA execution for the private NF4 expansion
operation in `engine/nf4_ggml.hpp`; CPU execution uses GGML's custom-op
callback. Packed storage contains NF4 nibbles, byte-quantized block statistics,
FP32 nested maxima and one FP32 offset. It retains double quantization and
expands into a temporary BF16/FP16 matrix for the existing matrix multiply.
The CUDA code uses the source-validated tables and separate rounded multiply/
add operations. Contiguous byte-offset views are supported without requiring
aligned float loads. Unrelated custom operations retain their existing behavior.

`nf4_storage_reference` checks 136 existing Python cases and eight additional
CUDA source cases covering all value/scale codes, zero scale groups and FP16
subnormal values. `test_nf4_storage_cuda` runs the same checks with `--cuda`.
Both executables optionally accept `--weights FILE` for the recorded full
Gemma projection manifest. They compare expanded bytes, independent matrix
outputs, compressed allocation size and preservation of input storage.
The tested CUDA candidate and source hashes are recorded in
`tests/fixtures/nf4_storage_validation.json`. To exercise a CUDA-enabled build:

```sh
build/test_nf4_storage_cuda \
  tests/fixtures/nf4_cases.json \
  tests/fixtures/nf4_f16_cases.json \
  tests/fixtures/nf4_storage_cases.json --cuda \
  --weights build/validation/nf4/real_weights.json
```

The optional full weight fixtures are local validation artifacts; omit
`--weights` to run the 144 checked-in cases. The CPU executable uses the same
arguments without `--cuda`.

`sdcpp-ltx-nf4-packed.patch` connects this operation to the raw LTX Gemma
loader and `Linear` blocks. Source metadata retains its original shape/type
and carries a separate NF4 compute precision. File-reader clones preserve
precision rules and CUDA geometry. Loading converts one projection at a time,
quantizes it and writes only packed bytes to its CPU/GPU destination; it never
rewrites source mappings. Model validation checks the packed destination size,
and placement estimates count packed parameter bytes. `Linear` expands its
weight for matrix multiplication, leaving graph allocation to reclaim the
temporary. The unrelated tensorwise INT8 path keeps its original behavior.

`ltx_nf4_loader_reference` now executes actual `Linear` graphs for 32 loading
cases, including precision selection before/after source discovery, direct
reads, mmap and lazy file-reader clones. Its optional `--weights FILE` mode
adds the two full Gemma projections, converting FP32 checkpoint data before
quantization and comparing with the recorded Python results.

This integration is deployed and its packaged checks and real repeated LTX
generations are recorded in `tests/fixtures/nf4_integration_validation.json`.
The existing worker's generic
offload policy still treats the whole conditioner together; Python retains
quantized Gemma separately. Full-model placement/offload parity, Quanto and
Python inference-output equivalence remain open.

`sdcpp-ltx-gemma-residency.patch` is the next, currently undeployed integration.
It retains NF4 Gemma's registered parameters on their compute backend while
leaving the projection module under its ordinary placement. Parameter memory
reporting supports a conditioner split across CPU/GPU. The hardware-only
`test_ltx_gemma_residency_cuda` target links the actual engine model manager
and checks placement, independent eviction/reload and complete unload.
Real full-model observations are in `tests/fixtures/gemma_residency_validation.json`.
The wrapper still needs post-load policy selection and distinct offload modes;
the engine prototype alone does not close those gaps.

`sdcpp-ltx-offload-policy.patch` applies after the separate Gemma residency
patch. It adds `--ltx-offload-mode` and `sd_set_ltx_offload_mode`: zero restores
automatic engine policy, one executes whole graphs, and two forces existing
graph-cut segments, disables prefetch and evicts staged weights at each segment
boundary. Resident quantized Gemma parameters remain on GPU. Invalid modes and
nonzero overrides for non-LTX models are rejected.

The private worker can change this execution policy on an existing context.
Changing full parameter placement still rebuilds the context. `LtxVideo`
initially loads with CPU offload, then queries the selected GPU again and uses
that post-load free-memory observation, the successful NF4 selection, and the
original request hint to select placement. Capacity failures enter the existing
sequential/reduced-generation recovery path.

`test_ltx_runtime` checks 45 recorded Python constructor cases for placement
only; Quanto tensor execution is not represented by its synthetic loader. Two
source cases that ignore failed sequential-hook installation remain excluded.
Its optional `--real-placement FILE` mode exercises the actual wrapper and
engine, records the chosen mode and process lifetime, and validates generation
recovery and teardown. The CUDA residency test also exercises the real segment
weight pipeline: sequential completion must immediately evict staged projector
weights while retaining Gemma, whereas automatic/model mode may keep the cache.
Graph segments are coarser than Python's per-module hooks; this implementation
does not establish equal peak memory or full numerical parity with Python.

`sdcpp-ltx-shared-cuts.patch` marks connector outputs and shared timestep
adaptations as outputs of the LTX graph prelude. Later transformer segments
consume those cached values instead of recursively executing their dependencies
again. Cut caches last for one graph invocation and are cleared on success or
failure; this does not cache conditioning across prompts or generations. The
engine reports executed-node and original-node counts when its segment plan
changes, so duplicate work can be checked in real generation logs.

This still differs from Diffusers' separate connector module, which runs before
its denoising loop. Native connector reuse across denoising calls and exact
Python module boundaries remain follow-up work.
