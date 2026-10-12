# Native patches

`ctranslate2-gb10-cuda.patch` fixes a GB10 CUDA build issue in ctranslate2.

The DFlash large-image cache fix (previously `dflash-pinned-image-positions.patch`)
is now included upstream in llama.cpp revision `aa94f2086147cd4bfbcff818ce998dffce5fcb39`,
bundled in xllamacpp 2026.10.11549. No local patch is required.

The CUDA and GB10 Docker images build xllamacpp from source via
`scripts/build_xllamacpp.py` to target specific GPU architectures not covered by
the published wheels. The DFlash pinned-position skip for M-RoPE image rows is
included in the upstream llama.cpp source at the pinned revision.

The CUDA build explicitly disables the optional NCCL collective backend: this
xllamacpp revision's binding linker omits NCCL even if CMake detects it, producing
an extension with unresolved NCCL symbols. The worker uses single-GPU inference;
this does not disable CUDA offload. GPU tensor-parallel/NCCL workloads are outside
the scope of this source build.
