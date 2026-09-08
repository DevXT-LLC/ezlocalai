# DFlash large-image hotfix

`dflash-pinned-image-positions.patch` targets llama.cpp
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`, bundled in xllamacpp 2026.9.10809.
It follows the narrowed position-based guard in
[upstream PR #28587](https://github.com/ggml-org/llama.cpp/pull/28587), revision
`4dc1fc0bd42beb28f71bc6531a11f1486cc1152f`. That PR is still under discussion,
not an accepted upstream fix; this is a temporary, locally validated hotfix.

M-RoPE image rows share a section-0 position. Injecting thousands of those rows
into the drafter's 2048-position sliding-window cache can exhaust its cells
before any become eligible for eviction. This fails even with free GPU memory.
The patch skips only those pinned-position **draft** embedding injections.
The target model still receives all image rows and verifies the proposed output.
Advancing audio/text embeddings are not skipped. Draft acceptance after images
can differ; this does not expand the target context or change target KV precision,
sampling parameters, model weights, or image resolution.

The CUDA Docker image builds the patched binding for SM86, SM89 and SM120.
Other images still install upstream wheels; native/other-backend users can build
the same fix with `python scripts/build_xllamacpp.py --install` (add `--cuda` or
`--hip` as appropriate). CMake, a compiler, Rust/cargo and the backend SDK are
required. Builds default to 20 jobs; use `--jobs` to reduce this on small hosts.
The installed marker `xllamacpp._ezlocalai_hotfix.HOTFIX` identifies patched wheels.
Reinstalling the official wheel removes the fix.

The CUDA build explicitly disables the optional NCCL collective backend: this
xllamacpp revision's binding linker omits NCCL even if CMake detects it, producing
an extension with unresolved NCCL symbols. The worker uses single-GPU inference;
this does not disable CUDA offload. GPU tensor-parallel/NCCL workloads are outside
the scope of this hotfix build.

Remove this patch/source-build override once an upstream xllamacpp release
contains a verified fix. Re-run the large-image regression suite before updating
the pin: a small single image is insufficient to catch the cache exhaustion.
