# CUDA 12.8 Isolated Build

2026-11-11. This is build qualification, not device correctness or performance
qualification. All GPU execution is reserved for the parent's measurement window.

## Completed Result

Both `test-backend-ops` and `llama-server` built and linked successfully from
clean source with CUDA 12.8.93 for SM86/89/120a. Build ran from
17:18:34 to 17:35:21 UTC (16 minutes 47 seconds). The compiler container exited
with code 0, `running=false`, `OOMKilled=false`; no compilation remains running.

Ready host bin directory for the parent's `--engine-root`:
`/home/josh/repos/xtsys/ezlocalai-prefill/build/validation/experimental-prefill-gdn-cuda128-20261111/build/bin`

- Bin directory: 105 MB; complete scratch tree: 548 MB.
- Latest sampled cgroup peak: 2,468,876,288 bytes (2.30 GiB), with zero OOM events.
  This was sampled shortly before completion, not a final peak measurement.
- All 10 host checks passed again; the audit file hashes remain unchanged.
- Compile/link log contains no compiler warnings or errors. Configure has the
  inherited `LLAMA_CURL` deprecation warning; the option is ignored in v0.9.0.
- `ldd` resolves all GGML/llama/mtmd/server libraries to `/validation/build/bin`.
  The normal GPU-free inspection reports only the expected missing driver
  `libcuda.so.1`. A second, GPU-free `ldd -r` with the image's driver stub in the
  loader path resolves everything with no undefined symbols. This is static
  relocation validation only; neither application nor a GPU kernel was run.
- `cuobjdump` confirms the three architecture images in the GDN object. Slice64
  has 188/188/238 registers and zero stack bytes for SM86/89/120a respectively.
  CUDA 12.8's SM120a output also reports 1,024 bytes of static shared memory,
  in addition to the kernel's dynamic allocation. The old V32/V128 resource
  entries are identical to the exact production image's GDN object entries.

Artifact SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `test-backend-ops` | `6bc78f74f048f4f41a34c3c6ff53049226756491cba9e9f44c9d31643633d73c` |
| `llama-server` | `b19b041625578c6f3c6c8d1294a361e3480c62d7403b896908397354eebc9db6` |
| `libggml-cuda.so.0.29.0` | `32a64b12e1bbc8a9d70e29ca60bc1ce0ba5005f5957d57401f09e5a1588df3d4` |

Evidence files in scratch include `build.log`, `configure.log`, `compiler.log`,
`container-limits.json`, `container-result.json`, `cgroup-progress.log`,
`artifact-sha256.txt`, `gdn-architectures.log`, `gdn-resource-usage.log`,
`production-gdn-resource-usage.log`, both `ldd-*.log` files, and
`static-relocations-with-driver-stub.log`. Device gates below remain pending.

## Location And Toolchain

Scratch root on the host:
`/home/josh/repos/xtsys/ezlocalai-prefill/build/validation/experimental-prefill-gdn-cuda128-20261111`

Compiler image (immutable ID):
`sha256:85764c1a7540b36d143faa494fdb3bac6927fe99b59c8d6a65f31261e2a6ee12`

- NVCC 12.8.93, GCC 13.3.0, CMake 3.28.3.
- Release, `GGML_NATIVE=OFF`, graphs enabled, NCCL disabled.
- SM86, SM89, SM120a (CMake maps requested `120-real` to `120a-real`).
- Generated CUDA `flags.make` is byte-identical to the production image's file.
- The full audited source was copied, including its existing local patch stack.
  Only the two experimental patches and copied production UI assets differ.
- No existing engine objects or libraries were reused. The production UI `dist`
  assets were copied to avoid downloads and frontend tooling during compilation.
- Build container: `experimental-prefill-gdn-cuda128-20261111`, runtime `runc`,
  network `none`, 4-CPU quota, 12 GiB memory, no swap allowance, 512 PID limit,
  UID/GID 1000:1000. No GPU devices or device requests; `NVIDIA_VISIBLE_DEVICES=void`.
- No deployed files, services, runtime workers, or audit source were modified.

## Reproduce

Use a fresh scratch directory and a unique container name when repeating this.
Do not rerun preparation against the already-patched source or overwrite artifacts.

```bash
repo=/home/josh/repos/xtsys/ezlocalai-prefill
audit=/home/josh/repos/xtsys/ezlocalai/build/validation/llama-v090-cuda-stage/source
work=$(mktemp -d "$repo/build/validation/experimental-prefill-gdn-cuda128.XXXXXX")
cp -a "$audit" "$work/source"
cd "$work/source"
git apply --check "$repo/patches/experimental-prefill-gdn-slice64.patch" \
  "$repo/patches/experimental-prefill-gdn-slice64-tests.patch"
git apply "$repo/patches/experimental-prefill-gdn-slice64.patch" \
  "$repo/patches/experimental-prefill-gdn-slice64-tests.patch"
docker run --name "experimental-prefill-gdn-$(date +%s)" \
  --runtime=runc --network=none --cpus=4 --memory=12g --memory-swap=12g \
  --pids-limit=512 --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/tmp \
  --mount "type=bind,src=$work,dst=/validation" \
  --mount "type=bind,src=$repo/benchmarks/experimental-prefill-gdn-slice64-full-build.sh,dst=/experimental-build.sh,readonly" \
  --entrypoint /bin/bash \
  sha256:85764c1a7540b36d143faa494fdb3bac6927fe99b59c8d6a65f31261e2a6ee12 \
  /experimental-build.sh
```

The script configures a new build and waits for both `test-backend-ops` and
`llama-server`, including all their dependencies. It records configuration,
compiler output, build logs, production build metadata, loader dependencies,
artifact hashes, and start/finish timestamps. A `finish-time.txt` is only written
after both targets and artifact checks succeed. It does not run either binary.

## Qualification Handoff

Run the operator gate before loading models, under the parent's owned drain.
This bounded smoke gate selects exactly seven correctness cases in the patched
evaluation list, including one 63-token AR fallback and six eligible MMA cases:

```bash
timeout --signal=TERM --kill-after=10s 180s \
  env GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=1 OMP_NUM_THREADS=4 \
  "$BIN/test-backend-ops" test -b CUDA0 -o GATED_DELTA_NET -j 1 \
  -p '^type=f32,head_count=16,head_size=128,n_seq_tokens=(63|64|65|128|129),n_seqs=[123],v_repeat=[124],permuted=[01],kda=0,K=1$'
```

Repeat in a separate process with the knob set to `0`. `BIN` must be the mounted
experimental absolute bin directory, with candidate-first `LD_LIBRARY_PATH`.
Require exit code 0, exactly `7/7 tests passed`, `Backend CUDA0: OK`, and seven
distinct `GATED_DELTA_NET(...): OK` case lines, after stripping ANSI escapes.
Reject unsupported/skipped cases, zero matches, errors, and timeout (124 or 137).
Do not accept the final overall `OK` alone: absent CUDA backends or zero selected
tests can otherwise appear to pass. The timeout bounds each mode to 180 seconds
plus at most 10 seconds for forced termination.

The seven tuples `(tokens, sequences, value-repeat, permuted)` are
`(63,1,2,0)`, `(64,1,4,0)`, `(65,1,1,0)`, `(65,3,4,0)`, `(128,1,2,0)`,
`(129,1,2,0)`, and `(129,2,4,1)`. Each uses H_k=16, D=128, scalar gating, K=1.
This is a bounded CPU-reference output/state check, not the complete acceptance
suite and not proof that the opt-in specialization actually launched. Confirm
the latter in the profile before interpreting performance results.

Use the complete `build/bin` directory, not just one executable or one replacement
library. The clean build contains its own `libggml*`, `libllama*`, server library,
and multimodal dependencies. Use the pinned image for its CUDA 12.8 runtime and
system libraries. Keep all candidate libraries ahead of the image libraries in
`LD_LIBRARY_PATH`; build RPATHs contain the container path `/validation/build/bin`.

After the parent explicitly releases a GPU, this is the isolated operator run
pattern (not executed as part of the build):

```bash
work=/home/josh/repos/xtsys/ezlocalai-prefill/build/validation/experimental-prefill-gdn-cuda128-20261111
docker run --rm --gpus "device=$GPU_UUID" --network=none \
  --mount "type=bind,src=$work,dst=/validation,readonly" \
  -e GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=1 \
  -e LD_LIBRARY_PATH=/validation/build/bin:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
  --entrypoint /validation/build/bin/test-backend-ops \
  sha256:85764c1a7540b36d143faa494fdb3bac6927fe99b59c8d6a65f31261e2a6ee12 \
  test -b CUDA0 -o GATED_DELTA_NET
```

Repeat in a new process with the knob set to `0` for the original schedule in the
same binary. For the eligible 32-case performance matrix use
`perf -b CUDA0 -o GATED_DELTA_NET -p 'head_count=16,head_size=128' --output csv`.
Follow the numerical, continuation, sanitizer, same-input, and model-quality
gates in [the experiment notes](experimental-prefill-gdn-slice64.md).

For a later server run, use `/validation/build/bin/llama-server` as the entrypoint
with the same candidate-first library path, a separate read-only model mount,
the parent's identical workload options, and a reserved non-production port.
Do not replace the worker's binaries or register this diagnostic server as a
production worker. Do not add the compiler's CUDA stub directory to runtime
`LD_LIBRARY_PATH`: device runs must load the real driver from the NVIDIA runtime.

The inherited snapshot eligibility concern remains unresolved: candidate
acceptance is K=1 prefill only. No MTP/snapshot correctness claim follows from
successful compilation.
