# Native Torch STT platform and GPU inference

The C++ `stt_torch_operations` adapter implements the Torch boundaries used by
`STT.py`: CUDA/cuDNN availability, device-0 total memory and Torch allocated
memory, the small convolution probe, cache release, synchronization and CUDA IPC
collection. Native objects release their shared owners directly. No Python
interpreter or Python runtime library participates in the native execution.

The reference is revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`, STT.py SHA-256
`d0dd8340c243d7ff9ba84f4e3babe3c41b0b0ce7c7dddf2565eb886ba3f21fe2`.
Captures execute its unchanged `_check_cudnn_available`, `get_available_vram_mb`,
CPU-thread helper, constructor and close methods. The external generator is
`/tmp/ezlocalai_stt_torch_oracle.py`; all repository runners are C++.

## Pinned native libraries

The reference CUDA Dockerfile installs Torch 2.9.1+cu128 and cuDNN 9.10.2.21.
The verified environment uses CUDA runtime 12.8.90, cuBLAS 12.8.4.1, cuDNN 9.10.2
(runtime version 91002), NumPy 2.4.6, and an NVIDIA GeForce RTX 3090 Ti with
25,258,688,512 bytes of reported device memory under driver 570.211.01. Both reference and native executions use
the C++ libraries from the pinned Torch package. Native linkage includes
torch_cpu, torch_cuda, c10 and c10_cuda, with no libtorch_python or libpython.

Library SHA-256 values:

- libtorch_cpu.so: `7918fc09ff644c9b667921100b924e33ea432c85737c2982b56283691b32a4d7`
- libtorch_cuda.so: `02250527966ae122ccfc89d0306736874f9c619ba04431871ef76175e7253b66`
- libcudnn.so.9: `3b68ea689be647bce63cb3c2d9edb589add415405a7e9f296d69ca029cae0b8b`

The standalone official LibTorch 2.9.1+cu128 archive was also inspected. Its
bundled cuDNN reports 9.8.0, so that archive is not the verified reference
combination. Merely matching Torch's version string does not establish matching
dependencies. The native target reads version metadata without executing Python.

## Comparisons

`stt_torch_cases.jsonl` contains 18 real-GPU scenarios: three retained allocation
sizes, two initial cuDNN flag configurations, and direct probe/normal constructor/
GPU constructor failure with CPU fallback. They compare exact allocated and
reserved memory, CPU/CUDA RNG-state SHA-256 values, restored cuDNN flags, model
arguments, batch/audio/collection event order, device/compute selection, available
memory and repeated close behavior. Model and audio boundaries are substituted
in this platform test; the Torch operations and allocations are real.

These 18 platform cases were recaptured from unchanged `STT.py` on 2026-10-18.
The prior capture reported 25,283,526,656 bytes and four fallback CPU threads;
the current machine reports the capacity above and selects eight threads.
Only those capacity/available-memory values and fallback thread arguments
changed. Allocation, RNG, flags and lifecycle traces otherwise stayed identical,
and all 18 refreshed cases pass in the CUDA image. The external capture script
SHA-256 is `6cfd6dd1e1ccfc119b61e43d06fb302bbbddb76effae6dda49535bb11c431bc1`.
The older real-model captures described below retain their original dependency
and thread settings; current large-v3-turbo captures are documented in
`audio_deployment_README.md`.

The source convolution output retains its autograd graph while empty_cache runs.
In the zero-external-allocation capture, 2,048 bytes remain allocated through that
call, then allocation drops to zero when the source probe returns while a 2 MiB
segment remains cached. The native probe returns a shared result whose lifetime
spans the wrapper's cache call, reproducing this behavior. It also restores all
six source cuDNN flag values and preserves the CPU/GPU random-generator effects.

`stt_torch_unavailable.json` captures native operations with CUDA_VISIBLE_DEVICES
empty: unavailable CUDA, compiled cuDNN availability, zero allocated memory,
no-op empty_cache, and exact errors from total-memory/synchronize/IPC calls.

`stt_runtime_model_modern_gpu.jsonl` contains 15 complete GPU float16 wrapper
calls. The generator is `/tmp/ezlocalai_stt_runtime_model_oracle.py` with
EZLOCALAI_STT_REAL_GPU=1. It uses real Torch, PyAudio, faster-whisper 1.2.0,
CTranslate2 4.6.1, native GPU inference and the same pinned models/audio/VAD
assets as the CPU wrapper fixtures. Both source and native runners assert that
the models remain on CUDA after every call, preventing a silent CPU fallback
from passing the GPU comparison. Text, structured output and errors are exact;
timestamps retain the existing 1e-12 arithmetic allowance. Model, audio,
tokenizer, vocabulary, configuration and VAD identities are checked.

The reference CUDA image explicitly places the pinned cuDNN directory on
LD_LIBRARY_PATH. The GPU model captures and native test likewise supply the
cuDNN/cuBLAS/CUDA runtime directories; otherwise the CTranslate2 wheel's cuDNN
loader can abort while resolving its component libraries. An initial capture
without that source environment failed and was not retained as a successful case.

## Scope and remaining work

The 2026-10-20 ROCm extension uses verified Torch 2.9.1+rocm6.4 native
libraries and the ROCm 6.4.1 SDK. `stt_torch_hip_unavailable.json` captures
the matching Python wheel with both visibility masks set to `-1`.
Compiled cuDNN availability is true in that Python build, even though
ATen's `hasCuDNN()` is false; the native adapter now preserves the Python
behavior. HIP allocator and device guards retain PyTorch's public CUDA
tensor naming. The unavailable-device, allocator and actual SDK ABI checks
pass without AMD hardware. The existing 18 real NVIDIA GPU cases and native
CUDA wake-word training also pass with the shared adapter.

`wakeword_training_rocm_cpu_cases.json` records five training cases from the
unchanged Python wake-word classes with the matching ROCm Torch/torchaudio
wheels. Native CPU training, checkpoint reload and cancellation pass against
that reference. CPU results differ slightly between the CUDA and ROCm Torch
builds; the original CUDA reference and tolerances were preserved.
These checks do not establish AMD GPU inference or training.
See `hip_torch_20261020_validation.json` for dependencies and evidence.

These captures verify the default CUDA availability path on one GPU and one
dependency combination. PYTORCH_NVML_BASED_CUDA_CHECK=1, fork poisoning, AMD GPU execution,
CPU-only LibTorch builds, alternate allocator backends, arbitrary exception
messages and other hardware remain unverified. The Python-specific garbage
collector has no native equivalent; the native model's shared-owner lifetimes
are compared at collection points in the policy/platform tests.

This optional target supplies the native Whisper adapter when configured.
Production dependency packaging, speech API/pool integration, tokenizer fallback,
preprocessor configuration, VAD packaging, microphone streams and diarization
remain incomplete. These results do not establish whole-service parity.
