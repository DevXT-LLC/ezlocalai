# Native worker reporting to Rust WorkerRouter

The current priority is worker reporting and CUDA API behavior. Video generation
is deferred. This work fixes concrete registration/heartbeat omissions: an empty
build version, missing embedding model/alias quantization, configured quantization
overriding the actual loaded file, and a loaded projector not adding vision
capability. Empty embedding-enable values now also report embedding context,
matching the source's enabled-by-default behavior.

`worker_version_cases.json` executes unchanged version functions from `Router.py`
in the original Python image. It covers 12 cleaning cases and 24 resolution cases:
environment precedence, marker files, Unicode/truncation, Git results, lightweight
Git metadata and missing-package fallback. The C++ runner also resolves a real
isolated Git repository and checks process caching.

Version resolution honors `EZLOCALAI_VERSION`, `EZLOCALAI_COMMIT`, `GIT_COMMIT`,
`.ezlocalai-version`, `VERSION`, Git and lightweight Git metadata in source order.
The final fallback identifies the native implementation using its project version
and a deterministic first-party runtime-source fingerprint. It does not pretend
to be the Python package version. The fingerprint includes source, headers, engine
adapters, patches, CMake and the CUDA Dockerfile; it is not a full image digest.
CMake tracks changes to those inputs. The default result is cached per process,
so overrides should be set before starting the worker.

`worker_model_reporting_cases.json` executes the unchanged full source
`WorkerHeartbeatClient._gather_state` with controlled model, memory and slot
boundaries. Its 23 cases compare public models, capabilities, quantization,
context, modality model names and residency extras. Cases include loaded-file
quants, split GGUF names, IQ quants, configured fallbacks, duplicate models,
replicas, embedding aliases/delegation/flags, invalid embedding context and a
custom loaded projector. A second source snapshot verifies that vision remains
advertised after unloading the model, as it does in the Python client.

The existing C++ heartbeat-capture runner compares both registration and heartbeat
payloads, then changes model state without restarting the worker. Its original
73 dependency/residency cases remain included. The first expanded run caught the
empty embedding flag's missing-context bug; that failing log is retained under
`build/validation/worker-reporting`.

`test_worker_router` launches the actual Rust router with native workers in direct
and tunnel modes. It checks build-version display, loaded-file quantization,
shared speech/text admission, partial preload and replica capacity, discovery,
JSON/SSE, concurrent requests, model switches, restart recovery and deregistration.
Two additional embedding-only scenarios verify model/alias context and quantization
in the Rust registry and dashboard, including router restart.

`worker_hip_gpu_cases.json` executes the unchanged Python GPU detection and
tier functions with simulated AMD device properties. Its eight complete GPU
snapshots cover ordering, hidden devices and property failures. They exposed
the native hardware backend label incorrectly remaining `cuda`; reporting now
uses `rocm` and `rocm:N` fallback names while Torch tensor devices retain CUDA
naming. The HIP metadata runner checks these captures through registration
and heartbeats. See `hip_torch_20261020_validation.json`.

`worker_heartbeat_shutdown_cases.json` captures the unchanged Python heartbeat
client against actual stalled/released loopback HTTP responses. Native checks
enforce the five-second task cancellation grace, separate deregistration
timeout and source request order. The regression failed at 10.28 seconds
before the runtime fix. See `heartbeat_shutdown_20261020_validation.json`;
image deployment status is tracked separately in the progress report.
The managed-engine scenarios also cancel live chat streams in both connection
modes, assert router/worker admission before and after interruption, observe
the engine-side disconnect and serve another request without an engine restart.
`stream_cancellation_20261020_validation.json` records corresponding live CUDA
checks at three interruption boundaries in each mode and Qwen speech recovery.

WorkerRouter's own identity, collections, routing-metadata, dashboard-workers
and worker-callback suites are run separately. Those tests exercise the Rust side
with its fixtures; they are not substitutes for the C++ worker integration tests.
The live CUDA check compares all reported field presence, idle slots, lifecycle,
model lists and actual llama process model/context/parallel/projector arguments,
and confirms the native version appears on the Rust dashboard.

These checks establish the specified reporting behavior. They do not prove every
worker endpoint, arbitrary dependency combinations, multiple-GPU execution or all
model inference behavior. Overall replacement parity remains partial.

The 2026-10-20 audit also checks GPU registration and heartbeat using the
simulated CUDA driver: two cards, reordered cards, one selected card, empty and
`-1` visibility, and three properties failures. Physical `nvidia-smi` devices
are no longer advertised when CUDA hides them. Failed per-device queries retain
Python's generic ordinal/zero-memory entry. See
`production_audit_20261020_validation.json` for the deployed image and successful
chat/SSE/vision checks over the production Rust router's reverse tunnel, followed
by restoration and revalidation of the original direct connection.

A subsequent real CPU check found that an explicit `LLAMA_CPP_MMPROJ_PATH`
outside the automatic model directory reached the owned engine but was absent
from its stored projector state. The worker therefore omitted vision from its
capabilities. Launch now retains that path. The actual Rust integration has a
regression with a model name lacking any automatic projector directory; it
checks the engine argument and vision registration/heartbeat in direct and
tunnel modes, including router restart. This fails before the fix and passes
afterward. Release acceptance is tracked in `PARITY_PROGRESS.md`.
