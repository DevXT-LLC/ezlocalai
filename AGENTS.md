# AGENTS.md

This is a guide for agents working in this repository. Read it before
changing code, building images or touching a deployment.

## What this is

ezlocalai is a C++17 local AI worker with an OpenAI-compatible API. It replaces
the Python ezlocalai worker behind the Rust **WorkerRouter**. The goal is
behavioral parity with the Python worker, except for documented intentional
corrections.

| Thing | Where |
| --- | --- |
| Python reference implementation (the parity oracle) | `../ezlocalai`, pinned at commit `0f7e9f4`. Tests read it through `EZLOCALAI_REFERENCE_ROOT`. |
| Rust router the worker registers with | [DevXT-LLC/WorkerRouter](https://github.com/DevXT-LLC/WorkerRouter) |
| Parity status and dated evidence log | `PARITY_PROGRESS.md`. The newest entries are at the top; older ones are history. |
| Per-route and per-module parity checklist | `parity_manifest.tsv`, checked with `build/parity_report` |
| Hardware still needing physical validation | `HARDWARE_TEST_PLAN.md` |

The Python router itself is **not** ported. Only the worker side of the
router protocol matters: registration, heartbeat, tunnel and reporting.

## Layout

```
src/                 worker, engines and adapters (native_server.cpp is ~20k lines: grep, don't read)
include/ezlocalai/   public headers, including the C ABI ezlocalai_c.h
tests/               C++ tests. tests/fixtures holds reference cases and *_validation.json records
cmake/               dependency installers and pinned per-platform manifests
                     (*-audio-dependencies.json, *-storyboard-dependencies.json)
engine/, patches/    stable-diffusion.cpp worker and the ordered sd.cpp patch stack
tts/                 Qwen TTS engine (links the llama.cpp build)
tools/               llama_cpp_tool, parity_report
voices/              the 22 default TTS voices, byte-identical to Python's. Uploads stay untracked
*.Dockerfile         cuda (also GB10 via EZLOCALAI_PLATFORM=gb10), cpu (x86, Pi), rocm, jetson
docker-compose*.yml  one profile per platform. Defaults mirror the Python compose files
```

## Build and test

Host builds need the native dependencies: LibTorch, CTranslate2, ONNX Runtime,
the stable-diffusion.cpp build and llama.cpp. The Docker images provide them.
For quick iteration:

- Use a container built from an image stage that already has
  `/opt/ezlocalai/deps`. Copy the tree in, then configure with the generated
  cache:
  `cmake -S . -B build -C /opt/ezlocalai/deps/native-cache.cmake -DEZLOCALAI_SDCPP_SOURCE=... -DEZLOCALAI_TTS_LLAMA_BUILD=...`.
- `ctest --test-dir build -R <name> --output-on-failure`. Run only the suites
  your change touches, then the package set.
- Each Dockerfile's final `ctest -R '^(...)$'` list is the **package gate**.
  When you add a test that must ship, add it to that regex and to the
  `cuda_package` CMake target.
- Package gates must run without `EZLOCALAI_REFERENCE_ROOT`: images do not
  include the Python checkout. Use explicit recorded-fixture modes for package
  checks and retain strict source-hash checks as separate developer tests.
- The compose gate runs on the host, because it needs the Docker client but no
  native dependencies:
  `cmake -S . -B build-compose -DEZLOCALAI_COMPOSE_GATE_ONLY=ON && make -C build-compose test_compose_profiles && ctest --test-dir build-compose`.
  It compares about 1,875 settings with the Python compose files. Run it after
  any compose change.

Fixture rules:

- `*_reference` tests run the Python source or its recorded cases, and many
  check SHA-256 hashes of Python files. A failure saying "source changed" means
  the reference checkout is missing or different, not that the code is wrong.
- Some tests hash files in **this** repository:
  - `tests/tts_process_fixture.cpp`
  - `include/ezlocalai/tts_unicode.hpp`
  - `tts/worker.cpp`

  If you edit one, re-pin the recorded hash in its fixture. Some of those
  fixtures are `.jsonl.gz`.

## Platform images

| Profile | Image tag | Notes |
| --- | --- | --- |
| `docker-compose-cuda.yml` | `ezlocalai:cuda` | x86_64, sm_86/89/120, CUDA 12.8 |
| `docker-compose-gb10.yml` | `ezlocalai:gb10` | aarch64, sm_121, CUDA 13. Torch cu130 and CTranslate2 built from source |
| `docker-compose.yml` | `ezlocalai:cpu` | x86_64 CPU |
| `docker-compose-rpi.yml` | `ezlocalai:rpi` / `ezlocalai:rpi5` | `EZLOCALAI_ARM_CPU_ARCH`: armv8-a for Pi 4B and 5, or `armv8.2-a+dotprod+fp16` for Pi 5 only |
| `docker-compose-rocm.yml` | `ezlocalai:rocm` | gfx1100 and `HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA3/3.5, including the Radeon 880M) |
| `docker-compose-jetson.yml` | `ezlocalai:jetson` | JetPack 6 / L4T R36.4, sm_87, Cortex-A78AE |

Build rules:

- Engines are built with `GGML_NATIVE=OFF`, so images run on every fleet CPU.
  Never reintroduce `-march=native`. Use `GGML_CPU_ARM_ARCH` for ARM targets.
- ARM images can be built on x86 under QEMU (`--platform linux/arm64`). They
  are slow, and four checks cannot pass under emulation, so emulated builds
  must pass
  `EZLOCALAI_PACKAGE_TEST_EXCLUDE='^(dotenv_startup|tunnel_framing_reference|ltx_runtime|media_engine)$'`.
  Native builds run the full gate.
- BuildKit reads the build context lazily. **Do not edit or move files in
  the tree while a build is running**; use a separate git worktree.

## Deployment and the router

- Cutover, verification and rollback: README, "Prebuilt images from the LAN
  registry".
  - Reuse the Python installation's `.env` with `--env-file`, and its data
    directories with the `EZLOCALAI_*_DIR` overrides.
  - Both workers use container name `ezlocalai`, so `docker compose down` the
    Python one first.
- The LAN registry is `192.168.1.243:5000`. Images go to
  `devxt/ezlocalai-<platform>:latest` (e.g. `devxt/ezlocalai-cuda:latest`,
  `devxt/ezlocalai-gb10:latest`). Build on devxt5090, push to the LAN
  registry, and workers pull from there. The C++ image replaces the Python
  worker — do not maintain both.
- Verify a deployment against the router, not just `/health`. Check the
  router's `/v1/router/workers` record (version, models, `cap_slots`,
  `model_context`, `extra`, gpus), then send a routed chat pinned with
  `"worker": "<label>"`. A worker's record should match a Python worker on
  the same hardware and `.env`.
- Before replacing a running container, tag the current image as a rollback.
  Afterwards, confirm the effective container environment and mounts are
  unchanged.

## Intentional divergences from Python

Keep these. Do not "fix" them back:

- **Failed text requests with no `FALLBACK_SERVER`** return 413
  `context_capacity_error`, 503 or the engine's 4xx. Python returns a 200
  "Local resources exhausted" completion.
- **Handoff:** a non-streaming request during a GPU handoff returns 503 with
  the `temporarily_unavailable` body.
- **Router fallback:** with `ROUTER_URL` set and no `FALLBACK_SERVER`, text
  the worker cannot serve goes to the router. Forwarded bodies carry
  `disable_fallback: true`, so there is at most one hop.
- **GPU memory** is read through NVML, with no CUDA context in the worker
  process. `cuMemGetInfo` is the fallback for GB10, HIP or missing NVML.
- **Tunnel read budgets** extend with `REQUEST_TIMEOUT`,
  `VIDEO_GENERATION_TIMEOUT` and `ACE_STEP_TIMEOUT`. Python uses a fixed 600 s.
- **LTX audio** is resampled to 24 kHz. Python labels 48 kHz audio as 24 kHz.
- **STT text and subtitle responses** carry `; charset=utf-8`.
- **Engine:** llama.cpp v0.8.0. Python pins b11549.
- **Qwen generator seed:** v0.8.0 resets codec RNG per utterance. Historical
  sequential audio hashes differ; the versioned CPU fixture retains exact
  Python-wrapper comparison with the same new engine.

## Scope

- LFM3-Omni-2B native realtime text/audio and chat image/audio/video input are in scope; see `REALTIME.md`. Its 128,000 context cap must never affect Qwen's 262,144 context.
- Qwen is the only required ordinary TTS engine; gTTS and Edge fallbacks are not
  required.
- PDF-to-speech is out of scope.
- The separate Gemma storage experiments in Python are out of scope. Complete
  LTX video is in scope, but video is the **last** priority, behind
  worker/router correctness.
- Multi-GPU is covered only by simulated topologies. There is no multi-GPU
  hardware.

## Recording evidence

- Add a dated section at the top of `PARITY_PROGRESS.md` saying what changed,
  why, and how it was validated. Include image IDs, test counts and rollback
  tags. Say plainly what was **not** validated.
- Put compact machine-readable evidence in
  `tests/fixtures/<topic>_<date>_validation.json`. Bulky logs and downloads go
  under `build/validation/`, which is ignored.
- Don't claim hardware validation from emulation or simulated drivers.

## Gotchas learned the hard way

- **Prompt-cache numerics.** With tiny models (0.8B), an identical greedy
  prompt answered after a prompt-cache hit can differ from the cold answer.
  Give acceptance prompts per-request prefixes.
- **QEMU timing.** Under QEMU, startup takes about 7 s and Qwen TTS can take
  over 15 minutes. Worker (`QWEN_TTS_TIMEOUT`) and router (`REQUEST_TIMEOUT`)
  defaults are 300 s.
- **`docker stats` and cgroup `memory.peak`** include reclaimable page cache
  from model reads. Hitting the limit is not an OOM; check `OOMKilled` and the
  logs.
- **Child environments.** llama-server and the other engines inherit the
  worker's environment (`fork`/`execvp`). Settings such as
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, needed for iGPU UMA, pass straight
  through from `.env`.
- **Disk.** The build host's disk is shared and fills fast: each CUDA image is
  about 19 GB. After each build cycle:
  - remove superseded `ezlocalai:*` tags;
  - prune this project's superseded BuildKit cache entries by ID, keeping the
    engine stages;
  - delete re-downloadable binaries under `build/validation/`.

  Never prune other projects' images, volumes or cache.
- **Don't remove other people's containers.** That includes long-running
  development containers you didn't create.
