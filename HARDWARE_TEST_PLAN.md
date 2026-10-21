# Hardware test plan

These are the deployment profiles that could not run on their real hardware
from the build host. That host has one RTX 3090 Ti and emulates ARM64 under
QEMU. Every profile below passes its package gate and the static review from
2026-10-20. What remains is physical execution.

Run each section on the named machine. Every section starts from a checkout of
this repository on the branch being deployed. Python is kept as the rollback.

The common checks are:

- **Health:** `curl -s localhost:8091/health` returns `{"status":"healthy"}`.
- **Router record:** the router's `/v1/router/workers` shows the host's
  `WORKER_LABEL` with a fresh heartbeat. Its capabilities, models, context and
  slots should match the Python worker's last record, and `version` should be
  the deployed commit.
- **Routed chat:** a chat sent through the router with `"worker": "<label>"`
  returns 200, both non-streaming and streaming. Add vision if the host serves
  it.
- **Logs:** `docker logs` shows no `not found`, `illegal instruction`, OOM or
  segfault lines.

## 1. RTX 5090 (DevXT5090), Qwen3.8-27B text and vision

**Status:** ready for a production trial. The image has `sm_120` kernels,
CUDA 12.8 and portable x86 CPU code. Per-card settings match Python:
- MTP draft length 3 for Qwen3.8-27B;
- KV cache `q4_0`, unless `MTP_SPEC_DRAFT_N_MAX_5090` or `KV_CACHE_TYPE_5090`
  is set;
- context from `LLM_MAX_TOKENS`.

**Steps:**

1. Cut over with the README's "Prebuilt images from the LAN registry"
   section, using `EZLOCALAI_CUDA_IMAGE=192.168.1.243:5000/devxt/ezlocalai-native-cuda:latest`.
   The driver must be 570 or newer.
2. Run the common checks.
3. Confirm `model_context` (262144 for Python on this host),
   `gpus[0].name` (`NVIDIA GeForce RTX 5090`) and tier 90.
4. Compare `nvidia-smi` and the router's free VRAM. Python reported about
   9.1 GB free; native should report a similar figure or slightly more, since
   it has no worker CUDA context.
5. Run a long prompt (30k tokens or more) and a follow-up turn. The second time
   to first token should show prompt-cache reuse.
6. **Pass:** chat, SSE and vision all return 200 for a day of normal traffic,
   with no circuit-open events on the router dashboard.

**Roll back** with the README rollback commands.

## 2. GB10 (DevXTgx10), every modality

**Status:** the image is published at
`192.168.1.243:5000/devxt/ezlocalai-native-gb10:latest` (`sm_121`, CUDA 13).
54/54 package checks pass under QEMU, and health, models and resources respond.

**Steps:**

1. Cut over with `docker-compose-gb10.yml` and `EZLOCALAI_GB10_IMAGE`, as in
   the README.
2. Run the common checks for text and vision. Then send one request each for
   embeddings, TTS, STT, image and music through the router.
   `build/validation/` holds the probe used on the 3090: the same
   inputs as `media_probe.py` (voices, embeddings, speech, speech/stream,
   transcription in three formats, a 512×512 image, a 10 s music clip).
3. GB10 memory is unified, so NVML reports none. Confirm `/v1/resources` free
   memory is plausible against `nvidia-smi` and system RAM; the worker uses the
   `cuMemGetInfo` fallback here.
4. Optional: build natively on the GB10 to run the full 58-check gate:
   `EZLOCALAI_PACKAGE_TEST_EXCLUDE= docker compose -f docker-compose-gb10.yml build`.
5. **Pass:** all modalities return 200, with outputs comparable to the Python
   GB10 worker's. `extra.loaded_llms` and slots should match Python's record
   (two text slots and two each for TTS, STT and embeddings).

## 3. AMD: Ryzen AI 9 365 laptop with Radeon 880M (RDNA 3.5 iGPU, ROCm 6.4.1)

**Status:**
- The package gate passes, along with CPU-fallback inference.
- Every engine resolves exactly one HIP runtime and one HSA runtime, which the
  gate checks.
- It has never run on an AMD GPU.

The image builds `gfx1100` kernels and sets `HSA_OVERRIDE_GFX_VERSION=11.0.0`,
so the 880M (`gfx1150`) runs RDNA3 code. This is the configuration the Python
profile documents for this laptop, and the usual way llama.cpp runs on RDNA 3.5
iGPUs.

**Memory.** The 880M has no VRAM of its own:

- The BIOS "UMA frame buffer" reserves a fixed slice of system RAM for it, often
  only 512 MB by default. ROCm reports that slice as VRAM.
- To let llama.cpp use ordinary system RAM as well, add
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` to the laptop's `.env`. llama.cpp then
  allocates with `hipMallocManaged`. llama-server inherits the worker's
  environment, so nothing else is needed.
- Alternatively, raise the UMA buffer in BIOS (8–16 GB) and keep normal
  allocation. Test both.

**Host requirements:**

- An amdgpu kernel driver with `/dev/kfd`: a recent kernel (6.10 or newer) or
  ROCm 6.4 DKMS.
- Docker. The ROCm userspace comes from the image.

**Steps (on the laptop):**

1. Pull or build the image. If it has been published, set
   `EZLOCALAI_ROCM_IMAGE=192.168.1.243:5000/devxt/ezlocalai-native-rocm:latest` and run
   `docker compose --env-file .env -f docker-compose-rocm.yml pull`. Otherwise run
   `docker compose -f docker-compose-rocm.yml build`.
2. Run `docker compose ... up -d`, then check inside the container:
   - `stat -c %g /dev/dri/renderD128 && id`. The render group ID must appear in
     `id`. If it doesn't, set `group_add` to the numeric ID.
   - `rocminfo | grep -E 'gfx|Marketing'` should list the 880M.
3. `curl :8091/v1/resources` should show the ROCm backend and the UMA VRAM size.
4. Send a chat with a small model first (Qwen3.5-4B Q4) while watching
   `rocm-smi --showmeminfo vram gtt`. The llama-server log should show `ROCm0`
   offload. Repeat with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` and compare
   tokens/s and how many GPU layers fit.
5. Send Whisper STT, Qwen TTS and one image generation. **Pass:** each returns
   200 and uses the GPU.
6. If the GPU isn't used:
   - `HSA_OVERRIDE_GFX_VERSION` must be `11.0.0`.
   - Rebuilding with `EZLOCALAI_HIP_ARCHITECTURES=gfx1150` and no override is
     the alternative, if ROCm 6.4.1's libraries include gfx1150 kernels.

## 4. Raspberry Pi 4B and Pi 5 (ARM64 CPU)

**Status:**
- Both images pass their package gates under QEMU.
- Real inference passes through Rust: chat, SSE, vision, embeddings and Whisper
  STT.
- Qwen TTS works but took 890 s under QEMU.

There are two images. Use the one that matches the board:

| Board | Image | CPU target |
| --- | --- | --- |
| Pi 4B (Cortex-A72) and Pi 5 | `ezlocalai:rpi` (`devxt/ezlocalai-native-rpi`) | `armv8-a` baseline |
| Pi 5 only (Cortex-A76) | `ezlocalai:rpi5` (`devxt/ezlocalai-native-rpi5`) | `armv8.2-a+dotprod+fp16`, faster quantized matmuls |

The Pi 5 image uses instructions the Pi 4B lacks and will crash on it with an
illegal instruction. Build either image locally with
`EZLOCALAI_ARM_CPU_ARCH` (empty for Pi 4B; `armv8.2-a+dotprod+fp16` and
`EZLOCALAI_RPI_IMAGE=ezlocalai:rpi5` for Pi 5).

**Steps (on each Pi, after pulling or building):**

1. `getconf PAGESIZE`. The Pi 5 kernel defaults to 16K pages; the container
   must still start and pass `/health`.
2. `up -d`, run the common checks, then embeddings, STT and TTS.
   - Watch `docker stats`. The Pi-sized x86 emulation peaked at 7.6 GB
     including page cache.
   - On 4 GB boards, check `dmesg` for OOM kills, and consider disabling
     TTS/STT or pointing them at another worker with `VOICE_SERVER`.
3. Time one Qwen TTS clip. If it exceeds 300 s, raise `QWEN_TTS_TIMEOUT` and the
   router's `REQUEST_TIMEOUT`, or use `VOICE_SERVER`. The same applies to Python.
4. Record Qwen3.5-0.8B tokens/s: `ezlocalai:rpi` on both boards, and
   `ezlocalai:rpi5` on the Pi 5.

## 5. Jetson Orin NX Developer Kit (Cortex-A78AE, sm_87)

**Host requirement.** JetPack 6.x (L4T R36), with Docker and the NVIDIA
Container Runtime (`docker info | grep -i runtime` must list `nvidia`). The
image is built on L4T R36.4 / CUDA 12.6. Unlike the Python profile, there is no
JetPack 5 (L4T R35) build. Upgrade the board to JetPack 6 first if it is
still on JetPack 5.

**Image:**

- LibTorch is compiled from source for `sm_87`.
- Engines target the Orin's Cortex-A78AE (`armv8.2-a+dotprod+fp16`), which every
  Orin module shares.
- FFmpeg includes the PNG codec and threads.
- Defaults follow Python's Jetson profile: Qwen3.5-4B Q4_K_XL, 8192 context,
  `f16` KV, full GPU offload, media models not kept loaded. These fit a 16 GB
  Orin NX; on 8 GB, consider disabling TTS or STT.
- Pull it if it has been published:
  `EZLOCALAI_JETSON_IMAGE=192.168.1.243:5000/devxt/ezlocalai-native-jetson:latest`.
  A native build on the board takes many hours and 30 GB or more of disk.

**Steps (on the Orin NX):**

1. Pull the image, or build it with
   `docker compose -f docker-compose-jetson.yml build`. A native build runs the
   full gate, including the `sm_87` architecture check.
2. Inside the container:
   - `ffmpeg -buildconf | grep -E 'zlib|pthreads'`;
   - `ldd /usr/local/bin/ezlocalai-native | grep avcodec` should show `.so.60`
     from `/usr/local/lib`;
   - `ldd /usr/local/bin/llama-server | grep libcuda` should resolve to the L4T
     driver.
3. `up -d`, run the common checks, and compare `/v1/resources` memory with
   `tegrastats`. NVML may be missing on Tegra, in which case the worker falls
   back to `cuMemGetInfo`.
4. Send STT, Qwen TTS and one short Wan or LTX video. The video proves the PNG
   frame path.

## 6. Multiple GPUs, tabled

There is no multi-GPU host available. Placement, `MAIN_GPU`, `TENSOR_SPLIT`
and per-device memory are covered by simulated-topology tests only. When a
multi-GPU host exists:

1. Check that `gpus[]` lists every visible card, and honors
   `CUDA_VISIBLE_DEVICES`.
2. Check that `MAIN_GPU` and `TENSOR_SPLIT` change the llama-server arguments
   and the actual per-card allocation.
3. Check that embeddings, TTS and STT land on the secondary card the way
   Python's placement chooses.
