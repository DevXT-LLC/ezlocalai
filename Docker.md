
# Run with Docker

You can choose to run with Docker or [Docker Compose](DockerCompose.md). Both are not needed. Instructions to run with Docker Compose can be found [here](DockerCompose.md).

## Docker Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using NVIDIA GPU)
- [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) (if using AMD GPU)

## Run without GPU support (CPU only)

Modify the `THREADS` environment variable to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

```bash
docker pull joshxt/ezlocalai:cpu
docker run -d --name ezlocalai -p 8091:8091 joshxt/ezlocalai:cpu -e THREADS="10" -e EZLOCALAI_API_KEY="" -v ./models:/app/models
```

## Run with NVIDIA GPU support (CUDA)

If you're using an NVIDIA GPU, you can use the CUDA version of the server. You must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed if using NVIDIA GPU.

Modify the `GPU_LAYERS`, `MAIN_GPU`, and `THREADS` environment variables to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

```bash
docker pull joshxt/ezlocalai:cuda
docker run -d --name ezlocalai -p 8091:8091 --gpus all joshxt/ezlocalai:cuda -e THREADS="10" -e GPU_LAYERS="20" -e MAIN_GPU="0" -e EZLOCALAI_API_KEY="" -v ./models:/app/models
```

## Run with AMD GPU support (ROCm)

If you're using an AMD GPU (Radeon RX 6000/7000/9000 series, Radeon PRO, or Ryzen APUs with integrated graphics like the Radeon 880M), you can use the ROCm version of the server. You must have [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) installed on your host system.

For Ryzen APUs (like the Radeon 880M in Ryzen AI 9 HX 365), you may need to set `HSA_OVERRIDE_GFX_VERSION` to match a supported architecture (e.g., `11.0.0` for RDNA 3.5).

```bash
docker pull joshxt/ezlocalai:rocm
docker run -d --name ezlocalai -p 8091:8091 \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    -e HSA_OVERRIDE_GFX_VERSION="11.0.0" \
    -e THREADS="10" \
    -e EZLOCALAI_API_KEY="" \
    -v ./models:/app/models \
    joshxt/ezlocalai:rocm
```

# Run with Docker Compose

You can choose to run with Docker or Docker Compose. Both are not needed.

## Docker Compose Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (NVIDIA GPU only)
- [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) (AMD GPU only)

## Environment Setup (Optional)

Assumptions will be made on all of these values if you choose to skip this step. Create a `.env` file if one does not exist and modify it to your needs. Here is an example `.env` file:

```env
EZLOCALAI_API_KEY=
THREADS=10
GPU_LAYERS=0
MAIN_GPU=0
```

- `EZLOCALAI_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.
- `THREADS` - The number of threads to use. Default is `your CPU core count minus 1`.

The following are only applicable to NVIDIA GPUs:

- `GPU_LAYERS` - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` - The GPU to use for the main model. Default is `0`.

The following are only applicable to AMD GPUs (ROCm):

- `HSA_OVERRIDE_GFX_VERSION` - Override the GPU architecture version. Useful for newer APUs like Radeon 880M. Default is `11.0.0` for RDNA 3.5.
- `HIP_VISIBLE_DEVICES` - The AMD GPU device to use. Default is `0`.

Make sure to move your `.env` file to the `ezlocalai` directory if you set one up.

## Run with Docker Compose (Without GPU - CPU only)

```bash
git clone https://github.com/DevXT-LLC/ezlocalai
cd ezlocalai
docker-compose pull
docker-compose up
```

## Run with Docker Compose (With NVIDIA GPU support)

You must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed if using NVIDIA GPU.

```bash
git clone https://github.com/DevXT-LLC/ezlocalai
cd ezlocalai
docker-compose -f docker-compose-cuda.yml pull
docker-compose -f docker-compose-cuda.yml up
```

## Run with Docker Compose (With AMD GPU support)

You must have [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) installed on your host system. This supports AMD Radeon RX 6000/7000/9000 series, Radeon PRO GPUs, and Ryzen APUs with integrated graphics (such as the Radeon 880M in Ryzen AI 9 HX 365).

```bash
git clone https://github.com/DevXT-LLC/ezlocalai
cd ezlocalai
docker-compose -f docker-compose-rocm.yml build
docker-compose -f docker-compose-rocm.yml up
```

**Note for Ryzen APUs:** If you're using a Ryzen APU with integrated graphics, you may need to adjust the `HSA_OVERRIDE_GFX_VERSION` environment variable in your `.env` file to match a supported architecture. For RDNA 3.5 APUs like the Radeon 880M, use `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

## Pull the shared CUDA image

CUDA builds for `main` run on the DevXT registry host through the Deploy API.
The `Publish CUDA through Deploy` workflow sends `DEPLOY_DOMAIN=api.ezlocal.ai`
using a direct `curl` request and exits once Deploy accepts it. It
runs independently of the CPU/RPi GitHub builds. Deploy can roll the published
image out to the configured workers after publication (see below).

The workflow requires Actions secrets `DEPLOY_URI` (the full `/deploy` URL),
`DEPLOY_SECRET`, and `DEPLOY_DOMAIN`. Organization secrets must allow this
repository; defining them at the organization alone does not grant access.
Install the Deploy API CUDA build update
before merging/enabling this workflow. Manual dispatch is supported on `main`.
The builder publishes current `origin/main`, including any newer queued commits.
The workflow does not poll or wait for the CUDA build. Its success confirms
request acceptance; view the returned event in the Deploy status dashboard or
query `/deploy/status/events/{event_id}` to check publication and cleanup.

To use the prebuilt CUDA image on another NVIDIA machine, keep a checkout for
its Compose configuration and `.env`, and set this in `.env`:

```dotenv
EZLOCALAI_CUDA_IMAGE=192.168.1.243:5000/devxt/ezlocalai-cuda:latest
```

Then pull and start without building:

```bash
docker compose -f docker-compose-cuda.yml pull ezlocalai
docker compose -f docker-compose-cuda.yml up -d --no-build ezlocalai
```

The Docker daemon must trust `192.168.1.243:5000` as an insecure registry for
this LAN HTTP registry, and NVIDIA Container Toolkit must be installed.
Leaving `EZLOCALAI_CUDA_IMAGE` unset preserves the local development image name
and `docker compose -f docker-compose-cuda.yml up -d --build` behavior.

On the registry host, manual publication is also available:

```bash
python3 scripts/publish_cuda_image.py
```

This builds `cuda.Dockerfile` with the existing Docker cache, pushes
`localhost:5000/devxt/ezlocalai-cuda:latest`, verifies the published digest, and
removes superseded manifests only after success. Registry deletion must be
enabled. Failed build/push attempts preserve the previous manifest; failed
cleanup is retried on the next successful publication. Old SHA tags pointing
to deleted digests are removed too. Tags sharing the current digest survive a
no-op build because deleting that digest would also delete `latest`.
The registry's existing offline maintenance job reclaims unreferenced layers.
No containers are restarted by this helper.

## Pull the shared GB10 image

The same Deploy trigger now accepts `image_variant: gb10`. The GitHub workflow
submits separate CUDA and GB10 requests and exits on acceptance for each; builds
and retention run on the configured build host. Install the Deploy GB10 API
update before enabling the workflow matrix, because it verifies the accepted
variant in the response.

GB10 is ARM64. Publication builds `gb10.Dockerfile` with
`--platform linux/arm64` and checks the built image architecture before pushing
`192.168.1.243:5000/devxt/ezlocalai-gb10:latest`. The registry host needs ARM64
binfmt/QEMU support when it is x86-64. Native ARM64 build hosts are also
supported through Deploy configuration. Emulated cold CUDA compilation can be
much slower than native compilation; the GB10 server build timeout defaults to
24 hours, while the GitHub runner still exits after its short request. Native
dependencies disable build-host CPU autodetection so QEMU-specific CPU
instructions do not get baked into the shared image; GPU kernels still target
GB10's `sm_121`.

On the GB10, set this in `.env`:

```dotenv
EZLOCALAI_GB10_IMAGE=192.168.1.243:5000/devxt/ezlocalai-gb10:latest
```

Then pull and start without compiling:

```bash
docker compose -f docker-compose-gb10.yml pull ezlocalai
docker compose -f docker-compose-gb10.yml up -d --no-build ezlocalai
```

The GB10 Docker daemon must trust the LAN HTTP registry, and NVIDIA Container
Toolkit must be configured. Leaving `EZLOCALAI_GB10_IMAGE` unset preserves the
local image name for native development builds.

Manual publication on the build host:

```bash
python3 scripts/publish_cuda_image.py --variant gb10
```

CUDA and GB10 use separate registry repositories, build checkouts, locks, and
retention state directories (`~/.local/state/ezlocalai-<variant>-publish`). Each
keeps only its own latest manifest and removes superseded versions after digest
verification. Publishing one variant does not clean up the other. Workers are
not restarted by publication.


## Managed rolling worker updates

The Deploy service can update configured CUDA and GB10 workers after the shared
image is published. The Deploy dashboard groups CUDA and GB10 under
`api.ezlocal.ai`, showing build progress and each worker's pull, drain, restart,
health-check, and cleanup status. GitHub exits after request acceptance.

Install the router maintenance update before enabling worker rollout in Deploy.
`GET /v1/router/workers` advertises `maintenance_protocol: 1` and reports each
worker's `draining`, `active_requests`, and `maintenance_busy` state. Drain/resume
use the same authentication as worker registration (`ROUTER_REGISTER_KEY`, falling
back to the router's `EZLOCALAI_API_KEY`; unchanged LAN behavior if neither is set).

The controller calls `POST /v1/router/drain` with `label`, a unique `owner`, and a
120-second lease, renewing it while it works. `require_idle: true` reserves an
idle worker atomically or returns 409 if a request won the race. Only one worker
can hold maintenance across the fleet. Draining workers continue heartbeats and
finish existing requests, but are excluded from new dispatch and affinity
selection. Long streams remain counted after their normal capacity lease expires.
The reservation follows the stable label when a restarted worker gets a new ID.
`POST /v1/router/resume` requires the same `label` and `owner`. Leases persist in
`ROUTER_DRAIN_FILE` (`/data/router-drains.json` by default), including across router
restarts, and expire if a deployment controller disappears.

Deploy updates the three 3090s first and waits for their active work to finish.
It then picks idle 4090/5090 workers, allowing interruption after 30 minutes if no
idle window occurs. GB10 can be interrupted immediately. Each target uses its
existing Compose file, project, `.env`, model directories and volumes, with a
temporary override selecting the verified image digest. No worker builds images.
Another worker is not restarted until the replacement passes Docker health and
fresh router registration checks. Failed replacements restore the cached previous
image and stop the rollout.

After health checks, cleanup removes stopped containers for that worker's Compose
project and unused previous worker images. In-use images and unrelated projects
are protected. `.ezlocalai-deploy-cleanup.json` records candidates before restart
so failed cleanup can be retried after a future successful update. Model/output
volumes and build caches are preserved. The registry separately removes superseded
manifests and reclaims blobs through its existing maintenance schedule.
