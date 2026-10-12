
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
runs independently of the CPU/RPi GitHub builds and does not deploy to workers.

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
