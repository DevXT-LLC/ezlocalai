
# Run with Docker

You can choose to run with Docker or [Docker Compose](DockerCompose.md). Both are not needed. Instructions to run with Docker Compose can be found [here](DockerCompose.md).

## Docker Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using NVIDIA GPU)

## Run without NVIDIA GPU support

Modify the `THREADS` environment variable to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

```bash
docker pull joshxt/ezlocalai:cpu
docker run -d --name ezlocalai -p 8091:8091 joshxt/ezlocalai:cpu -e THREADS="10" -e EZLOCALAI_API_KEY="" -v ./models:/app/models
```

## Run with NVIDIA GPU support

If you're using an NVIDIA GPU, you can use the CUDA version of the server. You must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed if using NVIDIA GPU.

Modify the `GPU_LAYERS`, `MAIN_GPU`, and `THREADS` environment variables to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

```bash
docker pull joshxt/ezlocalai:cuda
docker run -d --name ezlocalai -p 8091:8091 --gpus all joshxt/ezlocalai:cuda -e THREADS="10" -e GPU_LAYERS="20" -e MAIN_GPU="0" -e EZLOCALAI_API_KEY="" -v ./models:/app/models
```

# Run with Docker Compose

You can choose to run with Docker or Docker Compose. Both are not needed.

## Docker Compose Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using NVIDIA GPU)

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

Make sure to move your `.env` file to the `ezlocalai` directory if you set one up.

## Run with Docker Compose (Without NVIDIA GPU)

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
