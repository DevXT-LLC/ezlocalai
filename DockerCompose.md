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
LOCAL_LLM_API_KEY=
THREADS=10
BATCH_SIZE=512
GPU_LAYERS=0
MAIN_GPU=0
```

- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.
- `THREADS` - The number of threads to use. Default is `your CPU core count minus 1`.
- `BATCH_SIZE` - The batch size to use for batch generation. Default is `512`.
- `GPU_LAYERS` - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` - The GPU to use for the main model. Default is `0`.

Make sure to move your `.env` file to the `Local-LLM` directory if you set one up.

## Run with Docker Compose (Without NVIDIA GPU)

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
docker-compose pull
docker-compose up
```

## Run with Docker Compose (With NVIDIA GPU support)

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
docker-compose -f docker-compose-cuda.yml pull
docker-compose -f docker-compose-cuda.yml up
```
