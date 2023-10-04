# Local-LLM

- [Dockerhub](https://hub.docker.com/r/joshxt/local-llm/tags)
- [GitHub](https://github.com/Josh-XT/Local-LLM)

Local-LLM is a [llama.cpp](https://github.com/ggerganov/llama.cpp) server in Docker with OpenAI Style Endpoints that allows you to send the model name as the URL of the model from Hugging Face. It will automatically download the model from Hugging Face if it isn't already downloaded and configure the server for you. It automatically configures the server based on your CPU, RAM, and GPU. It is designed to be as easy as possible to get started with running local models.

## Table of Contents 📖

- [Local-LLM](#local-llm)
  - [Table of Contents 📖](#table-of-contents-)
  - [Environment Variables](#environment-variables)
  - [Run with Docker](#run-with-docker)
    - [Docker Prerequisites](#docker-prerequisites)
    - [Run with Docker (Without NVIDIA GPU)](#run-with-docker-without-nvidia-gpu)
    - [Run with Docker (With NVIDIA GPU support)](#run-with-docker-with-nvidia-gpu-support)
  - [Run with Docker Compose](#run-with-docker-compose)
    - [Docker Compose Prerequisites](#docker-compose-prerequisites)
    - [Environment Setup (Optional)](#environment-setup-optional)
    - [Run with Docker Compose (Without NVIDIA GPU)](#run-with-docker-compose-without-nvidia-gpu)
    - [Run with Docker Compose (With NVIDIA GPU support)](#run-with-docker-compose-with-nvidia-gpu-support)
  - [OpenAI Style Endpoint Usage](#openai-style-endpoint-usage)
  - [Shout Outs](#shout-outs)

## Environment Variables

Assumptions will be made on all of these values if you choose to accept the defaults.

- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.
- `THREADS` - The number of threads to use. Default is `your CPU core count minus 1`.
- `BATCH_SIZE` - The batch size to use for batch generation. Default is `512`.
- `GPU_LAYERS` - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` - The GPU to use for the main model. Default is `0`.

## Run with Docker

You can choose to run with Docker or Docker Compose. Both are not needed.

Run with docker without a `.env` file, just replace the environment variables with your desired settings.

### Docker Prerequisites

- [Docker](https://docs.docker.com/get-docker/)

### Run with Docker (Without NVIDIA GPU)

```bash
docker pull joshxt/local-llm:cpu
docker run -d --name local-llm -p 8091:8091 joshxt/local-llm:cpu -e THREADS="10" -e BATCH_SIZE="512" -e LOCAL_LLM_API_KEY=""
```

### Run with Docker (With NVIDIA GPU support)

If you're using an NVIDIA GPU, you can use the CUDA version of the server.

```bash
docker pull joshxt/local-llm:cuda
docker run -d --name local-llm -p 8091:8091 --gpus all joshxt/local-llm:cuda -e THREADS="10" -e BATCH_SIZE="512" -e GPU_LAYERS="0" -e MAIN_GPU="0" -e LOCAL_LLM_API_KEY=""
```

## Run with Docker Compose

You can choose to run with Docker or Docker Compose. Both are not needed.

### Docker Compose Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using NVIDIA GPU)

### Environment Setup (Optional)

Assumptions will be made on all of these values if you choose to skip this step. Create a `.env` file if one does not exist and modify it to your needs. Here is an example `.env` file:

```env
LOCAL_LLM_API_KEY=
THREADS=10
BATCH_SIZE=512
GPU_LAYERS=0
MAIN_GPU=0
```

Make sure to move your `.env` file to the `Local-LLM` directory if you set one up.

### Run with Docker Compose (Without NVIDIA GPU)

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
docker-compose pull
docker-compose up
```

### Run with Docker Compose (With NVIDIA GPU support)

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
docker-compose -f docker-compose-cuda.yml pull
docker-compose -f docker-compose-cuda.yml up
```

## OpenAI Style Endpoint Usage

OpenAI Style endpoints available at `http://localhost:8091/` by default. Documentation can be accessed at that url when the server is running. There are examples for each of the endpoints in the [Examples Jupyter Notebook](examples.ipynb).

## Shout Outs

- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - For constantly improving the ability for anyone to run local models. It is one of my favorite and most exciting projects on GitHub.
- [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - For making it easy to extend the functionality of llama.cpp in Python.
- [TheBloke](https://huggingface.co/TheBloke) - For helping enable the ability to run local models by quantizing them and sharing them with a great readme on how to use them in every repository.
- [Meta](https://meta.com) - For the absolutely earth shattering open source releases of the LLaMa models and all other contributions they have made to Open Source.
- [OpenAI](https://openai.com/) - For setting good standards for endpoints and making great models.
- As much as I hate to do it, I can't list all of the amazing people building and fine tuning local models, but you know who you are. Thank you for all of your hard work and contributions to the community!
