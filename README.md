# Local-LLM

- [Dockerhub](https://hub.docker.com/r/joshxt/local-llm/tags)
- [GitHub](https://github.com/Josh-XT/Local-LLM)

Local-LLM is a [llama.cpp](https://github.com/ggerganov/llama.cpp) server in Docker with OpenAI Style Endpoints that allows you to send the model name as the URL of the model from Hugging Face. It will automatically download the model from Hugging Face and configure the server for you. It automatically configures the server based on your CPU, RAM, and GPU. It is designed to be as easy as possible to get started with running local models.

## Table of Contents 📖

- [Local-LLM](#local-llm)
  - [Table of Contents 📖](#table-of-contents-)
  - [Prerequisites](#prerequisites)
  - [Clone the repository](#clone-the-repository)
  - [Environment Setup (Optional)](#environment-setup-optional)
  - [Run with Docker](#run-with-docker)
    - [CPU Only](#cpu-only)
    - [NVIDIA GPU](#nvidia-gpu)
  - [OpenAI Style Endpoint Usage](#openai-style-endpoint-usage)
  - [Shout Outs](#shout-outs)

## Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.10](https://www.python.org/downloads/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if using NVIDIA GPU)

If using Windows and trying to run locally, it is unsupported, but you will need [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and [Docker Desktop](https://docs.docker.com/docker-for-windows/install/) at a minimum in addition to the above.

## Clone the repository

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
```

## Environment Setup (Optional)

Assumptions will be made on all of these values if you choose to skip this step. Create a `.env` file if one does not exist and modify it to your needs. Here is an example `.env` file:

```env
MODEL_URL=TheBloke/Mistral-7B-OpenOrca-GGUF
THREADS=10
THREADS_BATCH=10
GPU_LAYERS=0
MAIN_GPU=0
BATCH_SIZE=512
LOCAL_LLM_API_KEY=
```

- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.
- `MODEL_URL` - The model URL or repository name to download from Hugging Face. Default is `TheBloke/Mistral-7B-OpenOrca-GGUF`.
- `THREADS` - The number of threads to use.
- `THREADS_BATCH` - The number of threads to use for batch generation, this will enable parallel generation of batches. Setting it to the same value as threads will disable batch generation.
- `BATCH_SIZE` - The batch size to use for batch generation. Default is `512`.
- `GPU_LAYERS` - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` - The GPU to use for the main model. Default is `0`.

## Run with Docker

### CPU Only

Run with docker without a `.env` file, just replace the environment variables with your desired settings:

```bash
docker pull joshxt/local-llm:cpu
docker run -d --name local-llm -p 8091:8091 joshxt/local-llm:cpu -e MODEL_URL="TheBloke/Mistral-7B-OpenOrca-GGUF" -e THREADS="10" -e THREADS_BATCH="10" -e BATCH_SIZE="512" -e GPU_LAYERS="0" -e MAIN_GPU="0" -e LOCAL_LLM_API_KEY=""
```

Or with docker-compose after setting up your `.env` file:

```bash
docker-compose pull
docker-compose up
```

### NVIDIA GPU

If you're using an NVIDIA GPU, you can use the CUDA version of the server.

Run with docker without a `.env` file, just replace the environment variables with your desired settings:

```bash
docker pull joshxt/local-llm:cuda
docker run -d --name local-llm -p 8091:8091 --gpus all joshxt/local-llm:cuda -e MODEL_URL="TheBloke/Mistral-7B-OpenOrca-GGUF" -e THREADS="10" -e THREADS_BATCH="10" -e BATCH_SIZE="512" -e GPU_LAYERS="0" -e MAIN_GPU="0" -e LOCAL_LLM_API_KEY=""
```

Or with docker-compose after setting up your `.env` file:

```bash
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
