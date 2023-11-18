# Local-LLM

- [Dockerhub](https://hub.docker.com/r/joshxt/local-llm/tags)
- [GitHub](https://github.com/Josh-XT/Local-LLM)

Local-LLM is a [llama.cpp](https://github.com/ggerganov/llama.cpp) server in Docker with OpenAI Style Endpoints that allows you to send the model name as the name of the model as it appears in the model list, for example `Mistral-7B-OpenOrca`. It will automatically download the model from Hugging Face if it isn't already downloaded and configure the server for you. It automatically configures the server based on your CPU, RAM, and GPU. It is designed to be as easy as possible to get started with running local models.

## Table of Contents 📖

- [Local-LLM](#local-llm)
  - [Table of Contents 📖](#table-of-contents-)
  - [Run with Docker](#run-with-docker)
    - [Prerequisites](#prerequisites)
    - [Run without NVIDIA GPU support](#run-without-nvidia-gpu-support)
    - [Run with NVIDIA GPU support](#run-with-nvidia-gpu-support)
  - [Run with Docker Compose](#run-with-docker-compose)
    - [Run without NVIDIA GPU support with Docker Compose](#run-without-nvidia-gpu-support-with-docker-compose)
    - [Run with NVIDIA GPU support with Docker Compose](#run-with-nvidia-gpu-support-with-docker-compose)
  - [OpenAI Style Endpoint Usage](#openai-style-endpoint-usage)
  - [Shout Outs](#shout-outs)

## Run with Docker

You can choose to run with Docker or [Docker Compose](DockerCompose.md). Both are not needed. Instructions to run with Docker Compose can be found [here](DockerCompose.md).

Replace the environment variables with your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.
- `THREADS` - The number of threads to use. Default is `your CPU core count minus 1`.

The following are only applicable to NVIDIA GPUs:

- `GPU_LAYERS` - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` - The GPU to use for the main model. Default is `0`.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)

### Run without NVIDIA GPU support

Modify the `THREADS` environment variable to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

```bash
docker pull joshxt/local-llm:cpu
docker run -d --name local-llm -p 8091:8091 joshxt/local-llm:cpu -e THREADS="10" -e LOCAL_LLM_API_KEY="" -v ./models:/app/models
```

### Run with NVIDIA GPU support

If you're using an NVIDIA GPU, you can use the CUDA version of the server. You must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed if using NVIDIA GPU.

Modify the `GPU_LAYERS`, `MAIN_GPU`, and `THREADS` environment variables to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

```bash
docker pull joshxt/local-llm:cuda
docker run -d --name local-llm -p 8091:8091 --gpus all joshxt/local-llm:cuda -e THREADS="10" -e GPU_LAYERS="20" -e MAIN_GPU="0" -e LOCAL_LLM_API_KEY="" -v ./models:/app/models
```

## Run with Docker Compose

You can choose to run with Docker Compose or Docker. Both are not needed.

Update the `.env` file with your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

### Run without NVIDIA GPU support with Docker Compose

```bash
docker-compose pull
docker-compose up
```

### Run with NVIDIA GPU support with Docker Compose

```bash
docker-compose -f docker-compose-cuda.yml pull
docker-compose -f docker-compose-cuda.yml up
```

## OpenAI Style Endpoint Usage

OpenAI Style endpoints available at `http://<YOUR LOCAL IP ADDRESS>:8091/v1` by default. Documentation can be accessed at that <http://localhost:8091> when the server is running. There are examples for each of the endpoints in the [Examples Jupyter Notebook](examples.ipynb).

## Shout Outs

- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - For constantly improving the ability for anyone to run local models. It is one of my favorite and most exciting projects on GitHub.
- [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - For making it easy to extend the functionality of llama.cpp in Python.
- [TheBloke](https://huggingface.co/TheBloke) - For helping enable the ability to run local models by quantizing them and sharing them with a great readme on how to use them in every repository.
- [Meta](https://meta.com) - For the absolutely earth shattering open source releases of the LLaMa models and all other contributions they have made to Open Source.
- [OpenAI](https://openai.com/) - For setting good standards for endpoints and making great models.
- [Hugging Face](https://huggingface.co/) - For making it easy to use and share models.
- As much as I hate to do it, I can't list all of the amazing people building and fine tuning local models, but you know who you are. Thank you for all of your hard work and contributions to the community!
