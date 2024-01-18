# Local-LLM

- [Dockerhub](https://hub.docker.com/r/joshxt/local-llm/tags)
- [GitHub](https://github.com/Josh-XT/Local-LLM)

Local-LLM is a [llama.cpp](https://github.com/ggerganov/llama.cpp) server in Docker with OpenAI Style Endpoints that allows you to send the model name as the name of the model as it appears in the model list, for example `phi-2-dpo`. It will automatically download the model from Hugging Face if it isn't already downloaded and configure the server for you. It automatically configures the server based on your CPU, RAM, and GPU. It is designed to be as easy as possible to get started with running local models.

## Table of Contents 📖

- [Local-LLM](#local-llm)
  - [Table of Contents 📖](#table-of-contents-)
  - [Environment Setup](#environment-setup)
  - [Run Local-LLM](#run-local-llm)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
  - [OpenAI Style Endpoint Usage](#openai-style-endpoint-usage)

## Environment Setup

Modify the `.env` file to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

Replace the environment variables with your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.
- `THREADS` - The number of threads to use. Default is `your CPU core count minus 1`.

The following are only applicable to NVIDIA GPUs:

- `GPU_LAYERS` - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` - The GPU to use for the main model. Default is `0`.

## Run Local-LLM

You can choose to run locally with the instructions below, or with [Docker](Docker.md). Both are not needed. Instructions to run with Docker or Docker Compose can be found [here](Docker.md).

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Python 3.10](https://www.python.org/downloads/)
- [PowerShell 7.X](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell?view=powershell-7.4)

### Installation

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
pip install -r requirements.txt
```

### Usage

Make your modifications to the `.env` file or proceed to accept defaults running on CPU without an API key.

```bash
./start.ps1
```

## OpenAI Style Endpoint Usage

OpenAI Style endpoints available at `http://<YOUR LOCAL IP ADDRESS>:8091/v1` by default. Documentation can be accessed at that <http://localhost:8091> when the server is running. There are examples for each of the endpoints in the [Examples Jupyter Notebook](tests/tests.ipynb).
