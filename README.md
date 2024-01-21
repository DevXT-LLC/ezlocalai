# Local-LLM

[![GitHub](https://img.shields.io/badge/GitHub-Local%20LLM-blue?logo=github&style=plastic)](https://github.com/Josh-XT/Local-LLM) [![Dockerhub](https://img.shields.io/badge/Docker-Local%20LLM-blue?logo=docker&style=plastic)](https://hub.docker.com/r/joshxt/local-llm)

Local-LLM is a simple [llama.cpp](https://github.com/ggerganov/llama.cpp) server that easily exposes a list of local language models to choose from to run on your own computer. It is designed to be as easy as possible to get started with running local models. It automatically handles downloading the model of your choice and configuring the server based on your CPU, RAM, and GPU. It also includes [OpenAI Style](https://pypi.org/project/openai/) endpoints for easy integration with other applications.

## Prerequisites

- [Git](https://git-scm.com/downloads)
- [PowerShell 7.X](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell?view=powershell-7.4)
- [Docker Desktop](https://docs.docker.com/docker-for-windows/install/) (Windows or Mac)

<details>
  <summary>Additional Linux Prerequisites</summary>

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

</details>

## Installation

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
```

Expand Environment Setup if you would like to modify the default environment variables, otherwise skip to Usage.

<details>
  <summary>Environment Setup (Optional)</summary>

None of the values need modified in order to run the server. If you are using an NVIDIA GPU, I would recommend setting the `GPU_LAYERS` and `MAIN_GPU` environment variables. If you plan to expose the server to the internet, I would recommend setting the `LOCAL_LLM_API_KEY` environment variable for security. `THREADS` is set to your CPU thread count minus 2 by default, if this causes significant performance issues, consider setting the `THREADS` environment variable manually to a lower number.

Modify the `.env` file to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

Replace the environment variables with your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key when accepting requests.
- `DEFAULT_MODEL` - The default model to use when no model is specified. Default is `phi-2-dpo`.
- `MULTI_SERVER` - This will run two servers, one with `zephyr-7b-beta` running on GPU, and one with `phi-2-dpo` running on CPU. If set, this will run both, otherwise it will only run one server.
- `AUTO_UPDATE` - Whether or not to automatically update Local-LLM. Default is `true`.
- `THREADS` - The number of CPU threads Local-LLM is allowed to use. Default is `your CPU thread count minus 2`.
- `GPU_LAYERS` (Only applicable to NVIDIA GPU) - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` (Only applicable to NVIDIA GPU) - The GPU to use for the main model. Default is `0`.

</details>

## Usage

```bash
./start.ps1
```

For examples on how to use the server to communicate with the models, see the [Examples Jupyter Notebook](tests/tests.ipynb).

## OpenAI Style Endpoint Usage

OpenAI Style endpoints available at `http://<YOUR LOCAL IP ADDRESS>:8091/v1/` by default. Documentation can be accessed at that <http://localhost:8091> when the server is running.
