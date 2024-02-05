# ezlocalai

[![GitHub](https://img.shields.io/badge/GitHub-Local%20LLM-blue?logo=github&style=plastic)](https://github.com/Josh-XT/ezlocalai) [![Dockerhub](https://img.shields.io/badge/Docker-Local%20LLM-blue?logo=docker&style=plastic)](https://hub.docker.com/r/joshxt/ezlocalai)

ezlocalai is a simple [llama.cpp](https://github.com/ggerganov/llama.cpp) server that easily exposes a list of local language models to choose from to run on your own computer. It is designed to be as easy as possible to get started with running local models. It automatically handles downloading the model of your choice and configuring the server based on your CPU, RAM, and GPU. It also includes [OpenAI Style](https://pypi.org/project/openai/) endpoints for easy integration with other applications. Additional functionality is built in for voice cloning text to speech and a voice to text for easy voice communication entirely offline after the initial setup.

## Prerequisites

- [Git](https://git-scm.com/downloads)
- [PowerShell 7.X](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell?view=powershell-7.4)
- [Docker Desktop](https://docs.docker.com/docker-for-windows/install/) (Windows or Mac)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA GPU only)

<details>
  <summary>Additional Linux Prerequisites</summary>

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

</details>

## Installation

```bash
git clone https://github.com/Josh-XT/ezlocalai
cd ezlocalai
```

### Environment Setup

Expand Environment Setup if you would like to modify the default environment variables, otherwise skip to Usage. All environment variables are optional and have useful defaults. Change the default model that starts with ezlocalai in your `.env` file.

<details>
  <summary>Environment Setup (Optional)</summary>

None of the values need modified in order to run the server. If you are using an NVIDIA GPU, I would recommend setting the `GPU_LAYERS` and `MAIN_GPU` environment variables. If you plan to expose the server to the internet, I would recommend setting the `EZLOCALAI_API_KEY` environment variable for security. `THREADS` is set to your CPU thread count minus 2 by default, if this causes significant performance issues, consider setting the `THREADS` environment variable manually to a lower number.

Modify the `.env` file to your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

Replace the environment variables with your desired settings. Assumptions will be made on all of these values if you choose to accept the defaults.

- `EZLOCALAI_API_KEY` - The API key to use for the server. If not set, the server will not require an API key when accepting requests.
- `DEFAULT_MODEL` - The default model to use when no model is specified. Default is `phi-2-dpo`.
- `WHISPER_MODEL` - The model to use for speech-to-text. Default is `base.en`.
- `AUTO_UPDATE` - Whether or not to automatically update ezlocalai. Default is `true`.
- `THREADS` - The number of CPU threads ezlocalai is allowed to use. Default is `your CPU thread count minus 2`.
- `GPU_LAYERS` (Only applicable to NVIDIA GPU) - The number of layers to use on the GPU. Default is `0`. ezlocalai will automatically determine the optimal number of layers to use based on your GPU's memory if it is set to 0 and you have an NVIDIA GPU.
- `MAIN_GPU` (Only applicable to NVIDIA GPU) - The GPU to use for the main model. Default is `0`.

</details>

## Usage

```bash
./start.ps1
```

For examples on how to use the server to communicate with the models, see the [Examples Jupyter Notebook](tests/tests.ipynb).

## OpenAI Style Endpoint Usage

OpenAI Style endpoints available at `http://<YOUR LOCAL IP ADDRESS>:8091/v1/` by default. Documentation can be accessed at that <http://localhost:8091> when the server is running.
