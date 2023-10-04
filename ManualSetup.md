# Manual Setup - Alternative to Local-LLM Configurator UI

Create a `.env` file if one does not exist and modify it to your needs. Here is an example `.env` file:

```env
MODEL_URL=TheBloke/Mistral-7B-OpenOrca-GGUF
MAX_TOKENS=8192
THREADS=10
THREADS_BATCH=10
GPU_LAYERS=0
MAIN_GPU=0
BATCH_SIZE=512
LOCAL_LLM_API_KEY=
```

- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.
- `MODEL_URL` - The model URL or repository name to download from Hugging Face. Default is `TheBloke/Mistral-7B-OpenOrca-GGUF`.
- `MAX_TOKENS` - The maximum number of tokens. Default is `8192`.
- `THREADS` - The number of threads to use.
- `THREADS_BATCH` - The number of threads to use for batch generation, this will enable parallel generation of batches. Setting it to the same value as threads will disable batch generation.
- `BATCH_SIZE` - The batch size to use for batch generation. Default is `512`.
- `GPU_LAYERS` - The number of layers to use on the GPU. Default is `0`.
- `MAIN_GPU` - The GPU to use for the main model. Default is `0`.
