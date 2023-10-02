# llamacpp-server in Docker with OpenAI Style Endpoints

This llamacpp server comes equipped with the OpenAI style endpoints that most software is familiar with. will allow you to start it with a `MODEL_URL` defined in the `.env` file instead of needing to manually go to Hugging Face and download the model on the server.

This is the default `.env` file, modify it to your needs:

```env
MODEL_URL=TheBloke/Mistral-7B-OpenOrca-GGUF
QUANT_TYPE=Q5_K_S
MAX_TOKENS=8192
THREADS=20
THREADS_BATCH=20
GPU_LAYERS=20
MAIN_GPU=0
BATCH_SIZE=512
UVICORN_WORKERS=2
LLAMACPP_API_KEY=
```

TheBloke sticks to the same naming convention for his models, so you can just use the model name and it will automatically download the model from Hugging Face. If the model repositories are not in the format he uses, you can use the full URL to the model of the download link.

## Clone the repository

```bash
git clone https://github.com/Josh-XT/llamacpp-server
cd llamacpp-server
```

Modify the `.env` file if desired before proceeding.

### NVIDIA GPU

If running without an NVIDIA GPU, you can start the server with:

```bash
docker-compose -f docker-compose-cuda.yml up
```

Or if you only want the OpenAPI Style endpoints exposed:

```bash
docker-compose -f docker-compose-cuda-openai.yml up
```

### CPU Only

If you are not running on an NVIDIA GPU, you can start the server with:

```bash
docker-compose up
```

Or if you only want the OpenAPI Style endpoints exposed:

```bash
docker-compose -f docker-compose-openai.yml up
```

The llamacpp server API is available at `http://localhost:8090` by default. The [documentation for the API is available here.](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#api-endpoints)

OpenAI Style endpoints available at `http://localhost:8091/` by default. Documentation can be accessed at that url when the server is running.
