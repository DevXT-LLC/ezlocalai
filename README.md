# llamacpp-server

This llamacpp server will allow you to start it with a `MODEL_URL` defined in the `.env` file instead of needing to manually go to Hugging Face and download the model on the server.

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
LLAMACPP_PORT=8080
LLAMACPP_IMAGE=full-cuda
```

TheBloke sticks to the same naming convention for his models, so you can just use the model name and it will automatically download the model from Hugging Face. If the model repositories are not in the format he uses, you can use the full URL to the model of the download link.

## Clone the repository

```bash
git clone https://github.com/Josh-XT/llamacpp-server
cd llamacpp-server
```

Modify the `.env` file if desired, then in your terminal you can start the server with:

```bash
docker-compose up --build
```

The server API is available at `http://localhost:8080` by default if your `LLAMACPP_PORT` is set to `8080` in the `.env` file.

[Documentation for the API is available here](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#api-endpoints)
