# Local-LLM

Local-LLM is a llama.cpp server in Docker with OpenAI Style Endpoints.

This server comes equipped with the OpenAI style endpoints that most software is familiar with. It will allow you to start it with a `MODEL_URL` defined in the `.env` file instead of needing to manually go to Hugging Face and download the model on the server.

TheBloke sticks to the same naming convention for his models, so you can just use the model repository name like `TheBloke/Mistral-7B-OpenOrca-GGUF` and it will automatically download the model from Hugging Face. If the model repositories are not in the format he uses, you can use the full URL to the model of the download link like `https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral.7b.q5_k_s.gguf` and it will download the quantized model from Hugging Face.

## Environment Set Up

Create a `.env` file if one does not exist and modify it to your needs. This is the default `.env` file if cloning the repository, modify it to your needs:

```env
MODEL_URL=TheBloke/Mistral-7B-OpenOrca-GGUF
QUANT_TYPE=Q4_K_M
MAX_TOKENS=8192
THREADS=20
THREADS_BATCH=20
GPU_LAYERS=20
MAIN_GPU=0
BATCH_SIZE=512
UVICORN_WORKERS=2
LOCAL_LLM_API_KEY=
```

- `MODEL_URL` - The model URL or repository name to download from Hugging Face. Default is `TheBloke/Mistral-7B-OpenOrca-GGUF`.
- `QUANT_TYPE` - The quantization type to use. Default is `Q4_K_M`.
- `MAX_TOKENS` - The maximum number of tokens. Default is `8192`.
- `THREADS` - The number of threads to use.
- `THREADS_BATCH` - The number of threads to use for batch generation, this will enable parallel generation of batches. Setting it to the same value as threads will disable batch generation.
- `GPU_LAYERS` - The number of layers to use on the GPU.
- `MAIN_GPU` - The GPU to use for the main model.
- `BATCH_SIZE` - The batch size to use for batch generation. Default is `512`.
- `UVICORN_WORKERS` - The number of uvicorn workers to use. Default is `2`.
- `LOCAL_LLM_API_KEY` - The API key to use for the server. If not set, the server will not require an API key.

## CPU Only

Run with docker:

```bash
docker pull joshxt/local-llm:cpu
docker run -d --name local-llm -p 8091:8091 joshxt/local-llm:cpu --env-file .env
```

Or with docker-compose:

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
docker-compose pull
docker-compose up
```

## NVIDIA GPU

If you're using an NVIDIA GPU, you can use the CUDA version of the server.

Run with docker:

```bash
docker pull joshxt/local-llm:cuda
docker run -d --name local-llm -p 8091:8091 --gpus all joshxt/local-llm:cuda --env-file .env
```

Or with docker-compose:

```bash
git clone https://github.com/Josh-XT/Local-LLM
cd Local-LLM
docker-compose -f docker-compose-cuda.yml pull
docker-compose -f docker-compose-cuda.yml up
```

## OpenAI Style Endpoint Usage

OpenAI Style endpoints available at `http://localhost:8091/` by default. Documentation can be accessed at that url when the server is running.

### Completion

```python
import openai

openai.api_base = "http://localhost:8091/v1"
openai.api_key = "YOUR API KEY IF YOU SET ONE IN THE .env FILE"
prompt = "Tell me something funny about llamas."

response = openai.Completion.create(
    engine="llamacpp",
    prompt=prompt,
    temperature=1.31,
    max_tokens=8192,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
    stream=False,
)
message = response.choices[0].text.strip()
print(message)
```

### Chat Completion

```python
import openai

openai.api_base = "http://localhost:8091/v1"
openai.api_key = "YOUR API KEY IF YOU SET ONE IN THE .env FILE"
prompt = "Tell me something funny about llamas."
messages = [{"role": "system", "content": prompt}]

response = openai.ChatCompletion.create(
    model="llamacpp",
    messages=messages,
    temperature=1.31,
    max_tokens=8192,
    top_p=1.0,
    n=1,
    stream=False,
)
message = response.choices[0].message.content.strip()
print(message)
```
