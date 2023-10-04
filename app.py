import requests
import time
import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from embedder import embed_text
from provider import LLM
from utils import (
    verify_api_key,
    get_tokens,
    make_res_data_stream,
)

app = FastAPI(title="Local-LLM Server", docs_url="/")

base_uri = "http://localhost:8080"


class Completions(BaseModel):
    model: str = None  # Model is actually the agent_name
    prompt: str = None
    suffix: str = None
    max_tokens: int = 100
    temperature: float = 0.9
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: int = None
    echo: bool = False
    stop: List[str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    logit_bias: Dict[str, float] = None
    user: str = None


class ChatCompletions(BaseModel):
    model: str = None  # Model is actually the agent_name
    messages: List[dict] = None
    functions: List[dict] = None
    function_call: str = None
    temperature: float = 0.9
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = None
    max_tokens: int = 8192
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Dict[str, float] = None
    user: str = None


class EmbeddingModel(BaseModel):
    input: str
    model: str
    user: str = None


# Chat completions endpoint
# https://platform.openai.com/docs/api-reference/chat
@app.post(
    "/v1/chat/completions", tags=["Completions"], dependencies=[Depends(verify_api_key)]
)
async def chat_completions(c: ChatCompletions, user=Depends(verify_api_key)):
    stream = c.stream
    messages = c.messages
    if len(messages) > 1:
        for message in messages:
            if message["role"] == "system":
                prompt = f"\nASSISTANT's RULE: {message.content}"
            elif message["role"] == "user":
                prompt = f"\nUSER: {message.content}"
            elif message["role"] == "assistant":
                prompt = f"\nASSISTANT: {message.content}"
    else:
        prompt = messages[0]["content"]
    tokens = get_tokens(prompt)
    if not stream:
        data = LLM(**c).instruct(prompt=prompt, tokens=tokens)
        return data
    else:

        async def generate():
            data = LLM(**c).instruct(prompt=c.prompt, tokens=tokens)
            yield "data: {}\n".format(json.dumps(data))
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    current_data = json.loads(decoded_line[6:])
                    yield "data: {}\n".format(json.dumps(current_data))

        return StreamingResponse(generate(), media_type="text/event-stream")


# Completions endpoint
# https://platform.openai.com/docs/api-reference/completions
@app.post(
    "/v1/completions", tags=["Completions"], dependencies=[Depends(verify_api_key)]
)
async def completions(c: Completions, user=Depends(verify_api_key)):
    stream = c.stream
    tokens = get_tokens(c.prompt)
    if not stream:
        data = LLM(**c).instruct(prompt=c.prompt, tokens=tokens)
        return data
    else:

        async def generate():
            data = LLM(**c).instruct(prompt=c.prompt, tokens=tokens)
            yield "data: {}\n".format(json.dumps(data))
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    current_data = json.loads(decoded_line[6:])
                    yield "data: {}\n".format(json.dumps(current_data))

        return StreamingResponse(generate(), media_type="text/event-stream")


# Embeddings endpoint
# https://platform.openai.com/docs/api-reference/embeddings
@app.post(
    "/v1/embeddings", tags=["Completions"], dependencies=[Depends(verify_api_key)]
)
async def embedding(embedding: EmbeddingModel, user=Depends(verify_api_key)):
    tokens = get_tokens(embedding.input)
    if tokens > 256:
        raise HTTPException("Input text is too long. Max length is 256 tokens.")
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embed_text(text=embedding.input),
                "index": 0,
            }
        ],
        "model": embedding.model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
