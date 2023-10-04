import requests
import time
import json
import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from embedder import embed_text
from llamaprovider import LlamaProvider
from utils import (
    verify_api_key,
    get_tokens,
    make_post_data,
    make_res_data,
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
    function_call = None
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatInput(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    mirostat: Optional[float] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    seed: Optional[int] = None
    logit_bias: Optional[dict] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    tokenize: Optional[bool] = False


class EmbeddingModel(BaseModel):
    input: str
    model: str


# Chat completions endpoint
# https://platform.openai.com/docs/api-reference/chat
@app.post(
    "/v1/chat/completions", tags=["Completions"], dependencies=[Depends(verify_api_key)]
)
async def chat_completions(chat_input: ChatInput, user=Depends(verify_api_key)):
    stream = chat_input.stream
    tokenize = chat_input.tokenize
    post_data = make_post_data(chat_input, chat=True, stream=stream)

    prompt_token = []
    if tokenize:
        token_data = requests.post(
            f"{base_uri}/tokenize",
            json={"content": post_data["prompt"]},
        ).json()
        prompt_token = token_data["tokens"]

    if not stream:
        data = requests.post(f"{base_uri}/completion", json=post_data).json()
        res_data = make_res_data(data, chat=True, prompt_token=prompt_token)
        return res_data
    else:

        async def generate():
            data = requests.post(
                f"{base_uri}/completion",
                json=post_data,
                stream=True,
            )
            time_now = int(time.time())
            res_data = make_res_data_stream(
                {}, chat=True, time_now=time_now, start=True
            )
            yield "data: {}\n".format(json.dumps(res_data))
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    res_data = make_res_data_stream(
                        json.loads(decoded_line[6:]), chat=True, time_now=time_now
                    )
                    yield "data: {}\n".format(json.dumps(res_data))

        return StreamingResponse(generate(), media_type="text/event-stream")


# Completions endpoint
# https://platform.openai.com/docs/api-reference/completions
@app.post(
    "/v1/completions", tags=["Completions"], dependencies=[Depends(verify_api_key)]
)
async def completion(chat_input: ChatInput, user=Depends(verify_api_key)):
    stream = chat_input.stream
    tokenize = chat_input.tokenize
    post_data = make_post_data(chat_input, chat=False, stream=stream)
    prompt_token = []
    if tokenize:
        token_data = requests.post(
            f"{base_uri}/tokenize",
            json={"content": post_data["prompt"]},
        ).json()
        prompt_token = token_data["tokens"]

    if not stream:
        data = requests.post(f"{base_uri}/completion", json=post_data).json()
        res_data = make_res_data(data, chat=False, prompt_token=prompt_token)
        return res_data
    else:

        async def generate():
            data = requests.post(
                f"{base_uri}/completion",
                json=post_data,
                stream=True,
            )
            time_now = int(time.time())
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    res_data = make_res_data_stream(
                        json.loads(decoded_line[6:]), chat=False, time_now=time_now
                    )
                    yield "data: {}\n".format(json.dumps(res_data))

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


@app.post(
    "/api/v1/completions", tags=["Completions"], dependencies=[Depends(verify_api_key)]
)
async def completion(prompt: Completions, user=Depends(verify_api_key)):
    # prompt.model is the agent name
    agent = Interactions(agent_name=prompt.model, user=user)
    agent_config = agent.agent.AGENT_CONFIG
    if "settings" in agent_config:
        if "AI_MODEL" in agent_config["settings"]:
            model = agent_config["settings"]["AI_MODEL"]
        else:
            model = "undefined"
    else:
        model = "undefined"
    response = await agent.run(
        user_input=prompt.prompt,
        prompt="Custom Input",
        context_results=3,
        shots=prompt.n,
    )
    characters = string.ascii_letters + string.digits
    prompt_tokens = get_tokens(prompt.prompt)
    completion_tokens = get_tokens(response)
    total_tokens = int(prompt_tokens) + int(completion_tokens)
    random_chars = "".join(random.choice(characters) for _ in range(15))
    res_model = {
        "id": f"cmpl-{random_chars}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "text": response,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    return res_model


@app.post(
    "/api/v1/chat/completions",
    tags=["Completions"],
    dependencies=[Depends(verify_api_key)],
)
async def chat_completion(prompt: Completions, user=Depends(verify_api_key)):
    # prompt.model is the agent name
    agent = Interactions(agent_name=prompt.model, user=user)
    agent_config = agent.agent.AGENT_CONFIG
    if "settings" in agent_config:
        if "AI_MODEL" in agent_config["settings"]:
            model = agent_config["settings"]["AI_MODEL"]
        else:
            model = "undefined"
    else:
        model = "undefined"
    response = await agent.run(
        user_input=prompt.prompt,
        prompt="Custom Input",
        context_results=3,
        shots=prompt.n,
    )
    characters = string.ascii_letters + string.digits
    prompt_tokens = get_tokens(prompt.prompt)
    completion_tokens = get_tokens(response)
    total_tokens = int(prompt_tokens) + int(completion_tokens)
    random_chars = "".join(random.choice(characters) for _ in range(15))
    res_model = {
        "id": f"chatcmpl-{random_chars}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": [
                    {
                        "role": "assistant",
                        "content": response,
                    },
                ],
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    return res_model
