from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import urllib.parse
import requests
import time
import json
import os


app = FastAPI(docs_url="/")

llama_api = "http://localhost:8080"
# Get api_key from environment LLAMACPP_API_KEY
api_key = os.environ.get("LLAMACPP_API_KEY", "")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatInput(BaseModel):
    messages: List[ChatMessage]
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


def is_present(data, key):
    return key in data


def convert_chat(messages):
    prompt = ""
    for line in messages:
        if line.role == "system":
            prompt += f"\nASSISTANT's RULE: {line.content}"
        elif line.role == "user":
            prompt += f"\nUSER: {line.content}"
        elif line.role == "assistant":
            prompt += f"\nASSISTANT: {line.content}"
    return prompt


def make_post_data(body, chat=False, stream=False):
    post_data = {}
    if chat:
        post_data["prompt"] = convert_chat(body.messages)
    else:
        post_data["prompt"] = body.prompt
    if is_present(body, "temperature"):
        post_data["temperature"] = body.temperature
    if is_present(body, "top_k"):
        post_data["top_k"] = body.top_k
    if is_present(body, "top_p"):
        post_data["top_p"] = body.top_p
    if is_present(body, "max_tokens"):
        post_data["n_predict"] = body.max_tokens
    if is_present(body, "presence_penalty"):
        post_data["presence_penalty"] = body.presence_penalty
    if is_present(body, "frequency_penalty"):
        post_data["frequency_penalty"] = body.frequency_penalty
    if is_present(body, "repeat_penalty"):
        post_data["repeat_penalty"] = body.repeat_penalty
    if is_present(body, "mirostat"):
        post_data["mirostat"] = body.mirostat
    if is_present(body, "mirostat_tau"):
        post_data["mirostat_tau"] = body.mirostat_tau
    if is_present(body, "mirostat_eta"):
        post_data["mirostat_eta"] = body.mirostat_eta
    if is_present(body, "seed"):
        post_data["seed"] = body.seed
    if is_present(body, "logit_bias"):
        post_data["logit_bias"] = [
            [int(token), body.logit_bias[token]] for token in body.logit_bias.keys()
        ]
    if is_present(body, "stop"):
        post_data["stop"] = body.stop
    else:
        post_data["stop"] = []
    post_data["n_keep"] = -1
    post_data["stream"] = stream
    return post_data


def make_res_data(data, chat=False, prompt_token=[]):
    res_data = {
        "id": "chatcmpl" if chat else "cmpl",
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "truncated": data["truncated"],
        "model": "LLaMA_CPP",
        "usage": {
            "prompt_tokens": data["tokens_evaluated"],
            "completion_tokens": data["tokens_predicted"],
            "total_tokens": data["tokens_evaluated"] + data["tokens_predicted"],
        },
    }
    if len(prompt_token) != 0:
        res_data["promptToken"] = prompt_token
    if chat:
        res_data["choices"] = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": data["content"],
                },
                "finish_reason": "stop"
                if (data["stopped_eos"] or data["stopped_word"])
                else "length",
            }
        ]
    else:
        res_data["choices"] = [
            {
                "text": data["content"],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
                if (data["stopped_eos"] or data["stopped_word"])
                else "length",
            }
        ]
    return res_data


def make_res_data_stream(data, chat=False, time_now=0, start=False):
    res_data = {
        "id": "chatcmpl" if chat else "cmpl",
        "object": "chat.completion.chunk" if chat else "text_completion.chunk",
        "created": time_now,
        "model": "LLaMA_CPP",
        "choices": [{"finish_reason": None, "index": 0}],
    }
    if chat:
        if start:
            res_data["choices"][0]["delta"] = {"role": "assistant"}
        else:
            res_data["choices"][0]["delta"] = {"content": data["content"]}
            if data["stop"]:
                res_data["choices"][0]["finish_reason"] = (
                    "stop"
                    if (data["stopped_eos"] or data["stopped_word"])
                    else "length"
                )
    else:
        res_data["choices"][0]["text"] = data["content"]
        if data["stop"]:
            res_data["choices"][0]["finish_reason"] = (
                "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
            )

    return res_data


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    chat_input: ChatInput, api_key: Optional[str] = Query(default="")
):
    if api_key != "" and api_key != api_key:
        raise HTTPException(status_code=403)

    stream = chat_input.stream
    tokenize = chat_input.tokenize
    post_data = make_post_data(chat_input, chat=True, stream=stream)

    prompt_token = []
    if tokenize:
        token_data = requests.post(
            urllib.parse.urljoin(llama_api, "/tokenize"),
            json={"content": post_data["prompt"]},
        ).json()
        prompt_token = token_data["tokens"]

    if not stream:
        data = requests.post(
            urllib.parse.urljoin(llama_api, "/completion"), json=post_data
        ).json()
        res_data = make_res_data(data, chat=True, prompt_token=prompt_token)
        return res_data
    else:

        async def generate():
            data = requests.post(
                urllib.parse.urljoin(llama_api, "/completion"),
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


@app.post("/completions")
@app.post("/v1/completions")
async def completion(chat_input: ChatInput, api_key: Optional[str] = Query(default="")):
    if api_key != "" and api_key != api_key:
        raise HTTPException(status_code=403)

    stream = chat_input.stream
    tokenize = chat_input.tokenize
    post_data = make_post_data(chat_input, chat=False, stream=stream)

    prompt_token = []
    if tokenize:
        token_data = requests.post(
            urllib.parse.urljoin(llama_api, "/tokenize"),
            json={"content": post_data["prompt"]},
        ).json()
        prompt_token = token_data["tokens"]

    if not stream:
        data = requests.post(
            urllib.parse.urljoin(llama_api, "/completion"), json=post_data
        ).json()
        res_data = make_res_data(data, chat=False, prompt_token=prompt_token)
        return res_data
    else:

        async def generate():
            data = requests.post(
                urllib.parse.urljoin(llama_api, "/completion"),
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
