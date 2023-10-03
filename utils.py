from fastapi import HTTPException, Header
import time
import os
import re
import tiktoken
import jwt

# Get api_key from environment LLAMACPP_API_KEY
LOCAL_LLM_API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "")
USING_JWT = True if os.environ.get("USING_JWT", "false").lower() == "true" else False
MODEL_MAX_TOKENS = os.environ.get("MAX_TOKENS", 8192)
try:
    with open(f"models/prompt.txt", "r") as f:
        prompt_template = f.read()
except:
    prompt_template = "{system_message}\n\n{prompt}"


def verify_api_key(authorization: str = Header(None)):
    if LOCAL_LLM_API_KEY:
        if authorization is None:
            raise HTTPException(
                status_code=401, detail="Authorization header is missing"
            )
        try:
            scheme, _, api_key = authorization.partition(" ")
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=401, detail="Invalid authentication scheme"
                )
            if USING_JWT:
                token = jwt.decode(
                    jwt=api_key,
                    key=LOCAL_LLM_API_KEY,
                    algorithms=["HS256"],
                )
                return token["email"]
            else:
                if api_key != LOCAL_LLM_API_KEY:
                    raise HTTPException(status_code=401, detail="Invalid API Key")
                return "USER"
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    else:
        return "USER"


def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def custom_format(string, **kwargs):
    if isinstance(string, list):
        string = "".join(str(x) for x in string)

    def replace(match):
        key = match.group(1)
        value = kwargs.get(key, match.group(0))
        if isinstance(value, list):
            return "".join(str(x) for x in value)
        else:
            return str(value)

    pattern = r"(?<!{){([^{}\n]+)}(?!})"
    result = re.sub(pattern, replace, string)
    return result


def format_prompt(prompt, system_message=""):
    formatted_prompt = custom_format(
        string=prompt_template, prompt=prompt, system_message=system_message
    )
    return formatted_prompt


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
    tokens = get_tokens(post_data["prompt"])
    try:
        soft_max = int(MODEL_MAX_TOKENS) - 10
    except:
        soft_max = 8182
    if tokens > soft_max:
        post_data["prompt"] = post_data["prompt"][-soft_max:]
    post_data["prompt"] = format_prompt(prompt=post_data["prompt"])
    if is_present(body, "temperature"):
        post_data["temperature"] = body.temperature
    if is_present(body, "top_k"):
        post_data["top_k"] = body.top_k
    if is_present(body, "top_p"):
        post_data["top_p"] = body.top_p
    if is_present(body, "max_tokens"):
        try:
            if int(body.max_tokens) > int(MODEL_MAX_TOKENS):
                body.max_tokens = int(MODEL_MAX_TOKENS)
        except:
            body.max_tokens = MODEL_MAX_TOKENS
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
