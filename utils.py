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
