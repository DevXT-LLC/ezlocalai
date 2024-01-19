from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from local_llm import LLM, streaming_generation, DEFAULT_MODEL
import os
import jwt


app = FastAPI(title="Local-LLM Server", docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DEFAULT_MODEL != "":
    default_llm = LLM(model=DEFAULT_MODEL)
else:
    default_llm = None


def verify_api_key(authorization: str = Header(None)):
    encryption_key = os.environ.get("LOCAL_LLM_API_KEY", "")
    using_jwt = (
        True if os.environ.get("USING_JWT", "false").lower() == "true" else False
    )
    if encryption_key:
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
            if using_jwt:
                token = jwt.decode(
                    jwt=api_key,
                    key=encryption_key,
                    algorithms=["HS256"],
                )
                return token["email"]
            else:
                if api_key != encryption_key:
                    raise HTTPException(status_code=401, detail="Invalid API Key")
                return "USER"
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    else:
        return "USER"


@app.get(
    "/v1/models",
    tags=["Models"],
    dependencies=[Depends(verify_api_key)],
)
async def models(user=Depends(verify_api_key)):
    if default_llm:
        return default_llm.models()
    models = LLM().models()
    return models


# Chat completions endpoint
# https://platform.openai.com/docs/api-reference/chat
class ChatCompletions(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[dict] = None
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 1.0
    functions: Optional[List[dict]] = None
    function_call: Optional[str] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 8192
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    system_message: Optional[str] = ""


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict


@app.post(
    "/v1/chat/completions",
    tags=["Completions"],
    dependencies=[Depends(verify_api_key)],
)
async def chat_completions(c: ChatCompletions, user=Depends(verify_api_key)):
    if DEFAULT_MODEL == c.model:
        if c.max_tokens:
            default_llm.params["max_tokens"] = c.max_tokens
        if c.temperature:
            default_llm.params["temperature"] = c.temperature
        if c.top_p:
            default_llm.params["top_p"] = c.top_p
        if c.logit_bias:
            default_llm.params["logit_bias"] = c.logit_bias
        if c.stop:
            default_llm.params["stop"].append(c.stop)
        if c.system_message:
            default_llm.params["system_message"] = c.system_message
    if not c.stream:
        if DEFAULT_MODEL == c.model:
            return default_llm.chat(messages=c.messages)
        return LLM(**c.model_dump()).chat(messages=c.messages)
    else:
        if DEFAULT_MODEL == c.model:
            return StreamingResponse(
                streaming_generation(data=default_llm.chat(messages=c.messages)),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            streaming_generation(data=LLM(**c.model_dump()).chat(messages=c.messages)),
            media_type="text/event-stream",
        )


# Completions endpoint
# https://platform.openai.com/docs/api-reference/completions
class Completions(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str = ""
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[List[str]] = None
    echo: Optional[bool] = False
    system_message: Optional[str] = ""
    user: Optional[str] = None
    format_prompt: Optional[bool] = True


class CompletionsResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict


@app.post(
    "/v1/completions",
    tags=["Completions"],
    dependencies=[Depends(verify_api_key)],
)
async def completions(c: Completions, user=Depends(verify_api_key)):
    if DEFAULT_MODEL == c.model:
        if c.max_tokens:
            default_llm.params["max_tokens"] = c.max_tokens
        if c.temperature:
            default_llm.params["temperature"] = c.temperature
        if c.top_p:
            default_llm.params["top_p"] = c.top_p
        if c.logit_bias:
            default_llm.params["logit_bias"] = c.logit_bias
        if c.stop:
            default_llm.params["stop"].append(c.stop)
        if c.system_message:
            default_llm.params["system_message"] = c.system_message
    if not c.stream:
        if DEFAULT_MODEL == c.model:
            return default_llm.completion(
                prompt=c.prompt, format_prompt=c.format_prompt
            )
        return LLM(**c.model_dump()).completion(
            prompt=c.prompt, format_prompt=c.format_prompt
        )
    else:
        if DEFAULT_MODEL == c.model:
            return StreamingResponse(
                streaming_generation(
                    data=default_llm.completion(
                        prompt=c.prompt, format_prompt=c.format_prompt
                    )
                ),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            streaming_generation(
                data=LLM(**c.model_dump()).completion(
                    prompt=c.prompt, format_prompt=c.format_prompt
                )
            ),
            media_type="text/event-stream",
        )


# Embeddings endpoint
# https://platform.openai.com/docs/api-reference/embeddings
class EmbeddingModel(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = DEFAULT_MODEL
    user: Optional[str] = None


class EmbeddingResponse(BaseModel):
    object: str
    data: List[dict]
    model: str
    usage: dict


@app.post(
    "/v1/engines/{model_name}/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)],
)
async def embedding(
    model_name: str, embedding: EmbeddingModel, user=Depends(verify_api_key)
):
    if DEFAULT_MODEL == embedding.model:
        return default_llm.embedding(input=embedding.input)
    return LLM(model=model_name).embedding(input=embedding.input)


@app.post(
    "/v1/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)],
)
async def embedding(embedding: EmbeddingModel, user=Depends(verify_api_key)):
    if DEFAULT_MODEL == embedding.model:
        return default_llm.embedding(input=embedding.input)
    return LLM(model=embedding.model).embedding(input=embedding.input)
