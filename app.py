from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from local_llm import LLM, streaming_generation
import os
import jwt


app = FastAPI(title="Local-LLM Server", docs_url="/")


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
    models = LLM().models()
    return models


# Chat completions endpoint
# https://platform.openai.com/docs/api-reference/chat
class ChatCompletions(BaseModel):
    model: str = "Mistral-7B-OpenOrca"
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
    if not c.stream:
        return LLM(**c.model_dump()).chat(messages=c.messages)
    else:
        return StreamingResponse(
            streaming_generation(data=LLM(**c.model_dump()).chat(messages=c.messages)),
            media_type="text/event-stream",
        )


# Completions endpoint
# https://platform.openai.com/docs/api-reference/completions
class Completions(BaseModel):
    model: str = "Mistral-7B-OpenOrca"
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
    if not c.stream:
        return LLM(**c.model_dump()).completion(prompt=c.prompt)
    else:
        return StreamingResponse(
            streaming_generation(
                data=LLM(**c.model_dump()).completion(prompt=c.prompt)
            ),
            media_type="text/event-stream",
        )


# Embeddings endpoint
# https://platform.openai.com/docs/api-reference/embeddings
class EmbeddingModel(BaseModel):
    input: str
    model: str = "Mistral-7B-OpenOrca"
    user: str = None


class EmbeddingResponse(BaseModel):
    object: str
    data: List[dict]
    model: str
    usage: dict


@app.post(
    "/v1/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)],
)
async def embedding(embedding: EmbeddingModel, user=Depends(verify_api_key)):
    return LLM(model=embedding.model).embedding(input=embedding.input)
