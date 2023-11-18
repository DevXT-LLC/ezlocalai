from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
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


# Embeddings endpoint
# https://platform.openai.com/docs/api-reference/embeddings
class EmbeddingModel(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "Mistral-7B-OpenOrca"
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
    return LLM(model=model_name).embedding(input=embedding.input)


@app.post(
    "/v1/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)],
)
async def embedding(embedding: EmbeddingModel, user=Depends(verify_api_key)):
    return LLM(model=embedding.model).embedding(input=embedding.input)
