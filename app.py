from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from local_llm.LLM import LLM, streaming_generation
from local_llm.STT import STT
from local_llm.CTTS import CTTS
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="Local-LLM Server", docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "zephyr-7b-beta")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

CURRENT_MODEL = DEFAULT_MODEL if DEFAULT_MODEL else "zephyr-7b-beta"
CURRENT_STT_MODEL = WHISPER_MODEL if WHISPER_MODEL else "base.en"
print(f"[LLM] {CURRENT_MODEL} model loading...")
LOADED_LLM = LLM(model=CURRENT_MODEL)
print(f"[STT] {WHISPER_MODEL} model loading...")
LOADED_STT = STT(model=WHISPER_MODEL)

print(f"[CTTS] xttsv2_2.0.2 model loading...")
LOADED_CTTS = CTTS()


def verify_api_key(authorization: str = Header(None)):
    encryption_key = os.environ.get("LOCAL_LLM_API_KEY", "")
    if encryption_key:
        if authorization is None:
            raise HTTPException(
                status_code=401, detail="Authorization header is missing"
            )
        try:
            if "bearer " in authorization.lower():
                scheme, _, api_key = authorization.partition(" ")
            else:
                api_key = authorization
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
    global LOADED_LLM
    if LOADED_LLM:
        return LOADED_LLM.models()
    models = LLM().models()
    return models


# For the completions and chat completions endpoints, we use extra_body for additional parameters.
# --------------------------------
# If `audio_format`` is present, the prompt will be transcribed to text.
#   It is assumed it is base64 encoded audio in the `audio_format`` specified.
# --------------------------------
# If `system_message`` is present, it will be used as the system message for the completion.
# --------------------------------
# If `voice`` is present, the completion will be converted to audio using the specified voice.
#   If not streaming, the audio will be returned in the response in the "audio" beside the "text" or "content" keys.
#   If streaming, the audio will be streamed in the response in audio/wav format.


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
    extra_body: Optional[dict] = {}


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
    global CURRENT_MODEL
    global LOADED_LLM
    if c.model:
        if CURRENT_MODEL != c.model:
            CURRENT_MODEL = c.model
            LOADED_LLM = LLM(model=c.model)
    if c.max_tokens:
        LOADED_LLM.params["max_tokens"] = c.max_tokens
    if c.temperature:
        LOADED_LLM.params["temperature"] = c.temperature
    if c.top_p:
        LOADED_LLM.params["top_p"] = c.top_p
    if c.logit_bias:
        LOADED_LLM.params["logit_bias"] = c.logit_bias
    if c.stop:
        LOADED_LLM.params["stop"].append(c.stop)
    if c.extra_body:
        if "audio_format" in c.extra_body:
            prompt = LOADED_STT.transcribe_audio(
                base64_audio=c.messages[-1]["content"],
                audio_format=c.extra_body["audio_format"],
            )
            c.messages[-1]["content"] = prompt
        if "system_message" in c.extra_body:
            LOADED_LLM.params["system_message"] = c.extra_body["system_message"]
    response = LOADED_LLM.chat(messages=c.messages)
    audio_response = None
    if c.extra_body:
        if "voice" in c.extra_body:
            text_response = response["messages"][1]["content"]
            language = c.extra_body["language"] if "language" in c.extra_body else "en"
            audio_response = LOADED_CTTS.generate(
                text=text_response, voice=c.extra_body["voice"], language=language
            )
            response["messages"][1]["audio"] = audio_response
    if not c.stream:
        return response
    else:
        if audio_response:
            return StreamingResponse(
                streaming_generation(data=audio_response),
                media_type="audio/wav",
            )
        return StreamingResponse(
            streaming_generation(data=LOADED_LLM.chat(messages=c.messages)),
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
    extra_body: Optional[dict] = {}
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
    global CURRENT_MODEL
    global LOADED_LLM
    if c.model:
        if CURRENT_MODEL != c.model:
            CURRENT_MODEL = c.model
            LOADED_LLM = LLM(model=c.model)
    if c.max_tokens:
        LOADED_LLM.params["max_tokens"] = c.max_tokens
    if c.temperature:
        LOADED_LLM.params["temperature"] = c.temperature
    if c.top_p:
        LOADED_LLM.params["top_p"] = c.top_p
    if c.logit_bias:
        LOADED_LLM.params["logit_bias"] = c.logit_bias
    if c.stop:
        LOADED_LLM.params["stop"].append(c.stop)
    if c.extra_body:
        if "audio_format" in c.extra_body:
            prompt = LOADED_STT.transcribe_audio(
                base64_audio=c.prompt, audio_format=c.extra_body["audio_format"]
            )
            c.prompt = prompt
        if "system_message" in c.extra_body:
            LOADED_LLM.params["system_message"] = c.extra_body["system_message"]
    response = LOADED_LLM.completion(prompt=c.prompt, format_prompt=c.format_prompt)
    audio_response = None
    if c.extra_body:
        if "voice" in c.extra_body:
            text_response = response["choices"][0]["text"]
            language = c.extra_body["language"] if "language" in c.extra_body else "en"
            audio_response = LOADED_CTTS.generate(
                text=text_response, voice=c.extra_body["voice"], language=language
            )
            response["choices"][0]["audio"] = audio_response
    if not c.stream:
        return response
    else:
        if audio_response:
            return StreamingResponse(
                streaming_generation(data=audio_response),
                media_type="audio/wav",
            )
        return StreamingResponse(
            streaming_generation(
                data=response["choices"][0]["text"], format_prompt=c.format_prompt
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
async def embedding(embedding: EmbeddingModel, user=Depends(verify_api_key)):
    global CURRENT_MODEL
    global LOADED_LLM
    if embedding.model:
        if CURRENT_MODEL != embedding.model:
            CURRENT_MODEL = embedding.model
            LOADED_LLM = LLM(model=embedding.model)
    return LOADED_LLM.embedding(input=embedding.input)


@app.post(
    "/v1/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)],
)
async def embedding(embedding: EmbeddingModel, user=Depends(verify_api_key)):
    global CURRENT_MODEL
    global LOADED_LLM
    if embedding.model:
        if CURRENT_MODEL != embedding.model:
            CURRENT_MODEL = embedding.model
            LOADED_LLM = LLM(model=embedding.model)
    return LOADED_LLM.embedding(input=embedding.input)


class SpeechToText(BaseModel):
    file: str  # The base64 encoded audio file
    audio_format: Optional[str] = "wav"
    model: Optional[str] = WHISPER_MODEL
    user: Optional[str] = None


@app.post(
    "/v1/audio/transcriptions",
    tags=["Speech to Text"],
    dependencies=[Depends(verify_api_key)],
)
async def speech_to_text(stt: SpeechToText, user=Depends(verify_api_key)):
    global LOADED_STT
    if stt.model:
        if CURRENT_STT_MODEL != stt.model:
            LOADED_STT = STT(model=stt.model)
    response = await LOADED_STT.transcribe_audio(
        base64_audio=stt.file, audio_format=stt.audio_format
    )
    return {"data": response}


class TextToSpeech(BaseModel):
    text: str
    voice: Optional[str] = "default"
    language: Optional[str] = "en"
    user: Optional[str] = None


@app.post(
    "/v1/audio/generation",
    tags=["Text to Speech"],
    dependencies=[Depends(verify_api_key)],
)
async def text_to_speech(tts: TextToSpeech, user=Depends(verify_api_key)):
    global LOADED_CTTS
    audio = await LOADED_CTTS.generate(
        text=tts.text, voice=tts.voice, language=tts.language
    )
    return {"data": audio}


@app.get(
    "/v1/audio/voices",
    tags=["Text to Speech"],
    dependencies=[Depends(verify_api_key)],
)
async def get_voices(user=Depends(verify_api_key)):
    global LOADED_CTTS
    return {"voices": LOADED_CTTS.voices}
