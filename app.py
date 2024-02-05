from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from ezlocalai.LLM import LLM, streaming_generation
from ezlocalai.STT import STT
from ezlocalai.CTTS import CTTS
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info(f"[CTTS] xttsv2_2.0.2 model loading. Please wait...")
LOADED_CTTS = CTTS()
logging.info(f"[CTTS] xttsv2_2.0.2 model loaded successfully.")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
CURRENT_STT_MODEL = WHISPER_MODEL if WHISPER_MODEL else "base.en"
logging.info(f"[STT] {CURRENT_STT_MODEL} model loading. Please wait...")
LOADED_STT = STT(model=CURRENT_STT_MODEL)
logging.info(f"[STT] {CURRENT_STT_MODEL} model loaded successfully.")

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "zephyr-7b-beta")
CURRENT_MODEL = DEFAULT_MODEL if DEFAULT_MODEL else "zephyr-7b-beta"
logging.info(f"[LLM] {CURRENT_MODEL} model loading. Please wait...")
LOADED_LLM = LLM(model=CURRENT_MODEL)
logging.info(f"[LLM] {CURRENT_MODEL} model loaded successfully.")

VISION_MODEL = os.getenv("VISION_MODEL", "")
LOADED_VISION_MODEL = None
if VISION_MODEL != "":
    LOADED_VISION_MODEL = LLM(model=VISION_MODEL)  # bakllava-1-7b
    logging.info(f"[ezlocalai] Vision is enabled.")

logging.info(f"[ezlocalai] Server is ready.")

app = FastAPI(title="ezlocalai Server", docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


def verify_api_key(authorization: str = Header(None)):
    encryption_key = os.environ.get("EZLOCALAI_API_KEY", "")
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


# For the completions and chat completions endpoints, we use extra_json for additional parameters.
# --------------------------------
# If `audio_format`` is present, the prompt will be transcribed to text.
#   It is assumed it is base64 encoded audio in the `audio_format`` specified.
# --------------------------------
# If `system_message`` is present, it will be used as the system message for the completion.
# --------------------------------
# If `voice`` is present, the completion will be converted to audio using the specified voice.
#   If not streaming, the audio will be returned in the response in the "audio" beside the "text" or "content" keys.
#   If streaming, the audio will be streamed in the response in audio/wav format.


def create_audio_control(audio: str):
    return f"""<audio controls><source src="data:audio/wav;base64,{audio}" type="audio/wav"></audio>"""


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


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict


async def get_response(data, completion_type="chat"):
    global CURRENT_MODEL
    global LOADED_LLM
    if data["model"]:
        if CURRENT_MODEL != data["model"]:
            CURRENT_MODEL = data["model"]
            LOADED_LLM = LLM(model=data["model"])
    if "stop" in data:
        new_stop = LOADED_LLM.params["stop"]
        new_stop.append(data["stop"])
        data["stop"] = new_stop
    if "audio_format" in data:
        base64_audio = (
            data["messages"][-1]["content"]
            if completion_type == "chat"
            else data["prompt"]
        )
        prompt = await LOADED_STT.transcribe_audio(
            base64_audio=base64_audio,
            audio_format=data["audio_format"],
        )
        if completion_type == "chat":
            data["messages"][-1]["content"] = prompt
        else:
            data["prompt"] = prompt
    if completion_type == "chat":
        response = LOADED_LLM.chat(**data)
    else:
        response = LOADED_LLM.completion(**data)
    audio_response = None
    if "voice" in data:
        if completion_type == "chat":
            text_response = response["messages"][1]["content"]
        else:
            text_response = response["choices"][0]["text"]
        language = data["language"] if "language" in data else "en"
        if "url_output" in data:
            url_output = data["url_output"].lower() == "true"
        else:
            url_output = True
        audio_response = await LOADED_CTTS.generate(
            text=text_response,
            voice=data["voice"],
            language=language,
            url_output=url_output,
        )
        audio_control = (
            audio_response if url_output else create_audio_control(audio_response)
        )
        if completion_type == "chat":
            response["messages"][1]["content"] = f"{text_response}\n{audio_control}"
        else:
            response["choices"][0]["text"] = f"{text_response}\n{audio_control}"
    return response, audio_response


@app.post(
    "/v1/chat/completions",
    tags=["Completions"],
    dependencies=[Depends(verify_api_key)],
)
async def chat_completions(
    c: ChatCompletions, request: Request, user=Depends(verify_api_key)
):
    data = await request.json()
    response, audio_response = await get_response(data=data, completion_type="chat")
    if audio_response:
        if audio_response.startswith("http"):
            return response
    if not c.stream:
        return response
    else:
        if audio_response:
            return StreamingResponse(
                content=audio_response,
                media_type="audio/wav",
            )
        return StreamingResponse(
            streaming_generation(data=response["messages"][1]["content"]),
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
async def completions(c: Completions, request: Request, user=Depends(verify_api_key)):
    response, audio_response = await get_response(
        data=await request.json(), completion_type="completion"
    )
    if audio_response:
        if audio_response.startswith("http"):
            return response
    if not c.stream:
        return response
    else:
        if audio_response:
            return StreamingResponse(
                content=audio_response,
                media_type="audio/wav",
            )
        return StreamingResponse(
            streaming_generation(data=response["choices"][0]["text"]),
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
