from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header,
    Request,
    Form,
    UploadFile,
    File,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from Pipes import Pipes
import base64
import os
import logging
import uuid
from dotenv import load_dotenv

load_dotenv()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "TheBloke/phi-2-dpo-GGUF")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
pipe = Pipes()
logging.info(f"[ezlocalai] Server is ready.")

app = FastAPI(title="ezlocalai Server", docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")
if NGROK_TOKEN:
    from pyngrok import ngrok

    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(8091)
    logging.info(f"[ngrok] Public Tunnel: {public_url.public_url}")
    ngrok_url = public_url.public_url

    def get_ngrok_url():
        global ngrok_url
        return ngrok_url

else:

    def get_ngrok_url():
        return "http://localhost:8091"


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
    return pipe.llm.models()


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


@app.post(
    "/v1/chat/completions",
    tags=["Completions"],
    dependencies=[Depends(verify_api_key)],
)
async def chat_completions(
    c: ChatCompletions, request: Request, user=Depends(verify_api_key)
):
    data = await request.json()
    response, audio_response = await pipe.get_response(
        data=data, completion_type="chat"
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
            content=response["messages"][1]["content"],
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
    response, audio_response = await pipe.get_response(
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
            content=response["choices"][0]["text"],
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
    "/v1/embeddings",
    tags=["Embeddings"],
    dependencies=[Depends(verify_api_key)],
)
async def embedding(embedding: EmbeddingModel, user=Depends(verify_api_key)):
    return pipe.embedder.get_embeddings(input=embedding.input)


# Audio Transcription endpoint
# https://platform.openai.com/docs/api-reference/audio/createTranscription
@app.post(
    "/v1/audio/transcriptions",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def speech_to_text(
    file: UploadFile = File(...),
    model: str = Form(WHISPER_MODEL),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamp_granularities: Optional[List[str]] = Form(["segment"]),
    user: str = Depends(verify_api_key),
):
    response = await pipe.stt.transcribe_audio(
        base64_audio=base64.b64encode(await file.read()).decode("utf-8"),
        audio_format=file.content_type,
        language=language,
        prompt=prompt,
        temperature=temperature,
    )
    return {"text": response}


# Audio Translation endpoint
# https://platform.openai.com/docs/api-reference/audio/createTranslation


@app.post(
    "/v1/audio/translations",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def audio_translations(
    file: UploadFile = File(...),
    model: str = Form(WHISPER_MODEL),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    user=Depends(verify_api_key),
):
    response = await pipe.stt.transcribe_audio(
        base64_audio=base64.b64encode(await file.read()).decode("utf-8"),
        audio_format=file.content_type,
        language=language,
        prompt=prompt,
        temperature=temperature,
        translate=True,
    )
    return {"text": response}


class TextToSpeech(BaseModel):
    input: str
    model: Optional[str] = "tts-1"
    voice: Optional[str] = "default"
    language: Optional[str] = "en"
    user: Optional[str] = None


@app.post(
    "/v1/audio/speech",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def text_to_speech(tts: TextToSpeech, user=Depends(verify_api_key)):
    if tts.input.startswith("data:"):
        if "pdf" in tts.input:
            audio = await pipe.pdf_to_audio(
                title=tts.user if tts.user else f"{uuid.uuid4().hex}",
                voice=tts.voice,
                pdf=tts.input,
                chunk_size=200,
            )
            return audio
        if "audio/" in tts.input:
            audio = await pipe.audio_to_audio(
                voice=tts.voice,
                audio=tts.input,
            )
            return audio
    audio = await pipe.ctts.generate(
        text=tts.input, voice=tts.voice, language=tts.language
    )
    return audio


@app.get(
    "/v1/audio/voices",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def get_voices(user=Depends(verify_api_key)):
    return {"voices": pipe.ctts.voices}


@app.post(
    "/v1/audio/voices",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def upload_voice(
    voice: str = Form("default"),
    file: UploadFile = File(...),
    user=Depends(verify_api_key),
):
    voice_name = voice
    file_path = os.path.join(os.getcwd(), "voices", f"{voice}.wav")
    if os.path.exists(file_path):
        i = 1
        while os.path.exists(file_path):
            file_path = os.path.join(os.getcwd(), "voices", f"{voice}-{i}.wav")
            voice_name = f"{voice}-{i}"
            i += 1
    with open(file_path, "wb") as audio_file:
        audio_file.write(await file.read())
    return {"detail": f"Voice {voice_name} has been uploaded."}
