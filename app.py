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
import time
from Globals import getenv

DEFAULT_MODEL = getenv("DEFAULT_MODEL")
WHISPER_MODEL = getenv("WHISPER_MODEL")
logging.basicConfig(
    level=getenv("LOG_LEVEL"),
    format=getenv("LOG_FORMAT"),
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
NGROK_TOKEN = getenv("NGROK_TOKEN")
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
    encryption_key = getenv("EZLOCALAI_API_KEY")
    if encryption_key:
        if encryption_key == "none":
            return "USER"
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

    if getenv("DEFAULT_MODEL") or getenv("VISION_MODEL"):
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

            # Create a generator that yields properly formatted SSE chunks
            def generate_stream():
                try:
                    import json

                    logging.info(
                        f"[STREAMING] Starting stream with response type: {type(response)}"
                    )
                    chunk_count = 0
                    for chunk in response:
                        chunk_count += 1
                        logging.debug(f"[STREAMING] Chunk {chunk_count}: {chunk}")
                        # Yield the complete chunk in SSE format
                        yield f"data: {json.dumps(chunk)}\n\n"
                    # Send the final [DONE] message
                    yield "data: [DONE]\n\n"
                    logging.info(
                        f"[STREAMING] Stream completed with {chunk_count} chunks"
                    )
                except Exception as e:
                    import traceback

                    logging.error(f"[STREAMING] Streaming error: {e}")
                    logging.error(
                        f"[STREAMING] Full traceback: {traceback.format_exc()}"
                    )
                    yield f'data: {{"error": "Streaming failed: {str(e)}"}}\n\n'

            return StreamingResponse(
                content=generate_stream(),
                media_type="text/event-stream",
            )
    else:
        raise HTTPException(status_code=404, detail="Chat completions are disabled.")


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
    if getenv("DEFAULT_MODEL") or getenv("VISION_MODEL"):
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
    else:
        raise HTTPException(status_code=404, detail="Completions are disabled.")


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
    if getenv("EMBEDDING_ENABLED").lower() == "true":
        return pipe.embedder.get_embeddings(input=embedding.input)
    else:
        raise HTTPException(status_code=404, detail="Embeddings are disabled.")


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
    if getenv("STT_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Speech to text is disabled.")
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
    if getenv("STT_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Speech to text is disabled.")
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
    if getenv("TTS_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Text to speech is disabled.")
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
    if getenv("TTS_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Text to speech is disabled.")
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
    if getenv("TTS_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Text to speech is disabled.")
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


# Image Generation endpoint
# https://platform.openai.com/docs/api-reference/images


class ImageCreation(BaseModel):
    prompt: str
    model: Optional[str] = "stabilityai/sdxl-turbo"
    n: Optional[int] = 1
    size: Optional[str] = "512x512"
    quality: Optional[str] = "hd"
    response_format: Optional[str] = "url"
    style: Optional[str] = "natural"


@app.post(
    "/v1/images/generations",
    tags=["Images"],
    dependencies=[Depends(verify_api_key)],
)
async def generate_image(
    image_creation: ImageCreation,
    user: str = Depends(verify_api_key),
):
    if getenv("SD_MODEL") == "":
        return {
            "created": int(time.time()),
            "data": [{"url": "https://demofree.sirv.com/nope-not-here.jpg"}],
        }
    images = []
    if int(image_creation.n) > 1:
        for i in range(image_creation.n):
            image = await pipe.generate_image(
                prompt=image_creation.prompt,
                response_format=image_creation.response_format,
                size=image_creation.size,
            )
            if image_creation.response_format == "url":
                images.append({"url": image})
            else:
                images.append({"b64_json": image})
        return {
            "created": int(time.time()),
            "data": images,
        }
    image = await pipe.generate_image(
        prompt=image_creation.prompt,
        response_format=image_creation.response_format,
        size=image_creation.size,
    )
    if image_creation.response_format == "url":
        return {
            "created": int(time.time()),
            "data": [{"url": image}],
        }
    return {
        "created": int(time.time()),
        "data": [{"b64_json": image}],
    }
