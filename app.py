import warnings

# Suppress SyntaxWarnings from third-party packages (e.g., pydub regex patterns)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header,
    Request,
    Form,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import struct
from Pipes import Pipes
from RequestQueue import RequestQueue
import base64
import os
import logging
import re
import uuid
import time
import asyncio
from Globals import getenv

DEFAULT_MODEL = getenv("DEFAULT_MODEL")
WHISPER_MODEL = getenv("WHISPER_MODEL")
logging.basicConfig(
    level=getenv("LOG_LEVEL"),
    format=getenv("LOG_FORMAT"),
)


# SRT/VTT timestamp formatting helpers
def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to VTT timestamp format: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def segments_to_srt(segments: list) -> str:
    """Convert segments to SRT format."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(seg["text"])
        srt_lines.append("")
    return "\n".join(srt_lines)


def segments_to_vtt(segments: list) -> str:
    """Convert segments to WebVTT format."""
    vtt_lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        vtt_lines.append(f"{start} --> {end}")
        vtt_lines.append(seg["text"])
        vtt_lines.append("")
    return "\n".join(vtt_lines)


# Initialize request queue
MAX_CONCURRENT_REQUESTS = int(getenv("MAX_CONCURRENT_REQUESTS", "1"))
MAX_QUEUE_SIZE = int(getenv("MAX_QUEUE_SIZE", "100"))
request_queue = RequestQueue(
    max_concurrent_requests=MAX_CONCURRENT_REQUESTS, max_queue_size=MAX_QUEUE_SIZE
)

pipe = Pipes()

app = FastAPI(title="ezlocalai Server", docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Queue management
@app.on_event("startup")
async def startup_event():
    await request_queue.start()

    # Initialize wake word manager in voice server mode
    from Pipes import is_voice_server_mode

    if is_voice_server_mode():
        logging.info(
            "[WakeWord] Voice server mode detected, initializing wake word manager"
        )
        try:
            from ezlocalai.WakeWord import WakeWordManager, set_wakeword_manager
            from pathlib import Path

            # In voice server mode, we can optionally share Chatterbox TTS with wake word training
            # This allows voice cloning for wake word samples
            chatterbox_model = None
            try:
                # Pre-load TTS to get the Chatterbox model for wake word training
                tts = pipe._get_tts()
                if hasattr(tts, "model"):
                    chatterbox_model = tts.model
                    logging.info(
                        "[WakeWord] Chatterbox TTS model available for wake word training"
                    )
            except Exception as e:
                logging.warning(f"[WakeWord] Could not get Chatterbox model: {e}")

            manager = WakeWordManager(
                chatterbox_model=chatterbox_model,
                voices_dir=Path(os.getcwd()) / "voices",
            )
            set_wakeword_manager(manager)
            logging.info(
                f"[WakeWord] Wake word manager initialized. "
                f"Models dir: {manager.models_dir}, "
                f"Existing models: {len(manager.list_available_models())}"
            )
        except Exception as e:
            logging.error(f"[WakeWord] Failed to initialize wake word manager: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    await request_queue.stop()


# Async wrapper for pipe.get_response
async def process_request_async(data: Dict, completion_type: str):
    """Async wrapper for pipe.get_response to handle it in the queue."""
    return await pipe.get_response(data, completion_type)


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
    return pipe.get_models()


@app.get(
    "/v1/resources",
    tags=["System"],
    dependencies=[Depends(verify_api_key)],
)
async def get_resources(user=Depends(verify_api_key)):
    """Get current resource status including VRAM usage, loaded models, and fallback info."""
    from Pipes import (
        get_resource_manager,
        get_fallback_client,
        should_use_ezlocalai_fallback,
    )

    resource_mgr = get_resource_manager()
    status = resource_mgr.get_status()

    # Add fallback information
    fallback_client = get_fallback_client()
    should_fallback, fallback_reason = should_use_ezlocalai_fallback()

    # Get combined memory info
    try:
        import psutil

        free_ram = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        free_ram = 0.0

    free_vram = resource_mgr.get_total_free_vram()

    status["fallback"] = {
        "configured": fallback_client.is_configured,
        "url": fallback_client.base_url if fallback_client.is_configured else None,
        "should_use_fallback": should_fallback,
        "reason": fallback_reason,
        "free_vram_gb": free_vram,
        "free_ram_gb": free_ram,
        "free_combined_gb": free_vram + free_ram,
        "memory_threshold_gb": float(getenv("FALLBACK_MEMORY_THRESHOLD", "8.0")),
    }

    return status


@app.get(
    "/v1/fallback/status",
    tags=["System"],
    dependencies=[Depends(verify_api_key)],
)
async def get_fallback_status(user=Depends(verify_api_key)):
    """Check the status and availability of the fallback ezlocalai server."""
    from Pipes import get_fallback_client, should_use_ezlocalai_fallback

    fallback_client = get_fallback_client()

    if not fallback_client.is_configured:
        return {
            "configured": False,
            "available": False,
            "reason": "No fallback server configured",
            "should_use": False,
        }

    available, avail_reason = await fallback_client.check_availability()
    should_fallback, fallback_reason = should_use_ezlocalai_fallback()

    # Try to get remote models if available
    remote_models = []
    if available:
        try:
            models_response = await fallback_client.get_models()
            remote_models = [
                m.get("id", m.get("name", "unknown"))
                for m in models_response.get("data", [])
            ]
        except:
            pass

    return {
        "configured": True,
        "url": fallback_client.base_url,
        "available": available,
        "availability_reason": avail_reason,
        "should_use": should_fallback,
        "should_use_reason": fallback_reason,
        "remote_models": remote_models,
    }


@app.get(
    "/v1/queue",
    tags=["System"],
    dependencies=[Depends(verify_api_key)],
)
async def get_queue_status(user=Depends(verify_api_key)):
    """Get current request queue status."""
    return request_queue.get_queue_status()


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

        # Add request timeout (configurable via environment variable)
        request_timeout = float(getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes default

        try:
            # Enqueue the request
            request_id = await request_queue.enqueue_request(
                data=data, completion_type="chat", processor_func=process_request_async
            )

            # Wait for the result
            response, audio_response = await request_queue.wait_for_result(
                request_id, timeout=request_timeout
            )

        except HTTPException:
            # Re-raise HTTP exceptions (queue full, etc.)
            raise
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Request timed out after {request_timeout} seconds",
            )
        except Exception as e:
            logging.error(f"[Chat Completions] Unexpected error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
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

                    chunk_count = 0
                    logging.info(
                        f"[STREAMING] Starting to iterate over response generator"
                    )
                    for chunk in response:
                        chunk_count += 1
                        if chunk_count <= 3 or chunk_count % 50 == 0:
                            logging.info(f"[STREAMING] Yielding chunk {chunk_count}")
                        # Yield the complete chunk in SSE format
                        yield f"data: {json.dumps(chunk)}\n\n"
                    logging.info(f"[STREAMING] Finished streaming {chunk_count} chunks")
                    # Send the final [DONE] message
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    import traceback

                    logging.error(f"[STREAMING] Streaming error: {e}")
                    logging.error(
                        f"[STREAMING] Full traceback: {traceback.format_exc()}"
                    )
                    # Don't expose exception details to client
                    yield f'data: {{"error": "Streaming failed"}}\n\n'

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
        data = await request.json()

        # Add request timeout (configurable via environment variable)
        request_timeout = float(getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes default

        try:
            # Enqueue the request
            request_id = await request_queue.enqueue_request(
                data=data,
                completion_type="completion",
                processor_func=process_request_async,
            )

            # Wait for the result
            response, audio_response = await request_queue.wait_for_result(
                request_id, timeout=request_timeout
            )

        except HTTPException:
            # Re-raise HTTP exceptions (queue full, etc.)
            raise
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"Request timed out after {request_timeout} seconds",
            )
        except Exception as e:
            logging.error(f"[Completions] Unexpected error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
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
    from Pipes import should_use_ezlocalai_fallback, get_fallback_client

    # Check if we should use fallback
    should_fallback, reason = should_use_ezlocalai_fallback()
    if should_fallback:
        fallback_client = get_fallback_client()
        if fallback_client.is_configured:
            available, _ = await fallback_client.check_availability()
            if available:
                logging.info(f"[Embeddings] Using fallback: {reason}")
                try:
                    return await fallback_client.forward_embeddings(
                        {
                            "input": embedding.input,
                            "model": embedding.model,
                        }
                    )
                except Exception as e:
                    logging.warning(f"[Embeddings] Fallback failed: {e}, using local")

    # Use local embeddings
    embedder = pipe._get_embedder()
    result = embedder.get_embeddings(input=embedding.input)
    pipe._destroy_embedder()
    return result


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

    from Pipes import get_voice_server_client, is_voice_server_mode

    # Read file content first (before any fallback attempts)
    file_content = await file.read()

    # Try voice server first if configured (not in voice server mode)
    if not is_voice_server_mode():
        voice_client = get_voice_server_client()
        if voice_client.is_configured:
            available, reason = await voice_client.check_availability()
            if available:
                logging.info(
                    f"[STT] Forwarding to voice server: {voice_client.base_url}"
                )
                response = await voice_client.forward_transcription(
                    file_content=file_content,
                    content_type=file.content_type,
                    model=model,
                    language=language,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature,
                )
                if response:
                    # Return the response as-is from voice server
                    if response_format == "text":
                        text_content = response.get("text", str(response))
                        return Response(content=text_content, media_type="text/plain")
                    return response
                logging.warning("[STT] Voice server failed, falling back to local")

    from Pipes import should_use_ezlocalai_fallback, get_fallback_client

    # Check if we should use fallback (general fallback server)
    should_fallback, reason = should_use_ezlocalai_fallback()
    if should_fallback:
        fallback_client = get_fallback_client()
        if fallback_client.is_configured:
            available, _ = await fallback_client.check_availability()
            if available:
                logging.info(f"[STT] Using fallback: {reason}")
                try:
                    response = await fallback_client.forward_transcription(
                        file_content=file_content,
                        content_type=file.content_type,
                        model=model,
                        language=language,
                        prompt=prompt,
                        response_format=response_format,
                        temperature=temperature,
                    )
                    # Return the response as-is from fallback
                    if response_format == "text":
                        text_content = response.get("text", str(response))
                        return Response(content=text_content, media_type="text/plain")
                    return response
                except Exception as e:
                    logging.warning(f"[STT] Fallback failed: {e}, using local")

    stt = pipe._get_stt()

    # Determine if we need segments based on response_format
    need_segments = response_format in ["verbose_json", "srt", "vtt"]

    response = await stt.transcribe_audio(
        base64_audio=base64.b64encode(file_content).decode("utf-8"),
        audio_format=file.content_type,
        language=language,
        prompt=prompt,
        temperature=temperature,
        return_segments=need_segments,
    )
    # In voice server mode, don't destroy STT - keep it loaded
    if not is_voice_server_mode():
        pipe._destroy_stt()

    # Format response based on response_format
    if response_format == "text":
        text_content = response["text"] if isinstance(response, dict) else response
        return Response(content=text_content, media_type="text/plain")

    elif response_format == "verbose_json":
        # Already in correct format with segments
        return response

    elif response_format == "srt":
        srt_content = segments_to_srt(response["segments"])
        return Response(content=srt_content, media_type="text/plain")

    elif response_format == "vtt":
        vtt_content = segments_to_vtt(response["segments"])
        return Response(content=vtt_content, media_type="text/vtt")

    else:  # json (default)
        if isinstance(response, dict):
            return {"text": response["text"]}
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

    from Pipes import is_voice_server_mode

    stt = pipe._get_stt()
    response = await stt.transcribe_audio(
        base64_audio=base64.b64encode(await file.read()).decode("utf-8"),
        audio_format=file.content_type,
        language=language,
        prompt=prompt,
        temperature=temperature,
        translate=True,
    )
    # In voice server mode, don't destroy STT - keep it loaded
    if not is_voice_server_mode():
        pipe._destroy_stt()
    return {"text": response}


# Helper function for batch translating subtitle segments
async def translate_segments_batch(
    segments: list, source_lang: str, target_lang: str, max_words: int = 4000
) -> list:
    """
    Translate multiple subtitle segments using the LLM in batches.
    Batches are sized by word count (~4000 words) to stay within context limits
    while minimizing the number of API calls.
    """
    translated_segments = []

    # Build batches based on word count, not fixed segment count
    batches = []
    current_batch = []
    current_word_count = 0

    for seg in segments:
        seg_words = len(seg["text"].split())
        if current_word_count + seg_words > max_words and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_word_count = 0
        current_batch.append(seg)
        current_word_count += seg_words

    if current_batch:
        batches.append(current_batch)

    logging.info(
        f"Translating {len(segments)} segments in {len(batches)} batches (~{max_words} words each)"
    )

    for batch_idx, batch in enumerate(batches):
        # Build numbered text block for batch translation
        text_block = "\n".join(
            [f"[{j+1}] {seg['text']}" for j, seg in enumerate(batch)]
        )

        messages = [
            {
                "role": "system",
                "content": f"""You are a professional subtitle translator. Translate the numbered subtitle lines from {source_lang} to {target_lang}.
Keep translations concise and natural for subtitles.
Maintain the exact same numbering format [N] for each line.
Return ONLY the translated lines with their numbers, nothing else.""",
            },
            {"role": "user", "content": text_block},
        ]

        # Call LLM for batch translation
        response, _ = await pipe.get_response(
            data={
                "model": getenv("DEFAULT_MODEL"),
                "messages": messages,
            },
            completion_type="chat",
        )

        # Parse response and map back to segments
        translated_text = ""
        if isinstance(response, dict) and "choices" in response:
            translated_text = response["choices"][0]["message"]["content"].strip()

        # Parse the numbered responses
        translations = {}
        for line in translated_text.split("\n"):
            match = re.match(r"\[(\d+)\]\s*(.+)", line.strip())
            if match:
                num = int(match.group(1))
                text = match.group(2).strip()
                translations[num] = text

        # Map translations back to segments
        for j, seg in enumerate(batch):
            translated_segments.append(
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": translations.get(
                        j + 1, seg["text"]
                    ),  # Fallback to original
                }
            )

        logging.info(
            f"Translated batch {batch_idx + 1}/{len(batches)} ({len(batch)} segments, ~{sum(len(s['text'].split()) for s in batch)} words)"
        )

    return translated_segments


@app.post(
    "/v1/audio/subtitles",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def generate_subtitles(
    file: UploadFile = File(...),
    model: str = Form(WHISPER_MODEL),
    source_language: Optional[str] = Form(None),
    target_languages: str = Form("en"),
    format: str = Form("vtt"),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    beam_size: int = Form(5),
    fast_mode: bool = Form(False),
    user: str = Depends(verify_api_key),
):
    """
    Generate subtitles in multiple languages from audio.

    - Transcribes audio to source language (or auto-detects)
    - Translates segments to each target language while preserving timing
    - Returns subtitles in requested format (srt, vtt, json)

    Parameters:
    - beam_size: Beam size for decoding (1=fastest, 5=default, higher=more accurate)
    - fast_mode: If True, uses beam_size=1 and disables context conditioning for ~2-3x speed

    Example: target_languages="en,ru" returns both English and Russian subtitles
    with synchronized timestamps.
    """
    if getenv("STT_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Speech to text is disabled.")

    from Pipes import is_voice_server_mode

    languages = [lang.strip() for lang in target_languages.split(",")]

    # Apply fast mode settings
    if fast_mode:
        beam_size = 1
        condition_on_previous = False
    else:
        condition_on_previous = True

    # Step 1: Transcribe audio to get segments with timing
    stt = pipe._get_stt()
    audio_bytes = await file.read()
    transcription = await stt.transcribe_audio(
        base64_audio=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_format=file.content_type,
        language=source_language,
        prompt=prompt,
        temperature=temperature,
        return_segments=True,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous,
    )
    # In voice server mode, don't destroy STT - keep it loaded
    if not is_voice_server_mode():
        pipe._destroy_stt()

    detected_language = transcription.get("language") or source_language or "en"
    source_segments = transcription["segments"]

    logging.info(
        f"[Subtitles] Detected language: {detected_language}, "
        f"Target languages: {languages}, "
        f"Segments: {len(source_segments)}"
    )

    result = {"source_language": detected_language, "subtitles": {}}

    # Step 2: Generate subtitles for each target language IN PARALLEL
    async def process_language(target_lang: str):
        logging.info(
            f"[Subtitles] Processing language: {target_lang}, "
            f"needs_translation: {target_lang != detected_language}"
        )
        if target_lang == detected_language:
            # No translation needed - use original segments
            return target_lang, source_segments
        else:
            # Batch translate segments while preserving timing
            # Uses ~4000 word batches to minimize API calls
            translated = await translate_segments_batch(
                segments=source_segments,
                source_lang=detected_language,
                target_lang=target_lang,
            )
            return target_lang, translated

    # Run all language translations in parallel
    translation_tasks = [process_language(lang) for lang in languages]
    translations = await asyncio.gather(*translation_tasks)

    # Format outputs
    for target_lang, translated_segments in translations:
        if format == "json":
            result["subtitles"][target_lang] = translated_segments
        elif format == "srt":
            result["subtitles"][target_lang] = segments_to_srt(translated_segments)
        elif format == "vtt":
            result["subtitles"][target_lang] = segments_to_vtt(translated_segments)

    return result


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

    from Pipes import get_voice_server_client, is_voice_server_mode

    # Try voice server first if configured (not in voice server mode)
    if not is_voice_server_mode():
        voice_client = get_voice_server_client()
        if voice_client.is_configured:
            available, reason = await voice_client.check_availability()
            if available:
                logging.info(
                    f"[TTS] Forwarding to voice server: {voice_client.base_url}"
                )
                audio_bytes = await voice_client.forward_tts(
                    text=tts.input,
                    voice=tts.voice,
                    language=tts.language,
                )
                if audio_bytes:
                    return Response(content=audio_bytes, media_type="audio/wav")
                logging.warning("[TTS] Voice server failed, falling back to local")

    from Pipes import should_use_ezlocalai_fallback, get_fallback_client

    # Check if we should use fallback (general fallback server)
    should_fallback, reason = should_use_ezlocalai_fallback()
    if should_fallback:
        fallback_client = get_fallback_client()
        if fallback_client.is_configured:
            available, _ = await fallback_client.check_availability()
            if available:
                logging.info(f"[TTS] Using fallback: {reason}")
                try:
                    audio_bytes = await fallback_client.forward_tts(
                        text=tts.input,
                        voice=tts.voice,
                        language=tts.language,
                    )
                    return Response(content=audio_bytes, media_type="audio/wav")
                except Exception as e:
                    logging.warning(f"[TTS] Fallback failed: {e}, using local")

    tts_model = pipe._get_tts()
    audio_b64 = await tts_model.generate(
        text=tts.input, voice=tts.voice, language=tts.language
    )
    # In voice server mode, don't destroy TTS - keep it loaded
    if not is_voice_server_mode():
        pipe._destroy_tts()
    # OpenAI SDK expects raw binary audio, not base64 JSON
    audio_bytes = base64.b64decode(audio_b64)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post(
    "/v1/audio/speech/stream",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def text_to_speech_stream(tts: TextToSpeech, user=Depends(verify_api_key)):
    """
    Stream TTS audio as it's generated, chunk by chunk.

    This enables real-time playback without waiting for the entire audio
    to be generated. Dramatically reduces time-to-first-word for long text.

    Response format (binary stream):
    - Header (8 bytes): sample_rate (uint32), bits (uint16), channels (uint16)
    - Chunks: chunk_size (uint32) + raw PCM data
    - End marker: chunk_size = 0

    Audio format: 24kHz, 16-bit, mono PCM
    """
    if getenv("TTS_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Text to speech is disabled.")

    from Pipes import is_voice_server_mode

    tts_model = pipe._get_tts()

    async def audio_stream_generator():
        try:
            async for chunk in tts_model.generate_stream(
                text=tts.input, voice=tts.voice, language=tts.language
            ):
                yield chunk
        finally:
            # In voice server mode, don't destroy TTS - keep it loaded
            if not is_voice_server_mode():
                pipe._destroy_tts()

    return StreamingResponse(
        audio_stream_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "pcm",
            "X-Sample-Rate": "24000",
            "X-Bits-Per-Sample": "16",
            "X-Channels": "1",
        },
    )


@app.websocket("/v1/audio/speech/ws")
async def tts_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional TTS streaming.

    This allows clients to:
    1. Send text incrementally as it's generated
    2. Receive audio chunks in real-time as TTS processes them

    Protocol:
    - Client sends JSON: {"text": "...", "voice": "default", "language": "en", "flush": false}
    - When "flush": true, generates TTS for accumulated text and streams audio
    - Server sends binary audio chunks (same format as /v1/audio/speech/stream)
    - Client can send {"done": true} to close gracefully

    Audio format: 24kHz, 16-bit, mono PCM
    Binary response format:
    - First message: 8-byte header (sample_rate u32, bits u16, channels u16)
    - Subsequent messages: raw PCM audio chunks
    - Final message: empty bytes to signal end
    """
    await websocket.accept()

    if getenv("TTS_ENABLED").lower() == "false":
        await websocket.close(code=1008, reason="TTS disabled")
        return

    from Pipes import is_voice_server_mode

    text_buffer = ""
    voice = "default"
    language = "en"
    header_sent = False

    try:
        tts_model = pipe._get_tts()

        while True:
            try:
                # Receive text from client
                data = await websocket.receive_json()

                if data.get("done"):
                    # Client signals done - flush any remaining text
                    if text_buffer.strip():
                        async for chunk in tts_model.generate_stream(
                            text=text_buffer.strip(), voice=voice, language=language
                        ):
                            if not header_sent:
                                # Send header first (8 bytes)
                                await websocket.send_bytes(chunk[:8])
                                header_sent = True
                                if len(chunk) > 8:
                                    await websocket.send_bytes(chunk[8:])
                            else:
                                await websocket.send_bytes(chunk)
                    # Send empty bytes to signal end
                    await websocket.send_bytes(b"")
                    break

                # Accumulate text
                if "text" in data:
                    text_buffer += data["text"]
                if "voice" in data:
                    voice = data["voice"]
                if "language" in data:
                    language = data["language"]

                # Check if we should flush (generate TTS for current buffer)
                if data.get("flush") and text_buffer.strip():
                    async for chunk in tts_model.generate_stream(
                        text=text_buffer.strip(), voice=voice, language=language
                    ):
                        if not header_sent:
                            # Send header first (8 bytes)
                            await websocket.send_bytes(chunk[:8])
                            header_sent = True
                            if len(chunk) > 8:
                                await websocket.send_bytes(chunk[8:])
                        else:
                            await websocket.send_bytes(chunk)
                    text_buffer = ""

            except WebSocketDisconnect:
                break

    except Exception as e:
        logging.error(f"TTS WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e)[:100])
        except:
            pass
    finally:
        if not is_voice_server_mode():
            pipe._destroy_tts()


@app.get(
    "/v1/audio/voices",
    tags=["Audio"],
    dependencies=[Depends(verify_api_key)],
)
async def get_voices(user=Depends(verify_api_key)):
    if getenv("TTS_ENABLED").lower() == "false":
        raise HTTPException(status_code=404, detail="Text to speech is disabled.")

    from Pipes import get_voice_server_client, is_voice_server_mode

    # Try voice server first if configured (not in voice server mode)
    if not is_voice_server_mode():
        voice_client = get_voice_server_client()
        if voice_client.is_configured:
            available, reason = await voice_client.check_availability()
            if available:
                voices = await voice_client.get_voices()
                if voices:
                    return voices

    tts_model = pipe._get_tts()
    voices = tts_model.voices
    # In voice server mode, don't destroy TTS - keep it loaded
    if not is_voice_server_mode():
        pipe._destroy_tts()
    return {"voices": voices}


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

    # Sanitize voice name to prevent path traversal
    import re

    # First sanitize the input to only allow safe characters
    if not voice or not isinstance(voice, str):
        voice = "default"
    # Remove any path separators and dangerous characters
    voice = re.sub(r'[/\\:*?"<>|]', "", voice)
    # Remove path traversal attempts
    voice = voice.replace("..", "")
    # Only allow alphanumeric, hyphen, underscore - strict allowlist
    voice = re.sub(r"[^a-zA-Z0-9_-]", "", voice)
    if not voice:
        voice = "default"
    voice = voice[:100]

    # Use basename to strip any remaining path components
    voice_name = os.path.basename(voice)
    if not voice_name:
        voice_name = "default"

    voices_dir = os.path.realpath(os.path.join(os.getcwd(), "voices"))
    os.makedirs(voices_dir, exist_ok=True)

    # Find unique filename
    base_name = voice_name
    i = 1
    while True:
        # Construct and normalize the path - CodeQL pattern from documentation
        fullpath = os.path.normpath(os.path.join(voices_dir, f"{voice_name}.wav"))
        # Verify with normalized version of path - exact CodeQL recommended pattern
        if not fullpath.startswith(voices_dir):
            raise HTTPException(status_code=400, detail="Invalid path")
        if not os.path.exists(fullpath):
            break
        voice_name = f"{base_name}-{i}"
        i += 1

    # fullpath is now verified safe - use it directly
    with open(fullpath, "wb") as audio_file:
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
    if getenv("IMG_MODEL") == "":
        # Check if fallback can handle image generation
        from Pipes import get_fallback_client

        fallback_client = get_fallback_client()
        if fallback_client.is_configured:
            available, _ = await fallback_client.check_availability()
            if available:
                logging.info("[IMG] No local model, using fallback")
                try:
                    return await fallback_client.forward_image_generation(
                        prompt=image_creation.prompt,
                        response_format=image_creation.response_format,
                        size=image_creation.size,
                    )
                except Exception as e:
                    logging.warning(f"[IMG] Fallback failed: {e}")
        return {
            "created": int(time.time()),
            "data": [{"url": "https://demofree.sirv.com/nope-not-here.jpg"}],
        }

    from Pipes import should_use_ezlocalai_fallback, get_fallback_client

    # Check if we should use fallback (IMG uses a lot of VRAM)
    should_fallback, reason = should_use_ezlocalai_fallback()
    if should_fallback:
        fallback_client = get_fallback_client()
        if fallback_client.is_configured:
            available, _ = await fallback_client.check_availability()
            if available:
                logging.info(f"[IMG] Using fallback: {reason}")
                try:
                    return await fallback_client.forward_image_generation(
                        prompt=image_creation.prompt,
                        response_format=image_creation.response_format,
                        size=image_creation.size,
                    )
                except Exception as e:
                    logging.warning(f"[IMG] Fallback failed: {e}, using local")

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


# Queue management endpoints
@app.get(
    "/v1/queue/status",
    tags=["Queue Management"],
    dependencies=[Depends(verify_api_key)],
)
async def get_queue_status(user=Depends(verify_api_key)):
    """Get current queue status and metrics."""
    return request_queue.get_queue_status()


@app.get(
    "/v1/queue/request/{request_id}",
    tags=["Queue Management"],
    dependencies=[Depends(verify_api_key)],
)
async def get_request_status(request_id: str, user=Depends(verify_api_key)):
    """Get status of a specific request."""
    status = request_queue.get_request_status(request_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    return status


# =============================================================================
# Wake Word Detection Endpoints
# =============================================================================
# Available when VOICE_SERVER=true or STT_ENABLED=true


class WakeWordTrainRequest(BaseModel):
    word: str
    sample_count: int = 500
    epochs: int = 50
    batch_size: int = 32


class WakeWordTrainResponse(BaseModel):
    job_id: str
    word: str
    status: str
    message: str
    estimated_minutes: Optional[int] = None
    check_status_url: str


class WakeWordJobStatusResponse(BaseModel):
    job_id: str
    word: str
    status: str
    progress: float
    current_stage: str
    error_message: Optional[str] = None
    estimated_completion: Optional[str] = None
    model_path: Optional[str] = None
    metrics: dict = {}
    created_at: str
    updated_at: str


class WakeWordModelInfo(BaseModel):
    word: str
    directory: str
    config: dict
    files: dict
    created_at: str


class WakeWordPredictRequest(BaseModel):
    audio_base64: str


class WakeWordPredictResponse(BaseModel):
    detected: bool
    confidence: float
    word: str


def is_wakeword_enabled() -> bool:
    """Check if wake word functionality is enabled."""
    from Pipes import is_voice_server_mode

    # Enable if in voice server mode OR if STT is enabled
    return is_voice_server_mode() or getenv("STT_ENABLED", "true").lower() == "true"


@app.get(
    "/v1/wakeword/models",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
)
async def list_wakeword_models(user=Depends(verify_api_key)):
    """List all available trained wake word models."""
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager

    manager = get_wakeword_manager()
    models = manager.list_available_models()
    return {"models": models, "total": len(models)}


@app.get(
    "/v1/wakeword/models/{word}",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
)
async def get_wakeword_model(
    word: str,
    format: str = "pytorch",
    user=Depends(verify_api_key),
):
    """
    Download a trained wake word model.

    Supported formats: pytorch, onnx, tflite
    """
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager
    from fastapi.responses import FileResponse

    manager = get_wakeword_manager()
    model_dir = manager.get_model_for_word(word)

    if model_dir:
        format_to_ext = {"pytorch": "pt", "onnx": "onnx", "tflite": "tflite"}
        ext = format_to_ext.get(format, "pt")
        model_file = model_dir / f"model.{ext}"

        if not model_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model format '{format}' not available for word '{word}'",
            )

        word_lower = word.lower().replace(" ", "_")
        return FileResponse(
            path=model_file,
            filename=f"{word_lower}_model.{ext}",
            media_type="application/octet-stream",
        )

    # Check if training is in progress
    job = manager.get_job_for_word(word)
    if job and job.status.value not in ["completed", "failed", "cancelled"]:
        return {
            "status": "training_in_progress",
            "job_id": job.job_id,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "message": f"Model for '{word}' is being trained. Check back soon.",
            "check_status_url": f"/v1/wakeword/jobs/{job.job_id}",
        }

    raise HTTPException(
        status_code=404,
        detail=f"No model found for word '{word}'. Use POST /v1/wakeword/train to create one.",
    )


@app.get(
    "/v1/wakeword/models/{word}/config",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
)
async def get_wakeword_model_config(word: str, user=Depends(verify_api_key)):
    """Get the configuration for a trained wake word model."""
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager
    import json

    manager = get_wakeword_manager()
    model_dir = manager.get_model_for_word(word)

    if not model_dir:
        raise HTTPException(status_code=404, detail=f"No model found for word '{word}'")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Model config not found")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


@app.delete(
    "/v1/wakeword/models/{word}",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
)
async def delete_wakeword_model(word: str, user=Depends(verify_api_key)):
    """Delete a trained wake word model."""
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager

    manager = get_wakeword_manager()

    if not manager.delete_model(word):
        raise HTTPException(status_code=404, detail=f"No model found for word '{word}'")

    return {"message": f"Model for '{word}' deleted successfully"}


@app.post(
    "/v1/wakeword/train",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
    response_model=WakeWordTrainResponse,
)
async def train_wakeword_model(
    request: WakeWordTrainRequest,
    user=Depends(verify_api_key),
):
    """
    Request training of a new wake word model.

    Training typically takes 10-20 minutes depending on sample_count and epochs.
    The model will use TTS engines (gTTS, Edge TTS, and optionally Chatterbox)
    to generate diverse training samples automatically.
    """
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager

    manager = get_wakeword_manager()
    word = request.word.strip()

    # Check if model already exists
    model_dir = manager.get_model_for_word(word)
    if model_dir:
        return WakeWordTrainResponse(
            job_id="",
            word=word,
            status="completed",
            message=f"Model for '{word}' already exists. Use GET /v1/wakeword/models/{word} to download.",
            check_status_url=f"/v1/wakeword/models/{word}",
        )

    # Check if training is already in progress
    existing_job = manager.get_job_for_word(word)
    if existing_job and existing_job.status.value not in [
        "completed",
        "failed",
        "cancelled",
    ]:
        return WakeWordTrainResponse(
            job_id=existing_job.job_id,
            word=word,
            status=existing_job.status.value,
            message=f"Training already in progress for '{word}'",
            estimated_minutes=15,
            check_status_url=f"/v1/wakeword/jobs/{existing_job.job_id}",
        )

    # Start new training job
    try:
        job = await manager.create_job(
            word=word,
            sample_count=request.sample_count,
            epochs=request.epochs,
            batch_size=request.batch_size,
        )

        return WakeWordTrainResponse(
            job_id=job.job_id,
            word=word,
            status=job.status.value,
            message=f"Training started for '{word}'. This typically takes 10-20 minutes.",
            estimated_minutes=15,
            check_status_url=f"/v1/wakeword/jobs/{job.job_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/v1/wakeword/jobs",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
)
async def list_wakeword_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    user=Depends(verify_api_key),
):
    """List wake word training jobs."""
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager, JobStatus

    manager = get_wakeword_manager()

    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {[s.value for s in JobStatus]}",
            )

    jobs = manager.list_jobs(status=status_filter, limit=limit)

    return [
        WakeWordJobStatusResponse(
            job_id=j.job_id,
            word=j.word,
            status=j.status.value,
            progress=j.progress,
            current_stage=j.current_stage,
            error_message=j.error_message,
            estimated_completion=(
                j.estimated_completion.isoformat() if j.estimated_completion else None
            ),
            model_path=j.model_path,
            metrics=j.metrics,
            created_at=j.created_at.isoformat(),
            updated_at=j.updated_at.isoformat(),
        )
        for j in jobs
    ]


@app.get(
    "/v1/wakeword/jobs/{job_id}",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
    response_model=WakeWordJobStatusResponse,
)
async def get_wakeword_job(job_id: str, user=Depends(verify_api_key)):
    """Get status of a specific wake word training job."""
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager

    manager = get_wakeword_manager()
    job = manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return WakeWordJobStatusResponse(
        job_id=job.job_id,
        word=job.word,
        status=job.status.value,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
        estimated_completion=(
            job.estimated_completion.isoformat() if job.estimated_completion else None
        ),
        model_path=job.model_path,
        metrics=job.metrics,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


@app.delete(
    "/v1/wakeword/jobs/{job_id}",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
)
async def cancel_wakeword_job(job_id: str, user=Depends(verify_api_key)):
    """Cancel a running wake word training job."""
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager

    manager = get_wakeword_manager()
    success = await manager.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Could not cancel job (may already be completed or cancelled)",
        )

    return {"message": "Job cancelled successfully"}


@app.post(
    "/v1/wakeword/predict/{word}",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
    response_model=WakeWordPredictResponse,
)
async def predict_wakeword(
    word: str,
    request: WakeWordPredictRequest,
    user=Depends(verify_api_key),
):
    """
    Predict if audio contains the wake word.

    Requires a trained model for the word.
    Send audio as base64-encoded WAV data.
    """
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager

    manager = get_wakeword_manager()
    model_dir = manager.get_model_for_word(word)

    if not model_dir:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for word '{word}'. Train one first with POST /v1/wakeword/train",
        )

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        detected, confidence = manager.predict(word, audio_bytes)

        return WakeWordPredictResponse(
            detected=detected,
            confidence=confidence,
            word=word,
        )
    except Exception as e:
        logging.error(f"Wake word prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/v1/wakeword/predict/{word}/file",
    tags=["Wake Word"],
    dependencies=[Depends(verify_api_key)],
    response_model=WakeWordPredictResponse,
)
async def predict_wakeword_file(
    word: str,
    file: UploadFile = File(...),
    user=Depends(verify_api_key),
):
    """
    Predict if uploaded audio file contains the wake word.

    Requires a trained model for the word.
    Accepts audio files (WAV, MP3, etc.).
    """
    if not is_wakeword_enabled():
        raise HTTPException(
            status_code=404,
            detail="Wake word detection is disabled. Set VOICE_SERVER=true to enable.",
        )

    from ezlocalai.WakeWord import get_wakeword_manager

    manager = get_wakeword_manager()
    model_dir = manager.get_model_for_word(word)

    if not model_dir:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for word '{word}'. Train one first with POST /v1/wakeword/train",
        )

    try:
        audio_bytes = await file.read()
        detected, confidence = manager.predict(word, audio_bytes)

        return WakeWordPredictResponse(
            detected=detected,
            confidence=confidence,
            word=word,
        )
    except Exception as e:
        logging.error(f"Wake word prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
