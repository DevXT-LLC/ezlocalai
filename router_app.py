"""Router-mode FastAPI app for ezlocalai.

This app does **no** local inference. It accepts the same OpenAI-compatible
endpoints as a normal ezlocalai server and proxies each request to the best
available worker that has registered itself via the router protocol.

Run with::

    ROUTER_MODE=true python start.py

(or set ``ROUTER_MODE=true`` in your env / docker-compose service.)

Endpoints exposed:

Router protocol
    * ``POST /v1/router/register``     — worker registration
    * ``POST /v1/router/heartbeat``    — periodic state update
    * ``POST /v1/router/deregister``   — graceful shutdown
    * ``GET  /v1/router/workers``      — admin: list registered workers
    * ``GET  /v1/router/health``       — router health (always reports queue ok)

OpenAI-compatible (proxied)
    * ``GET  /v1/models``
    * ``POST /v1/chat/completions``    (streaming + non-streaming)
    * ``POST /v1/completions``         (streaming + non-streaming)
    * ``POST /v1/embeddings``
    * ``POST /v1/audio/speech``
    * ``POST /v1/audio/speech/stream``
    * ``POST /v1/audio/transcriptions``
    * ``GET  /v1/audio/voices``
    * ``POST /v1/images/generations``
    * ``POST /v1/images/edits``
    * ``POST /v1/videos/generations``
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from Globals import getenv
from Router import (
    Router,
    WorkerInfo,
    WorkerRegistry,
    best_gpu_tier,
    get_registry,
    get_router,
)


logging.basicConfig(
    level=getenv("LOG_LEVEL"),
    format=getenv("LOG_FORMAT"),
    force=True,
)


app = FastAPI(title="ezlocalai Router", docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _expected_register_key() -> str:
    """Shared secret that workers must present to register/heartbeat."""
    return (getenv("ROUTER_REGISTER_KEY") or "").strip() or (
        getenv("EZLOCALAI_API_KEY") or ""
    ).strip()


def _expected_client_key() -> str:
    """API key that inference clients must present (same as a normal ezlocalai)."""
    return (getenv("EZLOCALAI_API_KEY") or "").strip()


def _check_bearer(authorization: Optional[str], expected: str) -> None:
    if not expected or expected == "none":
        return  # auth disabled
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split(None, 1)
    token = parts[1].strip() if len(parts) == 2 else parts[0].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def verify_client(authorization: Optional[str] = Header(None)) -> str:
    _check_bearer(authorization, _expected_client_key())
    return "USER"


def verify_worker(authorization: Optional[str] = Header(None)) -> str:
    _check_bearer(authorization, _expected_register_key())
    return "WORKER"


# ---------------------------------------------------------------------------
# Background pruner
# ---------------------------------------------------------------------------


_pruner_task: Optional[asyncio.Task] = None


async def _pruner_loop():
    registry = get_registry()
    while True:
        try:
            removed = registry.prune()
            for wid in removed:
                logging.info(f"[Router] Pruned stale worker {wid}")
        except Exception as e:  # pragma: no cover
            logging.debug(f"[Router] pruner error: {e}")
        await asyncio.sleep(max(2.0, registry.ttl / 3))


@app.on_event("startup")
async def _startup():
    global _pruner_task
    if _pruner_task is None or _pruner_task.done():
        _pruner_task = asyncio.create_task(_pruner_loop())
    register_key = _expected_register_key()
    client_key = _expected_client_key()
    auth_warnings = []
    if not register_key or register_key == "none":
        auth_warnings.append(
            "ROUTER_REGISTER_KEY/EZLOCALAI_API_KEY is empty — ANY worker that "
            "can reach this router can join the pool. Set EZLOCALAI_API_KEY (or "
            "ROUTER_REGISTER_KEY) before exposing the router publicly."
        )
    if not client_key or client_key == "none":
        auth_warnings.append(
            "EZLOCALAI_API_KEY is empty — ANY client can submit inference "
            "requests to the router."
        )
    for msg in auth_warnings:
        logging.warning(f"[Router] OPEN POOL: {msg}")
    logging.info(f"[Router] ezlocalai router ready (worker TTL={get_registry().ttl}s)")


@app.on_event("shutdown")
async def _shutdown():
    global _pruner_task
    if _pruner_task and not _pruner_task.done():
        _pruner_task.cancel()
        try:
            await _pruner_task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Router protocol endpoints (workers ↔ router)
# ---------------------------------------------------------------------------


@app.post("/v1/router/register", tags=["Router"])
async def router_register(
    request: Request,
    payload: Dict[str, Any],
    _: str = Depends(verify_worker),
):
    if not payload.get("worker_id"):
        raise HTTPException(status_code=400, detail="Missing field: worker_id")

    # Resolve a URL the router can call back on. Order:
    # 1. Whatever the worker explicitly reported (if non-loopback).
    # 2. http://<source-IP>:<reported port> (router fills in the source IP).
    # 3. Reject — the worker is unreachable.
    declared_url = str(payload.get("url") or "").strip().rstrip("/")
    port = int(payload.get("port") or 8091)
    source_host = request.client.host if request.client else None

    def _is_loopback(u: str) -> bool:
        u = u.lower()
        return (
            "://localhost" in u
            or "://127.0.0.1" in u
            or "://0.0.0.0" in u
            or "://::1" in u
        )

    url = declared_url
    if not url or _is_loopback(url):
        if not source_host:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Worker did not provide a public 'url' and the router "
                    "could not determine the source IP."
                ),
            )
        url = f"http://{source_host}:{port}"

    gpus = list(payload.get("gpus") or [])
    info = WorkerInfo(
        worker_id=str(payload["worker_id"]),
        label=str(payload.get("label") or payload["worker_id"]),
        url=url,
        api_key=str(payload.get("api_key") or ""),
        capabilities=list(payload.get("capabilities") or []),
        models=list(payload.get("models") or []),
        free_vram_gb=float(payload.get("free_vram_gb", 0.0) or 0.0),
        total_vram_gb=float(payload.get("total_vram_gb", 0.0) or 0.0),
        free_ram_gb=float(payload.get("free_ram_gb", 0.0) or 0.0),
        queue_depth=int(payload.get("queue_depth", 0) or 0),
        queue_capacity=int(payload.get("queue_capacity", 1) or 1),
        in_flight=int(payload.get("in_flight", 0) or 0),
        gpus=gpus,
        best_tier=int(payload.get("best_tier") or best_gpu_tier(gpus)),
        model_context={
            str(k): int(v) for k, v in (payload.get("model_context") or {}).items()
        },
        extra={
            k: v
            for k, v in payload.items()
            if k
            not in {
                "worker_id",
                "label",
                "url",
                "port",
                "api_key",
                "capabilities",
                "models",
                "free_vram_gb",
                "total_vram_gb",
                "free_ram_gb",
                "queue_depth",
                "queue_capacity",
                "in_flight",
                "gpus",
                "best_tier",
                "model_context",
            }
        },
    )
    info = get_registry().register(info)
    gpu_summary = (
        ", ".join(
            f"{g.get('name', '?')} ({g.get('total_vram_gb', 0):.0f}GB)"
            for g in info.gpus
        )
        or "no GPU"
    )
    logging.info(
        f"[Router] Registered worker {info.label} ({info.worker_id}) "
        f"@ {info.url} caps={info.capabilities} tier={info.best_tier} "
        f"gpus=[{gpu_summary}]"
    )
    return {"ok": True, "worker": info.to_public()}


@app.post("/v1/router/heartbeat", tags=["Router"])
async def router_heartbeat(payload: Dict[str, Any], _: str = Depends(verify_worker)):
    worker_id = payload.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="Missing worker_id")
    worker = get_registry().heartbeat(str(worker_id), payload)
    if worker is None:
        # Tell caller to re-register
        return JSONResponse(
            status_code=410,
            content={
                "ok": False,
                "reason": "unknown_worker",
                "should_reregister": True,
            },
        )
    return {"ok": True, "worker": worker.to_public()}


@app.post("/v1/router/deregister", tags=["Router"])
async def router_deregister(payload: Dict[str, Any], _: str = Depends(verify_worker)):
    worker_id = payload.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="Missing worker_id")
    removed = get_registry().deregister(str(worker_id))
    return {"ok": removed}


@app.get("/v1/router/workers", tags=["Router"])
async def router_workers(_: str = Depends(verify_client)):
    workers = get_registry().list_workers(alive_only=False)
    return {
        "object": "list",
        "data": [w.to_public() for w in workers],
        "ttl_seconds": get_registry().ttl,
    }


@app.get("/v1/router/health", tags=["Router"])
async def router_health():
    workers = get_registry().list_workers(alive_only=True)
    return {
        "status": "healthy",
        "alive_workers": len(workers),
        "ttl_seconds": get_registry().ttl,
    }


@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Worker selection helpers
# ---------------------------------------------------------------------------


def _wait_timeout() -> float:
    return float(getenv("ROUTER_WAIT_TIMEOUT", "120"))


async def _pick(capability: str, model: Optional[str] = None) -> WorkerInfo:
    router = get_router()
    worker = await router.wait_for_worker(capability, model, timeout=_wait_timeout())
    if worker is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No worker available for capability={capability!r}"
                + (f" model={model!r}" if model else "")
            ),
        )
    return worker


def _worker_headers(worker: WorkerInfo) -> Dict[str, str]:
    h = {}
    if worker.api_key and worker.api_key != "none":
        h["Authorization"] = f"Bearer {worker.api_key}"
    return h


# ---------------------------------------------------------------------------
# Generic proxy helpers
# ---------------------------------------------------------------------------


async def _proxy_json(
    worker: WorkerInfo,
    path: str,
    payload: Dict[str, Any],
    *,
    stream: bool = False,
    timeout: Optional[float] = None,
):
    """Forward a JSON POST to a worker. Returns either a dict or a StreamingResponse."""
    url = f"{worker.url}{path}"
    headers = {"Content-Type": "application/json", **_worker_headers(worker)}
    request_timeout = aiohttp.ClientTimeout(
        total=timeout or float(getenv("REQUEST_TIMEOUT", "300"))
    )
    registry = get_registry()
    registry.increment_in_flight(worker.worker_id, 1)

    if not stream:
        try:
            async with aiohttp.ClientSession(timeout=request_timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    body = await resp.read()
                    media_type = resp.headers.get("Content-Type", "application/json")
                    return Response(
                        content=body, status_code=resp.status, media_type=media_type
                    )
        finally:
            registry.increment_in_flight(worker.worker_id, -1)

    async def gen():
        # Open one persistent session for the whole stream
        session = aiohttp.ClientSession(timeout=request_timeout)
        try:
            resp = await session.post(url, json=payload, headers=headers)
            try:
                async for chunk in resp.content.iter_any():
                    if chunk:
                        yield chunk
            finally:
                resp.release()
        except Exception as e:
            logging.warning(f"[Router] Stream from {worker.url}{path} failed: {e}")
        finally:
            await session.close()
            registry.increment_in_flight(worker.worker_id, -1)

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _proxy_get(worker: WorkerInfo, path: str) -> Response:
    url = f"{worker.url}{path}"
    headers = _worker_headers(worker)
    request_timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=request_timeout) as session:
        async with session.get(url, headers=headers) as resp:
            body = await resp.read()
            media_type = resp.headers.get("Content-Type", "application/json")
            return Response(
                content=body, status_code=resp.status, media_type=media_type
            )


async def _proxy_multipart(
    worker: WorkerInfo,
    path: str,
    *,
    files: Dict[str, tuple],
    fields: Dict[str, str],
    timeout: Optional[float] = None,
) -> Response:
    """Forward a multipart/form-data POST to a worker.

    ``files`` is ``{field_name: (filename, content_bytes, content_type)}``.
    """
    url = f"{worker.url}{path}"
    headers = _worker_headers(worker)
    request_timeout = aiohttp.ClientTimeout(
        total=timeout or float(getenv("REQUEST_TIMEOUT", "300"))
    )
    registry = get_registry()
    registry.increment_in_flight(worker.worker_id, 1)
    try:
        data = aiohttp.FormData()
        for name, (fname, content, ctype) in files.items():
            data.add_field(name, content, filename=fname, content_type=ctype)
        for k, v in fields.items():
            if v is not None:
                data.add_field(k, str(v))
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.post(url, data=data, headers=headers) as resp:
                body = await resp.read()
                media_type = resp.headers.get("Content-Type", "application/json")
                return Response(
                    content=body, status_code=resp.status, media_type=media_type
                )
    finally:
        registry.increment_in_flight(worker.worker_id, -1)


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models", tags=["Models"])
async def models(_: str = Depends(verify_client)):
    """Aggregate model list across all live workers."""
    seen: Dict[str, Dict[str, Any]] = {}
    for w in get_registry().list_workers(alive_only=True):
        for m in w.models:
            seen.setdefault(
                m, {"id": m, "object": "model", "owned_by": "ezlocalai", "workers": []}
            )["workers"].append(w.label)
    return {"object": "list", "data": list(seen.values())}


@app.post("/v1/chat/completions", tags=["Chat"])
async def chat_completions(payload: Dict[str, Any], _: str = Depends(verify_client)):
    model = payload.get("model")
    # Vision detection: any message with image_url content -> vision
    needs_vision = False
    for msg in payload.get("messages", []) or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url",
                    "input_image",
                ):
                    needs_vision = True
                    break
        if needs_vision:
            break
    capability = "vision" if needs_vision else "text"
    worker = await _pick(capability, model)
    return await _proxy_json(
        worker,
        "/v1/chat/completions",
        payload,
        stream=bool(payload.get("stream")),
    )


@app.post("/v1/completions", tags=["Completions"])
async def completions(payload: Dict[str, Any], _: str = Depends(verify_client)):
    worker = await _pick("text", payload.get("model"))
    return await _proxy_json(
        worker, "/v1/completions", payload, stream=bool(payload.get("stream"))
    )


@app.post("/v1/embeddings", tags=["Embeddings"])
async def embeddings(payload: Dict[str, Any], _: str = Depends(verify_client)):
    # Embeddings count as text capability for now (most ezlocalai workers serve both)
    worker = (
        await _pick("embedding", payload.get("model"))
        if any("embedding" in w.capabilities for w in get_registry().list_workers())
        else await _pick("text", payload.get("model"))
    )
    return await _proxy_json(worker, "/v1/embeddings", payload, stream=False)


@app.post("/v1/audio/speech", tags=["Audio"])
async def audio_speech(payload: Dict[str, Any], _: str = Depends(verify_client)):
    worker = await _pick("voice", payload.get("model"))
    return await _proxy_json(worker, "/v1/audio/speech", payload, stream=False)


@app.post("/v1/audio/speech/stream", tags=["Audio"])
async def audio_speech_stream(payload: Dict[str, Any], _: str = Depends(verify_client)):
    worker = await _pick("voice", payload.get("model"))
    return await _proxy_json(worker, "/v1/audio/speech/stream", payload, stream=True)


@app.get("/v1/audio/voices", tags=["Audio"])
async def audio_voices(_: str = Depends(verify_client)):
    # Aggregate voices from any voice-capable worker
    voices: List[Dict[str, Any]] = []
    seen = set()
    for w in get_registry().list_workers(alive_only=True):
        if "voice" not in w.capabilities:
            continue
        try:
            resp = await _proxy_get(w, "/v1/audio/voices")
            data = json.loads(resp.body) if resp.body else {}
            for v in data.get("voices", []) or data.get("data", []) or []:
                key = v.get("id") if isinstance(v, dict) else v
                if key not in seen:
                    seen.add(key)
                    voices.append(v)
        except Exception as e:
            logging.debug(f"[Router] voices from {w.url} failed: {e}")
    return {"voices": voices}


@app.post("/v1/audio/transcriptions", tags=["Audio"])
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamps: Optional[bool] = Form(None),
    _: str = Depends(verify_client),
):
    worker = await _pick("voice", model)
    content = await file.read()
    return await _proxy_multipart(
        worker,
        "/v1/audio/transcriptions",
        files={
            "file": (
                file.filename or "audio.wav",
                content,
                file.content_type or "audio/wav",
            )
        },
        fields={
            "model": model,
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
            "timestamps": timestamps,
        },
    )


@app.post("/v1/images/generations", tags=["Images"])
async def images_generations(payload: Dict[str, Any], _: str = Depends(verify_client)):
    worker = await _pick("image", payload.get("model"))
    return await _proxy_json(worker, "/v1/images/generations", payload, stream=False)


@app.post("/v1/images/edits", tags=["Images"])
async def images_edits(request: Request, _: str = Depends(verify_client)):
    """Image edits use multipart; forward all parts as-is."""
    form = await request.form()
    files: Dict[str, tuple] = {}
    fields: Dict[str, str] = {}
    for key, value in form.multi_items():
        if hasattr(value, "read"):
            content = await value.read()
            files[key] = (
                getattr(value, "filename", None) or "file",
                content,
                getattr(value, "content_type", None) or "application/octet-stream",
            )
        else:
            fields[key] = str(value)
    worker = await _pick("image", fields.get("model"))
    return await _proxy_multipart(
        worker, "/v1/images/edits", files=files, fields=fields
    )


@app.post("/v1/videos/generations", tags=["Videos"])
async def videos_generations(payload: Dict[str, Any], _: str = Depends(verify_client)):
    worker = await _pick("video", payload.get("model"))
    return await _proxy_json(
        worker,
        "/v1/videos/generations",
        payload,
        stream=False,
        timeout=float(getenv("REQUEST_TIMEOUT", "1800")),
    )
