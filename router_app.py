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
import html
import json
import logging
import os
import re
import time
from datetime import datetime
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
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from Globals import getenv
from Router import (
    Router,
    WorkerInfo,
    WorkerRegistry,
    best_gpu_tier,
    get_registry,
    get_router,
)
from Tunnel import (
    TunnelConnection,
    get_tunnel_hub,
    is_tunnel_url,
    worker_id_from_tunnel_url,
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

# Persisted asset storage — the router downloads any image/video/audio file
# referenced in a worker response into this directory and serves it back
# under /outputs/<filename> so clients have a stable URL even when the
# originating worker is on a private network or behind a tunnel.
_OUTPUTS_DIR = os.path.abspath(os.environ.get("ROUTER_OUTPUTS_DIR", "outputs"))
os.makedirs(_OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=_OUTPUTS_DIR), name="outputs")


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


def _normalize_model_name(model: Optional[str]) -> str:
    """Collapse minor naming variants so usage stats aggregate correctly.

    Treats ``unsloth/Foo-GGUF`` and ``unsloth/Foo`` as the same model — the
    ``-GGUF`` suffix only describes the on-disk weight format and clients
    request both forms interchangeably.
    """
    if not model:
        return "unknown"
    m = str(model).strip()
    # Strip a trailing -GGUF / .GGUF (case-insensitive) so quantized and
    # non-quantized references collapse into one bucket.
    low = m.lower()
    for suffix in ("-gguf", ".gguf"):
        if low.endswith(suffix):
            m = m[: -len(suffix)]
            break
    return m


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


class UsageTracker:
    """Persists per-worker, per-model usage stats to a JSON file.

    Data shape::

        {
          "WorkerLabel": {
            "llm": {
              "ModelName": {
                "requests": 42,
                "prompt_tokens": 12000,
                "completion_tokens": 8000
              }
            },
            "tts":   {"requests": 15},
            "stt":   {"requests": 7},
            "image": {"requests": 3},
            "video": {"requests": 0},
            "embedding": {"requests": 5}
          }
        }
    """

    def __init__(self, path: str) -> None:
        self._path = path
        # History persists next to the usage file (same directory + suffix).
        base, ext = os.path.splitext(path)
        self._history_path = f"{base}.history{ext or '.json'}"
        self._lock: Optional[asyncio.Lock] = None
        self._data: Dict[str, Any] = {}
        # Capped ring buffer of recent LLM requests, persisted to disk.  Each entry:
        # {ts, worker, model, prompt_tokens, completion_tokens,
        #  prompt_ms, predicted_ms, prompt_tps, predicted_tps}
        self._history: List[Dict[str, Any]] = []
        self._history_max = int(getenv("USAGE_HISTORY_MAX", "500"))
        self._dirty: bool = False

    @property
    def _alock(self) -> asyncio.Lock:
        """Lazily created so it lives inside the running event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def load(self) -> None:
        """Load existing stats from disk (call at startup)."""
        try:
            with open(self._path) as fh:
                self._data = json.load(fh)
            logging.info(f"[Usage] Loaded stats from {self._path}")
        except FileNotFoundError:
            self._data = {}
            logging.info(
                f"[Usage] No existing stats file at {self._path}, starting fresh"
            )
        except Exception as e:
            self._data = {}
            logging.warning(
                f"[Usage] Failed to load {self._path}: {e} — starting fresh"
            )
        # History is persisted as a list of dict entries.
        try:
            with open(self._history_path) as fh:
                hist = json.load(fh)
            if isinstance(hist, list):
                # Trim if the file grew beyond the configured cap.
                self._history = hist[-self._history_max :]
                logging.info(
                    f"[Usage] Loaded {len(self._history)} history entries from "
                    f"{self._history_path}"
                )
        except FileNotFoundError:
            self._history = []
        except Exception as e:
            self._history = []
            logging.warning(
                f"[Usage] Failed to load history {self._history_path}: {e} — "
                "starting fresh"
            )

    def _save_sync(self) -> None:
        """Write atomically (tmp → rename). Runs in a thread executor — never call
        from the event loop thread directly."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        except Exception:
            pass
        tmp = self._path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self._data, fh, indent=2)
        os.replace(tmp, self._path)

    async def flush(self) -> None:
        """Write to disk if dirty. Runs the blocking I/O in a thread pool so the
        event loop is never blocked."""
        async with self._alock:
            if not self._dirty:
                return
            data_snapshot = json.loads(json.dumps(self._data))
            history_snapshot = list(self._history)
            self._dirty = False
        loop = asyncio.get_event_loop()
        _snap = data_snapshot
        _hist = history_snapshot
        _path = self._path
        _hist_path = self._history_path

        def _write() -> None:
            try:
                os.makedirs(os.path.dirname(_path) or ".", exist_ok=True)
            except Exception:
                pass
            tmp = _path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(_snap, fh, indent=2)
            os.replace(tmp, _path)
            # Persist history (no indent — it can grow large).
            tmp_h = _hist_path + ".tmp"
            with open(tmp_h, "w") as fh:
                json.dump(_hist, fh)
            os.replace(tmp_h, _hist_path)

        await loop.run_in_executor(None, _write)

    async def record_llm(
        self,
        label: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        timings: Optional[Dict[str, float]] = None,
    ) -> None:
        # Skip recording when we got nothing back — usually means the response
        # was an error (4xx/5xx) or the stream was aborted before the worker
        # emitted a usage/timings event. Recording 0/0 entries pollutes the
        # history with meaningless rows and drags down per-model averages.
        if int(prompt_tokens or 0) == 0 and int(completion_tokens or 0) == 0:
            return
        model = _normalize_model_name(model)
        prompt_ms = float((timings or {}).get("prompt_ms") or 0.0)
        predicted_ms = float((timings or {}).get("predicted_ms") or 0.0)
        total_ms = float((timings or {}).get("total_ms") or 0.0)
        prompt_tps = (prompt_tokens / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0
        predicted_tps = (
            (completion_tokens / (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0
        )
        async with self._alock:
            w = self._data.setdefault(label, {})
            llm = w.setdefault("llm", {})
            m = llm.setdefault(
                model,
                {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_tps_sum": 0.0,
                    "prompt_tps_n": 0,
                    "predicted_tps_sum": 0.0,
                    "predicted_tps_n": 0,
                },
            )
            m["requests"] += 1
            m["prompt_tokens"] += prompt_tokens
            m["completion_tokens"] += completion_tokens
            if prompt_tps > 0:
                m["prompt_tps_sum"] = float(m.get("prompt_tps_sum", 0.0)) + prompt_tps
                m["prompt_tps_n"] = int(m.get("prompt_tps_n", 0)) + 1
            if predicted_tps > 0:
                m["predicted_tps_sum"] = (
                    float(m.get("predicted_tps_sum", 0.0)) + predicted_tps
                )
                m["predicted_tps_n"] = int(m.get("predicted_tps_n", 0)) + 1
            self._history.append(
                {
                    "ts": time.time(),
                    "worker": label,
                    "model": model,
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "prompt_ms": prompt_ms,
                    "predicted_ms": predicted_ms,
                    "total_ms": total_ms,
                    "prompt_tps": prompt_tps,
                    "predicted_tps": predicted_tps,
                }
            )
            if len(self._history) > self._history_max:
                # Drop oldest in chunks to avoid O(n) churn on every insert.
                drop = len(self._history) - self._history_max
                del self._history[:drop]
            self._dirty = True

    async def record_cap(self, label: str, cap: str) -> None:
        async with self._alock:
            w = self._data.setdefault(label, {})
            c = w.setdefault(cap, {"requests": 0})
            c["requests"] += 1
            self._dirty = True

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current stats (no lock needed for reads)."""
        return json.loads(json.dumps(self._data))

    def history_snapshot(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` LLM request entries (newest last)."""
        if limit is None or limit >= len(self._history):
            return list(self._history)
        return list(self._history[-int(limit) :])


_usage = UsageTracker(getenv("USAGE_FILE", "/data/usage.json"))


# ---------------------------------------------------------------------------
# Streaming SSE token extractor
# ---------------------------------------------------------------------------


def _extract_tokens_from_sse_event(
    data_bytes: bytes, pt: int, ct: int, timings: Optional[Dict[str, float]] = None
) -> tuple:
    """Pull token counts (and optional timing data) out of a single decoded
    SSE ``data:`` payload.

    Supports both OpenAI's ``usage`` block (when the backend honors
    ``stream_options.include_usage``) and llama.cpp/xllamacpp's ``timings``
    block (which is emitted on the final chunk by default).

    If ``timings`` (a mutable dict) is provided, llama.cpp's millisecond
    counters are merged into it so the caller can derive tokens/sec.
    """
    try:
        obj = json.loads(data_bytes)
    except Exception:
        return pt, ct
    u = obj.get("usage")
    if isinstance(u, dict):
        pt = int(u.get("prompt_tokens") or pt)
        ct = int(u.get("completion_tokens") or ct)
    t = obj.get("timings")
    if isinstance(t, dict):
        pt = int(t.get("prompt_n") or pt)
        ct = int(t.get("predicted_n") or ct)
        if timings is not None:
            for k in ("prompt_ms", "predicted_ms"):
                v = t.get(k)
                if isinstance(v, (int, float)):
                    timings[k] = float(v)
    return pt, ct


async def _stream_with_token_extraction(
    body_iterator,
    record_callback,
):
    """Wrap an SSE body_iterator, pass chunks through unchanged, and call
    ``record_callback(prompt_tokens, completion_tokens, timings)`` exactly
    once when the stream ends. ``timings`` is a dict that may contain
    ``prompt_ms`` / ``predicted_ms`` if the backend emitted them.
    """
    pt = 0
    ct = 0
    timings: Dict[str, float] = {}
    buf = b""
    try:
        async for chunk in body_iterator:
            if not chunk:
                continue
            yield chunk
            buf += chunk
            # Process any complete SSE events in the buffer
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                # An event may have multiple lines (data:, event:, id:, ...).
                # We only care about ``data:`` lines.
                for line in event.split(b"\n"):
                    s = line.strip()
                    if not s.startswith(b"data:"):
                        continue
                    payload = s[5:].lstrip()
                    if not payload or payload == b"[DONE]":
                        continue
                    pt, ct = _extract_tokens_from_sse_event(payload, pt, ct, timings)
            # Cap buffer growth — only the most recent event matters for
            # token extraction, so trim aggressively if no delimiters appear.
            if len(buf) > 65536:
                buf = buf[-32768:]
        # Trailing data without a terminating ``\n\n``
        if buf:
            for line in buf.split(b"\n"):
                s = line.strip()
                if s.startswith(b"data:"):
                    payload = s[5:].lstrip()
                    if payload and payload != b"[DONE]":
                        pt, ct = _extract_tokens_from_sse_event(
                            payload, pt, ct, timings
                        )
    finally:
        await record_callback(pt, ct, timings)


# ---------------------------------------------------------------------------
# Background pruner + usage flusher
# ---------------------------------------------------------------------------

_pruner_task: Optional[asyncio.Task] = None
_usage_flush_task: Optional[asyncio.Task] = None


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


async def _usage_flush_loop():
    """Flush usage stats to disk every 10 seconds if dirty."""
    while True:
        await asyncio.sleep(10)
        try:
            await _usage.flush()
        except Exception as e:
            logging.warning(f"[Usage] Flush failed: {e}")


@app.on_event("startup")
async def _startup():
    global _pruner_task, _usage_flush_task
    _usage.load()
    if _pruner_task is None or _pruner_task.done():
        _pruner_task = asyncio.create_task(_pruner_loop())
    if _usage_flush_task is None or _usage_flush_task.done():
        _usage_flush_task = asyncio.create_task(_usage_flush_loop())
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
    global _pruner_task, _usage_flush_task
    # Flush any pending usage stats before exiting
    try:
        await _usage.flush()
    except Exception:
        pass
    for task in (_pruner_task, _usage_flush_task):
        if task and not task.done():
            task.cancel()
            try:
                await task
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
        total_ram_gb=float(payload.get("total_ram_gb", 0.0) or 0.0),
        queue_depth=int(payload.get("queue_depth", 0) or 0),
        queue_capacity=int(payload.get("queue_capacity", 1) or 1),
        in_flight=int(payload.get("in_flight", 0) or 0),
        cap_slots={
            str(k): WorkerInfo._normalize_slot(v)
            for k, v in (payload.get("cap_slots") or {}).items()
            if isinstance(v, dict)
        },
        model_slots={
            str(k): WorkerInfo._normalize_slot(v)
            for k, v in (payload.get("model_slots") or {}).items()
            if isinstance(v, dict)
        },
        gpus=gpus,
        best_tier=int(payload.get("best_tier") or best_gpu_tier(gpus)),
        model_context={
            str(k): int(v) for k, v in (payload.get("model_context") or {}).items()
        },
        model_quant={
            str(k): str(v) for k, v in (payload.get("model_quant") or {}).items() if v
        },
        cap_models={
            str(k): str(v) for k, v in (payload.get("cap_models") or {}).items() if v
        },
        version=str(payload.get("version") or ""),
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
                "total_ram_gb",
                "queue_depth",
                "queue_capacity",
                "in_flight",
                "cap_slots",
                "model_slots",
                "gpus",
                "best_tier",
                "model_context",
                "model_quant",
                "cap_models",
                "version",
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


@app.websocket("/v1/router/tunnel")
async def router_tunnel(websocket: WebSocket):
    """Reverse tunnel endpoint.

    A worker dials this URL with ``?worker_id=<id>`` and an
    ``Authorization: Bearer <key>`` header (when ROUTER_REGISTER_KEY /
    EZLOCALAI_API_KEY is set). The router multiplexes inbound HTTP requests
    to the worker through the resulting WebSocket — no inbound network
    access is required on the worker side.
    """
    expected = _expected_register_key()
    if expected:
        auth = websocket.headers.get("authorization") or ""
        provided = ""
        if auth.lower().startswith("bearer "):
            provided = auth.split(" ", 1)[1].strip()
        if provided != expected:
            await websocket.close(code=4401)
            return
    worker_id = websocket.query_params.get("worker_id") or ""
    if not worker_id:
        await websocket.close(code=4400)
        return
    await websocket.accept()
    hub = get_tunnel_hub()
    conn = TunnelConnection(worker_id=worker_id, ws=websocket, hub=hub)
    await hub.attach(conn)
    logging.info(f"[Router] Tunnel connected: {worker_id}")
    try:
        await conn.reader_loop()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.warning(f"[Router] Tunnel {worker_id} error: {e}")
    finally:
        await conn.close(reason="ws closed")
        logging.info(f"[Router] Tunnel disconnected: {worker_id}")


@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


# Human-readable labels for capability names used throughout the dashboard.
_CAP_LABELS: Dict[str, str] = {
    "text": "Text",
    "vision": "Vision",
    "text+vision": "Text + Vision",
    "image": "Image Generation",
    "tts": "Text-to-Speech",
    "stt": "Speech-to-Text",
    "embedding": "Embedding",
    "video": "Video",
}


def _cap_label(cap: str) -> str:
    return _CAP_LABELS.get(cap, cap.replace("-", " ").title())


def _aggregate_dashboard() -> Dict[str, Any]:
    """Build a JSON-serialisable summary of the entire pool."""
    registry = get_registry()
    alive = registry.list_workers(alive_only=True)
    stale = [w for w in registry.list_workers(alive_only=False) if w not in alive]

    total_capacity = sum(max(0, w.total_capacity()) for w in alive)
    total_in_flight = sum(max(0, w.total_busy()) for w in alive)
    total_queue_depth = sum(max(0, w.queue_depth) for w in alive)
    total_slots_left = sum(max(0, w.total_slots_left()) for w in alive)
    total_free_vram = sum(w.free_vram_gb for w in alive)
    total_vram = sum(w.total_vram_gb for w in alive)

    # Per-model rollup: which workers serve it, total parallel slots, max ctx
    model_rollup: Dict[str, Dict[str, Any]] = {}
    for w in alive:
        for model in w.models:
            model_capacity = max(1, w.capacity_for("text", model))
            slots_left = w.slots_left("text", model)
            entry = model_rollup.setdefault(
                model,
                {
                    "model": model,
                    "type": "text",
                    "worker_count": 0,
                    "total_capacity": 0,
                    "available_slots": 0,
                    "max_context": 0,
                    "best_tier": 0,
                    "workers": [],
                },
            )
            entry["worker_count"] += 1
            entry["total_capacity"] += model_capacity
            entry["available_slots"] += slots_left
            ctx = int(w.model_context.get(model, 0) or 0)
            if ctx > entry["max_context"]:
                entry["max_context"] = ctx
            if w.best_tier > entry["best_tier"]:
                entry["best_tier"] = w.best_tier
            if "vision" in w.capabilities and entry["type"] == "text":
                entry["type"] = "text+vision"
            entry["workers"].append(
                {
                    "label": w.label,
                    "worker_id": w.worker_id,
                    "best_tier": w.best_tier,
                    "context": ctx,
                    "slots_left": slots_left,
                    "queue_capacity": model_capacity,
                    "quant": w.model_quant.get(model, ""),
                }
            )
            q = w.model_quant.get(model)
            if q:
                entry.setdefault("quants", set()).add(q)

    # Non-text capability model entries: image, tts, stt, video workers
    # Each worker that has one of these caps gets a synthetic model row so the
    # Models section shows *all* running AI models, not just LLMs.
    _NON_TEXT_CAPS = ("image", "tts", "stt", "video")
    for w in alive:
        for cap in w.capabilities:
            if cap not in _NON_TEXT_CAPS:
                continue
            cap_label = _cap_label(cap)
            # Use the actual model name the worker reported for this capability,
            # falling back to the capability label if not reported yet.
            model_name = w.cap_models.get(cap) or cap_label
            # Key by cap+model_name so multiple workers running the same
            # model (e.g. two nodes both serving Chatterbox TTS) share one row.
            key = f"{cap}::{model_name}"
            entry = model_rollup.setdefault(
                key,
                {
                    "model": model_name,
                    "type": cap,
                    "worker_count": 0,
                    "total_capacity": 0,
                    "available_slots": 0,
                    "max_context": 0,
                    "best_tier": 0,
                    "workers": [],
                    "quants": set(),
                },
            )
            entry["worker_count"] += 1
            cap_capacity = max(1, w.capacity_for(cap))
            cap_slots_left = w.slots_left(cap)
            entry["total_capacity"] += cap_capacity
            entry["available_slots"] += cap_slots_left
            if w.best_tier > entry["best_tier"]:
                entry["best_tier"] = w.best_tier
            entry["workers"].append(
                {
                    "label": w.label,
                    "worker_id": w.worker_id,
                    "best_tier": w.best_tier,
                    "context": 0,
                    "slots_left": cap_slots_left,
                    "queue_capacity": cap_capacity,
                    "quant": "",
                }
            )

    # Per-capability rollup — text + vision are merged into a single "text+vision" entry
    # when a worker has both, to avoid double-counting on the dashboard.
    _CAP_DISPLAY_ORDER = [
        "text+vision",
        "text",
        "embedding",
        "image",
        "tts",
        "stt",
        "video",
    ]
    cap_rollup: Dict[str, Dict[str, Any]] = {}
    for w in alive:
        has_text = "text" in w.capabilities
        has_vision = "vision" in w.capabilities
        for cap in w.capabilities:
            # Skip the bare "vision" entry when the worker also does text;
            # it will be represented under "text+vision" instead.
            if cap == "vision" and has_text:
                continue
            merged_cap = "text+vision" if (cap == "text" and has_vision) else cap
            # Use the underlying text capacity for text+vision
            cap_key = "text" if merged_cap == "text+vision" else merged_cap
            cap_slots = max(1, w.capacity_for(cap_key))
            cap_slots_left = w.slots_left(cap_key)
            entry = cap_rollup.setdefault(
                merged_cap,
                {
                    "capability": merged_cap,
                    "worker_count": 0,
                    "total_capacity": 0,
                    "available_slots": 0,
                    "worker_labels": [],
                },
            )
            entry["worker_count"] += 1
            entry["total_capacity"] += cap_slots
            entry["available_slots"] += cap_slots_left
            entry["worker_labels"].append(w.label)

    pool_health = (
        "offline" if len(alive) == 0 else "degraded" if len(stale) > 0 else "healthy"
    )
    return {
        "generated_at": time.time(),
        "pool_health": pool_health,
        "router": {
            "ttl_seconds": registry.ttl,
            "wait_timeout": _wait_timeout(),
        },
        "totals": {
            "alive_workers": len(alive),
            "stale_workers": len(stale),
            "total_parallel_capacity": total_capacity,
            "total_in_flight": total_in_flight,
            "total_queue_depth": total_queue_depth,
            "total_available_slots": total_slots_left,
            "total_free_vram_gb": round(total_free_vram, 2),
            "total_vram_gb": round(total_vram, 2),
            "unique_models": len(model_rollup),
        },
        "capabilities": sorted(
            cap_rollup.values(),
            key=lambda x: (
                (
                    _CAP_DISPLAY_ORDER.index(x["capability"])
                    if x["capability"] in _CAP_DISPLAY_ORDER
                    else 99
                ),
                x["capability"],
            ),
        ),
        "models": sorted(
            [
                {**m, "quants": sorted(m.get("quants", set()))}
                for m in model_rollup.values()
            ],
            key=lambda x: (
                # Text/vision models first, then non-text by cap type, then alpha by name
                0 if x["type"] in ("text", "text+vision", "vision", "embedding") else 1,
                x.get("type", ""),
                x["model"].lower(),
            ),
        ),
        "workers": sorted(
            [_public_with_tunnel(w) for w in alive],
            key=lambda x: -x.get("best_tier", 0),
        )
        + sorted(
            [{**_public_with_tunnel(w), "stale": True} for w in stale],
            key=lambda x: -x.get("best_tier", 0),
        ),
        "usage": _usage.snapshot(),
        "history": _usage.history_snapshot(),
        "errors": _aggregate_recent_errors(alive + stale),
    }


def _public_with_tunnel(w) -> Dict[str, Any]:
    """Worker.to_public() plus tunnel status for tunneled workers."""
    pub = w.to_public()
    if is_tunnel_url(w.url):
        hub = get_tunnel_hub()
        wid = worker_id_from_tunnel_url(w.url)
        pub["tunnel"] = True
        pub["tunnel_connected"] = hub.is_connected(wid)
    else:
        pub["tunnel"] = False
        pub["tunnel_connected"] = None
    return pub


def _aggregate_recent_errors(workers) -> List[Dict[str, Any]]:
    """Flatten recent errors from every worker, newest first, capped."""
    out: List[Dict[str, Any]] = []
    for w in workers:
        for ev in getattr(w, "recent_errors", []) or []:
            out.append(
                {
                    "ts": ev.get("ts", 0),
                    "worker_id": w.worker_id,
                    "label": w.label,
                    "kind": ev.get("kind", ""),
                    "status": ev.get("status"),
                    "path": ev.get("path", ""),
                    "message": ev.get("message", ""),
                }
            )
    out.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return out[:100]


@app.get("/v1/router/dashboard", tags=["Router"])
async def router_dashboard_json(_: str = Depends(verify_client)):
    """JSON form of the dashboard data."""
    return _aggregate_dashboard()


@app.get("/v1/router/tunnels", tags=["Router"])
async def router_tunnels(_: str = Depends(verify_client)):
    """Currently connected reverse-tunnel WebSockets and their status
    correlated to registered workers."""
    hub = get_tunnel_hub()
    connected = hub.connected_workers()  # {worker_id: connected_at}
    now = time.time()
    workers = []
    for w in get_registry().list_workers(alive_only=False):
        if not is_tunnel_url(w.url):
            continue
        wid = worker_id_from_tunnel_url(w.url)
        connected_at = connected.get(wid)
        conn = hub.get(wid)
        last_close_reason = (
            getattr(conn, "last_close_reason", "") if conn is not None else ""
        )
        last_recv_age = None
        if conn is not None and getattr(conn, "last_recv", None):
            last_recv_age = round(now - conn.last_recv, 2)
        stats = hub.stats(wid)
        workers.append(
            {
                "worker_id": w.worker_id,
                "label": w.label,
                "tunnel_worker_id": wid,
                "connected": connected_at is not None,
                "connected_at": connected_at,
                "connected_for_seconds": (
                    (now - connected_at) if connected_at else None
                ),
                "last_recv_age_seconds": last_recv_age,
                "last_close_reason": last_close_reason,
                "connect_count": stats["connect_count"],
                "disconnect_history": stats["disconnect_history"],
                "alive": w.is_alive(get_registry().ttl),
                "last_heartbeat_age": round(now - w.last_heartbeat, 2),
            }
        )
    orphans = [
        {"tunnel_worker_id": wid, "connected_at": ts}
        for wid, ts in connected.items()
        if not any(
            wid == worker_id_from_tunnel_url(w.url)
            for w in get_registry().list_workers(alive_only=False)
        )
    ]
    return {"workers": workers, "orphan_tunnels": orphans}


@app.get("/v1/router/usage", tags=["Router"])
async def router_usage(_: str = Depends(verify_client)):
    """Historical per-worker usage stats (persisted across restarts)."""
    return _usage.snapshot()


@app.get("/v1/router/history", tags=["Router"])
async def router_history(limit: int = 200, _: str = Depends(verify_client)):
    """Recent LLM request history (in-memory ring buffer, newest last).

    Each entry contains: ``ts``, ``worker``, ``model``, ``prompt_tokens``,
    ``completion_tokens``, ``prompt_ms``, ``predicted_ms``, ``prompt_tps``,
    ``predicted_tps``.
    """
    return {"history": _usage.history_snapshot(limit=limit)}


@app.get("/v1/router/errors", tags=["Router"])
async def router_errors(_: str = Depends(verify_client)):
    """Per-worker recent error events plus circuit-breaker state.

    Returns a list of workers with their ``total_errors``, ``circuit_open``,
    ``circuit_open_until``, and the most recent error events.  Also returns
    a flat ``errors`` list across all workers, newest first, capped at 100.
    """
    registry = get_registry()
    workers = registry.list_workers(alive_only=False)
    return {
        "workers": [
            {
                "worker_id": w.worker_id,
                "label": w.label,
                "total_errors": w.total_errors,
                "circuit_open": w.is_circuit_open(),
                "circuit_open_until": w.circuit_open_until,
                "recent_errors": list(w.recent_errors or []),
            }
            for w in workers
        ],
        "errors": _aggregate_recent_errors(workers),
    }


# ---------------------------------------------------------------------------
# Dashboard visual helpers
# ---------------------------------------------------------------------------

# Per-capability CSS class names
_CAP_PILL_CLASS: Dict[str, str] = {
    "text": "cap-text",
    "vision": "cap-vision",
    "text+vision": "cap-vision",
    "image": "cap-image",
    "tts": "cap-tts",
    "stt": "cap-stt",
    "video": "cap-video",
    "embedding": "cap-embedding",
}


def _cap_pill(cap: str) -> str:
    """Colored pill span for a capability."""
    cls = _CAP_PILL_CLASS.get(cap, "")
    return f'<span class="pill {cls}">{_cap_label(cap)}</span>'


def _tier_badge(tier: int) -> str:
    """Color-coded tier indicator."""
    if tier >= 80:
        cls, star = "tier-gold", "⭐ "
    elif tier >= 50:
        cls, star = "tier-blue", ""
    elif tier >= 20:
        cls, star = "tier-green", ""
    elif tier >= 5:
        cls, star = "tier-warn", ""
    else:
        cls, star = "tier-muted", ""
    return f'<span class="{cls}">{star}tier {tier}</span>'


def _usage_bar(used_pct: float, width: str = "100%") -> str:
    """Mini horizontal usage bar. used_pct 0-100."""
    if used_pct >= 85:
        fill_cls = "crit"
    elif used_pct >= 60:
        fill_cls = "warn"
    else:
        fill_cls = "good"
    return (
        f'<div class="usage-bar" style="width:{width}">'
        f'<div class="usage-fill {fill_cls}" style="width:{used_pct:.0f}%"></div>'
        f"</div>"
    )


_HEALTH_STYLE = {
    "healthy": ("health-ok", "● Healthy"),
    "degraded": ("health-warn", "◐ Degraded"),
    "offline": ("health-crit", "○ Offline"),
}


def _version_link(version: str) -> str:
    """Render commit SHAs as links; other version IDs as plain text."""
    version = str(version or "").strip()
    if not version:
        return "—"
    safe_version = html.escape(version)
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", version):
        return safe_version
    url = f"https://github.com/DevXT-LLC/ezlocalai/commit/{version}"
    return f'<a href="{url}" target="_blank" rel="noopener">{safe_version}</a>'


def _render_dashboard_html(data: Dict[str, Any]) -> str:
    totals = data["totals"]
    router_meta = data["router"]
    pool_health = data.get("pool_health", "healthy")
    health_cls, health_label = _HEALTH_STYLE.get(
        pool_health, ("health-ok", "● Healthy")
    )

    # Capability cards
    def _cap_card(c: Dict[str, Any]) -> str:
        worker_pills = " ".join(
            f'<span class="pill cap-worker" title="{lbl}">{lbl}</span>'
            for lbl in c.get("worker_labels", [])
        )
        used_pct = 100 * (1 - c["available_slots"] / max(c["total_capacity"], 1))
        return f"""
        <div class="card cap">
          <div class="cap-header">{_cap_pill(c['capability'])}</div>
          <div class="cap-stats">
            <span><b>{c['worker_count']}</b> worker{'s' if c['worker_count'] != 1 else ''}</span>
            <span><b>{c['available_slots']}</b>/{c['total_capacity']} slots free</span>
          </div>
          {_usage_bar(used_pct, '100%')}
          {("<div class='cap-workers'>" + worker_pills + "</div>") if worker_pills else ""}
        </div>
        """

    cap_cards = (
        "".join(_cap_card(c) for c in data["capabilities"])
        or '<div class="muted">No capabilities advertised yet.</div>'
    )

    # Model rows
    def _model_row(m: Dict[str, Any]) -> str:
        worker_pills = "".join(
            f'<span class="pill" title="tier {w["best_tier"]} · {w["slots_left"]}/{w["queue_capacity"]} slots free · ctx {w["context"] or "?"} · quant {w.get("quant") or "?"}">'
            f'{w["label"]}</span>'
            for w in m["workers"]
        )
        mtype = m.get("type", "text")
        type_badge = _cap_pill(mtype) + " "
        ctx = f"{m['max_context']:,}" if m["max_context"] else "—"
        quants = ", ".join(m.get("quants") or []) or "—"
        return f"""
        <tr>
          <td class="mono">{type_badge}{m['model']}</td>
          <td class="num">{m['worker_count']}</td>
          <td class="num">{m['total_capacity']}</td>
          <td class="num">{m['available_slots']}</td>
          <td class="num">{'—' if mtype not in ('text', 'text+vision', 'vision', 'embedding') else ctx}</td>
          <td class="mono small">{'—' if mtype not in ('text', 'text+vision', 'vision', 'embedding') else quants}</td>
          <td class="num">{m['best_tier']}</td>
          <td>{worker_pills}</td>
        </tr>
        """

    model_rows = "".join(_model_row(m) for m in data["models"]) or (
        '<tr><td colspan="8" class="muted">No models loaded across the pool.</td></tr>'
    )

    # Worker rows
    def _worker_row(w: Dict[str, Any]) -> str:
        stale = w.get("stale")
        # Build accelerator summary — skip pure CPU entry if real accelerators exist
        raw_gpus = w.get("gpus") or []
        accel_gpus = [g for g in raw_gpus if g.get("index", 0) >= 0]
        cpu_entry = next((g for g in raw_gpus if g.get("backend") == "cpu"), None)
        display_gpus = accel_gpus if accel_gpus else ([cpu_entry] if cpu_entry else [])
        gpus = (
            ", ".join(
                f"{g.get('name', '?')}"
                + (
                    f" ({g.get('total_vram_gb', 0):.0f}GB VRAM)"
                    if g.get("total_vram_gb")
                    else (
                        f" ({w.get('total_ram_gb') or w.get('free_ram_gb') or 0:.0f}GB RAM)"
                        if g.get("backend") == "cpu"
                        and (w.get("total_ram_gb") or w.get("free_ram_gb"))
                        else ""
                    )
                )
                + (
                    f" [{g.get('backend','')}]"
                    if g.get("backend") not in ("cuda", None, "")
                    else ""
                )
                for g in display_gpus
            )
            or "CPU"
        )
        # VRAM / RAM cell with usage bar
        free_vram = float(w.get("free_vram_gb") or 0)
        total_vram = float(w.get("total_vram_gb") or 0)
        free_ram = float(w.get("free_ram_gb") or 0)
        total_ram = float(w.get("total_ram_gb") or 0)
        if total_vram > 0:
            used_pct = max(0.0, min(100.0, (1 - free_vram / total_vram) * 100))
            mem_cell = (
                f"{free_vram:.1f}/{total_vram:.0f} GB VRAM"
                f"{_usage_bar(used_pct, '80px')}"
            )
        elif total_ram > 0:
            used_pct = max(0.0, min(100.0, (1 - free_ram / total_ram) * 100))
            mem_cell = (
                f"{free_ram:.1f}/{total_ram:.0f} GB RAM"
                f"{_usage_bar(used_pct, '80px')}"
            )
        else:
            mem_cell = f"{free_ram:.1f} GB RAM free" if free_ram else "—"
        mq = w.get("model_quant") or {}
        mc = w.get("model_context") or {}

        def _fmt(name: str) -> str:
            ctx = int(mc.get(name, 0) or 0)
            quant = mq.get(name) or ""
            ctx_part = f" @ {ctx:,}" if ctx else ""
            quant_part = f" ({quant})" if quant else ""
            return f"{name}{ctx_part}{quant_part}"

        models = "<br>".join(_fmt(m) for m in (w.get("models") or [])) or "—"
        slots_left = int(w.get("slot_total_available", 0) or 0)
        slots_total = int(w.get("slot_total_capacity", 0) or 0)
        if slots_total <= 0:
            slots_total = int(w.get("queue_capacity", 1) or 1)
            slots_left = max(0, slots_total - int(w.get("queue_depth", 0) or 0))
        slot_pct = 0 if slots_total == 0 else (1 - slots_left / slots_total) * 100
        bar = f'<div class="bar"><div class="bar-fill" style="width:{slot_pct:.0f}%"></div></div>'
        last_hb = w.get("last_heartbeat_age", 0)
        tunnel_connected = w.get("tunnel_connected")
        if stale:
            status = "🔴 stale"
        elif tunnel_connected is False:
            status = '<span style="color:#f85149">🔌 tunnel-off</span>'
        elif slots_left > 0:
            status = "🟢 ready"
        else:
            status = "🟡 full"
        circuit_open = bool(w.get("circuit_open"))
        total_errors = int(w.get("total_errors", 0) or 0)
        if circuit_open:
            err_cell = (
                f'<span style="color:#f85149">🔴 OPEN</span>'
                f'<div class="muted small">{total_errors} total</div>'
            )
        elif total_errors:
            err_cell = (
                f'<span style="color:#d29922">{total_errors}</span>'
                f'<div class="muted small">recent ok</div>'
            )
        else:
            err_cell = '<span class="muted">0</span>'
        raw_caps = w.get("capabilities") or []
        # Merge text + vision into a single "text+vision" pill on the worker row
        if "text" in raw_caps and "vision" in raw_caps:
            display_caps = [
                "text+vision" if c == "text" else c for c in raw_caps if c != "vision"
            ]
        else:
            display_caps = raw_caps
        cap_slots = w.get("cap_slots") or {}

        def _cap_with_title(c: str) -> str:
            raw = cap_slots.get("text" if c == "text+vision" else c) or {}
            capacity = int(raw.get("capacity", 0) or 0)
            available = int(raw.get("available", 0) or 0)
            title = f"{available}/{capacity} slots free" if capacity else ""
            pill = _cap_pill(c)
            return (
                pill.replace("<span ", f'<span title="{title}" ', 1) if title else pill
            )

        cap_pills = " ".join(_cap_with_title(c) for c in display_caps) or "—"
        if w.get("tunnel"):
            url_cell = (
                f"{w['url']}"
                f'<div class="muted small">🔌 '
                f"{'connected' if tunnel_connected else 'disconnected'}</div>"
            )
        else:
            url_cell = w["url"]
        return f"""
        <tr class="{'stale' if stale else ''}">
          <td><b>{w['label']}</b><div class="muted small">{w['worker_id']}</div></td>
          <td>{status}</td>
          <td class="mono small">{url_cell}</td>
          <td>{gpus}<div class="muted small">{_tier_badge(w.get('best_tier', 0))}</div></td>
          <td class="num small">{mem_cell}</td>
          <td>{slots_left}/{slots_total} {bar}<div class="muted small">{w.get('slot_total_busy', w.get('in_flight', 0))} active/queued</div></td>
          <td class="small">{cap_pills}</td>
          <td class="small mono">{models}</td>
          <td class="num small">{last_hb:.0f}s</td>
          <td class="mono small">{_version_link(w.get('version') or '')}</td>
          <td class="small">{err_cell}</td>
        </tr>
        """

    worker_rows = "".join(_worker_row(w) for w in data["workers"]) or (
        '<tr><td colspan="11" class="muted">No workers registered.</td></tr>'
    )

    # Usage section — per-worker historical stats
    usage_data = data.get("usage") or {}

    def _usage_summary_row(label: str, wdata: Dict[str, Any]) -> str:
        llm = wdata.get("llm") or {}
        llm_reqs = sum(m.get("requests", 0) for m in llm.values())
        prompt_tok = sum(m.get("prompt_tokens", 0) for m in llm.values())
        comp_tok = sum(m.get("completion_tokens", 0) for m in llm.values())
        tts_reqs = (wdata.get("tts") or {}).get("requests", 0)
        stt_reqs = (wdata.get("stt") or {}).get("requests", 0)
        img_reqs = (wdata.get("image") or {}).get("requests", 0)
        vid_reqs = (wdata.get("video") or {}).get("requests", 0)
        emb_reqs = (wdata.get("embedding") or {}).get("requests", 0)
        total_tok = prompt_tok + comp_tok
        return f"""
        <tr>
          <td><b>{label}</b></td>
          <td class="num">{llm_reqs:,}</td>
          <td class="num">{prompt_tok:,}</td>
          <td class="num">{comp_tok:,}</td>
          <td class="num {'tok-hi' if total_tok > 0 else ''}">{total_tok:,}</td>
          <td class="num">{emb_reqs:,}</td>
          <td class="num">{tts_reqs:,}</td>
          <td class="num">{stt_reqs:,}</td>
          <td class="num">{img_reqs:,}</td>
          <td class="num">{vid_reqs:,}</td>
        </tr>"""

    # ---- Recent request history -------------------------------------------
    history = data.get("history") or []

    def _fmt_tps(v: float) -> str:
        if not v or v <= 0:
            return "—"
        return f"{v:,.1f}"

    # Per (worker, canonical model) speed averages — only entries with non-zero
    # timing data contribute (so we don't dilute averages with non-llama.cpp
    # backends that didn't report ``timings``).
    speed_by_wm: Dict[tuple, Dict[str, float]] = {}
    for h in history:
        key = (
            h.get("worker") or "unknown",
            _normalize_model_name(h.get("model") or ""),
        )
        s = speed_by_wm.setdefault(
            key,
            {
                "prompt_tps_sum": 0.0,
                "prompt_tps_n": 0,
                "predicted_tps_sum": 0.0,
                "predicted_tps_n": 0,
            },
        )
        ptps = float(h.get("prompt_tps") or 0.0)
        if ptps > 0:
            s["prompt_tps_sum"] += ptps
            s["prompt_tps_n"] += 1
        dtps = float(h.get("predicted_tps") or 0.0)
        if dtps > 0:
            s["predicted_tps_sum"] += dtps
            s["predicted_tps_n"] += 1

    def _avg_tps(key: tuple) -> tuple:
        s = speed_by_wm.get(key)
        if not s:
            return 0.0, 0.0
        ap = (s["prompt_tps_sum"] / s["prompt_tps_n"]) if s["prompt_tps_n"] else 0.0
        ad = (
            (s["predicted_tps_sum"] / s["predicted_tps_n"])
            if s["predicted_tps_n"]
            else 0.0
        )
        return ap, ad

    def _usage_model_rows(label: str, wdata: Dict[str, Any]) -> str:
        llm = wdata.get("llm") or {}
        # Merge variants that normalize to the same canonical name
        # (e.g. ``Foo`` and ``Foo-GGUF``) so the breakdown shows one row.
        merged: Dict[str, Dict[str, float]] = {}
        for raw_name, mdata in llm.items():
            canon = _normalize_model_name(raw_name)
            agg = merged.setdefault(
                canon,
                {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_tps_sum": 0.0,
                    "prompt_tps_n": 0,
                    "predicted_tps_sum": 0.0,
                    "predicted_tps_n": 0,
                },
            )
            agg["requests"] += int(mdata.get("requests", 0) or 0)
            agg["prompt_tokens"] += int(mdata.get("prompt_tokens", 0) or 0)
            agg["completion_tokens"] += int(mdata.get("completion_tokens", 0) or 0)
            agg["prompt_tps_sum"] += float(mdata.get("prompt_tps_sum", 0.0) or 0.0)
            agg["prompt_tps_n"] += int(mdata.get("prompt_tps_n", 0) or 0)
            agg["predicted_tps_sum"] += float(
                mdata.get("predicted_tps_sum", 0.0) or 0.0
            )
            agg["predicted_tps_n"] += int(mdata.get("predicted_tps_n", 0) or 0)
        rows = ""
        for model, mdata in sorted(
            merged.items(), key=lambda kv: -kv[1].get("requests", 0)
        ):
            reqs = mdata.get("requests", 0)
            pt = mdata.get("prompt_tokens", 0)
            ct = mdata.get("completion_tokens", 0)
            # Prefer cumulative averages from _data; fall back to recent
            # history (covers entries recorded before the cumulative fields
            # were added).
            ptn = int(mdata.get("prompt_tps_n", 0) or 0)
            dtn = int(mdata.get("predicted_tps_n", 0) or 0)
            if ptn or dtn:
                avg_p = (float(mdata["prompt_tps_sum"]) / ptn) if ptn else 0.0
                avg_d = (float(mdata["predicted_tps_sum"]) / dtn) if dtn else 0.0
                if not avg_p or not avg_d:
                    hp, hd = _avg_tps((label, model))
                    avg_p = avg_p or hp
                    avg_d = avg_d or hd
            else:
                avg_p, avg_d = _avg_tps((label, model))
            rows += f"""
        <tr>
          <td class="muted small">{label}</td>
          <td class="mono small">{model}</td>
          <td class="num small">{reqs:,}</td>
          <td class="num small">{pt:,}</td>
          <td class="num small">{ct:,}</td>
          <td class="num small">{pt + ct:,}</td>
          <td class="num small">{_fmt_tps(avg_p)}</td>
          <td class="num small">{_fmt_tps(avg_d)}</td>
        </tr>"""
        return rows

    usage_summary_rows = (
        "".join(_usage_summary_row(lbl, wd) for lbl, wd in sorted(usage_data.items()))
        or '<tr><td colspan="10" class="muted">No usage recorded yet.</td></tr>'
    )

    usage_model_rows = (
        "".join(
            _usage_model_rows(lbl, wd)
            for lbl, wd in sorted(usage_data.items())
            if wd.get("llm")
        )
        or '<tr><td colspan="8" class="muted">No LLM usage recorded yet.</td></tr>'
    )

    # Recent requests table (newest first, capped for display)
    def _hist_row(h: Dict[str, Any]) -> str:
        ts = h.get("ts") or 0
        when = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "—"
        total_ms = float(h.get("total_ms") or 0.0)
        if total_ms > 0:
            total_cell = (
                f"{total_ms/1000.0:.2f}s"
                if total_ms < 60_000
                else f"{total_ms/60000.0:.2f}m"
            )
        else:
            total_cell = '<span class="muted">—</span>'
        return f"""
        <tr>
          <td class="muted small">{when}</td>
          <td class="small"><b>{h.get('worker', '—')}</b></td>
          <td class="mono small">{h.get('model', '—')}</td>
          <td class="num small">{int(h.get('prompt_tokens') or 0):,}</td>
          <td class="num small">{int(h.get('completion_tokens') or 0):,}</td>
          <td class="num small">{_fmt_tps(float(h.get('prompt_tps') or 0))}</td>
          <td class="num small">{_fmt_tps(float(h.get('predicted_tps') or 0))}</td>
          <td class="num small">{total_cell}</td>
        </tr>"""

    recent = list(reversed(history))[:50]
    history_rows = (
        "".join(_hist_row(h) for h in recent)
        or '<tr><td colspan="8" class="muted">No requests yet.</td></tr>'
    )

    # Recent errors section
    errors = data.get("errors") or []

    def _err_row(e: Dict[str, Any]) -> str:
        ts = e.get("ts") or 0
        when = datetime.utcfromtimestamp(ts).strftime("%H:%M:%S") if ts else "—"
        kind = (e.get("kind") or "").replace("_", " ")
        status = e.get("status")
        status_cell = f'<span class="mono small">{status}</span>' if status else ""
        msg = (e.get("message") or "").replace("<", "&lt;").replace(">", "&gt;")
        return f"""
        <tr>
          <td class="muted small">{when}</td>
          <td class="small"><b>{e.get('label', '—')}</b></td>
          <td class="small">{kind}</td>
          <td class="small">{status_cell}</td>
          <td class="mono small">{e.get('path', '—')}</td>
          <td class="mono small">{msg}</td>
        </tr>"""

    error_rows = (
        "".join(_err_row(e) for e in errors[:50])
        or '<tr><td colspan="6" class="muted">No errors recorded.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ezlocalai Router</title>
<style>
  :root {{
    --bg: #0f1419; --fg: #e6edf3; --muted: #8b949e; --card: #161b22;
    --border: #30363d; --accent: #58a6ff; --warn: #f0883e; --ok: #3fb950;
    --crit: #f85149;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--fg); font: 14px/1.45 -apple-system,
         BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0;
         padding: 24px; }}
  h1 {{ margin: 0 0 4px; font-size: 22px; }}
  h2 {{ margin: 32px 0 12px; font-size: 16px; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.05em; }}
  .muted {{ color: var(--muted); }}
  .small {{ font-size: 12px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  code {{ background: rgba(110,118,129,0.2); padding: 1px 6px; border-radius: 4px; }}
  .banner {{ padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; }}
  .banner.warn {{ background: rgba(240,136,62,0.15); border: 1px solid var(--warn); }}
  /* Layout */
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 12px; }}
  .card {{ background: var(--card); border: 1px solid var(--border);
           border-radius: 8px; padding: 14px 16px; }}
  /* Stat cards */
  .stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 28px; font-weight: 700; margin-top: 4px;
                  background: linear-gradient(135deg, var(--fg), var(--accent));
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                  background-clip: text; }}
  .stat .sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
  /* Capability cards */
  .cap .cap-header {{ margin-bottom: 8px; }}
  .cap .cap-stats {{ display: flex; gap: 14px; margin-top: 6px;
                     color: var(--muted); font-size: 12px; }}
  /* Tables */
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: rgba(110,118,129,0.1); font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--muted); }}
  tr:last-child td {{ border-bottom: none; }}
  tr.stale {{ opacity: 0.5; }}
  tr:not(.stale):hover {{ background: rgba(88,166,255,0.04); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  /* Capability pills */
  .pill {{ display: inline-block; padding: 2px 9px; margin: 2px;
           border-radius: 999px; font-size: 11px; border: 1px solid; }}
  .cap-text    {{ background: rgba(88,166,255,0.15);  border-color: rgba(88,166,255,0.45); }}
  .cap-vision  {{ background: rgba(160,110,255,0.15); border-color: rgba(160,110,255,0.45); }}
  .cap-image   {{ background: rgba(240,136,62,0.15);  border-color: rgba(240,136,62,0.45); }}
  .cap-tts     {{ background: rgba(63,185,80,0.15);   border-color: rgba(63,185,80,0.45); }}
  .cap-stt     {{ background: rgba(50,200,180,0.15);  border-color: rgba(50,200,180,0.45); }}
  .cap-video   {{ background: rgba(245,80,80,0.15);   border-color: rgba(245,80,80,0.45); }}
  .cap-embedding {{ background: rgba(130,130,130,0.15); border-color: rgba(130,130,130,0.45); }}
  /* Tier badges */
  .tier-gold  {{ color: #ffd700; font-weight: 600; }}
  .tier-blue  {{ color: #58a6ff; }}
  .tier-green {{ color: #3fb950; }}
  .tier-warn  {{ color: #f0883e; }}
  .tier-muted {{ color: #8b949e; }}
  /* Usage / slot bars */
  .bar {{ display: inline-block; width: 80px; height: 6px; margin-left: 6px;
          background: rgba(110,118,129,0.25); border-radius: 3px;
          vertical-align: middle; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--ok), var(--warn)); }}
  .usage-bar {{ width: 100%; height: 4px; background: rgba(110,118,129,0.2);
                border-radius: 2px; margin-top: 8px; overflow: hidden;
                display: block; }}
  .usage-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}
  .usage-fill.good {{ background: var(--ok); }}
  .usage-fill.warn {{ background: var(--warn); }}
  .usage-fill.crit {{ background: var(--crit); }}
  /* Capability card worker labels */
  .cap-workers {{ margin-top: 8px; }}
  .cap-worker {{ background: rgba(110,118,129,0.12) !important; border-color: rgba(110,118,129,0.3) !important;
                 font-size: 10px !important; }}
  /* Header */
  .header {{ display: flex; justify-content: space-between; align-items: baseline;
             flex-wrap: wrap; gap: 12px; }}
  .header .meta {{ color: var(--muted); font-size: 12px; }}
  /* Pool health badge */
  .health-ok   {{ font-size: 14px; font-weight: 600; color: var(--ok); }}
  .health-warn {{ font-size: 14px; font-weight: 600; color: var(--warn); }}
  .health-crit {{ font-size: 14px; font-weight: 600; color: var(--crit); }}
  /* Usage table */
  .tok-hi {{ color: var(--accent); font-weight: 600; }}
</style>
</head>
<body>
  <div id="dash">
  <div class="header">
    <div>
      <h1>ezlocalai router</h1>
      <div class="meta">{totals['alive_workers']} worker{'s' if totals['alive_workers'] != 1 else ''} online
      · TTL {router_meta['ttl_seconds']:.0f}s · wait timeout {router_meta['wait_timeout']:.0f}s
      · auto-refresh 5s</div>
    </div>
    <span class="{health_cls}">{health_label}</span>
  </div>

    <div class="grid">
    <div class="card stat"><div class="label">Workers</div>
      <div class="value">{totals['alive_workers']}</div>
      <div class="sub">{totals['stale_workers']} stale</div></div>
    <div class="card stat"><div class="label">Parallel capacity</div>
      <div class="value">{totals['total_parallel_capacity']}</div>
      <div class="sub">{totals['total_available_slots']} slots free now</div></div>
    <div class="card stat"><div class="label">In flight</div>
      <div class="value">{totals['total_in_flight']}</div>
      <div class="sub">{totals['total_queue_depth']} queued</div></div>
    <div class="card stat">
      <div class="label">Free VRAM</div>
      <div class="value">{totals['total_free_vram_gb']:.0f} GB</div>
      {_usage_bar(100 * (1 - totals['total_free_vram_gb'] / max(totals['total_vram_gb'], 0.1)))}
      <div class="sub">of {totals['total_vram_gb']:.0f} GB total</div></div>
    <div class="card stat"><div class="label">Unique models</div>
      <div class="value">{totals['unique_models']}</div>
      <div class="sub">across the pool</div></div>
  </div>

  <h2>Capabilities</h2>
  <div class="grid">{cap_cards}</div>

  <h2>Models</h2>
  <table>
    <thead><tr>
      <th>Model</th><th class="num">Workers</th><th class="num">Total parallel</th>
      <th class="num">Available now</th><th class="num">Max context</th>
      <th>Quant(s)</th><th class="num">Best tier</th><th>Served by</th>
    </tr></thead>
    <tbody>{model_rows}</tbody>
  </table>

  <h2>Workers</h2>
  <table>
    <thead><tr>
      <th>Label</th><th>Status</th><th>URL</th><th>GPUs</th>
      <th>Free VRAM</th><th>Slots free</th><th>Capabilities</th>
      <th>Models</th><th class="num">Last hb</th><th>Version</th><th>Errors</th>
    </tr></thead>
    <tbody>{worker_rows}</tbody>
  </table>

  <h2>Usage history</h2>
  <table>
    <thead><tr>
      <th>Worker</th>
      <th class="num">LLM reqs</th><th class="num">Prompt tok</th>
      <th class="num">Completion tok</th><th class="num">Total tokens</th>
      <th class="num">Embedding reqs</th>
      <th class="num">TTS reqs</th><th class="num">STT reqs</th>
      <th class="num">Image reqs</th><th class="num">Video reqs</th>
    </tr></thead>
    <tbody>{usage_summary_rows}</tbody>
  </table>

  <h2>LLM model breakdown</h2>
  <table>
    <thead><tr>
      <th>Worker</th><th>Model</th>
      <th class="num">Requests</th><th class="num">Prompt tokens</th>
      <th class="num">Completion tokens</th><th class="num">Total tokens</th>
      <th class="num">Avg prompt t/s</th><th class="num">Avg output t/s</th>
    </tr></thead>
    <tbody>{usage_model_rows}</tbody>
  </table>

  <h2>Recent requests</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Worker</th><th>Model</th>
      <th class="num">Prompt tok</th><th class="num">Output tok</th>
      <th class="num">Prompt t/s</th><th class="num">Output t/s</th>
      <th class="num">Total time</th>
    </tr></thead>
    <tbody>{history_rows}</tbody>
  </table>

  <h2>Recent errors</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Worker</th><th>Kind</th><th>Status</th><th>Path</th><th>Message</th>
    </tr></thead>
    <tbody>{error_rows}</tbody>
  </table>
  </div>
<script>
(function() {{
  // In-place dashboard refresh: fetch the same page, parse it, and swap the
  // contents of #dash. Keeps scroll position and avoids the white flash you
  // get from <meta refresh>.
  var INTERVAL_MS = 5000;
  var inFlight = false;
  function tick() {{
    if (inFlight || document.hidden) return;
    inFlight = true;
    fetch(window.location.pathname + '?_=' + Date.now(), {{
      cache: 'no-store',
      credentials: 'same-origin',
      headers: {{ 'X-Dashboard-Refresh': '1' }}
    }})
      .then(function(r) {{ return r.ok ? r.text() : null; }})
      .then(function(html) {{
        if (!html) return;
        var doc = new DOMParser().parseFromString(html, 'text/html');
        var fresh = doc.getElementById('dash');
        var current = document.getElementById('dash');
        if (fresh && current) current.innerHTML = fresh.innerHTML;
      }})
      .catch(function() {{}})
      .finally(function() {{ inFlight = false; }});
  }}
  setInterval(tick, INTERVAL_MS);
}})();
</script>
</body>
</html>"""


@app.get("/dashboard", tags=["Router"], response_class=HTMLResponse)
async def router_dashboard():
    """Human-friendly HTML dashboard. Auto-refreshes every 5 seconds.

    Intentionally unauthenticated so the router operator can drop it on a
    private network without juggling browser auth — gate it at the reverse
    proxy layer if you expose the router publicly.
    """
    return HTMLResponse(_render_dashboard_html(_aggregate_dashboard()))


# ---------------------------------------------------------------------------
# Worker selection helpers
# ---------------------------------------------------------------------------


def _wait_timeout() -> float:
    return float(getenv("ROUTER_WAIT_TIMEOUT", "120"))


async def _pick(
    capability: str,
    model: Optional[str] = None,
    exclude: Optional[set] = None,
) -> WorkerInfo:
    router = get_router()
    # Pre-exclude tunneled workers whose WebSocket is not currently connected.
    # Without this the selector happily hands out a tunnel worker, the proxy
    # then 503s with "Tunnel ... is not connected", and we burn a retry.
    hub = get_tunnel_hub()
    registry = get_registry()
    pre_exclude = set(exclude or ())
    unavailable: List[str] = []
    for w in registry.list_workers(alive_only=True):
        if is_tunnel_url(w.url):
            wid = worker_id_from_tunnel_url(w.url)
            if not hub.is_connected(wid):
                pre_exclude.add(w.worker_id)
                unavailable.append(w.label)
    if unavailable:
        logging.info(
            f"[Router] tunnel offline, excluding from selection: {', '.join(unavailable)}"
        )
    worker = await router.wait_for_worker(
        capability, model, timeout=_wait_timeout(), exclude=pre_exclude
    )
    if worker is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No worker available for capability={capability!r}"
                + (f" model={model!r}" if model else "")
            ),
        )
    return worker


def _max_retries() -> int:
    """Number of additional workers to try on transient failure (default 2)."""
    try:
        return max(0, int(getenv("ROUTER_MAX_RETRIES", "2")))
    except (TypeError, ValueError):
        return 2


def _is_transient_failure(status: int) -> bool:
    """5xx and 408 are worth retrying on a different worker; 4xx are client errors."""
    return status == 408 or 500 <= status <= 599


def _worker_headers(worker: WorkerInfo) -> Dict[str, str]:
    h = {}
    if worker.api_key and worker.api_key != "none":
        h["Authorization"] = f"Bearer {worker.api_key}"
    return h


# ---------------------------------------------------------------------------
# Generic proxy helpers
# ---------------------------------------------------------------------------


async def _proxy_via_tunnel(
    worker: WorkerInfo,
    method: str,
    path: str,
    *,
    headers: Dict[str, str],
    body: Optional[bytes],
    stream: bool,
    timeout: Optional[float],
    capability: Optional[str] = None,
    model: Optional[str] = None,
):
    """Route a request to a tunneled worker through its open WebSocket."""
    hub = get_tunnel_hub()
    wid = worker_id_from_tunnel_url(worker.url)
    conn = hub.get(wid)
    if conn is None or conn.closed:
        try:
            get_registry().record_error(
                worker.worker_id,
                kind="tunnel_disconnect",
                path=path,
                message=f"Tunnel {wid} not connected",
                status=503,
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=503,
            detail=f"Tunnel for worker {worker.label} ({wid}) is not connected",
        )
    registry = get_registry()
    registry.increment_in_flight(
        worker.worker_id, 1, capability=capability, model=model
    )
    request_timeout = timeout or float(getenv("REQUEST_TIMEOUT", "300"))
    try:
        status, resp_headers, chunks = await conn.request(
            method,
            path,
            headers=headers,
            body=body,
            stream=stream,
            timeout=request_timeout,
        )
    except Exception:
        registry.increment_in_flight(
            worker.worker_id, -1, capability=capability, model=model
        )
        raise

    media_type = resp_headers.get("Content-Type") or resp_headers.get("content-type")
    if not stream:
        try:
            buf = bytearray()
            async for c in chunks:
                buf.extend(c)
            return Response(
                content=bytes(buf),
                status_code=status,
                media_type=media_type or "application/json",
            )
        finally:
            registry.increment_in_flight(
                worker.worker_id, -1, capability=capability, model=model
            )

    async def gen():
        try:
            async for c in chunks:
                yield c
        finally:
            registry.increment_in_flight(
                worker.worker_id, -1, capability=capability, model=model
            )

    return StreamingResponse(gen(), media_type=media_type or "text/event-stream")


async def _proxy_json(
    worker: WorkerInfo,
    path: str,
    payload: Dict[str, Any],
    *,
    stream: bool = False,
    timeout: Optional[float] = None,
    capability: Optional[str] = None,
    model: Optional[str] = None,
):
    """Forward a JSON POST to a worker. Returns either a dict or a StreamingResponse."""
    headers = {"Content-Type": "application/json", **_worker_headers(worker)}
    if is_tunnel_url(worker.url):
        return await _proxy_via_tunnel(
            worker,
            "POST",
            path,
            headers=headers,
            body=json.dumps(payload).encode("utf-8"),
            stream=stream,
            timeout=timeout,
            capability=capability,
            model=model,
        )
    url = f"{worker.url}{path}"
    request_timeout = aiohttp.ClientTimeout(
        total=timeout or float(getenv("REQUEST_TIMEOUT", "300")),
        connect=10,  # fail fast on connection errors, don't wait the full timeout
    )
    registry = get_registry()
    registry.increment_in_flight(
        worker.worker_id, 1, capability=capability, model=model
    )

    if not stream:
        try:
            async with aiohttp.ClientSession(timeout=request_timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    body = await resp.read()
                    media_type = resp.headers.get("Content-Type", "application/json")
                    return Response(
                        content=body, status_code=resp.status, media_type=media_type
                    )
        except Exception as e:
            logging.warning(f"[Router] POST {worker.url}{path} failed: {e}")
            registry.record_connection_failure(worker.worker_id)
            registry.record_error(
                worker.worker_id,
                kind="connection",
                path=path,
                message=f"{type(e).__name__}: {e}",
            )
            raise
        finally:
            registry.increment_in_flight(
                worker.worker_id, -1, capability=capability, model=model
            )

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
            registry.record_connection_failure(worker.worker_id)
            registry.record_error(
                worker.worker_id,
                kind="stream",
                path=path,
                message=f"{type(e).__name__}: {e}",
            )
        finally:
            await session.close()
            registry.increment_in_flight(
                worker.worker_id, -1, capability=capability, model=model
            )

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _proxy_get(worker: WorkerInfo, path: str) -> Response:
    headers = _worker_headers(worker)
    if is_tunnel_url(worker.url):
        return await _proxy_via_tunnel(
            worker, "GET", path, headers=headers, body=None, stream=False, timeout=30
        )
    url = f"{worker.url}{path}"
    request_timeout = aiohttp.ClientTimeout(total=30, connect=10)
    try:
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.get(url, headers=headers) as resp:
                body = await resp.read()
                media_type = resp.headers.get("Content-Type", "application/json")
                return Response(
                    content=body, status_code=resp.status, media_type=media_type
                )
    except Exception as e:
        logging.warning(f"[Router] GET {worker.url}{path} failed: {e}")
        registry = get_registry()
        registry.record_connection_failure(worker.worker_id)
        registry.record_error(
            worker.worker_id,
            kind="connection",
            path=path,
            message=f"{type(e).__name__}: {e}",
        )
        raise


# Matches absolute URLs that point at a worker's /outputs/<file> static mount.
# Filenames are restricted to a safe character set to prevent path traversal
# when we save them locally.
_ASSET_URL_RE = re.compile(
    r'(https?://[^\s"\'<>]+?/outputs/[A-Za-z0-9._/-]+)',
    re.IGNORECASE,
)


def _safe_outputs_path(rel_path: str) -> Optional[str]:
    """Resolve ``outputs/<rel>`` and reject anything escaping the outputs dir."""
    rel_path = rel_path.lstrip("/")
    if not rel_path.startswith("outputs/"):
        return None
    target = os.path.abspath(os.path.join(_OUTPUTS_DIR, rel_path[len("outputs/") :]))
    if not target.startswith(_OUTPUTS_DIR + os.sep) and target != _OUTPUTS_DIR:
        return None
    return target


async def _fetch_asset(worker: WorkerInfo, path: str) -> Optional[bytes]:
    """Download a single asset from a worker (tunnel or direct). Returns
    ``None`` on any error so the caller can fall back to the original URL."""
    headers = _worker_headers(worker)
    try:
        if is_tunnel_url(worker.url):
            resp = await _proxy_via_tunnel(
                worker,
                "GET",
                path,
                headers=headers,
                body=None,
                stream=False,
                timeout=120,
            )
            status = getattr(resp, "status_code", 200)
            if status != 200:
                logging.warning(
                    f"[Router] Asset fetch {worker.label}{path} -> HTTP {status}"
                )
                return None
            return bytes(getattr(resp, "body", b""))
        url = f"{worker.url}{path}"
        request_timeout = aiohttp.ClientTimeout(total=120, connect=10)
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.get(url, headers=headers) as resp_g:
                if resp_g.status != 200:
                    logging.warning(
                        f"[Router] Asset fetch {url} -> HTTP {resp_g.status}"
                    )
                    return None
                return await resp_g.read()
    except Exception as e:
        logging.warning(f"[Router] Asset fetch from {worker.label} failed: {e}")
        return None


async def _persist_response_assets(
    worker: WorkerInfo,
    resp: Response,
    request: Optional[Request] = None,
) -> Response:
    """Scan a JSON response body for worker asset URLs (``.../outputs/<file>``),
    download each one, save it to the router's local ``outputs/`` directory,
    and rewrite the URL to point at the router's own ``/outputs/<file>`` mount.

    Returns the (possibly modified) response. Non-JSON or empty responses
    pass through untouched.
    """
    if os.environ.get("ROUTER_PERSIST_ASSETS", "true").lower() in ("0", "false", "no"):
        return resp
    body = getattr(resp, "body", None)
    if not body:
        return resp
    try:
        text = body.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return resp
    if "/outputs/" not in text:
        return resp
    urls = list({m for m in _ASSET_URL_RE.findall(text)})
    if not urls:
        return resp

    # Build the public base URL for rewrites — prefer the request's own
    # base URL so links work behind reverse proxies; fall back to a
    # configured ROUTER_PUBLIC_URL or relative paths.
    base = ""
    if request is not None:
        base = str(request.base_url).rstrip("/")
    if not base:
        base = (os.environ.get("ROUTER_PUBLIC_URL") or "").rstrip("/")

    rewrites: Dict[str, str] = {}
    for url in urls:
        idx = url.lower().find("/outputs/")
        if idx < 0:
            continue
        worker_path = url[idx:]  # /outputs/<rest>
        local_target = _safe_outputs_path("outputs" + worker_path[len("/outputs") :])
        if local_target is None:
            logging.warning(f"[Router] Refusing unsafe asset path: {url}")
            continue
        # If we already have it cached, just rewrite without re-downloading.
        if not os.path.exists(local_target):
            content = await _fetch_asset(worker, worker_path)
            if content is None:
                continue
            try:
                os.makedirs(os.path.dirname(local_target), exist_ok=True)
                with open(local_target, "wb") as fh:
                    fh.write(content)
            except OSError as e:
                logging.warning(f"[Router] Failed to write {local_target}: {e}")
                continue
        rewrites[url] = f"{base}{worker_path}" if base else worker_path

    if not rewrites:
        return resp
    for old, new in rewrites.items():
        text = text.replace(old, new)
    new_body = text.encode("utf-8")
    media_type = (
        getattr(resp, "media_type", None)
        or resp.headers.get("Content-Type")
        or "application/json"
    )
    return Response(
        content=new_body, status_code=resp.status_code, media_type=media_type
    )


async def _proxy_multipart(
    worker: WorkerInfo,
    path: str,
    *,
    files: Dict[str, tuple],
    fields: Dict[str, Any],
    timeout: Optional[float] = None,
    capability: Optional[str] = None,
    model: Optional[str] = None,
) -> Response:
    """Forward a multipart/form-data POST to a worker.

    ``files`` is ``{field_name: (filename, content_bytes, content_type)}``.
    """
    headers = _worker_headers(worker)
    request_timeout = aiohttp.ClientTimeout(
        total=timeout or float(getenv("REQUEST_TIMEOUT", "300")),
        connect=10,
    )
    if is_tunnel_url(worker.url):
        # Build the multipart body with stdlib so we can ship it as a single
        # binary payload through the tunnel (no streaming MultipartWriter quirks).
        import secrets

        boundary = "----ezlocalai" + secrets.token_hex(12)
        parts: List[bytes] = []
        for k, v in fields.items():
            if v is None:
                continue
            values = v if isinstance(v, (list, tuple, set)) else [v]
            for item in values:
                if item is None:
                    continue
                parts.append(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
                )
                parts.append(str(item).encode("utf-8"))
                parts.append(b"\r\n")
        for name, (fname, content, ctype) in files.items():
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
                    f"Content-Type: {ctype}\r\n\r\n"
                ).encode()
            )
            parts.append(content)
            parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        body_bytes = b"".join(parts)
        mp_headers = dict(headers)
        mp_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        return await _proxy_via_tunnel(
            worker,
            "POST",
            path,
            headers=mp_headers,
            body=body_bytes,
            stream=False,
            timeout=timeout,
            capability=capability,
            model=model,
        )

    # Direct (non-tunneled) multipart upload via aiohttp.
    data = aiohttp.FormData()
    for name, (fname, content, ctype) in files.items():
        data.add_field(name, content, filename=fname, content_type=ctype)
    for k, v in fields.items():
        if v is None:
            continue
        values = v if isinstance(v, (list, tuple, set)) else [v]
        for item in values:
            if item is not None:
                data.add_field(k, str(item))

    url = f"{worker.url}{path}"
    registry = get_registry()
    registry.increment_in_flight(
        worker.worker_id, 1, capability=capability, model=model
    )
    try:
        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.post(url, data=data, headers=headers) as resp:
                body = await resp.read()
                media_type = resp.headers.get("Content-Type", "application/json")
                return Response(
                    content=body, status_code=resp.status, media_type=media_type
                )
    except Exception as e:
        logging.warning(f"[Router] Multipart POST {worker.url}{path} failed: {e}")
        registry.record_connection_failure(worker.worker_id)
        registry.record_error(
            worker.worker_id,
            kind="connection",
            path=path,
            message=f"{type(e).__name__}: {e}",
        )
        raise
    finally:
        registry.increment_in_flight(
            worker.worker_id, -1, capability=capability, model=model
        )


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
    model = payload.get("model") or "unknown"
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
    is_stream = bool(payload.get("stream"))
    return await _llm_proxy_with_retry(
        capability=capability,
        path="/v1/chat/completions",
        payload=payload,
        model=model,
        is_stream=is_stream,
    )


@app.post("/v1/completions", tags=["Completions"])
async def completions(payload: Dict[str, Any], _: str = Depends(verify_client)):
    model = payload.get("model") or "unknown"
    is_stream = bool(payload.get("stream"))
    return await _llm_proxy_with_retry(
        capability="text",
        path="/v1/completions",
        payload=payload,
        model=model,
        is_stream=is_stream,
    )


async def _llm_proxy_with_retry(
    *,
    capability: str,
    path: str,
    payload: Dict[str, Any],
    model: str,
    is_stream: bool,
):
    """Forward an LLM request, retrying on a different worker if the first
    pick fails connection-level or returns a transient (5xx) status. Once a
    streaming worker has started writing bytes back to the client we cannot
    safely retry — at that point we just record the partial usage and let
    the client see whatever the worker produced.
    """
    tried: set = set()
    last_error: Optional[Exception] = None
    last_status: Optional[int] = None
    max_attempts = 1 + _max_retries()
    request_started = time.monotonic()
    for attempt in range(max_attempts):
        worker = await _pick(capability, model, exclude=tried)
        tried.add(worker.worker_id)
        try:
            resp = await _proxy_json(
                worker,
                path,
                payload,
                stream=is_stream,
                capability=capability,
                model=model,
            )
        except Exception as e:
            last_error = e
            logging.warning(
                f"[Router] {path} attempt {attempt + 1}/{max_attempts} via "
                f"{worker.label} raised {type(e).__name__}: {e}"
            )
            continue

        if not is_stream:
            status = getattr(resp, "status_code", 200)
            if _is_transient_failure(status) and attempt + 1 < max_attempts:
                last_status = status
                logging.warning(
                    f"[Router] {path} attempt {attempt + 1}/{max_attempts} via "
                    f"{worker.label} returned {status}; retrying on next worker"
                )
                try:
                    body_snippet = (
                        resp.body.decode("utf-8", errors="replace")[:500]
                        if hasattr(resp, "body")
                        else ""
                    )
                except Exception:
                    body_snippet = ""
                get_registry().record_error(
                    worker.worker_id,
                    kind="http_5xx",
                    path=path,
                    message=body_snippet or f"HTTP {status}",
                    status=status,
                )
                continue
            # Final non-streaming response — extract tokens & timings if any
            pt, ct = 0, 0
            timings: Dict[str, float] = {}
            try:
                obj = json.loads(resp.body)
                u = obj.get("usage") or {}
                pt = int(u.get("prompt_tokens") or 0)
                ct = int(u.get("completion_tokens") or 0)
                t = obj.get("timings")
                if isinstance(t, dict):
                    if not pt:
                        pt = int(t.get("prompt_n") or 0)
                    if not ct:
                        ct = int(t.get("predicted_n") or 0)
                    for k in ("prompt_ms", "predicted_ms"):
                        v = t.get(k)
                        if isinstance(v, (int, float)):
                            timings[k] = float(v)
            except Exception:
                pass
            timings["total_ms"] = (time.monotonic() - request_started) * 1000.0
            await _usage.record_llm(worker.label, model, pt, ct, timings)
            return resp

        # Streaming: we cannot retry once bytes are flowing.  Wrap the
        # generator so token + timing extraction still happens.
        _wlabel, _model = worker.label, model

        async def _record(pt: int, ct: int, timings: Dict[str, float]):
            timings = dict(timings or {})
            timings["total_ms"] = (time.monotonic() - request_started) * 1000.0
            await _usage.record_llm(_wlabel, _model, pt, ct, timings)

        return StreamingResponse(
            _stream_with_token_extraction(resp.body_iterator, _record),
            media_type=resp.media_type,
        )

    # Exhausted retries
    if last_error is not None:
        raise HTTPException(
            status_code=502,
            detail=(
                f"All {max_attempts} worker attempts failed for {path}: "
                f"{type(last_error).__name__}: {last_error}"
            ),
        )
    raise HTTPException(
        status_code=502,
        detail=(
            f"All {max_attempts} worker attempts returned transient errors "
            f"(last status {last_status}) for {path}"
        ),
    )


@app.post("/v1/embeddings", tags=["Embeddings"])
async def embeddings(payload: Dict[str, Any], _: str = Depends(verify_client)):
    # Embeddings count as text capability for now (most ezlocalai workers serve both)
    worker = (
        await _pick("embedding", payload.get("model"))
        if any("embedding" in w.capabilities for w in get_registry().list_workers())
        else await _pick("text", payload.get("model"))
    )
    resp = await _proxy_json(
        worker,
        "/v1/embeddings",
        payload,
        stream=False,
        capability="embedding" if "embedding" in worker.capabilities else "text",
        model=payload.get("model"),
    )
    await _usage.record_cap(worker.label, "embedding")
    return resp


@app.post("/v1/audio/speech", tags=["Audio"])
async def audio_speech(
    payload: Dict[str, Any],
    request: Request,
    _: str = Depends(verify_client),
):
    worker = await _pick("tts", payload.get("model"))
    resp = await _proxy_json(
        worker, "/v1/audio/speech", payload, stream=False, capability="tts"
    )
    await _usage.record_cap(worker.label, "tts")
    return await _persist_response_assets(worker, resp, request)


@app.post("/v1/audio/speech/stream", tags=["Audio"])
async def audio_speech_stream(payload: Dict[str, Any], _: str = Depends(verify_client)):
    worker = await _pick("tts", payload.get("model"))
    resp = await _proxy_json(
        worker, "/v1/audio/speech/stream", payload, stream=True, capability="tts"
    )
    await _usage.record_cap(worker.label, "tts")
    return resp


@app.get("/v1/audio/voices", tags=["Audio"])
async def audio_voices(_: str = Depends(verify_client)):
    # Aggregate voices from any tts-capable worker
    voices: List[Dict[str, Any]] = []
    seen = set()
    for w in get_registry().list_workers(alive_only=True):
        if "tts" not in w.capabilities:
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
    timestamp_granularities: Optional[List[str]] = Form(None),
    timestamp_granularities_bracketed: Optional[List[str]] = Form(
        None, alias="timestamp_granularities[]"
    ),
    _: str = Depends(verify_client),
):
    worker = await _pick("stt", model)
    content = await file.read()
    granularities = timestamp_granularities_bracketed or timestamp_granularities
    resp = await _proxy_multipart(
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
            "timestamp_granularities[]": granularities,
        },
        capability="stt",
        model=model,
    )
    await _usage.record_cap(worker.label, "stt")
    return resp


@app.post("/v1/images/generations", tags=["Images"])
async def images_generations(
    payload: Dict[str, Any],
    request: Request,
    _: str = Depends(verify_client),
):
    worker = await _pick("image", payload.get("model"))
    resp = await _proxy_json(
        worker, "/v1/images/generations", payload, stream=False, capability="image"
    )
    await _usage.record_cap(worker.label, "image")
    return await _persist_response_assets(worker, resp, request)


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
    resp = await _proxy_multipart(
        worker,
        "/v1/images/edits",
        files=files,
        fields=fields,
        capability="image",
    )
    await _usage.record_cap(worker.label, "image")
    return await _persist_response_assets(worker, resp, request)


@app.post("/v1/videos/generations", tags=["Videos"])
async def videos_generations(
    payload: Dict[str, Any],
    request: Request,
    _: str = Depends(verify_client),
):
    worker = await _pick("video", payload.get("model"))
    resp = await _proxy_json(
        worker,
        "/v1/videos/generations",
        payload,
        stream=False,
        timeout=float(getenv("REQUEST_TIMEOUT", "1800")),
        capability="video",
    )
    await _usage.record_cap(worker.label, "video")
    return await _persist_response_assets(worker, resp, request)
