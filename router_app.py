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
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

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
        total_ram_gb=float(payload.get("total_ram_gb", 0.0) or 0.0),
        queue_depth=int(payload.get("queue_depth", 0) or 0),
        queue_capacity=int(payload.get("queue_capacity", 1) or 1),
        in_flight=int(payload.get("in_flight", 0) or 0),
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
                "gpus",
                "best_tier",
                "model_context",
                "model_quant",
                "cap_models",
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


def _cap_capacity(cap: str, worker_capacity: int) -> int:
    """Return the realistic concurrency for a given capability on one worker.

    image, stt, video: sequential by nature — always 1 slot.
    tts: fast and low-resource but we have no separate queue depth for it,
         so report 1 conservative slot (will not starve the text queue).
    text, vision, embedding: use the reported N_PARALLEL-sized queue capacity.
    """
    if cap in ("image", "stt", "video", "tts"):
        return 1
    return max(1, worker_capacity)


def _aggregate_dashboard() -> Dict[str, Any]:
    """Build a JSON-serialisable summary of the entire pool."""
    registry = get_registry()
    alive = registry.list_workers(alive_only=True)
    stale = [w for w in registry.list_workers(alive_only=False) if w not in alive]

    total_capacity = sum(max(1, w.queue_capacity) for w in alive)
    total_in_flight = sum(max(0, w.in_flight) for w in alive)
    total_queue_depth = sum(max(0, w.queue_depth) for w in alive)
    total_slots_left = sum(max(0, w.queue_capacity - w.queue_depth) for w in alive)
    total_free_vram = sum(w.free_vram_gb for w in alive)
    total_vram = sum(w.total_vram_gb for w in alive)

    # Per-model rollup: which workers serve it, total parallel slots, max ctx
    model_rollup: Dict[str, Dict[str, Any]] = {}
    for w in alive:
        slots_left = max(0, w.queue_capacity - w.queue_depth)
        for model in w.models:
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
            entry["total_capacity"] += max(1, w.queue_capacity)
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
                    "queue_capacity": w.queue_capacity,
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
            entry["total_capacity"] += _cap_capacity(cap, w.queue_capacity)
            entry["available_slots"] += _cap_capacity(cap, w.queue_capacity)
            if w.best_tier > entry["best_tier"]:
                entry["best_tier"] = w.best_tier
            entry["workers"].append(
                {
                    "label": w.label,
                    "worker_id": w.worker_id,
                    "best_tier": w.best_tier,
                    "context": 0,
                    "slots_left": _cap_capacity(cap, w.queue_capacity),
                    "queue_capacity": _cap_capacity(cap, w.queue_capacity),
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
            cap_slots = _cap_capacity(
                "text" if merged_cap == "text+vision" else merged_cap, w.queue_capacity
            )
            cap_in_flight = (
                w.in_flight if merged_cap in ("text", "text+vision", "embedding") else 0
            )
            cap_slots_left = max(0, cap_slots - cap_in_flight)
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
            [w.to_public() for w in alive], key=lambda x: -x.get("best_tier", 0)
        )
        + sorted(
            [{**w.to_public(), "stale": True} for w in stale],
            key=lambda x: -x.get("best_tier", 0),
        ),
    }


@app.get("/v1/router/dashboard", tags=["Router"])
async def router_dashboard_json(_: str = Depends(verify_client)):
    """JSON form of the dashboard data."""
    return _aggregate_dashboard()


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
        slots_left = max(
            0, int(w.get("queue_capacity", 1)) - int(w.get("queue_depth", 0))
        )
        slots_total = int(w.get("queue_capacity", 1))
        slot_pct = 0 if slots_total == 0 else (1 - slots_left / slots_total) * 100
        bar = f'<div class="bar"><div class="bar-fill" style="width:{slot_pct:.0f}%"></div></div>'
        last_hb = w.get("last_heartbeat_age", 0)
        status = "🔴 stale" if stale else ("🟢 ready" if slots_left > 0 else "🟡 full")
        raw_caps = w.get("capabilities") or []
        # Merge text + vision into a single "text+vision" pill on the worker row
        if "text" in raw_caps and "vision" in raw_caps:
            display_caps = [
                "text+vision" if c == "text" else c for c in raw_caps if c != "vision"
            ]
        else:
            display_caps = raw_caps
        cap_pills = " ".join(_cap_pill(c) for c in display_caps) or "—"
        return f"""
        <tr class="{'stale' if stale else ''}">
          <td><b>{w['label']}</b><div class="muted small">{w['worker_id']}</div></td>
          <td>{status}</td>
          <td class="mono small">{w['url']}</td>
          <td>{gpus}<div class="muted small">{_tier_badge(w.get('best_tier', 0))}</div></td>
          <td class="num small">{mem_cell}</td>
          <td>{slots_left}/{slots_total} {bar}<div class="muted small">{w.get('in_flight', 0)} in flight</div></td>
          <td class="small">{cap_pills}</td>
          <td class="small mono">{models}</td>
          <td class="num small">{last_hb:.0f}s</td>
        </tr>
        """

    worker_rows = "".join(_worker_row(w) for w in data["workers"]) or (
        '<tr><td colspan="9" class="muted">No workers registered.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ezlocalai Router</title>
<meta http-equiv="refresh" content="5" />
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
</style>
</head>
<body>
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
      <th>Models</th><th class="num">Last hb</th>
    </tr></thead>
    <tbody>{worker_rows}</tbody>
  </table>
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


async def _proxy_via_tunnel(
    worker: WorkerInfo,
    method: str,
    path: str,
    *,
    headers: Dict[str, str],
    body: Optional[bytes],
    stream: bool,
    timeout: Optional[float],
):
    """Route a request to a tunneled worker through its open WebSocket."""
    hub = get_tunnel_hub()
    wid = worker_id_from_tunnel_url(worker.url)
    conn = hub.get(wid)
    if conn is None or conn.closed:
        raise HTTPException(
            status_code=503,
            detail=f"Tunnel for worker {worker.label} ({wid}) is not connected",
        )
    registry = get_registry()
    registry.increment_in_flight(worker.worker_id, 1)
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
        registry.increment_in_flight(worker.worker_id, -1)
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
            registry.increment_in_flight(worker.worker_id, -1)

    async def gen():
        try:
            async for c in chunks:
                yield c
        finally:
            registry.increment_in_flight(worker.worker_id, -1)

    return StreamingResponse(gen(), media_type=media_type or "text/event-stream")


async def _proxy_json(
    worker: WorkerInfo,
    path: str,
    payload: Dict[str, Any],
    *,
    stream: bool = False,
    timeout: Optional[float] = None,
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
        )
    url = f"{worker.url}{path}"
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
    headers = _worker_headers(worker)
    if is_tunnel_url(worker.url):
        return await _proxy_via_tunnel(
            worker, "GET", path, headers=headers, body=None, stream=False, timeout=30
        )
    url = f"{worker.url}{path}"
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
    headers = _worker_headers(worker)
    request_timeout = aiohttp.ClientTimeout(
        total=timeout or float(getenv("REQUEST_TIMEOUT", "300"))
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
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
            )
            parts.append(str(v).encode("utf-8"))
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
        )

    # Direct (non-tunneled) multipart upload via aiohttp.
    data = aiohttp.FormData()
    for name, (fname, content, ctype) in files.items():
        data.add_field(name, content, filename=fname, content_type=ctype)
    for k, v in fields.items():
        if v is not None:
            data.add_field(k, str(v))

    url = f"{worker.url}{path}"
    registry = get_registry()
    registry.increment_in_flight(worker.worker_id, 1)
    try:
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
    worker = await _pick("tts", payload.get("model"))
    return await _proxy_json(worker, "/v1/audio/speech", payload, stream=False)


@app.post("/v1/audio/speech/stream", tags=["Audio"])
async def audio_speech_stream(payload: Dict[str, Any], _: str = Depends(verify_client)):
    worker = await _pick("tts", payload.get("model"))
    return await _proxy_json(worker, "/v1/audio/speech/stream", payload, stream=True)


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
    _: str = Depends(verify_client),
):
    worker = await _pick("stt", model)
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
