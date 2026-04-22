"""Router/load-balancer support for ezlocalai.

Two pieces live here:

1. ``WorkerRegistry`` + ``Router`` — used by the router process to keep track of
   worker ezlocalai instances that have registered themselves and to pick the
   best worker for a given request.

2. ``WorkerHeartbeatClient`` — used by every worker ezlocalai instance to
   register and heartbeat with a router (when ``ROUTER_URL`` is set).

The router exposes the same OpenAI-compatible API surface as a normal
ezlocalai server but does no inference itself. It selects a worker based on:

* Capability match (text / vision / voice / image / video / embedding)
* Whether the worker has the requested model
* Free VRAM and queue depth (more free, less queued = better score)
* Liveness (recent heartbeat)

The worker → router protocol is intentionally tiny: a ``register`` on startup,
periodic ``heartbeat`` POSTs, and a best-effort ``deregister`` on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from Globals import getenv


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

ALL_CAPABILITIES = {"text", "vision", "voice", "image", "video", "embedding"}


def detect_local_capabilities() -> List[str]:
    """Best-effort guess at what this ezlocalai instance can serve.

    Reads the same env vars the rest of the app uses so a worker advertises
    itself accurately based on its existing config (no extra env required).
    """
    caps: List[str] = []
    default_model = (getenv("DEFAULT_MODEL") or "").strip()
    voice_server = (getenv("VOICE_SERVER") or "").strip().lower()
    image_server = (getenv("IMAGE_SERVER") or "").strip().lower()
    text_server = (getenv("TEXT_SERVER") or "").strip().lower()
    img_model = (getenv("IMG_MODEL") or "").strip().lower()
    video_model = (getenv("VIDEO_MODEL") or "").strip().lower()
    tts_enabled = (getenv("TTS_ENABLED") or "true").strip().lower() == "true"
    stt_enabled = (getenv("STT_ENABLED") or "true").strip().lower() == "true"

    # Text/vision: any server that loads an LLM (and isn't dedicated to
    # voice/image only) can answer text. Vision is detected from common model
    # name hints.
    is_dedicated_voice = voice_server == "true"
    is_dedicated_image = image_server == "true"
    if default_model and not (is_dedicated_voice or is_dedicated_image):
        caps.append("text")
        lowered = default_model.lower()
        if any(tag in lowered for tag in ("vl", "vision", "qwen3.6", "qwen3.5-vl")):
            caps.append("vision")
    if (
        (
            text_server == "true"
            or (text_server == "" and not is_dedicated_voice and not is_dedicated_image)
        )
        and "text" not in caps
        and default_model
    ):
        caps.append("text")

    if tts_enabled or stt_enabled or voice_server == "true":
        caps.append("voice")

    if (img_model and img_model not in ("none", "")) or image_server == "true":
        caps.append("image")
    if video_model and video_model not in ("none", ""):
        caps.append("video")

    # De-dup, preserve order
    seen = set()
    deduped = []
    for c in caps:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Hardware tier scoring
# ---------------------------------------------------------------------------
# Higher tier = faster inference. Used to break ties when load is similar so
# requests prefer the 5090 over the 4090 over the 3090, etc. Unrecognized GPUs
# get TIER_DEFAULT_GPU; CPU-only workers get TIER_CPU.
# Substring match (case-insensitive) on torch's get_device_name() output.
GPU_TIERS: List[Tuple[str, int]] = [
    # Datacenter / Hopper / Blackwell
    ("h200", 100),
    ("h100", 95),
    ("b200", 110),
    ("a100", 80),
    ("a6000", 70),
    ("a40", 65),
    ("l40", 70),
    # RTX 50-series (Blackwell)
    ("5090", 90),
    ("5080", 75),
    ("5070 ti", 60),
    ("5070", 55),
    ("5060", 40),
    # RTX 40-series (Ada Lovelace)
    ("4090", 80),
    ("4080 super", 65),
    ("4080", 60),
    ("4070 ti super", 55),
    ("4070 ti", 50),
    ("4070", 45),
    ("4060 ti", 35),
    ("4060", 30),
    # RTX 30-series (Ampere)
    ("3090 ti", 55),
    ("3090", 50),
    ("3080 ti", 45),
    ("3080", 40),
    ("3070 ti", 32),
    ("3070", 30),
    ("3060 ti", 25),
    ("3060", 22),
    # RTX 20-series (Turing)
    ("2080 ti", 28),
    ("2080", 22),
    ("2070", 18),
    ("2060", 15),
    # GTX
    ("1080 ti", 14),
    ("1080", 12),
    ("1070", 10),
    # Jetson
    ("orin", 18),
    ("xavier", 8),
    # AMD
    ("mi300", 80),
    ("mi250", 65),
    ("7900", 50),
    # Apple Silicon
    ("m3 max", 35),
    ("m2 max", 30),
    ("m1 max", 25),
    ("m3 ultra", 45),
    ("m2 ultra", 40),
]
TIER_DEFAULT_GPU = 20
TIER_CPU = 1


def gpu_tier_for_name(name: str) -> int:
    """Look up a hardware tier score from a GPU model name string."""
    if not name:
        return TIER_CPU
    lname = name.lower()
    for needle, tier in GPU_TIERS:
        if needle in lname:
            return tier
    return TIER_DEFAULT_GPU


def detect_local_gpus() -> List[Dict[str, Any]]:
    """Best-effort enumeration of local CUDA devices with name + total VRAM.

    Returns a list like
    ``[{"index": 0, "name": "NVIDIA GeForce RTX 5090", "total_vram_gb": 32.0,
        "tier": 90}, ...]``
    Empty list when no CUDA is available.
    """
    gpus: List[Dict[str, Any]] = []
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return []
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                name = props.name
                total_gb = float(props.total_memory) / (1024**3)
            except Exception:
                name = f"cuda:{i}"
                total_gb = 0.0
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "total_vram_gb": round(total_gb, 2),
                    "tier": gpu_tier_for_name(name),
                }
            )
    except Exception as e:  # pragma: no cover - best effort
        logging.debug(f"[Router] GPU enumeration failed: {e}")
    return gpus


def best_gpu_tier(gpus: List[Dict[str, Any]]) -> int:
    """The fastest GPU's tier on this worker. Falls back to CPU tier."""
    if not gpus:
        return TIER_CPU
    return max(int(g.get("tier", TIER_DEFAULT_GPU)) for g in gpus)


# ---------------------------------------------------------------------------
# Worker registry (router-side)
# ---------------------------------------------------------------------------


@dataclass
class WorkerInfo:
    worker_id: str
    label: str
    url: str
    api_key: str = ""  # Key router should use when calling the worker
    capabilities: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    # Live state, updated each heartbeat
    free_vram_gb: float = 0.0
    total_vram_gb: float = 0.0
    free_ram_gb: float = 0.0
    queue_depth: int = 0
    queue_capacity: int = 1
    in_flight: int = 0
    # Hardware introspection (auto-reported by the worker)
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    best_tier: int = TIER_CPU
    # Per-model context window (max_tokens). Auto-reported when available.
    model_context: Dict[str, int] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "label": self.label,
            "url": self.url,
            "capabilities": list(self.capabilities),
            "models": list(self.models),
            "free_vram_gb": self.free_vram_gb,
            "total_vram_gb": self.total_vram_gb,
            "free_ram_gb": self.free_ram_gb,
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
            "in_flight": self.in_flight,
            "gpus": list(self.gpus),
            "best_tier": self.best_tier,
            "model_context": dict(self.model_context),
            "age_seconds": time.time() - self.registered_at,
            "last_heartbeat_age": time.time() - self.last_heartbeat,
            "extra": self.extra,
        }

    def is_alive(self, ttl: float) -> bool:
        return (time.time() - self.last_heartbeat) <= ttl

    def has_capacity(self) -> bool:
        # A worker is considered "free" if it has queue room and at least some VRAM
        return self.queue_depth < max(1, self.queue_capacity)

    def score(self) -> float:
        """Higher = better worker to send the next request to.

        Combines:
          * Hardware tier (5090 > 4090 > 3090 > … > CPU) — dominant factor
          * Free queue slots
          * Free VRAM
          * In-flight load penalty

        Tier dominates so a fast GPU sitting idle always beats a slower one
        sitting idle, but a heavily loaded fast GPU can lose to an idle slower
        one because the load penalty grows with in-flight requests.
        """
        slots_left = max(0, self.queue_capacity - self.queue_depth)
        return (
            self.best_tier * 10.0
            + slots_left * 5.0
            + self.free_vram_gb
            - self.in_flight * 4.0
        )


class WorkerRegistry:
    """Thread-safe in-memory registry of worker ezlocalai nodes."""

    def __init__(self, ttl_seconds: float):
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = RLock()
        self._ttl = ttl_seconds

    @property
    def ttl(self) -> float:
        return self._ttl

    def register(self, info: WorkerInfo) -> WorkerInfo:
        with self._lock:
            existing = self._workers.get(info.worker_id)
            if existing:
                # Preserve registered_at across re-registers
                info.registered_at = existing.registered_at
            self._workers[info.worker_id] = info
            return info

    def heartbeat(
        self, worker_id: str, payload: Dict[str, Any]
    ) -> Optional[WorkerInfo]:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return None
            worker.free_vram_gb = float(
                payload.get("free_vram_gb", worker.free_vram_gb)
            )
            worker.total_vram_gb = float(
                payload.get("total_vram_gb", worker.total_vram_gb)
            )
            worker.free_ram_gb = float(payload.get("free_ram_gb", worker.free_ram_gb))
            worker.queue_depth = int(payload.get("queue_depth", worker.queue_depth))
            worker.queue_capacity = int(
                payload.get("queue_capacity", worker.queue_capacity)
            )
            worker.in_flight = int(payload.get("in_flight", worker.in_flight))
            if "models" in payload:
                worker.models = list(payload["models"])
            if "capabilities" in payload:
                worker.capabilities = list(payload["capabilities"])
            if "gpus" in payload and isinstance(payload["gpus"], list):
                worker.gpus = list(payload["gpus"])
                worker.best_tier = best_gpu_tier(worker.gpus)
            if "model_context" in payload and isinstance(
                payload["model_context"], dict
            ):
                worker.model_context = {
                    str(k): int(v) for k, v in payload["model_context"].items()
                }
            if "extra" in payload and isinstance(payload["extra"], dict):
                worker.extra.update(payload["extra"])
            worker.last_heartbeat = time.time()
            return worker

    def deregister(self, worker_id: str) -> bool:
        with self._lock:
            return self._workers.pop(worker_id, None) is not None

    def prune(self) -> List[str]:
        removed: List[str] = []
        with self._lock:
            for wid, w in list(self._workers.items()):
                if not w.is_alive(self._ttl):
                    self._workers.pop(wid, None)
                    removed.append(wid)
        return removed

    def list_workers(self, alive_only: bool = True) -> List[WorkerInfo]:
        with self._lock:
            workers = list(self._workers.values())
        if alive_only:
            workers = [w for w in workers if w.is_alive(self._ttl)]
        return workers

    def increment_in_flight(self, worker_id: str, delta: int = 1) -> None:
        with self._lock:
            w = self._workers.get(worker_id)
            if w is not None:
                w.in_flight = max(0, w.in_flight + delta)


# ---------------------------------------------------------------------------
# Router (selection logic)
# ---------------------------------------------------------------------------


class Router:
    def __init__(self, registry: WorkerRegistry):
        self.registry = registry

    def select_worker(
        self,
        capability: str,
        model: Optional[str] = None,
    ) -> Optional[WorkerInfo]:
        """Pick the best worker matching capability + (optionally) model."""
        candidates = []
        for worker in self.registry.list_workers(alive_only=True):
            if capability not in worker.capabilities:
                continue
            if model and worker.models and model not in worker.models:
                # Allow base-name match (strip org prefix)
                base = model.split("/")[-1].lower()
                if not any(base in m.lower() for m in worker.models):
                    continue
            if not worker.has_capacity():
                continue
            candidates.append(worker)

        if not candidates:
            return None
        candidates.sort(key=lambda w: w.score(), reverse=True)
        return candidates[0]

    async def wait_for_worker(
        self,
        capability: str,
        model: Optional[str],
        timeout: float,
        poll_interval: float = 0.5,
    ) -> Optional[WorkerInfo]:
        """Block up to ``timeout`` seconds waiting for a free worker."""
        deadline = time.time() + max(0.0, timeout)
        while True:
            worker = self.select_worker(capability, model)
            if worker is not None:
                return worker
            if time.time() >= deadline:
                return None
            await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Worker -> Router heartbeat client
# ---------------------------------------------------------------------------


class WorkerHeartbeatClient:
    """Background client run on each worker ezlocalai instance.

    Registers with the router on start, sends periodic state updates, and
    deregisters cleanly on stop.
    """

    def __init__(
        self,
        router_url: str,
        worker_url: str = "",
        api_key: str = "",
        label: str = "",
        capabilities: Optional[List[str]] = None,
        interval: float = 10.0,
        worker_id: Optional[str] = None,
        worker_port: Optional[int] = None,
    ):
        self.router_url = router_url.rstrip("/")
        self.worker_url = (worker_url or "").rstrip("/")
        self.api_key = api_key
        self.label = label or socket.gethostname()
        self.capabilities = capabilities or detect_local_capabilities()
        self.interval = max(2.0, float(interval))
        # Port used so the router can build http://<source_ip>:<port> if the
        # worker did not (or could not) provide a public URL.
        try:
            self.worker_port = int(
                worker_port if worker_port is not None else getenv("PORT", "8091")
            )
        except (TypeError, ValueError):
            self.worker_port = 8091
        # Stable per-process ID (re-used across reconnects within the process)
        self.worker_id = worker_id or f"{self.label}-{uuid.uuid4().hex[:8]}"
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._registered = False

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "none":
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _gather_state(self) -> Dict[str, Any]:
        """Snapshot the local server's current state for a heartbeat payload."""
        free_vram = 0.0
        total_vram = 0.0
        free_ram = 0.0
        models: List[str] = []
        queue_depth = 0
        queue_capacity = 1
        in_flight = 0
        model_context: Dict[str, int] = {}
        try:
            from Pipes import get_resource_manager  # local import: optional dep

            mgr = get_resource_manager()
            free_vram = float(mgr.get_total_free_vram() or 0.0)
            total_vram = float(getattr(mgr, "total_vram", 0.0) or 0.0)
        except Exception as e:  # pragma: no cover - best-effort
            logging.debug(f"[Heartbeat] resource manager unavailable: {e}")
        try:
            import psutil

            free_ram = psutil.virtual_memory().available / (1024**3)
        except Exception:
            pass
        try:
            # Pull model list from the local Pipes singleton if available
            from app import pipe  # type: ignore

            if pipe is not None:
                data = pipe.get_models()
                models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                # Per-model context window from any persistent llm instances.
                # Falls back to the active llm if persistent_llms is empty.
                instances = dict(getattr(pipe, "persistent_llms", {}) or {})
                if not instances and getattr(pipe, "llm", None) is not None:
                    instances = {getattr(pipe.llm, "model_name", "default"): pipe.llm}
                for name, inst in instances.items():
                    n_ctx = 0
                    xlc = getattr(inst, "xlc_params", None)
                    if xlc is not None and getattr(xlc, "n_ctx", 0):
                        n_ctx = int(xlc.n_ctx)
                    elif hasattr(inst, "n_ctx"):
                        try:
                            n_ctx = int(inst.n_ctx)
                        except Exception:
                            n_ctx = 0
                    if n_ctx > 0:
                        model_context[str(name)] = n_ctx
        except Exception:
            pass
        try:
            from app import request_queue  # type: ignore

            status = request_queue.get_queue_status() if request_queue else {}
            queue_depth = int(status.get("queue_size", 0)) + int(
                status.get("processing_count", 0)
            )
            queue_capacity = int(status.get("max_concurrent", queue_capacity))
            in_flight = int(status.get("processing_count", 0))
        except Exception:
            pass

        gpus = detect_local_gpus()
        return {
            "free_vram_gb": free_vram,
            "total_vram_gb": total_vram,
            "free_ram_gb": free_ram,
            "queue_depth": queue_depth,
            "queue_capacity": queue_capacity,
            "in_flight": in_flight,
            "models": models,
            "capabilities": self.capabilities,
            "gpus": gpus,
            "best_tier": best_gpu_tier(gpus),
            "model_context": model_context,
        }

    async def _post(self, path: str, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        url = f"{self.router_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status >= 200 and resp.status < 300:
                        try:
                            return True, await resp.json()
                        except Exception:
                            return True, None
                    text = await resp.text()
                    logging.warning(
                        f"[Heartbeat] {path} -> HTTP {resp.status}: {text[:200]}"
                    )
                    return False, None
        except Exception as e:
            logging.debug(f"[Heartbeat] {path} failed: {e}")
            return False, None

    async def register(self) -> bool:
        state = await self._gather_state()
        payload = {
            "worker_id": self.worker_id,
            "label": self.label,
            "url": self.worker_url,
            "port": self.worker_port,
            "api_key": self.api_key,
            **state,
        }
        ok, _ = await self._post("/v1/router/register", payload)
        if ok:
            self._registered = True
            logging.info(
                f"[Heartbeat] Registered with router {self.router_url} "
                f"as {self.label} ({self.worker_id})"
            )
        return ok

    async def heartbeat_once(self) -> bool:
        state = await self._gather_state()
        payload = {"worker_id": self.worker_id, **state}
        ok, _ = await self._post("/v1/router/heartbeat", payload)
        if not ok and self._registered:
            # Router probably restarted — try to re-register next loop
            self._registered = False
        if not self._registered:
            return await self.register()
        return ok

    async def deregister(self) -> None:
        if not self._registered:
            return
        await self._post("/v1/router/deregister", {"worker_id": self.worker_id})
        self._registered = False

    async def _run(self) -> None:
        # Initial register attempt; retry on failure with backoff
        backoff = 2.0
        while not self._stop.is_set():
            if await self.register():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(60.0, backoff * 1.5)

        # Steady-state heartbeats
        while not self._stop.is_set():
            try:
                await self.heartbeat_once()
            except Exception as e:  # pragma: no cover
                logging.debug(f"[Heartbeat] loop error: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        try:
            await self.deregister()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singletons (lazy)
# ---------------------------------------------------------------------------

_registry: Optional[WorkerRegistry] = None
_router: Optional[Router] = None
_heartbeat_client: Optional[WorkerHeartbeatClient] = None


def get_registry() -> WorkerRegistry:
    global _registry
    if _registry is None:
        _registry = WorkerRegistry(ttl_seconds=float(getenv("ROUTER_WORKER_TTL", "30")))
    return _registry


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router(get_registry())
    return _router


def is_router_mode() -> bool:
    return (getenv("ROUTER_MODE", "false") or "").strip().lower() == "true"


def get_heartbeat_client() -> Optional[WorkerHeartbeatClient]:
    """Build (or return cached) heartbeat client when ROUTER_URL is configured."""
    global _heartbeat_client
    if _heartbeat_client is not None:
        return _heartbeat_client
    router_url = (getenv("ROUTER_URL") or "").strip()
    if not router_url:
        return None
    if is_router_mode():
        # A router does not register with itself
        return None
    # Use EZLOCALAI_URL as the worker's public callback URL. If unset or
    # loopback, the router substitutes the connection source IP automatically
    # (works for any LAN worker behind no NAT).
    worker_url = (getenv("EZLOCALAI_URL") or "").strip()
    api_key = (getenv("ROUTER_API_KEY") or "").strip() or (
        getenv("EZLOCALAI_API_KEY") or ""
    )
    label = (getenv("WORKER_LABEL") or "").strip() or socket.gethostname()
    capabilities = detect_local_capabilities()
    interval = float(getenv("WORKER_HEARTBEAT_INTERVAL", "10"))
    _heartbeat_client = WorkerHeartbeatClient(
        router_url=router_url,
        worker_url=worker_url,
        api_key=api_key,
        label=label,
        capabilities=capabilities,
        interval=interval,
    )
    return _heartbeat_client
