"""HTTP-over-WebSocket reverse tunnel for ezlocalai router/worker.

Lets a worker that cannot be reached directly (CGNAT, no public IP, friend's
home machine, etc.) still serve inference. The worker dials *out* to the
router — same direction as the existing heartbeat — and holds a long-lived
WebSocket open. The router multiplexes inference requests back through it.

Wire protocol (each frame is a JSON text message; bodies are base64 encoded):

    Router → Worker
        {"t":"req","id":"<uuid>","method":"POST","path":"/v1/...",
         "headers":{...},"body_b64":"...","stream":true|false}
        {"t":"cancel","id":"<uuid>"}

    Worker → Router
        {"t":"resp_start","id":"<uuid>","status":200,"headers":{...}}
        {"t":"resp_chunk","id":"<uuid>","data_b64":"..."}
        {"t":"resp_end","id":"<uuid>"}
        {"t":"resp_err","id":"<uuid>","error":"..."}
        {"t":"ping"} / {"t":"pong"}

Bodies up to ~16MB are sent in one frame; larger ones are not the typical
inference workload so kept simple. Streaming responses (SSE, audio) flow back
as a sequence of resp_chunk frames terminated by resp_end.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Tuple

import aiohttp


# ---------------------------------------------------------------------------
# Router-side hub — tracks connected tunnel workers and routes requests.
# ---------------------------------------------------------------------------


TUNNEL_URL_PREFIX = "tunnel://"


def tunnel_url(worker_id: str) -> str:
    """Sentinel URL stored in the registry for tunneled workers."""
    return f"{TUNNEL_URL_PREFIX}{worker_id}"


def is_tunnel_url(url: str) -> bool:
    return bool(url) and url.startswith(TUNNEL_URL_PREFIX)


def worker_id_from_tunnel_url(url: str) -> str:
    return url[len(TUNNEL_URL_PREFIX) :] if is_tunnel_url(url) else ""


class _PendingRequest:
    """In-flight tunnel request; collects response chunks for one HTTP call."""

    __slots__ = ("queue", "started", "status", "headers", "stream", "error")

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.started: asyncio.Event = asyncio.Event()
        self.status: int = 0
        self.headers: Dict[str, str] = {}
        self.stream: bool = False
        self.error: Optional[str] = None


class TunnelHub:
    """Per-router registry of connected tunnel WebSockets."""

    DISCONNECT_HISTORY_MAX = 20

    def __init__(self) -> None:
        self._connections: Dict[str, "TunnelConnection"] = {}
        self._lock = asyncio.Lock()
        # worker_id -> list[{"at": float, "duration": float, "reason": str}]
        self._disconnect_history: Dict[str, list] = {}
        self._connect_count: Dict[str, int] = {}

    async def attach(self, conn: "TunnelConnection") -> None:
        old: Optional["TunnelConnection"] = None
        async with self._lock:
            old = self._connections.get(conn.worker_id)
            self._connections[conn.worker_id] = conn
            self._connect_count[conn.worker_id] = (
                self._connect_count.get(conn.worker_id, 0) + 1
            )
        # Close the replaced socket outside the hub lock. ``close()`` calls
        # back into ``detach()``, so awaiting it while locked can deadlock the
        # entire tunnel hub during reconnects.
        if old and old is not conn:
            await old.close(reason="superseded")

    async def detach(self, conn: "TunnelConnection") -> None:
        async with self._lock:
            if self._connections.get(conn.worker_id) is conn:
                self._connections.pop(conn.worker_id, None)
            history = self._disconnect_history.setdefault(conn.worker_id, [])
            history.append(
                {
                    "at": time.time(),
                    "duration": time.time() - conn.connected_at,
                    "reason": conn.last_close_reason or "unknown",
                }
            )
            if len(history) > self.DISCONNECT_HISTORY_MAX:
                del history[: -self.DISCONNECT_HISTORY_MAX]

    def is_connected(self, worker_id: str) -> bool:
        c = self._connections.get(worker_id)
        return bool(c and not c.closed)

    def get(self, worker_id: str) -> Optional["TunnelConnection"]:
        return self._connections.get(worker_id)

    def connected_workers(self) -> Dict[str, float]:
        return {
            wid: c.connected_at for wid, c in self._connections.items() if not c.closed
        }

    def stats(self, worker_id: str) -> Dict[str, Any]:
        return {
            "connect_count": self._connect_count.get(worker_id, 0),
            "disconnect_history": list(self._disconnect_history.get(worker_id, [])),
        }


class TunnelConnection:
    """A single live worker tunnel as seen by the router."""

    # Tunable via env in router_app's WS handler if needed.
    PING_INTERVAL = 15.0  # send {t:ping} every N seconds
    PONG_TIMEOUT = 45.0  # close if no frame received in N seconds

    def __init__(self, worker_id: str, ws: Any, hub: TunnelHub) -> None:
        self.worker_id = worker_id
        self.ws = ws  # fastapi.WebSocket
        self.hub = hub
        self.connected_at = time.time()
        self.closed = False
        self.last_recv = time.time()
        self.last_close_reason: str = ""
        self._pending: Dict[str, _PendingRequest] = {}
        self._send_lock = asyncio.Lock()
        self._keepalive_task: Optional[asyncio.Task] = None

    async def close(self, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        self.last_close_reason = reason or self.last_close_reason or "closed"
        # Cancel in-flight requests
        for pid, p in list(self._pending.items()):
            p.error = reason or "tunnel closed"
            p.started.set()
            await p.queue.put(None)
        self._pending.clear()
        if self._keepalive_task and not self._keepalive_task.done():
            current_task = asyncio.current_task()
            if self._keepalive_task is not current_task:
                self._keepalive_task.cancel()
        try:
            try:
                await self.ws.close()
            except Exception:
                pass
        finally:
            await self.hub.detach(self)

    async def _send_json(self, msg: Dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(msg))

    async def _keepalive_loop(self) -> None:
        """Periodically ping the worker and close if it goes silent.

        Catches half-open connections (NAT timeout, silent TCP drop) that
        ``receive_text()`` would otherwise wait on for many minutes.
        """
        try:
            while not self.closed:
                await asyncio.sleep(self.PING_INTERVAL)
                if self.closed:
                    return
                if time.time() - self.last_recv > self.PONG_TIMEOUT:
                    self.last_close_reason = f"no frames in {int(time.time() - self.last_recv)}s (pong timeout)"
                    logging.warning(
                        f"[Tunnel:{self.worker_id}] {self.last_close_reason}, closing"
                    )
                    await self.close(reason=self.last_close_reason)
                    return
                try:
                    await self._send_json({"t": "ping"})
                except Exception as e:
                    self.last_close_reason = f"keepalive send failed: {e}"
                    await self.close(reason=self.last_close_reason)
                    return
        except asyncio.CancelledError:
            pass

    async def reader_loop(self) -> None:
        """Consume frames from the worker until the WS closes."""
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        try:
            while True:
                raw = await self.ws.receive_text()
                self.last_recv = time.time()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                t = msg.get("t")
                if t == "ping":
                    await self._send_json({"t": "pong"})
                    continue
                if t == "pong":
                    continue
                rid = msg.get("id")
                pending = self._pending.get(rid) if rid else None
                if pending is None:
                    continue
                if t == "resp_start":
                    pending.status = int(msg.get("status") or 0)
                    pending.headers = dict(msg.get("headers") or {})
                    pending.stream = bool(msg.get("stream"))
                    pending.started.set()
                elif t == "resp_chunk":
                    data = msg.get("data_b64") or ""
                    if data:
                        try:
                            await pending.queue.put(base64.b64decode(data))
                        except Exception:
                            pass
                elif t == "resp_end":
                    await pending.queue.put(None)
                elif t == "resp_err":
                    pending.error = str(msg.get("error") or "tunnel error")
                    pending.started.set()
                    await pending.queue.put(None)
        except Exception as e:
            self.last_close_reason = f"reader: {type(e).__name__}: {e}"
            logging.info(
                f"[Tunnel:{self.worker_id}] reader loop ended: {self.last_close_reason}"
            )
        finally:
            await self.close(reason=self.last_close_reason or "reader closed")

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
        stream: bool = False,
        timeout: float = 300.0,
    ) -> Tuple[int, Dict[str, str], AsyncIterator[bytes]]:
        """Send an HTTP request through the tunnel.

        Returns (status, response_headers, async-iterator of body chunks).
        For non-streaming responses the iterator yields one or more chunks
        followed by exhaustion; the caller can simply concat them.
        """
        if self.closed:
            raise RuntimeError(f"Tunnel for {self.worker_id} is closed")
        rid = uuid.uuid4().hex
        pending = _PendingRequest()
        self._pending[rid] = pending

        frame = {
            "t": "req",
            "id": rid,
            "method": method.upper(),
            "path": path,
            "headers": headers or {},
            "stream": bool(stream),
        }
        if body:
            frame["body_b64"] = base64.b64encode(body).decode("ascii")
        try:
            await self._send_json(frame)
            try:
                await asyncio.wait_for(pending.started.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Tunnel request to {self.worker_id}{path} timed out before headers"
                )
            if pending.error:
                raise RuntimeError(pending.error)

            async def chunks() -> AsyncIterator[bytes]:
                try:
                    while True:
                        item = await asyncio.wait_for(
                            pending.queue.get(), timeout=timeout
                        )
                        if item is None:
                            break
                        yield item
                finally:
                    self._pending.pop(rid, None)

            return pending.status, pending.headers, chunks()
        except Exception:
            self._pending.pop(rid, None)
            raise


# Module-level singleton hub for the router process.
_hub: Optional[TunnelHub] = None


def get_tunnel_hub() -> TunnelHub:
    global _hub
    if _hub is None:
        _hub = TunnelHub()
    return _hub


# ---------------------------------------------------------------------------
# Worker-side tunnel client — opens the WS and proxies inbound requests to
# the worker's local ezlocalai HTTP server.
# ---------------------------------------------------------------------------


class TunnelClient:
    """Persistent outbound WS connection from a worker to its router.

    The worker calls ``start()`` (background) and this loop reconnects on
    failure with backoff. Each inbound ``req`` frame is dispatched to the
    local ezlocalai HTTP server (``local_url``) and the response is streamed
    back as ``resp_*`` frames.
    """

    def __init__(
        self,
        router_ws_url: str,
        worker_id: str,
        local_url: str,
        api_key: str = "",
        ping_interval: float = 20.0,
    ) -> None:
        self.router_ws_url = router_ws_url
        self.worker_id = worker_id
        self.local_url = local_url.rstrip("/") or "http://localhost:8091"
        self.api_key = api_key
        self.ping_interval = ping_interval
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._send_lock = asyncio.Lock()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
        except Exception:
            pass
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            connected_at = time.time()
            try:
                await self._connect_once()
                # If we held the connection for >30s, treat as healthy and
                # reset backoff so a single transient drop doesn't push us
                # into long sleeps.
                if time.time() - connected_at > 30:
                    backoff = 2.0
            except Exception as e:
                logging.warning(
                    f"[Tunnel] connection to {self.router_ws_url} failed: "
                    f"{type(e).__name__}: {e}"
                )
            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            # Cap at 15s so a flapping tunnel recovers quickly once the
            # network stabilizes.
            backoff = min(15.0, backoff * 1.5)

    async def _connect_once(self) -> None:
        headers = {}
        if self.api_key and self.api_key != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"
        params = {"worker_id": self.worker_id}
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                self.router_ws_url,
                headers=headers,
                params=params,
                heartbeat=self.ping_interval,
                max_msg_size=64 * 1024 * 1024,
            ) as ws:
                self._ws = ws
                logging.info(
                    f"[Tunnel] Connected to {self.router_ws_url} as {self.worker_id}"
                )
                close_reason = "clean close"
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(msg.data)
                            except Exception:
                                continue
                            asyncio.create_task(self._handle(payload))
                        elif msg.type == aiohttp.WSMsgType.PING:
                            # aiohttp auto-pongs, just note activity
                            continue
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            close_reason = (
                                f"server CLOSE code={ws.close_code} "
                                f"extra={msg.extra!r}"
                            )
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            close_reason = f"CLOSED code={ws.close_code}"
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            close_reason = f"WS ERROR: {ws.exception()}"
                            break
                finally:
                    self._ws = None
                    logging.info(
                        f"[Tunnel] Disconnected from {self.router_ws_url}: {close_reason}"
                    )

    async def _send_json(self, msg: Dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return
        async with self._send_lock:
            await ws.send_str(json.dumps(msg))

    async def _send_chunk(self, rid: str, data: bytes) -> None:
        await self._send_json(
            {
                "t": "resp_chunk",
                "id": rid,
                "data_b64": base64.b64encode(data).decode("ascii"),
            }
        )

    async def _handle(self, msg: Dict[str, Any]) -> None:
        t = msg.get("t")
        if t == "ping":
            await self._send_json({"t": "pong"})
            return
        if t != "req":
            return
        rid = msg.get("id")
        if not rid:
            return
        method = (msg.get("method") or "GET").upper()
        path = msg.get("path") or "/"
        headers = dict(msg.get("headers") or {})
        body_b64 = msg.get("body_b64")
        body = base64.b64decode(body_b64) if body_b64 else None
        url = f"{self.local_url}{path}"
        if bool(msg.get("stream")):
            # Streaming inference can legitimately run far longer than 10
            # minutes. Use read-idle timeouts so keepalive chunks keep the
            # tunnel open without imposing a total wall-clock cap.
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=600)
        else:
            timeout = aiohttp.ClientTimeout(total=600)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method, url, headers=headers, data=body
                ) as resp:
                    await self._send_json(
                        {
                            "t": "resp_start",
                            "id": rid,
                            "status": resp.status,
                            "headers": {k: v for k, v in resp.headers.items()},
                            "stream": True,
                        }
                    )
                    async for chunk in resp.content.iter_any():
                        if chunk:
                            await self._send_chunk(rid, chunk)
                    await self._send_json({"t": "resp_end", "id": rid})
        except Exception as e:
            logging.warning(f"[Tunnel] local request {method} {path} failed: {e}")
            await self._send_json(
                {"t": "resp_err", "id": rid, "error": f"{type(e).__name__}: {e}"}
            )
