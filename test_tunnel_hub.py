import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp

from Tunnel import TunnelClient, TunnelConnection, TunnelHub


class FakeWebSocket:
    def __init__(self):
        self.close_count = 0
        self.sent = []

    async def close(self):
        self.close_count += 1

    async def send_text(self, data):
        self.sent.append(data)


class TunnelHubTests(unittest.IsolatedAsyncioTestCase):
    async def test_replacement_drains_existing_stream_without_interruption(self):
        hub = TunnelHub()
        old = TunnelConnection("worker", FakeWebSocket(), hub)
        await hub.attach(old)

        async def send(frame):
            pending = old._pending[frame["id"]]
            pending.status = 200
            pending.started.set()
            await pending.queue.put(b"first")

        old._send_json = send
        _, _, chunks = await old.request("POST", "/test", stream=True)
        self.assertEqual(await anext(chunks), b"first")
        pending = next(iter(old._pending.values()))
        new = TunnelConnection("worker", FakeWebSocket(), hub)
        await hub.attach(new)
        self.assertFalse(old.closed)
        self.assertTrue(old.draining)
        with self.assertRaises(RuntimeError):
            await old.request("POST", "/new")
        await pending.queue.put(b"last")
        await pending.queue.put(None)
        self.assertEqual([chunk async for chunk in chunks], [b"last"])
        self.assertTrue(old.closed)
        self.assertIs(hub.get("worker"), new)
        await new.close()

    async def test_cancelled_consumer_sends_cancel_to_worker(self):
        ws = FakeWebSocket()
        conn = TunnelConnection("worker", ws, TunnelHub())
        task = asyncio.create_task(conn.request("POST", "/test"))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(conn._pending)
        frames = [json.loads(frame) for frame in ws.sent]
        self.assertEqual([f["t"] for f in frames], ["req", "cancel"])
        self.assertEqual(frames[0]["id"], frames[1]["id"])

    async def test_successful_end_remains_clean(self):
        conn = TunnelConnection("worker", FakeWebSocket(), TunnelHub())

        async def send(frame):
            pending = conn._pending[frame["id"]]
            pending.status = 200
            pending.started.set()
            await pending.queue.put(b"ok")
            await pending.queue.put(None)

        conn._send_json = send
        _, _, chunks = await conn.request("POST", "/test", stream=True)
        self.assertEqual([part async for part in chunks], [b"ok"])
        self.assertFalse(conn._pending)

    async def test_error_after_headers_is_not_clean_eof(self):
        for partial in (False, True):
            conn = TunnelConnection("worker", FakeWebSocket(), TunnelHub())

            async def send(frame):
                pending = conn._pending[frame["id"]]
                pending.status = 200
                pending.started.set()

            conn._send_json = send
            _, _, chunks = await conn.request("POST", "/test", stream=True)
            pending = next(iter(conn._pending.values()))
            if partial:
                await pending.queue.put(b"partial")
                self.assertEqual(await anext(chunks), b"partial")
            await conn.close("worker disconnected")
            with self.assertRaisesRegex(RuntimeError, "worker disconnected"):
                await anext(chunks)
            self.assertFalse(conn._pending)

    async def test_attach_supersedes_existing_connection_without_deadlock(self):
        hub = TunnelHub()
        old_ws = FakeWebSocket()
        old = TunnelConnection("worker-1", old_ws, hub)
        await hub.attach(old)

        new = TunnelConnection("worker-1", FakeWebSocket(), hub)
        await asyncio.wait_for(hub.attach(new), timeout=1.0)

        self.assertTrue(old.closed)
        self.assertEqual(old_ws.close_count, 1)
        self.assertIs(hub.get("worker-1"), new)
        self.assertTrue(hub.is_connected("worker-1"))

        stats = hub.stats("worker-1")
        self.assertEqual(stats["connect_count"], 2)
        self.assertEqual(stats["disconnect_history"][0]["reason"], "superseded")

        await new.close(reason="test cleanup")

    async def test_close_from_keepalive_task_detaches_connection(self):
        hub = TunnelHub()
        conn = TunnelConnection("worker-1", FakeWebSocket(), hub)
        await hub.attach(conn)

        async def close_as_keepalive_task():
            conn._keepalive_task = asyncio.current_task()
            await conn.close(reason="pong timeout")

        await asyncio.wait_for(close_as_keepalive_task(), timeout=1.0)

        self.assertTrue(conn.closed)
        self.assertIsNone(hub.get("worker-1"))
        self.assertFalse(hub.is_connected("worker-1"))
        self.assertEqual(
            hub.stats("worker-1")["disconnect_history"][0]["reason"], "pong timeout"
        )


class TunnelClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_and_explicit_cancel_stop_local_requests(self):
        for explicit_cancel in (False, True):
            with self.subTest(explicit_cancel=explicit_cancel):

                class Socket:
                    closed = False
                    close_code = 1000

                    def __init__(self):
                        self.messages = asyncio.Queue()
                        self.sent = []

                    async def receive(self):
                        return await self.messages.get()

                    async def send_str(self, text):
                        self.sent.append(json.loads(text))

                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *args):
                        self.closed = True

                ws = Socket()

                class Session:
                    def __init__(self, **kwargs):
                        pass

                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *args):
                        pass

                    def ws_connect(self, *args, **kwargs):
                        self_options.update(kwargs)
                        return ws

                self_options = {}
                started, stopped = asyncio.Event(), asyncio.Event()
                client = TunnelClient("http://router/tunnel", "worker", "http://local")
                original = client._handle

                async def handle(message, socket):
                    if message.get("t") != "req":
                        return await original(message, socket)
                    self.assertIs(socket, ws)
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.set()

                client._handle = handle

                def message(payload):
                    return SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload)
                    )

                with patch("Tunnel.aiohttp.ClientSession", Session):
                    connection = asyncio.create_task(client._connect_once())
                    await ws.messages.put(message({"t": "req", "id": "request"}))
                    await asyncio.wait_for(started.wait(), 1)
                    await ws.messages.put(message({"t": "ping"}))
                    if explicit_cancel:
                        await ws.messages.put(message({"t": "cancel", "id": "request"}))
                        await asyncio.wait_for(stopped.wait(), 1)
                    await ws.messages.put(
                        SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, extra="test")
                    )
                    await asyncio.wait_for(connection, 1)
                    self.assertTrue(stopped.is_set())
                    self.assertIn({"t": "pong"}, ws.sent)
                    self.assertIsNone(self_options["heartbeat"])

    async def test_old_response_does_not_move_to_reconnected_socket(self):
        class Socket:
            closed = False

            def __init__(self):
                self.sent = []

            async def send_str(self, text):
                self.sent.append(json.loads(text))

        client = TunnelClient("http://router/tunnel", "worker", "http://local")
        old, new = Socket(), Socket()
        client._ws = new
        await client._send_chunk("old-request", b"old response", old)
        self.assertEqual(len(old.sent), 1)
        self.assertEqual(new.sent, [])


if __name__ == "__main__":
    unittest.main()
