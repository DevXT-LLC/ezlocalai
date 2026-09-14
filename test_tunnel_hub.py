import asyncio
import unittest

from Tunnel import TunnelConnection, TunnelHub


class FakeWebSocket:
    def __init__(self):
        self.close_count = 0
        self.sent = []

    async def close(self):
        self.close_count += 1

    async def send_text(self, data):
        self.sent.append(data)


class TunnelHubTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
