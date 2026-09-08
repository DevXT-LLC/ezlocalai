import asyncio
import ast
import struct
import threading
import unittest
import anyio
from pathlib import Path
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

from ezlocalai.AudioStreaming import stream_native_audio
from ezlocalai.TTSWebSocket import segment_events, stream_tts_session


HEADER = struct.pack("<IHH", 24000, 16, 1)
FRAME = struct.pack("<I", 4) + b"\x01\x00\x02\x00"
END = struct.pack("<I", 0)


async def fragments(data, size=1):
    for offset in range(0, len(data), size):
        yield data[offset : offset + size]


class FramingTests(unittest.IsolatedAsyncioTestCase):
    async def test_arbitrary_fragmentation(self):
        for size in (1, 2, 3, 7, 8, 20):
            events = [
                item
                async for item in segment_events(fragments(HEADER + FRAME + END, size))
            ]
            self.assertEqual(events[0], ("header", HEADER))
            self.assertEqual(b"".join(value for kind, value in events[1:]), FRAME)

    async def test_invalid_segments(self):
        for data in (
            b"",
            HEADER,
            HEADER + FRAME[:-1],
            HEADER + END + b"x",
            HEADER + struct.pack("<I", 3),
            HEADER + struct.pack("<I", 2**25),
            struct.pack("<IHH", 44100, 16, 1) + END,
        ):
            with self.assertRaises(ValueError):
                _ = [item async for item in segment_events(fragments(data))]


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_iterator_stays_on_one_thread(self):
        threads = []

        def generate(**kwargs):
            try:
                for i in range(5):
                    threads.append(threading.get_ident())
                    yield i
            finally:
                threads.append(threading.get_ident())

        model = SimpleNamespace(generate_voice_clone_stream=generate, close=Mock())
        self.assertEqual([v async for v in stream_native_audio(model)], list(range(5)))
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], threading.get_ident())
        model.close.assert_not_called()

    async def test_abandon_stops_native_and_closes_iterator(self):
        closed = threading.Event()
        generated = []

        def generate(**kwargs):
            try:
                for i in range(1000):
                    generated.append(i)
                    yield i
            finally:
                closed.set()

        model = SimpleNamespace(generate_voice_clone_stream=generate, close=Mock())
        async with aclosing(stream_native_audio(model)) as stream:
            self.assertEqual(await anext(stream), 0)
        self.assertTrue(closed.is_set())
        self.assertLess(len(generated), 10)
        model.close.assert_called_once()

    async def test_native_error_reaches_consumer(self):
        def generate(**kwargs):
            yield 1
            raise RuntimeError("native failure")

        model = SimpleNamespace(generate_voice_clone_stream=generate, close=Mock())
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            _ = [v async for v in stream_native_audio(model)]
        model.close.assert_called_once()

    async def test_anyio_disconnect_waits_for_native_teardown(self):
        stopped = threading.Event()
        iterator_closed = threading.Event()

        def generate(**kwargs):
            try:
                yield 1
                stopped.wait(timeout=2)
            finally:
                iterator_closed.set()

        model = SimpleNamespace(generate_voice_clone_stream=generate, close=stopped.set)
        with anyio.CancelScope() as scope:
            async with aclosing(stream_native_audio(model)) as stream:
                self.assertEqual(await anext(stream), 1)
                scope.cancel()
                await anyio.sleep(0)
        self.assertTrue(stopped.is_set())
        self.assertTrue(iterator_closed.is_set())


class SessionTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self):
        incoming = asyncio.Queue()
        outgoing = []
        started, resume = asyncio.Event(), asyncio.Event()
        generated = []

        async def generate(text, voice, language):
            generated.append((text, voice, language))
            yield HEADER
            started.set()
            await resume.wait()
            yield FRAME
            yield END

        async def send(data):
            outgoing.append(data)

        pipe = SimpleNamespace(
            _tts_lock=SimpleNamespace(
                acquire=AsyncMock(return_value={"handoff": False}), release=AsyncMock()
            ),
            _get_tts=Mock(return_value=SimpleNamespace(generate_stream=generate)),
            _destroy_tts=Mock(),
        )
        ws = SimpleNamespace(receive_json=incoming.get, send_bytes=send)
        return incoming, outgoing, started, resume, generated, pipe, ws

    async def test_text_received_during_audio_and_single_framing(self):
        incoming, outgoing, started, resume, generated, pipe, ws = self.make_session()
        task = asyncio.create_task(stream_tts_session(ws, pipe))
        await asyncio.sleep(0.01)
        pipe._tts_lock.acquire.assert_not_called()
        await incoming.put({"text": "First segment.", "flush": True})
        await asyncio.wait_for(started.wait(), 1)
        await incoming.put({"text": "Second segment.", "done": True})
        await asyncio.sleep(0.01)
        self.assertTrue(incoming.empty())  # Reader isn't blocked on synthesis.
        resume.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(
            [item[0] for item in generated], ["First segment.", "Second segment."]
        )
        self.assertEqual(b"".join(outgoing), HEADER + FRAME + FRAME + END)
        pipe._destroy_tts.assert_called_once_with(async_cleanup=False)
        pipe._tts_lock.release.assert_awaited_once()

    async def test_empty_session_never_leases(self):
        incoming, outgoing, _, _, _, pipe, ws = self.make_session()
        await incoming.put({"done": True})
        await stream_tts_session(ws, pipe)
        self.assertEqual(b"".join(outgoing), HEADER + END)
        pipe._tts_lock.acquire.assert_not_called()

    async def test_cancel_releases_active_lease(self):
        incoming, _, started, _, _, pipe, ws = self.make_session()
        await incoming.put({"text": "Hello.", "flush": True})
        task = asyncio.create_task(stream_tts_session(ws, pipe))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        pipe._tts_lock.release.assert_awaited_once()
        pipe._destroy_tts.assert_called_once()

    async def test_invalid_input_and_idle_timeout(self):
        for message in ({"text": "x" * 4097}, {"text": 1}, [], {"voice": 1}):
            incoming, _, _, _, _, pipe, ws = self.make_session()
            await incoming.put(message)
            with self.assertRaises(ValueError):
                await stream_tts_session(ws, pipe)
            pipe._tts_lock.acquire.assert_not_called()
        _, _, _, _, _, pipe, ws = self.make_session()
        with self.assertRaises(asyncio.TimeoutError):
            await stream_tts_session(ws, pipe, idle_timeout=0.01)
        pipe._tts_lock.acquire.assert_not_called()


class EndpointTests(unittest.TestCase):
    def test_actual_endpoint_auth_and_framed_response(self):
        from fastapi import (
            FastAPI,
            Header,
            HTTPException,
            WebSocket,
            WebSocketDisconnect,
        )
        from fastapi.testclient import TestClient
        import logging

        app = FastAPI()

        async def generate(*args):
            yield HEADER
            yield FRAME
            yield END

        pipe = SimpleNamespace(
            _tts_lock=SimpleNamespace(
                acquire=AsyncMock(return_value={}), release=AsyncMock()
            ),
            _get_tts=Mock(return_value=SimpleNamespace(generate_stream=generate)),
            _destroy_tts=Mock(),
        )
        namespace = dict(
            app=app,
            pipe=pipe,
            asyncio=asyncio,
            logging=logging,
            Header=Header,
            HTTPException=HTTPException,
            WebSocket=WebSocket,
            WebSocketDisconnect=WebSocketDisconnect,
            getenv=lambda key: "test-secret" if key == "EZLOCALAI_API_KEY" else "true",
        )
        tree = ast.parse((Path(__file__).parent / "app.py").read_text())
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"verify_api_key", "tts_websocket"}
        ]
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), "app.py", "exec"),
            namespace,
        )
        with TestClient(app) as client:
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                with self.assertRaises(WebSocketDisconnect) as failure:
                    with client.websocket_connect(
                        "/v1/audio/speech/ws", headers=headers
                    ):
                        pass
                self.assertEqual(failure.exception.code, 1008)
            pipe._get_tts.assert_not_called()
            with client.websocket_connect(
                "/v1/audio/speech/ws", headers={"Authorization": "Bearer test-secret"}
            ) as ws:
                ws.send_json({"text": "Hello.", "done": True})
                data = bytearray()
                while True:
                    chunk = ws.receive_bytes()
                    if not chunk:
                        break
                    data.extend(chunk)
                self.assertEqual(data, HEADER + FRAME + END)
        pipe._tts_lock.release.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
