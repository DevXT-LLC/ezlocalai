import os
import pathlib
import sys
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Router import WorkerInfo
import router_app


def _sse(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode("utf-8")


class RouterStreamingTests(unittest.IsolatedAsyncioTestCase):
    def test_sse_classifier_detects_nested_error(self):
        event = _sse(
            '{"error":{"message":"request exceeds context","type":"exceed_context_size_error"}}'
        ).strip()

        info = router_app._classify_sse_event(event)

        self.assertFalse(info["has_text"])
        self.assertTrue(info["terminal"])
        self.assertEqual(info["error_message"], "request exceeds context")

    async def test_stream_failover_retries_empty_worker_before_text(self):
        first = WorkerInfo(
            worker_id="first",
            label="first",
            url="http://first",
            capabilities=["text"],
            models=["model"],
        )
        second = WorkerInfo(
            worker_id="second",
            label="second",
            url="http://second",
            capabilities=["text"],
            models=["model"],
        )
        workers = [first, second]
        original_pick = router_app._pick
        original_iter = router_app._iter_worker_stream_bytes
        original_attempts = router_app._stream_max_attempts

        async def fake_pick(capability, model, exclude=None):
            for worker in workers:
                if worker.worker_id not in (exclude or set()):
                    return worker
            raise AssertionError("no fake worker left")

        async def fake_iter(worker, path, payload, *, capability=None, model=None):
            if worker.worker_id == "first":
                yield _sse(
                    '{"error":{"message":"request exceeds context","type":"exceed_context_size_error"}}'
                )
                return
            yield _sse('{"choices":[{"index":0,"delta":{"content":"ok"}}]}')
            yield _sse('{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}')

        try:
            router_app._pick = fake_pick
            router_app._iter_worker_stream_bytes = fake_iter
            router_app._stream_max_attempts = lambda capability: 2

            chunks = []
            async for chunk in router_app._llm_stream_with_worker_failover(
                capability="text",
                path="/v1/chat/completions",
                payload={"stream": True},
                model="model",
                request_started=0,
            ):
                chunks.append(chunk)
        finally:
            router_app._pick = original_pick
            router_app._iter_worker_stream_bytes = original_iter
            router_app._stream_max_attempts = original_attempts

        body = b"".join(chunks).decode("utf-8")
        self.assertIn('"content":"ok"', body)
        self.assertNotIn("request exceeds context", body)

    async def test_audio_speech_stream_uses_binary_pcm_response_metadata(self):
        worker = WorkerInfo(
            worker_id="tts-worker",
            label="tts-worker",
            url="http://tts-worker",
            capabilities=["tts"],
            models=[],
        )
        captured = {}
        original_pick = router_app._pick
        original_proxy = router_app._proxy_json
        original_record_cap = router_app._usage.record_cap

        async def fake_pick(capability, model, exclude=None):
            self.assertEqual(capability, "tts")
            return worker

        async def fake_proxy(worker_arg, path, payload, **kwargs):
            captured.update(kwargs)
            captured["path"] = path
            captured["payload"] = payload
            return router_app.Response(
                content=b"", media_type=kwargs["stream_media_type"]
            )

        async def fake_record_cap(label, capability):
            captured["usage"] = (label, capability)

        try:
            router_app._pick = fake_pick
            router_app._proxy_json = fake_proxy
            router_app._usage.record_cap = fake_record_cap

            response = await router_app.audio_speech_stream(
                {"model": "tts-1", "input": "Hello."}, _="test-client"
            )
        finally:
            router_app._pick = original_pick
            router_app._proxy_json = original_proxy
            router_app._usage.record_cap = original_record_cap

        self.assertEqual(captured["path"], "/v1/audio/speech/stream")
        self.assertTrue(captured["stream"])
        self.assertEqual(captured["capability"], "tts")
        self.assertEqual(captured["stream_media_type"], "application/octet-stream")
        self.assertEqual(captured["stream_headers"]["X-Audio-Format"], "pcm")
        self.assertEqual(captured["stream_headers"]["X-Sample-Rate"], "24000")
        self.assertEqual(response.media_type, "application/octet-stream")
        self.assertEqual(captured["usage"], ("tts-worker", "tts"))


class RouterTimeoutTests(unittest.TestCase):
    def test_stt_timeout_defaults_to_large_transcription_window(self):
        with patch.dict(
            os.environ,
            {"REQUEST_TIMEOUT": "300", "ROUTER_STT_TIMEOUT": ""},
        ):
            self.assertEqual(router_app._stt_timeout(), 7200.0)

    def test_stt_timeout_can_be_overridden(self):
        with patch.dict(os.environ, {"ROUTER_STT_TIMEOUT": "900"}):
            self.assertEqual(router_app._stt_timeout(), 900.0)


if __name__ == "__main__":
    unittest.main()
