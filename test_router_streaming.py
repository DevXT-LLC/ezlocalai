import os
import pathlib
import sys
import asyncio
import unittest
from unittest.mock import AsyncMock, patch


ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Router import WorkerInfo, WorkerRegistry
import router_app


def _sse(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode("utf-8")


class RouterStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_binary_stream_propagates_upstream_setup_error(self):
        worker = WorkerInfo(
            worker_id="tts-worker",
            label="tts-worker",
            url="http://tts-worker",
            capabilities=["tts"],
            models=[],
        )
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(worker)

        class FakeResponse:
            status = 503
            headers = {"Content-Type": "application/json"}

            async def read(self):
                return b'{"detail":"TTS model unavailable"}'

            def release(self):
                return None

        class FakeSession:
            def __init__(self, **_kwargs):
                self.closed = False

            async def post(self, *_args, **_kwargs):
                return FakeResponse()

            async def close(self):
                self.closed = True

        with patch("router_app.get_registry", return_value=registry), patch(
            "router_app.aiohttp.ClientSession", FakeSession
        ):
            response = await router_app._proxy_json(
                worker,
                "/v1/audio/speech/stream",
                {"input": "Hello"},
                stream=True,
                capability="tts",
                stream_media_type="application/octet-stream",
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.body, b'{"detail":"TTS model unavailable"}')
        self.assertEqual(worker.router_in_flight, 0)

    async def test_binary_stream_does_not_append_sse_error_payload(self):
        worker = WorkerInfo(
            worker_id="tts-worker",
            label="tts-worker",
            url="http://tts-worker",
            capabilities=["tts"],
            models=[],
        )
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(worker)

        class FakeContent:
            async def iter_any(self):
                yield b"\x01\x02audio"
                raise RuntimeError("voice worker disconnected")

        class FakeResponse:
            status = 200
            headers = {"Content-Type": "application/octet-stream"}
            content = FakeContent()

            def release(self):
                return None

        class FakeSession:
            def __init__(self, **_kwargs):
                pass

            async def post(self, *_args, **_kwargs):
                return FakeResponse()

            async def close(self):
                return None

        with patch("router_app.get_registry", return_value=registry), patch(
            "router_app.aiohttp.ClientSession", FakeSession
        ):
            response = await router_app._proxy_json(
                worker,
                "/v1/audio/speech/stream",
                {"input": "Hello"},
                stream=True,
                capability="tts",
                stream_media_type="application/octet-stream",
            )
            body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(body, b"\x01\x02audio")
        self.assertNotIn(b"data:", body)
        self.assertEqual(worker.router_in_flight, 0)

    async def test_tunnel_binary_stream_propagates_upstream_setup_error(self):
        worker = WorkerInfo(
            worker_id="tts-worker",
            label="tts-worker",
            url="tunnel://tts-worker",
            capabilities=["tts"],
            models=[],
        )
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(worker)

        async def error_chunks():
            yield b'{"detail":"TTS model unavailable"}'

        connection = type(
            "Connection",
            (),
            {
                "closed": False,
                "request": AsyncMock(
                    return_value=(
                        503,
                        {"Content-Type": "application/json"},
                        error_chunks(),
                    )
                ),
            },
        )()
        hub = type("Hub", (), {"get": lambda self, _wid: connection})()

        with patch("router_app.get_registry", return_value=registry), patch(
            "router_app.get_tunnel_hub", return_value=hub
        ):
            response = await router_app._proxy_via_tunnel(
                worker,
                "POST",
                "/v1/audio/speech/stream",
                headers={"Content-Type": "application/json"},
                body=b"{}",
                stream=True,
                timeout=30,
                capability="tts",
                stream_media_type="application/octet-stream",
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.body, b'{"detail":"TTS model unavailable"}')
        self.assertEqual(worker.router_in_flight, 0)

    def test_prompt_affinity_wait_default_covers_worker_heartbeat_release_lag(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROUTER_PROMPT_AFFINITY_WAIT", None)
            self.assertEqual(router_app._prompt_affinity_wait_timeout(), 15.0)

    async def test_prompt_affinity_reuses_the_same_available_worker(self):
        registry = WorkerRegistry(ttl_seconds=60)
        first = registry.register(
            WorkerInfo(
                worker_id="first",
                label="first",
                url="http://first",
                capabilities=["text"],
                models=["model"],
            )
        )
        second = registry.register(
            WorkerInfo(
                worker_id="second",
                label="second",
                url="http://second",
                capabilities=["text"],
                models=["model"],
            )
        )
        router = type(
            "FakeRouter",
            (),
            {"wait_for_worker": AsyncMock(side_effect=[first, second])},
        )()
        router_app._prompt_affinity.clear()

        with patch("router_app.get_registry", return_value=registry), patch(
            "router_app.get_router", return_value=router
        ):
            selected_first = await router_app._pick(
                "text", "model", affinity_key="text:model:conversation"
            )
            selected_second = await router_app._pick(
                "text", "model", affinity_key="text:model:conversation"
            )

        self.assertEqual(selected_first.worker_id, "first")
        self.assertEqual(selected_second.worker_id, "first")
        router.wait_for_worker.assert_awaited_once()

    async def test_prompt_affinity_waits_for_cached_worker_to_become_available(self):
        registry = WorkerRegistry(ttl_seconds=60)
        first = registry.register(
            WorkerInfo(
                worker_id="first",
                label="first",
                url="http://first",
                capabilities=["text"],
                models=["model"],
            )
        )
        router = type(
            "FakeRouter",
            (),
            {"wait_for_worker": AsyncMock(return_value=first)},
        )()
        router_app._prompt_affinity.clear()

        with patch("router_app.get_registry", return_value=registry), patch(
            "router_app.get_router", return_value=router
        ), patch.dict(os.environ, {"ROUTER_PROMPT_AFFINITY_WAIT": "0.25"}):
            selected_first = await router_app._pick(
                "text", "model", affinity_key="text:model:conversation"
            )
            first.queue_depth = 1

            async def release_cached_worker():
                await asyncio.sleep(0.03)
                first.queue_depth = 0

            release = asyncio.create_task(release_cached_worker())
            selected_second = await router_app._pick(
                "text", "model", affinity_key="text:model:conversation"
            )
            await release

        self.assertEqual(selected_first.worker_id, "first")
        self.assertEqual(selected_second.worker_id, "first")
        router.wait_for_worker.assert_awaited_once()

    async def test_temporary_spillover_does_not_replace_cache_home(self):
        registry = WorkerRegistry(ttl_seconds=60)
        first = registry.register(
            WorkerInfo(
                worker_id="first",
                label="first",
                url="http://first",
                capabilities=["text"],
                models=["model"],
            )
        )
        second = registry.register(
            WorkerInfo(
                worker_id="second",
                label="second",
                url="http://second",
                capabilities=["text"],
                models=["model"],
            )
        )
        router = type(
            "FakeRouter",
            (),
            {"wait_for_worker": AsyncMock(side_effect=[first, second])},
        )()
        affinity_key = "text:model:conversation"
        router_app._prompt_affinity.clear()

        with patch("router_app.get_registry", return_value=registry), patch(
            "router_app.get_router", return_value=router
        ), patch.dict(os.environ, {"ROUTER_PROMPT_AFFINITY_WAIT": "0.01"}):
            await router_app._pick("text", "model", affinity_key=affinity_key)
            first.queue_depth = 1
            selected_spillover = await router_app._pick(
                "text", "model", affinity_key=affinity_key
            )
            first.queue_depth = 0
            selected_after_spillover = await router_app._pick(
                "text", "model", affinity_key=affinity_key
            )

        self.assertEqual(selected_spillover.worker_id, "second")
        self.assertEqual(selected_after_spillover.worker_id, "first")
        self.assertEqual(router_app._prompt_affinity[affinity_key][0], "first")
        self.assertEqual(router.wait_for_worker.await_count, 2)

    def test_nested_prompt_softly_avoids_parent_cache_worker(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            WorkerInfo(
                worker_id="parent",
                label="parent",
                url="http://parent",
                capabilities=["text"],
                models=["model"],
            )
        )
        registry.register(
            WorkerInfo(
                worker_id="browser",
                label="browser",
                url="http://browser",
                capabilities=["text"],
                models=["model"],
            )
        )
        router_app._prompt_affinity.clear()
        parent_key = router_app._prompt_affinity_key(
            {"prompt_cache_key": "workconductor:agent:conversation"},
            "text",
            "model",
        )
        router_app._prompt_affinity[parent_key] = ("parent", router_app.time.time())

        with patch("router_app.get_registry", return_value=registry):
            avoided = router_app._prompt_cache_avoid_worker_ids(
                {
                    "prompt_cache_key": "workconductor-browser:agent:conversation",
                    "prompt_cache_avoid_key": "workconductor:agent:conversation",
                },
                "text",
                "model",
            )

        self.assertEqual(avoided, {"parent"})

    def test_nested_prompt_does_not_avoid_only_compatible_worker(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            WorkerInfo(
                worker_id="parent",
                label="parent",
                url="http://parent",
                capabilities=["text"],
                models=["model"],
            )
        )
        router_app._prompt_affinity.clear()
        parent_key = router_app._prompt_affinity_key(
            {"prompt_cache_key": "workconductor:agent:conversation"},
            "text",
            "model",
        )
        router_app._prompt_affinity[parent_key] = ("parent", router_app.time.time())

        with patch("router_app.get_registry", return_value=registry):
            avoided = router_app._prompt_cache_avoid_worker_ids(
                {"prompt_cache_avoid_key": "workconductor:agent:conversation"},
                "text",
                "model",
            )

        self.assertEqual(avoided, set())

    def test_llamacpp_timings_report_total_cached_and_evaluated_prompt_tokens(self):
        timings = {}

        prompt_tokens, completion_tokens = router_app._extract_tokens_from_sse_event(
            b'{"timings":{"cache_n":53,"prompt_n":4,"predicted_n":7,"prompt_ms":140}}',
            0,
            0,
            timings,
        )

        self.assertEqual(prompt_tokens, 57)
        self.assertEqual(completion_tokens, 7)
        self.assertEqual(timings["cached_prompt_tokens"], 53.0)
        self.assertEqual(timings["evaluated_prompt_tokens"], 4.0)

    def test_usage_cache_details_survive_partial_timing_data(self):
        timings = {}

        prompt_tokens, _ = router_app._extract_tokens_from_sse_event(
            b'{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":80}},'
            b'"timings":{"prompt_ms":120}}',
            0,
            0,
            timings,
        )

        self.assertEqual(prompt_tokens, 100)
        self.assertEqual(timings["cached_prompt_tokens"], 80.0)
        self.assertNotIn("evaluated_prompt_tokens", timings)

    def test_router_cache_metadata_is_not_forwarded_to_worker(self):
        worker = WorkerInfo(
            worker_id="worker",
            label="worker",
            url="http://worker",
            capabilities=["text"],
            models=["model"],
        )

        forwarded = router_app._worker_json_payload(
            worker,
            "/v1/chat/completions",
            {
                "model": "model",
                "prompt_cache_key": "conversation-key",
                "prompt_cache_avoid_key": "parent-key",
            },
        )

        self.assertNotIn("prompt_cache_key", forwarded)
        self.assertNotIn("prompt_cache_avoid_key", forwarded)
        self.assertEqual(forwarded["model"], "model")

    def test_sse_classifier_detects_nested_error(self):
        event = _sse(
            '{"error":{"message":"request exceeds context","type":"exceed_context_size_error"}}'
        ).strip()

        info = router_app._classify_sse_event(event)

        self.assertFalse(info["has_text"])
        self.assertTrue(info["terminal"])
        self.assertEqual(info["error_message"], "request exceeds context")
        self.assertEqual(info["error_type"], "exceed_context_size_error")

    async def test_stream_failover_retries_empty_worker_before_text(self):
        registry = WorkerRegistry(ttl_seconds=60)
        first = registry.register(
            WorkerInfo(
                worker_id="first",
                label="first",
                url="http://first",
                capabilities=["text"],
                models=["model"],
            )
        )
        second = registry.register(
            WorkerInfo(
                worker_id="second",
                label="second",
                url="http://second",
                capabilities=["text"],
                models=["model"],
            )
        )
        workers = [first, second]
        original_pick = router_app._pick
        original_iter = router_app._iter_worker_stream_bytes
        original_attempts = router_app._stream_max_attempts
        original_registry = router_app.get_registry

        async def fake_pick(capability, model, exclude=None, **kwargs):
            for worker in workers:
                if worker.worker_id not in (exclude or set()):
                    return worker
            raise AssertionError("no fake worker left")

        async def fake_iter(
            worker,
            path,
            payload,
            *,
            capability=None,
            model=None,
            reservation_id=None,
        ):
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
            router_app._stream_max_attempts = lambda capability, *args: 2
            router_app.get_registry = lambda: registry

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
            router_app.get_registry = original_registry

        body = b"".join(chunks).decode("utf-8")
        self.assertIn('"content":"ok"', body)
        self.assertNotIn("request exceeds context", body)

    async def test_stream_failover_preserves_all_worker_context_failure(self):
        registry = WorkerRegistry(ttl_seconds=60)
        workers = [
            registry.register(
                WorkerInfo(
                    worker_id="first",
                    label="first",
                    url="http://first",
                    capabilities=["text"],
                    models=["model"],
                    model_context={"model": 180_000},
                )
            ),
            registry.register(
                WorkerInfo(
                    worker_id="second",
                    label="second",
                    url="http://second",
                    capabilities=["text"],
                    models=["model"],
                    model_context={"model": 200_000},
                )
            ),
        ]
        original_pick = router_app._pick
        original_iter = router_app._iter_worker_stream_bytes
        original_attempts = router_app._stream_max_attempts
        original_registry = router_app.get_registry

        async def fake_pick(capability, model, exclude=None, **kwargs):
            for worker in workers:
                if worker.worker_id not in (exclude or set()):
                    return worker
            raise AssertionError("no fake worker left")

        async def fake_iter(
            worker,
            path,
            payload,
            *,
            capability=None,
            model=None,
            reservation_id=None,
        ):
            yield _sse(
                '{"error":{"message":"request exceeds context",'
                '"type":"exceed_context_size_error"}}'
            )

        try:
            router_app._pick = fake_pick
            router_app._iter_worker_stream_bytes = fake_iter
            router_app._stream_max_attempts = lambda capability, *args: 2
            router_app.get_registry = lambda: registry

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
            router_app.get_registry = original_registry

        body = b"".join(chunks).decode("utf-8")
        self.assertIn('"type": "context_capacity_error"', body)
        self.assertIn('"compact_and_retry": true', body)
        self.assertIn('"n_ctx": 200000', body)

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

        async def fake_record_cap(label, capability, **kwargs):
            captured["usage"] = (label, capability, kwargs)

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
        self.assertEqual(captured["usage"][0:2], ("tts-worker", "tts"))
        self.assertEqual(captured["usage"][2]["model"], "tts-1")
        self.assertEqual(captured["usage"][2]["outputs"], 1)


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
