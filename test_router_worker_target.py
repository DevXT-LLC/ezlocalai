"""Exercise strict targeting and vision forwarding through real local HTTP upstreams."""

import asyncio
import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import httpx
from aiohttp import web

import router_app
from Router import Router, WorkerInfo, WorkerRegistry
from scripts.router_vision_smoke import red_image_url


class WorkerTargetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = WorkerRegistry(ttl_seconds=60)
        self.router = Router(self.registry)
        self.calls = []
        self.status = 200

        async def upstream(request):
            body = await request.json()
            self.calls.append(
                (request.path, body, request.headers.get("Authorization"))
            )
            if self.status != 200:
                return web.json_response(
                    {"error": {"message": "vision unavailable"}}, status=self.status
                )
            if body.get("stream"):
                chunk = {"choices": [{"delta": {"content": "red"}}]}
                return web.Response(
                    text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                    content_type="text/event-stream",
                )
            return web.json_response({"choices": [{"message": {"content": "red"}}]})

        app = web.Application()
        app.router.add_post("/{provider}/v1/chat/completions", upstream)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.addAsyncCleanup(self.runner.cleanup)
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.local = self.registry.register(
            WorkerInfo(
                worker_id="local",
                label="DevXT5090",
                url=f"{base}/local",
                capabilities=["text", "vision"],
                models=["unsloth/Qwen3.8-27B-GGUF"],
                best_tier=100,
                queue_capacity=1,
            )
        )
        self.providers = [
            router_app._build_chutes_worker(api_key="test-chutes", model=""),
            router_app._build_openrouter_worker(api_key="test-openrouter", model=""),
        ]
        for worker in self.providers:
            worker.url = f"{base}/{worker.external_provider}"
            self.registry.register(worker)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in [
            ("get_registry", lambda: self.registry),
            ("get_router", lambda: self.router),
            ("_wait_timeout", lambda: 2),
            ("_record_llm_usage", AsyncMock()),
            ("_schedule_external_balance_refresh", lambda worker: None),
        ]:
            self.stack.enter_context(patch.object(router_app, name, value))
        self.stack.enter_context(
            patch.dict(
                router_app.app.dependency_overrides,
                {
                    router_app.verify_client: lambda: "test",
                },
            )
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=router_app.app), base_url="http://router"
        )
        self.addAsyncCleanup(self.client.aclose)

    def payload(self, worker="DevXT5090", stream=False):
        return {
            "worker": worker,
            "model": self.local.models[0],
            "stream": stream,
            "messages": [{"role": "user", "content": "Hello"}],
        }

    async def post(self, payload):
        return await self.client.post("/v1/chat/completions", json=payload)

    async def test_vision_forwarding_to_each_provider(self):
        for worker in self.providers:
            for stream in (False, True):
                for url in ("https://example.com/test.png", red_image_url()):
                    for image_type in ("image_url", "input_image"):
                        with self.subTest(
                            worker=worker.label, stream=stream, url=url, type=image_type
                        ):
                            image = {
                                "type": image_type,
                                "image_url": url,
                                "detail": "low",
                            }
                            if image_type == "image_url":
                                image = {
                                    "type": image_type,
                                    "image_url": {"url": url, "detail": "low"},
                                }
                            payload = self.payload(f" {worker.label.lower()} ", stream)
                            payload["messages"][0]["content"] = [
                                {"type": "text", "text": "What color?"},
                                image,
                            ]
                            original = copy.deepcopy(payload)
                            response = await self.post(payload)
                            self.assertEqual(response.status_code, 200, response.text)
                            self.assertIn("red", response.text)
                            path, body, auth = self.calls[-1]
                            self.assertEqual(
                                path, f"/{worker.external_provider}/v1/chat/completions"
                            )
                            self.assertEqual(auth, f"Bearer {worker.api_key}")
                            self.assertEqual(body["model"], worker.models[0])
                            self.assertEqual(
                                body["messages"][0]["content"][1],
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url, "detail": "low"},
                                },
                            )
                            self.assertNotIn("worker", body)
                            self.assertNotIn("disable_fallback", body)
                            self.assertEqual(payload, original)
                            self.assertEqual(worker.router_in_flight, 0)

    async def test_busy_target_queues_even_with_idle_alternatives(self):
        for worker in [self.local, *self.providers]:
            for stream in (False, True):
                with self.subTest(worker=worker.label, stream=stream):
                    reservations = [
                        self.registry.try_reserve_in_flight(
                            worker.worker_id, capability="text", model=worker.models[0]
                        )
                        for _ in range(worker.queue_capacity)
                    ]
                    before = len(self.calls)
                    task = asyncio.create_task(
                        self.post(self.payload(worker.label, stream))
                    )
                    try:
                        async with asyncio.timeout(1):
                            while not self.router.waiting_requests:
                                await asyncio.sleep(0.01)
                        self.assertFalse(task.done())
                        self.assertEqual(len(self.calls), before)
                        self.registry.release_in_flight(
                            worker.worker_id, reservations.pop()
                        )
                        response = await asyncio.wait_for(task, timeout=2)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(len(self.calls), before + 1)
                        self.assertEqual(
                            self.calls[-1][0],
                            f"/{worker.external_provider or 'local'}/v1/chat/completions",
                        )
                        if worker is self.local:
                            self.assertTrue(self.calls[-1][1]["disable_fallback"])
                            self.assertNotIn("worker", self.calls[-1][1])
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        for reservation in reservations:
                            self.registry.release_in_flight(
                                worker.worker_id, reservation
                            )
                    self.assertEqual(self.router.waiting_requests, 0)

    async def test_target_errors_never_fail_over(self):
        for stream in (False, True):
            for status in (400, 429, 503):
                self.status = status
                before = len(self.calls)
                response = await self.post(self.payload("chutes.ai", stream))
                self.assertEqual(len(self.calls), before + 1)
                self.assertIn("vision unavailable", response.text)
                self.assertEqual(response.status_code, 200 if stream else status)
                self.assertEqual(self.providers[0].router_in_flight, 0)

    async def test_invalid_targets_rejected_before_stream_headers(self):
        for stream in (False, True):
            for label, status in [(None, 400), (123, 400), ("", 400), ("missing", 404)]:
                response = await self.post(self.payload(label, stream))
                self.assertEqual(response.status_code, status)
            payload = self.payload("chutes.ai", stream)
            payload["disable_fallback"] = True
            self.assertEqual((await self.post(payload)).status_code, 400)
        self.assertEqual(self.calls, [])

    async def test_duplicate_offline_and_incompatible_labels(self):
        self.registry.register(
            WorkerInfo(worker_id="duplicate", label="DEVXT5090", url="http://unused")
        )
        self.assertEqual((await self.post(self.payload())).status_code, 400)
        self.registry.deregister("duplicate")
        self.local.last_heartbeat = 0
        self.assertEqual((await self.post(self.payload())).status_code, 503)
        self.local.persistent = True
        self.local.capabilities = ["text"]
        payload = self.payload()
        payload["messages"][0]["content"] = [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ]
        self.assertEqual((await self.post(payload)).status_code, 400)
        self.local.url = "tunnel://local"
        self.assertEqual((await self.post(self.payload())).status_code, 503)

    async def test_pin_overrides_cache_preferences(self):
        with patch.object(
            router_app,
            "_prompt_cache_avoid_worker_ids",
            return_value={self.local.worker_id},
        ):
            response = await self.post(self.payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.calls[-1][0], "/local/v1/chat/completions")

    async def test_dispatch_races_do_not_exhaust_attempts(self):
        reserve = self.registry.try_reserve_in_flight
        for stream in (False, True):
            misses = 5

            def race(*args, **kwargs):
                nonlocal misses
                if misses:
                    misses -= 1
                    return None
                return reserve(*args, **kwargs)

            with patch.object(self.registry, "try_reserve_in_flight", side_effect=race):
                response = await self.post(self.payload(stream=stream))
            self.assertEqual(response.status_code, 200)
            self.assertIn("red", response.text)

    async def test_target_queue_timeout_and_cancellation_cleanup(self):
        self.local.queue_depth = 1
        with patch.object(router_app, "_wait_timeout", return_value=0.02):
            self.assertEqual((await self.post(self.payload())).status_code, 503)
        self.assertEqual(self.router.waiting_requests, 0)
        task = asyncio.create_task(
            router_app._pick("text", self.local.models[0], worker_id="local")
        )
        async with asyncio.timeout(1):
            while not self.router.waiting_requests:
                await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.router.waiting_requests, 0)
        self.assertEqual(self.calls, [])

    def test_provider_pools_accept_mixed_requests_until_full(self):
        for worker, capacity in zip(self.providers, (10, 20)):
            registry = WorkerRegistry(ttl_seconds=60)
            registry.register(worker)
            router = Router(registry)
            reservations = []
            for index in range(capacity):
                capability = "text" if index % 2 else "vision"
                self.assertIs(
                    router.select_worker(capability, worker.models[0]), worker
                )
                reservation = registry.try_reserve_in_flight(
                    worker.worker_id, capability=capability, model=worker.models[0]
                )
                self.assertIsNotNone(reservation)
                reservations.append(reservation)
            for capability in ("text", "vision"):
                self.assertIsNone(router.select_worker(capability, worker.models[0]))
                self.assertIsNone(
                    registry.try_reserve_in_flight(
                        worker.worker_id, capability=capability, model=worker.models[0]
                    )
                )
            for reservation in reservations:
                registry.release_in_flight(worker.worker_id, reservation)
            self.assertEqual(worker.total_slots_left(), capacity)

    def test_image_conversion_does_not_mutate_original_messages(self):
        for worker in self.providers:
            payload = self.payload(worker.label)
            payload["messages"][0]["content"] = [
                {
                    "type": "input_image",
                    "image_url": red_image_url(),
                    "detail": "low",
                }
            ]
            original = copy.deepcopy(payload)
            forwarded = router_app._worker_json_payload(
                worker, "/v1/chat/completions", payload
            )
            self.assertEqual(payload, original)
            self.assertEqual(
                forwarded["messages"][0]["content"][0]["type"], "image_url"
            )


if __name__ == "__main__":
    unittest.main()
