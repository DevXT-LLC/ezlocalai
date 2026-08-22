import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from Router import Router, WorkerInfo, WorkerRegistry
import router_app


def _local_qwen_worker(*, busy: int = 0) -> WorkerInfo:
    model = "unsloth/Qwen3.8-27B-GGUF"
    slot = {
        "capacity": 1,
        "in_flight": busy,
        "queued": 0,
        "available": max(0, 1 - busy),
    }
    return WorkerInfo(
        worker_id="local-3090",
        label="Local 3090",
        url="http://local-3090",
        capabilities=["text", "vision"],
        models=[model],
        free_vram_gb=20.0,
        best_tier=50,
        cap_slots={"text": dict(slot), "vision": dict(slot)},
        model_slots={model: dict(slot)},
    )


class OpenRouterWorkerTests(unittest.TestCase):
    def test_worker_is_enabled_by_key_with_requested_defaults(self):
        worker = router_app._build_openrouter_worker(api_key="sk-or-test", model="")

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "OpenRouter.ai")
        self.assertEqual(worker.url, "https://openrouter.ai/api")
        self.assertEqual(worker.models, ["qwen/qwen3.8-27b"])
        self.assertEqual(worker.best_tier, 39)
        self.assertEqual(worker.priority_tier, 39)
        self.assertEqual(worker.capabilities, ["text", "vision"])
        self.assertEqual(worker.queue_capacity, 1000)
        self.assertEqual(worker.cap_slots["text"]["capacity"], 1000)
        self.assertEqual(worker.model_slots[worker.models[0]]["available"], 1000)
        self.assertTrue(worker.external_fallback)
        self.assertTrue(worker.persistent)
        self.assertEqual(
            router_app._worker_headers(worker),
            {"Authorization": "Bearer sk-or-test"},
        )
        self.assertNotIn("api_key", worker.to_public())

    def test_worker_is_disabled_without_key(self):
        self.assertIsNone(router_app._build_openrouter_worker(api_key=""))
        self.assertIsNone(router_app._build_openrouter_worker(api_key="none"))

    def test_tiers_order_local_then_chutes_then_openrouter(self):
        registry = WorkerRegistry(ttl_seconds=60)
        local = registry.register(_local_qwen_worker())
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))
        openrouter = registry.register(
            router_app._build_openrouter_worker(api_key="sk-or-test")
        )
        router = Router(registry)

        selected = router.select_worker(
            "vision", "qwen/qwen3.8-27b", allow_cross_model=False
        )
        self.assertIs(selected, local)

        local.cap_slots["text"]["in_flight"] = 1
        local.cap_slots["vision"]["in_flight"] = 1
        local.model_slots[local.models[0]]["in_flight"] = 1
        selected = router.select_worker(
            "vision", "qwen/qwen3.8-27b", allow_cross_model=False
        )
        self.assertIs(selected, chutes)
        self.assertGreater(
            chutes.score("vision", chutes.models[0]),
            openrouter.score("vision", openrouter.models[0]),
        )

        reservations = [
            registry.increment_in_flight(
                chutes.worker_id,
                capability="vision",
                model=chutes.models[0],
            )
            for _ in range(100)
        ]
        selected = router.select_worker(
            "vision", "qwen/qwen3.8-27b", allow_cross_model=False
        )
        self.assertIs(selected, openrouter)
        for reservation in reservations:
            registry.release_in_flight(chutes.worker_id, reservation)

    def test_openrouter_tracks_one_of_1000_slots(self):
        registry = WorkerRegistry(ttl_seconds=60)
        worker = registry.register(
            router_app._build_openrouter_worker(api_key="sk-or-test")
        )

        reservation = registry.increment_in_flight(
            worker.worker_id,
            capability="text",
            model=worker.models[0],
        )

        self.assertIsNotNone(reservation)
        self.assertEqual(worker.router_in_flight, 1)
        self.assertEqual(worker.slots_left("text", worker.models[0]), 999)
        registry.release_in_flight(worker.worker_id, reservation)
        self.assertEqual(worker.slots_left("text", worker.models[0]), 1000)

    def test_payload_uses_openrouter_model_and_strips_router_flag(self):
        worker = router_app._build_openrouter_worker(
            api_key="sk-or-test", model="custom/openrouter-model"
        )
        original = {
            "model": "local/model",
            "stream": True,
            "disable_fallback": False,
        }

        forwarded = router_app._worker_json_payload(
            worker, "/v1/chat/completions", original
        )

        self.assertEqual(forwarded["model"], "custom/openrouter-model")
        self.assertNotIn("disable_fallback", forwarded)
        self.assertNotIn("stream_options", forwarded)
        self.assertEqual(original["model"], "local/model")

    def test_dashboard_shows_tier_capacity_balance_and_grouped_model(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(_local_qwen_worker())
        worker = registry.register(
            router_app._build_openrouter_worker(api_key="sk-or-test")
        )
        worker.external_balance_usd = 42.125

        with (
            patch("router_app.get_registry", return_value=registry),
            patch(
                "router_app.get_router",
                return_value=SimpleNamespace(waiting_requests=0),
            ),
        ):
            data = router_app._aggregate_dashboard()

        public = next(w for w in data["workers"] if w["worker_id"] == worker.worker_id)
        self.assertEqual(public["priority_tier"], 39)
        html = router_app._render_dashboard_html(data)
        self.assertIn("OpenRouter.ai", html)
        self.assertIn("OpenRouter API", html)
        self.assertNotIn("OpenRouter API [api]", html)
        self.assertIn("tier 39", html)
        self.assertIn("0/1000", html)
        self.assertIn("$42.12 remaining", html)
        qwen_models = [m for m in data["models"] if m["model"] == "Qwen3.8-27B"]
        self.assertEqual(len(qwen_models), 1)
        self.assertEqual(qwen_models[0]["worker_count"], 2)

    def test_credit_and_key_balance_payloads(self):
        self.assertEqual(
            router_app._openrouter_balance_from_payload(
                {"data": {"total_credits": 100.5, "total_usage": 25.75}}
            ),
            74.75,
        )
        self.assertEqual(
            router_app._openrouter_balance_from_payload(
                {"data": {"limit_remaining": 12.25, "usage": 4.0}}
            ),
            12.25,
        )
        self.assertEqual(
            router_app._openrouter_balance_from_payload(
                {"data": {"limit": 20.0, "usage": 4.5}}
            ),
            15.5,
        )


class OpenRouterAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_initializer_registers_and_seeds_openrouter(self):
        worker = router_app._build_openrouter_worker(api_key="sk-or-test")
        refresh = AsyncMock(return_value=21.0)

        with (
            patch("router_app._sync_chutes_worker", return_value=None),
            patch("router_app._sync_openrouter_worker", return_value=worker),
            patch("router_app._refresh_external_balance", refresh),
        ):
            initialized = await router_app._initialize_external_workers()

        self.assertEqual(initialized, [worker])
        refresh.assert_awaited_once_with(worker)

    async def test_balance_refresh_uses_credits_endpoint(self):
        worker = router_app._build_openrouter_worker(api_key="sk-or-test")
        captured = []

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self, **_kwargs):
                return {"data": {"total_credits": 80.0, "total_usage": 7.25}}

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def get(self, url, headers):
                captured.append((url, headers))
                return FakeResponse()

        with patch("router_app.aiohttp.ClientSession", return_value=FakeSession()):
            balance = await router_app._refresh_external_balance(worker)

        self.assertEqual(balance, 72.75)
        self.assertEqual(worker.external_balance_usd, 72.75)
        self.assertGreater(worker.external_balance_updated_at, 0)
        self.assertEqual(captured[0][0], "https://openrouter.ai/api/v1/credits")
        self.assertEqual(captured[0][1], {"Authorization": "Bearer sk-or-test"})

    async def test_balance_refresh_falls_back_to_current_key_limit(self):
        worker = router_app._build_openrouter_worker(api_key="sk-or-test")
        captured = []

        class FakeResponse:
            def __init__(self, status, payload):
                self.status = status
                self.payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self, **_kwargs):
                return self.payload

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def get(self, url, headers):
                captured.append(url)
                if url == router_app.OPENROUTER_CREDITS_URL:
                    return FakeResponse(403, {})
                return FakeResponse(200, {"data": {"limit_remaining": 9.5}})

        with patch("router_app.aiohttp.ClientSession", return_value=FakeSession()):
            balance = await router_app._refresh_external_balance(worker)

        self.assertEqual(balance, 9.5)
        self.assertEqual(
            captured,
            [router_app.OPENROUTER_CREDITS_URL, router_app.OPENROUTER_KEY_URL],
        )

    async def test_usage_is_recorded_under_openrouter_and_grouped_model(self):
        worker = router_app._build_openrouter_worker(api_key="sk-or-test")
        with tempfile.TemporaryDirectory() as tmp:
            tracker = router_app.UsageTracker(f"{tmp}/usage.json")
            original_usage = router_app._usage
            try:
                router_app._usage = tracker
                await router_app._record_llm_usage(
                    worker,
                    "local/model",
                    11,
                    7,
                    {"total_ms": 25.0},
                )
            finally:
                router_app._usage = original_usage

        stats = tracker.snapshot()["OpenRouter.ai"]["llm"]["Qwen3.8-27B"]
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["prompt_tokens"], 11)
        self.assertEqual(stats["completion_tokens"], 7)

    async def test_disable_fallback_excludes_all_managed_providers(self):
        registry = WorkerRegistry(ttl_seconds=60)
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))
        openrouter = registry.register(
            router_app._build_openrouter_worker(api_key="sk-or-test")
        )
        local = registry.register(_local_qwen_worker())
        captured = {}

        class CapturingRouter:
            async def wait_for_worker(
                self, capability, model, timeout, poll_interval=0.5, exclude=None
            ):
                captured.update(timeout=timeout, exclude=set(exclude or ()))
                return local

        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app.get_router", return_value=CapturingRouter()),
        ):
            selected = await router_app._pick(
                "vision",
                "qwen/qwen3.8-27b",
                external_fallback_allowed=False,
                wait_indefinitely=True,
            )

        self.assertIs(selected, local)
        self.assertEqual(captured["timeout"], 0)
        self.assertIn(chutes.worker_id, captured["exclude"])
        self.assertIn(openrouter.worker_id, captured["exclude"])


if __name__ == "__main__":
    unittest.main()
