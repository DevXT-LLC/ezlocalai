import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from Router import Router, WorkerInfo, WorkerRegistry
import router_app


def _local_qwen_worker(*, busy: int = 0, tier: int = 50) -> WorkerInfo:
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
        best_tier=tier,
        cap_slots={"text": dict(slot), "vision": dict(slot)},
        model_slots={model: dict(slot)},
    )


class ChutesWorkerTests(unittest.TestCase):
    def test_worker_is_enabled_by_key_with_default_model_and_tier(self):
        worker = router_app._build_chutes_worker(api_key="cpk_test", model="")

        self.assertIsNotNone(worker)
        self.assertEqual(worker.models, ["Qwen/Qwen3.8-27B-TEE"])
        self.assertEqual(worker.best_tier, 50)
        self.assertEqual(worker.priority_tier, 50)
        self.assertEqual(worker.capabilities, ["text", "vision"])
        self.assertTrue(worker.external_fallback)
        self.assertTrue(worker.is_alive(ttl=0))
        self.assertEqual(
            router_app._worker_headers(worker),
            {"Authorization": "Bearer cpk_test"},
        )

    def test_worker_is_disabled_without_key(self):
        self.assertIsNone(router_app._build_chutes_worker(api_key=""))
        self.assertIsNone(router_app._build_chutes_worker(api_key="none"))

    def test_local_t50_worker_is_preferred_before_chutes(self):
        registry = WorkerRegistry(ttl_seconds=60)
        local = registry.register(_local_qwen_worker())
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        selected = Router(registry).select_worker(
            "vision", "Qwen/Qwen3.8-27B-TEE", allow_cross_model=False
        )

        self.assertIs(selected, local)
        self.assertIsNot(selected, chutes)

    def test_chutes_handles_overflow_when_local_t50_is_busy(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(_local_qwen_worker(busy=1))
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        selected = Router(registry).select_worker(
            "vision", "Qwen/Qwen3.8-27B-TEE", allow_cross_model=False
        )

        self.assertIs(selected, chutes)

    def test_chutes_does_not_take_a_local_dispatch_lease(self):
        registry = WorkerRegistry(ttl_seconds=60)
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        reservation = registry.increment_in_flight(
            chutes.worker_id,
            capability="vision",
            model=chutes.models[0],
        )

        self.assertIsNone(reservation)
        self.assertEqual(chutes.router_in_flight, 0)
        self.assertTrue(chutes.has_capacity("vision", chutes.models[0]))

    def test_chutes_payload_uses_provider_model_and_stream_usage(self):
        worker = router_app._build_chutes_worker(
            api_key="cpk_test", model="custom/chutes-model"
        )
        original = {
            "model": "local/model",
            "stream": True,
            "disable_fallback": False,
            "stream_options": {"some_option": True},
        }

        forwarded = router_app._worker_json_payload(
            worker, "/v1/chat/completions", original
        )

        self.assertEqual(forwarded["model"], "custom/chutes-model")
        self.assertNotIn("disable_fallback", forwarded)
        self.assertTrue(forwarded["stream_options"]["include_usage"])
        self.assertTrue(forwarded["stream_options"]["some_option"])
        self.assertEqual(original["model"], "local/model")
        self.assertIn("disable_fallback", original)

    def test_internal_worker_receives_disable_fallback_flag(self):
        internal = _local_qwen_worker()

        forwarded = router_app._worker_json_payload(
            internal,
            "/v1/chat/completions",
            {"model": internal.models[0], "disable_fallback": True},
        )

        self.assertTrue(forwarded["disable_fallback"])

    def test_dashboard_includes_chutes_as_t50_vlm(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        with (
            patch("router_app.get_registry", return_value=registry),
            patch(
                "router_app.get_router",
                return_value=SimpleNamespace(waiting_requests=0),
            ),
        ):
            data = router_app._aggregate_dashboard()

        worker = data["workers"][0]
        self.assertEqual(worker["label"], "Chutes API")
        self.assertEqual(worker["priority_tier"], 50)
        self.assertIn("vision", worker["capabilities"])
        html = router_app._render_dashboard_html(data)
        self.assertIn("Chutes API", html)
        self.assertIn("tier 50", html)
        self.assertIn("Chutes managed inference", html)


class ChutesRoutingRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_disable_fallback_excludes_chutes_and_waits_without_deadline(self):
        registry = WorkerRegistry(ttl_seconds=60)
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))
        local = registry.register(_local_qwen_worker())
        captured = {}

        class CapturingRouter:
            async def wait_for_worker(
                self, capability, model, timeout, poll_interval=0.5, exclude=None
            ):
                captured.update(
                    capability=capability,
                    model=model,
                    timeout=timeout,
                    exclude=set(exclude or ()),
                )
                return local

        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app.get_router", return_value=CapturingRouter()),
        ):
            selected = await router_app._pick(
                "vision",
                "Qwen/Qwen3.8-27B-TEE",
                external_fallback_allowed=False,
                wait_indefinitely=True,
            )

        self.assertIs(selected, local)
        self.assertEqual(captured["timeout"], 0)
        self.assertIn(chutes.worker_id, captured["exclude"])

    async def test_chat_disable_fallback_controls_router_selection(self):
        proxy = AsyncMock(return_value={"ok": True})
        payload = {
            "model": "local/model",
            "messages": [{"role": "user", "content": "hello"}],
            "disable_fallback": True,
        }

        with patch("router_app._llm_proxy_with_retry", proxy):
            result = await router_app.chat_completions(payload, _="client")

        self.assertEqual(result, {"ok": True})
        kwargs = proxy.await_args.kwargs
        self.assertFalse(kwargs["external_fallback_allowed"])
        self.assertTrue(kwargs["wait_indefinitely"])
        self.assertTrue(kwargs["payload"]["disable_fallback"])

    async def test_zero_token_chutes_response_still_counts_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = router_app.UsageTracker(f"{tmp}/usage.json")
            worker = router_app._build_chutes_worker(api_key="cpk_test")
            original_usage = router_app._usage
            try:
                router_app._usage = tracker
                await router_app._record_llm_usage(
                    worker,
                    "local/model",
                    0,
                    0,
                    {"total_ms": 12.0},
                )
            finally:
                router_app._usage = original_usage

        model_stats = tracker.snapshot()["Chutes API"]["llm"]["Qwen/Qwen3.8-27B-TEE"]
        self.assertEqual(model_stats["requests"], 1)
        self.assertEqual(model_stats["prompt_tokens"], 0)
        self.assertEqual(model_stats["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
