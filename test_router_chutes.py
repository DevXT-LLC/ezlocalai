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
        self.assertEqual(worker.label, "Chutes.ai")
        self.assertEqual(worker.best_tier, 45)
        self.assertEqual(worker.priority_tier, 45)
        self.assertEqual(worker.capabilities, ["text", "vision"])
        self.assertEqual(worker.queue_capacity, 100)
        self.assertEqual(worker.cap_slots["text"]["capacity"], 100)
        self.assertEqual(worker.cap_slots["vision"]["available"], 100)
        self.assertEqual(worker.model_slots[worker.models[0]]["capacity"], 100)
        self.assertTrue(worker.external_fallback)
        self.assertTrue(worker.is_alive(ttl=0))
        self.assertEqual(
            router_app._worker_headers(worker),
            {"Authorization": "Bearer cpk_test"},
        )

    def test_worker_is_disabled_without_key(self):
        self.assertIsNone(router_app._build_chutes_worker(api_key=""))
        self.assertIsNone(router_app._build_chutes_worker(api_key="none"))

    def test_comma_separated_models_are_trimmed_deduplicated_and_advertised(self):
        worker = router_app._build_chutes_worker(
            api_key="cpk_test",
            model=" Qwen/Qwen3.8-27B-TEE, Qwen/Qwen3.5-27B, qwen/qwen3.8-27b-tee ",
        )

        self.assertEqual(
            worker.models,
            ["Qwen/Qwen3.8-27B-TEE", "Qwen/Qwen3.5-27B"],
        )
        self.assertEqual(set(worker.model_slots), set(worker.models))

    def test_local_t50_worker_is_preferred_before_chutes_t45(self):
        registry = WorkerRegistry(ttl_seconds=60)
        local = registry.register(_local_qwen_worker())
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        selected = Router(registry).select_worker(
            "vision", "Qwen/Qwen3.8-27B-TEE", allow_cross_model=False
        )

        self.assertIs(selected, local)
        self.assertIsNot(selected, chutes)

    def test_tunneled_t50_worker_at_adjusted_t45_is_preferred_before_chutes(self):
        registry = WorkerRegistry(ttl_seconds=60)
        remote = _local_qwen_worker()
        remote.url = "tunnel://local-3090"
        remote = registry.register(remote)
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        self.assertEqual(remote.priority_tier, chutes.priority_tier)
        selected = Router(registry).select_worker(
            "vision", "Qwen/Qwen3.8-27B-TEE", allow_cross_model=False
        )

        self.assertIs(selected, remote)

    def test_chutes_handles_overflow_when_local_t50_is_busy(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(_local_qwen_worker(busy=1))
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        selected = Router(registry).select_worker(
            "vision", "Qwen/Qwen3.8-27B-TEE", allow_cross_model=False
        )

        self.assertIs(selected, chutes)

    def test_chutes_tracks_and_releases_one_of_100_dispatch_slots(self):
        registry = WorkerRegistry(ttl_seconds=60)
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        reservation = registry.increment_in_flight(
            chutes.worker_id,
            capability="vision",
            model=chutes.models[0],
        )

        self.assertIsNotNone(reservation)
        self.assertEqual(chutes.router_in_flight, 1)
        self.assertEqual(chutes.slots_left("vision", chutes.models[0]), 99)
        self.assertTrue(chutes.has_capacity("vision", chutes.models[0]))

        registry.release_in_flight(chutes.worker_id, reservation)

        self.assertEqual(chutes.router_in_flight, 0)
        self.assertEqual(chutes.slots_left("vision", chutes.models[0]), 100)

    def test_chutes_stops_accepting_requests_at_100_slots(self):
        registry = WorkerRegistry(ttl_seconds=60)
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))
        reservations = [
            registry.increment_in_flight(
                chutes.worker_id,
                capability="vision",
                model=chutes.models[0],
            )
            for _ in range(100)
        ]

        self.assertEqual(chutes.router_in_flight, 100)
        self.assertFalse(chutes.has_capacity("vision", chutes.models[0]))
        self.assertIsNone(
            Router(registry).select_worker(
                "vision", chutes.models[0], allow_cross_model=False
            )
        )

        for reservation in reservations:
            registry.release_in_flight(chutes.worker_id, reservation)

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

    def test_payload_selects_matching_model_from_configured_list(self):
        worker = router_app._build_chutes_worker(
            api_key="cpk_test",
            model="Qwen/Qwen3.8-27B-TEE,Qwen/Qwen3.5-27B",
        )

        forwarded = router_app._worker_json_payload(
            worker,
            "/v1/chat/completions",
            {"model": "Qwen/Qwen3.5-27B", "messages": []},
        )

        self.assertEqual(forwarded["model"], "Qwen/Qwen3.5-27B")
        self.assertEqual(
            router_app._worker_usage_model(worker, "Qwen/Qwen3.5-27B"),
            "Qwen/Qwen3.5-27B",
        )

    def test_qwen_payload_preserves_profile_and_vllm_thinking_control(self):
        worker = router_app._build_chutes_worker(api_key="cpk_test")

        forwarded = router_app._worker_json_payload(
            worker,
            "/v1/chat/completions",
            {
                "model": "qwen/qwen3.8-27b",
                "messages": [],
                "reasoning": {"enabled": False, "effort": "low"},
                "max_tokens": 123,
                "seed": 42,
            },
        )

        self.assertNotIn("reasoning", forwarded)
        self.assertEqual(forwarded["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(forwarded["temperature"], 0.7)
        self.assertEqual(forwarded["top_p"], 0.8)
        self.assertEqual(forwarded["top_k"], 20)
        self.assertEqual(forwarded["min_p"], 0.0)
        self.assertEqual(forwarded["presence_penalty"], 1.5)
        self.assertEqual(forwarded["repetition_penalty"], 1.0)
        self.assertEqual(forwarded["max_tokens"], 123)
        self.assertEqual(forwarded["seed"], 42)

    def test_internal_worker_receives_disable_fallback_flag(self):
        internal = _local_qwen_worker()

        forwarded = router_app._worker_json_payload(
            internal,
            "/v1/chat/completions",
            {"model": internal.models[0], "disable_fallback": True},
        )

        self.assertTrue(forwarded["disable_fallback"])

    def test_dashboard_includes_chutes_as_t45_vlm_with_cached_balance(self):
        registry = WorkerRegistry(ttl_seconds=60)
        chutes = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))
        chutes.external_balance_usd = 12.345

        with (
            patch("router_app.get_registry", return_value=registry),
            patch(
                "router_app.get_router",
                return_value=SimpleNamespace(waiting_requests=0),
            ),
        ):
            data = router_app._aggregate_dashboard()

        worker = data["workers"][0]
        self.assertEqual(worker["label"], "Chutes.ai")
        self.assertEqual(worker["priority_tier"], 45)
        self.assertIn("vision", worker["capabilities"])
        html = router_app._render_dashboard_html(data)
        self.assertIn("Chutes.ai", html)
        self.assertIn("tier 45", html)
        self.assertIn("Chutes API", html)
        self.assertNotIn("Chutes API [api]", html)
        self.assertIn("$12.35 remaining", html)
        self.assertIn("0/100", html)

    def test_chutes_and_local_qwen_share_dashboard_model_group(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(_local_qwen_worker())
        registry.register(router_app._build_chutes_worker(api_key="cpk_test"))

        with (
            patch("router_app.get_registry", return_value=registry),
            patch(
                "router_app.get_router",
                return_value=SimpleNamespace(waiting_requests=0),
            ),
        ):
            data = router_app._aggregate_dashboard()

        qwen_models = [m for m in data["models"] if m["model"] == "Qwen3.8-27B"]
        self.assertEqual(len(qwen_models), 1)
        self.assertEqual(qwen_models[0]["worker_count"], 2)

    def test_balance_parser_accepts_current_chutes_response(self):
        self.assertEqual(
            router_app._chutes_balance_from_payload({"balance": "19.875"}),
            19.875,
        )
        self.assertEqual(
            router_app._chutes_balance_from_payload(
                {"current_balance": {"effective_balance": 8.25}}
            ),
            8.25,
        )
        self.assertIsNone(router_app._chutes_balance_from_payload({"balance": "nan"}))


class ChutesRoutingRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_initializer_seeds_chutes_balance_after_registration(self):
        worker = router_app._build_chutes_worker(api_key="cpk_test")
        refresh = AsyncMock(return_value=14.5)

        with (
            patch("router_app._sync_chutes_worker", return_value=worker),
            patch("router_app._sync_openrouter_worker", return_value=None),
            patch("router_app._refresh_external_balance", refresh),
        ):
            initialized = await router_app._initialize_external_workers()

        self.assertEqual(initialized, [worker])
        refresh.assert_awaited_once_with(worker)

    async def test_balance_refresh_caches_users_me_response(self):
        worker = router_app._build_chutes_worker(api_key="cpk_test")
        captured = {}

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self, **_kwargs):
                return {"balance": 27.125}

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def get(self, url, headers):
                captured.update(url=url, headers=headers)
                return FakeResponse()

        with patch("router_app.aiohttp.ClientSession", return_value=FakeSession()):
            balance = await router_app._refresh_external_balance(worker)

        self.assertEqual(balance, 27.125)
        self.assertEqual(worker.external_balance_usd, 27.125)
        self.assertGreater(worker.external_balance_updated_at, 0)
        self.assertEqual(captured["url"], "https://api.chutes.ai/users/me")
        self.assertEqual(captured["headers"], {"Authorization": "Bearer cpk_test"})

    async def test_successful_chutes_completion_schedules_balance_refresh(self):
        registry = WorkerRegistry(ttl_seconds=60)
        worker = registry.register(router_app._build_chutes_worker(api_key="cpk_test"))
        response = router_app.Response(
            content=b'{"usage":{"prompt_tokens":2,"completion_tokens":3}}',
            status_code=200,
            media_type="application/json",
        )

        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app._pick", AsyncMock(return_value=worker)),
            patch("router_app._proxy_json", AsyncMock(return_value=response)),
            patch("router_app._record_llm_usage", AsyncMock()),
            patch("router_app._schedule_external_balance_refresh") as refresh,
        ):
            result = await router_app._llm_proxy_with_retry(
                capability="text",
                path="/v1/chat/completions",
                payload={"messages": [{"role": "user", "content": "hi"}]},
                model="Qwen3.8-27B",
                is_stream=False,
            )

        self.assertIs(result, response)
        refresh.assert_called_once_with(worker)

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

        model_stats = tracker.snapshot()["Chutes.ai"]["llm"]["Qwen3.8-27B"]
        self.assertEqual(model_stats["requests"], 1)
        self.assertEqual(model_stats["prompt_tokens"], 0)
        self.assertEqual(model_stats["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
