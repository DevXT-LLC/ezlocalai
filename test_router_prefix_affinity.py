"""Routing-only tests: no model, credentials, or GPU needed."""

import copy
import time
import unittest
from unittest.mock import AsyncMock, patch

from Router import WorkerInfo, WorkerRegistry
import router_app


def keys(text, model="model"):
    return router_app._system_prefix_affinity_keys(
        {"messages": [{"role": "system", "content": text}]}, "text", model
    )


class SystemPrefixTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.affinity_patch = patch.dict(router_app._prompt_affinity, {}, clear=True)
        self.affinity_patch.start()
        self.addCleanup(self.affinity_patch.stop)
        prefix_patch = patch.dict(router_app._system_prefix_affinity, {}, clear=True)
        prefix_patch.start()
        self.addCleanup(prefix_patch.stop)
        self.registry = WorkerRegistry(ttl_seconds=60)
        self.workers = []
        for name in ["short", "long", "cold"]:
            self.workers.append(
                self.registry.register(
                    WorkerInfo(
                        worker_id=name,
                        label=name,
                        url=f"http://{name}",
                        capabilities=["text"],
                        models=["model"],
                    )
                )
            )
        self.router = type(
            "Router", (), {"wait_for_worker": AsyncMock(return_value=self.workers[2])}
        )()
        for target, value in [
            ("get_registry", self.registry),
            ("get_router", self.router),
        ]:
            p = patch(f"router_app.{target}", return_value=value)
            p.start()
            self.addCleanup(p.stop)
        self.prefix = keys("a" * 16384 + "b" * 16384 + "conversation-specific suffix")

    def remember(self, prefix, worker):
        entries = (
            router_app._system_prefix_affinity
            if prefix in self.prefix
            else router_app._prompt_affinity
        )
        entries[prefix] = (worker.worker_id, time.time())

    def test_exact_incremental_prefix_preserves_suffix_independence(self):
        other = keys("a" * 16384 + "b" * 16384 + "different suffix")
        self.assertEqual(self.prefix[:2], other[:2])
        self.assertNotEqual(self.prefix[-1], other[-1])
        self.assertNotEqual(self.prefix[0], keys("A" + "a" * 16383)[0])
        self.assertNotEqual(self.prefix[0], keys("a" * 16384, "other-model")[0])

    def test_bounded_hashes_do_not_retain_prompt_or_modify_payload(self):
        payload = {
            "messages": [
                {"role": "system", "content": "secret-value " * 100000},
                {"role": "user", "content": "user input"},
            ]
        }
        before = copy.deepcopy(payload)
        result = router_app._system_prefix_affinity_keys(payload, "text", "model")
        self.assertEqual(len(result), 32)
        self.assertFalse(any("secret-value" in key for key in result))
        self.assertEqual(payload, before)

    def test_native_tools_and_template_settings_use_separate_cache_lanes(self):
        payload = {"messages": [{"role": "system", "content": "x" * 20000}]}
        baseline = router_app._system_prefix_affinity_keys(payload, "text", "model")
        for field, value in [
            ("tools", [{"type": "function", "function": {"name": "tool"}}]),
            ("chat_template_kwargs", {"enable_thinking": True}),
            ("chat_template", "custom"),
        ]:
            self.assertNotEqual(
                baseline,
                router_app._system_prefix_affinity_keys(
                    {**payload, field: value}, "text", "model"
                ),
            )

    def test_only_large_leading_text_system_messages_qualify(self):
        for messages in [
            None,
            [],
            "invalid",
            [None],
            [{"role": "user", "content": "x" * 20000}],
            [{"role": "system", "content": [{"type": "text", "text": "x" * 20000}]}],
            [{"role": "system", "content": "short"}],
        ]:
            self.assertEqual(
                router_app._system_prefix_affinity_keys(
                    {"messages": messages}, "text", "model"
                ),
                [],
            )
        self.assertEqual(
            router_app._system_prefix_affinity_keys(
                {"messages": [{"role": "system", "content": "x" * 20000}]},
                "tts",
                "model",
            ),
            [],
        )

    async def test_new_conversation_prefers_longest_matching_idle_prefix(self):
        self.remember(self.prefix[0], self.workers[0])
        self.remember(self.prefix[1], self.workers[1])
        worker = await router_app._pick(
            "text", "model", affinity_key="new", system_prefix_keys=self.prefix
        )
        self.assertEqual(worker.worker_id, "long")
        self.assertEqual(router_app._prompt_affinity["new"][0], "long")
        self.router.wait_for_worker.assert_not_awaited()

    async def test_busy_long_prefix_uses_idle_shorter_prefix_without_waiting(self):
        self.remember(self.prefix[0], self.workers[0])
        self.remember(self.prefix[1], self.workers[1])
        self.workers[1].queue_depth = 1
        with patch("router_app.asyncio.sleep", new_callable=AsyncMock) as sleep:
            worker = await router_app._pick(
                "text", "model", system_prefix_keys=self.prefix
            )
            sleep.assert_not_awaited()
        self.assertEqual(worker.worker_id, "short")

    async def test_busy_prefix_falls_back_to_normal_selector(self):
        self.remember(self.prefix[0], self.workers[0])
        self.workers[0].queue_depth = 1
        worker = await router_app._pick("text", "model", system_prefix_keys=self.prefix)
        self.assertEqual(worker.worker_id, "cold")
        self.router.wait_for_worker.assert_awaited_once()

    async def test_partial_prefix_does_not_downgrade_to_slower_hardware(self):
        self.remember(self.prefix[0], self.workers[0])
        self.workers[0].best_tier = 60
        self.workers[2].best_tier = 90
        worker = await router_app._pick("text", "model", system_prefix_keys=self.prefix)
        self.assertEqual(worker.worker_id, "cold")

    async def test_existing_conversation_owner_wins(self):
        self.remember("existing", self.workers[0])
        self.remember(self.prefix[-1], self.workers[1])
        worker = await router_app._pick(
            "text", "model", affinity_key="existing", system_prefix_keys=self.prefix
        )
        self.assertEqual(worker.worker_id, "short")

    async def test_busy_conversation_spills_normally_without_moving_home(self):
        self.remember("existing", self.workers[0])
        self.workers[0].queue_depth = 1
        self.remember(self.prefix[-1], self.workers[1])
        with patch("router_app._prompt_affinity_wait_timeout", return_value=0):
            worker = await router_app._pick(
                "text", "model", affinity_key="existing", system_prefix_keys=self.prefix
            )
        self.assertEqual(worker.worker_id, "cold")
        self.assertEqual(router_app._prompt_affinity["existing"][0], "short")

    async def test_excluded_parent_cache_worker_is_not_selected(self):
        self.remember(self.prefix[-1], self.workers[0])
        worker = await router_app._pick(
            "text", "model", exclude={"short"}, system_prefix_keys=self.prefix
        )
        self.assertEqual(worker.worker_id, "cold")

    async def test_stale_and_incompatible_prefixes_are_not_selected(self):
        router_app._system_prefix_affinity[self.prefix[0]] = ("short", 0)
        self.remember(self.prefix[-1], self.workers[1])
        self.workers[1].models = ["unrelated"]
        worker = await router_app._pick("text", "model", system_prefix_keys=self.prefix)
        self.assertEqual(worker.worker_id, "cold")

    def test_external_fallback_never_owns_system_prefix(self):
        self.workers[0].external_fallback = True
        router_app._remember_system_prefixes(self.prefix, self.workers[0])
        self.assertFalse(router_app._prompt_affinity)
        self.assertFalse(router_app._system_prefix_affinity)

    def test_prefix_hint_volume_does_not_evict_conversation_home(self):
        self.remember("existing", self.workers[0])
        for i in range(110):
            router_app._system_prefix_affinity[str(i)] = ("short", time.time())
        with patch.dict(router_app.os.environ, {"ROUTER_PROMPT_AFFINITY_MAX": "100"}):
            router_app._prune_prompt_affinity(time.time())
        self.assertEqual(len(router_app._system_prefix_affinity), 100)
        self.assertIn("existing", router_app._prompt_affinity)

    async def test_streaming_and_nonstreaming_requests_use_identical_prefix_hints(self):
        payload = {
            "model": "model",
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": "stable instructions " * 2000},
                {"role": "user", "content": "Do the task"},
            ],
            "prompt_cache_key": "fresh-conversation",
        }
        before = copy.deepcopy(payload)
        expected = router_app._system_prefix_affinity_keys(payload, "text", "model")

        async def fake_stream(worker, path, forwarded, **kwargs):
            self.assertEqual(forwarded, before)
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        for streaming in [False, True]:
            with self.subTest(streaming=streaming), patch(
                "router_app._pick", new_callable=AsyncMock, return_value=self.workers[0]
            ) as pick, patch(
                "router_app._proxy_json",
                new_callable=AsyncMock,
                return_value=router_app.JSONResponse({"choices": []}),
            ) as proxy, patch(
                "router_app._iter_worker_stream_bytes", fake_stream
            ), patch(
                "router_app._record_llm_usage", new_callable=AsyncMock
            ):
                response = await router_app._llm_proxy_with_retry(
                    capability="text",
                    path="/v1/chat/completions",
                    payload=payload,
                    model="model",
                    is_stream=streaming,
                )
                if streaming:
                    chunks = [chunk async for chunk in response.body_iterator]
                    self.assertTrue(any(b"ok" in chunk for chunk in chunks))
                else:
                    self.assertEqual(proxy.call_args.args[2], before)
                self.assertEqual(pick.call_args.kwargs["system_prefix_keys"], expected)
                self.assertEqual(payload, before)
                # Fake transports don't perform their production cleanup.
                self.workers[0].router_reservations.clear()
                self.workers[0]._refresh_router_reservations()


if __name__ == "__main__":
    unittest.main()
