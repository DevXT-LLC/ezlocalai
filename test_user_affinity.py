import asyncio
import time
import unittest
from unittest.mock import patch

from fastapi.responses import JSONResponse
from Router import Router, WorkerInfo, WorkerRegistry
from UserAffinity import UserAffinity, UserLane
import router_app


class LaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_cancel_and_independent_users(self):
        lanes = UserAffinity()
        entered = []
        gate = asyncio.Event()

        async def run(key, number):
            async with lanes.acquire(key) as lane:
                entered.append(number)
                if number == 0:
                    lane.worker_id = "gpu"
                    await gate.wait()
                if key == "same":
                    self.assertEqual(lane.worker_id, "gpu")

        first = asyncio.create_task(run("same", 0))
        await asyncio.sleep(0)
        pending = [asyncio.create_task(run("same", n)) for n in range(1, 5)]
        await asyncio.sleep(0)
        pending[1].cancel()
        await asyncio.gather(pending[1], return_exceptions=True)
        await run("other", 9)
        self.assertEqual(entered, [0, 9])
        gate.set()
        await asyncio.gather(first, *[p for p in pending if not p.cancelled()])
        self.assertEqual(entered, [0, 9, 1, 3, 4])
        self.assertEqual(lanes.lanes["same"].users, 0)

    async def test_idle_expiry_does_not_expire_active_or_waiting_lanes(self):
        lanes = UserAffinity()
        async with lanes.acquire("one", ttl=1) as first:
            first.worker_id = "gpu"
            first.touched = time.monotonic() - 100
            async with lanes.acquire("two", ttl=1):
                self.assertIs(lanes.lanes["one"], first)
        async with lanes.acquire("one", ttl=1) as reused:
            self.assertIs(reused, first)
        first.touched = time.monotonic() - 100
        async with lanes.acquire("one", ttl=1) as expired:
            self.assertIsNot(expired, first)
            self.assertIsNone(expired.worker_id)

    async def test_bounded_queues_do_not_evict_active_lanes(self):
        lanes = UserAffinity()
        async with lanes.acquire("one", max_pending=1):
            with self.assertRaises(OverflowError):
                async with lanes.acquire("one", max_pending=1):
                    self.fail("must not enter")
            with self.assertRaises(OverflowError):
                async with lanes.acquire("two", max_entries=1):
                    self.fail("must not enter")
        async with lanes.acquire("two", max_entries=1):
            self.assertEqual(list(lanes.lanes), ["two"])


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = WorkerRegistry(ttl_seconds=60)
        self.workers = [
            self.registry.register(
                WorkerInfo(
                    worker_id=name,
                    label=name,
                    url=f"http://{name}",
                    capabilities=["text", "vision"],
                    models=["model"],
                )
            )
            for name in ("home", "other")
        ]
        for name, value in [
            ("get_registry", self.registry),
            ("get_router", Router(self.registry)),
        ]:
            patcher = patch.object(router_app, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in [
            ("_user_affinity", UserAffinity()),
            ("_prompt_affinity", {}),
            ("_system_prefix_affinity", {}),
        ]:
            patcher = patch.object(router_app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(router_app, "_wait_timeout", return_value=0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_model_scoping_validation_and_metadata_stripping(self):
        payload = {"routing_affinity_key": "user", "prompt_cache_key": "project"}
        self.assertNotEqual(
            router_app._user_affinity_key(payload, "27b"),
            router_app._user_affinity_key(payload, "2b"),
        )
        self.assertNotIn("user", router_app._user_affinity_key(payload, "27b"))
        self.assertIsNone(router_app._user_affinity_key({}, "model"))
        for invalid in ("", [], "x" * 513):
            with self.assertRaises(router_app.HTTPException):
                router_app._user_affinity_key(
                    {"routing_affinity_key": invalid}, "model"
                )
        self.assertEqual(
            router_app._worker_json_payload(
                self.workers[0], "/v1/chat/completions", payload
            ),
            {},
        )
        self.assertIn("routing_affinity_key", payload)

    async def test_busy_home_waits_instead_of_using_idle_gpu_and_offline_fails_over(
        self,
    ):
        home, other = self.workers
        lane = UserLane(worker_id=home.worker_id)
        home.queue_depth = 1

        async def free_home():
            await asyncio.sleep(0.15)
            home.queue_depth = 0

        task = asyncio.create_task(free_home())
        with patch.object(router_app, "_prompt_affinity_wait_timeout", return_value=0):
            worker, reservation = await asyncio.wait_for(
                router_app._pick_and_reserve_llm("text", "model", user_lane=lane), 2
            )
        await task
        self.assertIs(worker, home)
        self.registry.release_in_flight(worker.worker_id, reservation)
        home.last_heartbeat = time.time() - 120
        worker, reservation = await asyncio.wait_for(
            router_app._pick_and_reserve_llm("text", "model", user_lane=lane), 2
        )
        self.assertIs(worker, other)
        self.assertEqual(lane.worker_id, other.worker_id)
        self.registry.release_in_flight(worker.worker_id, reservation)

    async def test_json_projects_share_worker_and_execute_fifo(self):
        entered = []
        gate = asyncio.Event()

        async def proxy(worker, path, payload, **kwargs):
            try:
                entered.append((payload["prompt_cache_key"], worker.worker_id))
                if len(entered) == 1:
                    await gate.wait()
                return JSONResponse({"ok": True})
            finally:
                self.registry.release_in_flight(
                    worker.worker_id, kwargs["reservation_id"]
                )

        async def request(project):
            return await router_app._llm_proxy_with_retry(
                capability="text",
                path="/v1/chat/completions",
                model="model",
                is_stream=False,
                payload={
                    "routing_affinity_key": "same-user",
                    "prompt_cache_key": project,
                },
            )

        with patch.object(router_app, "_proxy_json", side_effect=proxy):
            first = asyncio.create_task(request("one"))
            await asyncio.sleep(0)
            second = asyncio.create_task(request("two"))
            third = asyncio.create_task(request("three"))
            await asyncio.sleep(0.01)
            self.assertEqual(len(entered), 1)
            gate.set()
            await asyncio.gather(first, second, third)
        self.assertEqual([project for project, _ in entered], ["one", "two", "three"])
        self.assertEqual(len({worker for _, worker in entered}), 1)

    async def test_full_queue_returns_retryable_errors_and_explicit_target_bypasses_it(
        self,
    ):
        payload = {"routing_affinity_key": "same-user"}
        key = router_app._user_affinity_key(payload, "model")
        options = dict(
            capability="text",
            path="/v1/chat/completions",
            model="model",
            payload=payload,
        )
        with patch.dict("os.environ", {"ROUTER_USER_QUEUE_MAX": "1"}):
            async with router_app._user_affinity.acquire(key):
                with self.assertRaises(router_app.HTTPException) as raised:
                    await router_app._llm_proxy_with_retry(**options, is_stream=False)
                self.assertEqual(raised.exception.status_code, 429)
                self.assertEqual(raised.exception.headers["Retry-After"], "1")
                response = await router_app._llm_proxy_with_retry(
                    **options, is_stream=True
                )
                body = b"".join([chunk async for chunk in response.body_iterator])
                self.assertIn(b'"code": 429', body)
                self.assertIn(b"[DONE]", body)

                async def proxy(worker, path, payload, **kwargs):
                    self.assertEqual(worker.worker_id, "other")
                    self.registry.release_in_flight(
                        worker.worker_id, kwargs["reservation_id"]
                    )
                    return JSONResponse({"ok": True})

                with patch.object(router_app, "_proxy_json", side_effect=proxy):
                    response = await asyncio.wait_for(
                        router_app._llm_proxy_with_retry(
                            **options, is_stream=False, worker_id="other"
                        ),
                        2,
                    )
                self.assertEqual(response.status_code, 200)

    async def test_excluded_home_migrates_lane_on_retry(self):
        lane = UserLane(worker_id="home")
        worker, reservation = await asyncio.wait_for(
            router_app._pick_and_reserve_llm(
                "text", "model", user_lane=lane, exclude={"home"}, retrying=True
            ),
            2,
        )
        self.assertEqual(worker.worker_id, "other")
        self.assertEqual(lane.worker_id, "other")
        self.registry.release_in_flight(worker.worker_id, reservation)

    async def test_stream_disconnect_releases_lane_and_waiting_stream_gets_keepalive(
        self,
    ):
        entered = []
        closed = []

        async def stream(**kwargs):
            project = kwargs["payload"]["prompt_cache_key"]
            entered.append(project)
            try:
                yield b'data: {"choices":[]}\n\n'
                await asyncio.Event().wait()
            finally:
                closed.append(project)

        async def request(project):
            return await router_app._llm_proxy_with_retry(
                capability="text",
                path="/v1/chat/completions",
                model="model",
                is_stream=True,
                payload={
                    "routing_affinity_key": "same-user",
                    "prompt_cache_key": project,
                },
            )

        original_keepalive = router_app._with_stream_keepalive
        with patch.object(
            router_app, "_llm_stream_with_worker_failover", stream
        ), patch.object(
            router_app,
            "_with_stream_keepalive",
            lambda s: original_keepalive(s, interval=0.01),
        ):
            first, second = await request("one"), await request("two")
            self.assertTrue((await anext(first.body_iterator)).startswith(b"data:"))
            self.assertEqual(await anext(second.body_iterator), b": keepalive\n\n")
            self.assertEqual(entered, ["one"])
            await first.body_iterator.aclose()
            self.assertEqual(closed, ["one"])
            self.assertTrue((await anext(second.body_iterator)).startswith(b"data:"))
            await second.body_iterator.aclose()
        self.assertEqual(closed, ["one", "two"])
        self.assertTrue(
            all(lane.users == 0 for lane in router_app._user_affinity.lanes.values())
        )


if __name__ == "__main__":
    unittest.main()
