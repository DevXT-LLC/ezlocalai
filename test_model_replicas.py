"""Speculative replicas must own independent contexts and request lifetimes."""

import asyncio
import os
import threading
import types
import unittest
from unittest import mock

from test_pipes_stream_tracking import PipesStreamTrackingTests
from Pipes import Pipes
from Router import Router, WorkerInfo, WorkerRegistry

MODEL = "unsloth/Qwen3.8-27B-GGUF"


def configured_pipe(models=None, parallels="3", backend="mtp"):
    pipe = Pipes.__new__(Pipes)
    pipe.available_models = models or [MODEL]
    pipe.model_sources = {mid: mid.split("#")[0] for mid in pipe.available_models}
    pipe.model_replicas = {}
    for mid, source in pipe.model_sources.items():
        pipe.model_replicas.setdefault(source, []).append(mid)
    pipe.model_configs = {}
    env = {
        "MAIN_GPU": "0,1",
        "LLM_MAX_TOKENS": "262144,32768",
        "N_PARALLEL": parallels,
        "QUANT_TYPE": "Q3_K_XL,Q8_0",
    }
    with mock.patch(
        "Pipes.getenv", side_effect=lambda key, default=None: env.get(key, default)
    ), mock.patch("Pipes.get_gpu_count", return_value=2), mock.patch(
        "Pipes.speculative_backend", return_value=backend
    ):
        pipe._parse_per_model_configs()
    return pipe


class ReplicaConfigurationTests(unittest.TestCase):
    def test_mtp_and_dflash_expand_without_shifting_csv_assignments(self):
        for backend in ("mtp", "dflash"):
            with self.subTest(backend=backend):
                pipe = configured_pipe([MODEL, "other/model"], "3,1", backend)
                self.assertEqual(len(pipe.available_models), 4)
                self.assertEqual(len(pipe.model_replicas[MODEL]), 3)
                for mid in pipe.model_replicas[MODEL]:
                    cfg = pipe.model_configs[mid]
                    self.assertEqual(cfg["n_parallel"], 1)
                    self.assertEqual(cfg["max_tokens"], 262144)
                    self.assertEqual(cfg["quant_type"], "Q3_K_XL")
                    self.assertEqual(cfg["main_gpu"], 0)
                other = pipe.model_configs["other/model"]
                self.assertEqual(other["main_gpu"], 1)
                self.assertEqual(other["max_tokens"], 32768)
                self.assertEqual(other["quant_type"], "Q8_0")

    def test_existing_duplicates_get_unique_ids(self):
        pipe = configured_pipe([MODEL, MODEL + "#2"], "3,2")
        self.assertEqual(len(set(pipe.available_models)), 5)
        self.assertEqual(len(pipe.model_replicas[MODEL]), 5)

    def test_non_speculative_native_parallelism_is_preserved(self):
        pipe = configured_pipe(backend="none")
        self.assertEqual(pipe.available_models, [MODEL])
        self.assertEqual(pipe.model_configs[MODEL]["n_parallel"], 3)

    def test_speculative_auto_remains_one_instance(self):
        pipe = configured_pipe(parallels="0")
        self.assertEqual(pipe.available_models, [MODEL])
        self.assertEqual(pipe.model_configs[MODEL]["n_parallel"], 1)

    def test_three_22gb_instances_fit_h100_but_swap_on_40gb(self):
        pipe = configured_pipe()
        pipe._estimate_configured_llm_vram = mock.Mock(return_value=22)
        with mock.patch.dict(
            os.environ,
            {"LLM_MODEL_RESIDENCY": "auto", "LLM_MODEL_RESIDENCY_MARGIN_GB": "1.5"},
        ):
            for capacity, expected in ((80, "resident"), (40, "swap")):
                pipe.per_gpu_vram = [capacity]
                pipe._configure_llm_model_residency()
                self.assertEqual(pipe.llm_model_residency, expected)


class ReplicaRequestTests(unittest.IsolatedAsyncioTestCase):
    def pipe(self, streaming=True):
        pipe = PipesStreamTrackingTests._pipe(None)
        cfg = configured_pipe()
        for name in (
            "available_models",
            "model_configs",
            "model_sources",
            "model_replicas",
        ):
            setattr(pipe, name, getattr(cfg, name))
        pipe.llm_model_residency = "resident"
        pipe._resolve_slot_model = types.MethodType(Pipes._resolve_slot_model, pipe)
        pipe._get_response_internal = types.MethodType(
            Pipes._get_response_internal, pipe
        )
        pipe.local_uri = "http://localhost"
        pipe._using_large_model = False
        pipe.current_llm_name = MODEL
        pipe._is_vision_model = mock.Mock(return_value=False)
        pipe._find_non_vision_model = mock.Mock(return_value=None)
        pipe.persistent_llms = {}
        for mid in pipe.available_models:

            def chat(_mid=mid, **data):
                payload = {
                    "model": data["model"],
                    "replica": _mid,
                    "choices": [{"message": {"content": _mid}}],
                }
                return iter([payload]) if data.get("stream") else payload

            pipe.persistent_llms[mid] = types.SimpleNamespace(
                n_parallel=1, is_vision=False, params={}, chat=chat, completion=chat
            )

        def get_llm(mid, context):
            self.assertEqual(context, 262144)
            pipe.llm = pipe.persistent_llms[mid]
            pipe.current_llm_name = mid
            pipe.current_context = context
            return pipe.llm

        pipe._get_llm = mock.Mock(side_effect=get_llm)
        pipe._ensure_context_size = mock.Mock(
            side_effect=AssertionError("must not reload sibling")
        )
        pipe._reduce_gpu_layers = mock.Mock(
            side_effect=AssertionError("must not unload sibling")
        )
        return pipe

    def request(self, stream=True):
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
        }

    async def test_three_streams_lease_distinct_instances_fourth_waits(self):
        pipe = self.pipe()
        responses = await asyncio.gather(
            *(pipe.get_response(self.request()) for _ in range(3))
        )
        self.assertEqual(pipe._inference_count, 3)
        self.assertEqual(set(pipe._replica_inference_counts.values()), {1})
        chunks = [next(response[0]) for response in responses]
        self.assertEqual({c["replica"] for c in chunks}, set(pipe.available_models))
        self.assertEqual({c["model"] for c in chunks}, {MODEL})
        fourth = asyncio.create_task(pipe.get_response(self.request()))
        await asyncio.sleep(0.06)
        self.assertFalse(fourth.done())
        responses[1][0].close()
        replacement, _ = await asyncio.wait_for(fourth, 1)
        self.assertEqual(next(replacement)["replica"], chunks[1]["replica"])
        replacement.close()
        for response, _ in responses:
            response.close()
        self.assertEqual(pipe._inference_count, 0)
        self.assertEqual(pipe._replica_inference_counts, {})

    async def test_explicit_internal_replica_waits_for_that_instance(self):
        pipe = self.pipe()
        data = self.request()
        data["model"] = MODEL + "#2"
        first, _ = await pipe.get_response(data)
        self.assertEqual(next(first)["replica"], MODEL + "#2")
        data = self.request()
        data["model"] = MODEL + "#2"
        pending = asyncio.create_task(pipe.get_response(data))
        await asyncio.sleep(0.06)
        self.assertFalse(pending.done())
        first.close()
        second, _ = await asyncio.wait_for(pending, 1)
        self.assertEqual(next(second)["replica"], MODEL + "#2")
        second.close()
        self.assertEqual(pipe._replica_inference_counts, {})

    async def test_request_keeps_its_instance_across_async_preprocessing(self):
        pipe = self.pipe()
        arrived = 0
        barrier = asyncio.Event()

        async def describe(*args):
            nonlocal arrived
            arrived += 1
            if arrived == 3:
                barrier.set()
            await barrier.wait()
            return "an image"

        pipe._describe_images_with_vision_model = describe

        def req():
            data = self.request(False)
            data["messages"][0]["content"] = [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,eA=="},
                },
            ]
            return data

        results = await asyncio.wait_for(
            asyncio.gather(*(pipe.get_response(req()) for _ in range(3))), 2
        )
        self.assertEqual({r[0]["replica"] for r in results}, set(pipe.available_models))
        self.assertEqual(pipe._inference_count, 0)

    async def test_stream_context_error_releases_only_its_lease(self):
        pipe = self.pipe()

        def fail(**data):
            yield {"first": True}
            raise RuntimeError(
                "context size exceeded [n_prompt_tokens=300000, n_ctx=262144]"
            )

        pipe.persistent_llms[MODEL].chat = fail
        first, _ = await pipe.get_response(self.request())
        second, _ = await pipe.get_response(self.request())
        next(first)
        with self.assertRaisesRegex(RuntimeError, "context size"):
            next(first)
        self.assertEqual(pipe._inference_count, 1)
        self.assertEqual(next(second)["replica"], MODEL + "#2")
        second.close()
        pipe._ensure_context_size.assert_not_called()

    async def test_cancelled_native_call_holds_replica_until_thread_finishes(self):
        pipe = self.pipe()
        started, release = threading.Event(), threading.Event()

        def blocking(**data):
            started.set()
            release.wait(5)
            return {"choices": []}

        pipe.persistent_llms[MODEL].chat = blocking
        task = asyncio.create_task(pipe.get_response(self.request(False)))
        try:
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            self.assertEqual(pipe._replica_inference_counts, {MODEL: 1})
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(pipe._replica_inference_counts, {})

    async def test_stream_close_holds_lease_until_native_cancellation_finishes(self):
        pipe = self.pipe()
        native_done = threading.Event()
        pipe.persistent_llms[MODEL]._native_stream_done = native_done
        stream, _ = await pipe.get_response(self.request())
        stream.close()
        self.assertEqual(pipe._replica_inference_counts, {MODEL: 1})
        native_done.set()
        for _ in range(100):
            if pipe._inference_count == 0:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(pipe._inference_count, 0)
        self.assertEqual(pipe._replica_inference_counts, {})

    def test_snapshot_reports_resident_instances_and_full_public_model(self):
        pipe = self.pipe()
        pipe.resource_manager.get_model_active_count.return_value = 0
        pipe._is_vision_model.return_value = True
        pipe.llm = pipe.persistent_llms[MODEL]
        disabled = {
            key + "_ENABLED": "false"
            for key in ("TTS", "STT", "EMBEDDING", "IMAGE", "VIDEO", "MUSIC")
        }
        with mock.patch.dict(os.environ, disabled):
            snapshot = pipe.get_slot_capacity_snapshot()
        self.assertEqual(list(snapshot["model_slots"]), [MODEL])
        for state in (
            snapshot["model_slots"][MODEL],
            snapshot["cap_slots"]["text"],
            snapshot["cap_slots"]["vision"],
        ):
            self.assertEqual(state["capacity"], 3)
            self.assertEqual(state["instances"], 3)
        self.assertEqual(snapshot["slot_total_capacity"], 3)
        self.assertEqual(len(pipe.get_models()["data"]), 1)

    async def test_failed_replica_is_not_advertised_or_reserved(self):
        pipe = self.pipe()
        failed = MODEL + "#3"
        pipe.persistent_llms.pop(failed)
        pipe._failed_replica_ids = {failed}
        self.assertEqual(pipe._inference_capacity_for_model(MODEL), 2)
        slots = [await pipe._acquire_inference_slot(MODEL) for _ in range(2)]
        self.assertNotIn(failed, slots)
        for mid in slots:
            pipe._decrement_inference_count(MODEL, mid)


class ReplicaRouterTests(unittest.TestCase):
    def test_router_uses_idle_replica_and_reserves_last_slot(self):
        registry = WorkerRegistry(ttl_seconds=60)
        worker = WorkerInfo(
            worker_id="h100",
            label="H100",
            url="http://worker",
            models=[MODEL],
            capabilities=["text", "vision"],
        )
        registry.register(worker)
        registry.heartbeat(
            "h100",
            {"model_slots": {MODEL: {"capacity": 3, "in_flight": 2, "instances": 3}}},
        )
        router = Router(registry)
        with mock.patch.dict(os.environ, {"ROUTER_BUSY_SLOT_FALLBACK": "false"}):
            self.assertIs(router.select_worker(model=MODEL, capability="text"), worker)
            self.assertIs(
                router.select_worker(model=MODEL, capability="vision"), worker
            )
            registry.heartbeat(
                "h100",
                {
                    "model_slots": {
                        MODEL: {"capacity": 3, "in_flight": 3, "instances": 3}
                    }
                },
            )
            self.assertIsNone(router.select_worker(model=MODEL, capability="text"))
            registry.heartbeat(
                "h100", {"model_slots": {MODEL: {"capacity": 3, "in_flight": 1}}}
            )
            self.assertIsNone(router.select_worker(model=MODEL, capability="text"))
