import asyncio
import os
import threading
import time
import types
import unittest
from unittest import mock

from Pipes import ModelType, Pipes, _VoiceSlotGuard
from Router import WorkerInfo, WorkerRegistry


class LlmResidencyPolicyTests(unittest.TestCase):
    def test_auto_residency_selects_swap_when_models_overcommit_one_gpu(self):
        pipe = Pipes.__new__(Pipes)
        pipe.available_models = ["model-a", "model-b"]
        pipe.model_configs = {
            "model-a": {"main_gpu": 0, "max_tokens": 65536, "quant_type": "Q4"},
            "model-b": {"main_gpu": 0, "max_tokens": 65536, "quant_type": "Q4"},
        }
        pipe.model_sources = {"model-a": "model-a", "model-b": "model-b"}
        pipe.per_gpu_vram = [24.0]
        pipe._optimal_context = 65536
        pipe.llm_model_vram_estimates = {}

        with (
            mock.patch.dict(
                os.environ,
                {
                    "LLM_MODEL_RESIDENCY": "auto",
                    "LLM_MODEL_RESIDENCY_MARGIN_GB": "1.5",
                },
            ),
            mock.patch(
                "ezlocalai.LLM.download_model",
                side_effect=[("/tmp/a.gguf", None), ("/tmp/b.gguf", None)],
            ),
            mock.patch(
                "Pipes.estimate_model_vram_requirement", side_effect=[13.0, 14.0]
            ),
        ):
            pipe._configure_llm_model_residency()

        self.assertEqual(pipe.llm_model_residency, "swap")
        self.assertEqual(
            pipe.llm_model_vram_estimates, {"model-a": 13.0, "model-b": 14.0}
        )

    def test_swap_unloads_non_target_resident_model(self):
        pipe = Pipes.__new__(Pipes)
        resident = object()
        pipe.llm_model_residency = "swap"
        pipe.model_sources = {"model-a": "model-a", "model-b": "model-b"}
        pipe.persistent_llms = {"model-a": resident, "model-b": None}
        pipe.llm = resident
        pipe.current_llm_name = "model-a"
        pipe.current_context = 65536
        pipe.primary_llm = resident
        pipe.vision_llm = None
        pipe._using_large_model = False
        pipe._model_inference_counts = {"model-b": 1}
        pipe._inference_count_lock = threading.Lock()
        pipe.resource_manager = mock.Mock()
        pipe._record_model_lifecycle = mock.Mock()

        with mock.patch("Pipes.torch.cuda.is_available", return_value=False):
            pipe._unload_other_llms_for_swap_locked("model-b")

        self.assertIsNone(pipe.persistent_llms["model-a"])
        self.assertIsNone(pipe.llm)
        self.assertIsNone(pipe.primary_llm)
        pipe.resource_manager.unregister_model.assert_any_call(ModelType.LLM)


class LlmDependencySlotTests(unittest.TestCase):
    def test_active_llm_blocks_swap_models_and_handoff_services(self):
        pipe = Pipes.__new__(Pipes)
        pipe._llm_temporarily_unavailable = False
        pipe.llm_model_residency = "swap"
        pipe.available_models = ["model-a", "model-b"]
        pipe.persistent_llms = {"model-a": object(), "model-b": None}
        pipe.model_configs = {
            "model-a": {"n_parallel": 1, "max_tokens": 65536},
            "model-b": {"n_parallel": 1, "max_tokens": 65536},
        }
        pipe.model_sources = {"model-a": "model-a", "model-b": "model-b"}
        pipe._optimal_context = 65536
        pipe._inference_count = 1
        pipe._model_inference_counts = {"model-a": 1}
        pipe._inference_count_lock = threading.Lock()
        pipe.llm = types.SimpleNamespace(is_vision=False, n_parallel=1)
        pipe.current_llm_name = "model-a"
        pipe.resource_manager = mock.Mock()
        pipe.resource_manager.get_model_active_count.return_value = 0
        pipe._is_vision_model = mock.Mock(return_value=False)
        pipe._image_should_unload_llm_for_generation = mock.Mock(return_value=True)
        pipe._video_should_unload_llm_for_generation = mock.Mock(return_value=True)
        pipe._music_should_unload_llm_for_generation = mock.Mock(return_value=True)
        pipe._voice_should_unload_llm = mock.Mock(return_value=True)
        pipe._voice_handoff_active = mock.Mock(return_value=False)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "TTS_ENABLED": "true",
                    "STT_ENABLED": "true",
                    "EMBEDDING_ENABLED": "false",
                    "IMG_MODEL": "image-model",
                    "MUSIC_MODEL": "music-model",
                },
            ),
            mock.patch("Pipes.is_image_enabled", return_value=True),
            mock.patch("Pipes.is_video_enabled", return_value=True),
            mock.patch("Pipes.is_music_enabled", return_value=True),
            mock.patch("Pipes.has_image_server_url", return_value=False),
            mock.patch("Pipes.has_ace_step_server_url", return_value=True),
            mock.patch("Pipes.get_video_model_name", return_value="video-model"),
        ):
            snapshot = pipe.get_slot_capacity_snapshot()

        self.assertEqual(snapshot["llm_queue_capacity"], 1)
        self.assertEqual(snapshot["model_slots"]["model-a"]["available"], 0)
        self.assertEqual(snapshot["model_slots"]["model-b"]["available"], 0)
        self.assertEqual(snapshot["cap_slots"]["image"]["available"], 0)
        self.assertEqual(snapshot["cap_slots"]["video"]["available"], 0)
        self.assertEqual(snapshot["cap_slots"]["music"]["available"], 0)
        self.assertEqual(snapshot["cap_slots"]["tts"]["capacity"], 1)
        self.assertEqual(snapshot["cap_slots"]["tts"]["available"], 0)
        self.assertEqual(snapshot["cap_slots"]["stt"]["capacity"], 1)
        self.assertEqual(snapshot["cap_slots"]["stt"]["available"], 0)

    def test_router_text_reservation_blocks_declared_handoff_service(self):
        worker = WorkerInfo(
            worker_id="worker",
            label="worker",
            url="http://worker.local",
            capabilities=["text", "image"],
            models=["model-a"],
            cap_slots={
                "text": {"capacity": 1, "in_flight": 0, "queued": 0},
                "image": {"capacity": 1, "in_flight": 0, "queued": 0},
            },
            model_slots={"model-a": {"capacity": 1, "in_flight": 0, "queued": 0}},
            extra={"llm_unload_dependent_capabilities": ["image"]},
        )
        worker.router_reservations = {
            "request-1": {
                "expires_at": time.monotonic() + 15.0,
                "capability": "text",
                "model": "model-a",
            }
        }

        self.assertEqual(worker.slots_left(capability="image"), 0)

    def test_voice_reservation_excludes_text_and_sibling_voice_service(self):
        worker = WorkerInfo(
            worker_id="worker",
            label="worker",
            url="http://worker.local",
            capabilities=["text", "tts", "stt"],
            models=["model-a"],
            cap_slots={
                "text": {"capacity": 1, "in_flight": 0, "queued": 0},
                "tts": {"capacity": 1, "in_flight": 0, "queued": 0},
                "stt": {"capacity": 1, "in_flight": 0, "queued": 0},
            },
            model_slots={"model-a": {"capacity": 1, "in_flight": 0, "queued": 0}},
            extra={"llm_unload_dependent_capabilities": ["tts", "stt"]},
        )
        registry = WorkerRegistry(ttl_seconds=60, reservation_ttl_seconds=15)
        registry.register(worker)

        reservation = registry.try_reserve_in_flight(worker.worker_id, capability="tts")

        self.assertIsNotNone(reservation)
        self.assertEqual(worker.slots_left(capability="tts"), 0)
        self.assertEqual(worker.slots_left(capability="stt"), 0)
        self.assertEqual(worker.slots_left(capability="text", model="model-a"), 0)
        self.assertEqual(worker.total_capacity(), 1)
        self.assertEqual(worker.total_busy(), 1)


class VoiceHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_slot_unloads_voice_before_restoring_llm(self):
        pipe = Pipes.__new__(Pipes)
        pipe._voice_handoff_lock = asyncio.Lock()
        pipe._voice_handoff_state_lock = threading.Lock()
        pipe._voice_handoff_counts = {"tts": 0, "stt": 0}
        pipe._inference_count_lock = threading.Lock()
        pipe._llm_temporarily_unavailable = False
        pipe._voice_should_unload_llm = mock.Mock(return_value=True)
        pipe._wait_for_llm_idle_for_voice = mock.AsyncMock()
        handoff = {"loaded_models": ["model-a"], "active_model": "model-a"}
        pipe._unload_llms_for_service = mock.Mock(return_value=handoff)
        events = []
        pipe._destroy_tts = mock.Mock(side_effect=lambda **kwargs: events.append("tts"))
        pipe._destroy_stt = mock.Mock()
        pipe._restore_llms_after_service = mock.Mock(
            side_effect=lambda *args: events.append("llm")
        )
        guard = _VoiceSlotGuard(pipe, "tts", 1)

        async with guard:
            self.assertTrue(pipe._llm_temporarily_unavailable)
            self.assertTrue(pipe._voice_handoff_active("tts"))

        self.assertEqual(events, ["tts", "llm"])
        self.assertFalse(pipe._llm_temporarily_unavailable)
        self.assertFalse(pipe._voice_handoff_active())


class LlmSwapQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_model_request_waits_for_active_stream(self):
        pipe = Pipes.__new__(Pipes)
        pipe.llm_model_residency = "swap"
        pipe._inference_count = 1
        pipe._model_inference_counts = {"model-a": 1}
        pipe._inference_count_lock = threading.Lock()

        pending = asyncio.create_task(pipe._acquire_inference_slot("model-b"))
        await asyncio.sleep(0.02)
        self.assertFalse(pending.done())

        pipe._decrement_inference_count("model-a")
        await asyncio.wait_for(pending, timeout=0.5)

        self.assertEqual(pipe._inference_count, 1)
        self.assertEqual(pipe._model_inference_counts, {"model-b": 1})


class RouterReservationTests(unittest.TestCase):
    def test_atomic_reservation_blocks_cross_model_swap_race(self):
        registry = WorkerRegistry(ttl_seconds=60, reservation_ttl_seconds=15)
        registry.register(
            WorkerInfo(
                worker_id="worker",
                label="worker",
                url="http://worker.local",
                capabilities=["text"],
                models=["model-a", "model-b"],
                cap_slots={"text": {"capacity": 1, "in_flight": 0, "queued": 0}},
                model_slots={
                    "model-a": {"capacity": 1, "in_flight": 0, "queued": 0},
                    "model-b": {"capacity": 1, "in_flight": 0, "queued": 0},
                },
                extra={"llm_model_residency": "swap"},
            )
        )

        first = registry.try_reserve_in_flight(
            "worker", capability="text", model="model-a"
        )
        second = registry.try_reserve_in_flight(
            "worker", capability="text", model="model-b"
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)


class ImageHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_generation_restores_llm_after_forced_handoff(self):
        pipe = Pipes.__new__(Pipes)
        pipe._img_lock = asyncio.Lock()
        pipe.local_uri = "http://worker.local"
        pipe.resource_manager = mock.Mock()
        pipe._image_should_unload_llm_for_generation = mock.Mock(return_value=True)
        pipe._wait_for_llm_idle_for_image = mock.AsyncMock()
        handoff = {"loaded_models": ["model-a"], "active_model": "model-a"}
        pipe._unload_llms_for_service = mock.Mock(return_value=handoff)
        pipe._unload_aux_models_for_image = mock.Mock(return_value={})
        image_model = mock.Mock()
        image_model.generate.return_value = "http://worker.local/outputs/image.png"
        pipe._get_img = mock.Mock(return_value=image_model)
        pipe._destroy_img = mock.Mock()
        pipe._restore_llms_after_service = mock.Mock()
        pipe._restore_aux_models_after_image = mock.Mock()

        result = await pipe.generate_image("a stopwatch")

        self.assertEqual(result, "http://worker.local/outputs/image.png")
        pipe._destroy_img.assert_called_once_with(async_cleanup=False, force=True)
        pipe._restore_llms_after_service.assert_called_once_with(
            "image", handoff, "IMAGE_RELOAD_LLM_AFTER_GENERATION"
        )


if __name__ == "__main__":
    unittest.main()
