import asyncio
import threading
import types
import unittest
from unittest import mock

from Pipes import Pipes
from Router import WorkerInfo


class VideoAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_router_slots_unavailable_until_video_handoff_finishes(self):
        pipe = self.make_pipe()
        pipe.available_models = ["text-model"]
        pipe.persistent_llms = {"text-model": object()}
        pipe._llm_temporarily_unavailable = False
        pipe._inference_count_lock = threading.Lock()
        pipe._inference_count = 0
        pipe._model_inference_counts = {}
        pipe._resolved_parallel_for_model = lambda *a, **kw: 1
        pipe._resolve_source_model = lambda name: name
        pipe._is_vision_model = lambda name: False
        pipe._voice_should_unload_llm = lambda service: False
        pipe._voice_handoff_active = lambda: False
        pipe.resource_manager = types.SimpleNamespace(
            get_model_active_count=lambda _: 0
        )
        with (
            mock.patch("Pipes.getenv", side_effect=lambda key, default="": default),
            mock.patch("Pipes.has_voice_server_url", return_value=False),
            mock.patch("Pipes.has_embedding_server_url", return_value=False),
            mock.patch("Pipes.has_image_server_url", return_value=False),
            mock.patch("Pipes.is_image_enabled", return_value=False),
            mock.patch("Pipes.is_music_enabled", return_value=False),
            mock.patch("Pipes.is_video_enabled", return_value=True),
            mock.patch("Pipes.get_video_model_name", return_value="video-model"),
        ):
            idle = pipe.get_slot_capacity_snapshot()
            await pipe._video_lock.acquire()
            try:
                busy = pipe.get_slot_capacity_snapshot()
            finally:
                pipe._video_lock.release()
            restored = pipe.get_slot_capacity_snapshot()
        self.assertGreater(idle["cap_slots"]["embedding"]["available"], 0)
        self.assertEqual(busy["slot_total_available"], 0)
        self.assertEqual(busy["cap_slots"]["video"]["in_flight"], 1)
        self.assertEqual(restored, idle)
        worker = WorkerInfo(
            worker_id="gx10",
            label="gx10",
            url="http://worker",
            capabilities=list(busy["cap_slots"]),
            models=["text-model"],
            cap_slots=busy["cap_slots"],
            model_slots=busy["model_slots"],
        )
        self.assertEqual(worker.slots_left(capability="embedding"), 0)
        self.assertEqual(worker.slots_left(capability="text", model="text-model"), 0)

    def make_pipe(self):
        pipe = Pipes.__new__(Pipes)
        pipe._video_lock = asyncio.Lock()
        pipe._video_should_unload_llm_for_generation = lambda: False
        pipe._unload_llms_for_video = lambda: True
        pipe._unload_aux_models_for_video = lambda: {"embedding": True}
        pipe._reload_video_after_vram_handoff = mock.Mock()
        pipe._destroy_video = mock.Mock()
        pipe._restore_llms_after_video = mock.Mock()
        pipe._restore_aux_models_after_video = mock.Mock()
        return pipe

    async def wait_started(self, event):
        async def wait():
            while not event.is_set():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait(), timeout=2)

    async def test_event_loop_responsive_during_native_video(self):
        pipe = self.make_pipe()
        started, release = threading.Event(), threading.Event()
        loop_thread = threading.get_ident()
        native_threads = []

        def generate(**kwargs):
            native_threads.append(threading.get_ident())
            started.set()
            release.wait(timeout=3)
            return "outputs/video.mp4"

        pipe._generate_video_once = generate
        task = asyncio.create_task(pipe.generate_video("cup"))
        try:
            await self.wait_started(started)
            self.assertFalse(task.done())
            self.assertTrue(pipe._video_lock.locked())
            self.assertNotEqual(native_threads, [loop_thread])
            pipe._restore_llms_after_video.assert_not_called()
        finally:
            release.set()
        self.assertEqual(await task, "outputs/video.mp4")
        pipe._destroy_video.assert_called_once_with(async_cleanup=False, force=True)

    async def test_cancel_keeps_lock_and_models_until_native_completion(self):
        pipe = self.make_pipe()
        started, release = threading.Event(), threading.Event()
        calls = []

        def generate(**kwargs):
            calls.append(kwargs["prompt"])
            started.set()
            release.wait(timeout=3)
            return "outputs/video.mp4"

        pipe._generate_video_once = generate
        task = asyncio.create_task(pipe.generate_video("first"))
        second = None
        try:
            await self.wait_started(started)
            task.cancel()
            second = asyncio.create_task(pipe.generate_video("second"))
            await asyncio.sleep(0.02)
            task.cancel()  # Repeated cancellation must not release the lock.
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            self.assertEqual(calls, ["first"])
            pipe._destroy_video.assert_not_called()
            pipe._restore_aux_models_after_video.assert_not_called()
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        if second is not None:
            await second
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(pipe._destroy_video.call_count, 2)
        self.assertFalse(pipe._video_lock.locked())

    async def test_failure_releases_video_before_restoring_models(self):
        pipe = self.make_pipe()
        actions = mock.Mock()
        actions.attach_mock(pipe._destroy_video, "destroy")
        actions.attach_mock(pipe._restore_llms_after_video, "restore_llm")
        actions.attach_mock(pipe._restore_aux_models_after_video, "restore_aux")
        pipe._generate_video_once = mock.Mock(side_effect=RuntimeError("failed"))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await pipe.generate_video("cup")
        self.assertEqual(
            actions.mock_calls,
            [
                mock.call.destroy(async_cleanup=False, force=True),
                mock.call.restore_llm(True),
                mock.call.restore_aux({"embedding": True}),
            ],
        )
        self.assertFalse(pipe._video_lock.locked())
