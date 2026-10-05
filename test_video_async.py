import asyncio
import threading
import unittest
from unittest import mock

from Pipes import Pipes


class VideoAsyncTests(unittest.IsolatedAsyncioTestCase):
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
