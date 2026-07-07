import asyncio
import threading
import time
import unittest

from ezlocalai.STT import STT


class STTAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_audio_runs_sync_work_off_event_loop(self):
        stt = object.__new__(STT)
        event_loop_thread = threading.get_ident()
        sync_thread = None

        def fake_transcribe_sync(**kwargs):
            nonlocal sync_thread
            sync_thread = threading.get_ident()
            time.sleep(0.05)
            return kwargs["base64_audio"]

        stt._transcribe_audio_sync = fake_transcribe_sync

        task = asyncio.create_task(stt.transcribe_audio("ok"))
        tick = await asyncio.wait_for(
            asyncio.sleep(0.01, result="event-loop-free"),
            timeout=0.1,
        )
        result = await task

        self.assertEqual(tick, "event-loop-free")
        self.assertEqual(result, "ok")
        self.assertIsNotNone(sync_thread)
        self.assertNotEqual(sync_thread, event_loop_thread)


if __name__ == "__main__":
    unittest.main()
