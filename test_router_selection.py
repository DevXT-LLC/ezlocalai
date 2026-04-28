import unittest

from Router import Router, WorkerInfo, WorkerRegistry


class RouterSelectionTests(unittest.TestCase):
    def _router_with_worker(self, capability: str) -> Router:
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            WorkerInfo(
                worker_id=f"{capability}-worker",
                label=f"{capability.upper()} Worker",
                url="http://worker.local",
                capabilities=[capability],
                models=["some-llm-model"],
                cap_slots={
                    capability: {
                        "capacity": 1,
                        "in_flight": 0,
                        "queued": 0,
                        "available": 1,
                    }
                },
            )
        )
        return Router(registry)

    def test_stt_routes_by_capability_when_model_alias_is_not_advertised(self):
        router = self._router_with_worker("stt")

        worker = router.select_worker("stt", "whisper-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "STT Worker")

    def test_tts_routes_by_capability_when_model_alias_is_not_advertised(self):
        router = self._router_with_worker("tts")

        worker = router.select_worker("tts", "tts-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "TTS Worker")

    def test_image_routes_by_capability_when_model_alias_is_not_advertised(self):
        router = self._router_with_worker("image")

        worker = router.select_worker("image", "gpt-image-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "IMAGE Worker")

    def test_video_routes_by_capability_when_model_alias_is_not_advertised(self):
        router = self._router_with_worker("video")

        worker = router.select_worker("video", "sora", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "VIDEO Worker")

    def test_text_still_waits_for_requested_model_during_grace_period(self):
        router = self._router_with_worker("text")

        worker = router.select_worker(
            "text", "different-text-model", allow_cross_model=False
        )

        self.assertIsNone(worker)


if __name__ == "__main__":
    unittest.main()
