import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Router import Router, WorkerInfo, WorkerRegistry, gpu_tier_for_name
from router_app import router_heartbeat, router_register


class RouterGPUTierTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_gb10_registration_and_heartbeats_keep_tier_50(self):
        registry = WorkerRegistry(ttl_seconds=60)
        gpu = {"index": 0, "name": "NVIDIA GB10", "backend": "cuda", "tier": 20}
        payload = {
            "worker_id": "gb10",
            "url": "http://worker.local",
            "capabilities": ["stt"],
            "gpus": [gpu],
            "best_tier": 20,
        }
        request = SimpleNamespace(client=SimpleNamespace(host="worker.local"))
        with patch("router_app.get_registry", return_value=registry):
            registered = await router_register(request, payload)
            self.assertEqual(registered["worker"]["best_tier"], 50)
            for _ in range(2):
                heartbeat = await router_heartbeat(payload)
                self.assertEqual(heartbeat["worker"]["gpus"][0]["tier"], 50)
                self.assertEqual(heartbeat["worker"]["best_tier"], 50)
                self.assertEqual(heartbeat["worker"]["priority_tier"], 50)

        # The corrected tier must affect routing, not just dashboard rendering.
        registry.register(
            WorkerInfo(
                worker_id="tier-40",
                label="tier-40",
                url="http://other.local",
                capabilities=["stt"],
                best_tier=40,
            )
        )
        self.assertEqual(Router(registry).select_worker("stt").worker_id, "gb10")
        self.assertEqual(gpu["tier"], 20)
        self.assertEqual(gpu_tier_for_name("NVIDIA GB10"), 50)

    async def test_tier_correction_preserves_other_hardware_and_reported_tiers(self):
        cases = [
            ("NVIDIA GB10", "cuda", 0, 50, 50),
            ("NVIDIA GB10", "cuda", 0, 45, 45),
            ("Unknown Future GPU", "cuda", 0, 80, 80),
            ("Unknown Future GPU", "cuda", 0, 20, 20),
            ("", "cuda", 0, 20, 20),
            ("CPU", "cpu", -1, 2, 2),
        ]
        for name, backend, index, reported, expected in cases:
            with self.subTest(name=name, reported=reported):
                registry = WorkerRegistry(ttl_seconds=60)
                gpus = [dict(name=name, backend=backend, index=index, tier=reported)]
                worker = registry.register(
                    WorkerInfo(
                        worker_id="worker",
                        label="worker",
                        url="http://worker.local",
                        gpus=gpus,
                        best_tier=reported,
                    )
                )
                self.assertEqual(worker.best_tier, expected)
                registry.heartbeat("worker", {"gpus": gpus})
                self.assertEqual(worker.best_tier, expected)


if __name__ == "__main__":
    unittest.main()
