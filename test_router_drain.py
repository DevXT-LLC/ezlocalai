"""Maintenance admission must not race active streams or new worker dispatch."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import router_app
from Router import Router, WorkerInfo, WorkerRegistry, WorkerDrainingError


def worker(worker_id="one", label="DevXT3090J"):
    return WorkerInfo(
        worker_id=worker_id,
        label=label,
        url="http://worker",
        capabilities=["text", "vision"],
        models=["model"],
        best_tier=80,
    )


class DrainRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = WorkerRegistry(60, reservation_ttl_seconds=0.01)
        self.one = self.registry.register(worker())
        self.two = self.registry.register(worker("two", "DevXT5090"))

    def test_long_stream_remains_busy_after_capacity_lease_expires(self):
        token = self.registry.try_reserve_in_flight("one", "text", "model")
        self.one.router_reservations[token]["expires_at"] = 0
        self.assertEqual(self.one.effective_busy(), 0)
        self.assertEqual(self.one.maintenance_busy(), 1)
        with self.assertRaises(WorkerDrainingError):
            self.registry.drain(self.one.label, "rollout", require_idle=True)
        self.registry.release_in_flight("one", token)
        self.assertEqual(self.one.maintenance_busy(), 0)
        self.assertTrue(
            self.registry.drain(self.one.label, "rollout", require_idle=True)["ok"]
        )

    def test_drain_blocks_selection_affinity_and_dispatch_but_keeps_liveness(self):
        self.registry.drain(self.one.label, "rollout")
        self.assertTrue(self.one.is_alive(60))
        self.assertFalse(self.one.has_capacity("text", "model"))
        self.assertIsNone(self.registry.try_reserve_in_flight("one", "text", "model"))
        with self.assertRaises(WorkerDrainingError):
            self.registry.increment_in_flight("one", capability="text")
        selected = Router(self.registry).select_worker("text", "model")
        self.assertEqual(selected.worker_id, "two")
        self.assertFalse(
            router_app._eligible_affinity_worker(self.one, "text", "model", set())
        )
        self.assertTrue(self.one.to_public()["draining"])

    def test_active_request_finishes_normally_while_drained(self):
        token = self.registry.try_reserve_in_flight("one", "text", "model")
        self.registry.drain(self.one.label, "rollout")
        self.assertEqual(self.one.maintenance_busy(), 1)
        self.registry.release_in_flight("one", token)
        self.assertEqual(self.one.maintenance_busy(), 0)
        self.registry.resume(self.one.label, "rollout")
        self.assertFalse(self.one.is_draining())

    def test_heartbeat_and_new_worker_id_cannot_clear_drain(self):
        self.registry.drain(self.one.label, "rollout")
        self.registry.heartbeat("one", {"drain_owner": "", "drain_until": 0})
        self.assertTrue(self.one.is_draining())
        self.registry.deregister("one")
        replacement = self.registry.register(worker("replacement"))
        self.assertTrue(replacement.is_draining())
        self.registry.resume(replacement.label, "rollout")
        self.assertFalse(replacement.is_draining())

    def test_only_one_owner_and_one_worker_can_be_drained(self):
        self.registry.drain(self.one.label, "rollout")
        for label, owner in [(self.one.label, "other"), (self.two.label, "rollout")]:
            with self.assertRaises(WorkerDrainingError):
                self.registry.drain(label, owner)
        with self.assertRaises(WorkerDrainingError):
            self.registry.resume(self.one.label, "other")
        self.registry.deregister("one")
        self.assertTrue(self.registry.drain(self.one.label, "rollout")["ok"])

    def test_expired_controller_lease_recovers(self):
        self.registry.drain(self.one.label, "rollout")
        with patch("Router.time.time", return_value=time.time() + 121):
            self.assertFalse(self.one.is_draining())
            self.assertTrue(self.one.has_capacity())

    def test_drain_survives_router_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "drains.json")
            first = WorkerRegistry(60, drain_state_path=path)
            first.register(worker())
            first.drain(self.one.label, "rollout")
            second = WorkerRegistry(60, drain_state_path=path)
            replacement = second.register(worker("new-process"))
            self.assertTrue(replacement.is_draining())
            second.resume(replacement.label, "rollout")
            third = WorkerRegistry(60, drain_state_path=path)
            self.assertFalse(third.register(worker()).is_draining())

    def test_failed_persistence_does_not_acknowledge_a_drain(self):
        with patch.object(
            self.registry, "_save_drains", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                self.registry.drain(self.one.label, "rollout")
        self.assertFalse(self.one.is_draining())
        self.assertEqual(self.registry._drains, {})


class DrainApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_auth_idle_conflict_and_resume(self):
        registry = WorkerRegistry(60)
        target = registry.register(worker())
        with patch.object(
            router_app, "get_registry", return_value=registry
        ), patch.object(
            router_app, "_expected_register_key", return_value="test-drain-key"
        ), patch.object(
            router_app, "_expected_client_key", return_value=""
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=router_app.app),
                base_url="http://router",
            ) as client:
                payload = {
                    "label": target.label,
                    "owner": "rollout",
                    "require_idle": True,
                }
                self.assertEqual(
                    (await client.post("/v1/router/drain", json=payload)).status_code,
                    401,
                )
                headers = {"Authorization": "Bearer test-drain-key"}
                token = registry.try_reserve_in_flight(
                    target.worker_id, "text", "model"
                )
                self.assertEqual(
                    (
                        await client.post(
                            "/v1/router/drain", json=payload, headers=headers
                        )
                    ).status_code,
                    409,
                )
                registry.release_in_flight(target.worker_id, token)
                self.assertEqual(
                    (
                        await client.post(
                            "/v1/router/drain", json=payload, headers=headers
                        )
                    ).status_code,
                    200,
                )
                snapshot = (await client.get("/v1/router/workers")).json()
                self.assertEqual(snapshot["maintenance_protocol"], 1)
                self.assertTrue(snapshot["data"][0]["draining"])
                self.assertNotIn("drain_owner", snapshot["data"][0])
                self.assertEqual(
                    (
                        await client.post(
                            "/v1/router/resume", json=payload, headers=headers
                        )
                    ).status_code,
                    200,
                )
                self.assertFalse(target.is_draining())


if __name__ == "__main__":
    unittest.main()
