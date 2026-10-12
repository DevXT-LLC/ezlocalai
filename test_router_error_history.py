import os
import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Router import WorkerInfo, WorkerRegistry
from router_app import (
    _aggregate_dashboard,
    _aggregate_recent_errors,
    _render_dashboard_html,
    router_errors,
)


class ErrorHistoryTests(unittest.TestCase):
    def test_24_hour_cutoff_applies_to_archived_and_live_only_errors(self):
        now = 1800000000.0
        registry = WorkerRegistry(60)
        worker = registry.register(
            WorkerInfo(worker_id="one", label="Worker One", url="http://unused")
        )
        for timestamp, message in (
            (now - 86401, "expired"),
            (now - 86400, "exact cutoff"),
            (now - 86399, "inside window"),
        ):
            with patch("Router.time.time", return_value=timestamp):
                registry.record_error("one", "stream", "/test", message)
        worker.recent_errors.extend(
            [
                {"ts": now - 86400, "message": "expired live only"},
                {"ts": now - 1, "message": "recent live only"},
            ]
        )
        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app.time.time", return_value=now),
        ):
            self.assertEqual(
                [e["message"] for e in _aggregate_recent_errors([worker])],
                ["recent live only", "inside window"],
            )
        # Expiry needs no new failures or worker activity.
        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app.time.time", return_value=now + 86400),
        ):
            self.assertEqual(_aggregate_recent_errors([worker]), [])
        self.assertEqual(len(registry.error_history()), 3)
        self.assertEqual(worker.total_errors, 3)

    def test_dashboard_and_error_api_counts_expire_before_display_limits(self):
        now = 1800000000.0
        registry = WorkerRegistry(60)
        with patch("Router.time.time", return_value=now):
            worker = registry.register(
                WorkerInfo(worker_id="one", label="Worker One", url="http://unused")
            )
            for i in range(125):
                registry.record_error("one", "stream", "/test", f"failure {i}")
        with (
            patch("router_app.get_registry", return_value=registry),
            patch(
                "router_app.get_router",
                return_value=SimpleNamespace(waiting_requests=0),
            ),
            patch("router_app.time.time", return_value=now + 31),
        ):
            dashboard = _aggregate_dashboard()
            errors = asyncio.run(router_errors())
        for data in (dashboard, errors):
            self.assertEqual(data["workers"][0]["recent_error_count"], 125)
            self.assertEqual(len(data["workers"][0]["recent_errors"]), 125)
            self.assertEqual(len(data["errors"]), 100)
        self.assertLess(len(worker.recent_errors), 125)
        self.assertIn(">125</span>", _render_dashboard_html(dashboard))
        with (
            patch("router_app.get_registry", return_value=registry),
            patch(
                "router_app.get_router",
                return_value=SimpleNamespace(waiting_requests=0),
            ),
            patch("router_app.time.time", return_value=now + 86400),
        ):
            dashboard = _aggregate_dashboard()
            errors = asyncio.run(router_errors())
        for data in (dashboard, errors):
            self.assertEqual(data["workers"][0]["recent_error_count"], 0)
            self.assertEqual(data["workers"][0]["recent_errors"], [])
            self.assertEqual(data["workers"][0]["total_errors"], 125)
            self.assertEqual(data["errors"], [])
        html = _render_dashboard_html(dashboard)
        self.assertNotIn(">125</span>", html)
        self.assertNotIn("Recent errors · past 24 hours", html)

    def test_restarted_workers_do_not_duplicate_or_revive_expired_archive(self):
        now = 1800000000.0
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "errors.json")
            registry = WorkerRegistry(60, error_history_path=path)
            registry.register(
                WorkerInfo(worker_id="old", label="Worker One", url="http://unused")
            )
            with patch("Router.time.time", return_value=now):
                registry.record_error("old", "stream", "/test", "failure")
                restored = WorkerRegistry(60, error_history_path=path)
                worker = restored.register(
                    WorkerInfo(worker_id="new", label="Worker One", url="http://unused")
                )
            self.assertEqual(len(worker.recent_errors), 1)
            with (
                patch("router_app.get_registry", return_value=restored),
                patch("router_app.time.time", return_value=now),
            ):
                data = asyncio.run(router_errors())
                self.assertEqual(len(data["errors"]), 1)
                self.assertTrue(data["errors"][0]["offline"])
                self.assertEqual(data["workers"][0]["recent_error_count"], 1)
            with (
                patch("router_app.get_registry", return_value=restored),
                patch("router_app.time.time", return_value=now + 86400),
            ):
                self.assertEqual(_aggregate_recent_errors([worker]), [])
                self.assertEqual(_aggregate_recent_errors([]), [])

    def test_history_survives_registration_pruning_and_router_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "errors.json")
            registry = WorkerRegistry(60, error_history_path=path)
            worker = WorkerInfo(
                worker_id="one", label="Worker One", url="http://unused"
            )
            registry.register(worker)
            registry.record_error("one", "stream", "/test", "interrupted")
            replacement = WorkerInfo(
                worker_id="one", label="Worker One", url="http://unused"
            )
            registry.register(replacement)
            self.assertEqual(replacement.total_errors, 1)
            replacement.last_heartbeat = 0
            registry.prune()
            self.assertTrue(registry.error_history()[0]["offline"])
            # A late stream exception after pruning must also be retained.
            registry.record_error("one", "stream", "/test", "late failure")
            restored = WorkerRegistry(60, error_history_path=path)
            self.assertEqual(len(restored.error_history()), 2)
            self.assertEqual(restored.error_history()[0]["label"], "Worker One")
            with patch("router_app.get_registry", return_value=restored):
                self.assertEqual(len(_aggregate_recent_errors([])), 2)

    def test_bounded_history_and_deregistration(self):
        with patch.dict(os.environ, {"ROUTER_ERROR_ARCHIVE_MAX": "2"}):
            registry = WorkerRegistry(60)
        registry.register(WorkerInfo(worker_id="one", label="one", url="http://unused"))
        for i in range(3):
            registry.record_error("one", "stream", "/test", str(i))
        registry.deregister("one")
        self.assertEqual([e["message"] for e in registry.error_history()], ["2", "1"])

    def test_dashboard_does_not_duplicate_live_errors(self):
        registry = WorkerRegistry(60)
        registry.register(WorkerInfo(worker_id="one", label="one", url="http://unused"))
        registry.record_error("one", "stream", "/test", "failure")
        with patch("router_app.get_registry", return_value=registry):
            self.assertEqual(
                len(_aggregate_recent_errors(registry.list_workers(False))), 1
            )

    def test_new_worker_id_inherits_crash_loop_cooldown(self):
        with patch.dict(
            os.environ,
            {
                "ROUTER_ERROR_THRESHOLD": "3",
                "ROUTER_ERROR_WINDOW_SECONDS": "60",
                "ROUTER_CIRCUIT_COOLDOWN": "30",
            },
        ):
            registry = WorkerRegistry(60)
            original = registry.register(
                WorkerInfo(worker_id="old-id", label="4090", url="http://worker")
            )
            for _ in range(3):
                registry.record_error(
                    original.worker_id, "stream", "/v1/chat/completions", "crash"
                )
            registry.deregister(original.worker_id)
            replacement = registry.register(
                WorkerInfo(worker_id="new-id", label="4090", url="http://worker")
            )
        self.assertEqual(replacement.total_errors, 3)
        self.assertEqual(len(replacement.recent_errors), 3)
        self.assertTrue(replacement.is_circuit_open())
