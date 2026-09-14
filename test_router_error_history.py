import os
import tempfile
import unittest
from unittest.mock import patch

from Router import WorkerInfo, WorkerRegistry
from router_app import _aggregate_recent_errors


class ErrorHistoryTests(unittest.TestCase):
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
