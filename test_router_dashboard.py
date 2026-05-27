import unittest

from Router import WorkerInfo
from router_app import _public_with_tunnel, _tier_badge, _worker_priority_tier


class RouterDashboardTierTests(unittest.TestCase):
    def test_tunneled_worker_public_data_has_adjusted_priority_tier(self):
        worker = WorkerInfo(
            worker_id="tunnel-90",
            label="Tunnel 90",
            url="tunnel://tunnel-90",
            capabilities=["text"],
            best_tier=90,
        )

        public = _public_with_tunnel(worker)

        self.assertTrue(public["tunnel"])
        self.assertEqual(public["best_tier"], 90)
        self.assertEqual(public["priority_tier"], 85)
        self.assertEqual(_worker_priority_tier(public), 85)

    def test_tier_badge_shows_adjusted_and_hardware_tier(self):
        badge = _tier_badge(
            {
                "best_tier": 55,
                "priority_tier": 50,
                "tunnel": True,
            }
        )

        self.assertIn("priority tier 50", badge)
        self.assertIn("hw 55", badge)
        self.assertIn("tunnel penalty -5", badge)


if __name__ == "__main__":
    unittest.main()
