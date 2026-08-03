import unittest

from Router import WorkerInfo
from router_app import (
    _public_with_tunnel,
    _render_dashboard_html,
    _tier_badge,
    _worker_priority_tier,
)


def _dashboard_data(**overrides):
    data = {
        "pool_health": "healthy",
        "router": {"ttl_seconds": 60, "wait_timeout": 30},
        "totals": {
            "alive_workers": 1,
            "stale_workers": 0,
            "total_parallel_capacity": 1,
            "total_available_slots": 1,
            "total_in_flight": 0,
            "total_queue_depth": 0,
            "total_free_vram_gb": 0,
            "total_vram_gb": 0,
            "unique_models": 0,
        },
        "models": [],
        "workers": [],
        "usage": {},
        "usage_24h": {},
        "history": [],
        "errors": [],
    }
    data.update(overrides)
    return data


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

    def test_worker_models_column_shows_music_model(self):
        worker = WorkerInfo(
            worker_id="music-1",
            label="Studio",
            url="http://studio.local",
            capabilities=["music"],
            cap_models={"music": "Serveurperso/ACE-Step-1.5-GGUF"},
            cap_slots={
                "music": {
                    "capacity": 1,
                    "in_flight": 0,
                    "queued": 0,
                    "available": 1,
                }
            },
        )
        html = _render_dashboard_html(
            _dashboard_data(workers=[_public_with_tunnel(worker)])
        )

        self.assertIn('title="Serveurperso/ACE-Step-1.5-GGUF"', html)
        self.assertIn(">ACE-Step-1.5</span>", html)

    def test_worker_models_column_hides_music_video_duplicate_when_video_exists(self):
        worker = WorkerInfo(
            worker_id="video-1",
            label="Stage",
            url="http://stage.local",
            capabilities=["video", "music", "music_video"],
            cap_models={
                "video": "unsloth/LTX-2.3-GGUF",
                "music": "Serveurperso/ACE-Step-1.5-GGUF",
                "music_video": (
                    "Serveurperso/ACE-Step-1.5-GGUF + unsloth/LTX-2.3-GGUF"
                ),
            },
            cap_slots={
                "video": {"capacity": 1, "in_flight": 0, "queued": 0, "available": 1},
                "music": {"capacity": 1, "in_flight": 0, "queued": 0, "available": 1},
                "music_video": {
                    "capacity": 1,
                    "in_flight": 0,
                    "queued": 0,
                    "available": 1,
                },
            },
        )

        html = _render_dashboard_html(
            _dashboard_data(workers=[_public_with_tunnel(worker)])
        )

        self.assertIn('title="unsloth/LTX-2.3-GGUF"', html)
        self.assertIn('title="Serveurperso/ACE-Step-1.5-GGUF"', html)
        self.assertNotIn("Music Video", html)
        self.assertNotIn(
            'title="Serveurperso/ACE-Step-1.5-GGUF + unsloth/LTX-2.3-GGUF"',
            html,
        )

    def test_usage_summary_shows_music_requests(self):
        html = _render_dashboard_html(
            _dashboard_data(
                usage={
                    "Studio": {
                        "music": {"requests": 3},
                    }
                }
            )
        )

        self.assertIn("Music reqs", html)
        self.assertIn('<td class="num">3</td>', html)

    def test_usage_summary_counts_music_video_requests_as_video(self):
        html = _render_dashboard_html(
            _dashboard_data(
                usage={
                    "Studio": {
                        "video": {"requests": 2},
                        "music_video": {"requests": 3},
                    }
                }
            )
        )

        self.assertIn("Video reqs", html)
        self.assertNotIn("Music video reqs", html)
        self.assertIn('<td class="num">5</td>', html)


if __name__ == "__main__":
    unittest.main()
