import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Router import WorkerInfo, WorkerRegistry
from router_app import (
    UsageTracker,
    _aggregate_dashboard,
    _normalize_model_name,
    _public_with_tunnel,
    _render_dashboard_html,
    _tier_badge,
    _usage_from_history,
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
    def test_router_waiters_are_included_in_total_queue_depth(self):
        registry = WorkerRegistry(ttl_seconds=60)

        with (
            patch("router_app.get_registry", return_value=registry),
            patch(
                "router_app.get_router",
                return_value=SimpleNamespace(waiting_requests=3),
            ),
        ):
            data = _aggregate_dashboard()

        self.assertEqual(data["router"]["waiting_requests"], 3)
        self.assertEqual(data["totals"]["total_queue_depth"], 3)

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

    def test_router_reservation_is_consistent_across_dashboard_sections(self):
        idle = {"capacity": 1, "in_flight": 0, "queued": 0, "available": 1}
        registry = WorkerRegistry(ttl_seconds=60)
        worker = registry.register(
            WorkerInfo(
                worker_id="studio-1",
                label="Studio",
                url="http://studio.local",
                capabilities=["text", "tts", "stt"],
                models=["model-a"],
                cap_models={"tts": "tts-model", "stt": "stt-model"},
                cap_slots={
                    "text": dict(idle),
                    "tts": dict(idle),
                    "stt": dict(idle),
                },
                model_slots={"model-a": dict(idle)},
            )
        )
        registry.increment_in_flight(
            worker.worker_id, capability="text", model="model-a"
        )

        with patch("router_app.get_registry", return_value=registry):
            data = _aggregate_dashboard()

        models = {(entry["type"], entry["model"]): entry for entry in data["models"]}
        self.assertEqual(models[("text", "model-a")]["available_slots"], 0)
        self.assertEqual(models[("tts", "tts-model")]["available_slots"], 1)
        self.assertEqual(models[("stt", "stt-model")]["available_slots"], 1)

        html = _render_dashboard_html(data)
        self.assertIn(
            'title="1 in use of 1">1/1</b> ' '<span class="mono small" title="model-a"',
            html,
        )
        self.assertIn(
            'title="1 free">0/1</span> ' '<span class="mono small" title="tts-model"',
            html,
        )
        self.assertIn(
            'title="1 free">0/1</span> ' '<span class="mono small" title="stt-model"',
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

    def test_model_breakdown_shows_media_capability_models(self):
        html = _render_dashboard_html(
            _dashboard_data(
                usage={
                    "Studio": {
                        "tts": {"requests": 2},
                        "tts_models": {
                            "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {
                                "requests": 2,
                                "outputs": 2,
                                "total_ms_sum": 2500,
                            }
                        },
                        "image": {"requests": 1},
                        "image_models": {
                            "unsloth/FLUX.2-klein-4B-GGUF": {
                                "requests": 1,
                                "outputs": 1,
                                "total_ms_sum": 10_000,
                            }
                        },
                        "stt": {"requests": 1},
                        "stt_models": {
                            "Whisper large-v3-turbo": {
                                "requests": 1,
                                "outputs": 1,
                                "total_ms_sum": 5000,
                            }
                        },
                        "music": {"requests": 1},
                        "music_models": {
                            "Serveurperso/ACE-Step-1.5-GGUF": {
                                "requests": 1,
                                "outputs": 1,
                                "total_ms_sum": 120_000,
                            }
                        },
                    }
                }
            )
        )

        self.assertIn("Text-to-Speech", html)
        self.assertIn("Image Generation", html)
        self.assertIn("Speech-to-Text", html)
        self.assertIn("Music", html)
        self.assertIn("Qwen3-TTS-12Hz-0.6B-Base", html)
        self.assertIn("FLUX.2-klein-4B", html)
        self.assertIn("ACE-Step-1.5", html)
        self.assertIn("2.00m", html)

    def test_recent_requests_show_media_rows(self):
        html = _render_dashboard_html(
            _dashboard_data(
                history=[
                    {
                        "ts": 1000,
                        "kind": "image",
                        "worker": "Studio",
                        "model": "unsloth/FLUX.2-klein-4B-GGUF",
                        "outputs": 2,
                        "total_ms": 2500,
                    },
                    {
                        "ts": 1001,
                        "kind": "music_video",
                        "worker": "Studio",
                        "model": "unsloth/LTX-2.3-GGUF",
                        "outputs": 1,
                        "total_ms": 142_700,
                    },
                ]
            )
        )

        self.assertIn("<th>Time</th><th>Worker</th><th>Type</th><th>Model</th>", html)
        self.assertIn("Image Generation", html)
        self.assertIn("Video", html)
        self.assertIn("FLUX.2-klein-4B", html)
        self.assertIn("LTX-2.3", html)
        self.assertIn("2.38m", html)

    def test_mtp_and_non_mtp_models_share_usage_bucket(self):
        self.assertEqual(
            _normalize_model_name("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"),
            "unsloth/Qwen3.6-35B-A3B",
        )
        html = _render_dashboard_html(
            _dashboard_data(
                usage={
                    "Studio": {
                        "llm": {
                            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": {
                                "requests": 1,
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                            },
                            "unsloth/Qwen3.6-35B-A3B-GGUF": {
                                "requests": 2,
                                "prompt_tokens": 20,
                                "completion_tokens": 4,
                            },
                        }
                    }
                }
            )
        )

        self.assertIn("Qwen3.6-35B-A3B", html)
        self.assertIn('<td class="num small">3</td>', html)
        self.assertNotIn("Qwen3.6-35B-A3B-MTP", html)


class UsageTrackerMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_record_cap_adds_media_model_stats_and_history(self):
        tracker = UsageTracker("/tmp/ezlocalai-test-usage.json")

        await tracker.record_cap(
            "Studio",
            "image",
            model="unsloth/FLUX.2-klein-4B-GGUF",
            total_ms=1234,
            outputs=2,
        )

        snapshot = tracker.snapshot()
        model_stats = snapshot["Studio"]["image_models"]["unsloth/FLUX.2-klein-4B"]
        self.assertEqual(model_stats["requests"], 1)
        self.assertEqual(model_stats["outputs"], 2)
        history = tracker.history_snapshot()
        self.assertEqual(history[0]["kind"], "image")
        self.assertEqual(history[0]["model"], "unsloth/FLUX.2-klein-4B")

        rebuilt = _usage_from_history(history)
        self.assertEqual(
            rebuilt["Studio"]["image_models"]["unsloth/FLUX.2-klein-4B"]["requests"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
