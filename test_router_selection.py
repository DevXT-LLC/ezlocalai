import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from Router import (
    Router,
    WorkerInfo,
    WorkerRegistry,
    _configured_contexts_from_pipe,
    _usable_context_from_config,
    _version_from_git_metadata,
    detect_local_capabilities,
)


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

    def _text_worker(
        self,
        worker_id: str,
        model: str,
        *,
        best_tier: int,
        capacity: int = 1,
        busy: int = 0,
        url=None,
    ) -> WorkerInfo:
        return WorkerInfo(
            worker_id=worker_id,
            label=worker_id,
            url=url or f"http://{worker_id}.local",
            capabilities=["text"],
            models=[model],
            best_tier=best_tier,
            free_vram_gb=24.0,
            cap_slots={
                "text": {
                    "capacity": capacity,
                    "in_flight": busy,
                    "queued": 0,
                    "available": max(0, capacity - busy),
                }
            },
            model_slots={
                model: {
                    "capacity": capacity,
                    "in_flight": busy,
                    "queued": 0,
                    "available": max(0, capacity - busy),
                }
            },
        )

    def _capability_worker(
        self,
        worker_id: str,
        capabilities,
        *,
        best_tier: int,
        busy_by_cap=None,
    ) -> WorkerInfo:
        busy_by_cap = busy_by_cap or {}
        cap_slots = {}
        for cap in capabilities:
            busy = int(busy_by_cap.get(cap, 0))
            cap_slots[cap] = {
                "capacity": 1,
                "in_flight": busy,
                "queued": 0,
                "available": max(0, 1 - busy),
            }
        return WorkerInfo(
            worker_id=worker_id,
            label=worker_id,
            url=f"http://{worker_id}.local",
            capabilities=list(capabilities),
            models=["model-a"] if "text" in capabilities else [],
            best_tier=best_tier,
            free_vram_gb=24.0,
            cap_slots=cap_slots,
        )

    def test_tunneled_worker_priority_tier_is_penalized(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(self._text_worker("direct-86", "model-a", best_tier=86))
        tunnel_worker = registry.register(
            self._text_worker(
                "tunnel-90",
                "model-a",
                best_tier=90,
                url="tunnel://tunnel-90",
            )
        )
        router = Router(registry)

        worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "direct-86")
        self.assertEqual(tunnel_worker.priority_tier, 85)

    def test_tunneled_worker_penalty_applies_to_idle_spillover_window(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "tunnel-90",
                "model-a",
                best_tier=90,
                capacity=2,
                busy=1,
                url="tunnel://tunnel-90",
            )
        )
        registry.register(self._text_worker("direct-66", "model-a", best_tier=66))
        router = Router(registry)

        worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "direct-66")

    def test_busy_high_tier_spills_to_best_idle_worker_by_default(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "busy-5090",
                "model-a",
                best_tier=90,
                capacity=3,
                busy=1,
            )
        )
        registry.register(self._text_worker("idle-3090", "model-a", best_tier=55))
        router = Router(registry)

        worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "idle-3090")

    def test_idle_tier_window_can_hold_back_distant_idle_worker(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "busy-5090",
                "model-a",
                best_tier=90,
                capacity=3,
                busy=1,
            )
        )
        registry.register(self._text_worker("idle-3090", "model-a", best_tier=55))
        router = Router(registry)

        with patch.dict(os.environ, {"ROUTER_IDLE_TIER_WINDOW": "20"}):
            worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNone(worker)

    def test_busy_5090_spills_to_idle_4090_before_idle_3090(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "busy-5090",
                "model-a",
                best_tier=90,
                capacity=3,
                busy=1,
            )
        )
        registry.register(self._text_worker("idle-3090", "model-a", best_tier=55))
        registry.register(self._text_worker("idle-4090", "model-a", best_tier=80))
        router = Router(registry)

        worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "idle-4090")

    def test_all_busy_text_workers_queue_even_with_open_parallel_slots(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "busy-5090",
                "model-a",
                best_tier=90,
                capacity=3,
                busy=1,
            )
        )
        registry.register(
            self._text_worker(
                "busy-4090",
                "model-a",
                best_tier=80,
                capacity=2,
                busy=1,
            )
        )
        router = Router(registry)

        worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNone(worker)

    def test_busy_slot_fallback_can_restore_capacity_routing(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "busy-5090",
                "model-a",
                best_tier=90,
                capacity=3,
                busy=1,
            )
        )
        registry.register(self._text_worker("idle-3090", "model-a", best_tier=55))
        router = Router(registry)

        with patch.dict(
            os.environ,
            {"ROUTER_BUSY_SLOT_FALLBACK": "true", "ROUTER_IDLE_TIER_WINDOW": "0"},
        ):
            worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "busy-5090")

    def test_wide_idle_tier_window_allows_idle_3090_spillover(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "busy-5090",
                "model-a",
                best_tier=90,
                capacity=3,
                busy=1,
            )
        )
        registry.register(self._text_worker("idle-3090", "model-a", best_tier=55))
        router = Router(registry)

        with patch.dict(os.environ, {"ROUTER_IDLE_TIER_WINDOW": "100"}):
            worker = router.select_worker("text", "model-a", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "idle-3090")

    def test_stt_routes_by_capability_when_model_alias_is_not_advertised(self):
        router = self._router_with_worker("stt")

        worker = router.select_worker("stt", "whisper-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "STT Worker")

    def test_stt_prefers_dedicated_voice_worker_over_high_tier_text_worker(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._capability_worker(
                "mixed-5090", ["text", "vision", "stt"], best_tier=90
            )
        )
        registry.register(
            self._capability_worker("voice-3090", ["tts", "stt"], best_tier=55)
        )
        router = Router(registry)

        worker = router.select_worker("stt", "whisper-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "voice-3090")

    def test_stt_falls_back_to_mixed_worker_when_dedicated_voice_is_busy(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._capability_worker(
                "mixed-5090", ["text", "vision", "stt"], best_tier=90
            )
        )
        registry.register(
            self._capability_worker(
                "voice-3090",
                ["tts", "stt"],
                best_tier=55,
                busy_by_cap={"stt": 1},
            )
        )
        router = Router(registry)

        worker = router.select_worker("stt", "whisper-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "mixed-5090")

    def test_dedicated_voice_preference_can_be_disabled(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._capability_worker(
                "mixed-5090", ["text", "vision", "stt"], best_tier=90
            )
        )
        registry.register(
            self._capability_worker("voice-3090", ["tts", "stt"], best_tier=55)
        )
        router = Router(registry)

        with patch.dict(os.environ, {"ROUTER_PREFER_DEDICATED_CAPABILITIES": "none"}):
            worker = router.select_worker("stt", "whisper-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "mixed-5090")

    def test_tts_routes_by_capability_when_model_alias_is_not_advertised(self):
        router = self._router_with_worker("tts")

        worker = router.select_worker("tts", "tts-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "TTS Worker")

    def test_tts_prefers_dedicated_voice_worker_over_high_tier_text_worker(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._capability_worker(
                "mixed-5090", ["text", "vision", "tts"], best_tier=90
            )
        )
        registry.register(
            self._capability_worker("voice-3090", ["tts", "stt"], best_tier=55)
        )
        router = Router(registry)

        worker = router.select_worker("tts", "tts-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "voice-3090")

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

    def test_embedding_routes_by_capability_when_model_alias_is_not_advertised(self):
        router = self._router_with_worker("embedding")

        worker = router.select_worker(
            "embedding", "text-embedding-3-large", allow_cross_model=False
        )

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "EMBEDDING Worker")

    def test_text_still_waits_for_requested_model_during_grace_period(self):
        router = self._router_with_worker("text")

        worker = router.select_worker(
            "text", "different-text-model", allow_cross_model=False
        )

        self.assertIsNone(worker)

    def test_wait_for_worker_uses_available_fallback_without_default_grace(self):
        router = self._router_with_worker("text")

        worker = asyncio.run(
            router.wait_for_worker("text", "different-text-model", timeout=0)
        )

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "TEXT Worker")

    def test_wait_for_worker_positive_timeout_returns_none_when_pool_empty(self):
        router = Router(WorkerRegistry(ttl_seconds=60))

        worker = asyncio.run(
            router.wait_for_worker(
                "text", "model-a", timeout=0.01, poll_interval=0.001
            )
        )

        self.assertIsNone(worker)

    def test_text_prefers_mtp_variant_for_base_model_request(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "non-mtp-5090",
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                best_tier=90,
            )
        )
        registry.register(
            self._text_worker(
                "mtp-3090",
                "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
                best_tier=50,
            )
        )
        router = Router(registry)

        worker = router.select_worker(
            "text", "Qwen3.6-35B-A3B", allow_cross_model=False
        )

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "mtp-3090")

    def test_text_prefers_mtp_variant_for_non_mtp_gguf_request(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "non-mtp-5090",
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                best_tier=90,
            )
        )
        registry.register(
            self._text_worker(
                "mtp-3090",
                "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
                best_tier=50,
            )
        )
        router = Router(registry)

        worker = router.select_worker(
            "text", "unsloth/Qwen3.6-35B-A3B-GGUF", allow_cross_model=False
        )

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "mtp-3090")

    def test_text_uses_non_mtp_variant_when_mtp_is_saturated(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "non-mtp-5090",
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                best_tier=90,
            )
        )
        registry.register(
            self._text_worker(
                "mtp-3090",
                "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
                best_tier=50,
                busy=1,
            )
        )
        router = Router(registry)

        worker = router.select_worker(
            "text", "Qwen3.6-35B-A3B", allow_cross_model=False
        )

        self.assertIsNotNone(worker)
        self.assertEqual(worker.worker_id, "non-mtp-5090")

    def test_explicit_mtp_request_does_not_count_non_mtp_as_same_model(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            self._text_worker(
                "non-mtp-5090",
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                best_tier=90,
            )
        )
        router = Router(registry)

        worker = router.select_worker(
            "text",
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            allow_cross_model=False,
        )

        self.assertIsNone(worker)


class RuntimeVersionTests(unittest.TestCase):
    def test_version_from_lightweight_git_ref_metadata(self):
        sha = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as tmp:
            refs_dir = os.path.join(tmp, ".git", "refs", "heads")
            os.makedirs(refs_dir)
            with open(os.path.join(tmp, ".git", "HEAD"), "w", encoding="utf-8") as fh:
                fh.write("ref: refs/heads/main\n")
            with open(os.path.join(refs_dir, "main"), "w", encoding="utf-8") as fh:
                fh.write(f"{sha}\n")

            self.assertEqual(_version_from_git_metadata(tmp), sha[:7])

    def test_version_from_lightweight_packed_refs_metadata(self):
        sha = "abcdef0123456789abcdef0123456789abcdef01"
        with tempfile.TemporaryDirectory() as tmp:
            git_dir = os.path.join(tmp, ".git")
            os.makedirs(git_dir)
            with open(os.path.join(git_dir, "HEAD"), "w", encoding="utf-8") as fh:
                fh.write("ref: refs/heads/main\n")
            with open(
                os.path.join(git_dir, "packed-refs"), "w", encoding="utf-8"
            ) as fh:
                fh.write("# pack-refs with: peeled fully-peeled sorted\n")
                fh.write(f"{sha} refs/heads/main\n")

            self.assertEqual(_version_from_git_metadata(tmp), sha[:7])


class ConfiguredContextTests(unittest.TestCase):
    def test_context_from_config_divides_by_parallel_slots(self):
        self.assertEqual(_usable_context_from_config("1010000", "4"), 252500)

    def test_context_from_config_matches_auto_parallel(self):
        self.assertEqual(_usable_context_from_config("65536", "0"), 32768)

    def test_configured_contexts_from_pipe_use_unloaded_model_configs(self):
        class FakePipe:
            available_models = ["model-a", "model-a#2", "model-b"]
            model_configs = {
                "model-a": {"max_tokens": 100000, "n_parallel": 2},
                "model-a#2": {"max_tokens": 200000, "n_parallel": 4},
                "model-b": {"max_tokens": 65536, "n_parallel": 0},
            }

            def _resolve_source_model(self, model_name):
                return model_name.split("#", 1)[0]

        self.assertEqual(
            _configured_contexts_from_pipe(FakePipe()),
            {"model-a": 50000, "model-b": 32768},
        )


class CapabilityDetectionTests(unittest.TestCase):
    def _env(self, **overrides):
        env = {
            "DEFAULT_MODEL": "none",
            "VOICE_SERVER": "",
            "IMAGE_SERVER": "",
            "TEXT_SERVER": "",
            "IMAGE_ENABLED": "false",
            "IMG_MODEL": "",
            "VIDEO_ENABLED": "false",
            "VIDEO_MODEL": "",
            "TTS_ENABLED": "true",
            "TTS_N_PARALLEL": "1",
            "STT_ENABLED": "true",
            "STT_N_PARALLEL": "1",
            "EMBEDDING_ENABLED": "true",
        }
        env.update(overrides)
        return patch.dict(os.environ, env, clear=True)

    def test_default_model_none_does_not_advertise_text(self):
        with self._env(DEFAULT_MODEL="none"):
            caps = detect_local_capabilities()

        self.assertNotIn("text", caps)

    def test_disabled_tts_is_not_advertised(self):
        with self._env(TTS_ENABLED="false", STT_ENABLED="true"):
            caps = detect_local_capabilities()

        self.assertNotIn("tts", caps)
        self.assertIn("stt", caps)

    def test_voice_server_mode_does_not_override_disabled_voice_services(self):
        with self._env(VOICE_SERVER="true", TTS_ENABLED="false", STT_ENABLED="false"):
            caps = detect_local_capabilities()

        self.assertNotIn("tts", caps)
        self.assertNotIn("stt", caps)

    def test_disabled_embeddings_are_not_advertised(self):
        with self._env(EMBEDDING_ENABLED="false"):
            caps = detect_local_capabilities()

        self.assertNotIn("embedding", caps)

    def test_disabled_image_model_is_not_advertised(self):
        with self._env(IMAGE_ENABLED="false", IMG_MODEL="unsloth/FLUX.2-klein-4B-GGUF"):
            caps = detect_local_capabilities()

        self.assertNotIn("image", caps)

    def test_enabled_image_model_is_advertised(self):
        with self._env(IMAGE_ENABLED="true", IMG_MODEL="unsloth/FLUX.2-klein-4B-GGUF"):
            caps = detect_local_capabilities()

        self.assertIn("image", caps)

    def test_disabled_video_model_is_not_advertised(self):
        with self._env(VIDEO_ENABLED="false", VIDEO_MODEL="unsloth/LTX-2.3-GGUF"):
            caps = detect_local_capabilities()

        self.assertNotIn("video", caps)

    def test_enabled_video_model_is_advertised(self):
        with self._env(VIDEO_ENABLED="true", VIDEO_MODEL="unsloth/LTX-2.3-GGUF"):
            caps = detect_local_capabilities()

        self.assertIn("video", caps)

    def test_text_delegation_does_not_disable_embeddings(self):
        with self._env(TEXT_SERVER="http://text.local:8091"):
            caps = detect_local_capabilities()

        self.assertIn("embedding", caps)


if __name__ == "__main__":
    unittest.main()
