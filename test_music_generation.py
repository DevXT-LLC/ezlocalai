import base64
import json
import os
import tempfile
import unittest
from unittest.mock import call, patch

from ezlocalai.MUSIC import (
    MUSIC,
    ace_step_server_url,
    selected_ace_step_model_files,
)
from Router import Router, WorkerInfo, WorkerRegistry, detect_local_capabilities


HEAVY_METAL_PROMPT = (
    "Heavy metal anthem about the Pythagorean theorem, double-kick drums, "
    "distorted guitars, soaring vocals, and a triumphant chorus."
)

PYTHAGOREAN_LYRICS = """[Verse]
On a right triangle battlefield
Two short sides raise their shields

[Chorus]
A squared plus B squared, lightning in the night
Equals C squared, hypotenuse burning bright
"""


class MusicClientTests(unittest.IsolatedAsyncioTestCase):
    def test_blank_server_url_uses_internal_container_server(self):
        with patch.dict(
            os.environ,
            {
                "ACE_STEP_SERVER_URL": "",
                "ACE_STEP_HOST": "127.0.0.1",
                "ACE_STEP_PORT": "9099",
            },
            clear=True,
        ):
            self.assertEqual(ace_step_server_url(), "http://127.0.0.1:9099")
            self.assertEqual(MUSIC().server_url, "http://127.0.0.1:9099")

    def test_selected_ace_step_model_files_uses_all_required_ggufs_once(self):
        with patch.dict(
            os.environ,
            {
                "ACE_STEP_TEXT_ENCODER_MODEL": "Qwen3-Embedding-0.6B-Q8_0.gguf",
                "ACE_STEP_LM_MODEL": "acestep-5Hz-lm-4B-Q8_0.gguf",
                "ACE_STEP_DIT_MODEL": "acestep-v15-turbo-Q4_K_M.gguf",
                "ACE_STEP_VAE_MODEL": "vae-BF16.gguf",
            },
            clear=True,
        ):
            files = selected_ace_step_model_files()

        self.assertEqual(
            files,
            [
                "Qwen3-Embedding-0.6B-Q8_0.gguf",
                "acestep-5Hz-lm-4B-Q8_0.gguf",
                "acestep-v15-turbo-Q4_K_M.gguf",
                "vae-BF16.gguf",
            ],
        )

    def test_music_enabled_only_uses_4b_lm_default(self):
        with patch.dict(os.environ, {"MUSIC_ENABLED": "true"}, clear=True):
            files = selected_ace_step_model_files()
            music = MUSIC("http://ace.local")

        self.assertIn("acestep-5Hz-lm-4B-Q8_0.gguf", files)
        self.assertNotIn("acestep-5Hz-lm-0.6B-Q8_0.gguf", files)
        self.assertEqual(music.lm_model, "acestep-5Hz-lm-4B-Q8_0.gguf")

    def test_build_request_for_heavy_metal_pythagorean_song(self):
        music = MUSIC("http://ace.local")

        request = music.build_request(
            prompt=HEAVY_METAL_PROMPT,
            lyrics=PYTHAGOREAN_LYRICS,
            duration=12,
            seed=31415,
            n=2,
            output_format="wav16",
            inference_steps=8,
            shift=3.0,
            extra={"lm_temperature": 0.7},
        )

        self.assertEqual(request["task_type"], "text2music")
        self.assertEqual(request["caption"], HEAVY_METAL_PROMPT)
        self.assertIn("A squared plus B squared", request["lyrics"])
        self.assertEqual(request["duration"], 12)
        self.assertEqual(request["seed"], 31415)
        self.assertEqual(request["lm_batch_size"], 2)
        self.assertEqual(request["output_format"], "wav16")
        self.assertEqual(request["inference_steps"], 8)
        self.assertEqual(request["shift"], 3.0)
        self.assertEqual(request["lm_temperature"], 0.7)

    def test_build_request_fills_agent_friendly_defaults(self):
        music = MUSIC("http://ace.local")

        request = music.build_request(
            prompt=HEAVY_METAL_PROMPT,
            lyrics=PYTHAGOREAN_LYRICS,
            duration=45,
            keyscale="E minor",
        )

        self.assertEqual(request["output_format"], "wav16")
        self.assertEqual(request["duration"], 45)
        self.assertEqual(request["keyscale"], "E minor")
        self.assertEqual(request["bpm"], 128)
        self.assertEqual(request["timesignature"], "4/4")
        self.assertEqual(request["vocal_language"], "en")
        self.assertEqual(request["inference_steps"], 16)
        self.assertEqual(request["guidance_scale"], 1.0)
        self.assertEqual(request["shift"], 3.0)
        self.assertEqual(request["solver"], "euler")
        self.assertEqual(request["lm_model"], "acestep-5Hz-lm-4B-Q8_0.gguf")
        self.assertEqual(request["synth_model"], "acestep-v15-turbo-Q4_K_M.gguf")

    def test_parse_synth_multipart_extracts_audio_and_ignores_latents(self):
        music = MUSIC("http://ace.local")
        body = (
            b"--ace\r\n"
            b"Content-Type: audio/mpeg\r\n\r\n"
            b"ID3fake-mp3-bytes\r\n"
            b"--ace\r\n"
            b"Content-Type: application/octet-stream\r\n\r\n"
            b"latent-bytes\r\n"
            b"--ace--\r\n"
        )

        parts = music._parse_synth_result(body, "multipart/mixed; boundary=ace")

        self.assertEqual(parts, [(b"ID3fake-mp3-bytes", "audio/mpeg")])

    async def test_generate_runs_lm_then_synth_and_returns_base64_audio(self):
        class FakeMusic(MUSIC):
            def __init__(self):
                super().__init__("http://ace.local")
                self.calls = []

            async def _run_json_job(self, endpoint, payload):
                self.calls.append((endpoint, payload))
                if endpoint == "/lm":
                    return (
                        json.dumps(
                            [
                                {
                                    "caption": payload["caption"],
                                    "lyrics": payload["lyrics"],
                                    "audio_codes": "1,2,3,4",
                                    "duration": payload["duration"],
                                }
                            ]
                        ).encode("utf-8"),
                        "application/json",
                    )
                if endpoint == "/synth":
                    if payload["audio_codes"] != "1,2,3,4":
                        raise AssertionError(
                            "Synth payload did not receive audio codes"
                        )
                    return b"ID3fake-song", "audio/mpeg"
                raise AssertionError(f"Unexpected endpoint {endpoint}")

        music = FakeMusic()

        result = await music.generate(
            prompt=HEAVY_METAL_PROMPT,
            lyrics=PYTHAGOREAN_LYRICS,
            duration=10,
            response_format="b64_json",
            output_format="mp3",
            seed=31415,
        )

        self.assertEqual([call[0] for call in music.calls], ["/lm", "/synth"])
        lm_payload = music.calls[0][1]
        self.assertEqual(lm_payload["caption"], HEAVY_METAL_PROMPT)
        self.assertIn("hypotenuse", lm_payload["lyrics"])
        self.assertEqual(result["model"], "Serveurperso/ACE-Step-1.5-GGUF")
        audio = base64.b64decode(result["data"][0]["b64_json"])
        self.assertEqual(audio, b"ID3fake-song")
        self.assertEqual(result["data"][0]["content_type"], "audio/mpeg")

    async def test_generate_uses_available_model_when_request_alias_differs(self):
        class FakeMusic(MUSIC):
            def __init__(self):
                super().__init__("http://ace.local")
                self.calls = []

            async def _run_json_job(self, endpoint, payload):
                self.calls.append((endpoint, payload))
                if endpoint == "/lm":
                    return (
                        json.dumps(
                            {
                                "caption": payload["caption"],
                                "lyrics": payload["lyrics"],
                                "audio_codes": "5,6,7",
                                "duration": payload["duration"],
                                "bpm": payload["bpm"],
                                "vocal_language": payload["vocal_language"],
                            }
                        ).encode("utf-8"),
                        "application/json",
                    )
                if endpoint == "/synth":
                    return b"RIFFfake-wav", "audio/wav"
                raise AssertionError(f"Unexpected endpoint {endpoint}")

        music = FakeMusic()

        result = await music.generate(
            model="music-1",
            prompt=HEAVY_METAL_PROMPT,
            lyrics=PYTHAGOREAN_LYRICS,
            duration=45,
            response_format="b64_json",
            output_format="wav16",
            bpm=128,
            vocal_language="en",
            lm_model="acestep-5Hz-lm-4B-Q8_0.gguf",
            synth_model="acestep-v15-turbo-Q4_K_M.gguf",
            guidance_scale=1.0,
        )

        lm_payload = music.calls[0][1]
        self.assertEqual(result["model"], "Serveurperso/ACE-Step-1.5-GGUF")
        self.assertEqual(lm_payload["bpm"], 128)
        self.assertEqual(lm_payload["vocal_language"], "en")
        self.assertEqual(lm_payload["lm_model"], "acestep-5Hz-lm-4B-Q8_0.gguf")
        self.assertEqual(
            lm_payload["synth_model"], "acestep-v15-turbo-Q4_K_M.gguf"
        )
        self.assertEqual(lm_payload["guidance_scale"], 1.0)


class MusicRouterTests(unittest.TestCase):
    def test_music_routes_by_capability_when_model_alias_is_not_advertised(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            WorkerInfo(
                worker_id="music-worker",
                label="MUSIC Worker",
                url="http://music.local",
                capabilities=["music"],
                cap_slots={
                    "music": {
                        "capacity": 1,
                        "in_flight": 0,
                        "queued": 0,
                        "available": 1,
                    }
                },
            )
        )
        router = Router(registry)

        worker = router.select_worker("music", "music-1", allow_cross_model=False)

        self.assertIsNotNone(worker)
        self.assertEqual(worker.label, "MUSIC Worker")

    def test_enabled_music_is_advertised_with_internal_ace_step_default(self):
        with patch.dict(
            os.environ,
            {
                "DEFAULT_MODEL": "none",
                "VOICE_SERVER": "",
                "IMAGE_SERVER": "",
                "TEXT_SERVER": "",
                "EMBEDDING_SERVER": "",
                "TTS_ENABLED": "false",
                "STT_ENABLED": "false",
                "EMBEDDING_ENABLED": "false",
                "IMAGE_ENABLED": "false",
                "VIDEO_ENABLED": "false",
                "MUSIC_ENABLED": "true",
                "MUSIC_MODEL": "Serveurperso/ACE-Step-1.5-GGUF",
                "ACE_STEP_SERVER_URL": "",
            },
            clear=True,
        ):
            caps = detect_local_capabilities()

        self.assertIn("music", caps)

    def test_music_generation_slot_state_makes_text_unavailable(self):
        registry = WorkerRegistry(ttl_seconds=60)
        registry.register(
            WorkerInfo(
                worker_id="desktop",
                label="Desktop",
                url="http://desktop.local",
                capabilities=["text", "music"],
                models=["local-llm"],
                cap_slots={
                    "text": {
                        "capacity": 0,
                        "in_flight": 0,
                        "queued": 0,
                        "available": 0,
                    },
                    "music": {
                        "capacity": 1,
                        "in_flight": 1,
                        "queued": 0,
                        "available": 0,
                    },
                },
                model_slots={
                    "local-llm": {
                        "capacity": 0,
                        "in_flight": 0,
                        "queued": 0,
                        "available": 0,
                    }
                },
            )
        )
        router = Router(registry)

        worker = router.select_worker("text", "local-llm", allow_cross_model=False)

        self.assertIsNone(worker)


class MusicPrecacheTests(unittest.TestCase):
    def test_precache_music_model_downloads_selected_gguf_files(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with patch.dict(
                os.environ,
                {
                    "MUSIC_ENABLED": "true",
                    "ACE_STEP_PRECACHE": "true",
                    "ACE_STEP_SERVER_URL": "",
                    "ACE_STEP_MODEL_REPO": "Serveurperso/ACE-Step-1.5-GGUF",
                    "ACE_STEP_MODELS_DIR": model_dir,
                    "ACE_STEP_TEXT_ENCODER_MODEL": "text.gguf",
                    "ACE_STEP_LM_MODEL": "lm.gguf",
                    "ACE_STEP_DIT_MODEL": "dit.gguf",
                    "ACE_STEP_VAE_MODEL": "vae.gguf",
                },
                clear=True,
            ):
                with patch("precache.download_with_progress") as download:
                    from precache import precache_music_model

                    precache_music_model()

            download.assert_has_calls(
                [
                    call(
                        "Serveurperso/ACE-Step-1.5-GGUF",
                        filename="text.gguf",
                        local_dir=model_dir,
                    ),
                    call(
                        "Serveurperso/ACE-Step-1.5-GGUF",
                        filename="lm.gguf",
                        local_dir=model_dir,
                    ),
                    call(
                        "Serveurperso/ACE-Step-1.5-GGUF",
                        filename="dit.gguf",
                        local_dir=model_dir,
                    ),
                    call(
                        "Serveurperso/ACE-Step-1.5-GGUF",
                        filename="vae.gguf",
                        local_dir=model_dir,
                    ),
                ]
            )


@unittest.skipUnless(
    os.environ.get("ACE_STEP_LIVE_TEST", "").lower() == "true",
    "Set ACE_STEP_LIVE_TEST=true to run the live ACE-Step proof.",
)
class LiveAceStepMusicProofTest(unittest.IsolatedAsyncioTestCase):
    async def test_live_heavy_metal_pythagorean_song_generates_audio(self):
        music = MUSIC(
            os.environ.get("ACE_STEP_SERVER_URL"),
            timeout=float(os.environ.get("ACE_STEP_TIMEOUT", "1800")),
            poll_interval=float(os.environ.get("ACE_STEP_POLL_INTERVAL", "1.0")),
        )

        result = await music.generate(
            prompt=HEAVY_METAL_PROMPT,
            lyrics=PYTHAGOREAN_LYRICS,
            duration=10,
            response_format="b64_json",
            output_format="mp3",
            inference_steps=8,
            shift=3.0,
            seed=31415,
        )

        self.assertGreater(len(result["data"]), 0)
        audio = base64.b64decode(result["data"][0]["b64_json"])
        self.assertGreater(len(audio), 10_000)
        self.assertTrue(audio.startswith(b"ID3") or audio[:2] == b"\xff\xfb")


if __name__ == "__main__":
    unittest.main()
