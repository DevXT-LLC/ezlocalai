import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ezlocalai.Speculative import (
    speculative_backend,
    dflash_settings,
    download_dflash_model,
    DFLASH_REPO,
    DFLASH_FILE,
    DFLASH_REVISION,
)
from ezlocalai.LlamaTTS import (
    LlamaTTS,
    native_language,
    resolve_tts_model_id,
    download_tts_models,
    QWEN_TTS_MODEL,
    LEGACY_MODEL,
)


class SpeculativeConfigTests(unittest.TestCase):
    def test_auto_pairing_and_legacy_mtp(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            for name in (
                "unsloth/Qwen3.8-27B-GGUF",
                "Qwen3.8-27B",
                "Qwen3.8-27B-Q3_K_XL.gguf",
            ):
                self.assertEqual(speculative_backend(name), "mtp")
            self.assertEqual(speculative_backend("Qwen3.8-270B"), "none")
            self.assertEqual(speculative_backend("Qwen3.6-27B-GGUF"), "none")
            self.assertEqual(speculative_backend("Qwen3.5-35B-A3B-MTP-GGUF"), "mtp")

    def test_overrides_and_multiple_models(self):
        for setting, expected in (
            ("auto", "mtp"),
            ("", "mtp"),
            ("mtp", "mtp"),
            ("none", "none"),
            ("dflash2", "dflash2"),
            ("draft-dflash", "dflash2"),
        ):
            with mock.patch.dict(
                os.environ, {"LLM_SPECULATIVE_TYPE": setting}, clear=True
            ):
                self.assertEqual(speculative_backend("Qwen3.8-27B"), expected)
                self.assertEqual(speculative_backend("Qwen3.5-4B"), "none")

    def test_invalid_backend_fails_loudly(self):
        with mock.patch.dict(os.environ, {"LLM_SPECULATIVE_TYPE": "typo"}, clear=True):
            with self.assertRaises(ValueError):
                speculative_backend("Qwen3.8-27B")

    def test_dflash_does_not_inherit_mtp_knobs(self):
        with mock.patch.dict(
            os.environ,
            {"MTP_SPEC_DRAFT_N_MAX": "2", "MTP_SPEC_DRAFT_P_MIN": "0.9"},
            clear=True,
        ), mock.patch("ezlocalai.InferenceSettings.gpu_profile", return_value=("", 0)):
            self.assertEqual(dflash_settings(), (4, 0.0))

    def test_invalid_dflash_values(self):
        for env, values in (
            ("DFLASH_SPEC_DRAFT_N_MAX", ["0", "8", "bad"]),
            ("DFLASH_SPEC_DRAFT_P_MIN", ["nan", "inf", "-0.1", "1.1"]),
        ):
            for value in values:
                with mock.patch.dict(os.environ, {env: value}, clear=True):
                    with self.assertRaises(ValueError):
                        dflash_settings()

    def test_draft_download_is_revision_pinned(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "huggingface_hub.hf_hub_download", return_value="draft.gguf"
        ) as download:
            self.assertEqual(download_dflash_model(), "draft.gguf")
            download.assert_called_once_with(
                DFLASH_REPO, DFLASH_FILE, revision=DFLASH_REVISION
            )

    def test_local_draft_must_exist(self):
        with mock.patch.dict(
            os.environ, {"DFLASH_MODEL_PATH": "/nonexistent/draft.gguf"}, clear=True
        ):
            with self.assertRaises(FileNotFoundError):
                download_dflash_model()


class NativeTtsTests(unittest.TestCase):
    def test_stream_decodes_incremental_pcm_and_keeps_completed_worker(self):
        import base64
        import struct

        tts = LlamaTTS.__new__(LlamaTTS)
        tts._lock = threading.RLock()
        tts._process = mock.Mock()
        tts._process.poll.return_value = None
        tts.close = mock.Mock()
        tts._receive = mock.Mock(
            side_effect=[
                {
                    "pcm_f32": base64.b64encode(
                        struct.pack("<ff", 0.25, -0.5)
                    ).decode(),
                    "sample_rate": 24000,
                },
                {"ok": True},
            ]
        )
        stream = tts.generate_voice_clone_stream(
            text="Hello", language="en", ref_audio="voice.wav"
        )
        audio, rate = next(stream)
        self.assertEqual(audio.tolist(), [0.25, -0.5])
        self.assertEqual(rate, 24000)
        self.assertEqual(tts._receive.call_count, 1)
        self.assertEqual(list(stream), [])
        tts.close.assert_not_called()

    def test_abandoned_native_stream_kills_worker(self):
        import base64
        import struct

        tts = LlamaTTS.__new__(LlamaTTS)
        tts._lock = threading.RLock()
        tts._process = mock.Mock()
        tts._process.poll.return_value = None
        tts.close = mock.Mock()
        tts._receive = mock.Mock(
            return_value={
                "pcm_f32": base64.b64encode(struct.pack("<f", 0.25)).decode(),
                "sample_rate": 24000,
            }
        )
        stream = tts.generate_voice_clone_stream(
            text="Hello", language="en", ref_audio="voice.wav"
        )
        next(stream)
        stream.close()
        tts.close.assert_called_once()

    def test_cpu_worker_disables_backbone_and_projector_offload(self):
        process = mock.Mock()
        process.stdout = iter(['{"ready":true}\n'])
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "ezlocalai.LlamaTTS.download_tts_models", return_value=("talker", "codec")
        ), mock.patch(
            "ezlocalai.LlamaTTS.shutil.which", return_value="tts"
        ), mock.patch(
            "ezlocalai.LlamaTTS.subprocess.Popen", return_value=process
        ) as popen:
            tts = LlamaTTS("cpu")
            cmd = popen.call_args.args[0]
            self.assertIn("--no-mmproj-offload", cmd)
            self.assertEqual(cmd[cmd.index("-ngl") + 1], "0")
            self.assertEqual(cmd[cmd.index("--device") + 1], "none")
            self.assertEqual(cmd[cmd.index("-t") + 1], "20")
            tts._reader.join(timeout=2)

    def test_cuda_assignment_is_child_local(self):
        process = mock.Mock()
        process.stdout = iter(['{"ready":true}\n'])
        with mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "2,4"}, clear=True
        ), mock.patch(
            "ezlocalai.LlamaTTS.download_tts_models", return_value=("talker", "codec")
        ), mock.patch(
            "ezlocalai.LlamaTTS.shutil.which", return_value="tts"
        ), mock.patch(
            "ezlocalai.LlamaTTS.subprocess.Popen", return_value=process
        ) as popen:
            tts = LlamaTTS("cuda:1")
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "4")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,4")
            tts._reader.join(timeout=2)

    def test_legacy_env_migrates_same_base_model(self):
        with mock.patch.dict(os.environ, {"QWEN_TTS_MODEL": LEGACY_MODEL}, clear=True):
            self.assertEqual(resolve_tts_model_id(), QWEN_TTS_MODEL)

    def test_downloads_only_two_ggufs(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "huggingface_hub.hf_hub_download", side_effect=["talker", "codec"]
        ) as download:
            self.assertEqual(download_tts_models(), ("talker", "codec"))
            self.assertEqual(download.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs.get("revision") != "main"
                    for call in download.call_args_list
                )
            )

    def test_auto_language_never_sends_unsupported_auto(self):
        self.assertEqual(native_language("Auto", "Hello"), "english")
        self.assertEqual(native_language("Auto", "Привет"), "russian")
        self.assertEqual(native_language("Auto", "こんにちは"), "japanese")
        self.assertEqual(native_language("French", "Bonjour"), "french")

    def test_timeout_closes_native_worker(self):
        tts = LlamaTTS.__new__(LlamaTTS)
        tts._responses = queue.Queue()
        tts.timeout = 0.001
        with mock.patch.object(tts, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                tts._receive()
            close.assert_called_once()

    def test_close_waits_for_process_and_is_idempotent(self):
        tts = LlamaTTS.__new__(LlamaTTS)
        tts._lock = threading.RLock()
        tts._close_lock = threading.Lock()
        process = mock.Mock()
        process.poll.return_value = None
        tts._process = process
        tts.close()
        tts.close()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(tts._process)

    def test_close_can_interrupt_active_generation(self):
        tts = LlamaTTS.__new__(LlamaTTS)
        tts._lock = threading.Lock()
        tts._close_lock = threading.Lock()
        process = mock.Mock()
        process.poll.return_value = None
        tts._process = process
        with tts._lock:
            thread = threading.Thread(target=tts.close, daemon=True)
            thread.start()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
        process.terminate.assert_called_once()

    def test_closed_worker_cannot_generate(self):
        tts = LlamaTTS.__new__(LlamaTTS)
        tts._lock = threading.RLock()
        tts._process = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            tts.generate_voice_clone(
                text="test", language="English", ref_audio="voice.wav"
            )


if __name__ == "__main__":
    unittest.main()
