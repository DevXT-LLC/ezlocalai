"""Qwen TTS via a persistent llama.cpp/libmtmd subprocess, without qwen-tts.

Each instance owns one process. close() waits for native VRAM to be released
before the existing voice/LLM slot handoff can restore the language model.
"""

import base64
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

QWEN_TTS_MODEL = "Mouserat/qwen3-tts-0.6b-base-gguf"
QWEN_TTS_REVISION = "30954b7a9d99fad81064bf1dc35f881b029f9032"
LEGACY_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TTS_MODEL_FILE = "qwen-talker-0.6b-base-Q8_0.gguf"
TTS_MMPROJ_FILE = "qwen-tokenizer-12hz-f16.gguf"


def resolve_tts_model_id():
    name = os.getenv("QWEN_TTS_MODEL", QWEN_TTS_MODEL).strip() or QWEN_TTS_MODEL
    # Migrate existing .env files without downloading the Transformers weights.
    return QWEN_TTS_MODEL if name == LEGACY_MODEL else name


def download_tts_models():
    from huggingface_hub import hf_hub_download

    repo = resolve_tts_model_id()
    revision = os.getenv("QWEN_TTS_REVISION") or (
        QWEN_TTS_REVISION if repo == QWEN_TTS_MODEL else "main"
    )
    result = []
    for env, default in (
        ("QWEN_TTS_MODEL_FILE", TTS_MODEL_FILE),
        ("QWEN_TTS_MMPROJ_FILE", TTS_MMPROJ_FILE),
    ):
        filename = os.getenv(env, default).strip() or default
        if Path(filename).is_file():
            result.append(str(Path(filename).resolve()))
        else:
            result.append(hf_hub_download(repo, filename, revision=revision))
    return tuple(result)


def native_language(language, text):
    """libmtmd requires an explicit language; infer script for legacy auto mode."""
    value = language.lower()
    if value != "auto":
        return value
    for pattern, name in (
        (r"[\u3040-\u30ff]", "japanese"),
        (r"[\uac00-\ud7af]", "korean"),
        (r"[\u4e00-\u9fff]", "chinese"),
        (r"[\u0400-\u04ff]", "russian"),
    ):
        if re.search(pattern, text):
            return name
    return "english"


class LlamaTTS:
    def __init__(self, device="cpu", generation_kwargs=None):
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._process = None
        self._responses = queue.Queue()
        self.timeout = float(os.getenv("QWEN_TTS_TIMEOUT", "300"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("QWEN_TTS_TIMEOUT must be positive")
        self.device = device
        model, mmproj = download_tts_models()
        binary = os.getenv("QWEN_TTS_BIN", "/opt/ezlocalai-tts/build/bin/ezlocalai-tts")
        if "QWEN_TTS_BIN" not in os.environ and not Path(binary).is_file():
            executable = "ezlocalai-tts.exe" if os.name == "nt" else "ezlocalai-tts"
            binary = str(Path.home() / ".ezlocalai" / "tts-build" / "bin" / executable)
        if not shutil.which(binary):
            raise RuntimeError(
                f"Native TTS worker not found: {binary}. Rebuild the image or run "
                "python scripts/build_tts.py and set QWEN_TTS_BIN to its output."
            )
        kwargs = generation_kwargs or {}
        supported = {
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "seed",
        }
        unsupported = set(kwargs) - supported
        if unsupported:
            raise ValueError(
                f"Unsupported llama.cpp TTS generation settings: {sorted(unsupported)}"
            )
        cmd = [
            binary,
            "-m",
            model,
            "-mm",
            mmproj,
            "-c",
            os.getenv("QWEN_TTS_CONTEXT_SIZE", "4096"),
            "-b",
            "512",
            "-ub",
            "512",
            "-t",
            os.getenv("QWEN_TTS_THREADS", "20"),
            "--temp",
            str(kwargs.get("temperature", 0.9)),
            "--top-k",
            str(kwargs.get("top_k", 50)),
            "--top-p",
            str(kwargs.get("top_p", 1.0)),
            "--repeat-penalty",
            str(kwargs.get("repetition_penalty", 1.05)),
            "--flash-attn",
            "on",
        ]
        env = os.environ.copy()
        if "seed" in kwargs:
            cmd += ["--seed", str(int(kwargs["seed"]))]
        if device.startswith("cuda"):
            # Restrict the child only; never mutate the parent's CUDA assignment.
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            visible = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
            selected = (
                visible[index]
                if visible != [""] and index < len(visible)
                else str(index)
            )
            env["CUDA_VISIBLE_DEVICES"] = selected
            env["HIP_VISIBLE_DEVICES"] = selected
            cmd += ["-ngl", "999"]
        else:
            cmd += ["-ngl", "0", "--no-mmproj-offload", "--device", "none"]
        # A separate process prevents collisions with xllamacpp's shared libs.
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(
            target=self._read_responses,
            args=(self._process, self._responses),
            daemon=True,
        )
        self._reader.start()
        try:
            if self._receive().get("ready") is not True:
                raise RuntimeError("Native TTS worker did not become ready")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _read_responses(process, responses):
        try:
            for line in process.stdout:
                try:
                    value = json.loads(line)
                except ValueError:
                    logging.debug("[QTTS] Native output: %s", line.rstrip())
                    continue
                if isinstance(value, dict):
                    responses.put(value)
        finally:
            responses.put({"error": "Native TTS worker exited"})

    def __del__(self):
        # The reader does not retain this owner. An initialization failure in a
        # caller therefore cannot leave an orphaned model process behind.
        try:
            self.close()
        except Exception:
            pass

    def _receive(self):
        try:
            response = self._responses.get(timeout=self.timeout)
        except queue.Empty as error:
            self.close()
            raise RuntimeError(
                f"Native TTS worker on {getattr(self, 'device', 'unknown')} timed out and was stopped"
            ) from error
        if "error" in response:
            raise RuntimeError(
                f"Native TTS worker on {getattr(self, 'device', 'unknown')}: {response['error']}"
            )
        return response

    def generate_voice_clone(self, *, text, language, ref_audio, **kwargs):
        import soundfile as sf

        # This backend intentionally supports audio-only x-vector conditioning.
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("Native TTS worker is closed")
            with tempfile.TemporaryDirectory(prefix="ezlocalai-tts-") as directory:
                output = str(Path(directory) / "speech.wav")
                request = {
                    "text": text,
                    "speaker": str(Path(ref_audio).resolve()),
                    "output": output,
                    "language": native_language(language, text),
                    "max_new_tokens": int(kwargs.get("max_new_tokens", 320)),
                }
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                response = self._receive()
                if response.get("ok") is not True:
                    raise RuntimeError("Invalid native TTS response")
                if response.get("limit_reached"):
                    logging.warning(
                        "[QTTS] Audio frame limit reached; consider shorter chunks or a larger QWEN_TTS_MAX_NEW_TOKENS"
                    )
                audio, rate = sf.read(output, dtype="float32")
                return [audio], int(rate)

    def close(self):
        # Cancellation must be able to stop a child while generate() is waiting
        # for its reply under _lock. Do not wait for the full synthesis timeout.
        with self._close_lock:
            process, self._process = self._process, None
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
            if (
                hasattr(self, "_reader")
                and self._reader is not threading.current_thread()
            ):
                self._reader.join(timeout=2)

    def generate_voice_clone_stream(self, *, text, language, ref_audio, **kwargs):
        """Yield native float32 PCM windows while the talker is still decoding."""
        import numpy as np

        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("Native TTS worker is closed")
            complete = False
            request = {
                "text": text,
                "speaker": str(Path(ref_audio).resolve()),
                "language": native_language(language, text),
                "stream": True,
                "max_new_tokens": int(kwargs.get("max_new_tokens", 320)),
            }
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                while True:
                    response = self._receive()
                    if "pcm_f32" in response:
                        pcm = base64.b64decode(response["pcm_f32"], validate=True)
                        yield np.frombuffer(pcm, dtype="<f4"), int(
                            response["sample_rate"]
                        )
                    elif response.get("ok") is True:
                        complete = True
                        if response.get("limit_reached"):
                            logging.warning(
                                "[QTTS] Streaming audio reached the frame limit"
                            )
                        return
                    else:
                        raise RuntimeError("Invalid native TTS streaming response")
            finally:
                # An abandoned stream cannot leave unread PCM/done messages for
                # the next utterance, or keep an unleased process using VRAM.
                if not complete:
                    self.close()
