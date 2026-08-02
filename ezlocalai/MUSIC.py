import base64
import json
import logging
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from Globals import getenv


ACE_STEP_DEFAULT_MODEL = "Serveurperso/ACE-Step-1.5-GGUF"
ACE_STEP_DEFAULT_HOST = "127.0.0.1"
ACE_STEP_DEFAULT_PORT = "8085"
ACE_STEP_DEFAULT_MODELS_DIR = "models/ace-step"
ACE_STEP_DEFAULT_BIN = "/opt/acestep.cpp/build/ace-server"
ACE_STEP_DEFAULT_LM_MODEL = "acestep-5Hz-lm-4B-Q8_0.gguf"
ACE_STEP_DEFAULT_TEXT_ENCODER_MODEL = "Qwen3-Embedding-0.6B-Q8_0.gguf"
ACE_STEP_DEFAULT_DIT_MODEL = "acestep-v15-turbo-Q4_K_M.gguf"
ACE_STEP_DEFAULT_VAE_MODEL = "vae-BF16.gguf"
ACE_STEP_OUTPUT_FORMATS = {"mp3", "wav16", "wav24", "wav32"}
ACE_STEP_RESPONSE_FORMATS = {"url", "b64_json"}
ACE_STEP_AUDIO_TYPES = {
    "mp3": "audio/mpeg",
    "wav16": "audio/wav",
    "wav24": "audio/wav",
    "wav32": "audio/wav",
}

_ACE_STEP_PROCESS: Optional[subprocess.Popen] = None


def ace_step_internal_url() -> str:
    host = (
        getenv("ACE_STEP_HOST", ACE_STEP_DEFAULT_HOST) or ACE_STEP_DEFAULT_HOST
    ).strip()
    port = (
        getenv("ACE_STEP_PORT", ACE_STEP_DEFAULT_PORT) or ACE_STEP_DEFAULT_PORT
    ).strip()
    return f"http://{host}:{port}"


def ace_step_server_url() -> str:
    configured = (getenv("ACE_STEP_SERVER_URL") or "").strip()
    return configured.rstrip("/") if configured else ace_step_internal_url()


def ace_step_models_dir() -> str:
    return (
        getenv("ACE_STEP_MODELS_DIR", ACE_STEP_DEFAULT_MODELS_DIR)
        or ACE_STEP_DEFAULT_MODELS_DIR
    ).strip()


def selected_ace_step_model_files() -> List[str]:
    files = [
        getenv("ACE_STEP_TEXT_ENCODER_MODEL", ACE_STEP_DEFAULT_TEXT_ENCODER_MODEL),
        getenv("ACE_STEP_LM_MODEL", ACE_STEP_DEFAULT_LM_MODEL),
        getenv("ACE_STEP_DIT_MODEL", ACE_STEP_DEFAULT_DIT_MODEL),
        getenv("ACE_STEP_VAE_MODEL", ACE_STEP_DEFAULT_VAE_MODEL),
    ]
    seen = set()
    selected = []
    for filename in files:
        filename = (filename or "").strip()
        if filename and filename not in seen:
            seen.add(filename)
            selected.append(filename)
    return selected


def _truthy_env(name: str, default: str = "false") -> bool:
    return (getenv(name, default) or default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ace_step_health_sync(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"{url.rstrip('/')}/health", timeout=timeout
        ) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _ace_step_binary_path() -> str:
    return (
        getenv("ACE_STEP_BIN", ACE_STEP_DEFAULT_BIN) or ACE_STEP_DEFAULT_BIN
    ).strip()


def _configured_external_ace_step_url() -> bool:
    return bool((getenv("ACE_STEP_SERVER_URL") or "").strip())


def _ace_step_autostart_enabled() -> bool:
    return _truthy_env("ACE_STEP_AUTO_START", "true")


def _missing_ace_step_model_files(models_dir: str) -> List[str]:
    base = Path(models_dir)
    return [
        filename
        for filename in selected_ace_step_model_files()
        if not (base / filename).exists()
    ]


def build_ace_step_server_command() -> List[str]:
    cmd = [
        _ace_step_binary_path(),
        "--models",
        ace_step_models_dir(),
        "--host",
        getenv("ACE_STEP_HOST", ACE_STEP_DEFAULT_HOST) or ACE_STEP_DEFAULT_HOST,
        "--port",
        getenv("ACE_STEP_PORT", ACE_STEP_DEFAULT_PORT) or ACE_STEP_DEFAULT_PORT,
    ]

    adapters_dir = (getenv("ACE_STEP_ADAPTERS_DIR") or "").strip()
    if adapters_dir:
        cmd.extend(["--adapters", adapters_dir])
    if _truthy_env("ACE_STEP_KEEP_LOADED", "false"):
        cmd.append("--keep-loaded")

    for flag, env_name in (
        ("--max-batch", "ACE_STEP_MAX_BATCH"),
        ("--max-seq", "ACE_STEP_MAX_SEQ"),
        ("--vae-chunk", "ACE_STEP_VAE_CHUNK"),
        ("--vae-overlap", "ACE_STEP_VAE_OVERLAP"),
    ):
        value = (getenv(env_name) or "").strip()
        if value:
            cmd.extend([flag, value])

    extra_args = (getenv("ACE_STEP_EXTRA_ARGS") or "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


def start_internal_ace_step_server() -> Optional[subprocess.Popen]:
    """Start the container-local acestep.cpp server if this worker owns it."""
    global _ACE_STEP_PROCESS

    if not _truthy_env("MUSIC_ENABLED", "false"):
        return None
    if _configured_external_ace_step_url():
        logging.info(
            "[MUSIC] ACE_STEP_SERVER_URL configured; using external ACE-Step server"
        )
        return None
    if not _ace_step_autostart_enabled():
        logging.info(
            "[MUSIC] ACE_STEP_AUTO_START=false; expecting ACE-Step to be started manually"
        )
        return None

    url = ace_step_internal_url()
    if _ace_step_health_sync(url):
        logging.info(f"[MUSIC] ACE-Step server already healthy at {url}")
        return None

    binary = _ace_step_binary_path()
    if not os.path.exists(binary):
        logging.warning(
            f"[MUSIC] ACE-Step binary not found at {binary}; rebuild the Docker image with acestep.cpp support"
        )
        return None

    models_dir = ace_step_models_dir()
    missing = _missing_ace_step_model_files(models_dir)
    if missing:
        logging.warning(
            "[MUSIC] ACE-Step model directory is missing files: "
            f"{', '.join(missing)}. Run precache or disable ACE_STEP_PRECACHE=false only after downloading them."
        )

    if _ACE_STEP_PROCESS is not None and _ACE_STEP_PROCESS.poll() is None:
        return _ACE_STEP_PROCESS

    cmd = build_ace_step_server_command()
    logging.info(
        f"[MUSIC] Starting ACE-Step server: {' '.join(shlex.quote(part) for part in cmd)}"
    )
    try:
        _ACE_STEP_PROCESS = subprocess.Popen(cmd, cwd=os.getcwd())
    except Exception as e:
        logging.warning(f"[MUSIC] Failed to start ACE-Step server: {e}")
        _ACE_STEP_PROCESS = None
        return None

    startup_timeout = float(getenv("ACE_STEP_STARTUP_TIMEOUT", "30"))
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if _ACE_STEP_PROCESS.poll() is not None:
            logging.warning(
                f"[MUSIC] ACE-Step server exited early with code {_ACE_STEP_PROCESS.returncode}"
            )
            return _ACE_STEP_PROCESS
        if _ace_step_health_sync(url):
            logging.info(f"[MUSIC] ACE-Step server is ready at {url}")
            return _ACE_STEP_PROCESS
        time.sleep(0.5)

    logging.warning(
        f"[MUSIC] ACE-Step server did not become healthy within {startup_timeout:.0f}s; continuing startup"
    )
    return _ACE_STEP_PROCESS


class MusicGenerationError(RuntimeError):
    """Raised when the ACE-Step backend rejects or fails a music generation job."""


class MUSIC:
    """Client for an ACE-Step / acestep.cpp music generation server.

    ACE-Step GGUF models are served by acestep.cpp, which exposes a small
    async HTTP API. ezLocalai stays responsible for auth, routing, output URLs,
    and OpenAI-style response shapes while the native server owns model loading
    and synthesis.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        local_uri: Optional[str] = None,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        model: str = ACE_STEP_DEFAULT_MODEL,
        lm_model: Optional[str] = None,
        synth_model: Optional[str] = None,
    ):
        self.server_url = (server_url or ace_step_server_url()).rstrip("/")
        self.local_uri = local_uri
        self.timeout = float(
            timeout if timeout is not None else getenv("ACE_STEP_TIMEOUT", "1800")
        )
        self.poll_interval = float(
            poll_interval
            if poll_interval is not None
            else getenv("ACE_STEP_POLL_INTERVAL", "1.0")
        )
        self.model = model or ACE_STEP_DEFAULT_MODEL
        self.lm_model = lm_model or getenv(
            "ACE_STEP_LM_MODEL", ACE_STEP_DEFAULT_LM_MODEL
        )
        self.synth_model = synth_model or getenv(
            "ACE_STEP_DIT_MODEL", ACE_STEP_DEFAULT_DIT_MODEL
        )

    async def health(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def props(self) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/props",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise MusicGenerationError(
                        f"ACE-Step /props failed with HTTP {resp.status}: {text[:300]}"
                    )
                return await resp.json()

    def build_request(
        self,
        *,
        prompt: str,
        lyrics: Optional[str] = None,
        instrumental: bool = False,
        output_format: str = "mp3",
        duration: Optional[float] = None,
        seed: Optional[int] = None,
        n: int = 1,
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        timesignature: Optional[str] = None,
        vocal_language: Optional[str] = None,
        inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        shift: Optional[float] = None,
        solver: Optional[str] = None,
        lm_model: Optional[str] = None,
        synth_model: Optional[str] = None,
        audio_codes: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt is required")

        output_format = self._normalize_output_format(output_format)
        lm_model = lm_model if lm_model is not None else self.lm_model
        synth_model = synth_model if synth_model is not None else self.synth_model
        request: Dict[str, Any] = {
            "task_type": "text2music",
            "caption": str(prompt).strip(),
            "output_format": output_format,
            "lm_batch_size": max(1, int(n or 1)),
        }

        if instrumental:
            request["lyrics"] = "[Instrumental]"
        elif lyrics is not None:
            request["lyrics"] = str(lyrics)

        optional_fields = {
            "duration": duration,
            "seed": seed,
            "bpm": bpm,
            "keyscale": keyscale,
            "timesignature": timesignature,
            "vocal_language": vocal_language,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "shift": shift,
            "solver": solver,
            "lm_model": lm_model,
            "synth_model": synth_model,
            "audio_codes": audio_codes,
        }
        for key, value in optional_fields.items():
            if value not in (None, ""):
                request[key] = value

        if extra:
            for key, value in extra.items():
                if value is not None:
                    request[key] = value

        return request

    async def generate(
        self,
        *,
        prompt: str,
        lyrics: Optional[str] = None,
        instrumental: bool = False,
        response_format: str = "url",
        output_format: str = "mp3",
        model: Optional[str] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        response_format = self._normalize_response_format(response_format)
        display_model = model or self.model
        ace_request = self.build_request(
            prompt=prompt,
            lyrics=lyrics,
            instrumental=instrumental,
            output_format=output_format,
            n=n,
            **kwargs,
        )

        if ace_request.get("audio_codes"):
            synth_requests = [ace_request]
        else:
            lm_body, lm_content_type = await self._run_json_job("/lm", ace_request)
            synth_requests = self._parse_lm_result(lm_body, lm_content_type)
            if not synth_requests:
                raise MusicGenerationError("ACE-Step LM returned no synth requests")

        synth_payload: Any = (
            synth_requests[0] if len(synth_requests) == 1 else synth_requests
        )
        synth_body, synth_content_type = await self._run_json_job(
            "/synth", synth_payload
        )
        audio_parts = self._parse_synth_result(synth_body, synth_content_type)

        data = []
        for index, (audio_bytes, content_type) in enumerate(audio_parts):
            data.append(
                self._format_audio_result(
                    audio_bytes,
                    content_type=content_type,
                    response_format=response_format,
                    output_format=output_format,
                    index=index,
                    metadata=synth_requests[min(index, len(synth_requests) - 1)],
                )
            )

        return {
            "created": int(time.time()),
            "model": display_model,
            "data": data,
        }

    async def _run_json_job(self, endpoint: str, payload: Any) -> Tuple[bytes, str]:
        job_id = await self._submit_json(endpoint, payload)
        await self._wait_for_job(job_id)
        return await self._fetch_job_result(job_id)

    async def _submit_json(self, endpoint: str, payload: Any) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}{endpoint}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise MusicGenerationError(
                        f"ACE-Step {endpoint} submit failed with HTTP {resp.status}: {body[:300]}"
                    )
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    raise MusicGenerationError(
                        f"ACE-Step {endpoint} returned invalid JSON: {body[:300]}"
                    ) from e
                job_id = data.get("id")
                if not job_id:
                    raise MusicGenerationError(
                        f"ACE-Step {endpoint} response did not include a job id"
                    )
                return str(job_id)

    async def _wait_for_job(self, job_id: str) -> None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/job",
                    params={"id": job_id},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        raise MusicGenerationError(
                            f"ACE-Step job {job_id} poll failed with HTTP {resp.status}: {body[:300]}"
                        )
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as e:
                        raise MusicGenerationError(
                            f"ACE-Step job {job_id} returned invalid status JSON: {body[:300]}"
                        ) from e
                    status = str(data.get("status", "")).lower()
                    if status == "done":
                        return
                    if status in {"failed", "cancelled"}:
                        detail = data.get("error") or data.get("message") or body
                        raise MusicGenerationError(
                            f"ACE-Step job {job_id} ended with status {status}: {detail}"
                        )
            await self._sleep(self.poll_interval)
        raise MusicGenerationError(
            f"ACE-Step job {job_id} timed out after {self.timeout:.0f}s"
        )

    async def _fetch_job_result(self, job_id: str) -> Tuple[bytes, str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/job",
                params={"id": job_id, "result": "1"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                body = await resp.read()
                if resp.status != 200:
                    try:
                        detail = body.decode("utf-8", errors="replace")
                    except Exception:
                        detail = "<binary response>"
                    raise MusicGenerationError(
                        f"ACE-Step job {job_id} result failed with HTTP {resp.status}: {detail[:300]}"
                    )
                return body, resp.headers.get("Content-Type", "")

    async def _sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    def _parse_lm_result(self, body: bytes, content_type: str) -> List[Dict[str, Any]]:
        if "application/json" not in (content_type or "").lower():
            logging.debug("[MUSIC] Expected JSON LM result, got %s", content_type)
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception as e:
            raise MusicGenerationError("ACE-Step LM result was not valid JSON") from e
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        raise MusicGenerationError("ACE-Step LM result must be a JSON object or array")

    def _parse_synth_result(
        self, body: bytes, content_type: str
    ) -> List[Tuple[bytes, str]]:
        content_type = content_type or ""
        lower_type = content_type.lower()
        if lower_type.startswith("audio/"):
            return [(body, content_type)]

        if not lower_type.startswith("multipart/"):
            raise MusicGenerationError(
                f"ACE-Step synth result was {content_type or 'missing content-type'}, not audio or multipart"
            )

        message = BytesParser(policy=email_policy).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
        )
        parts: List[Tuple[bytes, str]] = []
        for part in message.iter_parts():
            part_type = part.get_content_type()
            if not part_type.startswith("audio/"):
                continue
            payload = part.get_payload(decode=True)
            if payload:
                parts.append((payload, part_type))
        if not parts:
            raise MusicGenerationError(
                "ACE-Step synth multipart result contained no audio parts"
            )
        return parts

    def _format_audio_result(
        self,
        audio_bytes: bytes,
        *,
        content_type: str,
        response_format: str,
        output_format: str,
        index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        content_type = content_type or ACE_STEP_AUDIO_TYPES.get(
            output_format, "audio/mpeg"
        )
        extension = "mp3" if "mpeg" in content_type or output_format == "mp3" else "wav"
        item: Dict[str, Any] = {
            "content_type": content_type,
            "format": output_format,
        }
        if metadata:
            item["lyrics"] = metadata.get("lyrics", "")
            item["caption"] = metadata.get("caption", "")
            for key in (
                "duration",
                "bpm",
                "keyscale",
                "timesignature",
                "vocal_language",
                "seed",
            ):
                if key in metadata:
                    item[key] = metadata[key]

        if response_format == "b64_json":
            item["b64_json"] = base64.b64encode(audio_bytes).decode("utf-8")
            return item

        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/{uuid.uuid4()}-{index}.{extension}"
        with open(filename, "wb") as audio_file:
            audio_file.write(audio_bytes)
        item["url"] = f"{self.local_uri}/{filename}" if self.local_uri else filename
        return item

    def _normalize_output_format(self, output_format: Optional[str]) -> str:
        output_format = (output_format or "mp3").strip().lower()
        if output_format not in ACE_STEP_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {sorted(ACE_STEP_OUTPUT_FORMATS)}"
            )
        return output_format

    def _normalize_response_format(self, response_format: Optional[str]) -> str:
        response_format = (response_format or "url").strip().lower()
        if response_format not in ACE_STEP_RESPONSE_FORMATS:
            raise ValueError(
                f"response_format must be one of {sorted(ACE_STEP_RESPONSE_FORMATS)}"
            )
        return response_format
