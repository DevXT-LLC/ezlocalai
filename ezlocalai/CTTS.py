import asyncio
import base64
import gc
import io
import logging
import os
import re
import uuid
from contextlib import aclosing
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ezlocalai.LlamaTTS import (
    LlamaTTS,
    QWEN_TTS_MODEL,
    resolve_tts_model_id,
)

from ezlocalai.AudioCache import AudioCache
from ezlocalai.AudioStreaming import stream_native_audio


MAX_CHUNK_CHARS = 350
STREAM_CHUNK_TARGET_CHARS = 280
STREAM_FIRST_CHUNK_TARGET_CHARS = 120
STREAM_MIN_FIRST_CHUNK_CHARS = 50
STREAM_WRITE_BYTES = 16384
STREAM_FRAME_DRAIN_SECONDS = 0.0
STREAM_FLUSH_SILENCE_MS = 0
SAFE_FILE_STEM_RE = re.compile(r"[^a-zA-Z0-9_-]")

LANGUAGE_ALIASES = {
    "": "Auto",
    "auto": "Auto",
    "automatic": "Auto",
    "multilingual": "Auto",
    "multi": "Auto",
    "en": "English",
    "eng": "English",
    "english": "English",
    "ru": "Russian",
    "rus": "Russian",
    "russian": "Russian",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}

ONES = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def number_to_words(n: int) -> str:
    """Convert an integer to English words."""
    if n < 0:
        return "negative " + number_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return (
            ONES[n // 100]
            + " hundred"
            + ("" if n % 100 == 0 else " " + number_to_words(n % 100))
        )
    if n < 1000000:
        return (
            number_to_words(n // 1000)
            + " thousand"
            + ("" if n % 1000 == 0 else " " + number_to_words(n % 1000))
        )
    return str(n)


def normalize_english_text_for_tts(text: str) -> str:
    """Normalize English-only text for smoother TTS output."""

    def time_to_words(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3).upper() if match.group(3) else ""

        if minute == 0:
            time_str = number_to_words(hour) + " o'clock"
        elif minute < 10:
            time_str = number_to_words(hour) + " oh " + number_to_words(minute)
        else:
            time_str = number_to_words(hour) + " " + number_to_words(minute)

        if period:
            time_str += " " + period.replace("AM", "A M").replace("PM", "P M")
        return time_str

    text = re.sub(r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?", time_to_words, text)

    def date_to_words(match):
        month = int(match.group(1))
        day = int(match.group(2))
        year = match.group(3) if match.group(3) else None
        months = [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        if not 1 <= month <= 12:
            return match.group(0)
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        result = months[month] + " " + number_to_words(day) + suffix
        if year:
            year_int = int(year)
            if year_int >= 2000:
                result += " " + number_to_words(year_int)
            else:
                result += (
                    " "
                    + number_to_words(year_int // 100)
                    + " "
                    + number_to_words(year_int % 100)
                )
        return result

    text = re.sub(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", date_to_words, text)
    text = re.sub(r"\b(\d{1,6})\b", lambda m: number_to_words(int(m.group(0))), text)
    return text


def clean_text_for_tts(text: str) -> str:
    """Light cleanup that keeps multilingual characters intact."""
    text = re.sub(r"([!?.。！？])\1+", r"\1", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.replace("#", "").strip()


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into sentence-sized chunks for lower latency and memory use."""
    return _split_text_into_chunks(text, max_chars, keep_short_text_together=True)


def split_text_into_stream_chunks(
    text: str,
    target_chars: int = STREAM_CHUNK_TARGET_CHARS,
    first_target_chars: Optional[int] = None,
    min_first_chars: Optional[int] = None,
) -> list[str]:
    """Split streaming text into natural speech units without dropping text."""
    target_chars = max(1, target_chars)
    first_target_chars = min(
        target_chars,
        max(
            1,
            (
                first_target_chars
                if first_target_chars is not None
                else STREAM_FIRST_CHUNK_TARGET_CHARS
            ),
        ),
    )
    min_first_chars = min(
        first_target_chars,
        max(
            1,
            (
                min_first_chars
                if min_first_chars is not None
                else STREAM_MIN_FIRST_CHUNK_CHARS
            ),
        ),
    )
    chunks = []
    current_chunk = ""
    current_sentences = 0

    def flush_current():
        nonlocal current_chunk, current_sentences
        if current_chunk:
            chunks.append(current_chunk.strip())
        current_chunk = ""
        current_sentences = 0

    for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue

        if not chunks and current_chunk and len(current_chunk) >= first_target_chars:
            flush_current()

        if len(sentence) > target_chars:
            flush_current()
            chunks.extend(split_text_into_chunks(sentence, target_chars))
            continue

        active_target = first_target_chars if not chunks else target_chars
        would_exceed = (
            current_chunk and len(current_chunk) + len(sentence) + 1 > active_target
        )
        if would_exceed:
            flush_current()

        current_chunk = (current_chunk + " " + sentence).strip()
        current_sentences += 1
        if (
            not chunks
            and len(current_chunk) >= min_first_chars
            and (current_sentences > 1 or len(current_chunk) >= first_target_chars)
        ):
            flush_current()

    flush_current()
    return chunks or ([text] if text else [])


def _split_text_into_chunks(
    text: str, max_chars: int, keep_short_text_together: bool
) -> list[str]:
    max_chars = max(1, max_chars)
    if keep_short_text_together and len(text) <= max_chars:
        return [text] if text else []

    sentence_pattern = r"(?<=[.!?。！？])\s+"
    sentences = re.split(sentence_pattern, text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue
        sub_chunk = ""
        for part in re.split(r"([,;:，；：])", chunk):
            part = part.strip()
            if not part:
                continue
            if sub_chunk and len(sub_chunk) + len(part) + 1 > max_chars:
                final_chunks.append(sub_chunk.strip())
                sub_chunk = part
            else:
                sub_chunk = (sub_chunk + " " + part).strip()
        if sub_chunk:
            final_chunks.extend(_force_split_words(sub_chunk, max_chars))

    return final_chunks or ([text] if text else [])


def _force_split_words(text: str, max_chars: int) -> list[str]:
    chunks = []
    current = ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > max_chars:
            chunks.append(current.strip())
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        chunks.append(current.strip())
    return chunks


def get_available_vram_mb(gpu_index: int = 0):
    """Get available VRAM in MB, accounting for non-PyTorch CUDA allocations."""
    if torch.cuda.is_available():
        try:
            free_memory, _ = torch.cuda.mem_get_info(gpu_index)
            return free_memory / (1024 * 1024)
        except Exception:
            try:
                free_memory = torch.cuda.get_device_properties(
                    gpu_index
                ).total_memory - torch.cuda.memory_allocated(gpu_index)
                return free_memory / (1024 * 1024)
            except Exception:
                return 0
    return 0


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("[QTTS] Invalid %s=%r; using %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logging.warning("[QTTS] Invalid %s=%r; using %s", name, value, default)
        return default


def _contains_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text)


def _safe_file_stem(value: Optional[str], default: str = "default") -> str:
    if not value or not isinstance(value, str):
        value = default
    value = value.strip().replace("\\", "/").rsplit("/", 1)[-1]
    if value.lower().endswith(".wav"):
        value = value[:-4]
    value = SAFE_FILE_STEM_RE.sub("", value.replace("..", ""))
    return (value or default)[:100]


def _contained_file_path(
    base_dir: str, file_stem: Optional[str], extension: str
) -> str:
    if extension not in {".wav", ".txt"}:
        raise ValueError("Invalid file extension")
    safe_file_name = f"{_safe_file_stem(file_stem)}{extension}"
    base_path = os.path.realpath(base_dir)
    full_path = os.path.realpath(os.path.join(base_path, safe_file_name))
    if full_path != base_path and full_path.startswith(base_path + os.sep):
        return full_path
    raise ValueError("Invalid file path")


class CTTS:
    """
    Qwen3-TTS wrapper with OpenAI-compatible ezlocalai endpoint behavior.

    The class name remains CTTS so existing Pipes/app imports keep working.
    """

    def __init__(self, cache_config=None, device=None):
        self._closed = False
        self.model_id = resolve_tts_model_id()
        self.max_chunk_chars = max(
            1, _env_int("QWEN_TTS_MAX_CHUNK_CHARS", MAX_CHUNK_CHARS)
        )
        self.stream_chunk_target_chars = min(
            self.max_chunk_chars,
            max(
                1,
                _env_int("QWEN_TTS_STREAM_CHUNK_CHARS", STREAM_CHUNK_TARGET_CHARS),
            ),
        )
        self.stream_first_chunk_target_chars = min(
            self.stream_chunk_target_chars,
            max(
                1,
                _env_int(
                    "QWEN_TTS_STREAM_FIRST_CHUNK_CHARS",
                    STREAM_FIRST_CHUNK_TARGET_CHARS,
                ),
            ),
        )
        self.stream_min_first_chunk_chars = min(
            self.stream_first_chunk_target_chars,
            max(
                1,
                _env_int(
                    "QWEN_TTS_STREAM_MIN_FIRST_CHUNK_CHARS",
                    STREAM_MIN_FIRST_CHUNK_CHARS,
                ),
            ),
        )
        self.stream_write_bytes = max(
            1024, _env_int("QWEN_TTS_STREAM_WRITE_BYTES", STREAM_WRITE_BYTES)
        )
        self.stream_frame_drain_seconds = max(
            0.0,
            _env_float(
                "QWEN_TTS_STREAM_FRAME_DRAIN_SECONDS", STREAM_FRAME_DRAIN_SECONDS
            ),
        )
        self.stream_flush_silence_ms = max(
            0, _env_int("QWEN_TTS_STREAM_FLUSH_SILENCE_MS", STREAM_FLUSH_SILENCE_MS)
        )
        self.non_streaming_mode = _env_bool("QWEN_TTS_NON_STREAMING_MODE", True)
        self.default_x_vector_only = True
        self.allow_cpu_fallback = _env_bool("QWEN_TTS_ALLOW_CPU_FALLBACK", True)
        self.generation_kwargs = self._load_generation_kwargs()

        self.device = self._select_device(device)
        logging.info(
            "[QTTS] Initializing llama.cpp Qwen-TTS %s on %s (audio-only cloning)",
            self.model_id,
            self.device,
        )
        self.model = self._load_model(self.device)
        self.sample_rate = 24000

        self.output_folder = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_folder, exist_ok=True)
        self.voices_path = os.path.join(os.getcwd(), "voices")
        os.makedirs(self.voices_path, exist_ok=True)
        self.voices = sorted(
            Path(file).stem
            for file in os.listdir(self.voices_path)
            if file.lower().endswith(".wav")
        )
        logging.info("[QTTS] Found %s voice(s): %s", len(self.voices), self.voices)

        self.cache = AudioCache(cache_config)
        self.use_cache = cache_config.get("enabled", True) if cache_config else True
        logging.info(
            "[QTTS] Audio caching %s", "enabled" if self.use_cache else "disabled"
        )

        if os.path.exists(os.path.join(self.voices_path, "default.wav")):
            try:
                self._generate_single_sample(
                    "Hello.",
                    self._voice_audio_path("default"),
                    "English",
                    generation_kwargs=self._warmup_generation_kwargs(),
                )
                logging.info("[QTTS] Warmup generation complete")
            except Exception as e:
                logging.warning("[QTTS] Warmup generation failed (non-fatal): %s", e)

        logging.info("[QTTS] Qwen-TTS initialized successfully")

    def _select_device(self, requested_device: Optional[str]) -> str:
        requested = str(requested_device or "").strip().lower()
        gpu_index = 0
        if requested.startswith("cuda:"):
            try:
                gpu_index = int(requested.split(":", 1)[1])
            except ValueError:
                gpu_index = 0

        min_vram_mb = _env_int("QWEN_TTS_MIN_VRAM_MB", 3500)
        available_vram = get_available_vram_mb(gpu_index)
        if requested == "cpu":
            return "cpu"
        if torch.cuda.is_available() and requested.startswith("cuda"):
            if available_vram < min_vram_mb:
                logging.warning(
                    "[QTTS] Only %.0fMB VRAM available on GPU %s (preferred %sMB); "
                    "trying requested CUDA device first",
                    available_vram,
                    gpu_index,
                    min_vram_mb,
                )
            return requested
        if (
            torch.cuda.is_available()
            and not requested
            and available_vram >= min_vram_mb
        ):
            return "cuda:0"
        if torch.cuda.is_available() and not self.allow_cpu_fallback:
            logging.warning(
                "[QTTS] Only %.0fMB VRAM available on GPU %s (preferred %sMB); "
                "trying CUDA because Qwen-TTS CPU fallback is disabled",
                available_vram,
                gpu_index,
                min_vram_mb,
            )
            return "cuda:0"
        if torch.cuda.is_available():
            logging.warning(
                "[QTTS] Only %.0fMB VRAM available on GPU %s, using CPU (need %sMB)",
                available_vram,
                gpu_index,
                min_vram_mb,
            )
            return "cpu"
        if not self.allow_cpu_fallback:
            raise RuntimeError(
                "[QTTS] CUDA is not available and QWEN_TTS_ALLOW_CPU_FALLBACK is "
                "not enabled. Set QWEN_TTS_ALLOW_CPU_FALLBACK=true or request "
                "device='cpu' to try CPU generation."
            )
        return "cpu"

    def _load_model(self, device: str):
        try:
            return LlamaTTS(device, self.generation_kwargs)
        except RuntimeError:
            if device.startswith("cuda") and self.allow_cpu_fallback:
                logging.exception("[QTTS] Native GPU initialization failed; trying CPU")
                self.device = "cpu"
                return LlamaTTS("cpu", self.generation_kwargs)
            raise

    def _load_generation_kwargs(self) -> dict:
        raw = os.getenv("QWEN_TTS_GENERATE_KWARGS", "").strip()
        kwargs = {}
        if not raw:
            parsed = {}
        else:
            try:
                import json

                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    logging.warning(
                        "[QTTS] QWEN_TTS_GENERATE_KWARGS must be a JSON object"
                    )
                    parsed = {}
            except Exception as e:
                logging.warning(
                    "[QTTS] Could not parse QWEN_TTS_GENERATE_KWARGS: %s", e
                )
                parsed = {}

        kwargs.update(parsed)
        max_tokens = _env_int("QWEN_TTS_MAX_NEW_TOKENS", 320)
        if (
            max_tokens > 0
            and "max_new_tokens" not in kwargs
            and "max_length" not in kwargs
        ):
            kwargs["max_new_tokens"] = max_tokens
        return kwargs

    def _warmup_generation_kwargs(self) -> dict:
        kwargs = dict(self.generation_kwargs)
        max_tokens = _env_int("QWEN_TTS_WARMUP_MAX_NEW_TOKENS", 32)
        if max_tokens <= 0:
            return kwargs

        configured = kwargs.get("max_new_tokens")
        if isinstance(configured, int) and configured > 0:
            kwargs["max_new_tokens"] = min(configured, max_tokens)
        else:
            kwargs["max_new_tokens"] = max_tokens
        return kwargs

    def _release_generation_memory(self):
        gc.collect()
        if (
            str(getattr(self, "device", "")).startswith("cuda")
            and torch.cuda.is_available()
        ):
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logging.debug("[QTTS] CUDA cache cleanup skipped: %s", e)

    def close(self):
        """Release the native TTS model even if callers still hold this wrapper."""
        self._closed = True
        model = getattr(self, "model", None)
        self.model = None
        if model is not None and hasattr(model, "close"):
            model.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _resolve_language(self, language: Optional[str], text: str) -> str:
        requested = LANGUAGE_ALIASES.get(str(language or "").strip().lower())
        if requested is None:
            requested = str(language).strip().title()
        if requested == "English" and _contains_cyrillic(text):
            return "Auto"
        return requested or "Auto"

    def _prepare_text(self, text: str, qwen_language: str) -> str:
        if qwen_language == "English" and not _contains_cyrillic(text):
            text = normalize_english_text_for_tts(text)
        return clean_text_for_tts(text)

    def _voice_audio_path(self, voice: Optional[str]) -> Optional[str]:
        voice_name = _safe_file_stem(voice)
        audio_path = _contained_file_path(self.voices_path, voice_name, ".wav")
        if os.path.exists(audio_path):
            return audio_path
        default_voice = _contained_file_path(self.voices_path, "default", ".wav")
        if os.path.exists(default_voice):
            logging.warning(
                "[QTTS] No voice file found for %r; using default.wav", voice
            )
            return default_voice
        logging.warning("[QTTS] No voice file found for %r and no default.wav", voice)
        return None

    def _voice_ref_text(self, audio_path: Optional[str]) -> Optional[str]:
        if not audio_path:
            return None
        voice_name = _safe_file_stem(os.path.basename(audio_path))
        sidecar = _contained_file_path(self.voices_path, voice_name, ".txt")
        if os.path.exists(sidecar):
            try:
                with open(sidecar, encoding="utf-8") as f:
                    return f.read().strip() or None
            except OSError as e:
                logging.warning(
                    "[QTTS] Could not read voice transcript %s: %s", sidecar, e
                )
        return None

    def _voice_clone_context(
        self, audio_path: Optional[str]
    ) -> tuple[Optional[str], bool, str]:
        # libmtmd uses the speaker encoder's x-vector, not transcript conditioning.
        # Existing .txt voice sidecars remain on disk for future upstream support.
        return None, True, ""

    async def generate(
        self,
        text,
        voice="default",
        language="auto",
        local_uri=None,
        output_file_name=None,
        use_cache=None,
    ):
        if use_cache is None:
            use_cache = self.use_cache

        voice_name = _safe_file_stem(voice)
        qwen_language = self._resolve_language(language, text)
        text = self._prepare_text(text, qwen_language)
        if not text:
            return ""

        audio_path = self._voice_audio_path(voice)
        ref_text, x_vector_only, transcript_hash = self._voice_clone_context(audio_path)
        cache_extra = {
            "engine": "llama.cpp-qwen-tts-v1",
            "model": self.model_id,
            "x_vector_only": x_vector_only,
            "ref_text_sha256": transcript_hash,
            "generation": self.generation_kwargs,
            "model_file": os.getenv("QWEN_TTS_MODEL_FILE", "default"),
            "mmproj_file": os.getenv("QWEN_TTS_MMPROJ_FILE", "default"),
            "revision": os.getenv("QWEN_TTS_REVISION", "default"),
        }
        if use_cache:
            cache_key = self.cache.generate_cache_key(
                text,
                voice_name,
                qwen_language,
                extra_params=cache_extra,
            )
            cached_file_path = os.path.join(
                self.output_folder, "cache", "audio", f"{cache_key}.wav"
            )
            if os.path.exists(cached_file_path):
                cached_audio = self.cache.get_cached_audio(cache_key)
                if cached_audio:
                    if local_uri:
                        return f"{local_uri}/outputs/cache/audio/{cache_key}.wav"
                    return base64.b64encode(cached_audio).decode("utf-8")

        chunks = split_text_into_chunks(text, self.max_chunk_chars)
        logging.info(
            "[QTTS] Generating %s chunk(s), language=%s, voice=%s, mode=%s",
            len(chunks),
            qwen_language,
            voice_name,
            "x-vector" if x_vector_only else "transcript",
        )

        audio_chunks = []
        sample_rate = self.sample_rate
        for i, chunk in enumerate(chunks):
            logging.debug(
                "[QTTS] Generating chunk %s/%s: %s", i + 1, len(chunks), chunk[:80]
            )
            wav, sample_rate = await asyncio.to_thread(
                self._generate_single_sample,
                chunk,
                audio_path,
                qwen_language,
                ref_text,
                x_vector_only,
            )
            if wav is not None and len(wav) > 0:
                audio_chunks.append(self._to_mono_float32(wav))

        if not audio_chunks:
            logging.warning("[QTTS] No audio generated")
            return ""

        final_audio = (
            audio_chunks[0]
            if len(audio_chunks) == 1
            else np.concatenate(audio_chunks, axis=0)
        )
        self.sample_rate = int(sample_rate)
        audio_data = self._array_to_wav_bytes(final_audio, self.sample_rate)
        if not audio_data:
            return ""

        if use_cache:
            cache_key = self.cache.generate_cache_key(
                text,
                voice_name,
                qwen_language,
                extra_params=cache_extra,
            )
            metadata = {
                "text": text,
                "voice": voice_name,
                "language": qwen_language,
                "generation_method": "llama.cpp-qwen-tts",
                "model": self.model_id,
                "chunks": len(chunks),
            }
            self.cache.store_cached_audio(cache_key, audio_data, metadata)

        if local_uri:
            output_stem = _safe_file_stem(output_file_name, uuid.uuid4().hex)
            output_file_name = f"{output_stem}.wav"
            output_file = _contained_file_path(self.output_folder, output_stem, ".wav")
            with open(output_file, "wb") as f:
                f.write(audio_data)
            return f"{local_uri}/outputs/{output_file_name}"
        return base64.b64encode(audio_data).decode("utf-8")

    def _generate_single_sample(
        self,
        text,
        audio_path,
        qwen_language,
        ref_text=None,
        x_vector_only=None,
        generation_kwargs=None,
    ):
        """Generate one Qwen-TTS sample and return (audio_array, sample_rate)."""
        if not audio_path:
            raise ValueError("[QTTS] Qwen Base TTS requires a reference voice wav")

        if x_vector_only is None:
            ref_text, x_vector_only, _ = self._voice_clone_context(audio_path)
        kwargs = (
            self.generation_kwargs if generation_kwargs is None else generation_kwargs
        )

        try:
            with torch.inference_mode():
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    language=qwen_language,
                    ref_audio=audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                    non_streaming_mode=self.non_streaming_mode,
                    **kwargs,
                )
            return wavs[0], int(sample_rate)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if getattr(self, "_closed", False):
                raise
            error = str(e).lower()
            if "out of memory" in error or "cuda" in error or "cudnn" in error:
                if not self.allow_cpu_fallback:
                    raise RuntimeError(
                        "[QTTS] GPU generation failed and QWEN_TTS_ALLOW_CPU_FALLBACK "
                        "is not enabled. Reduce TTS_N_PARALLEL, free VRAM, or set "
                        "QWEN_TTS_ALLOW_CPU_FALLBACK=true to try CPU generation."
                    ) from e
                logging.warning("[QTTS] GPU generation failed, reloading on CPU: %s", e)
                self._reload_on_cpu()
                with torch.inference_mode():
                    wavs, sample_rate = self.model.generate_voice_clone(
                        text=text,
                        language=qwen_language,
                        ref_audio=audio_path,
                        ref_text=ref_text,
                        x_vector_only_mode=x_vector_only,
                        non_streaming_mode=self.non_streaming_mode,
                        **kwargs,
                    )
                return wavs[0], int(sample_rate)
            logging.error("[QTTS] Error generating audio: %s", e)
            raise
        finally:
            self._release_generation_memory()

    def _reload_on_cpu(self):
        if self.model is not None:
            self.model.close()
        self._release_generation_memory()
        self.device = "cpu"
        self.model = LlamaTTS("cpu", self.generation_kwargs)

    def _to_mono_float32(self, wav) -> np.ndarray:
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim == 2:
            if audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
                audio = audio.T
            audio = audio.mean(axis=1)
        return np.ascontiguousarray(audio)

    def _array_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        try:
            audio = self._to_mono_float32(audio)
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="WAV")
            return buffer.getvalue()
        except Exception as e:
            logging.error("[QTTS] Error converting audio to WAV bytes: %s", e)
            return b""

    def _bytes_to_tensor(self, audio_bytes):
        try:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))
            waveform = torch.tensor(audio_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.T
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform
        except Exception as e:
            logging.error("[QTTS] Error converting bytes to tensor: %s", e)
            return None

    def _tensor_to_bytes(self, tensor):
        try:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            audio_np = tensor.detach().cpu().numpy()
            if audio_np.shape[0] <= 2:
                audio_np = audio_np.T
            return self._array_to_wav_bytes(audio_np, self.sample_rate)
        except Exception as e:
            logging.error("[QTTS] Error converting tensor to bytes: %s", e)
            return None

    def get_cache_stats(self):
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self, voice=None):
        """Clear the audio cache."""
        self.cache.clear_cache(voice=voice)
        logging.info(
            "[QTTS] Cache cleared for %s", f"voice: {voice}" if voice else "all voices"
        )

    async def generate_stream(
        self,
        text,
        voice="default",
        language="auto",
    ):
        """
        Generate TTS audio as a stream of PCM chunks.

        Emit native vocoder windows before the full text segment completes.
        Text segmentation is still needed: upstream cannot append text mid-prompt.
        """
        import struct

        qwen_language = self._resolve_language(language, text)
        text = self._prepare_text(text, qwen_language)
        audio_path = self._voice_audio_path(voice)
        ref_text, x_vector_only, _ = self._voice_clone_context(audio_path)
        chunks = split_text_into_stream_chunks(
            text,
            target_chars=self.stream_chunk_target_chars,
            first_target_chars=self.stream_first_chunk_target_chars,
            min_first_chars=self.stream_min_first_chunk_chars,
        )

        logging.info(
            "[QTTS] Streaming TTS: %s chunk(s), language=%s, chars=%s, "
            "stream_targets=%s/%s/%s, mode=%s",
            len(chunks),
            qwen_language,
            len(text),
            self.stream_first_chunk_target_chars,
            self.stream_chunk_target_chars,
            self.stream_min_first_chunk_chars,
            "x-vector" if x_vector_only else "transcript",
        )

        yield struct.pack("<IHH", self.sample_rate, 16, 1)

        for i, chunk in enumerate(chunks):
            try:
                async with aclosing(
                    stream_native_audio(
                        self.model,
                        text=chunk,
                        language=qwen_language,
                        ref_audio=audio_path,
                        **self.generation_kwargs,
                    )
                ) as stream:
                    async for wav, sample_rate in stream:
                        if int(sample_rate) != self.sample_rate:
                            raise RuntimeError(
                                "TTS sample rate changed after stream header"
                            )
                        pcm_data = self._array_to_pcm16_bytes(wav)
                        if not pcm_data:
                            continue
                        yield struct.pack("<I", len(pcm_data))
                        for offset in range(0, len(pcm_data), self.stream_write_bytes):
                            yield pcm_data[offset : offset + self.stream_write_bytes]
                            await asyncio.sleep(0)
                if i < len(chunks) - 1 and self.stream_flush_silence_ms > 0:
                    silence_samples = max(
                        1, int(self.sample_rate * self.stream_flush_silence_ms / 1000)
                    )
                    silence_data = b"\x00\x00" * silence_samples
                    yield struct.pack("<I", len(silence_data)) + silence_data
                if self.stream_frame_drain_seconds > 0:
                    await asyncio.sleep(self.stream_frame_drain_seconds)
            except BaseException as e:
                # The native bridge kills an interrupted worker. Never put that
                # dead instance back into a dedicated voice worker's warm pool.
                self.close()
                if isinstance(e, Exception):
                    logging.error(
                        "[QTTS] Error generating stream chunk %s: %s", i + 1, e
                    )
                raise

        yield struct.pack("<I", 0)
        logging.info("[QTTS] Streaming TTS complete")

    def _array_to_pcm16_bytes(self, wav) -> bytes:
        audio = self._to_mono_float32(wav)
        audio = np.nan_to_num(audio)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype("<i2").tobytes()
