import os
import re
import uuid
import base64
import torch
import torchaudio
import soundfile as sf
import logging
import gc
import io
from pathlib import Path
from huggingface_hub import hf_hub_download
from ezlocalai.AudioCache import AudioCache

# Use Chatterbox Turbo for faster inference (350M vs 500M params, single-step decoder)
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Turbo model repo ID
TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"
# Files needed for Turbo model
TURBO_FILES = [
    "ve.safetensors",
    "t3_turbo_v1.safetensors",
    "s3gen_meanflow.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "conds.pt",
]


def download_turbo_model() -> Path:
    """Download Chatterbox Turbo model files without requiring HF token."""
    local_path = None
    for fpath in TURBO_FILES:
        try:
            local_path = hf_hub_download(repo_id=TURBO_REPO_ID, filename=fpath)
        except Exception as e:
            logging.warning(f"[CTTS] Could not download {fpath}: {e}")
            continue
    if local_path is None:
        raise RuntimeError("Failed to download any Chatterbox Turbo files")
    return Path(local_path).parent


# Maximum characters per chunk for TTS generation
# Chatterbox struggles with long text, so we split into sentences
MAX_CHUNK_CHARS = 250

# Number to word conversion for TTS
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
    return str(n)  # For very large numbers, just return as-is


def normalize_text_for_tts(text: str) -> str:
    """Normalize text for better TTS output - convert numbers, dates, times to words."""

    # Convert time formats like "10:30 AM" or "2:45 PM" to words
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

    # Convert date formats like "12/24" or "12/24/2025" to words
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

        # Ordinal suffixes
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
                # Say "nineteen ninety five" for 1995
                result += (
                    " "
                    + number_to_words(year_int // 100)
                    + " "
                    + number_to_words(year_int % 100)
                )
        return result

    text = re.sub(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", date_to_words, text)

    # Convert standalone numbers to words (but not in middle of alphanumeric strings)
    def num_to_words_replace(match):
        num = int(match.group(0))
        return number_to_words(num)

    text = re.sub(r"\b(\d{1,6})\b", num_to_words_replace, text)

    return text


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Split text into sentence-based chunks for TTS generation.
    Chatterbox TTS produces nonsense/hallucinations on long text,
    so we need to process it in smaller chunks.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence boundaries (. ! ?)
    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence would exceed max, save current chunk and start new one
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Handle case where a single sentence is too long - split on commas or force split
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Try splitting on commas
            comma_parts = chunk.split(",")
            sub_chunk = ""
            for part in comma_parts:
                part = part.strip()
                if not part:
                    continue
                if sub_chunk and len(sub_chunk) + len(part) + 2 > max_chars:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = part
                else:
                    if sub_chunk:
                        sub_chunk += ", " + part
                    else:
                        sub_chunk = part
            if sub_chunk.strip():
                # Force split if still too long
                if len(sub_chunk) > max_chars:
                    words = sub_chunk.split()
                    word_chunk = ""
                    for word in words:
                        if word_chunk and len(word_chunk) + len(word) + 1 > max_chars:
                            final_chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk = (word_chunk + " " + word).strip()
                    if word_chunk:
                        final_chunks.append(word_chunk.strip())
                else:
                    final_chunks.append(sub_chunk.strip())

    return final_chunks if final_chunks else [text[:max_chars]]


def get_available_vram_mb():
    """Get available VRAM in MB."""
    if torch.cuda.is_available():
        try:
            free_memory = torch.cuda.get_device_properties(
                0
            ).total_memory - torch.cuda.memory_allocated(0)
            return free_memory / (1024 * 1024)
        except:
            return 0
    return 0


class CTTS:
    """
    Chatterbox Turbo TTS wrapper with voice cloning support.
    Turbo is a 350M parameter model with single-step decoder for faster inference.
    Supports paralinguistic tags like [laugh], [cough], [chuckle].
    Automatically falls back to CPU if GPU memory is insufficient.
    """

    def __init__(self, cache_config=None):
        # Check if there's enough VRAM for TTS (Turbo needs less VRAM than regular)
        available_vram = get_available_vram_mb()
        min_vram_mb = 1500  # 1.5GB minimum for Turbo (smaller than regular 500M model)

        if torch.cuda.is_available() and available_vram >= min_vram_mb:
            self.device = "cuda"
        else:
            self.device = "cpu"
            if torch.cuda.is_available():
                logging.debug(
                    f"[CTTS] Only {available_vram:.0f}MB VRAM available, using CPU (need {min_vram_mb}MB)"
                )

        logging.debug(f"[CTTS] Initializing Chatterbox Turbo TTS on {self.device}")

        # Download model files manually to avoid HF token requirement
        model_path = download_turbo_model()
        logging.debug(f"[CTTS] Model downloaded to {model_path}")

        try:
            self.model = ChatterboxTurboTTS.from_local(model_path, device=self.device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            # Check for OOM, CUDA errors, or cuDNN errors (version mismatch, etc.)
            if (
                "out of memory" in error_str
                or "cuda" in error_str
                or "cudnn" in error_str
            ):
                logging.warning(
                    f"[CTTS] GPU error during init, falling back to CPU: {e}"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                self.model = ChatterboxTurboTTS.from_local(model_path, device="cpu")
            else:
                raise

        self.sample_rate = self.model.sr

        self.output_folder = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_folder, exist_ok=True)
        self.voices_path = os.path.join(os.getcwd(), "voices")
        os.makedirs(self.voices_path, exist_ok=True)
        wav_files = []
        for file in os.listdir(self.voices_path):
            if file.endswith(".wav"):
                wav_files.append(file.replace(".wav", ""))
        self.voices = wav_files
        logging.debug(f"[CTTS] Found {len(self.voices)} voice(s): {self.voices}")

        # Initialize audio cache
        self.cache = AudioCache(cache_config)

        # Cache statistics tracking
        self.use_cache = cache_config.get("enabled", True) if cache_config else True
        logging.debug(
            f"[CTTS] Audio caching {'enabled' if self.use_cache else 'disabled'}"
        )

        # Pre-condition the default voice at load time for faster first request
        default_voice = os.path.join(self.voices_path, "default.wav")
        if os.path.exists(default_voice):
            try:
                logging.debug("[CTTS] Pre-conditioning default voice...")
                # prepare_conditionals loads the voice and creates embeddings
                # norm_loudness=False to work around dtype bug in chatterbox library
                self.model.prepare_conditionals(default_voice, norm_loudness=False)
                logging.debug("[CTTS] Default voice pre-conditioned successfully")

                # Do a warmup generation to fully initialize the model pipeline
                # This eliminates first-request latency from JIT compilation, etc.
                logging.debug("[CTTS] Warming up TTS with silent generation...")
                try:
                    _ = self._generate_single_sample("Hello.", default_voice)
                    logging.debug("[CTTS] TTS warmup complete")
                except Exception as warmup_err:
                    logging.warning(
                        f"[CTTS] Warmup generation failed (non-fatal): {warmup_err}"
                    )
            except Exception as e:
                logging.warning(f"[CTTS] Could not pre-condition default voice: {e}")

        logging.debug("[CTTS] Chatterbox Turbo TTS initialized successfully")

    async def generate(
        self,
        text,
        voice="default",
        language="en",
        local_uri=None,
        output_file_name=None,
        use_cache=None,  # Allow override of cache usage
    ):
        # Use cache setting from init if not explicitly overridden
        if use_cache is None:
            use_cache = self.use_cache

        # Normalize numbers, dates, times to words for better TTS
        text = normalize_text_for_tts(text)

        # Clean and normalize text
        # Remove remaining non-ASCII characters (Chatterbox TTS only handles English well)
        # This prevents "ba ba ba" garbage audio from Chinese, Arabic, etc.
        text = "".join(char for char in text if ord(char) < 128 or char in ".,;:!?-'\"")

        # Clean up repeated punctuation
        cleaned_string = re.sub(r"([!?.])\1+", r"\1", text)
        # Remove any remaining special characters except basic punctuation
        cleaned_string = re.sub(
            r'[^a-zA-Z0-9\s\.,;:!?\-\'"]',
            "",
            cleaned_string,
        )
        cleaned_string = re.sub(r"\n+", " ", cleaned_string)
        # Clean up multiple spaces that may result from removing characters
        cleaned_string = re.sub(r"\s+", " ", cleaned_string)
        text = cleaned_string.replace("#", "").strip()

        # Normalize voice name
        voice_name = voice
        if not voice.endswith(".wav"):
            voice = f"{voice}.wav"

        # Check cache first if enabled
        if use_cache:
            cache_key = self.cache.generate_cache_key(text, voice_name, language)

            # Check if cached file exists
            cached_file_path = os.path.join(
                self.output_folder, "cache", "audio", f"{cache_key}.wav"
            )

            if os.path.exists(cached_file_path):
                # Update cache statistics
                cached_audio = self.cache.get_cached_audio(cache_key)

                if cached_audio:
                    if local_uri:
                        # Return URL to existing cached file
                        return f"{local_uri}/outputs/cache/audio/{cache_key}.wav"
                    else:
                        # Return base64 encoded
                        return base64.b64encode(cached_audio).decode("utf-8")

        # If not cached or cache disabled, generate new audio
        audio_path = os.path.join(self.voices_path, voice)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.voices_path, "default.wav")
            if not os.path.exists(audio_path):
                logging.warning(
                    f"[CTTS] No voice file found for '{voice}' and no default.wav"
                )
                audio_path = None

        # Split long text into chunks to prevent hallucinations
        chunks = split_text_into_chunks(text)
        if len(chunks) > 1:
            logging.debug(f"[CTTS] Split text into {len(chunks)} chunks for generation")

        # Generate audio for each chunk and concatenate
        all_audio_tensors = []
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logging.debug(
                    f"[CTTS] Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}..."
                )
            chunk_audio = self._generate_single_sample(chunk, audio_path)
            if chunk_audio:
                # Convert bytes to tensor for concatenation
                chunk_tensor = self._bytes_to_tensor(chunk_audio)
                if chunk_tensor is not None:
                    all_audio_tensors.append(chunk_tensor)

        if not all_audio_tensors:
            logging.warning("[CTTS] No audio generated")
            return ""

        # Concatenate all audio chunks
        if len(all_audio_tensors) == 1:
            final_tensor = all_audio_tensors[0]
        else:
            final_tensor = torch.cat(all_audio_tensors, dim=1)

        # Convert back to bytes
        audio_data = self._tensor_to_bytes(final_tensor)

        if not audio_data:
            logging.warning("[CTTS] Failed to convert audio to bytes")
            return ""

        # Export final audio
        if not output_file_name:
            output_file_name = f"{uuid.uuid4().hex}.wav"
        output_file = os.path.join(self.output_folder, output_file_name)

        with open(output_file, "wb") as f:
            f.write(audio_data)

        # Store in cache if enabled
        if use_cache:
            cache_key = self.cache.generate_cache_key(text, voice_name, language)
            metadata = {
                "text": text,
                "voice": voice_name,
                "language": language,
                "generation_method": "chatterbox",
                "chunks": len(chunks),
            }
            self.cache.store_cached_audio(cache_key, audio_data, metadata)

        # Return result
        if local_uri:
            return f"{local_uri}/outputs/{output_file_name}"
        else:
            os.remove(output_file)
            return base64.b64encode(audio_data).decode("utf-8")

    def _bytes_to_tensor(self, audio_bytes):
        """Convert audio bytes to a torch tensor."""
        try:
            temp_file = os.path.join(
                self.output_folder, f"temp_read_{uuid.uuid4().hex}.wav"
            )
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
            # Use soundfile instead of torchaudio.load (PyTorch 2.9.1+ requires torchcodec)
            audio_np, sr = sf.read(temp_file)
            os.remove(temp_file)
            # Convert to tensor in channels-first format
            waveform = torch.tensor(audio_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            elif waveform.ndim == 2:
                waveform = (
                    waveform.T
                )  # Convert from (samples, channels) to (channels, samples)
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform
        except Exception as e:
            logging.error(f"[CTTS] Error converting bytes to tensor: {e}")
            return None

    def _tensor_to_bytes(self, tensor):
        """Convert a torch tensor to audio bytes."""
        try:
            temp_file = os.path.join(
                self.output_folder, f"temp_write_{uuid.uuid4().hex}.wav"
            )
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            # Use soundfile directly instead of torchaudio.save (PyTorch 2.9.1+ ignores backend param)
            audio_np = tensor.cpu().numpy()
            if audio_np.shape[0] <= 2:  # channels first format
                audio_np = audio_np.T  # soundfile expects (samples, channels)
            sf.write(temp_file, audio_np, self.sample_rate)
            with open(temp_file, "rb") as f:
                audio_data = f.read()
            os.remove(temp_file)
            return audio_data
        except Exception as e:
            logging.error(f"[CTTS] Error converting tensor to bytes: {e}")
            return None

    def _generate_single_sample(self, text, audio_path):
        """Generate a single audio sample using Chatterbox for a short text chunk.

        This method should be called with short text (< 250 chars) to avoid
        hallucinations. For longer text, use the generate() method which
        automatically chunks the text.

        Automatically falls back to CPU if GPU runs out of memory.
        """
        # Turbo requires an audio prompt for voice cloning
        # If no audio_path provided, use default voice
        if not audio_path or not os.path.exists(audio_path):
            default_voice = os.path.join(self.voices_path, "default.wav")
            if os.path.exists(default_voice):
                audio_path = default_voice
            else:
                raise ValueError(
                    "[CTTS] Turbo model requires an audio prompt. No default voice found."
                )

        try:
            # norm_loudness=False to work around dtype bug in chatterbox library
            # (norm_loudness converts float32 to float64 which breaks mel spectrogram)
            wav = self.model.generate(
                text, audio_prompt_path=audio_path, norm_loudness=False
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            # Check for OOM, CUDA errors, or cuDNN errors (version mismatch, etc.)
            if (
                "out of memory" in error_str
                or "cuda" in error_str
                or "cudnn" in error_str
            ):
                logging.warning(
                    f"[CTTS] GPU error during generation, reloading model on CPU: {e}"
                )
                # Free GPU memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reload model on CPU
                del self.model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.device = "cpu"
                model_path = download_turbo_model()
                self.model = ChatterboxTurboTTS.from_local(model_path, device="cpu")
                self.sample_rate = self.model.sr
                logging.debug("[CTTS] Model reloaded on CPU, retrying generation...")

                # Retry on CPU
                wav = self.model.generate(
                    text, audio_prompt_path=audio_path, norm_loudness=False
                )
            else:
                raise

        temp_file = os.path.join(self.output_folder, f"temp_{uuid.uuid4().hex}.wav")

        try:
            # Use soundfile directly instead of torchaudio.save (PyTorch 2.9.1+ ignores backend param)
            if isinstance(wav, torch.Tensor):
                audio_np = wav.cpu().numpy()
            else:
                import numpy as np

                audio_np = np.array(wav)

            # Ensure correct shape for soundfile (samples, channels) or (samples,) for mono
            if audio_np.ndim == 2 and audio_np.shape[0] <= 2:
                audio_np = (
                    audio_np.T
                )  # Convert from (channels, samples) to (samples, channels)
            elif audio_np.ndim == 1:
                pass  # Already in correct shape for mono

            sf.write(temp_file, audio_np, self.sample_rate)

            with open(temp_file, "rb") as f:
                audio_data = f.read()
            os.remove(temp_file)

            return audio_data

        except Exception as e:
            logging.error(f"[CTTS] Error generating audio: {e}")
            raise

    def get_cache_stats(self):
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self, voice=None):
        """Clear the audio cache."""
        self.cache.clear_cache(voice=voice)
        logging.debug(
            f"[CTTS] Cache cleared for {'voice: ' + voice if voice else 'all voices'}"
        )

    async def generate_stream(
        self,
        text,
        voice="default",
        language="en",
    ):
        """
        Generate TTS audio as a stream of PCM chunks.

        Yields raw PCM audio bytes (24kHz, 16-bit, mono) for each text chunk
        as it's generated. This enables real-time playback without waiting
        for the entire audio to be generated.

        The first yield includes audio format info as a header:
        - 4 bytes: sample rate (uint32, little-endian)
        - 2 bytes: bits per sample (uint16, little-endian)
        - 2 bytes: channels (uint16, little-endian)

        Subsequent yields are raw PCM data.
        """
        import struct

        # Normalize numbers, dates, times to words for better TTS
        text = normalize_text_for_tts(text)

        # Clean and normalize text
        cleaned_string = re.sub(r"([!?.])\1+", r"\1", text)
        cleaned_string = re.sub(
            r'[^a-zA-Z0-9\s\.,;:!?\-\'"\u0400-\u04FFÀ-ÿ\u0150\u0151\u0170\u0171]\$',
            "",
            cleaned_string,
        )
        cleaned_string = re.sub(r"\n+", " ", cleaned_string)
        text = cleaned_string.replace("#", "")

        # Normalize voice name
        if not voice.endswith(".wav"):
            voice = f"{voice}.wav"

        audio_path = os.path.join(self.voices_path, voice)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.voices_path, "default.wav")
            if not os.path.exists(audio_path):
                logging.warning(
                    f"[CTTS] No voice file found for '{voice}' and no default.wav"
                )
                audio_path = None

        # Split text into chunks
        chunks = split_text_into_chunks(text)
        logging.info(
            f"[CTTS] Streaming TTS: {len(chunks)} chunks for {len(text)} chars"
        )

        # Yield header with audio format info (sample_rate=24000, bits=16, channels=1)
        header = struct.pack("<IHH", self.sample_rate, 16, 1)
        yield header

        # Generate and yield each chunk
        for i, chunk in enumerate(chunks):
            logging.debug(
                f"[CTTS] Streaming chunk {i+1}/{len(chunks)}: {chunk[:50]}..."
            )

            try:
                # Generate audio for this chunk
                audio_bytes = self._generate_single_sample(chunk, audio_path)

                if audio_bytes:
                    # Extract raw PCM by finding the "data" chunk in WAV
                    # WAV files can have variable-length headers with metadata
                    pcm_data = None
                    if len(audio_bytes) > 12:
                        # Search for "data" chunk marker
                        data_pos = audio_bytes.find(b"data")
                        if data_pos >= 0 and data_pos + 8 <= len(audio_bytes):
                            # Read chunk size (4 bytes after "data")
                            chunk_size = struct.unpack(
                                "<I", audio_bytes[data_pos + 4 : data_pos + 8]
                            )[0]
                            pcm_start = data_pos + 8
                            if pcm_start + chunk_size <= len(audio_bytes):
                                pcm_data = audio_bytes[
                                    pcm_start : pcm_start + chunk_size
                                ]
                            else:
                                # Use remaining bytes if size is wrong
                                pcm_data = audio_bytes[pcm_start:]
                        else:
                            # Fallback: assume 44-byte header
                            pcm_data = audio_bytes[44:]

                    if pcm_data:
                        # Yield chunk size followed by PCM data
                        chunk_header = struct.pack("<I", len(pcm_data))
                        yield chunk_header + pcm_data
                        logging.debug(
                            f"[CTTS] Yielded {len(pcm_data)} bytes for chunk {i+1}"
                        )

            except Exception as e:
                logging.error(f"[CTTS] Error generating chunk {i+1}: {e}")
                continue

        # Yield end marker (zero-length chunk)
        yield struct.pack("<I", 0)
        logging.info("[CTTS] Streaming TTS complete")
