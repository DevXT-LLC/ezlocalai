import os
import re
import uuid
import base64
import torch
import torchaudio
import logging
import gc
import io
from ezlocalai.AudioCache import AudioCache

from chatterbox.tts import ChatterboxTTS

# Maximum characters per chunk for TTS generation
# Chatterbox struggles with long text, so we split into sentences
MAX_CHUNK_CHARS = 250


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
    Chatterbox TTS wrapper with voice cloning support.
    Automatically falls back to CPU if GPU memory is insufficient.
    """

    def __init__(self, cache_config=None):
        # Check if there's enough VRAM for TTS (need at least 2GB free)
        available_vram = get_available_vram_mb()
        min_vram_mb = 2000  # 2GB minimum for TTS

        if torch.cuda.is_available() and available_vram >= min_vram_mb:
            self.device = "cuda"
        else:
            self.device = "cpu"
            if torch.cuda.is_available():
                logging.debug(
                    f"[CTTS] Only {available_vram:.0f}MB VRAM available, using CPU (need {min_vram_mb}MB)"
                )

        logging.debug(f"[CTTS] Initializing Chatterbox TTS on {self.device}")

        try:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logging.warning(f"[CTTS] GPU OOM during init, falling back to CPU: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                self.model = ChatterboxTTS.from_pretrained(device="cpu")
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
        logging.debug("[CTTS] Chatterbox TTS initialized successfully")

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
            waveform, sr = torchaudio.load(temp_file)
            os.remove(temp_file)
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
            torchaudio.save(temp_file, tensor.cpu(), self.sample_rate)
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
        try:
            if audio_path and os.path.exists(audio_path):
                wav = self.model.generate(text, audio_prompt_path=audio_path)
            else:
                wav = self.model.generate(text)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logging.warning(
                    f"[CTTS] GPU OOM during generation, reloading model on CPU: {e}"
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
                self.model = ChatterboxTTS.from_pretrained(device="cpu")
                self.sample_rate = self.model.sr
                logging.debug("[CTTS] Model reloaded on CPU, retrying generation...")

                # Retry on CPU
                if audio_path and os.path.exists(audio_path):
                    wav = self.model.generate(text, audio_prompt_path=audio_path)
                else:
                    wav = self.model.generate(text)
            else:
                raise

        temp_file = os.path.join(self.output_folder, f"temp_{uuid.uuid4().hex}.wav")

        try:
            if isinstance(wav, torch.Tensor):
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                torchaudio.save(temp_file, wav.cpu(), self.sample_rate)
            else:
                wav_tensor = torch.tensor(wav)
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                torchaudio.save(temp_file, wav_tensor, self.sample_rate)

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
