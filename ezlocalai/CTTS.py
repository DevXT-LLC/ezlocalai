import os
import re
import uuid
import base64
import torch
import torchaudio
import logging
from ezlocalai.AudioCache import AudioCache

from chatterbox.tts import ChatterboxTTS


class CTTS:
    """
    Chatterbox TTS wrapper with voice cloning support.
    """

    def __init__(self, cache_config=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"[CTTS] Initializing Chatterbox TTS on {self.device}")

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
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
        logging.info(f"[CTTS] Found {len(self.voices)} voice(s): {self.voices}")

        # Initialize audio cache
        self.cache = AudioCache(cache_config)

        # Cache statistics tracking
        self.use_cache = cache_config.get("enabled", True) if cache_config else True
        logging.info(
            f"[CTTS] Audio caching {'enabled' if self.use_cache else 'disabled'}"
        )
        logging.info("[CTTS] Chatterbox TTS initialized successfully")

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

        # Generate audio directly (Chatterbox handles long text well)
        audio_data = self._generate_single_sample(text, audio_path)

        if not audio_data:
            logging.warning("[CTTS] No audio generated")
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
            }
            self.cache.store_cached_audio(cache_key, audio_data, metadata)

        # Return result
        if local_uri:
            return f"{local_uri}/outputs/{output_file_name}"
        else:
            os.remove(output_file)
            return base64.b64encode(audio_data).decode("utf-8")

    def _generate_single_sample(self, text, audio_path):
        """Generate a single audio sample using Chatterbox."""
        try:
            if audio_path and os.path.exists(audio_path):
                wav = self.model.generate(text, audio_prompt_path=audio_path)
            else:
                wav = self.model.generate(text)

            temp_file = os.path.join(self.output_folder, f"temp_{uuid.uuid4().hex}.wav")

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
        logging.info(
            f"[CTTS] Cache cleared for {'voice: ' + voice if voice else 'all voices'}"
        )
