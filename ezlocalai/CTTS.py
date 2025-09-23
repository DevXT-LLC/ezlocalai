import os
import re
import uuid
import base64
import torch
import torchaudio
import requests
import logging
from ezlocalai.Helpers import chunk_content
from ezlocalai.AudioCache import AudioCache
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pydub import AudioSegment

try:
    import deepspeed

    deepspeed_available = True
except:
    deepspeed_available = False


def download_xtts():
    files_to_download = {
        "LICENSE.txt": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true",
        "README.md": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true",
        "config.json": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true",
        "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true",
        "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/dvae.pth?download=true",
        "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/mel_stats.pth?download=true",
        "speakers_xtts.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/speakers_xtts.pth?download=true",
        "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true",
    }
    os.makedirs(os.path.join(os.getcwd(), "xttsv2_2.0.2"), exist_ok=True)
    for filename, url in files_to_download.items():
        destination = os.path.join(os.getcwd(), "xttsv2_2.0.2", filename)
        if not os.path.exists(destination):
            logging.info(f"[CTTS] Downloading {filename} for XTTSv2...")
            response = requests.get(url, stream=True)
            block_size = 1024  # 1 Kibibyte
            with open(destination, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)


class CTTS:
    def __init__(self, cache_config=None):
        global deepspeed_available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_dir = os.path.join(os.getcwd(), "xttsv2_2.0.2")
        download_xtts()
        config = XttsConfig()
        config.load_json(str(os.path.join(checkpoint_dir, "config.json")))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_dir),
            vocab_path=str(os.path.join(checkpoint_dir, "vocab.json")),
            use_deepspeed=deepspeed_available and self.device == "cuda",
        )
        self.model.to(self.device)

        # Fix for CPU mode: ensure the model is properly set to eval mode
        # This helps with the gpt_inference initialization on CPU
        self.model.eval()
        self.output_folder = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_folder, exist_ok=True)
        self.voices_path = os.path.join(os.getcwd(), "voices")
        wav_files = []
        for file in os.listdir(self.voices_path):
            if file.endswith(".wav"):
                wav_files.append(file.replace(".wav", ""))
        self.voices = wav_files

        # Initialize audio cache
        self.cache = AudioCache(cache_config)

        # Cache statistics tracking
        self.use_cache = cache_config.get("enabled", True) if cache_config else True
        logging.info(
            f"[CTTS] Audio caching {'enabled' if self.use_cache else 'disabled'}"
        )

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

        # Get conditioning latents
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[f"{audio_path}"],
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

        # Split text into chunks
        text_chunks = chunk_content(text)

        # Process chunks with cache optimization
        all_chunks_audio = []

        for chunk in text_chunks:
            if use_cache and len(chunk.strip()) > 0:
                # For caching, generate multiple samples and select best
                chunk_audio = await self._generate_with_cache_optimization(
                    chunk, language, gpt_cond_latent, speaker_embedding, voice_name
                )
            else:
                # Generate single sample without cache
                chunk_audio = self._generate_single_sample(
                    chunk, language, gpt_cond_latent, speaker_embedding
                )
            all_chunks_audio.append(chunk_audio)

        # Combine all chunks
        combined_audio = AudioSegment.empty()
        for audio_data in all_chunks_audio:
            # Convert bytes to AudioSegment
            temp_file = os.path.join(self.output_folder, f"temp_{uuid.uuid4().hex}.wav")
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            audio = AudioSegment.from_file(temp_file)
            combined_audio += audio
            combined_audio += AudioSegment.silent(duration=1000)
            os.remove(temp_file)

        # Export final audio
        if not output_file_name:
            output_file_name = f"{uuid.uuid4().hex}.wav"
        output_file = os.path.join(self.output_folder, output_file_name)
        combined_audio.export(output_file, format="wav")

        # Read final audio data
        with open(output_file, "rb") as file:
            final_audio_data = file.read()

        # Store in cache if enabled (for the complete text)
        if use_cache and len(text_chunks) == 1:  # Only cache single-chunk audio
            cache_key = self.cache.generate_cache_key(text, voice_name, language)
            metadata = {
                "text": text,
                "voice": voice_name,
                "language": language,
                "duration_ms": len(combined_audio),
                "generation_method": "single_generation",
            }
            self.cache.store_cached_audio(cache_key, final_audio_data, metadata)

        # Return result
        if local_uri:
            return f"{local_uri}/outputs/{output_file_name}"
        else:
            os.remove(output_file)
            return base64.b64encode(final_audio_data).decode("utf-8")

    async def _generate_with_cache_optimization(
        self, text, language, gpt_cond_latent, speaker_embedding, voice_name
    ):
        """Generate audio with multi-sample optimization for caching."""
        # Check if this specific chunk is already cached
        cache_key = self.cache.generate_cache_key(text, voice_name, language)
        cached_audio = self.cache.get_cached_audio(cache_key)

        if cached_audio:
            return cached_audio

        # Generate multiple samples using the cache's selection logic
        def generation_func(text, language, **kwargs):
            """Wrapper function for audio generation."""
            if self.device == "cpu":
                output = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    enable_text_splitting=False,
                    temperature=0.7,
                    repetition_penalty=10.0,
                )
            else:
                output = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    enable_text_splitting=True,
                    temperature=0.7,
                    repetition_penalty=10.0,
                )

            # Save to temporary file and return bytes
            temp_file = os.path.join(self.output_folder, f"temp_{uuid.uuid4().hex}.wav")
            torchaudio.save(temp_file, torch.tensor(output["wav"]).unsqueeze(0), 24000)
            with open(temp_file, "rb") as f:
                audio_data = f.read()
            os.remove(temp_file)
            return audio_data

        # Generate and select best audio
        generation_params = {
            "text": text,
            "language": language,
        }

        best_audio, metadata = self.cache.select_best_audio(
            [], generation_func, generation_params
        )

        # Add voice info to metadata and store in cache
        metadata["text"] = text
        metadata["voice"] = voice_name
        metadata["language"] = language

        self.cache.store_cached_audio(cache_key, best_audio, metadata)

        return best_audio

    def _generate_single_sample(
        self, text, language, gpt_cond_latent, speaker_embedding
    ):
        """Generate a single audio sample without caching."""
        if self.device == "cpu":
            output = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=False,
                temperature=0.7,
                repetition_penalty=10.0,
            )
        else:
            output = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=True,
                temperature=0.7,
                repetition_penalty=10.0,
            )

        # Save to temporary file and return bytes
        temp_file = os.path.join(self.output_folder, f"temp_{uuid.uuid4().hex}.wav")
        torchaudio.save(temp_file, torch.tensor(output["wav"]).unsqueeze(0), 24000)
        with open(temp_file, "rb") as f:
            audio_data = f.read()
        os.remove(temp_file)
        return audio_data

    def get_cache_stats(self):
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self, voice=None):
        """Clear the audio cache."""
        self.cache.clear_cache(voice=voice)
        logging.info(
            f"[CTTS] Cache cleared for {'voice: ' + voice if voice else 'all voices'}"
        )


if __name__ == "__main__":
    download_xtts()
