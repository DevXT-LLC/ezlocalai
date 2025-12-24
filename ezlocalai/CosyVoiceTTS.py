"""
CosyVoice TTS wrapper for ezlocalai.

Supports Fun-CosyVoice3-0.5B-2512 and Fun-CosyVoice3-0.5B-2512_RL models
with zero-shot voice cloning and streaming support.

Features:
- Multi-language support (9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian)
- Zero-shot voice cloning from audio samples
- Bi-directional streaming (text-in streaming and audio-out streaming)
- 150ms latency for streaming mode

Installation:
1. Clone CosyVoice repo into ezlocalai directory:
   cd /path/to/ezlocalai
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   
2. Install dependencies:
   pip install -r CosyVoice/requirements.txt
   
3. Download the model:
   from huggingface_hub import snapshot_download
   snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
   
4. Set TTS_PROVIDER=cosyvoice in your environment
"""

import os
import re
import sys
import uuid
import base64
import torch
import logging
import gc
import struct
from typing import Optional, Generator

# Add CosyVoice to path - it needs to be cloned into the ezlocalai directory
EZLOCALAI_ROOT = os.path.dirname(os.path.dirname(__file__))
COSYVOICE_PATH = os.path.join(EZLOCALAI_ROOT, "CosyVoice")

def _setup_cosyvoice_path():
    """Setup the Python path for CosyVoice imports."""
    if not os.path.exists(COSYVOICE_PATH):
        raise RuntimeError(
            f"CosyVoice not found at {COSYVOICE_PATH}. "
            "Please clone it with: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git"
        )
    
    # Add Matcha-TTS first (dependency)
    matcha_path = os.path.join(COSYVOICE_PATH, "third_party", "Matcha-TTS")
    if os.path.exists(matcha_path) and matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)
    
    # Add CosyVoice root
    if COSYVOICE_PATH not in sys.path:
        sys.path.insert(0, COSYVOICE_PATH)

# Maximum characters per chunk for TTS generation
MAX_CHUNK_CHARS = 500  # CosyVoice handles longer text than Chatterbox


def normalize_text_for_tts(text: str) -> str:
    """
    Normalize text for better TTS pronunciation.
    Converts times, dates, numbers, and common abbreviations to spoken form.
    """
    # Convert times like "12:45 PM" to "12 45 PM" for better pronunciation
    text = re.sub(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)', r'\1 \2 \3', text)
    # Convert 24-hour times like "14:30" to "14 30"
    text = re.sub(r'(\d{1,2}):(\d{2})(?!\d)', r'\1 \2', text)
    
    # Convert dates like "12/23/2025" to "12 23 2025"
    text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\1 \2 \3', text)
    text = re.sub(r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\1 \2 \3', text)
    
    # Expand common abbreviations
    text = re.sub(r'\bDr\.\s', 'Doctor ', text)
    text = re.sub(r'\bMr\.\s', 'Mister ', text)
    text = re.sub(r'\bMrs\.\s', 'Missus ', text)
    text = re.sub(r'\bMs\.\s', 'Miss ', text)
    text = re.sub(r'\betc\.\s', 'et cetera ', text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.\s', 'for example ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi\.e\.\s', 'that is ', text, flags=re.IGNORECASE)
    
    # Remove markdown-style formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    
    return text


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Split text into sentence-based chunks for TTS generation.
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

        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Handle case where a single sentence is too long
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
                if len(sub_chunk) > max_chars:
                    # Force split on words
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


class CosyVoiceTTS:
    """
    CosyVoice TTS wrapper with voice cloning support.
    
    Supports Fun-CosyVoice3-0.5B-2512 for high-quality multi-language TTS.
    """

    def __init__(
        self,
        model_name: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        cache_config: Optional[dict] = None,
    ):
        """
        Initialize CosyVoice TTS.
        
        Args:
            model_name: HuggingFace model name or local path
            cache_config: Audio cache configuration
        """
        self.model_name = model_name
        self.model = None
        self.sample_rate = 24000  # Default, will be updated when model loads
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.output_folder = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_folder, exist_ok=True)
        self.voices_path = os.path.join(os.getcwd(), "voices")
        os.makedirs(self.voices_path, exist_ok=True)
        
        # Find available voice files
        wav_files = []
        for file in os.listdir(self.voices_path):
            if file.endswith(".wav"):
                wav_files.append(file.replace(".wav", ""))
        self.voices = wav_files
        logging.debug(f"[CosyVoiceTTS] Found {len(self.voices)} voice(s): {self.voices}")

        # Initialize audio cache if provided
        if cache_config:
            from ezlocalai.AudioCache import AudioCache
            self.cache = AudioCache(cache_config)
            self.use_cache = cache_config.get("enabled", True)
        else:
            self.cache = None
            self.use_cache = False
            
        logging.debug(f"[CosyVoiceTTS] Initialized (model will be loaded on first use)")

    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if self.model is not None:
            return
            
        logging.info(f"[CosyVoiceTTS] Loading model: {self.model_name}")
        
        # CosyVoice requires CUDA - CPU mode produces garbage audio
        # If VRAM is insufficient, use VOICE_SERVER to offload to dedicated TTS server
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[CosyVoiceTTS] CUDA is required. CosyVoice does not work correctly on CPU. "
                "Use VOICE_SERVER environment variable to offload TTS to a dedicated server."
            )
        
        self.device = "cuda"
        
        # Check available VRAM
        available_vram = get_available_vram_mb()
        min_vram_mb = 4000  # CosyVoice needs about 4GB VRAM
        
        if available_vram < min_vram_mb:
            logging.warning(
                f"[CosyVoiceTTS] Only {available_vram:.0f}MB VRAM available, need ~{min_vram_mb}MB. "
                "TTS may fail with OOM. Consider using VOICE_SERVER to offload to dedicated TTS server."
            )
        
        # Setup path for CosyVoice imports
        _setup_cosyvoice_path()
        
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            
            # Determine model directory
            # Priority: 1) Direct path if exists, 2) pretrained_models/<basename>, 3) Download from HuggingFace
            model_dir = self.model_name
            
            if not os.path.exists(model_dir):
                # Try local pretrained_models path
                basename = os.path.basename(model_dir).replace("FunAudioLLM/", "")
                # Map HuggingFace name to local folder name
                if basename == "Fun-CosyVoice3-0.5B-2512":
                    local_name = "Fun-CosyVoice3-0.5B"
                elif basename == "Fun-CosyVoice3-0.5B-2512_RL":
                    local_name = "Fun-CosyVoice3-0.5B-RL"
                else:
                    local_name = basename
                    
                local_path = os.path.join(EZLOCALAI_ROOT, "pretrained_models", local_name)
                
                if os.path.exists(local_path):
                    model_dir = local_path
                    logging.info(f"[CosyVoiceTTS] Using local model at {model_dir}")
                else:
                    # Download from HuggingFace
                    logging.info(f"[CosyVoiceTTS] Downloading model from HuggingFace: {self.model_name}")
                    from huggingface_hub import snapshot_download
                    
                    # Create pretrained_models directory
                    pretrained_dir = os.path.join(EZLOCALAI_ROOT, "pretrained_models")
                    os.makedirs(pretrained_dir, exist_ok=True)
                    
                    model_dir = snapshot_download(
                        self.model_name, 
                        local_dir=local_path
                    )
            
            # Load with fp16 for GPU efficiency
            self.model = AutoModel(model_dir=model_dir, fp16=True)
            self.sample_rate = self.model.sample_rate
            
            logging.info(
                f"[CosyVoiceTTS] Model loaded successfully on {self.device}, "
                f"sample_rate={self.sample_rate}, fp16=True"
            )
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"[CosyVoiceTTS] Out of VRAM loading TTS model. "
                    f"CosyVoice requires ~4GB VRAM. Options:\n"
                    f"1. Use VOICE_SERVER to offload TTS to dedicated server\n"
                    f"2. Use a smaller LLM model\n"
                    f"3. Reduce LLM context size\n"
                    f"Original error: {e}"
                ) from e
            else:
                raise

    def _get_voice_path(self, voice: str) -> Optional[str]:
        """Get the path to a voice file."""
        if not voice.endswith(".wav"):
            voice = f"{voice}.wav"
            
        audio_path = os.path.join(self.voices_path, voice)
        if os.path.exists(audio_path):
            return audio_path
            
        # Try default
        default_path = os.path.join(self.voices_path, "default.wav")
        if os.path.exists(default_path):
            return default_path
            
        logging.warning(f"[CosyVoiceTTS] No voice file found for '{voice}'")
        return None

    async def generate(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        local_uri: Optional[str] = None,
        output_file_name: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> str:
        """
        Generate TTS audio from text.
        
        Args:
            text: Text to synthesize
            voice: Voice name (wav file in voices folder)
            language: Language code (en, zh, ja, ko, de, es, fr, it, ru)
            local_uri: Base URI for returning URLs instead of base64
            output_file_name: Custom output filename
            use_cache: Override cache usage
            
        Returns:
            Base64 encoded audio or URL to audio file
        """
        self._ensure_model_loaded()
        
        if use_cache is None:
            use_cache = self.use_cache
            
        # Clean text - CosyVoice handles multiple languages better than Chatterbox
        text = re.sub(r"([!?.])\1+", r"\1", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace("#", "").strip()
        
        # Normalize text for better TTS pronunciation (times, dates, abbreviations)
        text = normalize_text_for_tts(text)
        
        if not text:
            logging.warning("[CosyVoiceTTS] Empty text provided")
            return ""
            
        # Check cache
        voice_name = voice.replace(".wav", "") if voice.endswith(".wav") else voice
        if use_cache and self.cache:
            cache_key = self.cache.generate_cache_key(text, voice_name, language)
            cached_audio = self.cache.get_cached_audio(cache_key)
            if cached_audio:
                if local_uri:
                    cached_path = os.path.join(
                        self.output_folder, "cache", "audio", f"{cache_key}.wav"
                    )
                    if os.path.exists(cached_path):
                        return f"{local_uri}/outputs/cache/audio/{cache_key}.wav"
                return base64.b64encode(cached_audio).decode("utf-8")
        
        # Get voice file
        audio_path = self._get_voice_path(voice)
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        if len(chunks) > 1:
            logging.debug(f"[CosyVoiceTTS] Split text into {len(chunks)} chunks")
            
        # Generate audio for each chunk
        all_audio = []
        
        # CosyVoice3 uses inference_instruct2 for voice cloning with instructions
        # The instruction tells it how to speak (language, style, etc.)
        # Format: inference_instruct2(text, instruction + '<|endofprompt|>', reference_audio)
        instruct_prefix = "You are a helpful assistant. Speak in English with a clear and natural tone.<|endofprompt|>"
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logging.debug(f"[CosyVoiceTTS] Generating chunk {i+1}/{len(chunks)}")
                
            try:
                if audio_path:
                    # Use inference_instruct2 for voice cloning with style control
                    # This method takes (text_to_speak, instruction, reference_audio)
                    for result in self.model.inference_instruct2(
                        chunk, instruct_prefix, audio_path, stream=False
                    ):
                        audio_tensor = result['tts_speech']
                        all_audio.append(audio_tensor)
                else:
                    # Without reference audio, we can't do TTS with CosyVoice3
                    logging.error(f"[CosyVoiceTTS] No voice file available. CosyVoice3 requires a reference audio for voice cloning.")
                    return ""
                        
            except Exception as e:
                logging.error(f"[CosyVoiceTTS] Error generating chunk {i+1}: {e}")
                continue
                
        if not all_audio:
            logging.warning("[CosyVoiceTTS] No audio generated")
            return ""
            
        # Concatenate all audio
        if len(all_audio) == 1:
            final_audio = all_audio[0]
        else:
            final_audio = torch.cat(all_audio, dim=1)
            
        # Convert to bytes
        audio_bytes = self._tensor_to_wav_bytes(final_audio)
        
        if not audio_bytes:
            return ""
            
        # Save to file
        if not output_file_name:
            output_file_name = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(self.output_folder, output_file_name)
        
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
            
        # Store in cache
        if use_cache and self.cache:
            cache_key = self.cache.generate_cache_key(text, voice_name, language)
            metadata = {
                "text": text,
                "voice": voice_name,
                "language": language,
                "generation_method": "cosyvoice",
                "chunks": len(chunks),
            }
            self.cache.store_cached_audio(cache_key, audio_bytes, metadata)
            
        # Return result
        if local_uri:
            return f"{local_uri}/outputs/{output_file_name}"
        else:
            os.remove(output_path)
            return base64.b64encode(audio_bytes).decode("utf-8")

    def _tensor_to_wav_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert audio tensor to WAV bytes."""
        try:
            import soundfile as sf
            import io
            
            # Ensure tensor is on CPU and convert to numpy
            if tensor.dim() == 1:
                audio_np = tensor.cpu().numpy()
            else:
                audio_np = tensor.squeeze().cpu().numpy()
                
            # Write to BytesIO
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, self.sample_rate, format='WAV')
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logging.error(f"[CosyVoiceTTS] Error converting tensor to bytes: {e}")
            return b""

    def _extract_pcm_from_wav(self, wav_bytes: bytes) -> bytes:
        """Extract raw PCM data from WAV bytes."""
        # Search for "data" chunk marker
        data_pos = wav_bytes.find(b"data")
        if data_pos >= 0 and data_pos + 8 <= len(wav_bytes):
            chunk_size = struct.unpack("<I", wav_bytes[data_pos + 4:data_pos + 8])[0]
            pcm_start = data_pos + 8
            if pcm_start + chunk_size <= len(wav_bytes):
                return wav_bytes[pcm_start:pcm_start + chunk_size]
            return wav_bytes[pcm_start:]
        # Fallback: assume 44-byte header
        return wav_bytes[44:]

    async def generate_stream(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
    ) -> Generator[bytes, None, None]:
        """
        Generate TTS audio as a stream of PCM chunks.
        
        Yields raw PCM audio bytes (24kHz, 16-bit, mono) for real-time playback.
        
        First yield includes audio format header:
        - 4 bytes: sample rate (uint32, little-endian)
        - 2 bytes: bits per sample (uint16, little-endian)
        - 2 bytes: channels (uint16, little-endian)
        
        Subsequent yields are: size (uint32) + PCM data
        Final yield is: size = 0 (end marker)
        """
        self._ensure_model_loaded()
        
        # Clean text
        text = re.sub(r"([!?.])\1+", r"\1", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace("#", "").strip()
        
        # Normalize text for better TTS pronunciation (times, dates, abbreviations)
        text = normalize_text_for_tts(text)
        
        if not text:
            yield struct.pack("<IHH", self.sample_rate, 16, 1)
            yield struct.pack("<I", 0)
            return
            
        # Get voice file
        if not voice.endswith(".wav"):
            voice = f"{voice}.wav"
        audio_path = self._get_voice_path(voice)
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        logging.info(f"[CosyVoiceTTS] Streaming TTS: {len(chunks)} chunks for {len(text)} chars")
        
        # Yield header
        yield struct.pack("<IHH", self.sample_rate, 16, 1)
        
        # CosyVoice3 uses inference_instruct2 for voice cloning with instructions
        instruct_prefix = "You are a helpful assistant. Speak in English with a clear and natural tone.<|endofprompt|>"
        
        for i, chunk in enumerate(chunks):
            logging.debug(f"[CosyVoiceTTS] Streaming chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            try:
                if audio_path:
                    # Use streaming inference_instruct2 for voice cloning with style control
                    for result in self.model.inference_instruct2(
                        chunk, instruct_prefix, audio_path, stream=True
                    ):
                        audio_tensor = result['tts_speech']
                        wav_bytes = self._tensor_to_wav_bytes(audio_tensor)
                        if wav_bytes:
                            pcm_data = self._extract_pcm_from_wav(wav_bytes)
                            if pcm_data:
                                yield struct.pack("<I", len(pcm_data)) + pcm_data
                                logging.debug(f"[CosyVoiceTTS] Yielded {len(pcm_data)} bytes")
                else:
                    # CosyVoice3 requires a reference audio for voice cloning
                    logging.error(f"[CosyVoiceTTS] No voice file available. CosyVoice3 requires a reference audio for voice cloning.")
                    break
                                
            except Exception as e:
                logging.error(f"[CosyVoiceTTS] Error generating chunk {i+1}: {e}")
                continue
                
        # Yield end marker
        yield struct.pack("<I", 0)
        logging.info("[CosyVoiceTTS] Streaming TTS complete")

    def get_cache_stats(self):
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {}

    def clear_cache(self, voice: Optional[str] = None):
        """Clear the audio cache."""
        if self.cache:
            self.cache.clear_cache(voice=voice)
            logging.debug(
                f"[CosyVoiceTTS] Cache cleared for {'voice: ' + voice if voice else 'all voices'}"
            )
