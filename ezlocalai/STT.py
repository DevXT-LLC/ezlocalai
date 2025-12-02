import os
import base64
import uuid
import logging
import webrtcvad
import pyaudio
import wave
import threading
import numpy as np
import gc
import torch
from io import BytesIO
from faster_whisper import WhisperModel

# Suppress ALSA warnings in Docker containers without sound cards
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _check_cudnn_available():
    """Check if cuDNN is available for CTranslate2/faster-whisper."""
    if not torch.cuda.is_available():
        return False
    try:
        # Try to check if cuDNN is available via torch
        return torch.backends.cudnn.is_available()
    except:
        return False


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


class STT:
    def __init__(self, model="base", wake_functions={}):
        # Check if there's enough VRAM for STT (need at least 1GB free)
        available_vram = get_available_vram_mb()
        min_vram_mb = 1000  # 1GB minimum for Whisper

        # Check both VRAM availability and cuDNN availability
        cudnn_available = _check_cudnn_available()

        if (
            torch.cuda.is_available()
            and available_vram >= min_vram_mb
            and cudnn_available
        ):
            device = "cuda"
        else:
            device = "cpu"
            if torch.cuda.is_available():
                if not cudnn_available:
                    logging.warning(
                        "[STT] cuDNN not available, using CPU for Whisper (faster-whisper requires cuDNN for GPU)"
                    )
                elif available_vram < min_vram_mb:
                    logging.debug(
                        f"[STT] Only {available_vram:.0f}MB VRAM available, using CPU (need {min_vram_mb}MB)"
                    )

        logging.info(f"[STT] Loading Whisper {model} on {device}")
        self.device = device
        self.model_name = model

        try:
            self.w = WhisperModel(model, download_root="models", device=device)
        except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
            error_str = str(e).lower()
            # Check for various GPU-related errors including cuDNN issues
            is_gpu_error = any(
                x in error_str
                for x in [
                    "out of memory",
                    "cuda",
                    "cudnn",
                    "libcudnn",
                    "invalid handle",
                ]
            )
            if is_gpu_error and device == "cuda":
                logging.warning(f"[STT] GPU loading failed, falling back to CPU: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                self.w = WhisperModel(model, download_root="models", device="cpu")
            else:
                raise

        self.audio = pyaudio.PyAudio()
        self.wake_functions = wake_functions

    async def transcribe_audio(
        self,
        base64_audio,
        audio_format="wav",
        language=None,
        prompt=None,
        temperature=0.0,
        translate=False,
        return_segments=False,
        beam_size=5,
        condition_on_previous_text=True,
    ):
        if "/" in audio_format:
            audio_format = audio_format.split("/")[1]
        filename = f"{uuid.uuid4().hex}.wav"
        file_path = os.path.join(os.getcwd(), "outputs", filename)
        audio_data = base64.b64decode(base64_audio)
        with open(file_path, "wb") as audio_file:
            audio_file.write(audio_data)
        if not os.path.exists(file_path):
            raise RuntimeError(f"Failed to load audio.")

        transcribe_info = None
        try:
            segments, transcribe_info = self.w.transcribe(
                file_path,
                task="transcribe" if not translate else "translate",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=prompt,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                condition_on_previous_text=condition_on_previous_text,
            )
            segments = list(segments)
        except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
            error_str = str(e).lower()
            # Check for various GPU-related errors including cuDNN issues
            is_gpu_error = any(
                x in error_str
                for x in [
                    "out of memory",
                    "cuda",
                    "cudnn",
                    "libcudnn",
                    "invalid handle",
                ]
            )
            if is_gpu_error and self.device == "cuda":
                logging.warning(
                    f"[STT] GPU error during transcription, reloading on CPU: {e}"
                )
                # Free GPU memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reload model on CPU
                del self.w
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.device = "cpu"
                self.w = WhisperModel(
                    self.model_name, download_root="models", device="cpu"
                )
                logging.debug("[STT] Model reloaded on CPU, retrying transcription...")

                # Retry on CPU
                segments, transcribe_info = self.w.transcribe(
                    file_path,
                    task="transcribe" if not translate else "translate",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    initial_prompt=prompt,
                    language=language,
                    temperature=temperature,
                    beam_size=beam_size,
                    condition_on_previous_text=condition_on_previous_text,
                )
                segments = list(segments)
            else:
                os.remove(file_path)
                raise

        # Build full text and segment data
        user_input = ""
        segment_data = []
        for i, segment in enumerate(segments):
            user_input += segment.text
            segment_data.append(
                {
                    "id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )
        logging.debug(f"[STT] Transcribed User Input: {user_input}")
        os.remove(file_path)

        if return_segments:
            detected_language = None
            if transcribe_info and hasattr(transcribe_info, "language"):
                detected_language = transcribe_info.language
            elif language:
                detected_language = language
            return {
                "text": user_input.strip(),
                "segments": segment_data,
                "language": detected_language,
            }
        return user_input

    def listen(self):
        print("Listening for wake word...")
        vad = webrtcvad.Vad(1)
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=320,
        )
        frames = []
        silence_frames = 0
        while True:
            data = stream.read(320)
            frames.append(data)
            is_speech = vad.is_speech(data, 16000)
            if not is_speech:
                silence_frames += 1
                if silence_frames > 1 * 16000 / 320:
                    audio_data = b"".join(frames)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_np**2))
                    if rms > 500:
                        buffer = BytesIO()
                        with wave.open(buffer, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(16000)
                            wf.writeframes(b"".join(frames))
                        wav_buffer = buffer.getvalue()
                        base64_audio = base64.b64encode(wav_buffer).decode()
                        thread = threading.Thread(
                            target=self.transcribe_audio,
                            args=(base64_audio),
                        )
                        thread.start()
                    frames = []  # Clear frames after processing
                    silence_frames = 0
            else:
                silence_frames = 0
