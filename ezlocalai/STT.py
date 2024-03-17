import os
import base64
import uuid
import logging
import webrtcvad
import pyaudio
import wave
import threading
import numpy as np
from io import BytesIO
from faster_whisper import WhisperModel


class STT:
    def __init__(self, model="base", wake_functions={}):
        self.w = WhisperModel(model, download_root="models", device="cpu")
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
        segments, _ = self.w.transcribe(
            file_path,
            task="transcribe" if not translate else "translate",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=prompt,
            language=language,
            temperature=temperature,
        )
        segments = list(segments)
        user_input = ""
        for segment in segments:
            user_input += segment.text
        logging.info(f"[STT] Transcribed User Input: {user_input}")
        os.remove(file_path)
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
