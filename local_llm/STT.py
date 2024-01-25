import logging
import os
import base64
import io
import requests
import uuid
from whisper_cpp import Whisper
from pydub import AudioSegment


class STT:
    def __init__(self, model="base.en"):
        # https://huggingface.co/ggerganov/whisper.cpp
        # Models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, large-v1
        if model not in [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
        ]:
            self.model = "base.en"
        else:
            self.model = model
        os.makedirs(os.path.join(os.getcwd(), "models", "whispercpp"), exist_ok=True)
        self.model_path = os.path.join(
            os.getcwd(), "models", "whispercpp", f"ggml-{model}.bin"
        )
        if not os.path.exists(self.model_path):
            r = requests.get(
                f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin",
                allow_redirects=True,
            )
            open(self.model_path, "wb").write(r.content)
        self.w = Whisper(model_path=self.model_path)

    async def transcribe_audio_from_file(self, filename: str = "recording.wav"):
        file_path = os.path.join(os.getcwd(), "WORKSPACE", filename)
        if not os.path.exists(file_path):
            raise RuntimeError(f"Failed to load audio: {filename} does not exist.")
        self.w.transcribe(file_path)
        return self.w.output()

    async def transcribe_audio(self, base64_audio, audio_format="m4a"):
        filename = f"{uuid.uuid4().hex}.wav"
        if audio_format.lower() != "wav":
            audio_data = base64.b64decode(base64_audio)
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data), format=audio_format.lower()
            )
            audio_segment = audio_segment.set_frame_rate(16000)
            file_path = os.path.join(os.getcwd(), "WORKSPACE", filename)
            audio_segment.export(file_path, format="wav")
            with open(file_path, "rb") as f:
                audio = f.read()
            return f"{base64.b64encode(audio).decode('utf-8')}"
        user_input = await self.transcribe_audio_from_file(filename=filename)
        logging.info(f"[Whisper]: Transcribed User Input: {user_input}")
        return user_input
