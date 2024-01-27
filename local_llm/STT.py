import os
import base64
import io
import requests
import uuid
import logging
from whisper_cpp import Whisper
from pydub import AudioSegment


def download_whisper_model(model="base.en"):
    # https://huggingface.co/ggerganov/whisper.cpp
    if model not in [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
    ]:
        model = "base.en"
    os.makedirs(os.path.join(os.getcwd(), "whispercpp"), exist_ok=True)
    model_path = os.path.join(os.getcwd(), "whispercpp", f"ggml-{model}.bin")
    if not os.path.exists(model_path):
        r = requests.get(
            f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin",
            allow_redirects=True,
        )
        open(model_path, "wb").write(r.content)
    return model_path


class STT:
    def __init__(self, model="base.en"):
        model_path = download_whisper_model(model=model)
        self.w = Whisper(model_path=model_path, verbose=False)

    async def transcribe_audio(self, base64_audio, audio_format="m4a"):
        filename = f"{uuid.uuid4().hex}.wav"
        file_path = os.path.join(os.getcwd(), "outputs", filename)
        audio_data = base64.b64decode(base64_audio)
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_data), format=audio_format.lower()
        )
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_segment.export(file_path, format="wav")
        if not os.path.exists(file_path):
            raise RuntimeError(f"Failed to load audio.")
        self.w.transcribe(file_path)
        user_input = self.w.output(output_txt=False)
        logging.info(f"[STT] Transcribed User Input: {user_input}")
        user_input = user_input.replace("[BLANK_AUDIO]", "")
        os.remove(file_path)
        return user_input


if __name__ == "__main__":
    download_whisper_model()
