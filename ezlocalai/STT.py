import os
import base64
import io
import requests
import uuid
import logging
import webrtcvad
import pyaudio
import wave
import threading
import numpy as np
from io import BytesIO
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
    def __init__(self, model="base.en", wake_functions={}):
        model_path = download_whisper_model(model=model)
        self.w = Whisper(model_path=model_path, verbose=False)
        self.audio = pyaudio.PyAudio()
        self.wake_functions = wake_functions

    async def transcribe_audio(self, base64_audio, audio_format="wav"):
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
        if self.wake_functions != {}:
            for wake_word, wake_function in self.wake_functions.items():
                if wake_word.lower() in user_input.lower():
                    print("Wake word detected! Executing wake function...")
                    if wake_function:
                        try:
                            return wake_function(user_input)
                        except Exception as e:
                            logging.error(f"Failed to execute wake function: {e}")
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


if __name__ == "__main__":
    download_whisper_model()
