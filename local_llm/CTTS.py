import os
import re
import uuid
import numpy as np
import base64
import io
import wave
import torch
import torchaudio
import requests
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from fastapi.responses import JSONResponse, StreamingResponse

deepspeed_available = False
try:
    import deepspeed

    deepspeed_available = True
except ImportError:
    pass


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
    os.makedirs(os.path.join(os.getcwd(), "models", "xttsv2_2.0.2"), exist_ok=True)
    for filename, url in files_to_download.items():
        destination = os.path.join(os.getcwd(), "models", "xttsv2_2.0.2", filename)
        if not os.path.exists(destination):
            response = requests.get(url, stream=True)
            block_size = 1024  # 1 Kibibyte
            with open(destination, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)


class CTTS:
    def __init__(self):
        global deepspeed_available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.empty_cache()
        config = XttsConfig()
        checkpoint_dir = os.path.join(os.getcwd(), "models", "xttsv2_2.0.2")
        # Check if the model is downloaded
        if not os.path.exists(checkpoint_dir):
            print("Downloading XTTSv2 model...")
            download_xtts()
        config.load_json(str(os.path.join(checkpoint_dir, "config.json")))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_dir),
            vocab_path=str(os.path.join(checkpoint_dir, "vocab.json")),
            use_deepspeed=deepspeed_available,
        )
        self.model.to(self.device)

    async def get_voices(self):
        wav_files = []
        for file in os.listdir(os.path.join(os.getcwd(), "voices")):
            if file.endswith(".wav"):
                wav_files.append(file.replace(".wav", ""))
        return {"voices": wav_files}

    async def generate_audio(
        self,
        text,
        voice,
        output_file,
        language="en",
        streaming=False,
    ):
        cleaned_string = re.sub(r"([!?.])\1+", r"\1", text)
        cleaned_string = re.sub(
            r'[^a-zA-Z0-9\s\.,;:!?\-\'"\u0400-\u04FFÀ-ÿ\u0150\u0151\u0170\u0171]\$',
            "",
            cleaned_string,
        )
        if not voice.endswith(".wav"):
            voice = f"{voice}.wav"
        cleaned_string = re.sub(r"\n+", " ", cleaned_string)
        cleaned_string = cleaned_string.replace("#", "")
        text = cleaned_string
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[f"{os.getcwd()}/voices/{voice}"],
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )
        common_args = {
            "text": text,
            "language": language,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "temperature": 0.7,
            "length_penalty": float(self.model.config.length_penalty),
            "repetition_penalty": 10.0,
            "top_k": int(self.model.config.top_k),
            "top_p": float(self.model.config.top_p),
            "enable_text_splitting": True,
        }
        inference_func = (
            self.model.inference_stream if streaming else self.model.inference
        )
        if streaming:
            common_args["stream_chunk_size"] = 20
        output = inference_func(**common_args)
        if streaming:
            file_chunks = []
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as vfout:
                vfout.setnchannels(1)
                vfout.setsampwidth(2)
                vfout.setframerate(24000)
                vfout.writeframes(b"")
            wav_buf.seek(0)
            yield wav_buf.read()
            for i, chunk in enumerate(output):
                file_chunks.append(chunk)
                if isinstance(chunk, list):
                    chunk = torch.cat(chunk, dim=0)
                chunk = chunk.clone().detach().cpu().numpy()
                chunk = chunk[None, : int(chunk.shape[0])]
                chunk = np.clip(chunk, -1, 1)
                chunk = (chunk * 32767).astype(np.int16)
                yield chunk.tobytes()
        else:
            torchaudio.save(
                output_file, torch.tensor(output["wav"]).unsqueeze(0), 24000
            )

    async def generate(
        self,
        text: str = "",
        voice: str = "",
        language: str = "en",
        streaming: bool = False,
    ):
        output_file_name = f"{uuid.uuid4().hex}.wav"
        try:
            output_file_path = os.path.join(
                os.getcwd(), "outputs", f"{output_file_name}.wav"
            )
            response = await self.generate_audio(
                text=text,
                voice=voice,
                output_file=output_file_path,
                language=language,
                streaming=streaming,
            )
            if streaming:
                return StreamingResponse(response, media_type="audio/wav")
            else:
                with open(output_file_path, "rb") as file:
                    file_content = file.read()
                os.remove(output_file_path)
                return JSONResponse(
                    content={
                        "status": "success",
                        "data": base64.b64encode(file_content).decode("utf-8"),
                    },
                    status_code=200,
                )
        except Exception as e:
            return JSONResponse(
                content={"status": "generate-failure", "error": "An error occurred"},
                status_code=500,
            )


if __name__ == "__main__":
    download_xtts()
