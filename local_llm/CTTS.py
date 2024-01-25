import time
import os
import re
import uuid
import numpy as np
import base64
import io
import wave
from pathlib import Path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from fastapi.responses import JSONResponse, StreamingResponse

deepspeed_available = False
try:
    import deepspeed

    deepspeed_available = True
except ImportError:
    pass
if deepspeed_available:
    print("DeepSpeed enabled.")
else:
    print("DeepSpeed disabled.")


class CTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = XttsConfig()
        modeldownload_settings = {
            "base_path": "models",
            "model_path": "xttsv2_2.0.2",
            "files_to_download": {
                "LICENSE.txt": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true",
                "README.md": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true",
                "config.json": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true",
                "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true",
                "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/dvae.pth?download=true",
                "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/mel_stats.pth?download=true",
                "speakers_xtts.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/speakers_xtts.pth?download=true",
                "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true",
            },
        }
        modeldownload_base_path = Path(modeldownload_settings.get("base_path", ""))
        modeldownload_model_path = Path(modeldownload_settings.get("model_path", ""))
        self.params = {
            "activate": True,
            "autoplay": True,
            "deepspeed_activate": True,
            "low_vram": False if not torch.cuda.is_available() else False,
            "local_temperature": "0.7",
            "local_repetition_penalty": "10.0",
            "output_folder_wav_standalone": "outputs/",
            "remove_trailing_dots": False,
            "show_text": True,
            "voice": "female_01.wav",
        }
        self.home_dir = Path(__file__).parent.resolve()
        # check to see if a custom path has been set in modeldownload.json and use that path to load the model if so
        if str(modeldownload_base_path) == "models":
            config_path = (
                self.home_dir / "models" / modeldownload_model_path / "config.json"
            )
            vocab_path_dir = (
                self.home_dir / "models" / modeldownload_model_path / "vocab.json"
            )
            checkpoint_dir = self.home_dir / "models" / modeldownload_model_path
        else:
            print(
                f"[TTS Model] \033[94mInfo\033[0m Loading your custom model set in \033[93mmodeldownload.json\033[0m:",
                modeldownload_base_path / modeldownload_model_path,
            )
            config_path = (
                modeldownload_base_path / modeldownload_model_path / "config.json"
            )
            vocab_path_dir = (
                modeldownload_base_path / modeldownload_model_path / "vocab.json"
            )
            checkpoint_dir = modeldownload_base_path / modeldownload_model_path
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_dir),
            vocab_path=str(vocab_path_dir),
            use_deepspeed=self.params["deepspeed_activate"],
        )
        model.to(self.device)
        self.model = model
        # Set the output path for wav files
        output_directory = self.home_dir / self.params["output_folder_wav_standalone"]
        output_directory.mkdir(parents=True, exist_ok=True)

    async def switch_device(self):
        # Check if CUDA is available before performing GPU-related operations
        if torch.cuda.is_available():
            if self.device == "cuda":
                self.device = "cpu"
                self.model.to(self.device)
                torch.cuda.empty_cache()
            else:
                self.device == "cpu"
                self.device = "cuda"
                self.model.to(self.device)

    async def generate_audio(
        self,
        text,
        voice,
        output_file,
        language="en",
        temperature=0.7,
        repetition_penalty=10.0,
        streaming=False,
    ):
        if self.params["low_vram"] and self.device == "cpu":
            await self.switch_device()
        print(f"[TTS] {text}")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[f"{self.home_dir}/voices/{voice}"],
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )
        common_args = {
            "text": text,
            "language": language,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "temperature": float(temperature),
            "length_penalty": float(self.model.config.length_penalty),
            "repetition_penalty": float(repetition_penalty),
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

    async def get_voices(self):
        directory = self.home_dir / "voices"
        wav_files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")
        ]
        return {"voices": wav_files}

    async def generate(
        self,
        text: str = "",
        voice: str = "",
        language: str = "en",
        streaming: bool = False,
    ):
        output_file_name = f"{uuid.uuid4().hex}.wav"
        try:
            output_file_path = self.home_dir / "outputs" / f"{output_file_name}.wav"
            cleaned_string = re.sub(r"([!?.])\1+", r"\1", text)
            # Further clean to remove any other unwanted characters
            cleaned_string = re.sub(
                r'[^a-zA-Z0-9\s\.,;:!?\-\'"\u0400-\u04FFÀ-ÿ\u0150\u0151\u0170\u0171]\$',
                "",
                cleaned_string,
            )
            # Remove all newline characters (single or multiple)
            cleaned_string = re.sub(r"\n+", " ", cleaned_string)
            cleaned_string = cleaned_string.replace("#", "")
            response = await self.generate_audio(
                text=cleaned_string,
                voice=voice,
                language=language,
                temperature=self.params["local_temperature"],
                repetition_penalty=self.params["local_repetition_penalty"],
                output_file_path=output_file_path,
                streaming=streaming,
            )
            if streaming:
                return StreamingResponse(response, media_type="audio/wav")
            else:
                # Get content of output file
                with open(output_file_path, "rb") as file:
                    file_content = file.read()
                # Delete the output file
                os.remove(output_file_path)
                # Return the content of the output file in base64 format
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
