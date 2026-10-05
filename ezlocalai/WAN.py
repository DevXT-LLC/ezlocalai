"""Wan 2.2 A14B GGUF model assets for stable-diffusion.cpp."""

import os
import logging
import math
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

T2V_REPO = "QuantStack/Wan2.2-T2V-A14B-GGUF"
I2V_REPO = "QuantStack/Wan2.2-I2V-A14B-GGUF"
MODEL_REVISIONS = {
    T2V_REPO: "73eafba53a1a8f29254e4c77f92e74ea27d7cd6f",
    I2V_REPO: "6c6717459277b9cd1f72579d78a0fd62a79e57dc",
    "city96/umt5-xxl-encoder-gguf": "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7",
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged": "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
}
QUANT_TYPES = {
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q4_0",
    "Q4_1",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_0",
    "Q5_1",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
}


def is_wan_model(model):
    return str(model).lower() in {
        T2V_REPO.lower(),
        I2V_REPO.lower(),
        "wan2.2",
        "wan2.2-t2v",
        "wan2.2-i2v",
    }


def download_wan_models(image_to_video=False, quant=None):
    """Fetch one expert pair plus shared encoder/VAE into the normal HF cache."""
    from huggingface_hub import hf_hub_download

    quant = (quant or os.getenv("VIDEO_QUANT_TYPE", "Q4_K_M")).upper()
    if quant not in QUANT_TYPES:
        raise ValueError(f"Unsupported Wan quantization: {quant}")
    mode = "I2V" if image_to_video else "T2V"
    repo = I2V_REPO if image_to_video else T2V_REPO
    files = {
        "high": (repo, f"HighNoise/Wan2.2-{mode}-A14B-HighNoise-{quant}.gguf"),
        "low": (repo, f"LowNoise/Wan2.2-{mode}-A14B-LowNoise-{quant}.gguf"),
        "encoder": ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q8_0.gguf"),
        "vae": (
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "split_files/vae/wan_2.1_vae.safetensors",
        ),
    }
    return {
        name: hf_hub_download(
            model_repo,
            filename=filename,
            revision=MODEL_REVISIONS[model_repo],
            cache_dir=os.getenv("HF_HUB_CACHE", "models"),
        )
        for name, (model_repo, filename) in files.items()
    }


class WanVideo:
    """Wan 2.2 A14B through a short-lived, native CUDA subprocess.

    The high/low-noise expert pair is selected from the request's image input.
    Weights and CUDA allocations belong to sd-cli, so exiting the subprocess
    releases them before the worker restores its embedding and voice pools.
    Wan T2V/I2V produces silent video; music-video muxes ACE audio separately.
    """

    resident_vram_gb = 0.0

    def __init__(
        self, model=T2V_REPO, device="cuda", local_uri=None, generation_hint=None
    ):
        if not is_wan_model(model):
            raise ValueError(f"Unsupported Wan model: {model}")
        self.model = model
        self.device = device
        self.local_uri = local_uri
        self.gpu_residency = "subprocess"
        self.sdcli_bin = os.getenv(
            "SDCPP_BIN", "/opt/stable-diffusion.cpp/build/bin/sd-cli"
        )
        if not os.path.isfile(self.sdcli_bin):
            raise RuntimeError(
                f"Wan requires the stable-diffusion.cpp sd-cli binary: {self.sdcli_bin}"
            )

    def _command(
        self,
        assets,
        output,
        *,
        prompt,
        negative_prompt,
        width,
        height,
        frames,
        fps,
        steps,
        guidance_scale,
        image_path=None,
    ):
        # The upstream A14B example uses eight high-noise + ten low-noise steps.
        # Preserve the API's total requested step count across the two experts.
        high_steps = max(1, min(steps - 1, round(steps * 4 / 9)))
        cmd = [
            self.sdcli_bin,
            "-M",
            "vid_gen",
            "--diffusion-model",
            assets["low"],
            "--high-noise-diffusion-model",
            assets["high"],
            "--t5xxl",
            assets["encoder"],
            "--vae",
            assets["vae"],
            "-p",
            prompt,
            "-n",
            negative_prompt,
            "-W",
            str(width),
            "-H",
            str(height),
            "--video-frames",
            str(frames),
            "--fps",
            str(fps),
            "--steps",
            str(steps - high_steps),
            "--high-noise-steps",
            str(high_steps),
            "--cfg-scale",
            str(guidance_scale),
            "--high-noise-cfg-scale",
            str(guidance_scale),
            "--sampling-method",
            "euler",
            "--high-noise-sampling-method",
            "euler",
            "--flow-shift",
            "3.0",
            "--diffusion-fa",
            "--vae-tiling",
            "--temporal-tiling",
            "--seed",
            "42",
            "--output-begin-idx",
            "0",
            "-o",
            str(output),
        ]
        if self.device == "cpu":
            cmd += ["--backend", "cpu"]
        else:
            gpu = (
                self.device.split(":", 1)[1]
                if ":" in self.device
                else os.getenv("MAIN_GPU", "0")
            )
            cmd += ["--backend", f"cuda{int(gpu)}"]
        # Do not mmap weights on the GB10's shared memory. Native defaults use
        # ordinary host buffers, avoiding the slow file-backed GPU-copy path.
        if os.getenv("VIDEO_GPU_RESIDENCY", "auto").lower() != "full":
            cmd.append("--offload-to-cpu")
        if image_path is not None:
            cmd += ["-i", str(image_path)]
        return cmd

    def generate(
        self,
        prompt,
        negative_prompt="low quality, blurry, distorted, static",
        num_inference_steps=18,
        guidance_scale=3.5,
        num_frames=121,
        frame_rate=24,
        size="768x512",
        image=None,
        conditions=None,
        include_audio=True,
    ):
        if conditions:
            raise ValueError(
                "Wan 2.2 supports text-to-video and first-image conditioning; use image instead of conditions."
            )
        if (
            str(self.model).lower() in {I2V_REPO.lower(), "wan2.2-i2v"}
            and image is None
        ):
            raise ValueError("The configured Wan I2V model requires an image.")
        try:
            width, height = [int(x) for x in str(size).lower().split("x")]
            frames, fps, steps = (
                int(num_frames),
                int(frame_rate),
                int(num_inference_steps),
            )
            guidance_scale = float(guidance_scale)
        except (TypeError, ValueError):
            raise ValueError(
                "Invalid video dimensions, frame count or generation settings."
            ) from None
        if width < 64 or height < 64 or frames < 5 or fps < 1 or steps < 2:
            raise ValueError(
                "Video needs dimensions >=64, frames >=5, fps >=1 and at least two steps."
            )
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Video guidance scale must be finite and non-negative.")
        width, height = min(width, 1280) // 16 * 16, min(height, 1280) // 16 * 16
        frames = (frames - 1) // 4 * 4 + 1
        started = time.monotonic()
        assets = download_wan_models(image_to_video=image is not None)
        outputs = Path("outputs")
        outputs.mkdir(exist_ok=True)
        result_path = outputs / f"{uuid.uuid4()}.mp4"
        # sd-cli writes lossless frames; ffmpeg supplies the API's MP4 container.
        with tempfile.TemporaryDirectory(prefix="wan-") as tmp:
            work = Path(tmp)
            image_path = None
            if image is not None:
                from PIL import Image, ImageOps

                image_path = work / "input.png"
                ImageOps.fit(
                    image.convert("RGB"),
                    (width, height),
                    method=Image.Resampling.LANCZOS,
                ).save(image_path)
            pattern = work / "frame_%06d.png"
            cmd = self._command(
                assets,
                pattern,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                frames=frames,
                fps=fps,
                steps=steps,
                guidance_scale=guidance_scale,
                image_path=image_path,
            )
            logging.info(
                "[WAN] Starting %s, %sx%s, %s frames, %s total steps",
                "I2V" if image is not None else "T2V",
                width,
                height,
                frames,
                steps,
            )
            log_path = work / "generation.log"
            try:
                with log_path.open("wb") as log:
                    completed = subprocess.run(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=float(os.getenv("VIDEO_GENERATION_TIMEOUT", "3600")),
                    )
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size - 8192))
                    tail = log.read().decode(errors="replace")
                if completed.returncode:
                    logging.error("[WAN] Native generation failed: %s", tail)
                    raise RuntimeError(
                        f"Wan generation failed (sd-cli exit {completed.returncode})"
                    )
                logging.info("[WAN] Native generation complete: %s", tail[-3000:])
                if len(list(work.glob("frame_*.png"))) != frames:
                    raise RuntimeError(
                        "Wan did not produce the requested number of frames"
                    )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        str(fps),
                        "-start_number",
                        "0",
                        "-i",
                        str(pattern),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(result_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=180,
                )
                if not result_path.is_file() or result_path.stat().st_size == 0:
                    raise RuntimeError("Video encoding produced no output")
            except BaseException:
                result_path.unlink(missing_ok=True)
                raise
        logging.info("[WAN] Video ready in %.2fs", time.monotonic() - started)
        if self.local_uri:
            return f"{self.local_uri.rstrip('/')}/{result_path.as_posix()}"
        return result_path.as_posix()
