import logging
import os
import uuid
import fcntl
import subprocess
import tempfile
from PIL import Image
import io
import base64
import requests as http_requests

# Qwen-Image-2.1 GGUF requires stable-diffusion.cpp (sd-cli binary)
# This is a C++ inference engine, not Python diffusers.
# The sd-cli binary is built in the Docker image and its path is set via SDCPP_BIN env var.

import_success = True  # We always "succeed" at import; binary check happens in __init__


class IMG:
    """Image generation using Qwen-Image-2.1 GGUF via stable-diffusion.cpp (sd-cli).

    Qwen-Image-2.1 is a high-quality image generation model that runs through the
    stable-diffusion.cpp C++ engine as a subprocess. This avoids loading large
    PyTorch models into the Python process and allows efficient GPU utilization
    via flash attention and CPU offloading handled natively by sd-cli.

    Required model files (downloaded on first use):
    - Diffusion model: qwen_image_2.1-Q4_K.gguf from leejet/Qwen-Image-2.1-GGUF
    - VAE: qwen_image_2.1_vae_bf16.safetensors from Comfy-Org/Qwen-Image-2.1
    - Text encoder LLM: Qwen3VL-8B-Instruct-Q4_K_M.gguf from Qwen

    Environment variables:
    - SDCPP_BIN: Path to sd-cli binary (default: /opt/stable-diffusion.cpp/build/bin/sd-cli)
    - SDCPP_MODELS_DIR: Directory for model files (default: models/qwen-image)
    - IMG_MODEL: Set to "qwen-image-2.1" to enable, "none" to disable

    Model repos:
    - https://huggingface.co/leejet/Qwen-Image-2.1-GGUF
    - https://huggingface.co/Comfy-Org/Qwen-Image-2.1
    - https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF
    """

    # Model file definitions
    DIFFUSION_MODEL_REPO = "leejet/Qwen-Image-2.1-GGUF"
    DIFFUSION_MODEL_FILE = "qwen_image_2.1-Q4_K.gguf"

    VAE_REPO = "Comfy-Org/Qwen-Image-2.1"
    VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"

    LLM_REPO = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
    LLM_FILE = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"

    MMPROJ_REPO = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
    MMPROJ_FILE = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"

    # Default generation parameters for Qwen-Image-2.1
    DEFAULT_CFG_SCALE = 6.0
    DEFAULT_SAMPLING_METHOD = "euler"
    DEFAULT_STEPS = 20

    def __init__(
        self,
        model="qwen-image-2.1",
        device="cpu",
        local_uri=None,
    ):
        self.local_uri = local_uri
        self.device = device
        self.models_dir = os.environ.get("SDCPP_MODELS_DIR", "models/qwen-image")
        self.sdcli_bin = os.environ.get(
            "SDCPP_BIN", "/opt/stable-diffusion.cpp/build/bin/sd-cli"
        )

        # Verify binary exists
        if not os.path.isfile(self.sdcli_bin):
            logging.warning(
                f"[IMG] sd-cli binary not found at {self.sdcli_bin}. "
                "Image generation will be unavailable. "
                "Set SDCPP_BIN to the correct path or build stable-diffusion.cpp."
            )
            self._available = False
            return

        # Verify models are downloaded
        self._diffusion_path = os.path.join(self.models_dir, self.DIFFUSION_MODEL_FILE)
        self._vae_path = os.path.join(self.models_dir, self.VAE_FILE)
        self._llm_path = os.path.join(
            self.models_dir, "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
        )
        self._mmproj_path = os.path.join(
            self.models_dir, "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
        )

        if not all(
            os.path.isfile(p)
            for p in [
                self._diffusion_path,
                self._vae_path,
                self._llm_path,
                self._mmproj_path,
            ]
        ):
            logging.info("[IMG] Model files not found, downloading...")
            try:
                self._download_models()
            except Exception as e:
                logging.error(f"[IMG] Failed to download models: {e}")
                self._available = False
                return

        self._available = True
        logging.info(
            f"[IMG] Qwen-Image-2.1 ready. Binary: {self.sdcli_bin}, "
            f"Models: {self.models_dir}"
        )

    def _download_models(self):
        """Download all required model files from HuggingFace."""
        from huggingface_hub import hf_hub_download

        os.makedirs(self.models_dir, exist_ok=True)

        # Download diffusion model GGUF
        if not os.path.isfile(self._diffusion_path):
            logging.info(
                f"[IMG] Downloading diffusion model: {self.DIFFUSION_MODEL_REPO}/{self.DIFFUSION_MODEL_FILE}"
            )
            hf_hub_download(
                self.DIFFUSION_MODEL_REPO,
                filename=self.DIFFUSION_MODEL_FILE,
                cache_dir=self.models_dir,
                local_dir=self.models_dir,
            )

        # Download VAE safetensors
        if not os.path.isfile(self._vae_path):
            logging.info(f"[IMG] Downloading VAE: {self.VAE_REPO}/{self.VAE_FILE}")
            hf_hub_download(
                self.VAE_REPO,
                filename=self.VAE_FILE,
                cache_dir=self.models_dir,
                local_dir=self.models_dir,
            )

        # Download text encoder LLM GGUF
        if not os.path.isfile(self._llm_path):
            logging.info(
                f"[IMG] Downloading text encoder: {self.LLM_REPO}/{self.LLM_FILE}"
            )
            hf_hub_download(
                self.LLM_REPO,
                filename=self.LLM_FILE,
                cache_dir=self.models_dir,
                local_dir=self.models_dir,
            )

        # Download mmproj vision file for image editing
        if not os.path.isfile(self._mmproj_path):
            logging.info(
                f"[IMG] Downloading mmproj vision: {self.MMPROJ_REPO}/{self.MMPROJ_FILE}"
            )
            hf_hub_download(
                self.MMPROJ_REPO,
                filename=self.MMPROJ_FILE,
                cache_dir=self.models_dir,
                local_dir=self.models_dir,
            )

    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=None,
        guidance_scale=None,
        size="1024x1024",
        image=None,
        images=None,
        strength=0.75,
    ):
        """Generate a PNG, using the primary image as both init and Qwen reference.

        Additional references retain their order. Invalid inputs fail the request
        instead of silently generating without the user's visual context.
        """
        if not getattr(self, "_available", False):
            raise RuntimeError(
                "Qwen-Image-2.1 is unavailable (binary or models missing)"
            )
        try:
            width, height = map(int, size.split("x"))
        except (AttributeError, ValueError):
            raise ValueError("size must be WIDTHxHEIGHT") from None
        if any(v <= 0 or v % 32 for v in (width, height)):
            raise ValueError("Image dimensions must be positive multiples of 32")
        if not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if images is not None and not isinstance(images, (list, tuple)):
            raise ValueError("images must be a list")
        if len(images or []) > 10:
            raise ValueError("At most 10 additional reference images are supported")
        steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.DEFAULT_STEPS
        )
        cfg = guidance_scale if guidance_scale is not None else self.DEFAULT_CFG_SCALE
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        # A file lock also serializes sd-cli across Uvicorn worker processes.
        # Pipes retains its async lock until the worker thread finishes on cancellation.
        with open(os.path.join(self.models_dir, ".inference.lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with tempfile.TemporaryDirectory(prefix="qwen-image-") as tmp_dir:
                refs = ([image] if image is not None else []) + list(images or [])
                ref_paths = []
                for idx, source in enumerate(refs):
                    loaded = self._load_image(source)
                    if loaded is None:
                        raise ValueError(f"Invalid reference image at index {idx}")
                    path = os.path.join(tmp_dir, f"ref_{idx}.png")
                    with loaded:
                        loaded.save(path)
                    ref_paths.append(path)

                cmd = [
                    self.sdcli_bin,
                    "--diffusion-model",
                    self._diffusion_path,
                    "--vae",
                    self._vae_path,
                    "--llm",
                    self._llm_path,
                    "-p",
                    prompt,
                    "--cfg-scale",
                    str(cfg),
                    "--sampling-method",
                    self.DEFAULT_SAMPLING_METHOD,
                    "--steps",
                    str(steps),
                    "-H",
                    str(height),
                    "-W",
                    str(width),
                    "--diffusion-fa",
                ]
                if negative_prompt:
                    cmd.extend(["--negative-prompt", negative_prompt])
                if image is not None:
                    cmd.extend(
                        ["--init-img", ref_paths[0], "--strength", str(strength)]
                    )
                if ref_paths:
                    if not os.path.isfile(self._mmproj_path):
                        raise RuntimeError(
                            "Qwen image editing requires mmproj vision weights"
                        )
                    cmd.extend(["--llm_vision", self._mmproj_path])
                    for path in ref_paths:
                        cmd.extend(["-r", path])
                if self.device == "cpu":
                    cmd.extend(["--backend", "cpu"])
                elif str(self.device).startswith("cuda:"):
                    cmd.extend(["--backend", str(self.device).replace(":", "")])
                if self.device != "cpu" and self._should_offload():
                    cmd.append("--offload-to-cpu")
                tmp_output = os.path.join(tmp_dir, "output.png")
                cmd.extend(["-o", tmp_output])
                logging.info(
                    "[IMG] Generating %sx%s image, %s steps, %s references",
                    width,
                    height,
                    steps,
                    len(ref_paths),
                )
                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                except subprocess.TimeoutExpired as exc:
                    # subprocess.run kills and waits for the child before raising.
                    raise RuntimeError("sd-cli timed out after 600 seconds") from exc
                diagnostics = result.stdout.decode("utf-8", errors="replace")
                if result.returncode != 0:
                    logging.error(
                        "[IMG] sd-cli exit %s: %s",
                        result.returncode,
                        diagnostics[-4000:],
                    )
                    raise RuntimeError(
                        f"sd-cli failed (exit {result.returncode}); see server logs"
                    )
                # Never select arbitrary PNGs: those can be the user's input images.
                if not os.path.isfile(tmp_output):
                    logging.error(
                        "[IMG] sd-cli produced no output: %s", diagnostics[-4000:]
                    )
                    raise RuntimeError("sd-cli completed without an output image")
                with Image.open(tmp_output) as generated:
                    generated.load()
                    if generated.format != "PNG" or generated.size != (width, height):
                        raise RuntimeError(
                            "sd-cli returned an invalid PNG or unexpected dimensions"
                        )
                    if self.local_uri:
                        os.makedirs("outputs", exist_ok=True)
                        name = f"outputs/{uuid.uuid4()}.png"
                        generated.save(name)
                        return f"{self.local_uri.rstrip('/')}/{name}"
                with open(tmp_output, "rb") as output:
                    return base64.b64encode(output.read()).decode("ascii")

    def _should_offload(self):
        """Check if we should use CPU offload based on available VRAM."""
        try:
            import torch

            if not torch.cuda.is_available():
                return True
            free_mem, _ = torch.cuda.mem_get_info(self.device)
            free_gb = free_mem / (1024**3)
            # Qwen-Image-2.1 Q4_K needs ~15GB VRAM without offload
            return free_gb < 16.0
        except Exception:
            return True

    def _load_image(self, image_source):
        """Load an image from various sources (PIL, base64, URL).

        Args:
            image_source: PIL Image, base64 string, data URL, or HTTP URL

        Returns:
            PIL Image or None
        """
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        if not isinstance(image_source, str):
            return None
        try:
            if image_source.startswith(("http://", "https://")):
                with http_requests.get(
                    image_source, timeout=30, stream=True
                ) as response:
                    response.raise_for_status()
                    chunks = bytearray()
                    for chunk in response.iter_content(64 * 1024):
                        chunks.extend(chunk)
                        if len(chunks) > 32 * 1024 * 1024:
                            raise ValueError("Image exceeds 32 MiB")
                    image_bytes = bytes(chunks)
            else:
                encoded = image_source
                if image_source.startswith("data:"):
                    header, encoded = image_source.split(",", 1)
                    if not header.startswith("data:image/") or not header.endswith(
                        ";base64"
                    ):
                        raise ValueError("Expected a base64 image data URL")
                if len(encoded) > 45 * 1024 * 1024:
                    raise ValueError("Encoded image exceeds size limit")
                image_bytes = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(image_bytes)) as loaded:
                return loaded.convert("RGB")
        except (
            ValueError,
            OSError,
            http_requests.RequestException,
            Image.DecompressionBombError,
        ):
            logging.warning("[IMG] Could not load reference image")
            return None
