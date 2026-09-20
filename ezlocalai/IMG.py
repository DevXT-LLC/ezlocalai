import logging
import os
import uuid
import shutil
import subprocess
import tempfile
from PIL import Image
import gc
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
    - VAE: qwen_image_vae.safetensors from Comfy-Org/Qwen-Image_ComfyUI
    - Text encoder LLM: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf from mradermacher

    Environment variables:
    - SDCPP_BIN: Path to sd-cli binary (default: /opt/stable-diffusion.cpp/build/sd-cli)
    - SDCPP_MODELS_DIR: Directory for model files (default: models/qwen-image)
    - IMG_MODEL: Set to "qwen-image-2.1" to enable, "none" to disable

    Model repos:
    - https://huggingface.co/leejet/Qwen-Image-2.1-GGUF
    - https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI
    - https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF
    """

    # Model file definitions
    DIFFUSION_MODEL_REPO = "leejet/Qwen-Image-2.1-GGUF"
    DIFFUSION_MODEL_FILE = "qwen_image_2.1-Q4_K.gguf"

    VAE_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
    VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"

    LLM_REPO = "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF"
    LLM_FILE = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"

    # Default generation parameters for Qwen-Image-2.1
    DEFAULT_CFG_SCALE = 2.5
    DEFAULT_SAMPLING_METHOD = "euler"
    DEFAULT_FLOW_SHIFT = 3.0
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
            "SDCPP_BIN", "/opt/stable-diffusion.cpp/build/sd-cli"
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
        self._vae_path = os.path.join(self.models_dir, "qwen_image_vae.safetensors")
        self._llm_path = os.path.join(
            self.models_dir, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        )

        if not all(
            os.path.isfile(p) for p in [self._diffusion_path, self._vae_path, self._llm_path]
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
            logging.info(
                f"[IMG] Downloading VAE: {self.VAE_REPO}/{self.VAE_FILE}"
            )
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

    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=None,
        guidance_scale=None,
        size="1024x1024",
        image=None,
        strength=0.75,
    ):
        """Generate an image from a text prompt using Qwen-Image-2.1 via sd-cli.

        Supports both text-to-image and image-to-image (editing) generation.

        Args:
            prompt: Text description of the image to generate or edit
            negative_prompt: Unused (Qwen-Image uses CFG instead)
            num_inference_steps: Number of denoising steps (default: 20)
            guidance_scale: CFG scale (default: 2.5)
            size: Output image size as "WIDTHxHEIGHT"
            image: Optional input image for img2img editing. Accepts PIL Image,
                   base64 string, data URL, or HTTP URL. When provided, the
                   output will be a transformation of this input image guided
                   by the prompt.
            strength: Denoising strength for img2img (0.0-1.0). Lower values
                     preserve more of the original image; higher values allow
                     more creative transformation. Default 0.75. Only used
                     when image is provided.

        Returns:
            Path to saved image or PIL Image object, or None on failure
        """
        os.makedirs("outputs", exist_ok=True)
        new_file_name = f"outputs/{uuid.uuid4()}.png"

        if not getattr(self, "_available", False):
            logging.error("[IMG] Qwen-Image-2.1 not available (binary or models missing)")
            return None

        # Parse size
        width, height = map(int, size.split("x"))

        # Use defaults for Qwen-Image-2.1
        steps = num_inference_steps if num_inference_steps else self.DEFAULT_STEPS
        cfg = guidance_scale if guidance_scale is not None else self.DEFAULT_CFG_SCALE

        # Create temp directory for intermediate files
        tmp_dir = tempfile.mkdtemp()

        # Load input image for img2img if provided
        init_img_path = None
        if image is not None:
            loaded_img = self._load_image(image)
            if loaded_img is None:
                logging.error("[IMG] Failed to load input image for editing")
                return None
            # Save to temp file for sd-cli
            init_tmp = os.path.join(tmp_dir, "init_input.png")
            loaded_img.save(init_tmp)
            init_img_path = init_tmp

        # Build sd-cli command
        cmd = [
            self.sdcli_bin,
            "--diffusion-model", self._diffusion_path,
            "--vae", self._vae_path,
            "--llm", self._llm_path,
            "-p", prompt,
            "--cfg-scale", str(cfg),
            "--sampling-method", self.DEFAULT_SAMPLING_METHOD,
            "-H", str(height),
            "-W", str(width),
            "--diffusion-fa",
            "--flow-shift", str(self.DEFAULT_FLOW_SHIFT),
        ]

        # Add img2img flags when input image is provided
        if init_img_path:
            cmd.extend(["--init-img", init_img_path])
            cmd.extend(["--strength", str(strength)])
            logging.info(
                f"[IMG] Img2img mode: strength={strength}, "
                f"input={width}x{height}"
            )

        # Add CPU offload if device is CPU or low-VRAM GPU
        if self.device == "cpu" or self._should_offload():
            cmd.append("--offload-to-cpu")

        # Use a temp file for output, then move to final location
        tmp_output = os.path.join(tmp_dir, "output.png")
        cmd.extend(["-o", tmp_output])

        logging.info(f"[IMG] Generating image: {prompt[:80]}... ({width}x{height}, {steps} steps)")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,  # 10 minute timeout for generation
            )

            if result.returncode != 0:
                stderr_text = result.stderr.decode("utf-8", errors="replace")
                logging.error(
                    f"[IMG] sd-cli failed (exit {result.returncode}): {stderr_text[:500]}"
                )
                return None

            # Check if output file was created
            if not os.path.isfile(tmp_output):
                # sd-cli may have saved to a different name; check for any png in tmp_dir
                pngs = [f for f in os.listdir(tmp_dir) if f.endswith(".png")]
                if pngs:
                    tmp_output = os.path.join(tmp_dir, pngs[0])
                else:
                    logging.error("[IMG] sd-cli completed but no output file found")
                    return None

            # Read and save the generated image
            img = Image.open(tmp_output)
            img.save(new_file_name)

            if self.local_uri:
                return f"{self.local_uri}/{new_file_name}"
            return img

        except subprocess.TimeoutExpired:
            logging.error("[IMG] sd-cli timed out after 600 seconds")
            return None
        except Exception as e:
            logging.error(f"[IMG] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _should_offload(self):
        """Check if we should use CPU offload based on available VRAM."""
        try:
            import torch
            if not torch.cuda.is_available():
                return True
            _, total_mem = torch.cuda.mem_get_info(0)
            total_gb = total_mem / (1024**3)
            # Qwen-Image-2.1 Q4_K needs ~15GB VRAM without offload
            return total_gb < 16.0
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

        if isinstance(image_source, str):
            # Base64 data URL
            if image_source.startswith("data:"):
                try:
                    header, encoded = image_source.split(",", 1)
                    img_bytes = base64.b64decode(encoded)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    logging.error(f"[IMG] Failed to decode base64 image: {e}")
                    return None

            # HTTP URL
            if image_source.startswith("http://") or image_source.startswith(
                "https://"
            ):
                try:
                    response = http_requests.get(
                        image_source,
                        timeout=30,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    logging.error(f"[IMG] Failed to download image from URL: {e}")
                    return None

            # Try raw base64
            try:
                img_bytes = base64.b64decode(image_source)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass

        logging.error(f"[IMG] Unsupported image source type: {type(image_source)}")
        return None