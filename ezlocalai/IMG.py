import logging
import os
import uuid
import torch
from PIL import Image
import gc
import io
import base64
import requests as http_requests

# FLUX.2-klein-4B requires diffusers from source with Flux2KleinPipeline support
# pip install git+https://github.com/huggingface/diffusers
import_success = False

try:
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

    # Patch GGUFParameter to work with accelerate's cpu_offload.
    # accelerate moves parameters to meta device which loses quant_type,
    # causing KeyError(None) in GGML_QUANT_SIZES lookup.
    from diffusers.quantizers.gguf.utils import GGUFParameter

    _original_gguf_new = GGUFParameter.__new__

    def _patched_gguf_new(cls, data, requires_grad=False, quant_type=None):
        if quant_type is None:
            return torch.nn.Parameter.__new__(cls, data, requires_grad=requires_grad)
        return _original_gguf_new(
            cls, data, requires_grad=requires_grad, quant_type=quant_type
        )

    GGUFParameter.__new__ = _patched_gguf_new

    import_success = True
except (ImportError, RuntimeError, Exception) as e:
    logging.error(
        f"Failed to import Flux2KleinPipeline ({e}). Image generation will be unavailable. "
        "Install diffusers using 'pip install git+https://github.com/huggingface/diffusers'"
    )

# GGUF quant files available in unsloth/FLUX.2-klein-4B-GGUF
GGUF_QUANT_FILES = {
    "Q2_K": "flux-2-klein-4b-Q2_K.gguf",
    "Q3_K_S": "flux-2-klein-4b-Q3_K_S.gguf",
    "Q3_K_M": "flux-2-klein-4b-Q3_K_M.gguf",
    "Q4_0": "flux-2-klein-4b-Q4_0.gguf",
    "Q4_1": "flux-2-klein-4b-Q4_1.gguf",
    "Q4_K_S": "flux-2-klein-4b-Q4_K_S.gguf",
    "Q4_K_M": "flux-2-klein-4b-Q4_K_M.gguf",
    "Q5_0": "flux-2-klein-4b-Q5_0.gguf",
    "Q5_1": "flux-2-klein-4b-Q5_1.gguf",
    "Q5_K_S": "flux-2-klein-4b-Q5_K_S.gguf",
    "Q5_K_M": "flux-2-klein-4b-Q5_K_M.gguf",
    "Q6_K": "flux-2-klein-4b-Q6_K.gguf",
    "Q8_0": "flux-2-klein-4b-Q8_0.gguf",
    "BF16": "flux-2-klein-4b-BF16.gguf",
    "F16": "flux-2-klein-4b-F16.gguf",
}
DEFAULT_GGUF_QUANT = "Q4_K_M"
# Full-precision pipeline config repo (text_encoder, vae, scheduler, tokenizer)
FLUX2_KLEIN_CONFIG_REPO = "black-forest-labs/FLUX.2-klein-4B"
# Unsloth repo for GGUF quantized transformer files
UNSLOTH_REPO = "unsloth/FLUX.2-klein-4B-GGUF"


class IMG:
    """Image generation and editing using FLUX.2-klein-4B with GGUF quantization.

    FLUX.2-klein-4B is a fast 4B parameter image model from Black Forest Labs.
    It unifies text-to-image generation and image editing in a single architecture.

    Features:
    - Sub-second inference on high-end GPUs with only 4 steps
    - Text-to-image generation and image editing (image + text to image)
    - GGUF quantization for efficient memory usage (~2.6GB for Q4_K_M)
    - CPU offloading for memory-constrained devices
    - Runs on consumer GPUs (RTX 3090/4070 with ~13GB VRAM at full precision)

    Model: https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF
    """

    def __init__(
        self,
        model="unsloth/FLUX.2-klein-4B-GGUF",
        device="cpu",
        local_uri=None,
    ):
        global import_success
        self.local_uri = local_uri
        self.device = device
        self.pipe = None

        if not import_success:
            return

        self._load_pipeline(model, device)

    def _load_pipeline(self, model: str, device: str):
        """Load FLUX.2-klein-4B pipeline with GGUF-quantized transformer."""
        try:
            from huggingface_hub import hf_hub_download

            logging.info(f"[IMG] Loading FLUX.2-klein-4B GGUF ({model}) on {device}...")

            # Parse GPU index
            gpu_idx = 0
            is_cuda = device.startswith("cuda")
            if ":" in device:
                try:
                    gpu_idx = int(device.split(":")[1])
                except (ValueError, IndexError):
                    pass

            # Determine dtype
            if device == "cpu":
                dtype = torch.float32
            elif (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(gpu_idx)[0] >= 8
            ):
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            self.dtype = dtype

            # Download the GGUF transformer file
            gguf_filename = GGUF_QUANT_FILES.get(
                DEFAULT_GGUF_QUANT, GGUF_QUANT_FILES["Q4_K_M"]
            )

            logging.info(
                f"[IMG] Downloading GGUF transformer: {UNSLOTH_REPO}/{gguf_filename}"
            )
            gguf_path = hf_hub_download(
                UNSLOTH_REPO, filename=gguf_filename, cache_dir="models"
            )

            # Load GGUF-quantized transformer
            # We must download the transformer config from the klein repo first,
            # otherwise from_single_file auto-detects the GGUF as FLUX.2-dev (gated)
            logging.info("[IMG] Loading GGUF transformer...")
            config_path = hf_hub_download(
                FLUX2_KLEIN_CONFIG_REPO,
                "transformer/config.json",
                cache_dir="models",
            )

            config_dir = os.path.dirname(config_path)

            transformer = Flux2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
                config=config_dir,
            )

            # Load the pipeline with our GGUF transformer, pulling other
            # components (text_encoder, vae, scheduler, tokenizer) from config repo
            logging.info(
                f"[IMG] Loading pipeline components from {FLUX2_KLEIN_CONFIG_REPO}..."
            )
            self.pipe = Flux2KleinPipeline.from_pretrained(
                FLUX2_KLEIN_CONFIG_REPO,
                transformer=transformer,
                torch_dtype=dtype,
                cache_dir="models",
            )

            if is_cuda:
                # Choose offload strategy based on total GPU capacity:
                #  - >=10GB total: model CPU offload (moves whole modules, fast)
                #  - <10GB total: sequential CPU offload (per-layer, slower)
                # We use total VRAM, not free, because model_cpu_offload moves
                # modules to CPU after setup.
                _, total_mem = torch.cuda.mem_get_info(gpu_idx)
                total_gb = total_mem / (1024**3)
                use_model_offload = total_gb >= 10.0

                try:
                    if use_model_offload:
                        self.pipe.enable_model_cpu_offload(gpu_id=gpu_idx)
                        logging.info(
                            f"[IMG] Model CPU offload enabled on GPU {gpu_idx} "
                            f"({total_gb:.1f}GB total - whole-module transfers)"
                        )
                    else:
                        self.pipe.enable_sequential_cpu_offload(gpu_id=gpu_idx)
                        logging.info(
                            f"[IMG] Sequential CPU offload enabled on GPU {gpu_idx} "
                            f"({total_gb:.1f}GB total - per-layer transfers)"
                        )
                except Exception as e:
                    offload_type = "Model" if use_model_offload else "Sequential"
                    logging.warning(
                        f"[IMG] {offload_type} CPU offload failed ({e}), "
                        "falling back to CPU"
                    )
                    for name, component in self.pipe.components.items():
                        if isinstance(component, torch.nn.Module):
                            try:
                                component.to("cpu")
                            except Exception:
                                pass
            else:
                for name, component in self.pipe.components.items():
                    if isinstance(component, torch.nn.Module):
                        try:
                            component.to("cpu")
                        except Exception:
                            pass

            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass

            logging.info(
                f"[IMG] FLUX.2-klein-4B GGUF loaded successfully on {device} with dtype {dtype}"
            )

        except Exception as e:
            logging.error(f"[IMG] Failed to load FLUX.2-klein-4B: {e}")
            import traceback

            traceback.print_exc()
            self.pipe = None

    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=None,
        guidance_scale=None,
        size="1024x1024",
        image=None,
    ):
        """Generate an image from a text prompt, or edit an image with a text prompt.

        Args:
            prompt: Text description of the image to generate or editing instruction
            negative_prompt: Unused (FLUX.2-klein does not use negative prompts)
            num_inference_steps: Number of denoising steps (default: 4)
            guidance_scale: CFG scale (default: 4.0)
            size: Output image size as "WIDTHxHEIGHT"
            image: Optional input image for editing (PIL Image, base64 string, or URL)

        Returns:
            Path to saved image or PIL Image object, or None on failure
        """
        os.makedirs("outputs", exist_ok=True)
        new_file_name = f"outputs/{uuid.uuid4()}.png"

        if not self.pipe:
            return None

        # Parse size
        width, height = map(int, size.split("x"))

        # FLUX.2-klein defaults: 4 steps, guidance_scale 4.0
        steps = num_inference_steps if num_inference_steps else 4
        cfg = guidance_scale if guidance_scale is not None else 4.0

        # Clamp dimensions
        width = min(width, 1024)
        height = min(height, 1024)

        # Load input image if provided (for image editing)
        input_image = self._load_image(image) if image else None

        try:
            new_image = self._generate_flux2klein(
                prompt, width, height, steps, cfg, input_image
            )

            if new_image is None:
                return None

            # Resize if requested size differs from generation size
            if width != new_image.width or height != new_image.height:
                new_image = new_image.resize((width, height), resample=Image.LANCZOS)

            new_image.save(new_file_name)
            if self.local_uri:
                return f"{self.local_uri}/{new_file_name}"
            return new_image

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logging.warning(f"[IMG] GPU OOM during generation: {e}")
                return self._generate_cpu_fallback(
                    prompt, width, height, steps, cfg, input_image, new_file_name
                )
            raise

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

    def _generate_flux2klein(
        self, prompt, width, height, steps, guidance_scale, image=None
    ):
        """Generate or edit an image using FLUX.2-klein-4B."""
        # Determine device for generator
        gen_device = self.device
        if self.device.startswith("cuda") and hasattr(self.pipe, "_offload_gpu_id"):
            gen_device = "cuda"

        generator = torch.Generator(device=gen_device).manual_seed(42)

        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # If image provided, pass it for editing mode
        if image is not None:
            kwargs["image"] = image

        result = self.pipe(**kwargs)
        return result.images[0]

    def _generate_cpu_fallback(
        self, prompt, width, height, steps, cfg, image, output_file
    ):
        """Attempt generation with sequential CPU offload on OOM."""
        logging.warning("[IMG] Attempting sequential CPU offload fallback...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            if hasattr(self.pipe, "enable_sequential_cpu_offload"):
                try:
                    self.pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass

            generator = torch.Generator(device="cpu").manual_seed(42)

            kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "generator": generator,
            }

            if image is not None:
                kwargs["image"] = image

            result = self.pipe(**kwargs)
            new_image = result.images[0]
            new_image.save(output_file)

            if self.local_uri:
                return f"{self.local_uri}/{output_file}"
            return new_image

        except Exception as cpu_error:
            logging.error(f"[IMG] CPU fallback also failed: {cpu_error}")
            return None
