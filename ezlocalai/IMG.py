import logging
import uuid
import torch
from PIL import Image
import gc

# Z-Image requires the latest diffusers from source with ZImagePipeline support
# pip install git+https://github.com/huggingface/diffusers
ZIMAGE_AVAILABLE = False
SDXL_AVAILABLE = False

try:
    from diffusers import ZImagePipeline

    ZIMAGE_AVAILABLE = True
    import_success = True
except ImportError:
    try:
        # Fallback to older SDXL-Lightning if ZImagePipeline not available
        from diffusers import (
            StableDiffusionXLPipeline,
            UNet2DConditionModel,
            EulerDiscreteScheduler,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        SDXL_AVAILABLE = True
        import_success = True
        logging.warning(
            "[IMG] ZImagePipeline not available, will use SDXL-Lightning fallback. "
            "For Z-Image support, install diffusers from source: pip install git+https://github.com/huggingface/diffusers"
        )
    except ImportError:
        logging.error(
            "Failed to import diffusers. Please install diffusers using 'pip install git+https://github.com/huggingface/diffusers'"
        )
        import_success = False


class IMG:
    """Image generation using Z-Image-Turbo or SDXL-Lightning fallback.

    Z-Image-Turbo is a highly efficient 6B parameter image generation model from Tongyi-MAI.
    It offers sub-second inference on high-end GPUs and fits within 16GB VRAM.

    Features:
    - Fast generation with only 8 inference steps
    - High quality photorealistic images
    - Bilingual text rendering (English & Chinese)
    - CPU offloading for memory-constrained devices

    Model: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
    """

    def __init__(
        self,
        model="Tongyi-MAI/Z-Image-Turbo",
        device="cpu",
        local_uri=None,
    ):
        global import_success
        self.local_uri = local_uri
        self.device = device
        self.pipe = None
        self.model_type = None  # "zimage" or "sdxl_lightning"
        self.num_steps = 9  # Default for Z-Image

        if not import_success:
            return

        # Check if we should use Z-Image or fallback to SDXL-Lightning
        # Z-Image model names contain "Z-Image" or use default Tongyi-MAI model
        is_zimage_model = (
            "z-image" in model.lower()
            or "zimage" in model.lower()
            or model == "Tongyi-MAI/Z-Image-Turbo"
        )

        if is_zimage_model and ZIMAGE_AVAILABLE:
            self._load_zimage(model, device)
        elif SDXL_AVAILABLE:
            # Use SDXL-Lightning as fallback
            self._load_sdxl_lightning(device)
        else:
            logging.error("[IMG] No image generation backend available")
            self.pipe = None

    def _load_zimage(self, model: str, device: str):
        """Load Z-Image-Turbo model using diffusers ZImagePipeline."""
        try:
            from diffusers import ZImagePipeline

            logging.debug(f"[IMG] Loading Z-Image-Turbo ({model}) on {device}...")

            # Parse GPU index from device string (e.g. "cuda:1" -> 1)
            gpu_idx = 0
            is_cuda = device.startswith("cuda")
            if ":" in device:
                try:
                    gpu_idx = int(device.split(":")[1])
                except (ValueError, IndexError):
                    pass

            # Z-Image works best with bfloat16, but fall back to float16 if not supported
            if device == "cpu":
                dtype = torch.float32
            elif (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(gpu_idx)[0] >= 8
            ):
                # bfloat16 supported on Ampere (SM 8.0) and newer
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            self.dtype = dtype

            self.pipe = ZImagePipeline.from_pretrained(
                model,
                torch_dtype=dtype,
                cache_dir="models",
            )

            if is_cuda:
                # Z-Image-Turbo needs ~16GB VRAM. With other models loaded,
                # we almost always need CPU offloading for efficient memory use.
                # Sequential CPU offload is more aggressive but allows running
                # with other models loaded in VRAM.
                if torch.cuda.is_available():
                    free_vram_gb = torch.cuda.mem_get_info(gpu_idx)[0] / (1024**3)
                    total_vram_gb = torch.cuda.mem_get_info(gpu_idx)[1] / (1024**3)
                    logging.debug(
                        f"[IMG] GPU {gpu_idx} VRAM: {free_vram_gb:.1f}GB free / {total_vram_gb:.1f}GB total"
                    )

                    if free_vram_gb < 18:
                        # Use sequential CPU offload - moves each layer to GPU only when needed
                        # This is slower but works with limited VRAM
                        logging.debug(
                            f"[IMG] Limited VRAM on GPU {gpu_idx}, enabling sequential CPU offload..."
                        )
                        self.pipe.enable_sequential_cpu_offload(gpu_id=gpu_idx)
                    else:
                        self.pipe.to(device)
                else:
                    self.pipe.to(device)
            else:
                self.pipe.to(device)

            # Enable memory efficient attention if available
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass  # Not all pipelines support this

            # Enable VAE slicing for memory efficiency
            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass

            self.model_type = "zimage"
            self.num_steps = 9  # Z-Image uses 9 steps (results in 8 DiT forwards)
            logging.debug(
                f"[IMG] Z-Image-Turbo loaded successfully on {device} with dtype {dtype}"
            )

        except Exception as e:
            logging.error(f"[IMG] Failed to load Z-Image-Turbo: {e}")
            import traceback

            traceback.print_exc()
            self.pipe = None

    def _load_sdxl_lightning(self, device: str):
        """Fallback: Load SDXL-Lightning model."""
        try:
            from diffusers import (
                StableDiffusionXLPipeline,
                UNet2DConditionModel,
                EulerDiscreteScheduler,
            )
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            # 4-step is more stable than 2-step, especially on CPU
            self.num_steps = 4 if device == "cpu" else 2
            ckpt = f"sdxl_lightning_{self.num_steps}step_unet.safetensors"

            # CPU requires float32, GPU can use float16
            self.dtype = torch.float32 if device == "cpu" else torch.float16

            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"

            logging.debug(
                f"[IMG] Loading SDXL-Lightning {self.num_steps}-step on {device} with {self.dtype}..."
            )

            # Load UNet with SDXL-Lightning weights
            unet = UNet2DConditionModel.from_config(
                base,
                subfolder="unet",
                cache_dir="models",
            ).to(device, self.dtype)

            unet.load_state_dict(
                load_file(
                    hf_hub_download(repo, ckpt, cache_dir="models"),
                    device="cpu",
                )
            )

            # Build pipeline kwargs based on device
            pipe_kwargs = {
                "unet": unet,
                "torch_dtype": self.dtype,
                "cache_dir": "models",
            }
            if device != "cpu":
                pipe_kwargs["variant"] = "fp16"

            # Load pipeline with Lightning UNet
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                base,
                **pipe_kwargs,
            ).to(device)

            # Configure scheduler for Lightning (trailing timesteps)
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )

            self.pipe.enable_attention_slicing()
            self.pipe.safety_checker = None
            self.model_type = "sdxl_lightning"
            logging.debug(
                f"[IMG] SDXL-Lightning {self.num_steps}-step model loaded successfully on {device}."
            )
        except Exception as e:
            logging.error(f"[IMG] Failed to load SDXL-Lightning: {e}")
            import traceback

            traceback.print_exc()
            self.pipe = None

    def generate(
        self,
        prompt,
        negative_prompt="low resolution, grainy, distorted, blurry, ugly",
        num_inference_steps=None,
        guidance_scale=None,
        size="1024x1024",
    ):
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: Things to avoid (only used with guidance_scale > 0)
            num_inference_steps: Number of denoising steps (default: auto based on model)
            guidance_scale: CFG scale (Z-Image uses 0.0, SDXL-Lightning uses 0)
            size: Output image size as "WIDTHxHEIGHT"

        Returns:
            Path to saved image or PIL Image object, or None on failure
        """
        new_file_name = f"outputs/{uuid.uuid4()}.png"

        if not self.pipe:
            return None

        # Parse size
        width, height = map(int, size.split("x"))

        # Set model-specific defaults
        if self.model_type == "zimage":
            steps = num_inference_steps if num_inference_steps else 9
            cfg = guidance_scale if guidance_scale is not None else 0.0
            # Z-Image supports larger sizes natively
            max_size = 2048
        else:
            steps = num_inference_steps if num_inference_steps else self.num_steps
            cfg = guidance_scale if guidance_scale is not None else 0
            max_size = 1024

        # Clamp dimensions
        width = min(width, max_size)
        height = min(height, max_size)

        try:
            if self.model_type == "zimage":
                new_image = self._generate_zimage(prompt, width, height, steps, cfg)
            else:
                new_image = self._generate_sdxl(
                    prompt, negative_prompt, width, height, steps, cfg
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
                    prompt, negative_prompt, width, height, steps, cfg, new_file_name
                )
            raise

    def _generate_zimage(self, prompt, width, height, steps, guidance_scale):
        """Generate image using Z-Image-Turbo."""
        # Determine device for generator
        gen_device = self.device
        if self.device == "cuda" and hasattr(self.pipe, "_offload_gpu_id"):
            # When using CPU offload, generator should be on CUDA
            gen_device = "cuda"

        generator = torch.Generator(device=gen_device).manual_seed(42)

        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]

    def _generate_sdxl(
        self, prompt, negative_prompt, width, height, steps, guidance_scale
    ):
        """Generate image using SDXL-Lightning."""
        generator = torch.Generator(device=self.device).manual_seed(42)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if guidance_scale > 0 else None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        return result.images[0]

    def _generate_cpu_fallback(
        self, prompt, negative_prompt, width, height, steps, cfg, output_file
    ):
        """Attempt generation with sequential CPU offload on OOM."""
        logging.warning("[IMG] Attempting sequential CPU offload fallback...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Enable more aggressive sequential offloading
            if hasattr(self.pipe, "enable_sequential_cpu_offload"):
                try:
                    self.pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass

            generator = torch.Generator(device="cpu").manual_seed(42)

            if self.model_type == "zimage":
                result = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                )
            else:
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if cfg > 0 else None,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=width,
                    height=height,
                    generator=generator,
                )

            new_image = result.images[0]
            new_image.save(output_file)

            if self.local_uri:
                return f"{self.local_uri}/{output_file}"
            return new_image

        except Exception as cpu_error:
            logging.error(f"[IMG] CPU fallback also failed: {cpu_error}")
            return None
