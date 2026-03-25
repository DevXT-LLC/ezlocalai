import logging
import uuid
import torch
import gc

# LTX-2 requires diffusers with LTX2Pipeline support
LTX2_AVAILABLE = False

try:
    from diffusers import LTX2Pipeline

    LTX2_AVAILABLE = True
    import_success = True
except (ImportError, RuntimeError, Exception) as e:
    logging.warning(
        f"[VIDEO] LTX2Pipeline not available ({e}). Video generation will be unavailable. "
        "Install diffusers from source: pip install git+https://github.com/huggingface/diffusers"
    )
    import_success = False


class VIDEO:
    """Video generation using LTX-2 (Lightricks LTX Video) via diffusers.

    LTX-2 is a diffusion-based text-to-video model from Lightricks.
    It generates high-quality video from text prompts.

    Features:
    - Text-to-video generation
    - Configurable resolution and frame count
    - CPU offloading for memory-constrained devices
    - OOM recovery with sequential CPU offload fallback

    Model: https://huggingface.co/Lightricks/LTX-2
    """

    def __init__(
        self,
        model="Lightricks/LTX-2",
        device="cpu",
        local_uri=None,
    ):
        global import_success
        self.local_uri = local_uri
        self.device = device
        self.pipe = None
        self.dtype = None

        if not import_success:
            return

        self._load_ltx2(model, device)

    def _load_ltx2(self, model: str, device: str):
        """Load LTX-2 video generation pipeline."""
        try:
            from diffusers import LTX2Pipeline

            logging.debug(f"[VIDEO] Loading LTX-2 ({model}) on {device}...")

            # Parse GPU index from device string (e.g. "cuda:1" -> 1)
            gpu_idx = 0
            is_cuda = device.startswith("cuda")
            if ":" in device:
                try:
                    gpu_idx = int(device.split(":")[1])
                except (ValueError, IndexError):
                    pass

            # LTX-2 works best with bfloat16, fall back to float16 if not supported
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

            self.pipe = LTX2Pipeline.from_pretrained(
                model,
                torch_dtype=dtype,
                cache_dir="models",
            )

            if is_cuda:
                # LTX-2 needs significant VRAM (~20-24GB).
                # Use sequential CPU offload for efficient memory use.
                if torch.cuda.is_available():
                    free_vram_gb = torch.cuda.mem_get_info(gpu_idx)[0] / (1024**3)
                    total_vram_gb = torch.cuda.mem_get_info(gpu_idx)[1] / (1024**3)
                    logging.debug(
                        f"[VIDEO] GPU {gpu_idx} VRAM: {free_vram_gb:.1f}GB free / {total_vram_gb:.1f}GB total"
                    )

                    if free_vram_gb < 24:
                        # Use sequential CPU offload - moves each layer to GPU only when needed
                        logging.debug(
                            f"[VIDEO] Limited VRAM on GPU {gpu_idx}, enabling sequential CPU offload..."
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

            logging.debug(
                f"[VIDEO] LTX-2 loaded successfully on {device} with dtype {dtype}"
            )

        except Exception as e:
            logging.error(f"[VIDEO] Failed to load LTX-2: {e}")
            import traceback

            traceback.print_exc()
            self.pipe = None

    def generate(
        self,
        prompt,
        negative_prompt="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, blurry, shaky",
        num_inference_steps=40,
        guidance_scale=4.0,
        num_frames=121,
        frame_rate=24,
        size="768x512",
    ):
        """Generate a video from a text prompt.

        Args:
            prompt: Text description of the video to generate
            negative_prompt: Things to avoid in generation
            num_inference_steps: Number of denoising steps (default: 40)
            guidance_scale: CFG scale (default: 4.0)
            num_frames: Number of frames to generate (should be 8n+1, default: 121)
            frame_rate: Video frame rate (default: 24)
            size: Output video size as "WIDTHxHEIGHT" (dimensions should be divisible by 32)

        Returns:
            Path to saved video file or None on failure
        """
        new_file_name = f"outputs/{uuid.uuid4()}.mp4"

        if not self.pipe:
            return None

        # Parse size
        width, height = map(int, size.split("x"))

        # Ensure dimensions are divisible by 32 (LTX-2 requirement)
        width = (width // 32) * 32
        height = (height // 32) * 32

        # Ensure num_frames follows 8n+1 pattern
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
            logging.debug(
                f"[VIDEO] Adjusted num_frames to {num_frames} (must be 8n+1)"
            )

        # Clamp dimensions to reasonable maximums
        width = min(width, 1280)
        height = min(height, 1280)

        try:
            result = self._generate_ltx2(
                prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                num_frames,
                frame_rate,
            )

            if result is None:
                return None

            # Export frames to video file
            self._export_video(result, new_file_name, frame_rate)

            if self.local_uri:
                return f"{self.local_uri}/{new_file_name}"
            return new_file_name

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logging.warning(f"[VIDEO] GPU OOM during generation: {e}")
                return self._generate_cpu_fallback(
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    num_frames,
                    frame_rate,
                    new_file_name,
                )
            raise

    def _generate_ltx2(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_frames,
        frame_rate,
    ):
        """Generate video using LTX-2 pipeline."""
        gen_device = self.device
        if self.device == "cuda" and hasattr(self.pipe, "_offload_gpu_id"):
            gen_device = "cuda"

        generator = torch.Generator(device=gen_device).manual_seed(42)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result

    def _export_video(self, result, output_path, fps):
        """Export pipeline result to a video file.

        Uses diffusers export_to_video utility when available,
        falls back to OpenCV-based export.
        """
        try:
            from diffusers.utils import export_to_video

            # result.frames is a list of lists of PIL Images
            # Get the first (and typically only) batch
            frames = result.frames[0]
            export_to_video(frames, output_path, fps=fps)
            return
        except (ImportError, AttributeError, Exception) as e:
            logging.debug(f"[VIDEO] export_to_video not available: {e}")

        # Fallback: use OpenCV to write video
        try:
            import cv2
            import numpy as np

            frames = result.frames[0]
            if not frames:
                return

            # Get frame dimensions from first frame
            first_frame = np.array(frames[0])
            h, w = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for frame in frames:
                frame_np = np.array(frame)
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()

        except Exception as e:
            logging.error(f"[VIDEO] Failed to export video: {e}")

    def _generate_cpu_fallback(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_frames,
        frame_rate,
        output_file,
    ):
        """Attempt generation with sequential CPU offload on OOM."""
        logging.warning("[VIDEO] Attempting sequential CPU offload fallback...")
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

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            self._export_video(result, output_file, frame_rate)

            if self.local_uri:
                return f"{self.local_uri}/{output_file}"
            return output_file

        except Exception as cpu_error:
            logging.error(f"[VIDEO] CPU fallback also failed: {cpu_error}")
            return None
