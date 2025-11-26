import logging
import uuid
import torch
from PIL import Image

try:
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    import_success = True
except ImportError:
    logging.error(
        "Failed to import diffusers. Please install diffusers using 'pip install diffusers'"
    )
    import_success = False


class IMG:
    def __init__(
        self,
        model="ByteDance/SDXL-Lightning",
        device="cpu",
        local_uri=None,
    ):
        global import_success
        self.local_uri = local_uri
        self.device = device
        
        # Determine step count and checkpoint based on device
        # 4-step is more stable than 2-step, especially on CPU
        self.num_steps = 4 if device == "cpu" else 2
        ckpt = f"sdxl_lightning_{self.num_steps}step_unet.safetensors"
        
        # CPU requires float32, GPU can use float16
        self.dtype = torch.float32 if device == "cpu" else torch.float16
        
        if import_success:
            try:
                base = "stabilityai/stable-diffusion-xl-base-1.0"
                repo = "ByteDance/SDXL-Lightning"
                
                logging.info(f"[IMG] Loading SDXL-Lightning {self.num_steps}-step on {device} with {self.dtype}...")
                
                # Load UNet with SDXL-Lightning weights
                unet = UNet2DConditionModel.from_config(
                    base, 
                    subfolder="unet",
                    cache_dir="models",
                ).to(device, self.dtype)
                
                unet.load_state_dict(
                    load_file(
                        hf_hub_download(repo, ckpt, cache_dir="models"), 
                        device="cpu"  # Always load to CPU first, then move
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
                    self.pipe.scheduler.config, 
                    timestep_spacing="trailing"
                )
                
                self.pipe.enable_attention_slicing()
                self.pipe.safety_checker = None
                logging.info(f"[IMG] SDXL-Lightning {self.num_steps}-step model loaded successfully on {device}.")
            except Exception as e:
                logging.error(f"[IMG] Failed to load SDXL-Lightning: {e}")
                import traceback
                traceback.print_exc()
                self.pipe = None
        else:
            self.pipe = None

    def generate(
        self,
        prompt,
        negative_prompt="low resolution, grainy, distorted, blurry, ugly",
        num_inference_steps=None,
        guidance_scale=0,
        size="1024x1024",
    ):
        new_file_name = f"outputs/{uuid.uuid4()}.png"
        if self.pipe:
            # Use the step count the model was loaded with
            steps = num_inference_steps if num_inference_steps else self.num_steps
            generator = torch.Generator(device=self.device).manual_seed(0)
            width, height = map(int, size.split("x"))
            
            new_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if guidance_scale > 0 else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=min(width, 1024),
                height=min(height, 1024),
                generator=generator,
            ).images[0]
            
            # Resize if requested size differs from generation size
            if width != new_image.width or height != new_image.height:
                new_image = new_image.resize((width, height), resample=Image.LANCZOS)
            
            new_image.save(new_file_name)
            if self.local_uri:
                return f"{self.local_uri}/{new_file_name}"
            return new_image
        else:
            return None
