import logging
import uuid
import torch
from PIL import Image

try:
    from diffusers import DiffusionPipeline, LCMScheduler

    import_success = True
except ImportError:
    logging.error(
        "Failed to import diffusers. Please install diffusers using 'pip install diffusers'"
    )
    import_success = False


class IMG:
    def __init__(
        self,
        model="stabilityai/sdxl-turbo",
        device="cpu",
        local_uri=None,
    ):
        global import_success
        self.local_uri = local_uri
        if import_success:
            pipe = DiffusionPipeline.from_pretrained(
                model,
                cache_dir="models",
                torch_dtype=torch.float32,
                scheduler=LCMScheduler(beta_start=0.001, beta_end=0.01),
            ).to(device)
            self.pipe = pipe
            self.pipe.enable_attention_slicing()
            self.pipe.safety_checker = None
        else:
            self.pipe = None

    def generate(
        self,
        prompt,
        negative_prompt="low resolution, grainy, distorted",
        num_inference_steps=1,
        guidance_scale=0.0,
        size="512x512",
    ):
        new_file_name = f"outputs/{uuid.uuid4()}.png"
        if self.pipe:
            generator = torch.Generator(device="cpu").manual_seed(0)
            new_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            width, height = map(int, size.split("x"))
            if width != 512 and height != 512:
                upscaled_image = new_image.resize(
                    (width, height), resample=Image.LANCZOS
                )
                upscaled_image.save(new_file_name)
            else:
                new_image.save(new_file_name)
            if self.local_uri:
                return f"{self.local_uri}/{new_file_name}"
            return upscaled_image
        else:
            return None
