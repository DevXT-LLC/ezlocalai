import logging
import uuid
import torch

try:
    from diffusers import DiffusionPipeline

    import_success = True
except ImportError:
    logging.error(
        "Failed to import diffusers. Please install diffusers using 'pip install diffusers'"
    )
    import_success = False


class IMG:
    def __init__(self, model="stabilityai/sdxl-turbo"):
        global import_success
        if import_success:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = DiffusionPipeline.from_pretrained(
                model,
                cache_dir="models",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device)
            self.pipe = pipe
            self.pipe.safety_checker = None
        else:
            self.pipe = None

    def generate(
        self,
        prompt,
        negative_prompt="low resolution, grainy, distorted",
        num_inference_steps=1,
        guidance_scale=0.0,
        local_uri=None,
    ):
        new_file_name = f"outputs/{uuid.uuid4()}.png"
        if self.pipe:
            new_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            new_image.save(new_file_name)
            if local_uri:
                return f"{local_uri}/{new_file_name}"
            return new_image
        else:
            return None
