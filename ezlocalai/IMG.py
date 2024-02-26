import logging
import uuid
import torch

try:
    from diffusers import AutoPipelineForText2Image

    import_success = True
except ImportError:
    logging.error(
        "Failed to import diffusers. Please install diffusers using 'pip install diffusers'"
    )
    import_success = False

create_img_prompt = """Users message: {prompt} 
Assistant response: {response} 

**Act as a decision maker for creating stable diffusion images. Respond with a concise YES or NO answer on if it would make sense to generate an image based on the users message. No other explanation is needed!** Should an image be created to accompany the assistant response?
"""

img_prompt = """**Act as a STABLE DIFFUSION prompt generator.**

Users message: {prompt} 
Assistant response: {response}

Important rules to follow:
- Describe subjects in detail, specify image type (e.g., digital illustration), art style (e.g., steampunk), and background. Include art inspirations (e.g., Art Station, specific artists). Detail lighting, camera (type, lens, view), and render (resolution, style). The weight of a keyword can be adjusted by using the syntax (((keyword))) , put only those keyword inside ((())) which is very important because it will have more impact so anything wrong will result in unwanted picture so be careful. Realistic prompts: exclude artist, specify lens. Separate with double lines. Max 60 words, avoiding "real" for fantastical.
- Based on the message from the user and response of the assistant, you will need to generate one detailed stable diffusion image generation prompt based on the context of the conversation to accompany the assistant response.
- The prompt can only be up to 60 words long, so try to be concise while using enough descriptive words to make a proper prompt.
- Following all rules will result in a $2000 tip that you can spend on anything!
- Must be in python code block to be parsed out and only provide prompt in the code block, not any code. Do not write python code!
"""


class IMG:
    def __init__(self, model="stabilityai/sdxl-turbo"):
        global import_success
        if import_success:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = AutoPipelineForText2Image.from_pretrained(
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
