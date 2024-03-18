try:
    from deepseek_vl.models import VLChatProcessor
except ImportError:
    VLChatProcessor = None
import requests
import torch
from transformers import AutoModelForCausalLM
from typing import List, Dict
import PIL.Image


class VLM:
    def __init__(self, model="deepseek-ai/deepseek-vl-1.3b-chat"):
        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(model)
            self.tokenizer = self.vl_chat_processor.tokenizer
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                model, trust_remote_code=True
            )
            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        except:
            self.vl_chat_processor = None
            self.tokenizer = None
            self.vl_gpt = None

    def load_pil_images(
        self, conversations: List[Dict[str, str]]
    ) -> List[PIL.Image.Image]:
        pil_images = []

        for message in conversations:
            if "images" not in message:
                continue

            for image_path in message["images"]:
                pil_img = PIL.Image.open(image_path)
                pil_img = pil_img.convert("RGB")
                pil_images.append(pil_img)

        return pil_images

    def prompt_with_image(self, prompt, image_url: str) -> str:
        image = requests.get(image_url).content
        image_path = "./temp.png"
        with open(image_path, "wb") as f:
            f.write(image)
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image_path],
            },
            {"role": "Assistant", "content": ""},
        ]
        pil_images = self.load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        return answer

    def describe_image(self, image_url):
        return self.prompt_with_image(
            prompt="Describe each stage of this image.", image_url=image_url
        )
