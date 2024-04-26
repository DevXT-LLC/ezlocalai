try:
    from deepseek_vl.models import VLChatProcessor
except:
    VLChatProcessor = None
from transformers import AutoModelForCausalLM
from datetime import datetime
import requests
import torch
import PIL.Image
import uuid
import os
import base64
from ezlocalai.Helpers import get_tokens


class VLM:
    def __init__(self, model="deepseek-ai/deepseek-vl-1.3b-chat"):
        self.model = model.split("/")[-1]
        self.params = {}
        os.makedirs(os.path.join(os.getcwd(), "outputs"), exist_ok=True)
        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(model)
            self.tokenizer = self.vl_chat_processor.tokenizer
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                cache_dir=os.path.join(os.getcwd(), "models"),
            )
            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        except:
            self.vl_chat_processor = None
            self.tokenizer = None
            self.vl_gpt = None

    def chat(self, messages, **kwargs):
        pil_images = []
        images = []
        prompt = ""
        for message in messages:
            if isinstance(message["content"], str):
                role = message["role"] if "role" in message else "User"
                if role.lower() == "user":
                    prompt += f"{message['content']}\n\n"
                if role.lower() == "system":
                    prompt = f"System: {message['content']}\n\nUser: {prompt}"
            if isinstance(message["content"], list):
                for msg in message["content"]:
                    if "text" in msg:
                        role = message["role"] if "role" in message else "User"
                        if role.lower() == "user":
                            prompt += f"{msg['text']}\n\n"
                    if "image_url" in msg:
                        url = (
                            msg["image_url"]["url"]
                            if "url" in msg["image_url"]
                            else msg["image_url"]
                        )
                        image_path = f"./outputs/{uuid.uuid4().hex}.jpg"
                        if url.startswith("http"):
                            image = requests.get(url).content
                        else:
                            file_type = url.split(",")[0].split("/")[1].split(";")[0]
                            if file_type == "jpeg":
                                file_type = "jpg"
                            image_path = f"./outputs/{uuid.uuid4().hex}.{file_type}"
                            image = base64.b64decode(url.split(",")[1])
                        with open(image_path, "wb") as f:
                            f.write(image)
                        images.append(image_path)
                        pil_img = PIL.Image.open(image_path)
                        pil_img = pil_img.convert("RGB")
                        pil_images.append(pil_img)
        if len(images) > 0:
            for image in images:
                prompt = f"<image_placeholder> {prompt}"
        conversation = [
            {"role": "User", "content": prompt, "images": images},
            {"role": "Assistant", "content": ""},
        ]
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
            max_new_tokens=(
                1024 if "max_tokens" not in kwargs else int(kwargs["max_tokens"])
            ),
            do_sample=False,
            use_cache=True,
        )
        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        completion_tokens = get_tokens(answer)
        prompt_tokens = get_tokens(
            " ".join([message["content"] for message in conversation])
        )
        total_tokens = completion_tokens + prompt_tokens
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": answer, "role": "assistant"},
                    "logprobs": None,
                }
            ],
            "created": datetime.now().isoformat(),
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "model": self.model,
            "object": "chat.completion",
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            },
        }
        return data

    def completion(self, prompt, **kwargs):
        messages = [
            {"role": "User", "content": prompt},
        ]
        completion = self.chat(
            messages=messages,
            max_tokens=kwargs["max_tokens"] if "max_tokens" in kwargs else 1024,
        )
        data = {
            "choices": [
                {
                    "finish_reason": "length",
                    "index": 0,
                    "logprobs": None,
                    "text": completion["choices"][0]["message"]["content"],
                }
            ],
            "created": datetime.now().isoformat(),
            "id": f"cmpl-{uuid.uuid4().hex}",
            "model": self.model,
            "object": "text_completion",
            "usage": {
                "completion_tokens": completion["usage"]["completion_tokens"],
                "prompt_tokens": completion["usage"]["prompt_tokens"],
                "total_tokens": completion["usage"]["total_tokens"],
            },
        }
        return data

    def describe_image(self, image_url):
        messages = [
            {
                "role": "User",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "Describe each stage of this image.",
                    },
                ],
            },
        ]
        response = self.chat(
            messages=messages,
        )
        return response["choices"][0]["message"]["content"]

    def models(self):
        return [
            "deepseek-ai/deepseek-vl-1.3b-chat",
            "deepseek-ai/deepseek-vl-7b-chat",
        ]
