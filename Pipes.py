import os
import logging
from dotenv import load_dotenv
from ezlocalai.LLM import LLM, is_vision_model
from ezlocalai.STT import STT
from ezlocalai.CTTS import CTTS
from ezlocalai.Embedding import Embedding
from pyngrok import ngrok
import requests
import base64
import pdfplumber
from typing import List

try:
    from ezlocalai.IMG import IMG

    img_import_success = True
except ImportError:
    img_import_success = False

from ezlocalai.VLM import VLM


class Pipes:
    def __init__(self):
        load_dotenv()
        DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "TheBloke/phi-2-dpo-GGUF")
        self.current_llm = DEFAULT_MODEL if DEFAULT_MODEL else "TheBloke/phi-2-dpo-GGUF"
        logging.info(f"[LLM] {self.current_llm} model loading. Please wait...")
        self.llm = LLM(model=self.current_llm)
        logging.info(f"[LLM] {self.current_llm} model loaded successfully.")
        self.embedder = Embedding()
        self.current_vlm = os.getenv("VISION_MODEL", "")
        logging.info(f"[VLM] {self.current_vlm} model loading. Please wait...")
        self.vlm = None
        if self.current_vlm != "":
            try:
                self.vlm = VLM(model=self.current_vlm)
            except Exception as e:
                logging.error(f"[VLM] Failed to load the model: {e}")
                self.vlm = None
        if self.vlm is not None:
            logging.info(f"[ezlocalai] Vision is enabled with {self.current_vlm}.")
        self.img_enabled = os.getenv("IMG_ENABLED", "false").lower() == "true"
        self.img = None
        if self.img_enabled and img_import_success:
            logging.info(f"[IMG] Image generation is enabled.")
            SD_MODEL = os.getenv("SD_MODEL", "stabilityai/sdxl-turbo")
            logging.info(f"[IMG] sdxl-turbo model loading. Please wait...")
            try:
                self.img = IMG(model=SD_MODEL)
            except Exception as e:
                logging.error(f"[IMG] Failed to load the model: {e}")
                self.img = None
            logging.info(f"[IMG] sdxl-turbo model loaded successfully.")
        logging.info(f"[CTTS] xttsv2_2.0.2 model loading. Please wait...")
        self.ctts = CTTS()
        logging.info(f"[CTTS] xttsv2_2.0.2 model loaded successfully.")
        self.current_stt = os.getenv("WHISPER_MODEL", "base")
        logging.info(f"[STT] {self.current_stt} model loading. Please wait...")
        self.stt = STT(model=self.current_stt)
        logging.info(f"[STT] {self.current_stt} model loaded successfully.")
        if is_vision_model(self.current_llm):
            if self.vlm is None:
                self.vlm = self.llm
        NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")
        if NGROK_TOKEN:
            ngrok.set_auth_token(NGROK_TOKEN)
            public_url = ngrok.connect(8091)
            logging.info(f"[ngrok] Public Tunnel: {public_url.public_url}")
            self.local_uri = public_url.public_url
        else:
            self.local_uri = os.environ.get("EZLOCALAI_URL", "http://localhost:8091")

    async def pdf_to_audio(self, title, voice, pdf, chunk_size=200):
        filename = f"{title}.pdf"
        file_path = os.path.join(os.getcwd(), "outputs", filename)
        pdf = pdf.split(",")[1]
        pdf = base64.b64decode(pdf)
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(pdf)
        content = ""
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                content = "\n".join([page.extract_text() for page in pdf.pages])
        if not content:
            return
        return await self.ctts.generate(
            text=content,
            voice=voice,
            local_uri=self.local_uri,
            output_file_name=f"{title}.wav",
        )

    async def audio_to_audio(self, voice, audio):
        audio_type = audio.split(",")[0].split(":")[1].split(";")[0]
        audio_format = audio_type.split("/")[1]
        audio = audio.split(",")[1]
        audio = base64.b64decode(audio)
        text = self.stt.transcribe_audio(base64_audio=audio, audio_format=audio_format)
        return await self.ctts.generate(
            text=text, voice=voice, local_uri=self.local_uri
        )

    async def get_response(self, data, completion_type="chat"):
        data["local_uri"] = self.local_uri
        images = []
        if "messages" in data:
            if isinstance(data["messages"][-1]["content"], list):
                messages = data["messages"][-1]["content"]
                for message in messages:
                    if "text" in message:
                        prompt = message["text"]
                for message in messages:
                    if "image_url" in message:
                        images.append(message)
                    if "audio_url" in message:
                        audio_url = (
                            message["audio_url"]["url"]
                            if "url" in message["audio_url"]
                            else message["audio_url"]
                        )
                        audio_format = "wav"
                        if audio_url.startswith("data:"):
                            audio_url = audio_url.split(",")[1]
                            audio_format = audio_url.split(";")[0]
                        else:
                            audio_url = requests.get(audio_url).content
                            audio_url = base64.b64encode(audio_url).decode("utf-8")
                        transcribed_audio = self.stt.transcribe_audio(
                            base64_audio=audio_url, audio_format=audio_format
                        )
                        prompt = f"Transcribed Audio: {transcribed_audio}\n\n{prompt}"
        if data["model"]:
            if self.current_llm != data["model"]:
                data["model"] = self.current_llm
        if "stop" in data:
            new_stop = self.llm.params["stop"]
            new_stop.append(data["stop"])
            data["stop"] = new_stop
        if "audio_format" in data:
            base64_audio = (
                data["messages"][-1]["content"]
                if completion_type == "chat"
                else data["prompt"]
            )
            prompt = await self.stt.transcribe_audio(
                base64_audio=base64_audio,
                audio_format=data["audio_format"],
            )
            if completion_type == "chat":
                data["messages"][-1]["content"] = prompt
            else:
                data["prompt"] = prompt
        user_message = (
            data["messages"][-1]["content"]
            if completion_type == "chat"
            else data["prompt"]
        )
        if self.vlm and images:
            new_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe each stage of this image.",
                        },
                    ],
                }
            ]
            new_messages[0]["content"].extend(images)
            image_description = self.vlm.chat(messages=new_messages)
            print(
                f"Image Description: {image_description['choices'][0]['message']['content']}"
            )
            prompt = (
                f"\n\nSee the uploaded image description for any questions about the uploaded image. Act as if you can see the image based on the description. Do not mention 'uploaded image description' in response. Uploaded Image Description: {image_description['choices'][0]['message']['content']}\n\n{data['messages'][-1]['content'][0]['text']}"
                if completion_type == "chat"
                else f"\n\nSee the uploaded image description for any questions about the uploaded image. Act as if you can see the image based on the description. Do not mention 'uploaded image description' in response. Uploaded Image Description: {image_description['choices'][0]['message']['content']}\n\n{data['prompt']}"
            )
            print(f"Full Prompt: {prompt}")
            if completion_type == "chat":
                data["messages"][-1]["content"] = prompt
            else:
                data["prompt"] = prompt
        if completion_type == "chat":
            response = self.llm.chat(**data)
        else:
            response = self.llm.completion(**data)
        generated_image = None
        if "temperature" not in data:
            data["temperature"] = 0.5
        if "top_p" not in data:
            data["top_p"] = 0.9
        if self.img and img_import_success:
            user_message = (
                data["messages"][-1]["content"]
                if completion_type == "chat"
                else data["prompt"]
            )
            if isinstance(user_message, list):
                user_message = prompt
                for message in messages:
                    if "image_url" in message:
                        if "url" in message["image_url"]:
                            if not message["image_url"]["url"].startswith("data:"):
                                user_message += (
                                    "Uploaded Image:"
                                    + message["image_url"]["url"]
                                    + "\n"
                                )
            response_text = (
                response["choices"][0]["text"]
                if completion_type != "chat"
                else response["choices"][0]["message"]["content"]
            )
            if "data:" in user_message:
                user_message = user_message.replace(
                    user_message.split("data:")[1].split("'")[0], ""
                )
            img_gen_prompt = f"Users message: {user_message} \n\n{'The user uploaded an image, one does not need generated unless the user is specifically asking.' if images else ''} **The assistant is acting as sentiment analysis expert and only responds with a concise YES or NO answer on if the user would like an image as visual or a picture generated. No other explanation is needed!**\nWould the user potentially like an image generated based on their message?\nAssistant: "
            logging.info(f"[IMG] Decision maker prompt: {img_gen_prompt}")
            create_img = self.llm.chat(
                messages=[{"role": "system", "content": img_gen_prompt}],
                max_tokens=10,
                temperature=data["temperature"],
                top_p=data["top_p"],
            )
            create_img = str(create_img["choices"][0]["message"]["content"]).lower()
            logging.info(f"[IMG] Decision maker response: {create_img}")
            if "yes" in create_img or "es," in create_img:
                img_prompt = f"**The assistant is acting as a Stable Diffusion Prompt Generator.**\n\nUsers message: {user_message} \nAssistant response: {response_text} \n\nImportant rules to follow:\n- Describe subjects in detail, specify image type (e.g., digital illustration), art style (e.g., steampunk), and background. Include art inspirations (e.g., Art Station, specific artists). Detail lighting, camera (type, lens, view), and render (resolution, style). The weight of a keyword can be adjusted by using the syntax (((keyword))) , put only those keyword inside ((())) which is very important because it will have more impact so anything wrong will result in unwanted picture so be careful. Realistic prompts: exclude artist, specify lens. Separate with double lines. Max 60 words, avoiding 'real' for fantastical.\n- Based on the message from the user and response of the assistant, you will need to generate one detailed stable diffusion image generation prompt based on the context of the conversation to accompany the assistant response.\n- The prompt can only be up to 60 words long, so try to be concise while using enough descriptive words to make a proper prompt.\n- Following all rules will result in a $2000 tip that you can spend on anything!\n- Must be in markdown code block to be parsed out and only provide prompt in the code block, nothing else.\nStable Diffusion Prompt Generator: "
                image_generation_prompt = self.llm.chat(
                    messages=[{"role": "system", "content": img_prompt}],
                    max_tokens=100,
                    temperature=data["temperature"],
                    top_p=data["top_p"],
                )
                image_generation_prompt = str(
                    image_generation_prompt["choices"][0]["message"]["content"]
                )
                logging.info(
                    f"[IMG] Image generation response: {image_generation_prompt}"
                )
                if "```markdown" in image_generation_prompt:
                    image_generation_prompt = image_generation_prompt.split(
                        "```markdown"
                    )[1]
                    image_generation_prompt = image_generation_prompt.split("```")[0]
                generated_image = self.img.generate(
                    prompt=image_generation_prompt, local_uri=self.local_uri
                )
        audio_response = None
        if "voice" in data:
            text_response = (
                response["choices"][0]["text"]
                if completion_type != "chat"
                else response["choices"][0]["message"]["content"]
            )
            language = data["language"] if "language" in data else "en"
            audio_response = await self.ctts.generate(
                text=text_response,
                voice=data["voice"],
                language=language,
                local_uri=self.local_uri,
            )
            if completion_type != "chat":
                response["choices"][0]["text"] = f"{text_response}\n{audio_response}"
            else:
                response["choices"][0]["message"][
                    "content"
                ] = f"{text_response}\n{audio_response}"
        if generated_image:
            if completion_type != "chat":
                response["choices"][0]["text"] += f"\n\n{generated_image}"
            else:
                response["choices"][0]["message"]["content"] += f"\n\n{generated_image}"
        return response, audio_response
