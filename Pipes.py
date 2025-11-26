import os
import logging
from dotenv import load_dotenv
from ezlocalai.LLM import LLM
from ezlocalai.STT import STT
from ezlocalai.CTTS import CTTS
from ezlocalai.Embedding import Embedding
from pyngrok import ngrok
import requests
import base64
import pdfplumber
import json
from Globals import getenv

try:
    from ezlocalai.IMG import IMG

    img_import_success = True
except ImportError:
    img_import_success = False


class Pipes:
    def __init__(self):
        load_dotenv()
        global img_import_success
        self.current_llm = getenv("DEFAULT_MODEL")
        self.llm = None
        self.ctts = None
        self.stt = None
        self.embedder = None
        if self.current_llm.lower() != "none":
            logging.info(f"[LLM] {self.current_llm} model loading. Please wait...")
            self.llm = LLM(model=self.current_llm)
            logging.info(f"[LLM] {self.current_llm} model loaded successfully.")
            if self.llm.is_vision:
                logging.info(f"[LLM] Vision capability enabled for {self.current_llm}.")
        if getenv("EMBEDDING_ENABLED").lower() == "true":
            self.embedder = Embedding()
        if getenv("TTS_ENABLED").lower() == "true":
            logging.info(f"[CTTS] xttsv2_2.0.2 model loading. Please wait...")
            self.ctts = CTTS()
            logging.info(f"[CTTS] xttsv2_2.0.2 model loaded successfully.")
        if getenv("STT_ENABLED").lower() == "true":
            self.current_stt = getenv("WHISPER_MODEL")
            logging.info(f"[STT] {self.current_stt} model loading. Please wait...")
            self.stt = STT(model=self.current_stt)
            logging.info(f"[STT] {self.current_stt} model loaded successfully.")
        NGROK_TOKEN = getenv("NGROK_TOKEN")
        if NGROK_TOKEN:
            ngrok.set_auth_token(NGROK_TOKEN)
            public_url = ngrok.connect(8091)
            logging.info(f"[ngrok] Public Tunnel: {public_url.public_url}")
            self.local_uri = public_url.public_url
        else:
            self.local_uri = getenv("EZLOCALAI_URL")
        self.img_enabled = getenv("IMG_ENABLED").lower() == "true"
        self.img = None
        if img_import_success:
            logging.info(f"[IMG] Image generation is enabled.")
            SD_MODEL = getenv("SD_MODEL")  # stabilityai/sdxl-turbo
            if SD_MODEL:
                logging.info(f"[IMG] {SD_MODEL} model loading. Please wait...")
                img_device = getenv("IMG_DEVICE")
                try:
                    self.img = IMG(
                        model=SD_MODEL, local_uri=self.local_uri, device=img_device
                    )
                except Exception as e:
                    logging.error(f"[IMG] Failed to load the model: {e}")
                    self.img = None
            logging.info(f"[IMG] {SD_MODEL} model loaded successfully.")

    async def fallback_inference(self, messages):
        fallback_server = getenv("FALLBACK_SERVER")
        fallback_model = getenv("FALLBACK_MODEL")
        fallback_api_key = getenv("FALLBACK_API_KEY")
        if fallback_server == "":
            return "Unable to process request. Please try again later."
        from openai import Client

        client = Client(api_key=fallback_api_key, base_url=fallback_server)
        response = client.chat.completions.create(
            model=fallback_model, messages=messages
        )
        return response.choices[0].message.content

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

    async def generate_image(self, prompt, response_format="url", size="512x512"):
        if self.img:
            self.img.local_uri = self.local_uri if response_format == "url" else None
            new_image = self.img.generate(
                prompt=prompt,
                size=size,
            )
            self.img.local_uri = self.local_uri
            return new_image
        return ""

    async def get_response(self, data, completion_type="chat"):
        data["local_uri"] = self.local_uri
        images = []
        if "messages" in data:
            # Process messages to extract images and handle content types
            for i, message in enumerate(data["messages"]):
                if isinstance(message.get("content"), list):
                    # Extract text content and images from list format
                    text_content = ""
                    message_images = []
                    for content_item in message["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "text":
                                text_content += content_item.get("text", "")
                            elif "image_url" in content_item:
                                message_images.append(content_item)
                            elif "audio_url" in content_item:
                                audio_url = (
                                    content_item["audio_url"]["url"]
                                    if "url" in content_item["audio_url"]
                                    else content_item["audio_url"]
                                )
                                audio_format = "wav"
                                if audio_url.startswith("data:"):
                                    audio_url = audio_url.split(",")[1]
                                    audio_format = audio_url.split(";")[0]
                                else:
                                    audio_url = requests.get(audio_url).content
                                    audio_url = base64.b64encode(audio_url).decode(
                                        "utf-8"
                                    )
                                transcribed_audio = self.stt.transcribe_audio(
                                    base64_audio=audio_url, audio_format=audio_format
                                )
                                text_content = f"Transcribed Audio: {transcribed_audio}\n\n{text_content}"
                        elif isinstance(content_item, str):
                            text_content += content_item

                    # Collect images for later processing
                    if message_images:
                        images.extend(message_images)

                    # For non-vision models or non-user messages, convert to string
                    # For vision models with the last user message, we'll handle this later
                    if not (
                        self.llm
                        and self.llm.is_vision
                        and message_images
                        and i == len(data["messages"]) - 1
                    ):
                        data["messages"][i]["content"] = text_content

            # Legacy handling for the old format (keeping for backward compatibility)
            # Skip if we already collected images in the modern format
            if not images and isinstance(data["messages"][-1]["content"], list):
                messages = data["messages"][-1]["content"]
                prompt = ""
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
                # Convert list content back to string for LLM compatibility
                data["messages"][-1]["content"] = prompt
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
        # Handle images with vision-capable LLM
        if self.llm and self.llm.is_vision and images:
            # xllamacpp expects images in base64 data URL format (PNG or JPEG)
            # Convert any remote URLs to base64 data URLs, and convert WebP/other formats to PNG
            from PIL import Image as PILImage
            from io import BytesIO

            processed_images = []
            for img in images:
                if "image_url" in img:
                    img_url = (
                        img["image_url"].get("url", "")
                        if isinstance(img["image_url"], dict)
                        else img["image_url"]
                    )
                    if img_url and not img_url.startswith("data:"):
                        # Fetch remote image and convert to base64
                        try:
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            }
                            img_response = requests.get(
                                img_url, timeout=30, headers=headers
                            )
                            img_response.raise_for_status()
                            content_type = img_response.headers.get(
                                "Content-Type", "image/jpeg"
                            )

                            # llama.cpp mmproj only supports certain formats (not WebP)
                            # Convert any non-standard format to PNG
                            if content_type in [
                                "image/webp",
                                "image/gif",
                                "image/bmp",
                                "image/tiff",
                                "image/avif",
                            ]:
                                try:
                                    pil_img = PILImage.open(
                                        BytesIO(img_response.content)
                                    )
                                    # Convert to RGB if necessary (for RGBA or palette images)
                                    if pil_img.mode in ("RGBA", "P", "LA"):
                                        pil_img = pil_img.convert("RGB")
                                    buffer = BytesIO()
                                    pil_img.save(buffer, format="PNG")
                                    img_bytes = buffer.getvalue()
                                    content_type = "image/png"
                                    logging.info(
                                        f"[Vision] Converted {img_response.headers.get('Content-Type', 'unknown')} to PNG"
                                    )
                                except Exception as conv_err:
                                    logging.error(
                                        f"[Vision] Failed to convert image: {conv_err}"
                                    )
                                    img_bytes = img_response.content
                            else:
                                img_bytes = img_response.content

                            if not content_type.startswith("image/"):
                                content_type = "image/jpeg"
                            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                            processed_images.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{img_base64}"
                                    },
                                }
                            )
                            logging.info(
                                f"[Vision] Converted remote image to base64 ({len(img_base64)} chars)"
                            )
                        except Exception as e:
                            logging.error(
                                f"[Vision] Failed to fetch remote image {img_url}: {e}"
                            )
                            continue
                    else:
                        # Already a data URL - check if it needs conversion
                        if img_url.startswith("data:image/webp") or img_url.startswith(
                            "data:image/gif"
                        ):
                            try:
                                # Extract base64 data and convert
                                header, encoded = img_url.split(",", 1)
                                img_bytes = base64.b64decode(encoded)
                                pil_img = PILImage.open(BytesIO(img_bytes))
                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    pil_img = pil_img.convert("RGB")
                                buffer = BytesIO()
                                pil_img.save(buffer, format="PNG")
                                img_base64 = base64.b64encode(buffer.getvalue()).decode(
                                    "utf-8"
                                )
                                processed_images.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        },
                                    }
                                )
                                logging.info(
                                    f"[Vision] Converted data URL WebP/GIF to PNG"
                                )
                            except Exception as conv_err:
                                logging.error(
                                    f"[Vision] Failed to convert data URL: {conv_err}"
                                )
                                processed_images.append(img)
                        else:
                            processed_images.append(img)

            if completion_type == "chat":
                # Build proper multimodal message with text + images
                user_text = data["messages"][-1]["content"]
                if isinstance(user_text, list):
                    # Extract text from list content format
                    user_text = " ".join(
                        [
                            item.get("text", "")
                            for item in user_text
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                    )

                if processed_images:
                    # Create message with text + images in xllamacpp expected format
                    multimodal_content = [{"type": "text", "text": user_text}]
                    multimodal_content.extend(processed_images)
                    data["messages"][-1]["content"] = multimodal_content
                    logging.info(
                        f"[Vision] Sending multimodal message with {len(processed_images)} image(s)"
                    )
                else:
                    # No images could be processed, fall back to text-only
                    data["messages"][-1]["content"] = user_text
                    logging.warning(
                        f"[Vision] No images could be processed, falling back to text-only"
                    )
        if completion_type == "chat":
            try:
                response = self.llm.chat(**data)
            except Exception as e:
                import traceback

                logging.error(f"[LLM] Chat completion failed: {e}")
                logging.error(f"[LLM] Full traceback: {traceback.format_exc()}")
                logging.error(f"[LLM] Data that caused failure: {data}")
                response = await self.fallback_inference(data["messages"])
        else:
            try:
                response = self.llm.completion(**data)
            except Exception as e:
                import traceback

                logging.error(f"[LLM] Completion failed: {e}")
                logging.error(f"[LLM] Full traceback: {traceback.format_exc()}")
                logging.error(f"[LLM] Data that caused failure: {data}")
                response = await self.fallback_inference(
                    [{"role": "user", "content": data.get("prompt", "")}]
                )
        generated_image = None
        if "temperature" not in data:
            data["temperature"] = 0.5
        if "top_p" not in data:
            data["top_p"] = 0.9
        if self.img_enabled and img_import_success and self.img:
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
            try:
                create_img = self.llm.chat(
                    messages=[{"role": "system", "content": img_gen_prompt}],
                    max_tokens=10,
                    temperature=data["temperature"],
                    top_p=data["top_p"],
                )
            except:
                create_img = await self.fallback_inference(
                    [{"role": "system", "content": img_gen_prompt}]
                )
            create_img = str(create_img["choices"][0]["message"]["content"]).lower()
            logging.info(f"[IMG] Decision maker response: {create_img}")
            if "yes" in create_img or "es," in create_img:
                img_prompt = f"**The assistant is acting as a Stable Diffusion Prompt Generator.**\n\nUsers message: {user_message} \nAssistant response: {response_text} \n\nImportant rules to follow:\n- Describe subjects in detail, specify image type (e.g., digital illustration), art style (e.g., steampunk), and background. Include art inspirations (e.g., Art Station, specific artists). Detail lighting, camera (type, lens, view), and render (resolution, style). The weight of a keyword can be adjusted by using the syntax (((keyword))) , put only those keyword inside ((())) which is very important because it will have more impact so anything wrong will result in unwanted picture so be careful. Realistic prompts: exclude artist, specify lens. Separate with double lines. Max 60 words, avoiding 'real' for fantastical.\n- Based on the message from the user and response of the assistant, you will need to generate one detailed stable diffusion image generation prompt based on the context of the conversation to accompany the assistant response.\n- The prompt can only be up to 60 words long, so try to be concise while using enough descriptive words to make a proper prompt.\n- Following all rules will result in a $2000 tip that you can spend on anything!\n- Must be in markdown code block to be parsed out and only provide prompt in the code block, nothing else.\nStable Diffusion Prompt Generator: "
                try:
                    image_generation_prompt = self.llm.chat(
                        messages=[{"role": "system", "content": img_prompt}],
                        max_tokens=100,
                        temperature=data["temperature"],
                        top_p=data["top_p"],
                    )
                except:
                    image_generation_prompt = await self.fallback_inference(
                        [{"role": "system", "content": img_prompt}]
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
                generated_image = self.img.generate(prompt=image_generation_prompt)
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
        # Only log JSON if response is not a generator (streaming mode)
        if not hasattr(response, "__next__"):
            logging.info(f"[ezlocalai] {json.dumps(response, indent=2)}")
        else:
            logging.info(f"[ezlocalai] Streaming response generated")
        return response, audio_response
