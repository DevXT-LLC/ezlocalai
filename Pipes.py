import os
import logging
from dotenv import load_dotenv
from ezlocalai.LLM import LLM, is_vision_model
from ezlocalai.STT import STT
from ezlocalai.CTTS import CTTS
from ezlocalai.Embedding import Embedding
from ezlocalai.Helpers import chunk_content_by_tokens
from pydub import AudioSegment
from datetime import datetime
from Globals import getenv
from pyngrok import ngrok
import requests
import base64
import pdfplumber
import zipfile
import docx2txt
import pandas as pd
import random
import json
import io

try:
    from ezlocalai.IMG import IMG

    img_import_success = True
except ImportError:
    img_import_success = False

from ezlocalai.VLM import VLM


async def file_to_text(file_path: str = ""):
    """
    Learn from a file

    Args:
        file_path (str, optional): Path to the file. Defaults to "".

    Returns:
        str: Response from the agent
    """
    file_content = ""
    file_name = os.path.basename(file_path)
    logging.info(f"File path: {file_path}")
    file_type = file_name.split(".")[-1]
    if file_type == "pdf":
        with pdfplumber.open(file_path) as pdf:
            content = "\n".join([page.extract_text() for page in pdf.pages])
            file_content += content
    elif file_path.endswith(".zip"):
        extracted_zip_folder_name = f"extracted_{file_name.replace('.zip', '_zip')}"
        new_folder = os.path.join(os.path.dirname(file_path), extracted_zip_folder_name)
        file_content += f"Content from the zip file uploaded named `{file_name}`:\n"
        with zipfile.ZipFile(file_path, "r") as zipObj:
            zipObj.extractall(path=new_folder)
        # Iterate over every file that was extracted including subdirectories
        for root, dirs, files in os.walk(new_folder):
            for name in files:
                file_content += f"Content from file uploaded named `{name}`:\n"
                file_content += await file_to_text(file_path=os.path.join(root, name))
        return file_content
    elif file_path.endswith(".doc") or file_path.endswith(".docx"):
        file_content = docx2txt.process(file_path)
    elif file_type == "csv":
        with open(file_path, "r") as f:
            file_content = f.read()
    elif file_type == "xlsx" or file_type == "xls":
        xl = pd.ExcelFile(file_path)
        if len(xl.sheet_names) > 1:
            sheet_count = len(xl.sheet_names)
            for i, sheet_name in enumerate(xl.sheet_names, 1):
                df = xl.parse(sheet_name)
                csv_file_path = file_path.replace(f".{file_type}", f"_{i}.csv")
                df.to_csv(csv_file_path, index=False)
        else:
            df = pd.read_excel(file_path)
            csv_file_path = file_path.replace(f".{file_type}", ".csv")
            df.to_csv(csv_file_path, index=False)
        with open(csv_file_path, "r") as f:
            file_content = f.read()
    else:
        with open(file_path, "r") as f:
            file_content = f.read()
    return file_content


class Pipes:
    def __init__(self):
        load_dotenv()
        global img_import_success
        self.current_llm = getenv("DEFAULT_MODEL")
        self.current_vlm = getenv("VISION_MODEL")
        self.llm = None
        self.vlm = None
        self.ctts = None
        self.stt = None
        self.embedder = None
        if self.current_llm.lower() != "none":
            logging.info(f"[LLM] {self.current_llm} model loading. Please wait...")
            self.llm = LLM(model=self.current_llm)
            logging.info(f"[LLM] {self.current_llm} model loaded successfully.")
        if getenv("EMBEDDING_ENABLED").lower() == "true":
            self.embedder = Embedding()
        if self.current_vlm != "":
            logging.info(f"[VLM] {self.current_vlm} model loading. Please wait...")
            try:
                self.vlm = VLM(model=self.current_vlm)
            except Exception as e:
                logging.error(f"[VLM] Failed to load the model: {e}")
                self.vlm = None
            if self.vlm is not None:
                logging.info(f"[ezlocalai] Vision is enabled with {self.current_vlm}.")
        if getenv("TTS_ENABLED").lower() == "true":
            logging.info(f"[CTTS] xttsv2_2.0.2 model loading. Please wait...")
            self.ctts = CTTS()
            logging.info(f"[CTTS] xttsv2_2.0.2 model loaded successfully.")
        if getenv("STT_ENABLED").lower() == "true":
            self.current_stt = getenv("WHISPER_MODEL")
            logging.info(f"[STT] {self.current_stt} model loading. Please wait...")
            self.stt = STT(model=self.current_stt)
            logging.info(f"[STT] {self.current_stt} model loaded successfully.")
        if is_vision_model(self.current_llm):
            if self.vlm is None:
                self.vlm = self.llm
        if self.current_llm == "none" and self.vlm is not None:
            self.llm = self.vlm
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
        return response, audio_response

    async def create_audiobook(
        self,
        content,
        voice,
        language="en",
    ):
        string_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file_name = f"audiobook_{string_timestamp}"
        # Step 1: Chunk the book content into paragraphs
        paragraphs = chunk_content_by_tokens(content)

        # Step 2: Extract characters, their lines, genders, and maintain order
        characters = {}
        ordered_content = []

        def find_similar_character(name):
            # Check for exact match first
            if name in characters:
                return name
            # Check for case-insensitive match
            lower_name = name.lower()
            for char in characters:
                if char.lower() == lower_name:
                    return char
            # Check for partial matches (e.g., "Mr. Smith" vs "Smith")
            for char in characters:
                if name in char or char in name:
                    return char
            return None

        for paragraph in paragraphs:
            # Inject a list of characters we know so far.
            prompt = f"""## Characters we know so far:
{json.dumps(characters, indent=4)}

## Paragraph
{paragraph}

## System
Analyze the text in the paragraph and extract:
1. All character names and their genders (male, female, or unknown.) Use best judgement based on hisortical uses of a name to determine gender. Attempt to normalize character names to match existing characters if possible.
2. Lines spoken by each character
3. Narrator lines (not spoken by any character)

Provide the result in JSON format:
{{
    "characters": [
        {{"name": "character1", "gender": "male/female/unknown"}},
        {{"name": "character2", "gender": "male/female/unknown"}},
        ...
    ],
    "content": [
        {{"type": "narrator", "text": "narrator line"}},
        {{"type": "character", "name": "character1", "text": "character1 line"}},
        {{"type": "narrator", "text": "narrator line"}},
        {{"type": "character", "name": "character2", "text": "character2 line"}},
        ...
    ]
}}
Ensure the content array preserves the original order of narration and dialogue."""

            response = await self.llm.completion(prompt=prompt)
            result_text = response["choices"][0]["text"]

            # Strip out code block markers if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            try:
                result = json.loads(result_text)
                for char in result.get("characters", []):
                    similar_char = find_similar_character(char["name"])
                    if similar_char:
                        # Use the existing character name
                        char["name"] = similar_char
                    else:
                        # Add new character
                        characters[char["name"]] = char["gender"]

                # Update content with potentially merged character names
                for item in result.get("content", []):
                    if item["type"] == "character":
                        similar_char = find_similar_character(item["name"])
                        if similar_char:
                            item["name"] = similar_char
                    ordered_content.append(item)

            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON from LLM response: {result_text}")
                continue

        # Step 3: Translate the content if necessary
        if language != "en":
            translated_content = []
            for item in ordered_content:
                translation_prompt = f"""## Original text:{item['text']}\n\n## System\nTranslate the original text to {language}.\nReturn only the translated text without any additional commentary."""
                translation_response = await self.llm.completion(
                    prompt=translation_prompt
                )
                translated_text = translation_response["choices"][0]["text"].strip()
                translated_item = item.copy()
                translated_item["text"] = translated_text
                translated_content.append(translated_item)
            ordered_content = translated_content

        # Step 4: Assign voices to characters based on gender
        character_voices = {}
        male_voices = [f"male-{i}" for i in range(1, 101)]
        female_voices = [f"female-{i}" for i in range(1, 101)]
        unknown_voices = male_voices + female_voices
        random.shuffle(male_voices)
        random.shuffle(female_voices)
        random.shuffle(unknown_voices)

        for character, gender in characters.items():
            if gender == "male" and male_voices:
                character_voices[character] = male_voices.pop()
            elif gender == "female" and female_voices:
                character_voices[character] = female_voices.pop()
            elif unknown_voices:
                character_voices[character] = unknown_voices.pop()
            else:
                logging.warning(
                    f"Ran out of voices. Reusing voices for character: {character}"
                )
                character_voices[character] = random.choice(male_voices + female_voices)

        # Step 5: Generate audio for each item in ordered_content
        audio_segments = []
        text_output = []

        for item in ordered_content:
            if item["type"] == "narrator":
                try:
                    audio = await self.ctts.generate(
                        text=item["text"], voice=voice, language=language
                    )
                    audio_segments.append(base64.b64decode(audio))
                    text_output.append(f"Narrator: {item['text']}")
                except Exception as e:
                    logging.error(
                        f"Failed to generate audio for narrator text: {item['text'][:50]}... Error: {str(e)}"
                    )
            elif item["type"] == "character":
                character_voice = character_voices.get(item["name"], voice)
                try:
                    audio = await self.ctts.generate(
                        text=item["text"], voice=character_voice, language=language
                    )
                    audio_segments.append(base64.b64decode(audio))
                    text_output.append(f"{item['name']}: {item['text']}")
                except Exception as e:
                    logging.error(
                        f"Failed to generate audio for character {item['name']}: {item['text'][:50]}... Error: {str(e)}"
                    )

        # Step 6: Combine all audio segments
        combined_audio = AudioSegment.empty()
        for audio_data in audio_segments:
            try:
                audio = AudioSegment.from_wav(io.BytesIO(audio_data))
                combined_audio += audio
                combined_audio += AudioSegment.silent(
                    duration=500
                )  # 0.5 second pause between segments
            except Exception as e:
                logging.error(f"Failed to process audio segment. Error: {str(e)}")
        outputs = os.path.join(os.getcwd(), "outputs")
        # Step 7: Export the final audiobook
        audio_output_path = os.path.join(outputs, f"{output_file_name}.mp3")
        combined_audio.export(audio_output_path, format="mp3")

        # Step 8: Save the text output
        text_output_path = os.path.join(outputs, f"{output_file_name}.txt")
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(text_output))
        return {
            "audio_file": f"{self.local_uri}/outputs/{output_file_name}.mp3",
            "text_file": f"{self.local_uri}/outputs/{output_file_name}.txt",
            "character_voices": character_voices,
        }
