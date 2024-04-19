import streamlit as st
import openai
import requests
import base64
import os
import re
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv()
# Show logo but resize to the screen https://devxt.com/wp-content/uploads/2023/01/Logo-1024x316.png
st.image(
    "https://devxt.com/wp-content/uploads/2023/01/Logo-1024x316.png",
    width=300,
)

EZLOCALAI_SERVER = os.getenv("EZLOCALAI_URL", "http://localhost:8091")
EZLOCALAI_API_KEY = os.getenv("EZLOCALAI_API_KEY", "none")
DEFAULT_LLM = os.getenv("DEFAULT_MODEL", "TheBloke/phi-2-dpo-GGUF")
VISION_MODEL = os.getenv("VISION_MODEL", None)
SD_MODEL = os.getenv("SD_MODEL", "stabilityai/sdxl-turbo")
IMG_ENABLED = os.getenv("IMG_ENABLED", "false")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

if "/" in DEFAULT_LLM:
    link_to_model = f"https://huggingface.co/{DEFAULT_LLM}"
else:
    link_to_model = f"https://huggingface.co/models?search={DEFAULT_LLM}"
st.markdown(
    f"""
    [![GitHub](https://img.shields.io/badge/GitHub-ezLocalai-blue?logo=github&style=plastic)](https://github.com/DevXT-LLC/ezlocalai) [![Dockerhub](https://img.shields.io/badge/Docker-ezlocalai-blue?logo=docker&style=plastic)](https://hub.docker.com/r/joshxt/ezlocalai)
    ## ezLocal.ai Demo
    **Language model:** [{DEFAULT_LLM}]({link_to_model})
    """
)
if VISION_MODEL:
    st.markdown(
        f"""
        **Vision model:** [{VISION_MODEL}](https://huggingface.co/{VISION_MODEL})
        """
    )
if IMG_ENABLED.lower() == "true":
    st.markdown(
        f"""
        **Image Generation model:** [{SD_MODEL}](https://huggingface.co/{SD_MODEL})
        """
    )
openai.base_url = f"{EZLOCALAI_SERVER}/v1/"
openai.api_key = EZLOCALAI_API_KEY if EZLOCALAI_API_KEY else EZLOCALAI_SERVER
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"{EZLOCALAI_API_KEY}",
    "ngrok-skip-browser-warning": "true",
}
waiting_for_server = False


def get_voices():
    return ["Morgan_Freeman", "DukeNukem", "HAL9000"]
    # Commented for speed, but this works if we want to get the voices from the server
    """
    global EZLOCALAI_SERVER
    global HEADERS
    global waiting_for_server
    while True:
        try:
            voices = requests.get(
                f"{EZLOCALAI_SERVER}/v1/audio/voices", headers=HEADERS
            )
            voices = voices.json()["voices"]
            waiting_for_server = False
            return voices
        except:
            if waiting_for_server == False:
                st.spinner("Waiting for server to start...")
            waiting_for_server = True
            time.sleep(1)
    """


def display_content(content):
    global EZLOCALAI_SERVER
    global HEADERS
    outputs_url = f"{EZLOCALAI_SERVER}/outputs/"
    os.makedirs("outputs", exist_ok=True)
    if "http://localhost:8091/outputs/" in content:
        if outputs_url != "http://localhost:8091/outputs/":
            content = content.replace("http://localhost:8091/outputs/", outputs_url)
    if "<audio controls>" in content or " " not in content:
        try:
            audio_response = content.split("data:audio/wav;base64,")[1].split('" type')[
                0
            ]
        except:
            audio_response = content
        file_name = f"outputs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.wav"
        with open(file_name, "wb") as fh:
            fh.write(base64.b64decode(audio_response))
        st.audio(file_name, format="audio/wav", start_time=0)
    if outputs_url in content:
        urls = re.findall(f"{re.escape(outputs_url)}[^\"' ]+", content)
        urls = urls[0].split("\n\n")
        for url in urls:
            file_name = url.split("/")[-1]
            url = f"{outputs_url}{file_name}"
            data = requests.get(url, headers=HEADERS).content
            if url.endswith(".jpg") or url.endswith(".png"):
                content = content.replace(url, "")
                st.image(data, use_column_width=True)
            elif url.endswith(".mp4"):
                content = content.replace(url, "")
                st.audio(data, format="audio/mp4", start_time=0)
            elif url.endswith(".wav"):
                content = content.replace(url, "")
                st.audio(data, format="audio/wav", start_time=0)
    st.markdown(content, unsafe_allow_html=True)


show_advanced_options = st.checkbox(
    "Show Advanced Options", key="show_advanced_options"
)
if show_advanced_options:
    SYSTEM_MESSAGE = st.text_area(
        "System Prompt",
        "",
    )
    DEFAULT_MAX_TOKENS = st.number_input(
        "Max Output Tokens", min_value=10, max_value=300000, value=1024
    )
    DEFAULT_TEMPERATURE = st.number_input(
        "Temperature", min_value=0.0, max_value=1.0, value=0.5
    )
    DEFAULT_TOP_P = st.number_input("Top P", min_value=0.0, max_value=1.0, value=0.9)
else:
    SYSTEM_MESSAGE = ""
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.5
    DEFAULT_TOP_P = 0.9
with st.form("chat"):
    voice_drop_down = st.selectbox(
        "Text-to-Speech Response Voice", ["None"] + get_voices(), index=0
    )
    uploaded_file = st.file_uploader("Upload an image")
    prompt = st.text_area("Your Message:", "Describe each stage of this image.")
    send = st.form_submit_button("Send")
    if prompt != "" and send:
        start_time = time.time()
        st.markdown("---")
        st.spinner("Thinking...")
        messages = []
        if SYSTEM_MESSAGE != "":
            messages.append({"role": "system", "content": SYSTEM_MESSAGE})
        if uploaded_file:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": f"{uploaded_file.type.split('/')[0]}_url",
                            f"{uploaded_file.type.split('/')[0]}_url": {
                                "url": f"data:{uploaded_file.type};base64,{base64.b64encode(uploaded_file.read()).decode('utf-8')}",
                            },
                        },
                    ],
                },
            )
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, use_column_width=True)
        if messages == []:
            messages = [
                {"role": "user", "content": prompt},
            ]
        extra_body = None if voice_drop_down == "None" else {"voice": voice_drop_down}
        response = openai.chat.completions.create(
            model=DEFAULT_LLM,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            top_p=DEFAULT_TOP_P,
            stream=False,
            extra_body=extra_body,
        )
        display_content(response.choices[0].message.content)
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        # If response time is longer than 60 seconds, split the response time into minutes and seconds
        if elapsed_time > 60:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            st.success(f"Response time: {minutes} minutes and {seconds:.2f} seconds")
        else:
            st.success(f"Response time: {elapsed_time:.2f} seconds")
        st.balloons()
