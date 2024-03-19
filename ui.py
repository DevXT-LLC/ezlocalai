import streamlit as st
import openai
import requests
import time
import base64
import os
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
st.title("ezLocalai")

EZLOCALAI_SERVER = os.getenv("EZLOCALAI_URL", "http://localhost:8091")
EZLOCALAI_API_KEY = os.getenv("EZLOCALAI_API_KEY", "none")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "phi-2-dpo")
openai.base_url = f"{EZLOCALAI_SERVER}/v1/"
openai.api_key = EZLOCALAI_API_KEY if EZLOCALAI_API_KEY else EZLOCALAI_SERVER
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"{EZLOCALAI_API_KEY}",
    "ngrok-skip-browser-warning": "true",
}


def get_voices():
    global EZLOCALAI_SERVER
    global HEADERS
    voices = requests.get(f"{EZLOCALAI_SERVER}/v1/audio/voices", headers=HEADERS)
    return voices.json()


waiting_for_server = False

while True:
    try:
        voices = get_voices()
        break
    except:
        if waiting_for_server == False:
            st.spinner("Waiting for server to start...")
        waiting_for_server = True
        time.sleep(1)
waiting_for_server = False


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


with st.form("chat"):
    SYSTEM_MESSAGE = st.text_area(
        "System Prompt",
        "The assistant is acting as a creative writer. All of your text responses are transcribed to audio and sent to the user. Be concise with all responses. After the request is fulfilled, end with </s>.",
    )
    DEFAULT_MAX_TOKENS = st.number_input(
        "Max Output Tokens", min_value=10, max_value=300000, value=256
    )
    DEFAULT_TEMPERATURE = st.number_input(
        "Temperature", min_value=0.0, max_value=1.0, value=0.5
    )
    DEFAULT_TOP_P = st.number_input("Top P", min_value=0.0, max_value=1.0, value=0.9)
    voice_drop_down = st.selectbox(
        "Text-to-Speech Response Voice", ["None"] + voices["voices"], index=0
    )
    uploaded_file = st.file_uploader("Upload an image")
    prompt = st.text_area("Your Message:", "Describe each stage of this image.")
    send = st.form_submit_button("Send")
    if prompt != "" and send:
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
        extra_body = {} if voice_drop_down == "None" else {"voice": voice_drop_down}
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
        st.balloons()
