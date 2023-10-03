import streamlit as st
import requests
from bs4 import BeautifulSoup
from configure import auto_configure
from GetModel import get_model
from dotenv import load_dotenv
import os
import GPUtil
import psutil
import subprocess

load_dotenv()
CURRENT_MODEL = os.getenv("MODEL_URL", "TheBloke/Mistral-7B-OpenOrca-GGUF")
QUANT_TYPE = os.getenv("QUANT_TYPE", "Q4_K_M")
MAX_TOKENS = os.getenv("MAX_TOKENS", 8192)
THREADS = os.getenv("THREADS", 4)
THREADS_BATCH = os.getenv("THREADS_BATCH", 4)
GPU_LAYERS = os.getenv("GPU_LAYERS", 0)
MAIN_GPU = os.getenv("MAIN_GPU", 0)
BATCH_SIZE = os.getenv("BATCH_SIZE", 512)
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
st.set_page_config(
    page_title="Local-LLM",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Local-LLM Configurator")
is_container_running = subprocess.run(
    ["docker-compose", "ps", "-q", "llm-server"], capture_output=True
).stdout


try:
    gpus = GPUtil.getGPUs()
except:
    gpus = "None"


psutil.cpu_stats()
ram_in_gb = psutil.virtual_memory().total / 1024**3
# Round the ram to closest full number
ram_in_gb = round(ram_in_gb)
if st.button("Update Server"):
    if gpus == "None":
        subprocess.run(["docker-compose", "pull"])
    else:
        if "nvidia" in gpus[0].name.lower():
            subprocess.run(["docker-compose", "-f", "docker-compose-cuda.yml", "pull"])
        else:
            subprocess.run(["docker-compose", "pull"])
    st.experimental_rerun()

if is_container_running:
    # Stop server button
    if st.button("Stop Server"):
        if gpus == "None":
            subprocess.run(["docker-compose", "down"])
        else:
            if "nvidia" in gpus[0].name.lower():
                subprocess.run(
                    ["docker-compose", "-f", "docker-compose-cuda.yml", "down"]
                )
            else:
                subprocess.run(["docker-compose", "down"])
        st.experimental_rerun()
else:
    # Start server button
    if st.button(
        "Start Server",
        help="Start the Local-LLM server. This will download the model if you do not already have it, configure it, and start the server.",
    ):
        subprocess.run(["docker-compose", "up"])
        st.experimental_rerun()

st.markdown(
    "### About your computer\n\n"
    f"- **GPU Information:** {gpus}\n\n"
    f"- **CPU Threads:** {psutil.cpu_count()}\n"
    f"- **RAM:** {ram_in_gb} GB"
)


def get_models():
    response = requests.get(
        "https://huggingface.co/TheBloke?search_models=GGUF&sort_models=modified"
    )
    soup = BeautifulSoup(response.text, "html.parser")
    model_names = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/TheBloke/") and href.endswith("-GGUF"):
            base_name = href[10:-5]
            model_names.append({base_name: href[1:]})
    return model_names


manual_config = st.checkbox("Manually Configure Environment")

model_names = get_models()
with st.form("configure"):
    model_keys = [list(model.keys())[0] for model in model_names]
    quantization_types = [
        "Q4_K_M",
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_0",
        "Q4_K_S",
        "Q5_0",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    ]
    current_model_name = CURRENT_MODEL.split("/")[1].replace("-GGUF", "")
    default_index = (
        model_keys.index(current_model_name) if current_model_name in model_keys else 0
    )
    model_name = st.selectbox(
        "Select a Model",
        model_keys,
        index=default_index,
        help="The model URL or repository name to download from Hugging Face.",
    )
    model_url = list(model_names[default_index].values())[0]
    # Link to the model
    st.markdown(
        f"[Click here for more information about {model_name} on Hugging Face](https://huggingface.co/{model_url})"
    )
    local_llm_api_key = st.text_input(
        "Local-LLM API Key",
        value=LOCAL_LLM_API_KEY,
        help="The API key to use for the server. If not set, the server will not require an API key.",
    )

    if manual_config:
        quantization_type = st.selectbox(
            "Quantization Type",
            quantization_types,
            index=quantization_types.index(QUANT_TYPE),
            help="The quantization type to use. Default is Q4_K_M.",
        )
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            value=int(MAX_TOKENS),
            help="The maximum number of tokens. Default is 8192.",
        )
        threads = st.number_input(
            "Threads",
            min_value=1,
            value=int(THREADS),
            help="The number of threads to use.",
        )
        threads_batch = st.number_input(
            "Threads Batch",
            min_value=1,
            value=int(THREADS_BATCH),
            help="The number of threads to use for batch generation, this will enable parallel generation of batches. Setting it to the same value as threads will disable batch generation.",
        )
        gpu_layers = st.number_input(
            "GPU Layers",
            min_value=0,
            value=int(GPU_LAYERS),
            help="The number of layers to use on the GPU. Default is 0.",
        )
        main_gpu = st.number_input(
            "Main GPU",
            min_value=0,
            value=int(MAIN_GPU),
            help="The GPU to use for the main model. Default is 0.",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            value=int(BATCH_SIZE),
            help="The batch size to use for batch generation. Default is 512.",
        )

        if st.form_submit_button("Save Configuration"):
            with open(".env", "w") as f:
                f.write(
                    f"MODEL_URL={model_names[model_keys.index(model_name)][model_name]}\n"
                )
                f.write(f"QUANT_TYPE={quantization_type}\n")
                f.write(f"MAX_TOKENS={max_tokens}\n")
                f.write(f"THREADS={threads}\n")
                f.write(f"THREADS_BATCH={threads_batch}\n")
                f.write(f"GPU_LAYERS={gpu_layers}\n")
                f.write(f"MAIN_GPU={main_gpu}\n")
                f.write(f"BATCH_SIZE={batch_size}\n")
                f.write(f"LOCAL_LLM_API_KEY={local_llm_api_key}\n")
            st.experimental_rerun()
    else:
        quantization_type = QUANT_TYPE
        max_tokens = MAX_TOKENS
        threads = THREADS
        threads_batch = THREADS_BATCH
        gpu_layers = GPU_LAYERS
        main_gpu = MAIN_GPU
        batch_size = BATCH_SIZE
        if st.form_submit_button("Auto Configure Server"):
            model_url = list(model_names[default_index].values())[0]
            auto_configure(model_url=model_url, api_key=local_llm_api_key)
            get_model(model_url=model_url, quant_type=QUANT_TYPE)
            st.experimental_rerun()
