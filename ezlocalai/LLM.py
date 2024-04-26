from llama_cpp import Llama, llama_chat_format
from huggingface_hub import hf_hub_download
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import os
import re
import requests
import psutil
import torch
import logging


DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "TheBloke/phi-2-dpo-GGUF")


def get_vision_models():
    return [
        "mys/ggml_bakllava-1",
        "mys/ggml_llava-v1.5-7b",
        "mys/ggml_llava-v1.5-13b",
    ]


def get_models():
    response = requests.get("https://huggingface.co/models?sort=modified&search=gguf")
    soup = BeautifulSoup(response.text, "html.parser")
    model_names = []
    if soup:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.endswith("-GGUF"):
                model_names.append(href[1:])
    model_names.append(get_vision_models())
    return model_names


def is_vision_model(model_name="") -> bool:
    if model_name == "":
        return False
    model_name = model_name.lower()
    for model in get_vision_models():
        for key in model:
            if model_name == key.lower():
                return True
    return False


def download_llm(model_name="", models_dir="models"):
    if model_name != "":
        global DEFAULT_MODEL
        model_name = DEFAULT_MODEL
    if "/" not in model_name:
        model_name = "TheBloke/" + model_name + "-GGUF"
    ram = round(psutil.virtual_memory().total / 1024**3)
    if ram > 16:
        default_quantization_type = "Q5_K_M"
    else:
        default_quantization_type = "Q4_K_M"
    quantization_type = os.environ.get("QUANT_TYPE", default_quantization_type)
    model = model_name.split("/")[-1].replace("-GGUF", "")
    filename = model + f".{quantization_type}.gguf"
    models_dir = os.path.join(models_dir, model)
    os.makedirs(models_dir, exist_ok=True)
    clip_file = None
    if model_name.split("/")[0] == "mys":
        filename = f"ggml-model-q5_k.gguf"
        clip_file = f"mmproj-model-f16.gguf"
    # Check if file exists
    if clip_file:
        if not os.path.exists(f"{models_dir}/{clip_file}"):
            hf_hub_download(
                repo_id=model_name,
                filename=clip_file,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
            )
    if os.path.exists(f"{models_dir}/{filename}"):
        return f"{models_dir}/{filename}"
    try:
        file_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
    except:
        filename = model + f".q4_k_m.gguf"
        file_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
    return file_path


def get_clip_path(model_name="", models_dir="models"):
    if model_name == "":
        global DEFAULT_MODEL
        model_name = DEFAULT_MODEL
    if os.path.exists(f"{models_dir}/{model_name}/mmproj-model-f16.gguf"):
        return f"{models_dir}/{model_name}/mmproj-model-f16.gguf"
    else:
        return ""


def clean(
    message: str = "",
    stop_tokens: List[str] = [
        "<|im_end|",
        "<|im_end|>",
        "</|im_end|>",
        "</s>",
        "<s>",
        "User:",
        "### \n###",
        "[/INST]",
        "<|eot_id|>",
        "<|end_of_text|>",
        "assistant\n\n",
    ],
):
    if message == "":
        return message
    for token in stop_tokens:
        if token in message:
            message = message.split(token)[0]
    message = message.strip()
    if message.startswith("\n "):
        message = message[3:]
    if message.endswith("\n\n"):
        message = message[:-4]
    if message.startswith(" "):
        message = message[1:]
    if message.endswith("\n"):
        message = message[:-3]
    if "[Insert " in message:
        message = re.sub(r"\[Insert.*?\]", "", message)
    return message


class LLM:
    def __init__(
        self,
        stop: List[str] = [],
        temperature: float = 1.31,
        max_tokens: int = 0,
        top_p: float = 0.95,
        min_p: float = 0.05,
        stream: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        model: str = "",
        models_dir: str = "./models",
        system_message: str = "",
        **kwargs,
    ):
        global DEFAULT_MODEL
        MAIN_GPU = os.environ.get("MAIN_GPU", 0)
        GPU_LAYERS = os.environ.get("GPU_LAYERS", 0)
        if torch.cuda.is_available() and int(GPU_LAYERS) == -1:
            # 5GB VRAM reserved for TTS and STT.
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3) - 5
            if vram == 3:
                vram = 1
            if vram <= 0:
                vram = 0
            logging.info(f"[LLM] {vram}GB of available VRAM detected.")
            GPU_LAYERS = vram - 1 if vram > 0 else 0
        if GPU_LAYERS == -2:
            GPU_LAYERS = -1
        logging.info(
            f"[LLM] Loading {DEFAULT_MODEL} with {GPU_LAYERS if GPU_LAYERS != -1 else 'all'} GPU layers. Please wait..."
        )
        self.params = {}
        self.model_name = DEFAULT_MODEL
        chat_handler = None
        if self.model_name != "":
            self.params["model_path"] = download_llm(
                model_name=self.model_name, models_dir=models_dir
            )
            if max_tokens != 0:
                self.params["max_tokens"] = 4096
            else:
                self.params["max_tokens"] = max_tokens
            if is_vision_model(model_name=self.model_name):
                clip_path = get_clip_path(
                    model_name=self.model_name, models_dir=models_dir
                )
                if clip_path != "":
                    chat_handler = llama_chat_format.Llava15ChatHandler(
                        clip_model_path=clip_path, verbose=True
                    )
        else:
            self.params["model_path"] = ""
            self.params["max_tokens"] = 8192
        self.params["n_ctx"] = int(os.environ.get("LLM_MAX_TOKENS", 4096))
        self.params["verbose"] = True
        self.system_message = system_message
        self.params["mirostat_mode"] = 2
        self.params["top_k"] = 20 if "top_k" not in kwargs else kwargs["top_k"]
        self.params["stop"] = [
            "<|im_end|",
            "<|im_end|>",
            "</|im_end|>",
            "</s>",
            "<s>",
            "User:",
            "### \n###",
            "[/INST]",
            "<|eot_id|>",
            "<|end_of_text|>",
            "assistant\n\n",
        ]
        if stop != []:
            if isinstance(stop, str):
                self.params["stop"].append(stop)
            else:
                try:
                    for stop_string in stop:
                        if stop_string not in self.params["stop"]:
                            if stop_string != None:
                                self.params["stop"].append(stop_string)
                except:
                    if stop != None:
                        self.params["stop"].append(stop)

        self.params["temperature"] = temperature if temperature else 1.31
        self.params["top_p"] = top_p if top_p else 0.95
        self.params["min_p"] = min_p if min_p else 0.05
        self.params["stream"] = stream if stream else False
        self.params["presence_penalty"] = presence_penalty if presence_penalty else 0.0
        self.params["frequency_penalty"] = (
            frequency_penalty if frequency_penalty else 0.0
        )
        self.params["repetition_penalty"] = 1.2
        self.params["logit_bias"] = logit_bias if logit_bias else None
        self.params["n_gpu_layers"] = int(GPU_LAYERS) if GPU_LAYERS else 0
        self.params["main_gpu"] = int(MAIN_GPU) if MAIN_GPU else 0
        if "batch_size" in kwargs:
            self.params["n_batch"] = (
                int(kwargs["batch_size"]) if kwargs["batch_size"] else 1024
            )
        else:
            self.params["n_batch"] = int(os.environ.get("LLM_BATCH_SIZE", 1024))
        if self.model_name != "":
            self.lcpp = Llama(
                **self.params,
                chat_handler=chat_handler,
                logits_all=True if chat_handler else False,
            )
        else:
            self.lcpp = None
        self.model_list = get_models()

    def generate(
        self,
        prompt,
        max_tokens=None,
        temperature=None,
        top_p=None,
        min_p=None,
        top_k=None,
        logit_bias=None,
        mirostat_mode=None,
        frequency_penalty=None,
        presence_penalty=None,
        stream=None,
        model=None,
        system_message=None,
        **kwargs,
    ):
        messages = [
            {
                "role": "system",
                "content": (
                    self.system_message if system_message is None else system_message
                ),
            }
        ]
        if isinstance(prompt, list):
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                },
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                },
            )
        data = self.lcpp.create_chat_completion(
            messages=messages,
            max_tokens=(
                self.params["max_tokens"] if max_tokens is None else int(max_tokens)
            ),
            temperature=(
                self.params["temperature"]
                if temperature is None
                else float(temperature)
            ),
            top_p=self.params["top_p"] if top_p is None else float(top_p),
            min_p=self.params["min_p"] if min_p is None else float(min_p),
            stop=self.params["stop"],
            top_k=self.params["top_k"] if top_k is None else int(top_k),
            logit_bias=self.params["logit_bias"] if logit_bias is None else logit_bias,
            mirostat_mode=(
                self.params["mirostat_mode"]
                if mirostat_mode is None
                else int(mirostat_mode)
            ),
            frequency_penalty=(
                self.params["frequency_penalty"]
                if frequency_penalty is None
                else float(frequency_penalty)
            ),
            presence_penalty=(
                self.params["presence_penalty"]
                if presence_penalty is None
                else float(presence_penalty)
            ),
            stream=self.params["stream"] if stream is None else stream,
            model=self.model_name if model is None else model,
        )
        data["model"] = self.model_name
        return data

    def completion(self, prompt, **kwargs):
        data = self.generate(prompt=prompt, **kwargs)
        data["choices"][0]["message"]["content"] = clean(
            message=data["choices"][0]["message"]["content"],
            stop_tokens=self.params["stop"],
        )
        data["choices"][0]["text"] = data["choices"][0]["message"]["content"]
        return data

    def chat(self, messages, **kwargs):
        user_input = messages[-1]["content"]
        data = self.generate(prompt=user_input, **kwargs)
        data["choices"][0]["message"]["content"] = clean(
            message=data["choices"][0]["message"]["content"],
            stop_tokens=self.params["stop"],
        )
        return data

    def models(self):
        return self.model_list


if __name__ == "__main__":
    logging.info(f"[LLM] Downloading {DEFAULT_MODEL} model...")
    download_llm(model_name=DEFAULT_MODEL, models_dir="models")
