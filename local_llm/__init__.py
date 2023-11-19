from llama_cpp import Llama, llama_chat_format
from bs4 import BeautifulSoup
from typing import List
import os
import re
import requests
import tiktoken
import json
import psutil
import GPUtil


def get_sys_info():
    try:
        gpus = GPUtil.getGPUs()
    except:
        gpus = "None"
    ram = psutil.virtual_memory().total / 1024**3
    ram = round(ram)
    threads = psutil.cpu_count()
    return gpus, ram, threads


gpus, ram, threads = get_sys_info()
if gpus == "None":
    GPU_LAYERS = 0
    MAIN_GPU = 0
else:
    GPU_LAYERS = os.environ.get("GPU_LAYERS", 0)
    MAIN_GPU = os.environ.get("MAIN_GPU", 0)
THREADS = os.environ.get("THREADS", threads - 2)
DOWNLOAD_MODELS = (
    True if os.environ.get("DOWNLOAD_MODELS", "true").lower() == "true" else False
)


def get_models():
    try:
        response = requests.get(
            "https://huggingface.co/TheBloke?search_models=GGUF&sort_models=modified"
        )
        soup = BeautifulSoup(response.text, "html.parser")
    except:
        soup = None
    model_names = [
        {"bakllava-1-7b": "mys/ggml_bakllava-1"},
        {"llava-v1.5-7b": "mys/ggml_llava-v1.5-7b"},
        {"llava-v1.5-13b": "mys/ggml_llava-v1.5-13b"},
    ]
    if soup:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("/TheBloke/") and href.endswith("-GGUF"):
                base_name = href[10:-5]
                model_names.append({base_name: href[1:]})
    return model_names


def get_model_url(model_name="Mistral-7B-OpenOrca"):
    model_url = ""
    try:
        models = get_models()
        for model in models:
            for key in model:
                if model_name.lower() == key.lower():
                    model_url = model[key]
                    break
        if model_url == "":
            raise Exception(
                f"Model not found. Choose from one of these models: {', '.join(models.keys())}"
            )
    except:
        model_url = f"https://huggingface.co/TheBloke/{model_name}-GGUF"
    return model_url


def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def get_model_name(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF"):
    model_name = model_url.split("/")[-1].replace("-GGUF", "").lower()
    return model_name


def get_readme(model_name="Mistral-7B-OpenOrca", models_dir="models"):
    model_url = get_model_url(model_name=model_name)
    model_name = model_name.lower()
    if not os.path.exists(f"{models_dir}/{model_name}/README.md"):
        readme_url = f"https://huggingface.co/{model_url}/raw/main/README.md"
        with requests.get(readme_url, stream=True, allow_redirects=True) as r:
            with open(f"{models_dir}/{model_name}/README.md", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    with open(f"{models_dir}/{model_name}/README.md", "r") as f:
        readme = f.read()
    return readme


def get_max_tokens(model_name="Mistral-7B-OpenOrca", models_dir="models"):
    readme = get_readme(model_name=model_name, models_dir=models_dir)
    if "200k" in readme:
        return 200000
    if "131072" in readme or "128k" in readme:
        return 131072
    if "65536" in readme or "64k" in readme:
        return 65536
    if "32768" in readme or "32k" in readme:
        return 32768
    if "16384" in readme or "16k" in readme:
        return 16384
    if "8192" in readme or "8k" in readme:
        return 8192
    if "4096" in readme or "4k" in readme:
        return 4096
    if "2048" in readme or "2k" in readme:
        return 2048
    return 8192


def get_prompt(model_name="Mistral-7B-OpenOrca", models_dir="models"):
    model_name = model_name.lower()
    if os.path.exists(f"{models_dir}/{model_name}/prompt.txt"):
        with open(f"{models_dir}/{model_name}/prompt.txt", "r") as f:
            prompt_template = f.read()
        return prompt_template
    readme = get_readme(model_name=model_name, models_dir=models_dir)
    try:
        prompt_template = readme.split("prompt_template: '")[1].split("'")[0]
    except:
        prompt_template = ""
    if prompt_template == "":
        prompt_template = "{system_message}\n\n{prompt}"
    return prompt_template


def get_model(model_name="Mistral-7B-OpenOrca", models_dir="models"):
    if ram > 16:
        default_quantization_type = "Q5_K_M"
    else:
        default_quantization_type = "Q4_K_M"
    quantization_type = os.environ.get("QUANT_TYPE", default_quantization_type)
    model_url = get_model_url(model_name=model_name)
    model_name = model_name.lower()
    file_path = f"{models_dir}/{model_name}/{model_name}.{quantization_type}.gguf"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(f"{models_dir}/{model_name}"):
        os.makedirs(f"{models_dir}/{model_name}")
    if not os.path.exists(file_path):
        if DOWNLOAD_MODELS is False:
            raise Exception("Model not found.")
        clip_url = ""
        if model_url.startswith("mys/"):
            # Multimodal models
            url = (
                f"https://huggingface.co/{model_url}/resolve/main/ggml-model-q5_k.gguf"
            )
            clip_url = (
                f"https://huggingface.co/{model_url}/resolve/main/mmproj-model-f16.gguf"
            )
        else:
            url = (
                (
                    model_url
                    if "https://" in model_url
                    else f"https://huggingface.co/{model_url}/resolve/main/{model_name}.{quantization_type}.gguf"
                )
                if model_name != "mistrallite-7b"
                else f"https://huggingface.co/TheBloke/MistralLite-7B-GGUF/resolve/main/mistrallite.{quantization_type}.gguf"
            )
        print(f"Downloading {model_name}...")
        with requests.get(url, stream=True, allow_redirects=True) as r:
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        if clip_url != "":
            print(f"Downloading {model_name} CLIP...")
            with requests.get(clip_url, stream=True, allow_redirects=True) as r:
                with open(
                    f"{models_dir}/{model_name}/mmproj-model-f16.gguf", "wb"
                ) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    return file_path


def custom_format(string, **kwargs):
    if isinstance(string, list):
        string = "".join(str(x) for x in string)

    def replace(match):
        key = match.group(1)
        value = kwargs.get(key, match.group(0))
        if isinstance(value, list):
            return "".join(str(x) for x in value)
        else:
            return str(value)

    pattern = r"(?<!{){([^{}\n]+)}(?!})"
    result = re.sub(pattern, replace, string)
    return result


def format_prompt(prompt, prompt_template, system_message=""):
    formatted_prompt = custom_format(
        string=prompt_template, prompt=prompt, system_message=system_message
    )
    return formatted_prompt


async def streaming_generation(data):
    yield "data: {}\n".format(json.dumps(data))
    for line in data.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            current_data = json.loads(decoded_line[6:])
            yield "data: {}\n".format(json.dumps(current_data))


def clean(message: str = ""):
    if message.startswith("\n "):
        message = message[3:]
    if message.endswith("\n\n  "):
        message = message[:-4]
    if message.startswith(" "):
        message = message[1:]
    if message.endswith("\n  "):
        message = message[:-3]
    return message


class LLM:
    def __init__(
        self,
        stop: List[str] = [],
        max_tokens: int = 0,
        temperature: float = 1.31,
        top_p: float = 1.0,
        stream: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: list = [],
        model: str = "",
        models_dir: str = "./models",
        system_message: str = "",
        **kwargs,
    ):
        self.params = {}
        self.model_name = model
        if model != "":
            self.params["model_path"] = get_model(
                model_name=self.model_name, models_dir=models_dir
            )
            model_max_tokens = get_max_tokens(
                model_name=self.model_name, models_dir=models_dir
            )
            self.prompt_template = get_prompt(
                model_name=self.model_name, models_dir=models_dir
            )
            if "llava" in self.model_name:
                self.params["chat_handler"] = llama_chat_format.Llava15ChatHandler(
                    clip_model_path=self.params["model_path"], verbose=False
                )
        else:
            self.params["model_path"] = ""
            model_max_tokens = 8192
            self.prompt_template = "{system_message}\n\n{prompt}"
        try:
            self.max_tokens = (
                int(max_tokens)
                if max_tokens and int(max_tokens) > 0
                else model_max_tokens
            )
        except:
            self.max_tokens = model_max_tokens
        self.params["n_ctx"] = self.max_tokens
        self.params["verbose"] = False
        self.system_message = system_message
        self.params["mirostat_mode"] = 2
        self.params["top_k"] = 20 if "top_k" not in kwargs else kwargs["top_k"]
        self.params["stop"] = ["<|im_end|>", "</|im_end|>", "</s>"]
        if stop != []:
            if isinstance(stop, str):
                self.params["stop"].append(stop)
            else:
                try:
                    for stop_string in stop:
                        if stop_string not in self.params["stop"]:
                            self.params["stop"].append(stop_string)
                except:
                    self.params["stop"].append(stop)
        if temperature:
            self.params["temperature"] = temperature
        if top_p:
            self.params["top_p"] = top_p
        if stream:
            self.params["stream"] = stream
        if presence_penalty:
            self.params["presence_penalty"] = presence_penalty
        if frequency_penalty:
            self.params["frequency_penalty"] = frequency_penalty
        if logit_bias:
            self.params["logit_bias"] = logit_bias
        if THREADS:
            self.params["n_threads"] = int(THREADS)
        if GPU_LAYERS:
            self.params["n_gpu_layers"] = int(GPU_LAYERS)
        if MAIN_GPU:
            self.params["main_gpu"] = int(MAIN_GPU)
        if "batch_size" in kwargs:
            self.params["n_batch"] = int(kwargs["batch_size"])

    def generate(self, prompt):
        formatted_prompt = format_prompt(
            prompt=prompt,
            prompt_template=self.prompt_template,
            system_message=self.system_message,
        )
        tokens = get_tokens(formatted_prompt)
        self.params["n_predict"] = int(self.max_tokens) - tokens
        llm = Llama(**self.params)
        data = llm(prompt=formatted_prompt)
        data["model"] = self.model_name
        return data

    def completion(self, prompt):
        data = self.generate(prompt=prompt)
        data["choices"][0]["text"] = clean(
            params=self.params, message=data["choices"][0]["text"]
        )
        return data

    def chat(self, messages):
        if len(messages) > 1:
            for message in messages:
                if message["role"] == "system":
                    prompt = f"\nASSISTANT's RULE: {message.content}"
                elif message["role"] == "user":
                    prompt = f"\nUSER: {message.content}"
                elif message["role"] == "assistant":
                    prompt = f"\nASSISTANT: {message.content}"
        else:
            try:
                prompt = messages[0]["content"]
            except:
                prompt = messages
        data = self.generate(prompt=prompt)
        messages = [{"role": "user", "content": prompt}]
        message = clean(message=data["choices"][0]["text"])
        messages.append({"role": "assistant", "content": message})
        data["messages"] = messages
        del data["choices"]
        return data

    def embedding(self, input):
        llm = Llama(embedding=True, **self.params)
        embeddings = llm.create_embedding(input=input, model=self.model_name)
        embeddings["model"] = self.model_name
        return embeddings

    def models(self):
        models = get_models()
        model_list = []
        for model in models:
            for key in model:
                model_list.append(key)
        return model_list
