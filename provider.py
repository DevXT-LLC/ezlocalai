from llama_cpp import Llama
import os
import re
import requests
import tiktoken
import json

THREADS = os.environ.get("THREADS")
GPU_LAYERS = os.environ.get("GPU_LAYERS")
MAIN_GPU = os.environ.get("MAIN_GPU")
BATCH_SIZE = os.environ.get("BATCH_SIZE")
DOWNLOAD_MODELS = (
    True if os.environ.get("DOWNLOAD_MODELS", "true").lower() == "true" else False
)


def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def get_model_name(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF"):
    model_name = model_url.split("/")[-1].replace("-GGUF", "").lower()
    return model_name


def get_readme(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF"):
    model_name = get_model_name(model_url=model_url)
    if not os.path.exists(f"models/{model_name}/README.md"):
        readme_url = f"https://huggingface.co/{model_url}/raw/main/README.md"
        with requests.get(readme_url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(f"models/{model_name}/README.md", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    with open(f"models/{model_name}/README.md", "r") as f:
        readme = f.read()
    return readme


def get_prompt(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF"):
    model_name = get_model_name(model_url=model_url)
    if os.path.exists(f"models/{model_name}/prompt.txt"):
        with open(f"models/{model_name}/prompt.txt", "r") as f:
            prompt_template = f.read()
        return prompt_template
    readme = get_readme(model_url)
    prompt_template = readme.split("prompt_template: '")[1].split("'")[0]
    if prompt_template == "":
        prompt_template = "{system_message}\n\n{prompt}"
    return prompt_template


def get_model(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF", quant_type="Q4_K_M"):
    model_name = get_model_name(model_url=model_url)
    file_path = f"models/{model_name}/{model_name}.{quant_type}.gguf"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(file_path):
        if DOWNLOAD_MODELS is False:
            raise Exception("Model not found.")
        url = (
            model_url
            if "https://" in model_url
            else f"https://huggingface.co/{model_url}/resolve/main/{model_name}.{quant_type}.gguf"
        )
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    prompt_template = get_prompt(model_url)
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


class LLM:
    def __init__(
        self,
        stop: str = "<|im_end|>",
        max_tokens: int = 8192,
        temperature: float = 1.31,
        top_p: float = 1.0,
        stream: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: list = [],
        model: str = "TheBloke/Mistral-7B-OpenOrca-GGUF",
        **kwargs,
    ):
        self.params = {}
        self.model = model
        self.params["model_path"] = get_model(model_url=model)
        self.max_tokens = max_tokens if max_tokens else 8192
        self.params["n_ctx"] = self.max_tokens
        self.params["verbose"] = False
        if stop:
            self.params["stop"] = stop
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
        if BATCH_SIZE:
            self.params["n_batch"] = int(BATCH_SIZE)

    def generate(self, prompt):
        prompt_template = get_prompt(model_url=self.model)
        formatted_prompt = format_prompt(
            prompt=prompt, prompt_template=prompt_template, system_message=""
        )
        tokens = get_tokens(formatted_prompt)
        self.params["n_predict"] = int(self.max_tokens) - tokens
        llm = Llama(**self.params)
        data = llm(prompt=formatted_prompt)
        return data

    def completion(self, prompt):
        data = self.generate(prompt=prompt)
        message = data["choices"][0]["text"]
        if message.startswith("\n "):
            message = message[3:]
        if message.endswith("\n\n  "):
            message = message[:-4]
        if self.params["stop"] in message:
            message = message.split(self.params["stop"])[0]
        data["choices"][0]["text"] = message
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
        message = data["choices"][0]["text"]
        if message.startswith("\n "):
            message = message[3:]
        if message.endswith("\n\n  "):
            message = message[:-4]
        if self.params["stop"] in message:
            message = message.split(self.params["stop"])[0]
        messages.append({"role": "assistant", "content": message})
        data["messages"] = messages
        return data

    def embedding(self, prompt):
        llm = Llama(embedding=True, **self.params)
        embeddings = llm.create_embedding(prompt=prompt)
        return embeddings
