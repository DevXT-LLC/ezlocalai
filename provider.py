from llama_cpp import Llama
import os
import re
import requests


def get_readme(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF"):
    model_name = model_url.split("/")[-1].replace("-GGUF", "").lower()
    if not os.path.exists(f"models/{model_name}.README.md"):
        readme_url = f"https://huggingface.co/{model_url}/raw/main/README.md"
        with requests.get(readme_url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(f"models/{model_name}.README.md", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    with open(f"models/{model_name}.README.md", "r") as f:
        readme = f.read()
    return readme


def get_prompt(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF"):
    model_name = model_url.split("/")[-1].replace("-GGUF", "").lower()
    if os.path.exists(f"models/{model_name}.prompt.txt"):
        with open(f"models/{model_name}.prompt.txt", "r") as f:
            prompt_template = f.read()
        return prompt_template
    readme = get_readme(model_url)
    prompt_template = readme.split("prompt_template: '")[1].split("'")[0]
    if prompt_template == "":
        prompt_template = "{system_message}\n\n{prompt}"
    return prompt_template


def get_model(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF", quant_type="Q4_K_M"):
    model_name = model_url.split("/")[-1].replace("-GGUF", "").lower()
    file_path = f"models/{model_name}.{quant_type}.gguf"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(file_path):
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
    with open(f"models/prompt.txt", "w") as f:
        f.write(prompt_template)
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
        self.model_path = get_model(model_url=model)
        self.max_tokens = max_tokens if max_tokens else 8192
        self.params["n_ctx"] = self.max_tokens
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

    def instruct(self, prompt, tokens: int = 0):
        self.params["n_predict"] = int(self.max_tokens) - tokens
        llm = Llama(**self.params, model_path=self.model_path)
        prompt_template = get_prompt(model_url=self.model)
        formatted_prompt = format_prompt(
            prompt=prompt, prompt_template=prompt_template, system_message=""
        )
        data = llm(prompt=formatted_prompt)
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
