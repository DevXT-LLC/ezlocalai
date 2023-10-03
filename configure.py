import re
import sys
import subprocess
import logging
import asyncio
import psutil

try:
    import argparse
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "argparse"])
    import argparse
from GetModel import get_readme

try:
    from g4f.Provider import RetryProvider
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "g4f"])
    from g4f.Provider import RetryProvider
from g4f.models import ModelUtils, gpt_35_turbo, default

try:
    import GPUtil
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "GPUtil"])
    import GPUtil


async def get_model_url(prompt):
    model_response = await Gpt4freeProvider().instruct(prompt=prompt)
    # Strip out anything that isn't the url from the response
    extract_model_url = model_response.split("https://huggingface.co/")[1].split(
        ".gguf"
    )[0]
    if not extract_model_url:
        return await get_model_url(prompt)
    new_model_url = f"https://huggingface.co/{extract_model_url}.gguf"
    return new_model_url


async def auto_configure(model_url="TheBloke/Mistral-7B-OpenOrca-GGUF"):
    readme = get_readme(model_url)
    table = readme.split("Provided files\n")[1].split("\n\n")[0]
    with open("hardwarereqs.txt", "r") as f:
        hardware_requirements = f.read()
    prompt = f"Readme: {hardware_requirements}\n\n{table}\n\n"
    prompt = "## Current Computer Specifications\n"
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]
    gpu_name = gpu.name
    gpu_memory = gpu.memoryTotal
    cpu_name = psutil.cpu_freq().current
    ram = psutil.virtual_memory().total
    prompt += f"- GPU: {gpu_name} ({gpu_memory} MB)\n"
    prompt += f"- CPU: {cpu_name} MHz\n"
    prompt += f"- RAM: {ram} MB\n\n"
    get_model_prompt = f"{prompt}**Determine the best model to run for a machine with the computer specifications provided. Ensure that the model will not utilize more than 80% of the resources. Return only the model URL of the best one.**"
    new_model_url = get_model_url(get_model_prompt)
    # How many GPU layers, threads, and batch size?
    get_settings_prompt = f"{prompt}The default batch size is 512. The default GPU layers is set to 0. If not running an NVIDIA GPU, GPU Layers should be 0. **How many GPU layers, CPU threads, and batch size should we use when running llama.cpp on this computer?** Respond like this: `layers: 0, threads: 4, batch size: 512`"
    settings_response = await Gpt4freeProvider().instruct(prompt=get_settings_prompt)
    # Strip out anything that isn't the settings from the response
    gpu_layers = settings_response.split("layers: ")[1].split(", threads: ")[0]
    if not gpu_layers:
        gpu_layers = 0
    cpu_threads = settings_response.split(", threads: ")[1].split(", batch size: ")[0]
    if not cpu_threads:
        cpu_threads = 4
    batch_size = settings_response.split(", batch size: ")[1]
    if not batch_size:
        batch_size = 512
    get_max_tokens = f"{prompt}**Does anything indicate what the token limit is? Something like 8k, 16k, 32k, something like that would tell us.  If so, just respond with a python code block with the number.  For example, 8k would be 8192.**"
    max_tokens_response = await Gpt4freeProvider().instruct(prompt=get_max_tokens)
    # Strip out anything but numbers from max_tokens_response
    max_tokens = re.sub("[^0-9]", "", max_tokens_response)
    if not max_tokens:
        max_tokens = 8192
    if max_tokens >= 32768:
        # Run two 16k context models in parallel if over 32k tokens
        threads_batch = cpu_threads / 2
    else:
        threads_batch = cpu_threads
    # Now we need to create a .env file with the settings
    quantization_type = new_model_url.split(".Q")[1].split(".gguf")[0]
    with open(".env", "w") as f:
        f.write(f"GPU_LAYERS={gpu_layers}\n")
        f.write(f"THREADS={cpu_threads}\n")
        f.write(f"THREADS_BATCH={threads_batch}\n")
        f.write(f"BATCH_SIZE={batch_size}\n")
        f.write(f"MODEL_URL={model_url}\n")
        f.write(f"MAX_TOKENS={max_tokens}\n")
        f.write(f"QUANT_TYPE=Q{quantization_type}\n")
        f.write(f"MAIN_GPU=0")
    return model_url


class Gpt4freeProvider:
    def __init__(
        self,
        AI_MODEL: str = gpt_35_turbo.name,
        MAX_TOKENS: int = 4096,
        AI_TEMPERATURE: float = 0.0,
        AI_TOP_P: float = 1.0,
        WAIT_BETWEEN_REQUESTS: int = 1,
        WAIT_AFTER_FAILURE: int = 3,
        **kwargs,
    ):
        self.requirements = ["g4f", "httpx"]
        if not AI_MODEL:
            self.AI_MODEL = default
        elif AI_MODEL in ModelUtils.convert:
            self.AI_MODEL = ModelUtils.convert[AI_MODEL]
        else:
            raise ValueError(f"Model not found: {AI_MODEL}")
        self.AI_TEMPERATURE = AI_TEMPERATURE if AI_TEMPERATURE else 0.7
        self.MAX_TOKENS = MAX_TOKENS if MAX_TOKENS else 4096
        self.AI_TOP_P = AI_TOP_P if AI_TOP_P else 0.7
        self.WAIT_BETWEEN_REQUESTS = (
            WAIT_BETWEEN_REQUESTS if WAIT_BETWEEN_REQUESTS else 1
        )
        self.WAIT_AFTER_FAILURE = WAIT_AFTER_FAILURE if WAIT_AFTER_FAILURE else 3

    async def instruct(self, prompt, tokens: int = 0):
        max_new_tokens = (
            int(self.MAX_TOKENS) - int(tokens) if tokens > 0 else self.MAX_TOKENS
        )
        model = self.AI_MODEL
        provider = model.best_provider
        if provider:
            append_model = f" and model: {model.name}" if model.name else ""
            logging.info(f"[Gpt4Free] Use provider: {provider.__name__}{append_model}")
        try:
            return await asyncio.gather(
                provider.create_async(
                    model=model.name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=float(self.AI_TEMPERATURE),
                    top_p=float(self.AI_TOP_P),
                ),
                asyncio.sleep(int(self.WAIT_BETWEEN_REQUESTS)),
            )[0]
        except Exception as e:
            if int(self.WAIT_AFTER_FAILURE) > 0:
                await asyncio.sleep(int(self.WAIT_AFTER_FAILURE))
            raise e
        finally:
            if provider and isinstance(provider, RetryProvider):
                if hasattr(provider, "exceptions"):
                    for provider_name in provider.exceptions:
                        error = provider.exceptions[provider_name]
                        logging.error(f"[Gpt4Free] {provider_name}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", type=str, default="None")
    args = parser.parse_args()
    model_url = args.model_url
    if model_url != "None":
        asyncio.run(auto_configure(model_url))
