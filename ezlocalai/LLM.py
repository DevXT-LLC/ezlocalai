import xllamacpp as xlc
from huggingface_hub import hf_hub_download
from typing import List, Optional, Dict
import os
import re
import torch
import logging
import json
from Globals import getenv

DEFAULT_MODEL = getenv("DEFAULT_MODEL")


def get_models():
    """Return a list of available models."""
    return [
        {"id": DEFAULT_MODEL, "object": "model", "owned_by": "ezlocalai"},
    ]


def download_model(model_name: str = "", models_dir: str = "models") -> tuple:
    """
    Download a model from HuggingFace Hub.
    Returns tuple of (model_path, mmproj_path) where mmproj_path may be None.
    """
    global DEFAULT_MODEL
    model_name = model_name if model_name else DEFAULT_MODEL

    if "/" not in model_name:
        model_name = "TheBloke/" + model_name + "-GGUF"

    quantization_type = getenv("QUANT_TYPE")
    model = model_name.split("/")[-1].split("-GGUF")[0]
    model_dir = os.path.join(models_dir, model)
    os.makedirs(model_dir, exist_ok=True)

    # Try to find or download multimodal projector files (for vision models)
    mmproj_path = None
    potential_mmproj_files = [
        # Common naming conventions for vision model projectors
        "mmproj-F16.gguf",
        "mmproj-BF16.gguf",
        "mmproj-F32.gguf",
        "mmproj-f16.gguf",
        "mmproj-model-f16.gguf",
        f"{model}-mmproj-f16.gguf",
        "mmproj.gguf",
        f"{model.lower()}-mmproj-f16.gguf",
    ]

    for mmproj_file in potential_mmproj_files:
        mmproj_filepath = os.path.join(model_dir, mmproj_file)
        if os.path.exists(mmproj_filepath):
            mmproj_path = mmproj_filepath
            break
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=mmproj_file,
                local_dir=model_dir,
            )
            mmproj_path = mmproj_filepath
            logging.info(f"[LLM] Downloaded mmproj: {mmproj_file}")
            break
        except Exception:
            pass

    # Look for the main model file
    potential_filenames = [
        f"{model}.{quantization_type}.gguf",
        f"{model}-{quantization_type}.gguf",
        f"{model}.{quantization_type.lower()}.gguf",
        f"{model}-{quantization_type.lower()}.gguf",
        f"{model}.Q4_K_M.gguf",
        f"{model}.q4_k_m.gguf",
        f"{model}-Q4_K_M.gguf",
        f"{model}-q4_k_m.gguf",
        "ggml-model-q5_k.gguf",
        "ggml-model-f16.gguf",
    ]

    # Check if model already exists
    for filename in potential_filenames:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            return filepath, mmproj_path

    # Download the model
    logging.info(f"[LLM] Downloading {model}...")
    for filename in potential_filenames:
        filepath = os.path.join(model_dir, filename)
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=model_dir,
            )
            logging.info(f"[LLM] Downloaded {model} successfully!")
            return filepath, mmproj_path
        except Exception:
            pass

    raise FileNotFoundError(f"No suitable model file found for {model_name}")


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
) -> str:
    """Clean up generated text by removing stop tokens and extra whitespace."""
    if not message:
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

        MAIN_GPU = int(getenv("MAIN_GPU"))
        GPU_LAYERS = int(getenv("GPU_LAYERS"))

        if torch.cuda.is_available() and GPU_LAYERS == -1:
            # Reserve 5GB VRAM for TTS and STT
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3) - 5
            if vram == 3:
                vram = 1
            if vram <= 0:
                vram = 0
            logging.info(f"[LLM] {vram}GB of available VRAM detected.")
            GPU_LAYERS = vram - 1 if vram > 0 else 0
        if GPU_LAYERS == -2:
            GPU_LAYERS = -1

        self.model_name = model if model else DEFAULT_MODEL
        self.system_message = system_message
        self.params = {}

        # Initialize stop tokens
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
        if stop:
            if isinstance(stop, str):
                self.params["stop"].append(stop)
            else:
                for stop_string in stop:
                    if stop_string and stop_string not in self.params["stop"]:
                        self.params["stop"].append(stop_string)

        self.params["temperature"] = temperature if temperature else 1.31
        self.params["top_p"] = top_p if top_p else 0.95
        self.params["min_p"] = min_p if min_p else 0.05
        self.params["stream"] = stream
        self.params["presence_penalty"] = presence_penalty
        self.params["frequency_penalty"] = frequency_penalty
        self.params["logit_bias"] = logit_bias
        self.params["max_tokens"] = (
            int(getenv("LLM_MAX_TOKENS")) if max_tokens == 0 else max_tokens
        )
        self.params["top_k"] = kwargs.get("top_k", 20)

        # Download model and get paths
        model_path, mmproj_path = download_model(
            model_name=self.model_name, models_dir=models_dir
        )

        # Initialize xllamacpp
        logging.info(f"[LLM] Loading {self.model_name} with xllamacpp...")

        self.xlc_params = xlc.CommonParams()
        self.xlc_params.model.path = model_path
        self.xlc_params.n_ctx = int(getenv("LLM_MAX_TOKENS"))
        self.xlc_params.n_batch = int(getenv("LLM_BATCH_SIZE"))
        self.xlc_params.n_gpu_layers = GPU_LAYERS
        self.xlc_params.main_gpu = MAIN_GPU
        self.xlc_params.warmup = True

        # Set multimodal projector path if available (for vision models)
        self.is_vision = False
        if mmproj_path:
            self.xlc_params.mmproj.path = mmproj_path
            self.is_vision = True
            logging.info(f"[LLM] Vision enabled with mmproj: {mmproj_path}")

        # Create the server instance
        self.server = xlc.Server(self.xlc_params)

        self.model_list = get_models()
        logging.info(f"[LLM] {self.model_name} loaded successfully with xllamacpp.")

    def chat(self, messages: List[Dict], **kwargs) -> dict:
        """Handle chat completions using xllamacpp server."""
        # Build the request payload
        chat_request = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.params["max_tokens"]),
            "temperature": kwargs.get("temperature", self.params["temperature"]),
            "top_p": kwargs.get("top_p", self.params["top_p"]),
            "stream": kwargs.get("stream", False),
        }

        # Add system message if not present
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and self.system_message:
            chat_request["messages"] = [
                {"role": "system", "content": self.system_message}
            ] + messages

        # Call xllamacpp server
        result = self.server.handle_chat_completions(chat_request)

        if isinstance(result, dict) and "error" in result:
            logging.error(f"[LLM] Chat completion error: {result}")
            raise Exception(result.get("error", {}).get("message", "Unknown error"))

        # Clean the response content
        if isinstance(result, dict) and result.get("choices") and not kwargs.get("stream", False):
            content = result["choices"][0].get("message", {}).get("content", "")
            result["choices"][0]["message"]["content"] = clean(
                message=content,
                stop_tokens=self.params["stop"],
            )

        return result

    def completion(self, prompt: str, **kwargs) -> dict:
        """Handle text completions using xllamacpp server."""
        completion_request = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.params["max_tokens"]),
            "temperature": kwargs.get("temperature", self.params["temperature"]),
            "top_p": kwargs.get("top_p", self.params["top_p"]),
            "stream": kwargs.get("stream", False),
        }

        result = self.server.handle_completions(completion_request)

        if isinstance(result, dict) and "error" in result:
            logging.error(f"[LLM] Completion error: {result}")
            raise Exception(result.get("error", {}).get("message", "Unknown error"))

        # Clean the response text and add text field for compatibility
        if isinstance(result, dict) and result.get("choices") and not kwargs.get("stream", False):
            text = result["choices"][0].get("text", "")
            if not text:
                # If text is empty, try to get from message content
                text = result["choices"][0].get("message", {}).get("content", "")
            result["choices"][0]["text"] = clean(
                message=text,
                stop_tokens=self.params["stop"],
            )

        return result

    def generate(self, prompt, **kwargs) -> dict:
        """Generate text using chat format."""
        messages = [{"role": "user", "content": prompt}]
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        return self.chat(messages=messages, **kwargs)

    def models(self) -> List[dict]:
        """Return list of available models."""
        return self.model_list


if __name__ == "__main__":
    logging.info(f"[LLM] Downloading {DEFAULT_MODEL} model...")
    download_model(model_name=DEFAULT_MODEL, models_dir="models")
