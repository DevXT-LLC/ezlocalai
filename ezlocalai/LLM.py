import xllamacpp as xlc
from huggingface_hub import hf_hub_download
from typing import List, Optional, Dict, Tuple
import os
import re
import torch
import logging
import json
import math
from Globals import getenv

DEFAULT_MODEL = getenv("DEFAULT_MODEL")


def get_gpu_count() -> int:
    """Get the number of available CUDA GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_free_vram_per_gpu() -> List[float]:
    """Get FREE (available) VRAM for each GPU in GB.

    Uses torch.cuda.mem_get_info() which returns actual free memory
    accounting for other processes using the GPU.
    """
    if not torch.cuda.is_available():
        return []

    free_vram = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_vram.append(free / (1024**3))
    return free_vram


def get_total_vram_per_gpu() -> List[float]:
    """Get total VRAM for each GPU in GB."""
    if not torch.cuda.is_available():
        return []

    total_vram = []
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        total_vram.append(total / (1024**3))
    return total_vram


def get_total_vram_all_gpus() -> float:
    """Get total VRAM across all GPUs in GB."""
    if torch.cuda.is_available():
        total = 0.0
        for i in range(torch.cuda.device_count()):
            total += torch.cuda.get_device_properties(i).total_memory
        return total / (1024**3)
    return 0.0


def get_total_free_vram() -> float:
    """Get total FREE VRAM across all GPUs in GB."""
    return sum(get_free_vram_per_gpu())


def calculate_tensor_split_from_free_vram() -> list:
    """Calculate tensor split ratios based on FREE (available) VRAM per GPU.

    Returns a list of 128 floats (xllamacpp expects exactly 128).
    Non-zero values indicate relative FREE VRAM proportions for each GPU.
    """
    if not torch.cuda.is_available():
        return [0.0] * 128

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return [0.0] * 128

    # Get FREE VRAM for each GPU
    free_vram_per_gpu = get_free_vram_per_gpu()
    total_free_vram = sum(free_vram_per_gpu)

    # Calculate proportional splits based on FREE VRAM
    tensor_split = [0.0] * 128
    for i, free_vram in enumerate(free_vram_per_gpu):
        tensor_split[i] = free_vram / total_free_vram if total_free_vram > 0 else 0.0

    return tensor_split


def calculate_tensor_split() -> list:
    """Calculate tensor split ratios based on available VRAM per GPU.

    DEPRECATED: Use calculate_tensor_split_from_free_vram() for accurate splits.
    This function uses total VRAM which doesn't account for other processes.

    Returns a list of 128 floats (xllamacpp expects exactly 128).
    Non-zero values indicate relative VRAM proportions for each GPU.
    """
    if not torch.cuda.is_available():
        return [0.0] * 128

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return [0.0] * 128

    # Get VRAM for each GPU
    vram_per_gpu = []
    for i in range(gpu_count):
        vram = torch.cuda.get_device_properties(i).total_memory
        vram_per_gpu.append(vram)

    total_vram = sum(vram_per_gpu)

    # Calculate proportional splits
    tensor_split = [0.0] * 128
    for i, vram in enumerate(vram_per_gpu):
        tensor_split[i] = vram / total_vram if total_vram > 0 else 0.0

    return tensor_split


def parse_tensor_split_env() -> list:
    """Parse TENSOR_SPLIT environment variable.

    Format: comma-separated floats, e.g., "0.5,0.5" for two equal GPUs.
    Returns None if not set or empty, otherwise returns 128-element list.
    """
    tensor_split_str = getenv("TENSOR_SPLIT")
    if not tensor_split_str or tensor_split_str.strip() == "":
        return None

    try:
        values = [float(v.strip()) for v in tensor_split_str.split(",") if v.strip()]
        if not values:
            return None

        # Pad to 128 elements
        tensor_split = [0.0] * 128
        for i, v in enumerate(values[:128]):
            tensor_split[i] = v
        return tensor_split
    except ValueError:
        logging.warning(f"[LLM] Invalid TENSOR_SPLIT format: {tensor_split_str}")
        return None


def get_models():
    """Return a list of available models from DEFAULT_MODEL config."""
    model_config = getenv("DEFAULT_MODEL")
    models = []
    if model_config.lower() != "none":
        for model_entry in model_config.split(","):
            model_entry = model_entry.strip()
            if model_entry:
                # Parse model@max_tokens format
                if "@" in model_entry:
                    model_name = model_entry.rsplit("@", 1)[0]
                else:
                    model_name = model_entry
                models.append(
                    {"id": model_name, "object": "model", "owned_by": "ezlocalai"}
                )
    return models


def download_model(model_name: str = "", models_dir: str = "models") -> tuple:
    """
    Download a model from HuggingFace Hub.
    Returns tuple of (model_path, mmproj_path) where mmproj_path may be None.

    First checks if any GGUF file already exists in the model directory (from startup download),
    and uses that to avoid downloading a different quantization.
    """
    global DEFAULT_MODEL
    model_name = model_name if model_name else DEFAULT_MODEL

    if "/" not in model_name:
        model_name = "unsloth/" + model_name + "-GGUF"

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
            logging.debug(f"[LLM] Downloaded mmproj: {mmproj_file}")
            break
        except Exception:
            pass

    # First, check if any GGUF model file already exists in the directory
    # This ensures we use whatever was downloaded at startup rather than re-downloading
    if os.path.exists(model_dir):
        existing_gguf_files = [
            f
            for f in os.listdir(model_dir)
            if f.endswith(".gguf") and "mmproj" not in f.lower()
        ]
        if existing_gguf_files:
            # Use the first existing model file (there should typically be only one)
            existing_file = existing_gguf_files[0]
            filepath = os.path.join(model_dir, existing_file)
            logging.debug(f"[LLM] Using existing model: {existing_file}")
            return filepath, mmproj_path

    # No existing model found - download based on QUANT_TYPE preference
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

    # Download the model
    logging.debug(f"[LLM] Downloading {model}...")
    for filename in potential_filenames:
        filepath = os.path.join(model_dir, filename)
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=model_dir,
            )
            logging.debug(f"[LLM] Downloaded {model} successfully!")
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
        gpu_layers: int = None,  # Override GPU_LAYERS env var if provided
        main_gpu: int = None,  # Override MAIN_GPU env var if provided
        tensor_split: list = None,  # Override tensor split if provided
        batch_size: int = None,  # Override LLM_BATCH_SIZE env var if provided
        **kwargs,
    ):
        global DEFAULT_MODEL

        # Use provided main_gpu if specified, otherwise fall back to env var
        MAIN_GPU = main_gpu if main_gpu is not None else int(getenv("MAIN_GPU", "0"))

        # Use provided gpu_layers if specified, otherwise fall back to env var or auto-detect
        gpu_layers_env = getenv("GPU_LAYERS", "")
        if gpu_layers is not None:
            GPU_LAYERS = gpu_layers
        elif gpu_layers_env:
            GPU_LAYERS = int(gpu_layers_env)
        else:
            # Auto-detect: use -1 which triggers VRAM-based calculation on GPU, 0 on CPU
            GPU_LAYERS = -1 if torch.cuda.is_available() else 0

        # Multi-GPU detection and smart GPU selection
        self.gpu_count = get_gpu_count()
        self.tensor_split = tensor_split  # Allow external tensor_split to be passed
        self.main_gpu = MAIN_GPU

        if torch.cuda.is_available() and GPU_LAYERS == -1:
            # Get FREE VRAM per GPU (not total) - this accounts for other processes
            free_vram_per_gpu = get_free_vram_per_gpu()
            total_free_vram = sum(free_vram_per_gpu)
            total_vram = get_total_vram_all_gpus()

            if self.gpu_count > 1:
                logging.debug(f"[LLM] Multi-GPU detected: {self.gpu_count} GPUs")
                for i, free in enumerate(free_vram_per_gpu):
                    total_gpu = torch.cuda.get_device_properties(i).total_memory / (
                        1024**3
                    )
                    logging.debug(
                        f"[LLM]   GPU {i}: {free:.1f}GB free / {total_gpu:.1f}GB total"
                    )
                logging.debug(
                    f"[LLM] Total: {total_free_vram:.1f}GB free / {round(total_vram)}GB total"
                )

                # GPU strategy is managed by Pipes.determine_gpu_strategy.
                # Only fall back to env-based tensor split if no strategy was provided
                # (i.e. LLM was created directly without going through _load_llm_resilient).
                if self.tensor_split is None:
                    self.tensor_split = parse_tensor_split_env()

                if self.tensor_split:
                    logging.debug(
                        f"[LLM] Tensor split ratios: {self.tensor_split[:self.gpu_count]}"
                    )
                else:
                    logging.warning(
                        f"[LLM] Multi-GPU detected but no tensor_split set - "
                        f"llama.cpp will auto-distribute across all GPUs by VRAM ratio"
                    )
            else:
                logging.debug(
                    f"[LLM] {total_free_vram:.1f}GB of available VRAM detected."
                )

            # -1 means "offload all layers to GPU" in xllamacpp
            # Only fall back to CPU (0) if there's essentially no VRAM
            GPU_LAYERS = -1 if total_free_vram > 1 else 0
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
        # Use provided max_tokens, or fall back to env var
        effective_max_tokens = (
            max_tokens if max_tokens > 0 else int(getenv("LLM_MAX_TOKENS"))
        )
        self.params["max_tokens"] = effective_max_tokens
        self.params["top_k"] = kwargs.get("top_k", 20)

        # Download model and get paths
        model_path, mmproj_path = download_model(
            model_name=self.model_name, models_dir=models_dir
        )

        # Initialize xllamacpp
        logging.debug(
            f"[LLM] Loading {self.model_name} with xllamacpp (context: {effective_max_tokens})"
        )

        self.xlc_params = xlc.CommonParams()
        self.xlc_params.model.path = model_path
        self.xlc_params.n_ctx = effective_max_tokens
        # Use provided batch_size, or scale dynamically based on context size
        if batch_size is not None:
            self.xlc_params.n_batch = batch_size
        else:
            default_batch = int(getenv("LLM_BATCH_SIZE", "2048"))
            self.xlc_params.n_batch = default_batch
        # Set n_ubatch (physical micro-batch / compute graph size) to reduce
        # compute buffer VRAM usage. Default 512 -> 493MB compute buffer.
        # Reducing to 256 roughly halves it, giving more room for KV/mmproj.
        default_ubatch = int(getenv("LLM_UBATCH_SIZE", "256"))
        self.xlc_params.n_ubatch = min(default_ubatch, self.xlc_params.n_batch)
        logging.debug(
            f"[LLM] Batch size: {self.xlc_params.n_batch}, ubatch: {self.xlc_params.n_ubatch} for context {effective_max_tokens}"
        )
        self.xlc_params.n_gpu_layers = GPU_LAYERS
        self.xlc_params.main_gpu = (
            self.main_gpu
        )  # Use self.main_gpu which may be overridden
        self.xlc_params.warmup = True

        # Enable flash attention for significantly faster inference on CUDA
        self.xlc_params.flash_attn_type = (
            xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED
        )

        # Quantize KV cache to reduce VRAM usage (~8x savings vs f16)
        # q4_0 enables full 262K context on 24GB GPUs for models like Qwen3.5
        # where only 10/40 layers use standard attention (rest are DeltaNet state-space)
        self.xlc_params.cache_type_k = xlc.ggml_type.GGML_TYPE_Q4_0
        self.xlc_params.cache_type_v = xlc.ggml_type.GGML_TYPE_Q4_0

        # Apply tensor split for multi-GPU setups
        if self.tensor_split and self.gpu_count > 1:
            try:
                # xllamacpp expects tensor_split as a list of 128 floats
                for i, ratio in enumerate(self.tensor_split):
                    self.xlc_params.tensor_split[i] = ratio
                logging.debug(
                    f"[LLM] Applied tensor split across {self.gpu_count} GPUs (main_gpu={self.main_gpu})"
                )
            except Exception as e:
                logging.warning(f"[LLM] Failed to set tensor_split: {e}")
        else:
            logging.debug(
                f"[LLM] Loading on GPU {self.main_gpu} only (no tensor split)"
            )

        # Set multimodal projector path if available (for vision models)
        self.is_vision = False
        if mmproj_path:
            self.xlc_params.mmproj.path = mmproj_path
            self.is_vision = True
            logging.debug(f"[LLM] Vision enabled with mmproj: {mmproj_path}")

        # Create the server instance
        self.server = xlc.Server(self.xlc_params)

        # Verify server initialized correctly with a minimal test completion
        try:
            test_result = self.server.handle_completions(
                {
                    "prompt": "Hi",
                    "max_tokens": 1,
                }
            )
            if isinstance(test_result, dict) and "error" in test_result:
                raise RuntimeError(f"LLM server test failed: {test_result}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize LLM server for {self.model_name}: {e}"
            )

        self.model_list = get_models()
        logging.debug(f"[LLM] {self.model_name} loaded successfully with xllamacpp.")

    def chat(self, messages: List[Dict], **kwargs):
        """Handle chat completions using xllamacpp server.

        Returns:
            dict: Non-streaming response with choices
            generator: Streaming response yielding chunk dicts when stream=True
        """
        stream = kwargs.get("stream", False)
        logging.debug(
            f"[LLM] chat() called with stream={stream}, kwargs keys: {kwargs.keys()}"
        )

        # Build the request payload
        chat_request = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.params["max_tokens"]),
            "temperature": kwargs.get("temperature", self.params["temperature"]),
            "top_p": kwargs.get("top_p", self.params["top_p"]),
            "stream": stream,
        }

        # Forward chat_template_kwargs for thinking mode control
        # e.g. {"enable_thinking": False} to disable <think> tags
        if "chat_template_kwargs" in kwargs:
            chat_request["chat_template_kwargs"] = kwargs["chat_template_kwargs"]

        # Add system message if not present
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and self.system_message:
            chat_request["messages"] = [
                {"role": "system", "content": self.system_message}
            ] + messages

        # Handle streaming with callback
        if stream:
            logging.debug(f"[LLM] Calling _chat_stream for streaming response")
            return self._chat_stream(chat_request)

        # Non-streaming call
        result = self.server.handle_chat_completions(chat_request)

        # Check for error - xllamacpp can return errors in two formats:
        # 1. {"error": {"message": "...", "n_prompt_tokens": ...}}
        # 2. {"code": 400, "message": "...", "n_prompt_tokens": ...}  (top-level)
        if isinstance(result, dict):
            error_info = None
            if "error" in result:
                error_info = result.get("error", {})
            elif (
                result.get("code") == 400
                or result.get("type") == "exceed_context_size_error"
            ):
                # Error at top level
                error_info = result

            if error_info:
                logging.error(f"[LLM] Chat completion error: {result}")
                error_msg = error_info.get("message", "Unknown error")
                # Include token counts in error message for context size handling
                n_prompt_tokens = error_info.get("n_prompt_tokens")
                n_ctx = error_info.get("n_ctx")
                if n_prompt_tokens:
                    error_msg = f"{error_msg} [n_prompt_tokens={n_prompt_tokens}, n_ctx={n_ctx or 'unknown'}]"
                raise Exception(error_msg)

        # Clean the response content (only for non-streaming)
        if isinstance(result, dict) and result.get("choices"):
            content = result["choices"][0].get("message", {}).get("content", "")
            result["choices"][0]["message"]["content"] = clean(
                message=content,
                stop_tokens=self.params["stop"],
            )

        # Ensure the response contains the actual model name used
        if isinstance(result, dict):
            result["model"] = self.model_name

        return result

    def _chat_stream(self, chat_request: dict):
        """Handle streaming chat completions using xllamacpp callback.

        Returns a generator that yields OpenAI-compatible chunk dicts.

        xllamacpp's handle_chat_completions is synchronous - it blocks and calls
        the callback with chunks during execution. The callback receives either:
        - An array of chunk dicts (for streaming responses)
        - A single dict (for partial/final responses)

        We use a thread to run it and a queue to collect chunks for the generator to yield.
        """
        import queue
        import threading
        import time

        # Queue to collect chunks from callback
        chunk_queue = queue.Queue()
        generation_complete = threading.Event()
        error_holder = [None]  # Use list to allow modification in nested function

        def streaming_callback(chunk_data):
            """Callback function called by xllamacpp for streaming chunks.

            Args:
                chunk_data: Can be:
                    - A list of chunk dicts (xllamacpp bundles streaming deltas)
                    - A single chunk dict
                    - An error dict with 'code' key

            Returns:
                False to continue receiving chunks, True to stop early
            """
            try:
                logging.debug(
                    f"[LLM] Stream callback received: type={type(chunk_data)}"
                )

                # Check for error response
                if isinstance(chunk_data, dict) and "code" in chunk_data:
                    logging.error(f"[LLM] Stream callback error: {chunk_data}")
                    # Include token counts in error message for context size handling
                    error_msg = chunk_data.get("message", str(chunk_data))
                    n_prompt_tokens = chunk_data.get("n_prompt_tokens")
                    n_ctx = chunk_data.get("n_ctx")
                    if n_prompt_tokens:
                        error_msg = f"{error_msg} [n_prompt_tokens={n_prompt_tokens}, n_ctx={n_ctx or 'unknown'}]"
                    error_holder[0] = Exception(error_msg)
                    return True  # Stop on error

                # xllamacpp returns a list of deltas for streaming
                if isinstance(chunk_data, list):
                    logging.debug(f"[LLM] Received {len(chunk_data)} chunks in array")
                    for chunk in chunk_data:
                        chunk_queue.put(chunk)
                else:
                    # Single chunk
                    logging.debug(f"[LLM] Received single chunk")
                    chunk_queue.put(chunk_data)

                return False  # Continue receiving chunks
            except Exception as e:
                logging.error(f"[LLM] Streaming callback error: {e}")
                error_holder[0] = e
                return True  # Stop on error

        def run_inference():
            """Run the inference in a separate thread."""
            try:
                # Ensure stream=True in the request
                request_copy = chat_request.copy()
                request_copy["stream"] = True
                logging.debug(f"[LLM] Starting streaming inference with callback")
                # Pass callback as second positional argument (not keyword)
                # xllamacpp will call streaming_callback for each chunk/batch
                result = self.server.handle_chat_completions(
                    request_copy, streaming_callback
                )
                logging.debug(
                    f"[LLM] handle_chat_completions returned: type={type(result)}"
                )

                # No need to check result since xllamacpp calls the callback directly
                logging.debug(f"[LLM] Streaming inference completed")
            except Exception as e:
                logging.error(f"[LLM] Streaming inference error: {e}")
                import traceback

                logging.error(f"[LLM] Traceback: {traceback.format_exc()}")
                error_holder[0] = e
            finally:
                generation_complete.set()

        # Start inference in background thread
        inference_thread = threading.Thread(target=run_inference, daemon=True)
        inference_thread.start()

        # Yield chunks as they come in
        chunk_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        chunks_yielded = 0
        last_keepalive = time.time()
        keepalive_interval = 5.0  # Send keepalive every 5 seconds during processing

        while not generation_complete.is_set() or not chunk_queue.empty():
            try:
                chunk_data = chunk_queue.get(timeout=0.1)
                chunks_yielded += 1
                logging.debug(
                    f"[LLM] Processing chunk {chunks_yielded}: {type(chunk_data)}"
                )

                # Check if this is already in OpenAI format
                if isinstance(chunk_data, dict):
                    if "choices" in chunk_data:
                        # Already in correct format, yield directly
                        logging.debug(
                            f"[LLM] Yielding OpenAI format chunk with choices"
                        )
                        yield chunk_data
                    elif "content" in chunk_data or "delta" in chunk_data:
                        # Wrap in OpenAI format
                        content = chunk_data.get(
                            "content", chunk_data.get("delta", {}).get("content", "")
                        )
                        yield {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    else:
                        # Unknown dict format, log it
                        logging.debug(f"[LLM] Unknown chunk dict format: {chunk_data}")
                elif isinstance(chunk_data, str):
                    # JSON string - parse it
                    try:
                        import json

                        parsed = json.loads(chunk_data)
                        if "choices" in parsed:
                            yield parsed
                        else:
                            logging.debug(
                                f"[LLM] Parsed JSON without choices: {parsed}"
                            )
                    except json.JSONDecodeError:
                        # Raw text chunk
                        yield {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk_data},
                                    "finish_reason": None,
                                }
                            ],
                        }
                else:
                    logging.debug(f"[LLM] Unexpected chunk type: {type(chunk_data)}")
            except queue.Empty:
                # Send keepalive during long waits (e.g., prompt processing)
                # This prevents client timeouts during the initial processing phase
                now = time.time()
                if now - last_keepalive >= keepalive_interval:
                    last_keepalive = now
                    # Yield an empty delta chunk as keepalive
                    yield {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},  # Empty delta acts as keepalive
                                "finish_reason": None,
                            }
                        ],
                    }
                continue
            except Exception as e:
                logging.error(f"[LLM] Error processing stream chunk: {e}")
                continue

        # Wait for thread to complete
        inference_thread.join(timeout=5.0)

        logging.debug(f"[LLM] Stream complete, yielded {chunks_yielded} chunks")

        # Check for errors
        if error_holder[0]:
            raise error_holder[0]

        # Only yield final stop chunk if we didn't get one from xllamacpp
        # The last chunk from xllamacpp should have finish_reason set
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

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

        # Check for error - xllamacpp can return errors in two formats:
        # 1. {"error": {"message": "...", "n_prompt_tokens": ...}}
        # 2. {"code": 400, "message": "...", "n_prompt_tokens": ...}  (top-level)
        if isinstance(result, dict):
            error_info = None
            if "error" in result:
                error_info = result.get("error", {})
            elif (
                result.get("code") == 400
                or result.get("type") == "exceed_context_size_error"
            ):
                # Error at top level
                error_info = result

            if error_info:
                logging.error(f"[LLM] Completion error: {result}")
                error_msg = error_info.get("message", "Unknown error")
                # Include token counts in error message for context size handling
                n_prompt_tokens = error_info.get("n_prompt_tokens")
                n_ctx = error_info.get("n_ctx")
                if n_prompt_tokens:
                    error_msg = f"{error_msg} [n_prompt_tokens={n_prompt_tokens}, n_ctx={n_ctx or 'unknown'}]"
                raise Exception(error_msg)

        # Clean the response text and add text field for compatibility
        if (
            isinstance(result, dict)
            and result.get("choices")
            and not kwargs.get("stream", False)
        ):
            text = result["choices"][0].get("text", "")
            if not text:
                # If text is empty, try to get from message content
                text = result["choices"][0].get("message", {}).get("content", "")
            result["choices"][0]["text"] = clean(
                message=text,
                stop_tokens=self.params["stop"],
            )

        # Ensure the response contains the actual model name used
        if isinstance(result, dict):
            result["model"] = self.model_name

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
    logging.debug(f"[LLM] Downloading {DEFAULT_MODEL} model...")
    download_model(model_name=DEFAULT_MODEL, models_dir="models")
