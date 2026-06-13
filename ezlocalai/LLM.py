import xllamacpp as xlc
from huggingface_hub import hf_hub_download, list_repo_files
from typing import Any, List, Optional, Dict, Tuple
import os
import re
import torch
import logging
import json
import math
from Globals import getenv

DEFAULT_MODEL = getenv("DEFAULT_MODEL")
MTP_SPEC_DRAFT_P_MIN_DEFAULT = 0.25
MTP_SPEC_DRAFT_N_MAX_MAX = 16


STREAM_CONTENT_KEYS = (
    "content",
    "text",
    "token",
    "response",
    "output",
    "output_text",
    "generated_text",
)
STREAM_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def _stream_text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_stream_text_value(item) for item in value)
    if isinstance(value, dict):
        for key in (*STREAM_CONTENT_KEYS, *STREAM_REASONING_KEYS):
            text = _stream_text_value(value.get(key))
            if text:
                return text
    return ""


def _first_stream_text(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        text = _stream_text_value(mapping.get(key))
        if text:
            return text
    return ""


def normalize_stream_chunk_delta(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw streaming chunk content into an OpenAI delta object."""
    raw_delta = chunk_data.get("delta", {})
    if not isinstance(raw_delta, dict):
        raw_delta = {}
    delta: Dict[str, Any] = {}
    content = _first_stream_text(chunk_data, STREAM_CONTENT_KEYS) or _first_stream_text(
        raw_delta, STREAM_CONTENT_KEYS
    )
    reasoning = _first_stream_text(
        chunk_data, STREAM_REASONING_KEYS
    ) or _first_stream_text(raw_delta, STREAM_REASONING_KEYS)
    if content:
        delta["content"] = content
    elif reasoning:
        delta["reasoning_content"] = reasoning
    return delta


def stream_chunk_has_assistant_text(chunk_data: Any) -> bool:
    """Return True when a streaming chunk contains content or reasoning text."""
    if isinstance(chunk_data, dict):
        choices = chunk_data.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    if stream_chunk_has_assistant_text(choice.get("delta", {})):
                        return True
                    if stream_chunk_has_assistant_text(choice.get("message", {})):
                        return True
                    if normalize_stream_chunk_delta(choice):
                        return True
            return False
        return bool(normalize_stream_chunk_delta(chunk_data))
    if isinstance(chunk_data, str):
        if not chunk_data.strip():
            return False
        try:
            return stream_chunk_has_assistant_text(json.loads(chunk_data))
        except Exception:
            return True
    return False


def stream_chunk_finish_reason(chunk_data: Any) -> str:
    if isinstance(chunk_data, dict):
        choices = chunk_data.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    reason = choice.get("finish_reason")
                    if reason:
                        return str(reason)
            return ""
        reason = chunk_data.get("finish_reason")
        return str(reason) if reason else ""
    if isinstance(chunk_data, str) and chunk_data.strip():
        try:
            return stream_chunk_finish_reason(json.loads(chunk_data))
        except Exception:
            return ""
    return ""


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


def is_mtp_model(model_name: str) -> bool:
    """Return True for GGUF repos with built-in multi-token prediction heads."""
    return "-mtp" in (model_name or "").lower()


def get_model_size_billions(model_name: str) -> float:
    """Best-effort parse of the largest parameter-size marker from a model name."""
    sizes = [
        float(match)
        for match in re.findall(
            r"(?<![a-z0-9])(\d+(?:\.\d+)?)b(?:[^a-z0-9]|$)",
            (model_name or "").lower(),
        )
    ]
    return max(sizes) if sizes else 0.0


def get_mtp_spec_draft_n_max(
    main_gpu: int = 0, model_name: str = ""
) -> Tuple[int, float]:
    """Choose MTP draft length from the primary GPU's total VRAM."""
    total_vram = get_total_vram_per_gpu()
    raw_override = str(getenv("MTP_SPEC_DRAFT_N_MAX", "auto") or "auto").strip()

    card_vram_gb = 0.0
    if total_vram:
        gpu_idx = main_gpu if 0 <= main_gpu < len(total_vram) else 0
        card_vram_gb = total_vram[gpu_idx]

    if raw_override.lower() not in {"", "auto", "0"}:
        try:
            override = int(raw_override)
        except (TypeError, ValueError):
            logging.warning(
                "[LLM] Invalid MTP_SPEC_DRAFT_N_MAX=%r; using auto", raw_override
            )
        else:
            return min(max(1, override), MTP_SPEC_DRAFT_N_MAX_MAX), card_vram_gb

    model_size_b = get_model_size_billions(model_name)
    if 0 < model_size_b <= 4 and card_vram_gb >= 20:
        return 4, card_vram_gb

    # Large MTP models lose acceptance quickly with deeper drafts. In practice
    # the 35B-A3B 5090/4090/3090 tests favored n_max=2 even when VRAM had room.
    return 2, card_vram_gb


def get_mtp_spec_draft_p_min() -> float:
    """Minimum draft token probability used for MTP speculative decoding."""
    raw_value = getenv("MTP_SPEC_DRAFT_P_MIN", str(MTP_SPEC_DRAFT_P_MIN_DEFAULT))
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logging.warning(
            "[LLM] Invalid MTP_SPEC_DRAFT_P_MIN=%r; using %.2f",
            raw_value,
            MTP_SPEC_DRAFT_P_MIN_DEFAULT,
        )
        return MTP_SPEC_DRAFT_P_MIN_DEFAULT

    if value < 0.0 or value > 1.0:
        clamped = min(max(value, 0.0), 1.0)
        logging.warning(
            "[LLM] MTP_SPEC_DRAFT_P_MIN=%s is outside 0.0-1.0; using %.2f",
            raw_value,
            clamped,
        )
        return clamped

    return value


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


def calculate_auto_batch_sizes(
    main_gpu: int = 0, effective_max_tokens: int = 0
) -> Tuple[int, int, str]:
    """Choose n_batch and n_ubatch from available VRAM and GPU generation.

    ``n_batch`` controls the logical prompt-processing chunk size. ``n_ubatch``
    controls the physical compute graph size and is the larger VRAM consumer.
    Bigger values improve prompt/prefill throughput on fast GPUs, but can OOM
    tight cards. This keeps explicit env overrides available while making the
    default adapt to the machine instead of using one fixed value everywhere.
    """
    if not torch.cuda.is_available():
        return 1024, 256, "cpu/default"

    try:
        device_count = torch.cuda.device_count()
        gpu_idx = main_gpu if 0 <= main_gpu < device_count else 0
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        cc = torch.cuda.get_device_capability(gpu_idx)
        cc_num = cc[0] * 10 + cc[1]
    except Exception as e:
        logging.debug(f"[LLM] auto batch VRAM probe failed: {e}; using safe defaults")
        return 2048, 512, "probe-failed/default"

    # Safe defaults for older/smaller GPUs.
    n_batch = 1024
    n_ubatch = 256

    if cc_num >= 89:  # Ada / Hopper / Blackwell
        if free_gb >= 28:
            n_batch, n_ubatch = 8192, 2048
        elif free_gb >= 20:
            n_batch, n_ubatch = 4096, 1024
        elif free_gb >= 12:
            n_batch, n_ubatch = 2048, 512
        elif free_gb >= 8:
            n_batch, n_ubatch = 1024, 512
    elif cc_num >= 80:  # Ampere
        if free_gb >= 22 and effective_max_tokens and effective_max_tokens <= 300_000:
            n_batch, n_ubatch = 4096, 1024
        elif free_gb >= 12:
            n_batch, n_ubatch = 2048, 512
        elif free_gb >= 8:
            n_batch, n_ubatch = 1024, 512
    else:
        if free_gb >= 16:
            n_batch, n_ubatch = 2048, 512

    if effective_max_tokens:
        n_batch = min(n_batch, max(1, int(effective_max_tokens)))
        n_ubatch = min(n_ubatch, n_batch)

    reason = (
        f"GPU {gpu_idx}, free={free_gb:.1f}GB/{total_gb:.1f}GB, "
        f"CC {cc_num // 10}.{cc_num % 10}"
    )
    return n_batch, n_ubatch, reason


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


def download_model(
    model_name: str = "",
    models_dir: str = "models",
    quantization_type: str = None,
    skip_mmproj: bool = False,
) -> tuple:
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

    if quantization_type is None:
        quantization_type = getenv("QUANT_TYPE")
    if quantization_type and "," in quantization_type:
        # If QUANT_TYPE is comma-separated and no explicit per-model override
        # was provided, default to the first value for backward compatibility.
        quantization_type = quantization_type.split(",", 1)[0].strip()
    model = model_name.split("/")[-1].split("-GGUF")[0]
    model_dir = os.path.join(models_dir, model)
    os.makedirs(model_dir, exist_ok=True)

    # Try to find or download multimodal projector files (for vision models).
    # First scan the local directory for ANY mmproj-like file so a pre-placed
    # file (e.g. mmproj-BF16.gguf) is recognised even if it isn't the first
    # entry in our hard-coded preference list. Without this scan the loop
    # below would issue a HEAD/GET to HF for every variant before reaching
    # the user's file, redownloading content unnecessarily.
    mmproj_path = None
    if not skip_mmproj and os.path.isdir(model_dir):
        for fname in os.listdir(model_dir):
            if "mmproj" in fname.lower() and fname.endswith(".gguf"):
                mmproj_path = os.path.join(model_dir, fname)
                break

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

    if not skip_mmproj and mmproj_path is None:
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
            # Prefer a file that matches the requested QUANT_TYPE
            matching_files = [
                f
                for f in existing_gguf_files
                if quantization_type and quantization_type in f
            ]
            if matching_files:
                existing_file = matching_files[0]
            elif not quantization_type:
                # No specific quant requested, use whatever exists
                existing_file = existing_gguf_files[0]
            else:
                # Existing files don't match requested quant - need to download
                existing_file = None
            if existing_file:
                filepath = os.path.join(model_dir, existing_file)
                logging.debug(f"[LLM] Using existing model: {existing_file}")
                return filepath, mmproj_path

    # No existing model found - use list_repo_files + pattern matching
    # to find the best quantization, consistent with precache.py and Pipes.py
    logging.debug(f"[LLM] Downloading {model}...")
    try:
        files = list_repo_files(model_name)
        gguf_files = [
            f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()
        ]

        if gguf_files:
            best_file = None
            patterns = [quantization_type, "Q4_K", "Q5_K", "Q6_K", "Q8"]
            for pattern in patterns:
                for f in gguf_files:
                    if pattern in f:
                        best_file = f
                        break
                if best_file:
                    break

            if not best_file:
                best_file = gguf_files[0]

            filepath = os.path.join(model_dir, best_file)
            hf_hub_download(
                repo_id=model_name,
                filename=best_file,
                local_dir=model_dir,
            )
            logging.debug(f"[LLM] Downloaded {model} ({best_file}) successfully!")
            return filepath, mmproj_path
    except Exception as e:
        logging.error(f"[LLM] Failed to list/download from {model_name}: {e}")

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
        n_parallel: int = None,  # Override N_PARALLEL env var if provided
        quant_type: str = None,  # Override QUANT_TYPE env var if provided
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
        # Context size for the model (how much input it can accept)
        effective_max_tokens = (
            max_tokens if max_tokens > 0 else int(getenv("LLM_MAX_TOKENS"))
        )
        # Default output limit (how many tokens the model can generate per request).
        # This is separate from context size — context_size controls n_ctx (input),
        # while this controls the default max_tokens for generation (output).
        # Using context_size as output limit causes runaway generation (40K+ tokens).
        self.params["max_tokens"] = int(getenv("LLM_MAX_OUTPUT_TOKENS", "8192"))
        self.params["top_k"] = kwargs.get("top_k", 20)

        # Download model and get paths
        model_path, mmproj_path = download_model(
            model_name=self.model_name,
            models_dir=models_dir,
            quantization_type=quant_type,
        )

        # Auto-prefer fastest single GPU when the model fits on it alone.
        #
        # On a mixed-GPU host (e.g. 5090 + 3090) llama.cpp's default behaviour
        # is to split the model across BOTH cards by VRAM ratio.  That tanks
        # prompt-processing throughput because every PP step has to wait for
        # the slowest GPU.  When the model file plus a reasonable margin for
        # KV + compute buffer fits in the fastest GPU's free VRAM, we restrict
        # tensor_split to put 100% there and pin main_gpu to that index.
        #
        # Disable with LLM_PREFER_FASTEST_GPU=false, or override with explicit
        # TENSOR_SPLIT or MAIN_GPU env vars.
        try:
            prefer_fastest = getenv(
                "LLM_PREFER_FASTEST_GPU", "true"
            ).strip().lower() not in ("false", "0", "no")
            user_split = (getenv("TENSOR_SPLIT") or "").strip()
            user_main = getenv("MAIN_GPU")
            if (
                prefer_fastest
                and not user_split
                and user_main is None
                and self.gpu_count > 1
                and self.tensor_split is None
                and torch.cuda.is_available()
                and os.path.exists(model_path)
            ):
                model_bytes = os.path.getsize(model_path)
                model_gb = model_bytes / (1024**3)
                # Headroom multiplier: model weights + KV cache + compute buffer.
                # 1.4x is a reasonable lower bound for typical contexts; users
                # can tune via LLM_SINGLE_GPU_HEADROOM (e.g. 1.6 for big ctx).
                try:
                    headroom = float(getenv("LLM_SINGLE_GPU_HEADROOM", "1.4"))
                except ValueError:
                    headroom = 1.4
                needed_gb = model_gb * headroom
                # Rank GPUs by compute capability (highest first), tie-break
                # by free VRAM.
                free_per_gpu = get_free_vram_per_gpu()
                ranked = []
                for i in range(self.gpu_count):
                    try:
                        cc = torch.cuda.get_device_capability(i)
                        cc_num = cc[0] * 10 + cc[1]
                    except Exception:
                        cc_num = 0
                    ranked.append(
                        (cc_num, free_per_gpu[i] if i < len(free_per_gpu) else 0.0, i)
                    )
                ranked.sort(key=lambda x: (-x[0], -x[1]))
                # Pick the first GPU (by rank) whose free VRAM covers needed_gb
                chosen = None
                for cc_num, free_gb, idx in ranked:
                    if free_gb >= needed_gb:
                        chosen = (idx, cc_num, free_gb)
                        break
                if chosen is not None:
                    idx, cc_num, free_gb = chosen
                    split = [0.0] * 128
                    split[idx] = 1.0
                    self.tensor_split = split
                    self.main_gpu = idx
                    logging.info(
                        f"[LLM] Single-GPU mode: model {model_gb:.1f}GB * "
                        f"{headroom:.2f} = {needed_gb:.1f}GB needed, GPU {idx} "
                        f"(CC {cc_num // 10}.{cc_num % 10}) has {free_gb:.1f}GB "
                        f"free; pinning to GPU {idx} for max prompt-processing speed"
                    )
                else:
                    logging.info(
                        f"[LLM] Multi-GPU split required: model {model_gb:.1f}GB * "
                        f"{headroom:.2f} = {needed_gb:.1f}GB needed, no single GPU "
                        f"has enough free VRAM (free per GPU: "
                        f"{[round(f, 1) for f in free_per_gpu[:self.gpu_count]]})"
                    )
        except Exception as e:
            logging.debug(f"[LLM] single-GPU auto-pick skipped: {e}")

        # Initialize xllamacpp
        logging.debug(
            f"[LLM] Loading {self.model_name} with xllamacpp (context: {effective_max_tokens})"
        )

        self.xlc_params = xlc.CommonParams()
        self.xlc_params.model.path = model_path
        self.xlc_params.n_ctx = effective_max_tokens

        auto_batch, auto_ubatch, auto_batch_reason = calculate_auto_batch_sizes(
            main_gpu=self.main_gpu, effective_max_tokens=effective_max_tokens
        )

        # Use provided batch_size, explicit LLM_BATCH_SIZE, or auto-size from
        # available VRAM. ``auto``/``0`` means adapt to the current GPU.
        if batch_size is not None:
            self.xlc_params.n_batch = batch_size
            batch_source = "override"
        else:
            batch_env = (getenv("LLM_BATCH_SIZE", "auto") or "auto").strip().lower()
            if batch_env in ("", "auto", "0"):
                self.xlc_params.n_batch = auto_batch
                batch_source = f"auto ({auto_batch_reason})"
            else:
                try:
                    self.xlc_params.n_batch = int(batch_env)
                    batch_source = "env"
                except ValueError:
                    self.xlc_params.n_batch = auto_batch
                    batch_source = f"invalid-env-auto ({auto_batch_reason})"
                    logging.warning(
                        f"[LLM] Invalid LLM_BATCH_SIZE={batch_env!r}, using auto"
                    )
        # Set n_ubatch (physical micro-batch / compute graph size). This is
        # the dominant prompt-processing throughput knob: larger ubatch means
        # each compute step processes more tokens, which lets fast GPUs
        # (4090/5090) actually saturate their tensor cores. Cost: VRAM for
        # the compute buffer scales roughly linearly with ubatch.
        #
        # Auto-scaling: when LLM_UBATCH_SIZE is unset (or "auto"), pick a
        # value based on the device's compute capability so high-end GPUs
        # are not bottlenecked at 256 — but stay conservative on older /
        # smaller cards where the compute buffer matters more.
        ubatch_env = (getenv("LLM_UBATCH_SIZE") or "auto").strip().lower()
        if ubatch_env in ("", "auto", "0"):
            default_ubatch = auto_ubatch
            ubatch_source = f"auto ({auto_batch_reason})"
        else:
            try:
                default_ubatch = int(ubatch_env)
                ubatch_source = "env"
            except ValueError:
                logging.warning(
                    f"[LLM] Invalid LLM_UBATCH_SIZE={ubatch_env!r}, falling back to 256"
                )
                default_ubatch = 256
                ubatch_source = "invalid-env-default"
        self.xlc_params.n_ubatch = min(default_ubatch, self.xlc_params.n_batch)
        logging.info(
            f"[LLM] Batch size: {self.xlc_params.n_batch} ({batch_source}), "
            f"ubatch: {self.xlc_params.n_ubatch} ({ubatch_source}) "
            f"for context {effective_max_tokens}"
        )
        # Prompt cache (host-RAM checkpoint cache).
        # SWA / hybrid / recurrent models (e.g. Qwen3, Gemma2) cannot reuse
        # cached prompt state across requests — llama.cpp logs
        # "forcing full prompt re-processing due to lack of cache data"
        # every request, and the cache machinery becomes pure overhead
        # (100-200ms/request + growing host RAM + KV pressure).
        # Default to disabled; set LLM_PROMPT_CACHE_MIB > 0 to opt in.
        try:
            cache_mib = int(getenv("LLM_PROMPT_CACHE_MIB", "0"))
        except ValueError:
            cache_mib = 0
        self.xlc_params.cache_ram_mib = cache_mib
        self.xlc_params.n_gpu_layers = GPU_LAYERS
        self.gpu_layers = GPU_LAYERS  # Expose for runtime inspection
        self.xlc_params.main_gpu = (
            self.main_gpu
        )  # Use self.main_gpu which may be overridden
        self.xlc_params.warmup = True

        # Set reasoning budget to limit thinking tokens per response
        # -1 = unlimited (default), 0 = disable thinking, N = max thinking tokens
        reasoning_budget = int(getenv("REASONING_BUDGET", "-1"))
        if reasoning_budget >= 0:
            self.xlc_params.reasoning_budget = reasoning_budget
            logging.info(f"[LLM] Reasoning budget set to {reasoning_budget} tokens")
        else:
            logging.debug("[LLM] Reasoning budget unlimited (default)")

        # Enable flash attention for significantly faster inference on CUDA
        self.xlc_params.flash_attn_type = (
            xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED
        )

        # KV cache type: q4_0 saves ~8x VRAM vs f16, but requires FA_ALL_QUANTS
        # compiled into the CUDA backend. Use f16 on Jetson/embedded where FA_ALL_QUANTS
        # may not be available, or q4_0 on desktop GPUs with ample VRAM.
        kv_cache_type = getenv("KV_CACHE_TYPE", "q4_0").lower().strip()
        kv_type_map = {
            "f16": xlc.ggml_type.GGML_TYPE_F16,
            "f32": xlc.ggml_type.GGML_TYPE_F32,
            "q8_0": xlc.ggml_type.GGML_TYPE_Q8_0,
            "q4_0": xlc.ggml_type.GGML_TYPE_Q4_0,
        }
        kv_type = kv_type_map.get(kv_cache_type, xlc.ggml_type.GGML_TYPE_Q4_0)
        self.xlc_params.cache_type_k = kv_type
        self.xlc_params.cache_type_v = kv_type
        if kv_cache_type != "q4_0":
            logging.info(f"[LLM] KV cache type: {kv_cache_type}")

        # Parallel inference: n_parallel creates multiple KV cache slots within
        # one model load. The total KV cache (n_ctx) is divided across slots, so
        # VRAM stays constant — only per-slot context shrinks.
        # 0 = auto-scale, 1 = single slot, N = fixed.
        if n_parallel is not None:
            resolved_parallel = n_parallel
        else:
            resolved_parallel = int(getenv("N_PARALLEL", "0"))

        if resolved_parallel == 0:
            # Auto-scale: target ~32K tokens per slot to handle most conversations
            # comfortably, capped at 16 to balance parallelism vs per-slot context.
            # VRAM is constant regardless of n_parallel — only per-slot context changes.
            target_per_slot = 32768
            auto_parallel = max(1, effective_max_tokens // target_per_slot)
            resolved_parallel = min(auto_parallel, 16)

        if is_mtp_model(self.model_name) and resolved_parallel != 1:
            logging.info(
                f"[LLM] MTP model detected; forcing n_parallel=1 "
                f"(requested {resolved_parallel})"
            )
            resolved_parallel = 1

        self.n_parallel = resolved_parallel
        self.xlc_params.n_parallel = resolved_parallel
        # cont_batching is True by default in xllamacpp, but be explicit
        self.xlc_params.cont_batching = True

        if is_mtp_model(self.model_name):
            spec_draft_n_max, card_vram_gb = get_mtp_spec_draft_n_max(
                self.main_gpu, self.model_name
            )
            spec_draft_p_min = get_mtp_spec_draft_p_min()
            self.xlc_params.speculative.types = [
                xlc.common_speculative_type.COMMON_SPECULATIVE_TYPE_DRAFT_MTP
            ]
            self.xlc_params.speculative.draft.n_max = spec_draft_n_max
            self.xlc_params.speculative.draft.p_min = spec_draft_p_min
            logging.info(
                f"[LLM] MTP speculative decoding enabled: spec_type=draft-mtp, "
                f"spec_draft_n_max={spec_draft_n_max}, "
                f"spec_draft_p_min={spec_draft_p_min:.2f} "
                f"(GPU {self.main_gpu}, total VRAM {card_vram_gb:.1f}GB)"
            )

        if resolved_parallel > 1:
            per_slot_ctx = effective_max_tokens // resolved_parallel
            logging.info(
                f"[LLM] Parallel inference: {resolved_parallel} slots, "
                f"{per_slot_ctx:,} tokens/slot "
                f"(total context: {effective_max_tokens:,})"
            )
        else:
            logging.debug("[LLM] Single inference slot")

        # Apply tensor split for multi-GPU setups
        if self.tensor_split and self.gpu_count > 1:
            try:
                # Check if this is a single-GPU tensor_split (only one GPU has weight > 0)
                active_gpus = sum(
                    1 for i in range(self.gpu_count) if self.tensor_split[i] > 0
                )
                if active_gpus <= 1:
                    # Single-GPU: use SPLIT_MODE_NONE to load entire model on main_gpu.
                    # SPLIT_MODE_LAYER (default) would still distribute KV/compute
                    # across all visible GPUs even with tensor_split=[1,0,...].
                    self.xlc_params.split_mode = (
                        xlc.llama_split_mode.LLAMA_SPLIT_MODE_NONE
                    )
                    logging.info(
                        f"[LLM] Single-GPU mode: entire model on GPU {self.main_gpu} "
                        f"(split_mode=NONE, fit_params=off)"
                    )
                else:
                    # Multi-GPU tensor split: set ratios for layer distribution
                    for i, ratio in enumerate(self.tensor_split):
                        self.xlc_params.tensor_split[i] = ratio
                    logging.info(
                        f"[LLM] Tensor split across {active_gpus} GPUs "
                        f"(main_gpu={self.main_gpu}, fit_params=off)"
                    )
                # Disable llama.cpp's auto-fit which overrides our split config
                # and redistributes across all GPUs ignoring our explicit settings.
                self.xlc_params.fit_params = False
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

    def _apply_sampling_kwargs(self, request: dict, kwargs: dict) -> dict:
        """Forward llama.cpp sampling knobs accepted outside the OpenAI subset."""
        for key in ("top_k", "min_p", "presence_penalty", "frequency_penalty"):
            if key in kwargs:
                request[key] = kwargs[key]

        if "repeat_penalty" in kwargs:
            request["repeat_penalty"] = kwargs["repeat_penalty"]
        elif "repetition_penalty" in kwargs:
            request["repeat_penalty"] = kwargs["repetition_penalty"]

        return request

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
        chat_request = self._apply_sampling_kwargs(chat_request, kwargs)

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
        cancel_event = (
            threading.Event()
        )  # Set when generator is closed to stop inference
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
                if cancel_event.is_set():
                    logging.debug("[LLM] cancel_event set, stopping stream callback")
                    return True

                logging.debug(
                    f"[LLM] Stream callback received: type={type(chunk_data)}"
                )

                # Check for error response. xllamacpp can deliver streaming
                # errors either at the top level or nested under "error".
                # Treat both as hard stream errors so clients see the real
                # cause instead of a misleading empty-stream retry loop.
                error_info = None
                if isinstance(chunk_data, dict):
                    if "error" in chunk_data:
                        nested_error = chunk_data.get("error")
                        error_info = (
                            nested_error
                            if isinstance(nested_error, dict)
                            else {"message": nested_error}
                        )
                    elif (
                        "code" in chunk_data
                        or chunk_data.get("type") == "exceed_context_size_error"
                    ):
                        error_info = chunk_data

                if error_info is not None:
                    logging.error(f"[LLM] Stream callback error: {chunk_data}")
                    # Include token counts in error message for context size handling
                    error_msg = error_info.get("message", str(error_info))
                    n_prompt_tokens = error_info.get("n_prompt_tokens")
                    n_ctx = error_info.get("n_ctx")
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
        assistant_chunks_yielded = 0
        pending_final_chunk = None
        last_keepalive = time.time()
        keepalive_interval = 5.0  # Send keepalive every 5 seconds during processing

        try:
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
                            has_assistant_text = stream_chunk_has_assistant_text(
                                chunk_data
                            )
                            if has_assistant_text:
                                assistant_chunks_yielded += 1
                            if not has_assistant_text and stream_chunk_finish_reason(
                                chunk_data
                            ):
                                pending_final_chunk = chunk_data
                                continue
                            yield chunk_data
                        else:
                            # Wrap in OpenAI format while preserving reasoning
                            # deltas. Some llama.cpp/xllamacpp builds stream
                            # reasoning_content separately from answer content;
                            # dropping it creates empty visible streams for
                            # clients that know how to render thinking.
                            delta = normalize_stream_chunk_delta(chunk_data)
                            if not delta:
                                logging.warning(
                                    f"[LLM] Unknown stream chunk dict format without assistant text: {chunk_data}"
                                )
                                continue
                            assistant_chunks_yielded += 1
                            yield {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": self.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": chunk_data.get(
                                            "finish_reason"
                                        ),
                                    }
                                ],
                            }
                    elif isinstance(chunk_data, str):
                        # JSON string - parse it
                        try:
                            import json

                            parsed = json.loads(chunk_data)
                            if "choices" in parsed:
                                has_assistant_text = stream_chunk_has_assistant_text(
                                    parsed
                                )
                                if has_assistant_text:
                                    assistant_chunks_yielded += 1
                                if (
                                    not has_assistant_text
                                    and stream_chunk_finish_reason(parsed)
                                ):
                                    pending_final_chunk = parsed
                                    continue
                                yield parsed
                            else:
                                delta = normalize_stream_chunk_delta(parsed)
                                if delta:
                                    assistant_chunks_yielded += 1
                                    yield {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": self.model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": delta,
                                                "finish_reason": parsed.get(
                                                    "finish_reason"
                                                ),
                                            }
                                        ],
                                    }
                                else:
                                    logging.warning(
                                        f"[LLM] Parsed JSON stream chunk without assistant text: {parsed}"
                                    )
                        except json.JSONDecodeError:
                            # Raw text chunk
                            assistant_chunks_yielded += 1
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
                        logging.debug(
                            f"[LLM] Unexpected chunk type: {type(chunk_data)}"
                        )
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

        except GeneratorExit:
            # Consumer closed the generator (e.g., client disconnected or AGiXT
            # broke out of the streaming loop for continuation).
            # Signal the callback to return True on its next invocation, which
            # makes the C++ process_handler_response() exit -> handle_chat_completions()
            # returns -> the C++ response object is destroyed -> slot freed.
            logging.info(
                "[LLM] GeneratorExit received, setting cancel_event to free slot"
            )
            cancel_event.set()
            # Wait briefly for the inference thread to notice and finish,
            # but only if we're not already on that thread (which can happen
            # when close() propagates through nested generators).
            if threading.current_thread() is not inference_thread:
                inference_thread.join(timeout=5.0)
            return

        # Wait for thread to complete
        inference_thread.join(timeout=5.0)

        logging.debug(f"[LLM] Stream complete, yielded {chunks_yielded} chunks")

        # Check for errors
        if error_holder[0]:
            raise error_holder[0]

        if assistant_chunks_yielded == 0:
            message = (
                "LLM stream completed without any assistant text. "
                f"raw_chunks={chunks_yielded}; model={self.model_name}"
            )
            logging.error(f"[LLM] {message}")
            yield {
                "error": {
                    "message": message,
                    "type": "empty_stream",
                }
            }
            return

        if pending_final_chunk is not None:
            yield pending_final_chunk
            return

        # Only synthesize a final stop chunk if xllamacpp did not send one.
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
        completion_request = self._apply_sampling_kwargs(completion_request, kwargs)

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
