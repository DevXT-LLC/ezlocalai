import os
import logging
import time
import math
import threading
import queue
from dotenv import load_dotenv
from ezlocalai.LLM import (
    LLM,
    get_free_vram_per_gpu,
    get_total_vram_per_gpu,
    calculate_tensor_split_from_free_vram,
)
from ezlocalai.CTTS import CTTS
from pyngrok import ngrok
import requests
import base64
import pdfplumber
import json
from Globals import getenv
import gc
import torch
from typing import Tuple, Optional, Dict, Any, List

try:
    from ezlocalai.IMG import IMG

    img_import_success = True
except ImportError:
    img_import_success = False

# Background cleanup queue for async model unloading
_cleanup_queue = queue.Queue()
_cleanup_thread = None
_cleanup_thread_lock = threading.Lock()


def _cleanup_worker():
    """Background worker thread that handles model cleanup asynchronously.

    This allows responses to be returned to the user immediately while
    model unloading and VRAM cleanup happens in the background.
    """
    while True:
        try:
            cleanup_task = _cleanup_queue.get(timeout=1.0)
            if cleanup_task is None:
                # Shutdown signal
                break

            cleanup_func, args, kwargs = cleanup_task
            try:
                cleanup_func(*args, **kwargs)
            except Exception as e:
                logging.error(f"[Cleanup] Background cleanup error: {e}")
        except queue.Empty:
            continue


def _ensure_cleanup_thread():
    """Ensure the background cleanup thread is running."""
    global _cleanup_thread
    with _cleanup_thread_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True)
            _cleanup_thread.start()
            logging.debug("[Cleanup] Background cleanup thread started")


def _schedule_cleanup(cleanup_func, *args, **kwargs):
    """Schedule a cleanup function to run in the background."""
    _ensure_cleanup_thread()
    _cleanup_queue.put((cleanup_func, args, kwargs))


# xllamacpp memory estimation
try:
    import xllamacpp
    from xllamacpp import estimate_gpu_layers, get_device_info
    from huggingface_hub import hf_hub_download

    xllamacpp_available = True
except ImportError:
    xllamacpp_available = False


def get_gpu_count() -> int:
    """Get the number of available CUDA GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_available_vram_gb(gpu_index: int = None) -> float:
    """Get available VRAM in GB, rounded down to nearest 1GB for safety margin.

    Args:
        gpu_index: Specific GPU index, or None to sum all GPUs for multi-GPU setups.
    """
    if torch.cuda.is_available():
        if gpu_index is not None:
            # Single GPU
            if gpu_index < torch.cuda.device_count():
                total = torch.cuda.get_device_properties(gpu_index).total_memory
                return math.floor(total / (1024**3))
            return 0.0
        else:
            # Sum all GPUs for total available VRAM budget
            total_vram = 0.0
            for i in range(torch.cuda.device_count()):
                total_vram += torch.cuda.get_device_properties(i).total_memory
            return math.floor(total_vram / (1024**3))
    return 0.0


def get_free_vram_gb(gpu_index: int = None) -> float:
    """Get FREE (available) VRAM in GB, accounting for other processes.

    Args:
        gpu_index: Specific GPU index, or None to sum all GPUs.
    """
    if not torch.cuda.is_available():
        return 0.0

    if gpu_index is not None:
        if gpu_index < torch.cuda.device_count():
            free, _ = torch.cuda.mem_get_info(gpu_index)
            return free / (1024**3)
        return 0.0
    else:
        total_free = 0.0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            total_free += free
        return total_free / (1024**3)


def get_per_gpu_vram_gb() -> list:
    """Get VRAM for each GPU as a list of GB values."""
    if torch.cuda.is_available():
        vram_list = []
        for i in range(torch.cuda.device_count()):
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            vram_list.append(math.floor(vram_gb))
        return vram_list
    return []


def get_per_gpu_free_vram_gb() -> list:
    """Get FREE VRAM for each GPU as a list of GB values.

    Uses torch.cuda.mem_get_info() which accounts for other processes.
    """
    if torch.cuda.is_available():
        free_list = []
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            free_list.append(free / (1024**3))
        return free_list
    return []


def calculate_context_size(estimated_prompt_tokens: int) -> int:
    """Calculate context size with fixed 8k headspace for generation.

    Simply adds 8k to the estimated prompt tokens to provide headspace
    for response generation without over-allocating.
    """
    return estimated_prompt_tokens + 8192


def get_vram_usage_gb(gpu_index: int = 0) -> float:
    """Get current VRAM usage in GB for a specific GPU."""
    if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        return torch.cuda.memory_allocated(gpu_index) / (1024**3)
    return 0.0


def get_total_vram_gb(gpu_index: int = None) -> float:
    """Get total VRAM in GB.

    Args:
        gpu_index: Specific GPU index, or None to sum all GPUs.
    """
    if torch.cuda.is_available():
        if gpu_index is not None:
            if gpu_index < torch.cuda.device_count():
                return torch.cuda.get_device_properties(gpu_index).total_memory / (
                    1024**3
                )
            return 0.0
        else:
            # Sum all GPUs
            total = 0.0
            for i in range(torch.cuda.device_count()):
                total += torch.cuda.get_device_properties(i).total_memory
            return total / (1024**3)
    return 0.0


def calculate_tensor_split() -> list:
    """Calculate tensor split ratios based on available VRAM per GPU.

    DEPRECATED: Use calculate_tensor_split_from_free_vram() for accurate splits.

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


def extract_gpu_generation_score(gpu_name: str) -> Tuple[int, int, int]:
    """Extract a capability score from NVIDIA GPU name.

    Parses GPU names like "NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3090",
    "NVIDIA A100", etc. and returns a tuple for comparison.

    Returns:
        Tuple of (generation, tier, variant) where higher is better.
        - generation: 50 for 5000 series, 40 for 4000 series, etc.
        - tier: 90 for x090, 80 for x080, 70 for x070, etc.
        - variant: Ti/Super variants get bonus points
    """
    import re

    gpu_name_upper = gpu_name.upper()

    # Default low score for unknown GPUs
    generation = 0
    tier = 0
    variant = 0

    # Check for datacenter/professional cards first (A100, H100, etc.)
    datacenter_match = re.search(r"\b([AH])(\d{2,3})\b", gpu_name_upper)
    if datacenter_match:
        prefix = datacenter_match.group(1)
        number = int(datacenter_match.group(2))
        # H100 > A100 > A40, etc.
        if prefix == "H":
            generation = 100
        elif prefix == "A":
            generation = 90
        tier = number
        return (generation, tier, variant)

    # RTX/GTX consumer cards - extract the model number
    # Matches: RTX 5090, RTX 4090, RTX 3090, GTX 1080, etc.
    rtx_match = re.search(r"\b(?:RTX|GTX)\s*(\d)(\d{2,3})\b", gpu_name_upper)
    if rtx_match:
        gen_digit = int(rtx_match.group(1))  # 5, 4, 3, 2, 1
        tier_digits = rtx_match.group(2)  # 090, 080, 070, 80, 70

        # Normalize generation (5xxx = 50, 4xxx = 40, etc.)
        generation = gen_digit * 10

        # Normalize tier (90, 80, 70, 60, 50)
        if len(tier_digits) == 3:
            tier = int(tier_digits[0:2])  # 090 -> 90
        else:
            tier = int(tier_digits)  # 80 -> 80

        # Check for Ti/Super variants
        if "TI" in gpu_name_upper:
            variant = 5
        elif "SUPER" in gpu_name_upper:
            variant = 3

    # Quadro cards
    quadro_match = re.search(r"QUADRO\s*(?:RTX\s*)?(\d+)", gpu_name_upper)
    if quadro_match:
        number = int(quadro_match.group(1))
        generation = 35  # Place between GTX and newer RTX
        tier = number // 100 if number >= 1000 else number // 10

    return (generation, tier, variant)


def get_gpu_capability_ranking() -> List[Tuple[int, float, str]]:
    """Get GPUs ranked by capability (most powerful first).

    Returns:
        List of tuples: (gpu_index, capability_score, gpu_name)
        Sorted by capability_score descending (most powerful first).
    """
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_name = props.name

        # Get generation/tier score from name
        gen_score = extract_gpu_generation_score(gpu_name)

        # Also factor in compute capability and VRAM as tiebreakers
        compute_capability = props.major * 10 + props.minor
        total_vram_gb = props.total_memory / (1024**3)

        # Combined score: prioritize generation, then tier, then variant,
        # then compute capability, then VRAM
        # Score = gen*10000 + tier*100 + variant*10 + compute_cap + vram/100
        capability_score = (
            gen_score[0] * 10000
            + gen_score[1] * 100
            + gen_score[2] * 10
            + compute_capability
            + total_vram_gb / 100
        )

        gpu_info.append((i, capability_score, gpu_name))

    # Sort by capability score descending
    gpu_info.sort(key=lambda x: x[1], reverse=True)

    return gpu_info


def get_gpus_by_priority() -> List[int]:
    """Get GPU indices ordered by capability (most powerful first).

    Returns:
        List of GPU indices, ordered from most to least capable.
    """
    ranking = get_gpu_capability_ranking()
    return [gpu_idx for gpu_idx, _, _ in ranking]


def estimate_model_vram_requirement(
    model_path: str, context_size: int, projectors: list = None
) -> float:
    """Estimate VRAM requirement for a model in GB.

    Uses xllamacpp's estimate_gpu_layers to get memory estimates.
    Returns estimated VRAM in GB, or a conservative estimate if xllamacpp is unavailable.
    """
    if not xllamacpp_available:
        # Conservative fallback: assume 0.5GB per 1K context + base model size
        # This is very rough but better than nothing
        return 8.0 + (context_size / 1000) * 0.5

    try:
        # Create a fake GPU with unlimited memory to get total requirement
        fake_gpu = {
            "name": "Virtual GPU",
            "memory_free": 1024 * 1024 * 1024 * 1024,  # 1TB
            "memory_total": 1024 * 1024 * 1024 * 1024,
        }

        result = estimate_gpu_layers(
            gpus=[fake_gpu],
            model_path=model_path,
            projectors=projectors or [],
            context_length=context_size,
            batch_size=2048,
            num_parallel=1,
            kv_cache_type="f16",
        )

        # Extract memory requirement
        if hasattr(result, "memory"):
            return result.memory / (1024**3)
        elif isinstance(result, dict) and "memory" in result:
            return result["memory"] / (1024**3)

        # Fallback estimation
        return 8.0 + (context_size / 1000) * 0.5

    except Exception as e:
        logging.warning(f"[GPU Selection] Failed to estimate VRAM: {e}")
        return 8.0 + (context_size / 1000) * 0.5


def _estimate_optimal_layers(
    model_path: str,
    context_size: int,
    projectors: list = None,
    available_vram_per_gpu: list = None,
) -> int:
    """Estimate optimal number of GPU layers for partial offloading.

    Uses xllamacpp's estimate_gpu_layers to determine how many layers
    can fit in the available VRAM.

    Args:
        model_path: Path to the model file
        context_size: Context window size
        projectors: List of projector paths (for vision models)
        available_vram_per_gpu: List of available VRAM per GPU in GB

    Returns:
        Number of layers that can fit on GPU, or 0 if estimation fails
    """
    if not xllamacpp_available or not available_vram_per_gpu:
        return 0

    try:
        # Build GPU info list for xllamacpp
        gpus = []
        for i, avail_gb in enumerate(available_vram_per_gpu):
            avail_bytes = int(avail_gb * 1024 * 1024 * 1024)
            gpus.append(
                {
                    "name": f"GPU {i}",
                    "memory_free": avail_bytes,
                    "memory_total": avail_bytes,  # Use available as total for estimation
                }
            )

        result = estimate_gpu_layers(
            gpus=gpus,
            model_path=model_path,
            projectors=projectors or [],
            context_length=context_size,
            batch_size=2048,
            num_parallel=1,
            kv_cache_type="f16",
        )

        # Extract layer count from result
        if hasattr(result, "layers"):
            return result.layers
        elif isinstance(result, dict):
            return result.get(
                "layers", result.get("gpu_layers", result.get("n_gpu_layers", 0))
            )
        elif isinstance(result, int):
            return result

        return 0

    except Exception as e:
        logging.warning(f"[GPU Selection] Failed to estimate optimal layers: {e}")
        return 0


def determine_gpu_strategy(
    model_path: str,
    context_size: int,
    projectors: list = None,
    reserved_vram: float = 5.0,
) -> Dict[str, Any]:
    """Determine optimal GPU loading strategy based on available VRAM and GPU capability.

    Smart GPU selection logic (GPUs ordered by capability, not nvidia-smi index):
    1. If the most powerful GPU has enough free VRAM → load on it only
    2. If combined GPUs have enough → tensor split across them (weighted by free VRAM)
    3. If most powerful is full but another GPU has enough → load on that GPU only
    4. Otherwise → fall back to CPU

    Args:
        model_path: Path to the model file
        context_size: Context window size
        projectors: List of projector paths (for vision models)
        reserved_vram: VRAM to reserve for TTS/STT (default 5GB)

    Returns:
        Dict with keys:
        - main_gpu: GPU index to use as primary (0, 1, etc.)
        - tensor_split: List of 128 floats for tensor splitting, or None for single GPU
        - gpu_layers: Number of GPU layers, or 0 for CPU-only
        - strategy: Description of the strategy ("gpu0", "gpu1", "tensor_split", "cpu")
    """
    gpu_count = get_gpu_count()

    if gpu_count == 0 or not torch.cuda.is_available():
        return {"main_gpu": 0, "tensor_split": None, "gpu_layers": 0, "strategy": "cpu"}

    # Get GPU capability ranking (most powerful first)
    gpu_ranking = get_gpu_capability_ranking()
    gpus_by_priority = [gpu_idx for gpu_idx, _, _ in gpu_ranking]

    # Log GPU ranking (debug-level to reduce noise)
    logging.debug(f"[GPU Selection] GPU capability ranking (most powerful first):")
    for gpu_idx, score, name in gpu_ranking:
        logging.debug(f"[GPU Selection]   GPU {gpu_idx}: {name} (score: {score:.1f})")

    # Get free VRAM for each GPU
    free_vram = get_per_gpu_free_vram_gb()
    total_vram = get_per_gpu_vram_gb()

    # Estimate model VRAM requirement
    estimated_vram = estimate_model_vram_requirement(
        model_path, context_size, projectors
    )

    # Account for reserved VRAM (distribute across GPUs proportionally)
    reserved_per_gpu = reserved_vram / gpu_count if gpu_count > 0 else 0
    available_vram = [max(0, free - reserved_per_gpu) for free in free_vram]

    logging.debug(
        f"[GPU Selection] Model VRAM estimate: {estimated_vram:.1f}GB for {context_size//1000}k context"
    )
    for i in range(gpu_count):
        logging.debug(
            f"[GPU Selection]   GPU {i}: {free_vram[i]:.1f}GB free, "
            f"{available_vram[i]:.1f}GB available after {reserved_per_gpu:.1f}GB reservation"
        )

    # Single GPU case
    if gpu_count == 1:
        if available_vram[0] >= estimated_vram:
            return {
                "main_gpu": 0,
                "tensor_split": None,
                "gpu_layers": -1,  # Auto-detect (all layers on GPU)
                "strategy": "gpu0",
            }
        else:
            # Not enough VRAM for full model - try partial offloading
            # Use xllamacpp to estimate how many layers can fit
            optimal_layers = _estimate_optimal_layers(
                model_path, context_size, projectors, [available_vram[0]]
            )
            if optimal_layers and optimal_layers > 0:
                logging.info(
                    f"[GPU Selection] GPU 0 has {available_vram[0]:.1f}GB available, "
                    f"need {estimated_vram:.1f}GB - using partial offload ({optimal_layers} layers on GPU)"
                )
                return {
                    "main_gpu": 0,
                    "tensor_split": None,
                    "gpu_layers": optimal_layers,
                    "strategy": "gpu0_partial",
                }
            else:
                # Fall back to CPU
                logging.warning(
                    f"[GPU Selection] GPU 0 has {available_vram[0]:.1f}GB available, "
                    f"need {estimated_vram:.1f}GB, falling back to CPU"
                )
                return {
                    "main_gpu": 0,
                    "tensor_split": None,
                    "gpu_layers": 0,
                    "strategy": "cpu",
                }

    # Multi-GPU case - use capability ranking
    # Strategy 1: Try most powerful GPU alone
    primary_gpu = gpus_by_priority[0]
    primary_name = gpu_ranking[0][2]

    if available_vram[primary_gpu] >= estimated_vram:
        logging.debug(
            f"[GPU Selection] Primary GPU {primary_gpu} ({primary_name}) has {available_vram[primary_gpu]:.1f}GB available, "
            f"sufficient for {estimated_vram:.1f}GB model - loading on GPU {primary_gpu} only"
        )
        return {
            "main_gpu": primary_gpu,
            "tensor_split": None,
            "gpu_layers": -1,
            "strategy": f"gpu{primary_gpu}",
        }

    # Strategy 2: Try tensor split across all GPUs
    total_available = sum(available_vram)
    if total_available >= estimated_vram:
        # Calculate tensor split based on FREE VRAM proportions
        tensor_split = [0.0] * 128
        for i, avail in enumerate(available_vram):
            tensor_split[i] = avail / total_available if total_available > 0 else 0.0

        logging.debug(
            f"[GPU Selection] Tensor splitting across {gpu_count} GPUs "
            f"(total: {total_available:.1f}GB available, need: {estimated_vram:.1f}GB)"
        )
        logging.debug(f"[GPU Selection] Split ratios: {tensor_split[:gpu_count]}")

        # Use the most powerful GPU as main_gpu for tensor split
        return {
            "main_gpu": primary_gpu,
            "tensor_split": tensor_split,
            "gpu_layers": -1,
            "strategy": "tensor_split",
        }

    # Strategy 3: Try other GPUs in order of capability
    for gpu_idx in gpus_by_priority[1:]:  # Skip the primary GPU we already tried
        if available_vram[gpu_idx] >= estimated_vram:
            gpu_name = next(
                (name for idx, _, name in gpu_ranking if idx == gpu_idx),
                f"GPU {gpu_idx}",
            )
            logging.debug(
                f"[GPU Selection] Primary GPU {primary_gpu} full ({available_vram[primary_gpu]:.1f}GB), "
                f"but GPU {gpu_idx} ({gpu_name}) has {available_vram[gpu_idx]:.1f}GB - loading on GPU {gpu_idx} only"
            )
            return {
                "main_gpu": gpu_idx,
                "tensor_split": None,
                "gpu_layers": -1,
                "strategy": f"gpu{gpu_idx}",
            }

    # Strategy 4: Try partial offloading with tensor split
    # Model doesn't fit entirely but we can offload some layers to GPUs
    total_available = sum(available_vram)
    if total_available > 2:  # At least 2GB available across all GPUs
        optimal_layers = _estimate_optimal_layers(
            model_path, context_size, projectors, available_vram
        )
        if optimal_layers and optimal_layers > 0:
            # Calculate tensor split for partial offloading
            tensor_split = [0.0] * 128
            for i, avail in enumerate(available_vram):
                tensor_split[i] = (
                    avail / total_available if total_available > 0 else 0.0
                )

            logging.info(
                f"[GPU Selection] Partial offload: {optimal_layers} layers across {gpu_count} GPUs "
                f"(total: {total_available:.1f}GB available, need: {estimated_vram:.1f}GB for full model)"
            )
            return {
                "main_gpu": primary_gpu,
                "tensor_split": tensor_split,
                "gpu_layers": optimal_layers,
                "strategy": "tensor_split_partial",
            }

    # Strategy 5: Fall back to CPU
    logging.warning(
        f"[GPU Selection] Insufficient VRAM across all GPUs "
        f"(need {estimated_vram:.1f}GB, have {total_available:.1f}GB), falling back to CPU"
    )
    return {"main_gpu": 0, "tensor_split": None, "gpu_layers": 0, "strategy": "cpu"}


# Model-specific config overrides for optimal inference settings
# These override user-provided values for known models
MODEL_CONFIG_OVERRIDES = {
    "unsloth/Qwen3-VL-4B-Instruct-GGUF": {
        "top_p": 0.8,
        "top_k": 20,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    "unsloth/Qwen3-4B-Instruct-2507-GGUF": {
        "top_p": 1.0,
        "top_k": 40,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 2.0,
    },
    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF": {
        "top_p": 0.8,
        "top_k": 20,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
    },
}


class Pipes:
    def __init__(self):
        load_dotenv()
        global img_import_success

        # Lock for model access - prevents race conditions when multiple
        # requests try to load/switch models simultaneously
        self._model_lock = threading.Lock()
        # Track if inference is currently in progress
        self._inference_in_progress = False

        # Check if precache already ran (models already downloaded/warmed)
        from pathlib import Path

        precache_done = Path("/tmp/ezlocalai_precache.done").exists()
        if precache_done:
            logging.debug(
                "[Init] Precache completed, skipping redundant warmup operations"
            )

        # Auto-detect multi-GPU configuration
        self.gpu_count = get_gpu_count()
        self.per_gpu_vram = get_per_gpu_vram_gb()

        # Auto-detect total VRAM budget across all GPUs (rounded down to nearest 1GB for safety margin)
        self.vram_budget_gb = get_available_vram_gb()

        # Calculate tensor split for multi-GPU
        self.tensor_split = calculate_tensor_split()

        if self.vram_budget_gb > 0:
            if self.gpu_count > 1:
                logging.debug(
                    f"[VRAM] Multi-GPU detected: {self.gpu_count} GPUs, "
                    f"{self.vram_budget_gb}GB total VRAM budget "
                    f"(per GPU: {self.per_gpu_vram})"
                )
            else:
                logging.debug(
                    f"[VRAM] Auto-detected {self.vram_budget_gb}GB VRAM budget"
                )

        # Parse model list: "model1,model2" (simple comma-separated)
        model_config = getenv("DEFAULT_MODEL")
        self.available_models = []  # List of model names
        self.calibrated_gpu_layers = {}  # {model_name: {context: gpu_layers}}

        # Persistent LLM instances (kept loaded to avoid reload overhead)
        self.primary_llm = (
            None  # First non-vision model (or first model if all are vision)
        )
        self.primary_llm_name = None
        self.primary_llm_context = None

        self.vision_llm = None  # Vision model (if different from primary)
        self.vision_llm_name = None
        self.vision_llm_context = None

        # Active LLM pointer (points to one of the above, or a temp large model)
        self.llm = None
        self.current_llm_name = None
        self.current_context = None  # Track current context size

        # Track if we're using a "large" model that should be unloaded after use
        self._using_large_model = False

        if model_config.lower() != "none":
            for model_entry in model_config.split(","):
                model_name = model_entry.strip()
                # Strip any legacy @tokens suffix for backward compat
                if "@" in model_name:
                    model_name = model_name.rsplit("@", 1)[0]
                if model_name and model_name not in self.available_models:
                    self.available_models.append(model_name)

            # Pre-load persistent LLMs (primary + vision if different)
            if self.available_models:
                # Pre-load with 32k context to avoid reloads on typical requests
                # (requests add estimated prompt tokens + 16k headroom, so need buffer)
                default_context = int(getenv("LLM_MAX_TOKENS", "32768"))

                # Find primary (first non-vision) and vision models
                primary_model = None
                vision_model = None

                for model_name in self.available_models:
                    is_vision = self._is_vision_model(model_name)
                    if is_vision and vision_model is None:
                        vision_model = model_name
                    if not is_vision and primary_model is None:
                        primary_model = model_name

                # If no non-vision model, use first model as primary
                if primary_model is None:
                    primary_model = self.available_models[0]
                    # If primary is vision, don't load it twice
                    if primary_model == vision_model:
                        vision_model = None

                # Load primary model
                logging.info(f"[LLM] Pre-loading primary model: {primary_model}...")
                start_time = time.time()
                try:
                    self.primary_llm = self._load_llm_resilient(
                        model_name=primary_model,
                        max_tokens=default_context,
                    )
                    self.primary_llm_name = primary_model
                    self.primary_llm_context = default_context
                    self.llm = self.primary_llm
                    self.current_llm_name = primary_model
                    self.current_context = default_context
                    load_time = time.time() - start_time
                    logging.info(
                        f"[LLM] Primary model {primary_model} loaded in {load_time:.1f}s"
                    )
                except Exception as e:
                    logging.warning(
                        f"[LLM] Failed to pre-load primary model {primary_model}: {e}"
                    )

                # Load vision model if different from primary
                if vision_model and vision_model != primary_model:
                    logging.info(f"[LLM] Pre-loading vision model: {vision_model}...")
                    start_time = time.time()
                    try:
                        self.vision_llm = self._load_llm_resilient(
                            model_name=vision_model,
                            max_tokens=default_context,
                        )
                        self.vision_llm_name = vision_model
                        self.vision_llm_context = default_context
                        load_time = time.time() - start_time
                        logging.info(
                            f"[LLM] Vision model {vision_model} loaded in {load_time:.1f}s"
                        )
                    except Exception as e:
                        logging.warning(
                            f"[LLM] Failed to pre-load vision model {vision_model}: {e}"
                        )

        # TTS initialization - skip warmup if precache already did it
        self.ctts = None
        if getenv("TTS_ENABLED").lower() == "true":
            if precache_done:
                # Precache already warmed the TTS cache, skip loading/unloading
                logging.debug(
                    "[CTTS] TTS cache already warmed by precache, will lazy-load on first request"
                )
            else:
                # No precache - warm the cache now (first run or single-worker mode)
                logging.debug("[CTTS] Preloading Chatterbox TTS to warm cache...")
                start_time = time.time()
                self.ctts = CTTS()
                load_time = time.time() - start_time
                logging.debug(
                    f"[CTTS] Chatterbox TTS preloaded in {load_time:.2f}s, unloading to free VRAM..."
                )
                self._destroy_tts()
                logging.debug(
                    "[CTTS] TTS unloaded, will lazy-load on first TTS request."
                )

        # Lazy-loaded models (loaded on first use, destroyed after)
        self.stt = None
        self.embedder = None
        self.img = None
        self.current_stt = getenv("WHISPER_MODEL")

        NGROK_TOKEN = getenv("NGROK_TOKEN")
        if NGROK_TOKEN:
            ngrok.set_auth_token(NGROK_TOKEN)
            public_url = ngrok.connect(8091)
            logging.info(f"[ngrok] Public Tunnel: {public_url.public_url}")
            self.local_uri = public_url.public_url
        else:
            self.local_uri = getenv("EZLOCALAI_URL")

        logging.info(f"[Server] Ready!")

    def _load_llm_resilient(
        self,
        model_name: str,
        max_tokens: int,
        gpu_layers: int = None,
        main_gpu: int = None,
        tensor_split: list = None,
    ) -> "LLM":
        """Load an LLM with smart GPU selection and resilient fallback.

        This method uses smart GPU selection to determine the optimal loading strategy:
        1. If GPU 0 has enough VRAM → load on GPU 0 only
        2. If GPU 0 + GPU 1 together have enough → tensor split across both
        3. If GPU 0 is full but GPU 1 has enough → load on GPU 1 only
        4. Otherwise → fall back to CPU

        Args:
            model_name: Name of the model to load
            max_tokens: Context size for the model
            gpu_layers: Number of GPU layers (None for smart auto-detect)
            main_gpu: Primary GPU index (None for smart auto-detect)
            tensor_split: Tensor split ratios (None for smart auto-detect)

        Returns:
            LLM instance

        Raises:
            Exception: Only if all loading strategies fail
        """
        # Get model path for VRAM estimation
        from ezlocalai.LLM import download_model

        try:
            model_path, mmproj_path = download_model(
                model_name=model_name, models_dir="./models"
            )
            projectors = [mmproj_path] if mmproj_path else []
        except Exception as e:
            logging.warning(f"[LLM] Could not get model path for estimation: {e}")
            model_path = None
            projectors = []

        # Use smart GPU selection if no explicit configuration provided
        if (
            gpu_layers is None
            and main_gpu is None
            and tensor_split is None
            and model_path
        ):
            strategy = determine_gpu_strategy(
                model_path=model_path,
                context_size=max_tokens,
                projectors=projectors,
                reserved_vram=5.0,  # Reserve 5GB for TTS/STT
            )

            gpu_layers = strategy["gpu_layers"]
            main_gpu = strategy["main_gpu"]
            tensor_split = strategy["tensor_split"]

            logging.debug(
                f"[LLM] Smart GPU selection: strategy='{strategy['strategy']}', "
                f"main_gpu={main_gpu}, gpu_layers={gpu_layers}, "
                f"tensor_split={'enabled' if tensor_split else 'disabled'}"
            )

        # First attempt: Try with determined/specified configuration
        try:
            logging.debug(
                f"[LLM] Attempting to load {model_name} (main_gpu={main_gpu}, "
                f"gpu_layers={gpu_layers or 'auto'}, tensor_split={'yes' if tensor_split else 'no'})..."
            )
            llm = LLM(
                model=model_name,
                max_tokens=max_tokens,
                gpu_layers=gpu_layers,
                main_gpu=main_gpu,
                tensor_split=tensor_split,
            )
            return llm
        except Exception as gpu_error:
            error_str = str(gpu_error).lower()
            # Check if this is a resource exhaustion error
            is_resource_error = any(
                x in error_str
                for x in [
                    "out of memory",
                    "cuda",
                    "vram",
                    "gpu",
                    "allocat",
                    "memory",
                    "resource",
                ]
            )

            if is_resource_error and gpu_layers != 0:
                logging.warning(
                    f"[LLM] GPU loading failed for {model_name}: {gpu_error}"
                )
                logging.debug(f"[LLM] Falling back to CPU-only mode (gpu_layers=0)...")

                # Second attempt: Force CPU-only mode
                try:
                    llm = LLM(model=model_name, max_tokens=max_tokens, gpu_layers=0)
                    logging.debug(
                        f"[LLM] {model_name} loaded successfully in CPU-only mode"
                    )
                    return llm
                except Exception as cpu_error:
                    logging.error(
                        f"[LLM] CPU fallback also failed for {model_name}: {cpu_error}"
                    )
                    raise cpu_error
            else:
                # Not a resource error, or already at gpu_layers=0
                raise gpu_error

    def _calibrate_model(self, model_name: str, max_tokens: int) -> int:
        """Calibrate a single model to find optimal GPU layers.

        First tries xllamacpp's native estimate_gpu_layers for a fast estimation,
        then falls back to binary search if needed.
        """
        # Try native estimation first if available
        if xllamacpp_available:
            try:
                estimated = self._estimate_layers_native(model_name, max_tokens)
                if estimated is not None:
                    return estimated
            except Exception as e:
                logging.warning(
                    f"[Calibration] Native estimation failed: {e}, falling back to binary search"
                )

        # Fallback: Binary search calibration
        return self._calibrate_binary_search(model_name, max_tokens)

    def _estimate_layers_native(self, model_name: str, max_tokens: int) -> int:
        """Use xllamacpp's native GPU layer estimation.

        Returns optimal GPU layers or None if estimation fails.
        """
        logging.debug(
            f"[Calibration] Using native estimation for {model_name} (budget: {self.vram_budget_gb}GB)"
        )

        # Get GPU info
        devices = get_device_info()
        gpus = []
        gpu_idx = 0
        for dev in devices:
            # Check if it's a GPU device
            dev_type = str(dev.get("type", ""))
            if "GPU" in dev_type:
                # Use per-GPU VRAM budget, not total budget
                # self.per_gpu_vram has the individual VRAM for each GPU
                if gpu_idx < len(self.per_gpu_vram):
                    per_gpu_budget = self.per_gpu_vram[gpu_idx]
                else:
                    per_gpu_budget = self.vram_budget_gb / max(
                        1, len(self.per_gpu_vram)
                    )
                budget_bytes = per_gpu_budget * 1024 * 1024 * 1024
                gpus.append(
                    {
                        "name": dev["name"],
                        "memory_free": budget_bytes,  # Use per-GPU budget
                        "memory_total": dev["memory_total"],
                    }
                )
                gpu_idx += 1

        if not gpus:
            logging.warning("[Calibration] No GPU devices found for estimation")
            return None

        logging.debug(
            f"[Calibration] GPUs: {[g['name'] + ' (' + str(round(g['memory_free']/1e9, 1)) + 'GB budget)' for g in gpus]}"
        )

        # Download model file to get path
        try:
            model_path = self._get_model_path(model_name)
            if not model_path:
                return None

            logging.debug(f"[Calibration] Model path: {model_path}")

            # Get projector if this might be a vision model
            projectors = []
            vision_proj = self._get_vision_projector_path(model_name)
            if vision_proj:
                projectors.append(vision_proj)

            # Estimate GPU layers
            result = estimate_gpu_layers(
                gpus=gpus,
                model_path=model_path,
                projectors=projectors,
                context_length=max_tokens,
                batch_size=2048,
                num_parallel=1,
                kv_cache_type="f16",
            )

            logging.debug(f"[Calibration] Native estimation result: {result}")

            # Extract layer count from result
            # xllamacpp returns MemoryEstimate object with .layers attribute
            if hasattr(result, "layers"):
                gpu_layers = result.layers
            elif isinstance(result, dict):
                gpu_layers = result.get(
                    "layers", result.get("gpu_layers", result.get("n_gpu_layers", 0))
                )
            elif isinstance(result, int):
                gpu_layers = result
            else:
                logging.warning(
                    f"[Calibration] Unexpected result format: {type(result)}"
                )
                return None

            logging.debug(
                f"[Calibration] {model_name} estimated to {gpu_layers} GPU layers (native)"
            )
            return gpu_layers

        except Exception as e:
            logging.error(f"[Calibration] Native estimation error: {e}")
            return None

    def _get_model_path(self, model_name: str) -> str:
        """Get the local path to a model file."""
        try:
            # Parse model name: "org/repo" -> download the GGUF file
            if "/" in model_name:
                # Try common GGUF patterns
                parts = model_name.split("/")
                repo_id = model_name

                # Try to find the GGUF file
                from huggingface_hub import list_repo_files

                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]

                if not gguf_files:
                    return None

                # Prefer QUANT_TYPE from env, then fall back to common quantizations
                quant_type = getenv("QUANT_TYPE")
                best_file = None
                patterns = [
                    quant_type,
                    "Q4_K_M",
                    "Q4_K_XL",
                    "Q4_K",
                    "Q5_K",
                    "Q6_K",
                    "Q8",
                ]
                for pattern in patterns:
                    for f in gguf_files:
                        if pattern in f:
                            best_file = f
                            break
                    if best_file:
                        break

                if not best_file:
                    best_file = gguf_files[0]

                return hf_hub_download(repo_id, best_file)

            return None
        except Exception as e:
            logging.error(f"[Calibration] Failed to get model path: {e}")
            return None

    def _get_vision_projector_path(self, model_name: str) -> str:
        """Get vision projector path if this is a vision model."""
        try:
            if "/" in model_name:
                from huggingface_hub import list_repo_files

                files = list_repo_files(model_name)

                # Look for mmproj files
                for f in files:
                    if "mmproj" in f.lower() and f.endswith(".gguf"):
                        return hf_hub_download(model_name, f)
            return None
        except:
            return None

    def _is_vision_model(self, model_name: str) -> bool:
        """Check if a model has vision capability (has mmproj file)."""
        # Cache results to avoid repeated HuggingFace API calls
        if not hasattr(self, "_vision_model_cache"):
            self._vision_model_cache = {}

        if model_name in self._vision_model_cache:
            return self._vision_model_cache[model_name]

        has_vision = self._get_vision_projector_path(model_name) is not None
        self._vision_model_cache[model_name] = has_vision
        return has_vision

    def _find_vision_model(self) -> str:
        """Find a vision-capable model from available models."""
        for model_name in self.available_models:
            if self._is_vision_model(model_name):
                return model_name
        return None

    def _find_non_vision_model(self) -> str:
        """Find the first non-vision model from available models.

        Returns None if all models are vision models.
        """
        for model_name in self.available_models:
            if not self._is_vision_model(model_name):
                return model_name
        return None

    async def _describe_images_with_vision_model(
        self, images: list, user_text: str
    ) -> str:
        """Use a vision model to describe images, then return descriptions for use with non-vision model.

        This enables non-vision models to respond about images by using a vision model
        to first describe what's in the images.
        """
        vision_model = self._find_vision_model()
        if not vision_model:
            logging.warning(
                "[Vision Fallback] No vision model available to describe images"
            )
            return None

        try:
            # Load vision model (will destroy any current model first)
            self._get_llm(vision_model, 16384)  # Use 16k context for image description

            if not self.llm.is_vision:
                logging.error(
                    f"[Vision Fallback] {vision_model} failed to load with vision capability"
                )
                return None

            # Process images same way as in get_response
            from PIL import Image as PILImage
            from io import BytesIO

            processed_images = []
            for img in images:
                if "image_url" in img:
                    img_url = (
                        img["image_url"].get("url", "")
                        if isinstance(img["image_url"], dict)
                        else img["image_url"]
                    )
                    if img_url and not img_url.startswith("data:"):
                        try:
                            headers = {"User-Agent": "Mozilla/5.0"}
                            img_response = requests.get(
                                img_url, timeout=30, headers=headers
                            )
                            img_response.raise_for_status()
                            content_type = img_response.headers.get(
                                "Content-Type", "image/jpeg"
                            )

                            # Convert unsupported formats to PNG
                            if content_type in [
                                "image/webp",
                                "image/gif",
                                "image/bmp",
                                "image/tiff",
                                "image/avif",
                            ]:
                                pil_img = PILImage.open(BytesIO(img_response.content))
                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    pil_img = pil_img.convert("RGB")
                                buffer = BytesIO()
                                pil_img.save(buffer, format="PNG")
                                img_bytes = buffer.getvalue()
                                content_type = "image/png"
                            else:
                                img_bytes = img_response.content

                            if not content_type.startswith("image/"):
                                content_type = "image/jpeg"
                            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                            processed_images.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{img_base64}"
                                    },
                                }
                            )
                        except Exception as e:
                            logging.error(
                                f"[Vision Fallback] Failed to fetch image: {e}"
                            )
                            continue
                    else:
                        # Already a data URL
                        processed_images.append(img)

            if not processed_images:
                return None

            # Build multimodal message asking to describe the images
            describe_prompt = f"Describe the contents of the image(s) in detail. The user's question about the image(s) is: {user_text}"
            multimodal_content = [{"type": "text", "text": describe_prompt}]
            multimodal_content.extend(processed_images)

            # Get description from vision model
            response = self.llm.chat(
                messages=[{"role": "user", "content": multimodal_content}],
                local_uri=self.local_uri,
                max_tokens=2048,
                temperature=0.3,
            )

            # Extract the text response
            if hasattr(response, "choices") and response.choices:
                description = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                description = response["choices"][0]["message"]["content"]
            else:
                description = str(response)

            return description

        except Exception as e:
            logging.error(f"[Vision Fallback] Failed to describe images: {e}")
            return None
        # Note: LLM will be destroyed at the end of get_response() - no need to swap back

    def _calibrate_binary_search(self, model_name: str, max_tokens: int) -> int:
        """Calibrate using binary search (fallback method).

        Uses binary search to efficiently find the highest GPU layer count that
        fits within VRAM budget. Returns the optimal number of GPU layers.
        """
        # Binary search for optimal layers - start at 70 as most models have 40-80 layers
        low = 0
        high = 70
        best_layers = 0  # Default to CPU if nothing works

        logging.debug(
            f"[Calibration] Binary search for {model_name} (budget: {self.vram_budget_gb}GB)"
        )

        while low <= high:
            mid = (low + high) // 2

            try:
                # Clear VRAM before test
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logging.debug(
                    f"[Calibration] Testing {model_name} with {mid} GPU layers"
                )

                # Try to load the model
                test_llm = LLM(model=model_name, max_tokens=max_tokens, gpu_layers=mid)

                # Check VRAM usage
                vram_used = get_vram_usage_gb()

                # Unload test model
                del test_llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if vram_used <= self.vram_budget_gb:
                    # Fits! Try higher
                    best_layers = mid
                    logging.debug(
                        f"[Calibration] {mid} layers OK ({vram_used:.1f}GB), trying higher"
                    )
                    low = mid + 1
                else:
                    # Too much VRAM, try lower
                    logging.debug(
                        f"[Calibration] {mid} layers too high ({vram_used:.1f}GB), trying lower"
                    )
                    high = mid - 1

            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "out of memory" in error_msg
                    or "oom" in error_msg
                    or "cuda" in error_msg
                ):
                    logging.warning(
                        f"[Calibration] OOM at {mid} layers, trying lower..."
                    )
                else:
                    logging.error(f"[Calibration] Error at {mid} layers: {e}")

                # Cleanup after error
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # OOM means too many layers
                high = mid - 1

        logging.debug(
            f"[Calibration] {model_name} calibrated to {best_layers} GPU layers"
        )
        return best_layers

    def _get_gpu_layers_for_model(self, model_name: str, context_size: int) -> int:
        """Get GPU layers for a model at a specific context size.

        If not pre-calibrated for this context, calibrate now and cache.
        """
        if self.vram_budget_gb <= 0:
            return None  # No GPU available

        # Check cache for this model+context combo
        if model_name in self.calibrated_gpu_layers:
            if context_size in self.calibrated_gpu_layers[model_name]:
                return self.calibrated_gpu_layers[model_name][context_size]

        # Need to calibrate for this context size
        logging.debug(
            f"[Calibration] Calibrating {model_name} for {context_size//1000}k context"
        )
        calibrated = self._calibrate_model(model_name, context_size)

        # Cache it
        if model_name not in self.calibrated_gpu_layers:
            self.calibrated_gpu_layers[model_name] = {}
        self.calibrated_gpu_layers[model_name][context_size] = calibrated

        return calibrated

    def get_models(self):
        """Return list of available models without loading the LLM.

        This allows /v1/models endpoint to work even when LLM is not loaded.
        """
        from ezlocalai.LLM import get_models

        return get_models()

    def _ensure_context_size(self, required_context: int):
        """Reload LLM with larger context if needed using smart GPU selection.

        Thread-safe: Uses _model_lock to prevent race conditions.

        IMPORTANT: When increasing context size, we must unload ALL models (including
        vision model) to free GPU VRAM before reloading. Otherwise the GPU will be
        full and the model will fall back to CPU, which is very slow.
        """
        with self._model_lock:
            if self.current_context and self.current_context >= required_context:
                # Current context is sufficient
                return

            # Need to reload with larger context
            logging.info(
                f"[LLM] Context {self.current_context//1000 if self.current_context else 0}k insufficient for {required_context:,} tokens, reloading at {required_context//1000}k"
            )

            model_name = self.current_llm_name

            # Store which models were loaded so we know what to potentially reload later
            had_vision_model = self.vision_llm is not None
            vision_model_name = self.vision_llm_name

            # Unload ALL models to free GPU VRAM for the larger context
            logging.info(
                "[LLM] Unloading all models to free GPU VRAM for larger context..."
            )

            # Unload vision model first
            if self.vision_llm:
                logging.debug(f"[LLM] Unloading vision model {self.vision_llm_name}")
                del self.vision_llm
                self.vision_llm = None
                self.vision_llm_name = None
                self.vision_llm_context = None

            # Unload primary model
            if self.primary_llm:
                logging.debug(f"[LLM] Unloading primary model {self.primary_llm_name}")
                del self.primary_llm
                self.primary_llm = None
                self.primary_llm_name = None
                self.primary_llm_context = None

            # Unload current active model (might be same as primary)
            if self.llm:
                del self.llm
                self.llm = None

            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for CUDA operations to complete
                # Log available VRAM after cleanup
                free_vram = get_free_vram_gb()
                logging.info(f"[LLM] After cleanup: {free_vram:.1f}GB VRAM free")

            # Load with new context - let smart GPU selection determine configuration
            start_time = time.time()
            self.llm = self._load_llm_resilient(
                model_name=model_name,
                max_tokens=required_context,
                # gpu_layers, main_gpu, tensor_split determined by smart selection
            )
            self.current_llm_name = model_name
            self.current_context = required_context

            # Also update primary model reference if this was the primary
            if (
                model_name == self.available_models[0]
                if self.available_models
                else None
            ):
                self.primary_llm = self.llm
                self.primary_llm_name = model_name
                self.primary_llm_context = required_context

            load_time = time.time() - start_time
            logging.info(
                f"[LLM] {model_name} reloaded at {required_context//1000}k context in {load_time:.2f}s"
            )

            # Note: We intentionally do NOT reload the vision model here.
            # It will be lazy-loaded on demand if needed, keeping VRAM available
            # for the large context model.

    def _swap_llm(self, requested_model: str, required_context: int = None):
        """Hot-swap to a different LLM if needed with smart GPU selection.

        Thread-safe: Uses _model_lock to prevent race conditions.

        Args:
            requested_model: Model name to swap to
            required_context: Minimum context size needed
        """
        with self._model_lock:
            # Check if this is a known model
            target_model = None

            for model_name in self.available_models:
                # Match by exact name or by the short name (last part after /)
                short_name = model_name.split("/")[-1].lower()
                requested_short = requested_model.split("/")[-1].lower()
                if (
                    model_name.lower() == requested_model.lower()
                    or short_name == requested_short
                ):
                    target_model = model_name
                    break

            if target_model is None:
                # Model not in available list, use current
                return

            # Determine context size
            target_context = required_context if required_context else 16384

            # Check if we already have this model loaded with sufficient context
            if (
                self.current_llm_name == target_model
                and self.current_context
                and self.current_context >= target_context
            ):
                return

            # Swap models - must unload old model first to free VRAM
            logging.debug(
                f"[LLM] Swapping to {target_model} at {target_context//1000}k context"
            )
            start_time = time.time()

            # Store old model info in case we need to rollback
            old_model_name = self.current_llm_name
            old_context = self.current_context or 16384

            # Destroy current LLM first to free VRAM
            if self.llm:
                logging.debug(f"[LLM] Unloading {old_model_name} to free VRAM")
                del self.llm
                self.llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Try to load new LLM with smart GPU selection
            try:
                self.llm = self._load_llm_resilient(
                    model_name=target_model,
                    max_tokens=target_context,
                    # Let smart GPU selection handle gpu_layers, main_gpu, tensor_split
                )
                self.current_llm_name = target_model
                self.current_context = target_context
                load_time = time.time() - start_time
                logging.debug(f"[LLM] {target_model} loaded in {load_time:.2f}s")
                if self.llm.is_vision:
                    logging.debug(f"[LLM] Vision capability enabled for {target_model}")
            except Exception as e:
                logging.error(f"[LLM] Failed to load {target_model}: {e}")
                # Rollback to old model with smart GPU selection
                logging.debug(f"[LLM] Rolling back to {old_model_name}")
                try:
                    self.llm = self._load_llm_resilient(
                        model_name=old_model_name,
                        max_tokens=old_context,
                        # Let smart GPU selection handle configuration
                    )
                    self.current_llm_name = old_model_name
                    self.current_context = old_context
                    logging.debug(f"[LLM] Rolled back to {old_model_name}")
                except Exception as rollback_error:
                    logging.error(
                        f"[LLM] CRITICAL: Failed to rollback to {old_model_name}: {rollback_error}"
                    )
                    # Last resort - try to load first available model at 16k with CPU fallback
                    for model_name in self.available_models:
                        try:
                            self.llm = self._load_llm_resilient(
                                model_name=model_name,
                                max_tokens=16384,
                                gpu_layers=0,  # Force CPU to maximize chance of success
                            )
                            self.current_llm_name = model_name
                            self.current_context = 16384
                            logging.debug(
                                f"[LLM] Recovered with {model_name} (CPU mode)"
                            )
                            break
                        except:
                            continue

    def _get_embedder(self):
        """Lazy load embedding model on demand."""
        if self.embedder is None:
            from ezlocalai.Embedding import Embedding

            logging.debug("[Embedding] Loading BGE-M3 on demand")
            start_time = time.time()
            self.embedder = Embedding()
            logging.debug(
                f"[Embedding] BGE-M3 loaded in {time.time() - start_time:.2f}s"
            )
        return self.embedder

    def _destroy_embedder_sync(self, embedder_ref):
        """Synchronous embedder destruction."""
        try:
            start_time = time.time()
            del embedder_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - start_time
            logging.debug(f"[Embedding] BGE-M3 unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[Embedding] Error during cleanup: {e}")

    def _destroy_embedder(self, async_cleanup: bool = True):
        """Destroy embedding model to free resources."""
        if self.embedder is not None:
            embedder_ref = self.embedder
            self.embedder = None

            if async_cleanup:
                logging.debug("[Embedding] Scheduling async unload...")
                _schedule_cleanup(self._destroy_embedder_sync, embedder_ref)
            else:
                self._destroy_embedder_sync(embedder_ref)

    def _get_stt(self):
        """Lazy load STT model on demand."""
        if self.stt is None:
            from ezlocalai.STT import STT

            logging.debug(f"[STT] Loading {self.current_stt} on demand")
            start_time = time.time()
            self.stt = STT(model=self.current_stt)
            logging.debug(
                f"[STT] {self.current_stt} loaded in {time.time() - start_time:.2f}s"
            )
        return self.stt

    def _destroy_stt_sync(self, stt_ref):
        """Synchronous STT destruction."""
        try:
            start_time = time.time()
            del stt_ref
            gc.collect()
            cleanup_time = time.time() - start_time
            logging.debug(f"[STT] Whisper unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[STT] Error during cleanup: {e}")

    def _destroy_stt(self, async_cleanup: bool = True):
        """Destroy STT model to free resources."""
        if self.stt is not None:
            stt_ref = self.stt
            self.stt = None

            if async_cleanup:
                logging.debug("[STT] Scheduling async unload...")
                _schedule_cleanup(self._destroy_stt_sync, stt_ref)
            else:
                self._destroy_stt_sync(stt_ref)

    def _get_img(self):
        """Lazy load IMG model on demand."""
        global img_import_success
        if self.img is None and img_import_success:
            IMG_MODEL = getenv("IMG_MODEL")
            if IMG_MODEL:
                logging.debug(f"[IMG] Loading {IMG_MODEL} on demand")
                start_time = time.time()
                # Auto-detect CUDA for image generation
                img_device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    self.img = IMG(
                        model=IMG_MODEL, local_uri=self.local_uri, device=img_device
                    )
                    logging.debug(
                        f"[IMG] {IMG_MODEL} loaded on {img_device} in {time.time() - start_time:.2f}s"
                    )
                except Exception as e:
                    logging.error(f"[IMG] Failed to load the model: {e}")
                    self.img = None
        return self.img

    def _destroy_img_sync(self, img_ref):
        """Synchronous IMG destruction."""
        try:
            start_time = time.time()
            del img_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - start_time
            logging.debug(f"[IMG] SDXL-Lightning unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[IMG] Error during cleanup: {e}")

    def _destroy_img(self, async_cleanup: bool = True):
        """Destroy IMG model to free resources."""
        if self.img is not None:
            img_ref = self.img
            self.img = None

            if async_cleanup:
                logging.debug("[IMG] Scheduling async unload...")
                _schedule_cleanup(self._destroy_img_sync, img_ref)
            else:
                self._destroy_img_sync(img_ref)

    def _get_llm(self, model_name: str = None, context_size: int = 16384):
        """Get LLM instance, using persistent models when possible.

        Persistent models (primary_llm, vision_llm) are kept loaded to avoid
        reload overhead for frequent requests. Large models are loaded on demand
        and unloaded after use.

        Uses smart GPU selection to determine optimal loading strategy:
        - GPU 0 only if it has enough VRAM
        - Tensor split if GPU 0 needs help from other GPUs
        - GPU 1 alone if GPU 0 is fully occupied
        - CPU fallback if no GPU has sufficient VRAM

        Thread-safe: Uses _model_lock to prevent race conditions when multiple
        requests try to access/load models simultaneously.
        """
        if model_name is None:
            model_name = self.available_models[0] if self.available_models else None

        if model_name is None:
            logging.warning("[LLM] No model available to load")
            return None

        # Acquire lock to prevent race conditions during model check/load
        with self._model_lock:
            logging.debug(
                f"[LLM _get_llm] Requested model='{model_name}', context={context_size}"
            )
            logging.debug(
                f"[LLM _get_llm] Current: llm={self.llm is not None}, name='{self.current_llm_name}', ctx={self.current_context}"
            )
            logging.debug(
                f"[LLM _get_llm] Primary: llm={self.primary_llm is not None}, name='{self.primary_llm_name}', ctx={self.primary_llm_context}"
            )
            logging.debug(
                f"[LLM _get_llm] Vision: llm={self.vision_llm is not None}, name='{self.vision_llm_name}', ctx={self.vision_llm_context}"
            )

            # Check if we already have the right model loaded with sufficient context
            if (
                self.llm is not None
                and self.current_llm_name == model_name
                and self.current_context
                and self.current_context >= context_size
            ):
                logging.debug(f"[LLM _get_llm] Using current self.llm (already loaded)")
                return self.llm

            # Check if this is one of our persistent models
            is_primary = model_name == self.primary_llm_name
            is_vision = model_name == self.vision_llm_name
            logging.debug(
                f"[LLM _get_llm] is_primary={is_primary}, is_vision={is_vision}"
            )

            # If we were using a large model, unload it first
            if self._using_large_model and self.llm is not None:
                logging.debug(
                    f"[LLM] Unloading large model to switch back to persistent model"
                )
                self._destroy_llm_temp()
                self._using_large_model = False

            # Try to use persistent models
            if is_primary and self.primary_llm is not None:
                if (
                    self.primary_llm_context
                    and self.primary_llm_context >= context_size
                ):
                    logging.debug(f"[LLM] Using persistent primary model: {model_name}")
                    self.llm = self.primary_llm
                    self.current_llm_name = model_name
                    self.current_context = self.primary_llm_context
                    return self.llm
                else:
                    # Need to reload primary with larger context
                    logging.info(
                        f"[LLM] Reloading primary model with larger context: {context_size//1000}k (had {self.primary_llm_context})"
                    )
                    del self.primary_llm
                    self.primary_llm = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if is_vision and self.vision_llm is not None:
                if self.vision_llm_context and self.vision_llm_context >= context_size:
                    logging.debug(f"[LLM] Using persistent vision model: {model_name}")
                    self.llm = self.vision_llm
                    self.current_llm_name = model_name
                    self.current_context = self.vision_llm_context
                    return self.llm
                else:
                    # Need to reload vision with larger context
                    logging.info(
                        f"[LLM] Reloading vision model with larger context: {context_size//1000}k (had {self.vision_llm_context})"
                    )
                    del self.vision_llm
                    self.vision_llm = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Loading a new model - check if it's a "large" model (not primary or vision)
            is_large_model = not is_primary and not is_vision
            logging.debug(f"[LLM] Loading new model - is_large_model={is_large_model}")

            if is_large_model:
                # Unload persistent models temporarily to make room for large model
                logging.debug(
                    f"[LLM] Loading large model {model_name}, temporarily unloading persistent models"
                )
                if self.primary_llm is not None:
                    del self.primary_llm
                    self.primary_llm = None
                if self.vision_llm is not None:
                    del self.vision_llm
                    self.vision_llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._using_large_model = True

            logging.debug(
                f"[LLM] Loading {model_name} (context: {context_size//1000}k)"
            )
            start_time = time.time()

            # Let _load_llm_resilient handle smart GPU selection
            new_llm = self._load_llm_resilient(
                model_name=model_name,
                max_tokens=context_size,
            )

            # Store in appropriate slot
            if is_primary:
                self.primary_llm = new_llm
                self.primary_llm_name = model_name
                self.primary_llm_context = context_size
            elif is_vision:
                self.vision_llm = new_llm
                self.vision_llm_name = model_name
                self.vision_llm_context = context_size

            self.llm = new_llm
            self.current_llm_name = model_name
            self.current_context = context_size

            load_time = time.time() - start_time
            logging.debug(f"[LLM] {model_name} loaded in {load_time:.2f}s")
            if self.llm.is_vision:
                logging.debug(f"[LLM] Vision capability enabled for {model_name}")

            return self.llm

    def _destroy_llm_temp(self):
        """Destroy temporary (large) LLM without affecting persistent models.

        Note: This method should only be called while holding _model_lock.
        """
        if self.llm is not None and self._using_large_model:
            logging.debug(
                f"[LLM] Unloading temporary large model: {self.current_llm_name}"
            )
            del self.llm
            self.llm = None
            self.current_llm_name = None
            self.current_context = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._using_large_model = False

    def _reload_persistent_models(self):
        """Reload persistent models after using a large model.

        Thread-safe: Uses _model_lock to prevent race conditions.
        """
        with self._model_lock:
            default_context = int(getenv("LLM_MAX_TOKENS", "8192"))

            # Reload primary if it was set
            if self.primary_llm_name and self.primary_llm is None:
                logging.debug(
                    f"[LLM] Reloading persistent primary model: {self.primary_llm_name}"
                )
                try:
                    self.primary_llm = self._load_llm_resilient(
                        model_name=self.primary_llm_name,
                        max_tokens=self.primary_llm_context or default_context,
                    )
                    self.llm = self.primary_llm
                    self.current_llm_name = self.primary_llm_name
                    self.current_context = self.primary_llm_context or default_context
                except Exception as e:
                    logging.warning(f"[LLM] Failed to reload primary model: {e}")

            # Reload vision if it was set and different from primary
            if (
                self.vision_llm_name
                and self.vision_llm is None
                and self.vision_llm_name != self.primary_llm_name
            ):
                logging.debug(
                    f"[LLM] Reloading persistent vision model: {self.vision_llm_name}"
                )
                try:
                    self.vision_llm = self._load_llm_resilient(
                        model_name=self.vision_llm_name,
                        max_tokens=self.vision_llm_context or default_context,
                    )
                except Exception as e:
                    logging.warning(f"[LLM] Failed to reload vision model: {e}")

    def _destroy_llm_sync(self, llm_ref, model_name: str):
        """Synchronous LLM destruction - runs the actual cleanup.

        This is called either directly or from a background thread.
        """
        try:
            start_time = time.time()
            del llm_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            cleanup_time = time.time() - start_time
            logging.debug(
                f"[LLM] {model_name} unloaded, VRAM freed in {cleanup_time:.2f}s"
            )
        except Exception as e:
            logging.error(f"[LLM] Error during cleanup of {model_name}: {e}")

    def _destroy_llm(self, async_cleanup: bool = True):
        """Destroy LLM to free VRAM.

        This is called after each inference to ensure VRAM is freed.
        The model will be reloaded on the next request via _get_llm().

        Thread-safe: Uses _model_lock to prevent race conditions.

        Args:
            async_cleanup: If True (default), run cleanup in background thread
                          for faster response times. If False, run synchronously.
        """
        with self._model_lock:
            if self.llm is not None:
                model_name = self.current_llm_name or "LLM"
                llm_ref = self.llm

                # Clear references immediately so new requests can load fresh
                self.llm = None
                self.current_llm_name = None
                self.current_context = None

                if async_cleanup:
                    logging.debug(f"[LLM] Scheduling async unload of {model_name}")
                    _schedule_cleanup(self._destroy_llm_sync, llm_ref, model_name)
                else:
                    logging.debug(f"[LLM] Unloading {model_name} synchronously")
                    self._destroy_llm_sync(llm_ref, model_name)

    def _get_tts(self):
        """Lazy load TTS model on demand."""
        if self.ctts is None:
            logging.debug("[CTTS] Loading Chatterbox TTS on demand")
            start_time = time.time()
            self.ctts = CTTS()
            logging.debug(
                f"[CTTS] Chatterbox TTS loaded in {time.time() - start_time:.2f}s"
            )
        return self.ctts

    def _destroy_tts_sync(self, tts_ref):
        """Synchronous TTS destruction."""
        try:
            start_time = time.time()
            del tts_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - start_time
            logging.debug(f"[CTTS] Chatterbox TTS unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[CTTS] Error during TTS cleanup: {e}")

    def _destroy_tts(self, async_cleanup: bool = True):
        """Destroy TTS model to free resources."""
        if self.ctts is not None:
            tts_ref = self.ctts
            self.ctts = None

            if async_cleanup:
                logging.debug("[CTTS] Scheduling async unload...")
                _schedule_cleanup(self._destroy_tts_sync, tts_ref)
            else:
                self._destroy_tts_sync(tts_ref)

    async def fallback_inference(self, messages):
        fallback_server = getenv("FALLBACK_SERVER")
        fallback_model = getenv("FALLBACK_MODEL")
        fallback_api_key = getenv("FALLBACK_API_KEY")
        if fallback_server == "":
            return "Unable to process request. Please try again later."
        from openai import Client

        client = Client(api_key=fallback_api_key, base_url=fallback_server)
        response = client.chat.completions.create(
            model=fallback_model, messages=messages
        )
        return response.choices[0].message.content

    async def pdf_to_audio(self, title, voice, pdf, chunk_size=200):
        # Sanitize title to prevent path traversal
        import re

        # First sanitize the input to only allow safe characters
        if not title or not isinstance(title, str):
            title = "output"
        # Remove any path separators and dangerous characters
        title = re.sub(r'[/\\:*?"<>|]', "", title)
        # Remove path traversal attempts
        title = title.replace("..", "")
        # Only allow alphanumeric, hyphen, underscore, space - strict allowlist
        title = re.sub(r"[^a-zA-Z0-9_\- ]", "", title)
        if not title:
            title = "output"
        title = title[:100]

        # Use basename to strip any remaining path components
        safe_title = os.path.basename(title)
        if not safe_title:
            safe_title = "output"

        outputs_dir = os.path.realpath(os.path.join(os.getcwd(), "outputs"))
        os.makedirs(outputs_dir, exist_ok=True)

        # Construct and normalize the path - CodeQL pattern from documentation
        fullpath = os.path.normpath(os.path.join(outputs_dir, f"{safe_title}.pdf"))
        # Verify with normalized version of path - exact CodeQL recommended pattern
        if not fullpath.startswith(outputs_dir):
            raise ValueError("Invalid path - potential path traversal")

        pdf_data = pdf.split(",")[1]
        pdf_bytes = base64.b64decode(pdf_data)
        # fullpath is verified safe - use it directly
        with open(fullpath, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)

        content = ""
        if fullpath.endswith(".pdf"):
            # fullpath was already verified above, use it directly
            with pdfplumber.open(fullpath) as pdf_doc:
                content = "\n".join([page.extract_text() for page in pdf_doc.pages])
        if not content:
            return
        tts = self._get_tts()
        result = await tts.generate(
            text=content,
            voice=voice,
            local_uri=self.local_uri,
            output_file_name=f"{safe_title}.wav",
        )
        self._destroy_tts()
        return result

    async def audio_to_audio(self, voice, audio):
        audio_type = audio.split(",")[0].split(":")[1].split(";")[0]
        audio_format = audio_type.split("/")[1]
        audio = audio.split(",")[1]
        audio = base64.b64decode(audio)
        stt = self._get_stt()
        text = stt.transcribe_audio(base64_audio=audio, audio_format=audio_format)
        self._destroy_stt()
        tts = self._get_tts()
        result = await tts.generate(text=text, voice=voice, local_uri=self.local_uri)
        self._destroy_tts()
        return result

    async def generate_image(self, prompt, response_format="url", size="512x512"):
        img = self._get_img()
        if img:
            img.local_uri = self.local_uri if response_format == "url" else None
            new_image = img.generate(
                prompt=prompt,
                size=size,
            )
            self._destroy_img()
            return new_image
        return ""

    def _apply_model_config_overrides(self, data: dict) -> dict:
        """Apply model-specific config overrides if the current model has them defined.

        Overrides only apply to parameters defined in MODEL_CONFIG_OVERRIDES for the
        current model. User-provided values are used for any parameter not in overrides.
        """
        if self.current_llm_name and self.current_llm_name in MODEL_CONFIG_OVERRIDES:
            overrides = MODEL_CONFIG_OVERRIDES[self.current_llm_name]
            for key, value in overrides.items():
                data[key] = value
            logging.debug(
                f"[Config] Applied model overrides for {self.current_llm_name}: {overrides}"
            )
        return data

    async def get_response(self, data, completion_type="chat"):
        data["local_uri"] = self.local_uri
        # Apply model-specific config overrides
        data = self._apply_model_config_overrides(data)
        images = []
        if "messages" in data:
            # Process messages to extract images and handle content types
            for i, message in enumerate(data["messages"]):
                if isinstance(message.get("content"), list):
                    # Extract text content and images from list format
                    text_content = ""
                    message_images = []
                    for content_item in message["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "text":
                                text_content += content_item.get("text", "")
                            elif "image_url" in content_item:
                                message_images.append(content_item)
                            elif "audio_url" in content_item:
                                audio_url = (
                                    content_item["audio_url"]["url"]
                                    if "url" in content_item["audio_url"]
                                    else content_item["audio_url"]
                                )
                                audio_format = "wav"
                                if audio_url.startswith("data:"):
                                    audio_url = audio_url.split(",")[1]
                                    audio_format = audio_url.split(";")[0]
                                else:
                                    audio_url = requests.get(audio_url).content
                                    audio_url = base64.b64encode(audio_url).decode(
                                        "utf-8"
                                    )
                                stt = self._get_stt()
                                transcribed_audio = stt.transcribe_audio(
                                    base64_audio=audio_url, audio_format=audio_format
                                )
                                self._destroy_stt()
                                text_content = f"Transcribed Audio: {transcribed_audio}\n\n{text_content}"
                        elif isinstance(content_item, str):
                            text_content += content_item

                    # Collect images for later processing
                    if message_images:
                        images.extend(message_images)

                    # For non-vision models or non-user messages, convert to string
                    # For vision models with the last user message, we'll handle this later
                    if not (
                        self.llm
                        and self.llm.is_vision
                        and message_images
                        and i == len(data["messages"]) - 1
                    ):
                        data["messages"][i]["content"] = text_content

            # Legacy handling for the old format (keeping for backward compatibility)
            # Skip if we already collected images in the modern format
            if not images and isinstance(data["messages"][-1]["content"], list):
                messages = data["messages"][-1]["content"]
                prompt = ""
                for message in messages:
                    if "text" in message:
                        prompt = message["text"]
                for message in messages:
                    if "image_url" in message:
                        images.append(message)
                    if "audio_url" in message:
                        audio_url = (
                            message["audio_url"]["url"]
                            if "url" in message["audio_url"]
                            else message["audio_url"]
                        )
                        audio_format = "wav"
                        if audio_url.startswith("data:"):
                            audio_url = audio_url.split(",")[1]
                            audio_format = audio_url.split(";")[0]
                        else:
                            audio_url = requests.get(audio_url).content
                            audio_url = base64.b64encode(audio_url).decode("utf-8")
                        stt = self._get_stt()
                        transcribed_audio = stt.transcribe_audio(
                            base64_audio=audio_url, audio_format=audio_format
                        )
                        self._destroy_stt()
                        prompt = f"Transcribed Audio: {transcribed_audio}\n\n{prompt}"
                # Convert list content back to string for LLM compatibility
                data["messages"][-1]["content"] = prompt

        # Use current context size - don't pre-estimate tokens
        # The model's actual tokenizer will determine if context is sufficient
        # If context is exceeded, error handling will reload with larger context
        # This avoids inaccurate character-based estimation causing unnecessary reloads
        required_context = (
            self.current_context
            if self.current_context
            else int(getenv("LLM_MAX_TOKENS", "16384"))
        )

        # Lazy load LLM with requested model and context size
        # Determine target model
        requested_model = data.get("model")
        target_model = None

        if requested_model and self.available_models:
            # Find matching model name from available models
            for model_name in self.available_models:
                short_name = model_name.split("/")[-1].lower()
                requested_short = requested_model.split("/")[-1].lower()
                if (
                    model_name.lower() == requested_model.lower()
                    or short_name == requested_short
                ):
                    target_model = model_name
                    break

            # If requested model not found in available models, fallback to first available
            if target_model is None:
                target_model = self.available_models[0]
                logging.debug(
                    f"[LLM] Requested model '{requested_model}' not available, using '{target_model}'"
                )
        elif self.available_models:
            # No model requested, use first available
            target_model = self.available_models[0]

        # Vision model fallback: If the target model is a vision model but no images
        # are in the request, fall back to a non-vision model if one is available.
        # This optimizes resource usage since vision models are heavier.
        if target_model and not images and self._is_vision_model(target_model):
            non_vision_model = self._find_non_vision_model()
            if non_vision_model:
                logging.debug(
                    f"[LLM] Request to vision model '{target_model}' has no images, "
                    f"falling back to non-vision model '{non_vision_model}'"
                )
                target_model = non_vision_model

        # Lazy load the LLM with calculated context (estimated prompt tokens + 16k headspace)
        self._get_llm(target_model, required_context)
        data["model"] = self.current_llm_name

        if "stop" in data:
            new_stop = self.llm.params["stop"]
            new_stop.append(data["stop"])
            data["stop"] = new_stop
        if "audio_format" in data:
            base64_audio = (
                data["messages"][-1]["content"]
                if completion_type == "chat"
                else data["prompt"]
            )
            stt = self._get_stt()
            prompt = stt.transcribe_audio(
                base64_audio=base64_audio,
                audio_format=data["audio_format"],
            )
            self._destroy_stt()
            if completion_type == "chat":
                data["messages"][-1]["content"] = prompt
            else:
                data["prompt"] = prompt
        user_message = (
            data["messages"][-1]["content"]
            if completion_type == "chat"
            else data["prompt"]
        )
        # Handle images with vision-capable LLM
        if self.llm and self.llm.is_vision and images:
            # xllamacpp expects images in base64 data URL format (PNG or JPEG)
            # Convert any remote URLs to base64 data URLs, and convert WebP/other formats to PNG
            from PIL import Image as PILImage
            from io import BytesIO

            processed_images = []
            for img in images:
                if "image_url" in img:
                    img_url = (
                        img["image_url"].get("url", "")
                        if isinstance(img["image_url"], dict)
                        else img["image_url"]
                    )
                    if img_url and not img_url.startswith("data:"):
                        # Fetch remote image and convert to base64
                        try:
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            }
                            img_response = requests.get(
                                img_url, timeout=30, headers=headers
                            )
                            img_response.raise_for_status()
                            content_type = img_response.headers.get(
                                "Content-Type", "image/jpeg"
                            )

                            # llama.cpp mmproj only supports certain formats (not WebP)
                            # Convert any non-standard format to PNG
                            if content_type in [
                                "image/webp",
                                "image/gif",
                                "image/bmp",
                                "image/tiff",
                                "image/avif",
                            ]:
                                try:
                                    pil_img = PILImage.open(
                                        BytesIO(img_response.content)
                                    )
                                    # Convert to RGB if necessary (for RGBA or palette images)
                                    if pil_img.mode in ("RGBA", "P", "LA"):
                                        pil_img = pil_img.convert("RGB")
                                    buffer = BytesIO()
                                    pil_img.save(buffer, format="PNG")
                                    img_bytes = buffer.getvalue()
                                    content_type = "image/png"
                                    logging.debug(
                                        f"[Vision] Converted {img_response.headers.get('Content-Type', 'unknown')} to PNG"
                                    )
                                except Exception as conv_err:
                                    logging.error(
                                        f"[Vision] Failed to convert image: {conv_err}"
                                    )
                                    img_bytes = img_response.content
                            else:
                                img_bytes = img_response.content

                            if not content_type.startswith("image/"):
                                content_type = "image/jpeg"
                            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                            processed_images.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{img_base64}"
                                    },
                                }
                            )
                            logging.debug(
                                f"[Vision] Converted remote image to base64 ({len(img_base64)} chars)"
                            )
                        except Exception as e:
                            logging.error(
                                f"[Vision] Failed to fetch remote image {img_url}: {e}"
                            )
                            continue
                    else:
                        # Already a data URL - check if it needs conversion
                        if img_url.startswith("data:image/webp") or img_url.startswith(
                            "data:image/gif"
                        ):
                            try:
                                # Extract base64 data and convert
                                header, encoded = img_url.split(",", 1)
                                img_bytes = base64.b64decode(encoded)
                                pil_img = PILImage.open(BytesIO(img_bytes))
                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    pil_img = pil_img.convert("RGB")
                                buffer = BytesIO()
                                pil_img.save(buffer, format="PNG")
                                img_base64 = base64.b64encode(buffer.getvalue()).decode(
                                    "utf-8"
                                )
                                processed_images.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        },
                                    }
                                )
                                logging.debug(
                                    f"[Vision] Converted data URL WebP/GIF to PNG"
                                )
                            except Exception as conv_err:
                                logging.error(
                                    f"[Vision] Failed to convert data URL: {conv_err}"
                                )
                                processed_images.append(img)
                        else:
                            processed_images.append(img)

            if completion_type == "chat":
                # Build proper multimodal message with text + images
                user_text = data["messages"][-1]["content"]
                if isinstance(user_text, list):
                    # Extract text from list content format
                    user_text = " ".join(
                        [
                            item.get("text", "")
                            for item in user_text
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                    )

                if processed_images:
                    # Create message with text + images in xllamacpp expected format
                    multimodal_content = [{"type": "text", "text": user_text}]
                    multimodal_content.extend(processed_images)
                    data["messages"][-1]["content"] = multimodal_content
                    logging.debug(
                        f"[Vision] Sending multimodal message with {len(processed_images)} image(s)"
                    )
                else:
                    # No images could be processed, fall back to text-only
                    data["messages"][-1]["content"] = user_text
                    logging.warning(
                        f"[Vision] No images could be processed, falling back to text-only"
                    )
        elif images and self.llm and not self.llm.is_vision:
            # Non-vision model received images - use vision model fallback
            logging.debug(
                f"[Vision Fallback] Non-vision model {self.current_llm_name} received {len(images)} image(s), using vision fallback"
            )
            user_text = (
                user_message if isinstance(user_message, str) else str(user_message)
            )

            # Get image description from vision model
            image_description = await self._describe_images_with_vision_model(
                images, user_text
            )

            if image_description:
                # Prepend image description to user message
                enhanced_message = f"[Image Description: {image_description}]\n\nUser's question: {user_text}"
                if completion_type == "chat":
                    data["messages"][-1]["content"] = enhanced_message
                else:
                    data["prompt"] = enhanced_message
                logging.debug(
                    "[Vision Fallback] Enhanced prompt with image description"
                )
            else:
                logging.warning(
                    "[Vision Fallback] Could not get image description, proceeding without images"
                )

        # Helper function to detect context size errors and retry with larger context
        def _is_context_error(error_msg: str) -> bool:
            error_lower = error_msg.lower()
            return any(
                pattern in error_lower
                for pattern in [
                    "context size",
                    "context length",
                    "exceeds",
                    "too long",
                    "token limit",
                    "max_tokens",
                    "maximum context",
                ]
            )

        def _estimate_prompt_tokens(messages_or_prompt, completion_type: str) -> int:
            """Estimate prompt tokens using character count approximation.

            This is intentionally aggressive to ensure we pre-allocate enough context
            for streaming requests where lazy errors can't be retried.

            We use chars/2.5 (instead of chars/4) because:
            - Code and technical content tokenizes at higher rates
            - Chat templates add overhead (role markers, special tokens)
            - Better to overestimate and have headroom than underestimate and fail
            """
            total_chars = 0
            if completion_type == "chat" and isinstance(messages_or_prompt, list):
                for msg in messages_or_prompt:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        # Multimodal content
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    total_chars += len(item.get("text", ""))
                                elif "image_url" in item:
                                    # Images add significant tokens for vision models
                                    # Estimate ~1000 tokens per image
                                    total_chars += 4000
                            elif isinstance(item, str):
                                total_chars += len(item)
                    # Add role overhead (special tokens, markers)
                    total_chars += 50  # More realistic overhead per message
            else:
                # Completion mode or string content
                total_chars = len(str(messages_or_prompt))

            # Aggressive estimate: chars/2.5 + 20% buffer for chat template overhead
            # This ensures we don't underestimate for code/technical content
            estimated_tokens = int((total_chars / 2.5) * 1.2)
            return estimated_tokens

        async def _try_inference_with_context_retry(
            chat_mode: bool, data: dict
        ) -> dict:
            """Try inference, and if context error occurs, reload model with larger context and retry.

            For streaming requests, we pre-estimate tokens and ensure sufficient context
            BEFORE starting the stream, since streaming errors occur lazily during iteration
            and cannot be retried mid-stream.
            """
            max_retries = 3
            current_context = self.current_context or 16384
            is_streaming = data.get("stream", False)

            # For streaming requests, pre-estimate tokens and ensure context size
            # This prevents the lazy context error during stream iteration
            if is_streaming:
                messages_or_prompt = (
                    data.get("messages") if chat_mode else data.get("prompt", "")
                )
                estimated_tokens = _estimate_prompt_tokens(
                    messages_or_prompt, "chat" if chat_mode else "completion"
                )
                required_context = calculate_context_size(estimated_tokens)

                logging.info(
                    f"[LLM] Streaming pre-check: estimated {estimated_tokens:,} tokens, "
                    f"required context {required_context:,}, current context {current_context:,}"
                )

                if required_context > current_context:
                    logging.info(
                        f"[LLM] Streaming request: estimated {estimated_tokens:,} tokens, "
                        f"pre-loading {required_context//1024}k context (current: {current_context//1024}k)"
                    )
                    self._ensure_context_size(required_context)
                    current_context = required_context
            else:
                logging.info(
                    f"[LLM] Non-streaming request, stream={data.get('stream')}"
                )

            for attempt in range(max_retries):
                try:
                    if chat_mode:
                        logging.info(
                            f"[LLM] Calling llm.chat with stream={data.get('stream', False)}, context={self.current_context}"
                        )
                        result = self.llm.chat(**data)
                        logging.info(f"[LLM] llm.chat returned type: {type(result)}")
                        return result
                    else:
                        return self.llm.completion(**data)
                except Exception as e:
                    error_msg = str(e)
                    if _is_context_error(error_msg) and attempt < max_retries - 1:
                        # Try to extract n_prompt_tokens from error message
                        # Format: "... [n_prompt_tokens=21922, n_ctx=16384]"
                        import re

                        prompt_tokens_match = re.search(
                            r"n_prompt_tokens=(\d+)", error_msg
                        )
                        if prompt_tokens_match:
                            needed_tokens = int(prompt_tokens_match.group(1))
                            logging.debug(
                                f"[LLM] Extracted n_prompt_tokens={needed_tokens} from error"
                            )
                        else:
                            # Fallback: try to find any large number
                            numbers = re.findall(r"(\d+)", error_msg)
                            if numbers:
                                # Use the largest number as required context
                                needed_tokens = max(int(n) for n in numbers)
                            else:
                                # Double current context as fallback
                                needed_tokens = current_context * 2

                        # Calculate new context: actual tokens needed + 16k headspace
                        new_context = calculate_context_size(needed_tokens)

                        if new_context > current_context:
                            logging.warning(
                                f"[LLM] Context error detected ({needed_tokens} prompt tokens), reloading with {new_context//1024}k context..."
                            )
                            self._ensure_context_size(new_context)
                            current_context = new_context
                            continue

                    # Not a context error or max retries reached, raise
                    raise

            # Should not reach here, but just in case
            if chat_mode:
                return self.llm.chat(**data)
            else:
                return self.llm.completion(**data)

        # Check if local LLM is available, if not use fallback server
        if self.llm is None:
            logging.warning("[LLM] No local model available, using fallback server...")
            if completion_type == "chat":
                response = await self.fallback_inference(data["messages"])
                # Wrap in expected format
                response = {
                    "choices": [{"message": {"content": response}}],
                    "model": "fallback",
                }
            else:
                response = await self.fallback_inference(
                    [{"role": "user", "content": data.get("prompt", "")}]
                )
                response = {"choices": [{"text": response}], "model": "fallback"}
        elif completion_type == "chat":
            try:
                response = await _try_inference_with_context_retry(
                    chat_mode=True, data=data
                )
            except Exception as e:
                import traceback

                logging.error(f"[LLM] Chat completion failed: {e}")
                logging.error(f"[LLM] Full traceback: {traceback.format_exc()}")
                logging.error(f"[LLM] Data that caused failure: {data}")
                response = await self.fallback_inference(data["messages"])
        else:
            try:
                response = await _try_inference_with_context_retry(
                    chat_mode=False, data=data
                )
            except Exception as e:
                import traceback

                logging.error(f"[LLM] Completion failed: {e}")
                logging.error(f"[LLM] Full traceback: {traceback.format_exc()}")
                logging.error(f"[LLM] Data that caused failure: {data}")
                response = await self.fallback_inference(
                    [{"role": "user", "content": data.get("prompt", "")}]
                )
        generated_image = None
        if "temperature" not in data:
            data["temperature"] = 0.5
        if "top_p" not in data:
            data["top_p"] = 0.9
        # IMG is lazy loaded - try to get it if IMG_MODEL is configured
        img = self._get_img() if getenv("IMG_MODEL") else None
        if img_import_success and img:
            user_message = (
                data["messages"][-1]["content"]
                if completion_type == "chat"
                else data["prompt"]
            )
            if isinstance(user_message, list):
                user_message = prompt
                for message in messages:
                    if "image_url" in message:
                        if "url" in message["image_url"]:
                            if not message["image_url"]["url"].startswith("data:"):
                                user_message += (
                                    "Uploaded Image:"
                                    + message["image_url"]["url"]
                                    + "\n"
                                )
            response_text = (
                response["choices"][0]["text"]
                if completion_type != "chat"
                else response["choices"][0]["message"]["content"]
            )
            if "data:" in user_message:
                user_message = user_message.replace(
                    user_message.split("data:")[1].split("'")[0], ""
                )
            img_gen_prompt = f"Users message: {user_message} \n\n{'The user uploaded an image, one does not need generated unless the user is specifically asking.' if images else ''} **The assistant is acting as sentiment analysis expert and only responds with a concise YES or NO answer on if the user would like an image as visual or a picture generated. No other explanation is needed!**\nWould the user potentially like an image generated based on their message?\nAssistant: "
            logging.debug(f"[IMG] Decision maker prompt: {img_gen_prompt}")
            try:
                create_img = self.llm.chat(
                    messages=[{"role": "system", "content": img_gen_prompt}],
                    max_tokens=10,
                    temperature=data["temperature"],
                    top_p=data["top_p"],
                )
            except:
                create_img = await self.fallback_inference(
                    [{"role": "system", "content": img_gen_prompt}]
                )
            create_img = str(create_img["choices"][0]["message"]["content"]).lower()
            logging.debug(f"[IMG] Decision maker response: {create_img}")
            if "yes" in create_img or "es," in create_img:
                img_prompt = f"**The assistant is acting as a Stable Diffusion Prompt Generator.**\n\nUsers message: {user_message} \nAssistant response: {response_text} \n\nImportant rules to follow:\n- Describe subjects in detail, specify image type (e.g., digital illustration), art style (e.g., steampunk), and background. Include art inspirations (e.g., Art Station, specific artists). Detail lighting, camera (type, lens, view), and render (resolution, style). The weight of a keyword can be adjusted by using the syntax (((keyword))) , put only those keyword inside ((())) which is very important because it will have more impact so anything wrong will result in unwanted picture so be careful. Realistic prompts: exclude artist, specify lens. Separate with double lines. Max 60 words, avoiding 'real' for fantastical.\n- Based on the message from the user and response of the assistant, you will need to generate one detailed stable diffusion image generation prompt based on the context of the conversation to accompany the assistant response.\n- The prompt can only be up to 60 words long, so try to be concise while using enough descriptive words to make a proper prompt.\n- Following all rules will result in a $2000 tip that you can spend on anything!\n- Must be in markdown code block to be parsed out and only provide prompt in the code block, nothing else.\nStable Diffusion Prompt Generator: "
                try:
                    image_generation_prompt = self.llm.chat(
                        messages=[{"role": "system", "content": img_prompt}],
                        max_tokens=100,
                        temperature=data["temperature"],
                        top_p=data["top_p"],
                    )
                except:
                    image_generation_prompt = await self.fallback_inference(
                        [{"role": "system", "content": img_prompt}]
                    )
                image_generation_prompt = str(
                    image_generation_prompt["choices"][0]["message"]["content"]
                )
                logging.debug(
                    f"[IMG] Image generation response: {image_generation_prompt}"
                )
                if "```markdown" in image_generation_prompt:
                    image_generation_prompt = image_generation_prompt.split(
                        "```markdown"
                    )[1]
                    image_generation_prompt = image_generation_prompt.split("```")[0]
                generated_image = self.img.generate(prompt=image_generation_prompt)
            # Destroy IMG model after use to free VRAM (even if no image was generated)
            self._destroy_img()
        audio_response = None
        if "voice" in data:
            text_response = (
                response["choices"][0]["text"]
                if completion_type != "chat"
                else response["choices"][0]["message"]["content"]
            )
            language = data["language"] if "language" in data else "en"
            tts = self._get_tts()
            audio_response = await tts.generate(
                text=text_response,
                voice=data["voice"],
                language=language,
                local_uri=self.local_uri,
            )
            self._destroy_tts()
            if completion_type != "chat":
                response["choices"][0]["text"] = f"{text_response}\n{audio_response}"
            else:
                response["choices"][0]["message"][
                    "content"
                ] = f"{text_response}\n{audio_response}"
        if generated_image:
            if completion_type != "chat":
                response["choices"][0]["text"] += f"\n\n{generated_image}"
            else:
                response["choices"][0]["message"]["content"] += f"\n\n{generated_image}"

        # Only log JSON if response is not a generator (streaming mode)
        is_streaming = (
            hasattr(response, "__next__")
            or hasattr(response, "__iter__")
            and not isinstance(response, (dict, list))
        )

        if not is_streaming:
            logging.debug(f"[ezlocalai] {json.dumps(response, indent=2)}")
            # Keep the model loaded - no need to reload after each request
            # The higher context model works fine for smaller prompts too
        else:
            logging.debug(f"[ezlocalai] Streaming response generated")
            # For streaming, wrap the generator to handle cleanup after consumption
            original_response = response
            using_large = self._using_large_model
            pipes_self = self  # Capture self for use in wrapper
            data_copy = data.copy()  # Capture data for potential retry

            def streaming_wrapper():
                try:
                    for chunk in original_response:
                        yield chunk
                except Exception as e:
                    error_msg = str(e)
                    # Check if this is a context size error
                    if _is_context_error(error_msg):
                        # Extract token count from error if available
                        import re

                        prompt_tokens_match = re.search(
                            r"n_prompt_tokens=(\d+)", error_msg
                        )
                        if prompt_tokens_match:
                            needed_tokens = int(prompt_tokens_match.group(1))
                        else:
                            # Try to find any number that looks like a token count
                            numbers = re.findall(r"(\d{4,})", error_msg)  # 4+ digits
                            needed_tokens = (
                                max(int(n) for n in numbers) if numbers else 0
                            )

                        logging.error(
                            f"[STREAMING] Context size error during streaming. "
                            f"Prompt required {needed_tokens:,} tokens but context was insufficient. "
                            f"The model will be reloaded with larger context for the next request."
                        )
                        # Pre-load larger context for next request
                        if needed_tokens > 0:
                            new_context = calculate_context_size(needed_tokens)
                            pipes_self._ensure_context_size(new_context)
                    # Re-raise the error to let caller handle it
                    raise
                # No cleanup needed - keep the model loaded for subsequent requests

            response = streaming_wrapper()

        return response, audio_response
