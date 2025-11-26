import os
import logging
import time
import math
from dotenv import load_dotenv
from ezlocalai.LLM import LLM
from ezlocalai.CTTS import CTTS
from pyngrok import ngrok
import requests
import base64
import pdfplumber
import json
from Globals import getenv
import gc
import torch

try:
    from ezlocalai.IMG import IMG

    img_import_success = True
except ImportError:
    img_import_success = False

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


def get_per_gpu_vram_gb() -> list:
    """Get VRAM for each GPU as a list of GB values."""
    if torch.cuda.is_available():
        vram_list = []
        for i in range(torch.cuda.device_count()):
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            vram_list.append(math.floor(vram_gb))
        return vram_list
    return []


def round_context_to_32k(token_count: int) -> int:
    """Round up token count to nearest 32k for context sizing."""
    return math.ceil(token_count / 32768) * 32768


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

        # Auto-detect multi-GPU configuration
        self.gpu_count = get_gpu_count()
        self.per_gpu_vram = get_per_gpu_vram_gb()

        # Auto-detect total VRAM budget across all GPUs (rounded down to nearest 1GB for safety margin)
        self.vram_budget_gb = get_available_vram_gb()

        # Calculate tensor split for multi-GPU
        self.tensor_split = calculate_tensor_split()

        if self.vram_budget_gb > 0:
            if self.gpu_count > 1:
                logging.info(
                    f"[VRAM] Multi-GPU detected: {self.gpu_count} GPUs, "
                    f"{self.vram_budget_gb}GB total VRAM budget "
                    f"(per GPU: {self.per_gpu_vram})"
                )
            else:
                logging.info(
                    f"[VRAM] Auto-detected {self.vram_budget_gb}GB VRAM budget"
                )

        # Parse model list: "model1,model2" (simple comma-separated)
        model_config = getenv("DEFAULT_MODEL")
        self.available_models = []  # List of model names
        self.calibrated_gpu_layers = {}  # {model_name: {context: gpu_layers}}
        self.current_llm_name = None
        self.current_context = None  # Track current context size
        self.llm = None

        if model_config.lower() != "none":
            for model_entry in model_config.split(","):
                model_name = model_entry.strip()
                # Strip any legacy @tokens suffix for backward compat
                if "@" in model_name:
                    model_name = model_name.rsplit("@", 1)[0]
                if model_name and model_name not in self.available_models:
                    self.available_models.append(model_name)

            # Pre-calibrate models at 32k context (baseline) if VRAM available
            if (
                self.vram_budget_gb > 0
                and torch.cuda.is_available()
                and self.available_models
            ):
                base_context = 32768
                logging.info(
                    f"[Calibration] Pre-calibrating {len(self.available_models)} models at {base_context//1000}k context..."
                )
                for model_name in self.available_models:
                    if model_name not in self.calibrated_gpu_layers:
                        self.calibrated_gpu_layers[model_name] = {}
                    calibrated = self._calibrate_model(model_name, base_context)
                    self.calibrated_gpu_layers[model_name][base_context] = calibrated
                logging.info(
                    f"[Calibration] Models calibrated at {base_context//1000}k: {[(m, self.calibrated_gpu_layers[m][base_context]) for m in self.available_models]}"
                )

            # Load the first model with baseline 32k context
            if self.available_models:
                first_model = self.available_models[0]
                base_context = 32768
                gpu_layers = self.calibrated_gpu_layers.get(first_model, {}).get(
                    base_context
                )
                logging.info(
                    f"[LLM] {first_model} loading (context: {base_context//1000}k, gpu_layers: {gpu_layers or 'auto'}). Please wait..."
                )
                start_time = time.time()
                self.llm = LLM(
                    model=first_model, max_tokens=base_context, gpu_layers=gpu_layers
                )
                load_time = time.time() - start_time
                self.current_llm_name = first_model
                self.current_context = base_context
                logging.info(f"[LLM] {first_model} loaded in {load_time:.2f}s.")
                if self.llm.is_vision:
                    logging.info(f"[LLM] Vision capability enabled for {first_model}.")

                # Log available models
                if len(self.available_models) > 1:
                    logging.info(
                        f"[LLM] Available models: {', '.join(self.available_models)}"
                    )

        # Preload TTS to warm the cache, then unload to free VRAM
        # This way TTS loads fast (~5s) when needed via lazy loading
        self.ctts = None
        if getenv("TTS_ENABLED").lower() == "true":
            logging.info("[CTTS] Preloading Chatterbox TTS to warm cache...")
            start_time = time.time()
            self.ctts = CTTS()
            load_time = time.time() - start_time
            logging.info(
                f"[CTTS] Chatterbox TTS preloaded in {load_time:.2f}s, unloading to free VRAM..."
            )
            self._destroy_tts()
            logging.info("[CTTS] TTS unloaded, will lazy-load on first TTS request.")

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
        logging.info(
            f"[Calibration] Using native estimation for {model_name} (budget: {self.vram_budget_gb}GB)..."
        )

        # Get GPU info
        devices = get_device_info()
        gpus = []
        for dev in devices:
            # Check if it's a GPU device
            dev_type = str(dev.get("type", ""))
            if "GPU" in dev_type:
                # Use VRAM budget as available memory for estimation
                # This is the target we want to fit within, not current free memory
                budget_bytes = self.vram_budget_gb * 1024 * 1024 * 1024
                gpus.append(
                    {
                        "name": dev["name"],
                        "memory_free": budget_bytes,  # Use budget, not actual free
                        "memory_total": dev["memory_total"],
                    }
                )

        if not gpus:
            logging.warning("[Calibration] No GPU devices found for estimation")
            return None

        logging.info(
            f"[Calibration] GPUs: {[g['name'] + ' (' + str(round(g['memory_free']/1e9, 1)) + 'GB budget)' for g in gpus]}"
        )

        # Download model file to get path
        try:
            model_path = self._get_model_path(model_name)
            if not model_path:
                return None

            logging.info(f"[Calibration] Model path: {model_path}")

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

            logging.info(f"[Calibration] Native estimation result: {result}")

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

            logging.info(
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

        # Remember current model
        original_model = self.current_llm_name
        original_context = self.current_context

        try:
            # Swap to vision model temporarily
            logging.info(
                f"[Vision Fallback] Swapping to {vision_model} to describe {len(images)} image(s)"
            )
            self._swap_llm(vision_model, 32768)  # Use 32k context for image description

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

            logging.info(
                f"[Vision Fallback] Got image description ({len(description)} chars)"
            )
            return description

        except Exception as e:
            logging.error(f"[Vision Fallback] Failed to describe images: {e}")
            return None
        finally:
            # Swap back to original model
            if original_model and original_model != self.current_llm_name:
                logging.info(f"[Vision Fallback] Swapping back to {original_model}")
                self._swap_llm(original_model, original_context)

    def _calibrate_binary_search(self, model_name: str, max_tokens: int) -> int:
        """Calibrate using binary search (fallback method).

        Uses binary search to efficiently find the highest GPU layer count that
        fits within VRAM budget. Returns the optimal number of GPU layers.
        """
        # Binary search for optimal layers - start at 70 as most models have 40-80 layers
        low = 0
        high = 70
        best_layers = 0  # Default to CPU if nothing works

        logging.info(
            f"[Calibration] Binary search for {model_name} (budget: {self.vram_budget_gb}GB)..."
        )

        while low <= high:
            mid = (low + high) // 2

            try:
                # Clear VRAM before test
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logging.info(
                    f"[Calibration] Testing {model_name} with {mid} GPU layers..."
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
                    logging.info(
                        f"[Calibration] {mid} layers OK ({vram_used:.1f}GB), trying higher..."
                    )
                    low = mid + 1
                else:
                    # Too much VRAM, try lower
                    logging.info(
                        f"[Calibration] {mid} layers too high ({vram_used:.1f}GB), trying lower..."
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

        logging.info(
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
        logging.info(
            f"[Calibration] Calibrating {model_name} for {context_size//1000}k context..."
        )
        calibrated = self._calibrate_model(model_name, context_size)

        # Cache it
        if model_name not in self.calibrated_gpu_layers:
            self.calibrated_gpu_layers[model_name] = {}
        self.calibrated_gpu_layers[model_name][context_size] = calibrated

        return calibrated

    def _ensure_context_size(self, required_context: int):
        """Reload LLM with larger context if needed.

        Context is rounded up to nearest 32k to avoid frequent reloads.
        """
        rounded_context = round_context_to_32k(required_context)

        if self.current_context and self.current_context >= rounded_context:
            # Current context is sufficient
            return

        # Need to reload with larger context
        logging.info(
            f"[LLM] Context {self.current_context//1000 if self.current_context else 0}k insufficient for {required_context:,} tokens, reloading at {rounded_context//1000}k..."
        )

        model_name = self.current_llm_name
        gpu_layers = self._get_gpu_layers_for_model(model_name, rounded_context)

        # Unload current model
        if self.llm:
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load with new context
        start_time = time.time()
        self.llm = LLM(
            model=model_name, max_tokens=rounded_context, gpu_layers=gpu_layers
        )
        self.current_context = rounded_context
        load_time = time.time() - start_time
        logging.info(
            f"[LLM] {model_name} reloaded at {rounded_context//1000}k context ({gpu_layers} GPU layers) in {load_time:.2f}s"
        )

    def _swap_llm(self, requested_model: str, required_context: int = None):
        """Hot-swap to a different LLM if needed.

        Args:
            requested_model: Model name to swap to
            required_context: Minimum context size needed (will be rounded up to 32k)
        """
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

        # Determine context size (round up to nearest 32k)
        target_context = (
            round_context_to_32k(required_context) if required_context else 32768
        )

        # Check if we already have this model loaded with sufficient context
        if (
            self.current_llm_name == target_model
            and self.current_context
            and self.current_context >= target_context
        ):
            return

        # Get calibrated GPU layers for target context
        target_gpu_layers = self._get_gpu_layers_for_model(target_model, target_context)

        # Swap models - must unload old model first to free VRAM
        logging.info(
            f"[LLM] Swapping to {target_model} at {target_context//1000}k context ({target_gpu_layers} GPU layers)..."
        )
        start_time = time.time()

        # Store old model info in case we need to rollback
        old_model_name = self.current_llm_name
        old_context = self.current_context or 32768
        old_gpu_layers = (
            self._get_gpu_layers_for_model(old_model_name, old_context)
            if old_model_name
            else None
        )

        # Destroy current LLM first to free VRAM
        if self.llm:
            logging.info(f"[LLM] Unloading {old_model_name} to free VRAM...")
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Try to load new LLM
        try:
            self.llm = LLM(
                model=target_model,
                max_tokens=target_context,
                gpu_layers=target_gpu_layers,
            )
            self.current_llm_name = target_model
            self.current_context = target_context
            load_time = time.time() - start_time
            logging.info(f"[LLM] {target_model} loaded in {load_time:.2f}s.")
            if self.llm.is_vision:
                logging.info(f"[LLM] Vision capability enabled for {target_model}.")
        except Exception as e:
            logging.error(f"[LLM] Failed to load {target_model}: {e}")
            # Rollback to old model
            logging.info(f"[LLM] Rolling back to {old_model_name}...")
            try:
                self.llm = LLM(
                    model=old_model_name,
                    max_tokens=old_context,
                    gpu_layers=old_gpu_layers,
                )
                self.current_llm_name = old_model_name
                self.current_context = old_context
                logging.info(f"[LLM] Rolled back to {old_model_name}")
            except Exception as rollback_error:
                logging.error(
                    f"[LLM] CRITICAL: Failed to rollback to {old_model_name}: {rollback_error}"
                )
                # Last resort - try to load first available model at 32k
                for model_name in self.available_models:
                    try:
                        gpu_layers = self._get_gpu_layers_for_model(model_name, 32768)
                        self.llm = LLM(
                            model=model_name, max_tokens=32768, gpu_layers=gpu_layers
                        )
                        self.current_llm_name = model_name
                        self.current_context = 32768
                        logging.info(f"[LLM] Recovered with {model_name}")
                        break
                    except:
                        continue

    def _get_embedder(self):
        """Lazy load embedding model on demand."""
        if self.embedder is None:
            from ezlocalai.Embedding import Embedding

            logging.info("[Embedding] Loading BGE-M3 on demand...")
            start_time = time.time()
            self.embedder = Embedding()
            logging.info(
                f"[Embedding] BGE-M3 loaded in {time.time() - start_time:.2f}s."
            )
        return self.embedder

    def _destroy_embedder(self):
        """Destroy embedding model to free resources."""
        if self.embedder is not None:
            del self.embedder
            self.embedder = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("[Embedding] BGE-M3 unloaded to free resources.")

    def _get_stt(self):
        """Lazy load STT model on demand."""
        if self.stt is None:
            from ezlocalai.STT import STT

            logging.info(f"[STT] Loading {self.current_stt} on demand...")
            start_time = time.time()
            self.stt = STT(model=self.current_stt)
            logging.info(
                f"[STT] {self.current_stt} loaded in {time.time() - start_time:.2f}s."
            )
        return self.stt

    def _destroy_stt(self):
        """Destroy STT model to free resources."""
        if self.stt is not None:
            del self.stt
            self.stt = None
            gc.collect()
            logging.info("[STT] Whisper unloaded to free resources.")

    def _get_img(self):
        """Lazy load IMG model on demand."""
        global img_import_success
        if self.img is None and img_import_success:
            IMG_MODEL = getenv("IMG_MODEL")
            if IMG_MODEL:
                logging.info(f"[IMG] Loading {IMG_MODEL} on demand...")
                start_time = time.time()
                # Auto-detect CUDA for image generation
                img_device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    self.img = IMG(
                        model=IMG_MODEL, local_uri=self.local_uri, device=img_device
                    )
                    logging.info(
                        f"[IMG] {IMG_MODEL} loaded on {img_device} in {time.time() - start_time:.2f}s."
                    )
                except Exception as e:
                    logging.error(f"[IMG] Failed to load the model: {e}")
                    self.img = None
        return self.img

    def _destroy_img(self):
        """Destroy IMG model to free resources."""
        if self.img is not None:
            del self.img
            self.img = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("[IMG] SDXL-Lightning unloaded to free resources.")

    def _get_tts(self):
        """Lazy load TTS model on demand."""
        if self.ctts is None:
            logging.info("[CTTS] Loading Chatterbox TTS on demand...")
            start_time = time.time()
            self.ctts = CTTS()
            logging.info(
                f"[CTTS] Chatterbox TTS loaded in {time.time() - start_time:.2f}s."
            )
        return self.ctts

    def _destroy_tts(self):
        """Destroy TTS model to free resources."""
        if self.ctts is not None:
            del self.ctts
            self.ctts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("[CTTS] Chatterbox TTS unloaded to free resources.")

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
        filename = f"{title}.pdf"
        file_path = os.path.join(os.getcwd(), "outputs", filename)
        pdf = pdf.split(",")[1]
        pdf = base64.b64decode(pdf)
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(pdf)
        content = ""
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                content = "\n".join([page.extract_text() for page in pdf.pages])
        if not content:
            return
        tts = self._get_tts()
        return await tts.generate(
            text=content,
            voice=voice,
            local_uri=self.local_uri,
            output_file_name=f"{title}.wav",
        )

    async def audio_to_audio(self, voice, audio):
        audio_type = audio.split(",")[0].split(":")[1].split(";")[0]
        audio_format = audio_type.split("/")[1]
        audio = audio.split(",")[1]
        audio = base64.b64decode(audio)
        stt = self._get_stt()
        text = stt.transcribe_audio(base64_audio=audio, audio_format=audio_format)
        self._destroy_stt()
        tts = self._get_tts()
        return await tts.generate(text=text, voice=voice, local_uri=self.local_uri)

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

        # Estimate token count for context sizing
        # Rough estimate: 4 chars per token + max_tokens for generation headroom
        total_chars = 0
        if completion_type == "chat" and "messages" in data:
            for msg in data["messages"]:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            total_chars += len(item.get("text", ""))
        elif "prompt" in data:
            total_chars = len(data.get("prompt", ""))

        # Estimate tokens (4 chars/token) + generation tokens + buffer
        estimated_prompt_tokens = total_chars // 4
        max_tokens = data.get("max_tokens", 2048)
        required_context = estimated_prompt_tokens + max_tokens + 1000  # 1k buffer

        if data["model"]:
            # Check if we need to swap to a different model (with context size)
            self._swap_llm(data["model"], required_context)
            data["model"] = self.current_llm_name
        else:
            # Same model, but maybe need larger context
            self._ensure_context_size(required_context)
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
                                    logging.info(
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
                            logging.info(
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
                                logging.info(
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
                    logging.info(
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
            logging.info(
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
                logging.info("[Vision Fallback] Enhanced prompt with image description")
            else:
                logging.warning(
                    "[Vision Fallback] Could not get image description, proceeding without images"
                )

        if completion_type == "chat":
            try:
                response = self.llm.chat(**data)
            except Exception as e:
                import traceback

                logging.error(f"[LLM] Chat completion failed: {e}")
                logging.error(f"[LLM] Full traceback: {traceback.format_exc()}")
                logging.error(f"[LLM] Data that caused failure: {data}")
                response = await self.fallback_inference(data["messages"])
        else:
            try:
                response = self.llm.completion(**data)
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
            logging.info(f"[IMG] Decision maker prompt: {img_gen_prompt}")
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
            logging.info(f"[IMG] Decision maker response: {create_img}")
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
                logging.info(
                    f"[IMG] Image generation response: {image_generation_prompt}"
                )
                if "```markdown" in image_generation_prompt:
                    image_generation_prompt = image_generation_prompt.split(
                        "```markdown"
                    )[1]
                    image_generation_prompt = image_generation_prompt.split("```")[0]
                generated_image = self.img.generate(prompt=image_generation_prompt)
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
        if not hasattr(response, "__next__"):
            logging.info(f"[ezlocalai] {json.dumps(response, indent=2)}")
        else:
            logging.info(f"[ezlocalai] Streaming response generated")
        return response, audio_response
