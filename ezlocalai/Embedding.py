import logging
import math
import os

import torch
import xllamacpp as xlc

from Globals import getenv
from ezlocalai.LLM import download_model


def _get_int_env(name: str, default: int) -> int:
    value = getenv(name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        logging.warning("[Embedding] Invalid %s=%r; using %s", name, value, default)
        return default


def _get_float_env(name: str, default: float) -> float:
    value = getenv(name, str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        logging.warning("[Embedding] Invalid %s=%r; using %s", name, value, default)
        return default


def _field_value(reader, *keys):
    for key in keys:
        field = reader.fields.get(key)
        if field is None:
            continue
        try:
            value = field.parts[-1]
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, list):
                if (
                    key == "general.architecture"
                    and value
                    and all(isinstance(x, int) for x in value)
                ):
                    try:
                        return bytes(value).decode("utf-8")
                    except Exception:
                        return value[0] if len(value) == 1 else value
                return value[0] if len(value) == 1 else value
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore")
            return value
        except Exception:
            continue
    return None


def _get_model_metadata(model_path: str) -> dict:
    metadata = {
        "layers": 28,
        "kv_heads": 8,
        "key_length": 128,
        "value_length": 128,
    }
    try:
        import gguf

        reader = gguf.GGUFReader(model_path)
        architecture = _field_value(reader, "general.architecture")
        prefixes = []
        if architecture:
            prefixes.append(str(architecture))
        prefixes.extend(["qwen3", "qwen2", "llama"])

        for prefix in dict.fromkeys(prefixes):
            layers = _field_value(reader, f"{prefix}.block_count")
            kv_heads = _field_value(reader, f"{prefix}.attention.head_count_kv")
            key_length = _field_value(reader, f"{prefix}.attention.key_length")
            value_length = _field_value(reader, f"{prefix}.attention.value_length")

            if layers:
                metadata["layers"] = int(layers)
            if kv_heads:
                metadata["kv_heads"] = int(kv_heads)
            if key_length:
                metadata["key_length"] = int(key_length)
            if value_length:
                metadata["value_length"] = int(value_length)
            if layers or kv_heads or key_length or value_length:
                break
    except Exception as e:
        logging.debug("[Embedding] GGUF metadata read failed: %s", e)
    return metadata


def _cache_type_bytes(cache_type: str) -> float:
    cache_type = (cache_type or "f16").strip().lower()
    if cache_type in ("f32", "float32"):
        return 4.0
    if cache_type in ("q8_0", "q8"):
        return 1.0
    if cache_type in ("q4_0", "q4"):
        return 0.5
    return 2.0


def _estimate_embedding_vram(
    model_path: str,
    context_length: int,
    n_parallel: int,
    ubatch_size: int,
    cache_type: str,
) -> dict:
    metadata = _get_model_metadata(model_path)
    layers = max(1, int(metadata["layers"]))
    total_ctx = max(1, int(context_length)) * max(1, int(n_parallel))
    file_size_gb = os.path.getsize(model_path) / (1024**3)

    cache_bytes = _cache_type_bytes(cache_type)
    kv_per_layer_gb = (
        total_ctx
        * int(metadata["kv_heads"])
        * (int(metadata["key_length"]) + int(metadata["value_length"]))
        * cache_bytes
    ) / (1024**3)

    # Prompt-processing buffers scale mostly with ubatch. The coefficient is
    # intentionally conservative and calibrated from the 8k ubatch OOM observed
    # on the local 3090 Ti.
    compute_gb = 0.25 + (max(1, int(ubatch_size)) * 0.0007)
    layer_weight_gb = (file_size_gb * 0.9) / layers
    fixed_weight_gb = file_size_gb * 0.1
    per_layer_gb = layer_weight_gb + kv_per_layer_gb
    total_gb = fixed_weight_gb + compute_gb + (per_layer_gb * layers)

    return {
        "layers": layers,
        "total_gb": total_gb,
        "fixed_gb": fixed_weight_gb + compute_gb,
        "per_layer_gb": per_layer_gb,
        "weights_gb": file_size_gb,
        "kv_gb": kv_per_layer_gb * layers,
        "compute_gb": compute_gb,
    }


def _get_cuda_free_gb(gpu_index: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        gpu_index = gpu_index if 0 <= gpu_index < torch.cuda.device_count() else 0
        free, _ = torch.cuda.mem_get_info(gpu_index)
        return free / (1024**3)
    except Exception as e:
        logging.warning("[Embedding] Could not probe CUDA free memory: %s", e)
        return 0.0


def _is_memory_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in ("out of memory", "cuda", "vram", "alloc", "memory", "resource")
    )


def _normalize(vector):
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm <= 0:
        return vector
    return [float(x) / norm for x in vector]


class Embedding:
    def __init__(
        self,
        model_name: str = None,
        models_dir: str = "./models",
        force_cpu: bool = False,
        main_gpu: int = None,
        gpu_layers: int = None,
    ):
        self.model_name = model_name or getenv("EMBEDDING_MODEL")
        self.model_alias = getenv("EMBEDDING_MODEL_ALIAS") or self.model_name
        self.quant_type = getenv("EMBEDDING_QUANT_TYPE", "Q8_0")
        self.context_length = _get_int_env("EMBEDDING_CONTEXT_LENGTH", 32768)
        self.n_parallel = max(1, _get_int_env("EMBEDDING_N_PARALLEL", 1))
        self.batch_size = min(
            _get_int_env("EMBEDDING_BATCH_SIZE", 512), self.context_length
        )
        self.ubatch_size = min(
            _get_int_env("EMBEDDING_UBATCH_SIZE", 512), self.batch_size
        )
        if self.batch_size > self.ubatch_size:
            logging.info(
                "[Embedding] EMBEDDING_BATCH_SIZE=%s is larger than "
                "EMBEDDING_UBATCH_SIZE=%s; using %s for both",
                self.batch_size,
                self.ubatch_size,
                self.ubatch_size,
            )
            self.batch_size = self.ubatch_size
        self.kv_cache_type = getenv("EMBEDDING_KV_CACHE_TYPE", "f16").strip().lower()

        model_path, _ = download_model(
            model_name=self.model_name,
            models_dir=models_dir,
            quantization_type=self.quant_type,
            skip_mmproj=True,
        )

        if gpu_layers is None:
            raw_gpu_layers = getenv("EMBEDDING_GPU_LAYERS", "auto")
            if raw_gpu_layers and raw_gpu_layers.strip().lower() not in (
                "",
                "auto",
                "-2",
            ):
                try:
                    gpu_layers = int(raw_gpu_layers)
                except ValueError:
                    logging.warning(
                        "[Embedding] Invalid EMBEDDING_GPU_LAYERS=%r; using auto",
                        raw_gpu_layers,
                    )
                    gpu_layers = None

        if main_gpu is None:
            raw_main_gpu = getenv("EMBEDDING_MAIN_GPU", "")
            if not raw_main_gpu:
                raw_main_gpu = getenv("MAIN_GPU", "0")
            try:
                main_gpu = int(raw_main_gpu)
            except (TypeError, ValueError):
                logging.warning(
                    "[Embedding] Invalid embedding main GPU %r; using 0", raw_main_gpu
                )
                main_gpu = 0

        self.estimated_vram = _estimate_embedding_vram(
            model_path,
            self.context_length,
            self.n_parallel,
            self.ubatch_size,
            self.kv_cache_type,
        )
        self.estimated_vram_gb = float(self.estimated_vram["total_gb"])
        self.gpu_layers = self._resolve_gpu_layers(
            gpu_layers=gpu_layers,
            force_cpu=force_cpu,
            main_gpu=main_gpu,
        )
        self.estimated_gpu_vram_gb = self._estimate_gpu_vram_for_layers(self.gpu_layers)
        self.device = (
            "cuda" if self.gpu_layers != 0 and torch.cuda.is_available() else "cpu"
        )

        logging.info(
            "[Embedding] VRAM estimate for %s: %.2fGB total "
            "(weights=%.2fGB, kv=%.2fGB, compute=%.2fGB, cache=%s)",
            self.model_name,
            self.estimated_vram["total_gb"],
            self.estimated_vram["weights_gb"],
            self.estimated_vram["kv_gb"],
            self.estimated_vram["compute_gb"],
            self.kv_cache_type,
        )

        fallback_layers = self._fallback_gpu_layers(self.gpu_layers)
        last_error = None
        for attempt_layers in fallback_layers:
            params = self._build_params(model_path, main_gpu, attempt_layers)
            self.gpu_layers = attempt_layers
            self.estimated_gpu_vram_gb = self._estimate_gpu_vram_for_layers(
                attempt_layers
            )
            self.device = (
                "cuda" if self.gpu_layers != 0 and torch.cuda.is_available() else "cpu"
            )
            try:
                logging.info(
                    "[Embedding] Loading %s (%s, ctx=%s/slot, slots=%s, "
                    "total_ctx=%s, batch=%s, ubatch=%s, gpu_layers=%s, device=%s)",
                    self.model_name,
                    self.quant_type,
                    self.context_length,
                    self.n_parallel,
                    params.n_ctx,
                    self.batch_size,
                    self.ubatch_size,
                    self.gpu_layers,
                    self.device,
                )
                self.server = xlc.Server(params)
                return
            except Exception as e:
                last_error = e
                if attempt_layers == 0 or not _is_memory_error(e):
                    break
                logging.warning(
                    "[Embedding] Load failed with gpu_layers=%s: %s; retrying lower",
                    attempt_layers,
                    e,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        raise RuntimeError(
            f"Failed to initialize embedding model {self.model_name}: {last_error}"
        )

    def _resolve_gpu_layers(
        self,
        gpu_layers: int = None,
        force_cpu: bool = False,
        main_gpu: int = 0,
    ) -> int:
        if force_cpu or not torch.cuda.is_available():
            return 0
        if gpu_layers is not None:
            return gpu_layers

        free_gb = _get_cuda_free_gb(main_gpu)
        safety_gb = max(0.0, _get_float_env("EMBEDDING_VRAM_SAFETY_MARGIN", 0.75))
        total_gb = float(self.estimated_vram["total_gb"])
        if free_gb >= total_gb + safety_gb:
            logging.info(
                "[Embedding] GPU %s has %.2fGB free; full offload fits "
                "(need %.2fGB + %.2fGB margin)",
                main_gpu,
                free_gb,
                total_gb,
                safety_gb,
            )
            return -1

        budget_gb = free_gb - safety_gb - float(self.estimated_vram["fixed_gb"])
        per_layer_gb = max(0.001, float(self.estimated_vram["per_layer_gb"]))
        layers = int(self.estimated_vram["layers"])
        selected_layers = max(0, min(layers, int(budget_gb / per_layer_gb)))

        if selected_layers <= 0:
            logging.warning(
                "[Embedding] GPU %s has %.2fGB free, below estimated fixed "
                "embedding overhead %.2fGB + %.2fGB margin; using CPU",
                main_gpu,
                free_gb,
                self.estimated_vram["fixed_gb"],
                safety_gb,
            )
            return 0

        if selected_layers >= layers:
            return -1

        logging.info(
            "[Embedding] GPU %s has %.2fGB free; using partial offload "
            "(%s/%s layers on GPU, %.2fGB safety margin)",
            main_gpu,
            free_gb,
            selected_layers,
            layers,
            safety_gb,
        )
        return selected_layers

    def _fallback_gpu_layers(self, gpu_layers: int) -> list:
        if gpu_layers == 0:
            return [0]
        layers = int(self.estimated_vram["layers"])
        start = layers if gpu_layers < 0 else min(gpu_layers, layers)
        candidates = [gpu_layers]
        for fraction in (0.75, 0.5, 0.25):
            candidate = max(1, int(start * fraction))
            if candidate not in candidates:
                candidates.append(candidate)
        if 0 not in candidates:
            candidates.append(0)
        return candidates

    def _estimate_gpu_vram_for_layers(self, gpu_layers: int) -> float:
        if gpu_layers == 0:
            return 0.0
        layers = int(self.estimated_vram["layers"])
        selected_layers = layers if gpu_layers < 0 else min(gpu_layers, layers)
        return float(self.estimated_vram["fixed_gb"]) + (
            float(self.estimated_vram["per_layer_gb"]) * selected_layers
        )

    def _build_params(self, model_path: str, main_gpu: int, gpu_layers: int):
        params = xlc.CommonParams()
        params.model.path = model_path
        params.embedding = True
        params.n_ctx = self.context_length * self.n_parallel
        params.n_batch = self.batch_size
        params.n_ubatch = self.ubatch_size
        params.n_parallel = self.n_parallel
        params.n_gpu_layers = gpu_layers
        params.main_gpu = main_gpu
        params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_LAST
        params.cache_ram_mib = 0
        params.cache_prompt = False

        try:
            params.fit_params = False
        except Exception:
            pass

        kv_type_map = {
            "f16": getattr(xlc.ggml_type, "GGML_TYPE_F16", None),
            "f32": getattr(xlc.ggml_type, "GGML_TYPE_F32", None),
            "q8_0": getattr(xlc.ggml_type, "GGML_TYPE_Q8_0", None),
            "q4_0": getattr(xlc.ggml_type, "GGML_TYPE_Q4_0", None),
        }
        kv_type = kv_type_map.get(self.kv_cache_type)
        if kv_type is not None:
            try:
                params.cache_type_k = kv_type
                params.cache_type_v = kv_type
            except Exception:
                pass

        try:
            params.embd_normalize = int(getenv("EMBEDDING_NORMALIZE", "2"))
        except (TypeError, ValueError):
            params.embd_normalize = 2

        try:
            params.flash_attn_type = (
                xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED
            )
        except Exception:
            pass
        return params

    def _resolve_dimensions(self, dimensions):
        configured = getenv("EMBEDDING_DIMENSIONS", "")
        if dimensions is None and configured:
            dimensions = configured
        if dimensions in (None, ""):
            return None
        try:
            dimensions = int(dimensions)
        except (TypeError, ValueError):
            raise ValueError("dimensions must be an integer")
        if dimensions < 32 or dimensions > 1024:
            raise ValueError("Qwen3-Embedding-0.6B dimensions must be 32..1024")
        return dimensions

    def get_embeddings(self, input, model: str = None, dimensions=None):
        request_model = model or self.model_alias
        result = self.server.handle_embeddings(
            {
                "input": input,
                "model": request_model,
            }
        )

        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])

        resolved_dimensions = self._resolve_dimensions(dimensions)
        if resolved_dimensions:
            for item in result.get("data", []):
                embedding = item.get("embedding")
                if embedding and len(embedding) > resolved_dimensions:
                    item["embedding"] = _normalize(embedding[:resolved_dimensions])

        if isinstance(result, dict):
            result["model"] = request_model

        return result
