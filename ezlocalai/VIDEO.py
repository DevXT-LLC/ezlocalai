import logging
import uuid
import torch
import gc
import os

# Enable expandable segments to reduce CUDA memory fragmentation.
# This is critical for sequential CPU offload where layers are repeatedly
# moved to/from GPU, causing fragmentation over 40+ inference steps.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# LTX-2.3 requires diffusers with LTX2Pipeline and GGUF support
import_success = False

try:
    from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import (
        LTX2ImageToVideoPipeline,
    )
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import (
        LTX2ConditionPipeline,
        LTX2VideoCondition,
    )
    from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

    # Patch GGUFParameter to work with accelerate's cpu_offload.
    # accelerate moves parameters to meta device which loses quant_type,
    # causing KeyError(None) in GGML_QUANT_SIZES lookup.
    from diffusers.quantizers.gguf.utils import GGUFParameter

    _original_gguf_new = GGUFParameter.__new__

    def _patched_gguf_new(cls, data, requires_grad=False, quant_type=None):
        if quant_type is None:
            return torch.nn.Parameter.__new__(cls, data, requires_grad=requires_grad)
        return _original_gguf_new(
            cls, data, requires_grad=requires_grad, quant_type=quant_type
        )

    GGUFParameter.__new__ = _patched_gguf_new

    import_success = True
except (ImportError, RuntimeError, Exception) as e:
    logging.warning(
        f"[VIDEO] LTX2Pipeline/GGUF not available ({e}). Video generation will be unavailable. "
        "Install diffusers from source: pip install git+https://github.com/huggingface/diffusers"
    )


class _CPUEncoderBridge(torch.nn.Module):
    """Wraps a CPU-resident quantized text encoder for a GPU-offloaded pipeline.

    ``enable_sequential_cpu_offload`` sets the pipeline's execution device to
    GPU, so input_ids arrive on cuda.  This bridge transparently moves them to
    CPU for the real encoder, then moves output hidden_states back to GPU.
    The real encoder is stored outside ``nn.Module`` registration so the
    pipeline's offload hooks never inspect (or break) its quantized weights.
    """

    def __init__(self, encoder, target_device):
        super().__init__()
        # Store encoder outside nn.Module submodule registry
        object.__setattr__(self, "_cpu_encoder", encoder)
        object.__setattr__(self, "_target_device", target_device)

    # Forward proxy attributes the pipeline may inspect.
    @property
    def config(self):
        return self._cpu_encoder.config

    @property
    def dtype(self):
        return self._cpu_encoder.dtype

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            input_ids = input_ids.cpu()
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()
        out = self._cpu_encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        # Move hidden_states to GPU so the transformer can consume them.
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            out["hidden_states"] = tuple(
                h.to(self._target_device) for h in out.hidden_states
            )
        return out


# GGUF quant files available in unsloth/LTX-2.3-GGUF
GGUF_QUANT_FILES = {
    "Q4_K_M": "ltx-2.3-22b-dev-Q4_K_M.gguf",
    "Q8_0": "ltx-2.3-22b-dev-Q8_0.gguf",
}
DEFAULT_GGUF_QUANT = "Q4_K_M"
# Full-precision pipeline config repo (text_encoder, vae, scheduler, etc.)
LTX2_CONFIG_REPO = "Lightricks/LTX-2"
# Unsloth repo for GGUF + matching connector text projections
UNSLOTH_REPO = "unsloth/LTX-2.3-GGUF"
UNSLOTH_CONNECTOR_FILE = (
    "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors"
)


class VIDEO:
    """Video generation using LTX-2.3 with GGUF-quantized transformer.

    Supports multiple generation modes:
      - Text-to-video (+ audio): prompt only
      - Image-to-video (+ audio): image + optional prompt
      - Video-to-video (+ audio): conditioning frames at arbitrary indices + optional prompt

    All modes produce both video frames and synchronized audio.

    Uses a Q4_K_M GGUF transformer from unsloth/LTX-2.3-GGUF for efficient
    inference, combined with the full pipeline components from Lightricks/LTX-2.
    Sequential CPU offload keeps peak VRAM usage low.

    Model: https://huggingface.co/unsloth/LTX-2.3-GGUF
    """

    AUDIO_SAMPLE_RATE = 24000

    def __init__(
        self,
        model="unsloth/LTX-2.3-GGUF",
        device="cpu",
        local_uri=None,
    ):
        global import_success
        self.local_uri = local_uri
        self.device = device
        self.pipe = None  # LTX2Pipeline (text-to-video)
        self.pipe_i2v = None  # LTX2ImageToVideoPipeline
        self.pipe_cond = None  # LTX2ConditionPipeline
        self.dtype = None

        if not import_success:
            return

        self._load_pipeline(model, device)

    def _load_pipeline(self, model: str, device: str):
        """Load LTX-2.3 pipeline with GGUF-quantized transformer."""
        try:
            from huggingface_hub import hf_hub_download

            logging.info(f"[VIDEO] Loading LTX-2.3 GGUF ({model}) on {device}...")

            # Parse GPU index from device string (e.g. "cuda:1" -> 1)
            gpu_idx = 0
            is_cuda = device.startswith("cuda")
            if ":" in device:
                try:
                    gpu_idx = int(device.split(":")[1])
                except (ValueError, IndexError):
                    pass

            # Determine dtype
            if device == "cpu":
                dtype = torch.bfloat16
            elif (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(gpu_idx)[0] >= 8
            ):
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            self.dtype = dtype

            # Download the GGUF transformer file
            gguf_filename = GGUF_QUANT_FILES[DEFAULT_GGUF_QUANT]
            logging.info(
                f"[VIDEO] Downloading GGUF transformer: {model}/{gguf_filename}"
            )
            gguf_path = hf_hub_download(
                model, filename=gguf_filename, cache_dir="models"
            )

            # Load GGUF-quantized transformer
            # LTX-2.3 has architectural differences from LTX-2.0:
            #  - cross_attn_mod/audio_cross_attn_mod: 9 mod params vs 6
            #  - gated_attn/audio_gated_attn: gate logits in attention layers
            #  - use_prompt_embeddings=False: uses connectors instead of caption_projection
            #  - rope_type="interleaved" instead of "split"
            # We must override the config from Lightricks/LTX-2 (which is LTX-2.0)
            logging.info("[VIDEO] Loading GGUF transformer...")
            config_path = hf_hub_download(
                LTX2_CONFIG_REPO,
                "transformer/config.json",
                cache_dir="models",
            )
            import json
            import os

            with open(config_path) as f:
                transformer_config = json.load(f)

            # LTX-2.3 config overrides
            transformer_config["cross_attn_mod"] = True
            transformer_config["audio_cross_attn_mod"] = True
            transformer_config["gated_attn"] = True
            transformer_config["audio_gated_attn"] = True
            transformer_config["use_prompt_embeddings"] = False
            transformer_config["rope_type"] = "interleaved"

            # Write modified config to a local directory (from_single_file needs a path)
            config_dir = os.path.join("models", "_ltx23_transformer_config")
            os.makedirs(config_dir, exist_ok=True)
            with open(os.path.join(config_dir, "config.json"), "w") as f:
                json.dump(transformer_config, f)

            # Patch the key conversion function to handle LTX-2.3 specific mappings
            self._patch_ltx2_conversion()

            transformer = LTX2VideoTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
                config=config_dir,
            )

            # Load LTX-2.3 connectors from GGUF + unsloth text projections.
            # The Lightricks/LTX-2 HF repo has LTX-2.0 connectors (3840-dim)
            # but LTX-2.3 uses 4096-dim video / 2048-dim audio connectors.
            # The GGUF contains the connector attention blocks (removed by the
            # diffusers conversion fn). Unsloth provides matching text projections.
            logging.info("[VIDEO] Loading LTX-2.3 connectors from GGUF...")
            connectors = self._load_connectors_from_gguf(gguf_path, dtype)

            # Load text encoder (Gemma 3 12B).
            # Strategy hierarchy:
            #  1. GPU BNB 4-bit (~3GB VRAM) when >=12GB free VRAM
            #  2. CPU quanto INT8 (~12GB RAM) — fast, uses VNNI/AVX-512
            #  3. CPU bfloat16 (~24GB RAM) — fallback if quanto unavailable
            from transformers import Gemma3ForConditionalGeneration

            text_encoder = None
            if is_cuda:
                free_mem, _ = torch.cuda.mem_get_info(gpu_idx)
                free_gb = free_mem / (1024**3)
                logging.info(f"[VIDEO] GPU {gpu_idx} has {free_gb:.1f}GB free VRAM")
                if free_gb >= 12.0:
                    try:
                        from transformers import BitsAndBytesConfig

                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        logging.info(
                            "[VIDEO] Loading text encoder with bitsandbytes 4-bit (~3GB)..."
                        )
                        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                            LTX2_CONFIG_REPO,
                            subfolder="text_encoder",
                            quantization_config=bnb_config,
                            torch_dtype=dtype,
                            device_map={"": f"cuda:{gpu_idx}"},
                            cache_dir="models",
                        )
                        logging.info(
                            "[VIDEO] Text encoder loaded with 4-bit quantization"
                        )
                    except Exception as e:
                        logging.warning(
                            f"[VIDEO] bitsandbytes 4-bit failed ({e}), "
                            "falling back to CPU text encoder"
                        )
                        text_encoder = None
                else:
                    logging.info(
                        f"[VIDEO] Only {free_gb:.1f}GB free VRAM, skipping "
                        "bitsandbytes (need 12GB). Using CPU text encoder."
                    )

            # CPU path: try quanto INT8 first (~12GB), fall back to bf16 (~24GB)
            if text_encoder is None:
                try:
                    from transformers import QuantoConfig

                    quanto_config = QuantoConfig(weights="int8")
                    logging.info(
                        "[VIDEO] Loading text encoder on CPU with "
                        "quanto INT8 (~12GB)..."
                    )
                    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                        LTX2_CONFIG_REPO,
                        subfolder="text_encoder",
                        quantization_config=quanto_config,
                        torch_dtype=dtype,
                        cache_dir="models",
                    )
                    logging.info("[VIDEO] Text encoder loaded with quanto INT8")
                except Exception as e:
                    logging.warning(
                        f"[VIDEO] quanto INT8 failed ({e}), "
                        "falling back to bf16 on CPU"
                    )
                    text_encoder = None

            if text_encoder is None:
                logging.info("[VIDEO] Loading text encoder on CPU (bf16, ~24GB)...")
                text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                    LTX2_CONFIG_REPO,
                    subfolder="text_encoder",
                    torch_dtype=dtype,
                    cache_dir="models",
                )

            # Track whether text encoder uses BNB 4-bit (GPU-resident) or
            # quanto INT8 (CPU-resident).  Both use custom tensor types that
            # cannot survive the meta-device round-trip from sequential offload.
            text_encoder_is_bnb = getattr(
                text_encoder, "is_loaded_in_4bit", False
            ) or getattr(text_encoder, "is_loaded_in_8bit", False)
            text_encoder_is_quanto = not text_encoder_is_bnb and getattr(
                text_encoder, "is_quantized", False
            )

            # Load the rest of the pipeline from the config repo.
            logging.info(
                f"[VIDEO] Loading pipeline components from {LTX2_CONFIG_REPO}..."
            )
            self.pipe = LTX2Pipeline.from_pretrained(
                LTX2_CONFIG_REPO,
                transformer=transformer,
                connectors=connectors,
                text_encoder=text_encoder,
                torch_dtype=dtype,
                cache_dir="models",
                ignore_patterns=[
                    "transformer/diffusion_pytorch_model*",
                    "text_encoder/diffusion_pytorch_model*",
                    "text_encoder/model-*",
                    "connectors/diffusion_pytorch_model*",
                    "latent_upsampler/*",
                    "*.mp4",
                    "ltx-2-*.safetensors",
                ],
            )

            if is_cuda:
                # Choose offload strategy based on total GPU capacity:
                #  - >=40GB total: model CPU offload (moves whole modules, fast)
                #  - <40GB total: sequential CPU offload (moves individual layers,
                #    ~300MB each, slower but fits in limited VRAM)
                # The Q4_K_M GGUF transformer is ~14GB. Model CPU offload loads
                # the entire module onto GPU during forward pass, so you need
                # the 14GB model + activation memory + text encoder + VAE.
                # 24GB GPUs OOM with model offload; 40GB+ (A100, A6000) can use it.
                _, total_mem = torch.cuda.mem_get_info(gpu_idx)
                total_gb = total_mem / (1024**3)
                use_model_offload = total_gb >= 40.0

                if text_encoder_is_bnb or text_encoder_is_quanto:
                    # Both BNB and quanto use custom tensor types that cannot
                    # survive the meta-device round-trip that sequential offload
                    # uses.  Swap out the text encoder so it isn't touched.
                    saved_te = self.pipe.text_encoder
                    self.pipe.text_encoder = None
                try:
                    if use_model_offload:
                        self.pipe.enable_model_cpu_offload(gpu_id=gpu_idx)
                        logging.info(
                            f"[VIDEO] Model CPU offload enabled on GPU {gpu_idx} "
                            f"({total_gb:.1f}GB total - whole-module transfers)"
                        )
                    else:
                        self.pipe.enable_sequential_cpu_offload(gpu_id=gpu_idx)
                        logging.info(
                            f"[VIDEO] Sequential CPU offload enabled on GPU {gpu_idx} "
                            f"({total_gb:.1f}GB total - per-layer transfers)"
                        )
                except Exception as e:
                    offload_type = "Model" if use_model_offload else "Sequential"
                    logging.warning(
                        f"[VIDEO] {offload_type} CPU offload failed ({e}), "
                        "falling back to CPU"
                    )
                    import traceback

                    traceback.print_exc()
                    for name, component in self.pipe.components.items():
                        if isinstance(component, torch.nn.Module):
                            try:
                                component.to("cpu")
                            except Exception:
                                pass

                if text_encoder_is_bnb:
                    # Restore the BNB text encoder (stays on GPU at ~3GB)
                    self.pipe.text_encoder = saved_te
                    logging.info(
                        f"[VIDEO] BNB text encoder kept on cuda:{gpu_idx} (~3GB)"
                    )
                elif text_encoder_is_quanto:
                    # Wrap quanto encoder in a bridge that moves tensors
                    # between CPU (where quanto lives) and GPU (where the
                    # pipeline's execution device is set by offload hooks).
                    self.pipe.text_encoder = _CPUEncoderBridge(
                        saved_te, torch.device(f"cuda:{gpu_idx}")
                    )
                    logging.info(
                        "[VIDEO] Quanto INT8 text encoder bridged "
                        f"(CPU → cuda:{gpu_idx})"
                    )
            else:
                for name, component in self.pipe.components.items():
                    if isinstance(component, torch.nn.Module):
                        component.to("cpu")

            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass

            try:
                self.pipe.enable_vae_tiling()
                logging.info("[VIDEO] VAE spatial tiling enabled (reduces decode VRAM)")
            except Exception:
                pass

            # Enable framewise (temporal) decoding — decodes frames in chunks
            # instead of all 121 at once.  Without this, VAE decode tries to
            # allocate ~19GB for the full frame tensor, causing OOM on 24GB GPUs.
            try:
                self.pipe.vae.use_framewise_decoding = True
                self.pipe.vae.use_framewise_encoding = True
                logging.info("[VIDEO] VAE framewise decoding enabled (temporal tiling)")
            except Exception:
                pass

            # Build image-to-video and condition pipelines from the same components.
            # They share all weights (no extra memory), just different __call__ logic.
            components = self.pipe.components
            self.pipe_i2v = LTX2ImageToVideoPipeline(**components)
            self.pipe_cond = LTX2ConditionPipeline(
                **{k: v for k, v in components.items() if k != "processor"}
            )
            # Copy over accelerate hooks so offloading works on all pipelines
            if hasattr(self.pipe, "_all_hooks") and self.pipe._all_hooks:
                self.pipe_i2v._all_hooks = self.pipe._all_hooks
                self.pipe_cond._all_hooks = self.pipe._all_hooks

            logging.info(
                f"[VIDEO] LTX-2.3 GGUF loaded successfully on {device} with dtype {dtype}"
            )

        except Exception as e:
            logging.error(f"[VIDEO] Failed to load LTX-2.3: {e}")
            import traceback

            traceback.print_exc()
            self.pipe = None

    def _load_connectors_from_gguf(self, gguf_path: str, dtype):
        """Build LTX-2.3 connectors from GGUF attention blocks + unsloth text projections.

        The Lightricks/LTX-2 HF repo has LTX-2.0 connectors (3840-dim) but the
        LTX-2.3 GGUF model uses 4096-dim video / 2048-dim audio connectors with
        per-modality projections. This method:
          1. Extracts connector attention block weights from the GGUF
          2. Downloads text projection weights from unsloth
          3. Creates a properly configured LTX2TextConnectors
          4. Loads the combined state dict
        """
        import gguf as gguf_lib
        from gguf import GGUFReader
        from huggingface_hub import hf_hub_download
        from diffusers.quantizers.gguf.utils import (
            GGUFParameter,
            SUPPORTED_GGUF_QUANT_TYPES,
            dequantize_gguf_tensor,
        )
        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2TextConnectors
        import safetensors.torch

        # Step 1: Extract and dequantize connector tensors from GGUF
        logging.info("[VIDEO] Extracting connector weights from GGUF...")
        reader = GGUFReader(gguf_path)
        gguf_connector_tensors = {}
        # Track which tensors were dequantized from GGUF-order (need transpose)
        needs_transpose = set()
        for tensor in reader.tensors:
            name = tensor.name
            if not (
                name.startswith("video_embeddings_connector.")
                or name.startswith("audio_embeddings_connector.")
            ):
                continue
            quant_type = tensor.tensor_type
            is_quantized = quant_type not in [
                gguf_lib.GGMLQuantizationType.F32,
                gguf_lib.GGMLQuantizationType.F16,
                gguf_lib.GGMLQuantizationType.BF16,
            ]
            weights = torch.from_numpy(tensor.data.copy())
            if is_quantized:
                # Quantized: dequantize returns flat data, reshape to GGUF metadata shape
                # GGUF metadata stores shapes as [in, out], needs transpose to PyTorch [out, in]
                stored_shape = tuple(int(s) for s in tensor.shape)
                param = GGUFParameter(weights, quant_type=quant_type)
                dequantized = dequantize_gguf_tensor(param)
                dequantized = dequantized.reshape(stored_shape)
                gguf_connector_tensors[name] = dequantized.to(dtype)
                if len(stored_shape) == 2:
                    needs_transpose.add(name)
            else:
                # Non-quantized: gguf library already provides data in PyTorch shape order
                gguf_connector_tensors[name] = weights.to(dtype)
        del reader

        logging.info(
            f"[VIDEO] Extracted {len(gguf_connector_tensors)} connector tensors from GGUF"
        )

        # Step 2: Remap GGUF keys to diffusers LTX2TextConnectors format
        # GGUF stores weights as [in_features, out_features], PyTorch needs [out_features, in_features]
        connector_state_dict = {}
        for gguf_key, tensor in gguf_connector_tensors.items():
            new_key = gguf_key
            new_key = new_key.replace("video_embeddings_connector.", "video_connector.")
            new_key = new_key.replace("audio_embeddings_connector.", "audio_connector.")
            new_key = new_key.replace("transformer_1d_blocks.", "transformer_blocks.")
            new_key = new_key.replace(".k_norm.", ".norm_k.")
            new_key = new_key.replace(".q_norm.", ".norm_q.")

            # Transpose 2D quantized tensors (GGUF metadata [in, out] -> PyTorch [out, in])
            # Non-quantized tensors are already in PyTorch shape order from the gguf library
            if gguf_key in needs_transpose:
                tensor = tensor.T.contiguous()

            connector_state_dict[new_key] = tensor

        del gguf_connector_tensors

        # Step 3: Download and remap text projection weights from unsloth
        logging.info("[VIDEO] Downloading connector text projections from unsloth...")
        proj_path = hf_hub_download(
            UNSLOTH_REPO, filename=UNSLOTH_CONNECTOR_FILE, cache_dir="models"
        )
        proj_tensors = safetensors.torch.load_file(proj_path)
        proj_key_map = {
            "text_embedding_projection.video_aggregate_embed.weight": "video_text_proj_in.weight",
            "text_embedding_projection.video_aggregate_embed.bias": "video_text_proj_in.bias",
            "text_embedding_projection.audio_aggregate_embed.weight": "audio_text_proj_in.weight",
            "text_embedding_projection.audio_aggregate_embed.bias": "audio_text_proj_in.bias",
        }
        for old_key, new_key in proj_key_map.items():
            if old_key in proj_tensors:
                connector_state_dict[new_key] = proj_tensors[old_key].to(dtype)
        del proj_tensors

        # Step 4: Create LTX2TextConnectors with correct LTX-2.3 config
        # LTX-2.3 uses per-modality projections with 4096-dim video / 2048-dim audio
        logging.info("[VIDEO] Creating LTX-2.3 connectors model...")
        connectors = LTX2TextConnectors(
            caption_channels=3840,
            text_proj_in_factor=49,
            video_connector_num_attention_heads=32,
            video_connector_attention_head_dim=128,
            video_connector_num_layers=8,
            video_connector_num_learnable_registers=128,
            video_gated_attn=True,
            audio_connector_num_attention_heads=32,
            audio_connector_attention_head_dim=64,
            audio_connector_num_layers=8,
            audio_connector_num_learnable_registers=128,
            audio_gated_attn=True,
            connector_rope_base_seq_len=4096,
            rope_theta=10000.0,
            rope_double_precision=True,
            causal_temporal_positioning=False,
            rope_type="interleaved",
            per_modality_projections=True,
            video_hidden_dim=4096,
            audio_hidden_dim=2048,
            proj_bias=True,
        )

        # Step 5: Load the combined state dict
        missing, unexpected = connectors.load_state_dict(
            connector_state_dict, strict=False
        )
        if missing:
            logging.warning(f"[VIDEO] Connector missing keys: {missing[:5]}...")
        if unexpected:
            logging.warning(f"[VIDEO] Connector unexpected keys: {unexpected[:5]}...")
        logging.info("[VIDEO] LTX-2.3 connectors loaded successfully")

        del connector_state_dict
        gc.collect()

        # Ensure all weights are in the correct dtype
        connectors = connectors.to(dtype)

        return connectors

    @staticmethod
    def _patch_ltx2_conversion():
        """Patch the diffusers LTX2 key conversion to handle LTX-2.3 weight names.

        The built-in convert_ltx2_transformer_to_diffusers doesn't remap:
          - prompt_adaln_single -> prompt_adaln
          - audio_prompt_adaln_single -> audio_prompt_adaln
        This causes those weights to be skipped, leaving meta tensors.
        """
        from diffusers.loaders import single_file_utils

        original_fn = single_file_utils.convert_ltx2_transformer_to_diffusers

        def patched_convert(checkpoint, **kwargs):
            converted = original_fn(checkpoint, **kwargs)
            # Remap LTX-2.3 prompt adaln keys
            for key in list(converted.keys()):
                if key.startswith("prompt_adaln_single."):
                    new_key = key.replace("prompt_adaln_single.", "prompt_adaln.", 1)
                    converted[new_key] = converted.pop(key)
                elif key.startswith("audio_prompt_adaln_single."):
                    new_key = key.replace(
                        "audio_prompt_adaln_single.", "audio_prompt_adaln.", 1
                    )
                    converted[new_key] = converted.pop(key)
            return converted

        single_file_utils.convert_ltx2_transformer_to_diffusers = patched_convert
        # Also patch the reference in the mapping dict
        from diffusers.loaders import single_file_model

        for cls_name, cfg in single_file_model.SINGLE_FILE_LOADABLE_CLASSES.items():
            if cfg.get("checkpoint_mapping_fn") is original_fn:
                cfg["checkpoint_mapping_fn"] = patched_convert

    def generate(
        self,
        prompt,
        negative_prompt="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, blurry, shaky",
        num_inference_steps=40,
        guidance_scale=4.0,
        num_frames=121,
        frame_rate=24,
        size="768x512",
        image=None,
        conditions=None,
    ):
        """Generate a video, optionally conditioned on image/video frames.

        Args:
            image: PIL Image for image-to-video mode. First frame stays fixed.
            conditions: List of dicts with keys 'image' (PIL or path), 'index' (int),
                        'strength' (float 0-1) for multi-frame conditioning (video-to-video).

        Returns:
            Path to saved video file (MP4 with audio) or None on failure
        """
        new_file_name = f"outputs/{uuid.uuid4()}.mp4"

        if not self.pipe:
            return None

        # Parse size
        width, height = map(int, size.split("x"))

        # Ensure dimensions are divisible by 32 (LTX requirement)
        width = (width // 32) * 32
        height = (height // 32) * 32

        # Ensure num_frames follows 8n+1 pattern
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
            logging.debug(f"[VIDEO] Adjusted num_frames to {num_frames} (must be 8n+1)")

        # Clamp dimensions to reasonable maximums
        width = min(width, 1280)
        height = min(height, 1280)

        # Minimum 2 inference steps required (1 step causes NaN in scheduler
        # when shift_terminal is configured with the flow matching scheduler)
        num_inference_steps = max(num_inference_steps, 2)

        # Determine generation mode
        if conditions and self.pipe_cond:
            mode = "condition"
        elif image is not None and self.pipe_i2v:
            mode = "i2v"
        else:
            mode = "t2v"

        try:
            result = self._run_pipeline(
                mode=mode,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                frame_rate=frame_rate,
                image=image,
                conditions=conditions,
            )

            if result is None:
                return None

            self._export_video(result, new_file_name, frame_rate)

            if self.local_uri:
                return f"{self.local_uri}/{new_file_name}"
            return new_file_name

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logging.warning(f"[VIDEO] GPU OOM during generation: {e}")
                return self._generate_cpu_fallback(
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    num_frames,
                    frame_rate,
                    new_file_name,
                )
            raise

    def _run_pipeline(
        self,
        mode,
        prompt,
        negative_prompt,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_frames,
        frame_rate,
        image=None,
        conditions=None,
    ):
        """Run the appropriate LTX-2.3 pipeline based on mode."""
        generator = torch.Generator(device="cpu").manual_seed(42)

        common_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        if mode == "i2v":
            logging.info("[VIDEO] Running image-to-video pipeline")
            return self.pipe_i2v(image=image, **common_kwargs)
        elif mode == "condition":
            logging.info("[VIDEO] Running condition pipeline (video-to-video)")
            ltx_conditions = self._build_conditions(conditions)
            return self.pipe_cond(conditions=ltx_conditions, **common_kwargs)
        else:
            logging.info("[VIDEO] Running text-to-video pipeline")
            return self.pipe(**common_kwargs)

    @staticmethod
    def _build_conditions(conditions):
        """Convert condition dicts to LTX2VideoCondition objects."""
        from PIL import Image

        ltx_conditions = []
        for cond in conditions:
            frames = cond.get("image") or cond.get("frames")
            if isinstance(frames, str):
                frames = Image.open(frames).convert("RGB")
            ltx_conditions.append(
                LTX2VideoCondition(
                    frames=frames,
                    index=cond.get("index", 0),
                    strength=cond.get("strength", 1.0),
                )
            )
        return ltx_conditions

    def _export_video(self, result, output_path, fps):
        """Export pipeline result to a video file, with audio if available."""
        has_audio = hasattr(result, "audio") and result.audio is not None
        frames = result.frames[0]

        if has_audio:
            # Use PyAV to mux video + audio into a single MP4
            try:
                from diffusers.pipelines.ltx2.export_utils import encode_video

                encode_video(
                    video=frames,
                    fps=fps,
                    audio=result.audio,
                    audio_sample_rate=self.AUDIO_SAMPLE_RATE,
                    output_path=output_path,
                )
                return
            except Exception as e:
                logging.warning(
                    f"[VIDEO] encode_video (audio+video) failed: {e}, "
                    "falling back to video-only export"
                )

        # Video-only export
        try:
            from diffusers.utils import export_to_video

            export_to_video(frames, output_path, fps=fps)
            return
        except (ImportError, AttributeError, Exception) as e:
            logging.debug(f"[VIDEO] export_to_video not available: {e}")

        # Fallback: use OpenCV to write video
        try:
            import cv2
            import numpy as np

            if not frames:
                return

            first_frame = np.array(frames[0])
            h, w = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for frame in frames:
                frame_np = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()

        except Exception as e:
            logging.error(f"[VIDEO] Failed to export video: {e}")

    def _generate_cpu_fallback(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_frames,
        frame_rate,
        output_file,
    ):
        """Attempt generation at reduced resolution after GPU OOM.

        Only retries at lower resolution on GPU.  Does NOT attempt full CPU
        fallback because BNB 4-bit text encoder tensors are GPU-only and
        moving them to CPU produces corrupted/garbage output.
        """
        logging.warning("[VIDEO] Attempting reduced-resolution GPU retry...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reduce resolution and frame count to fit in VRAM
        reduced_width = min(width, 512)
        reduced_height = min(height, 320)
        reduced_frames = min(num_frames, 41)  # 5*8+1
        if (reduced_frames - 1) % 8 != 0:
            reduced_frames = ((reduced_frames - 1) // 8) * 8 + 1

        try:
            logging.info(
                f"[VIDEO] Retrying at {reduced_width}x{reduced_height}, "
                f"{reduced_frames} frames (down from {width}x{height}, {num_frames})"
            )
            generator = torch.Generator(device="cpu").manual_seed(42)

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=reduced_width,
                height=reduced_height,
                num_frames=reduced_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            self._export_video(result, output_file, frame_rate)

            if self.local_uri:
                return f"{self.local_uri}/{output_file}"
            return output_file

        except Exception as retry_error:
            logging.error(
                f"[VIDEO] Reduced resolution also failed: {retry_error}. "
                "Cannot fall back to CPU (BNB 4-bit text encoder is GPU-only)."
            )
            return None
