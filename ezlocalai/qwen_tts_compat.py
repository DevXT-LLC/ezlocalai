"""Compatibility shims for qwen-tts with newer Transformers releases."""

from __future__ import annotations

import logging


_APPLIED = False


def _transformers_major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def apply_qwen_tts_transformers_compat() -> None:
    """Patch qwen-tts 0.1.x assumptions that changed in Transformers 5.x."""
    global _APPLIED
    if _APPLIED:
        return

    try:
        import torch
        import transformers
    except Exception:
        return

    if _transformers_major(getattr(transformers, "__version__", "")) < 5:
        _APPLIED = True
        return

    _patch_check_model_inputs()
    _patch_default_rope(torch)

    try:
        _patch_qwen_modules(torch)
    except ImportError:
        raise
    except Exception as exc:
        logging.warning("[QTTS] Failed to apply Transformers 5 compatibility: %s", exc)
        raise

    _APPLIED = True


def repair_qwen_tts_rotary_buffers(model) -> int:
    """Recompute Qwen-TTS RoPE buffers that can be corrupted by weight loading."""
    try:
        import torch
    except Exception:
        return 0

    module = getattr(model, "model", model)
    if not hasattr(module, "named_modules"):
        return 0

    repaired = 0
    for name, rotary in module.named_modules():
        if not (
            hasattr(rotary, "config")
            and hasattr(rotary, "inv_freq")
            and hasattr(rotary, "rope_init_fn")
        ):
            continue
        try:
            current = rotary.inv_freq
            inv_freq, attention_scaling = rotary.rope_init_fn(
                config=rotary.config,
                device=current.device,
            )
            inv_freq = inv_freq.to(device=current.device, dtype=current.dtype)
            if not torch.isfinite(inv_freq).all():
                logging.warning("[QTTS] Skipping non-finite RoPE repair for %s", name)
                continue
            rotary.register_buffer("inv_freq", inv_freq, persistent=False)
            rotary.original_inv_freq = rotary.inv_freq
            rotary.attention_scaling = attention_scaling
            repaired += 1
        except Exception as exc:
            logging.warning("[QTTS] Could not repair RoPE buffer %s: %s", name, exc)

    if repaired:
        logging.info("[QTTS] Recomputed %s Qwen-TTS RoPE buffer(s)", repaired)
    return repaired


def _patch_check_model_inputs() -> None:
    from transformers.utils import generic as tf_generic

    original = tf_generic.check_model_inputs
    if getattr(original, "_ezlocalai_qwen_compat", False):
        return

    def check_model_inputs_compat(func=None):
        if func is None:
            return original
        return original(func)

    check_model_inputs_compat._ezlocalai_qwen_compat = True
    tf_generic.check_model_inputs = check_model_inputs_compat


def _patch_default_rope(torch) -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def default_rope_parameters(
        config=None,
        device=None,
        seq_len=None,
        layer_type=None,
        **kwargs,
    ):
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters


def _patch_qwen_modules(torch) -> None:
    from transformers import masking_utils
    from transformers.utils import generic as tf_generic

    original_causal_mask = masking_utils.create_causal_mask
    original_sliding_mask = masking_utils.create_sliding_window_causal_mask
    original_log_level = tf_generic.logger.level
    masking_utils.create_causal_mask = _create_qwen_causal_mask
    masking_utils.create_sliding_window_causal_mask = _create_qwen_causal_mask
    tf_generic.logger.setLevel(logging.CRITICAL)
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
        import qwen_tts.core.models.modeling_qwen3_tts as qwen_modeling
        import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as qwen_tokenizer_modeling
    finally:
        masking_utils.create_causal_mask = original_causal_mask
        masking_utils.create_sliding_window_causal_mask = original_sliding_mask
        tf_generic.logger.setLevel(original_log_level)

    _patch_talker_config(Qwen3TTSTalkerConfig)
    qwen_modeling.create_causal_mask = _create_qwen_causal_mask
    qwen_modeling.create_sliding_window_causal_mask = _create_qwen_causal_mask
    qwen_tokenizer_modeling.create_causal_mask = _create_qwen_causal_mask
    _patch_multimodal_rope(qwen_modeling)


def _patch_talker_config(config_class) -> None:
    original = config_class.__init__
    if getattr(original, "_ezlocalai_qwen_compat", False):
        return

    def init_compat(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if getattr(self, "pad_token_id", None) is None:
            self.pad_token_id = getattr(self, "codec_pad_id", None)
        if getattr(self, "bos_token_id", None) is None:
            self.bos_token_id = getattr(self, "codec_bos_id", None)
        if getattr(self, "eos_token_id", None) is None:
            self.eos_token_id = getattr(self, "codec_eos_token_id", None)

    init_compat._ezlocalai_qwen_compat = True
    config_class.__init__ = init_compat


def _patch_multimodal_rope(qwen_modeling) -> None:
    original = qwen_modeling.apply_multimodal_rotary_pos_emb
    if getattr(original, "_ezlocalai_qwen_compat", False):
        return

    def rope_compat(q, k, cos, sin, *args, **kwargs):
        q_len = q.shape[-2]
        if cos.shape[-2] != q_len:
            cos = cos[..., -q_len:, :]
            sin = sin[..., -q_len:, :]
        return original(q, k, cos, sin, *args, **kwargs)

    rope_compat._ezlocalai_qwen_compat = True
    qwen_modeling.apply_multimodal_rotary_pos_emb = rope_compat


def _create_qwen_causal_mask(
    config,
    inputs_embeds=None,
    attention_mask=None,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
    input_embeds=None,
    **kwargs,
):
    import torch

    if inputs_embeds is None:
        inputs_embeds = input_embeds
    if inputs_embeds is None:
        return None
    if attention_mask is not None and getattr(attention_mask, "dim", lambda: 0)() == 4:
        return attention_mask

    batch_size, query_length = inputs_embeds.shape[:2]
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    past_seen = 0
    try:
        if past_key_values is not None:
            past_seen = int(past_key_values.get_seq_length())
    except Exception:
        past_seen = 0

    if cache_position is None:
        cache_position = torch.arange(
            past_seen,
            past_seen + query_length,
            device=device,
        )
    else:
        cache_position = cache_position.to(device)

    target_length = past_seen + query_length
    if cache_position.numel():
        target_length = max(target_length, int(cache_position.max().item()) + 1)
    if attention_mask is not None:
        target_length = max(target_length, int(attention_mask.shape[-1]))

    min_dtype = torch.finfo(dtype).min
    key_positions = torch.arange(target_length, device=device)
    allowed = key_positions.unsqueeze(0) <= cache_position.reshape(-1, 1)
    causal_mask = torch.full(
        (query_length, target_length),
        min_dtype,
        dtype=dtype,
        device=device,
    )
    causal_mask = causal_mask.masked_fill(allowed, 0)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1).clone()

    if attention_mask is not None:
        mask_length = min(attention_mask.shape[-1], target_length)
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
            :, None, None, :mask_length
        ].to(device)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)

    return causal_mask
