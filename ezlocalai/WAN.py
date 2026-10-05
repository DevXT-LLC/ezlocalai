"""Wan 2.2 A14B GGUF model assets for stable-diffusion.cpp."""

import os

T2V_REPO = "QuantStack/Wan2.2-T2V-A14B-GGUF"
I2V_REPO = "QuantStack/Wan2.2-I2V-A14B-GGUF"
MODEL_REVISIONS = {
    T2V_REPO: "73eafba53a1a8f29254e4c77f92e74ea27d7cd6f",
    I2V_REPO: "6c6717459277b9cd1f72579d78a0fd62a79e57dc",
    "city96/umt5-xxl-encoder-gguf": "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7",
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged": "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
}
QUANT_TYPES = {
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q4_0",
    "Q4_1",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_0",
    "Q5_1",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
}


def download_wan_models(image_to_video=False, quant=None):
    """Fetch one expert pair plus shared encoder/VAE into the normal HF cache."""
    from huggingface_hub import hf_hub_download

    quant = (quant or os.getenv("VIDEO_QUANT_TYPE", "Q4_K_M")).upper()
    if quant not in QUANT_TYPES:
        raise ValueError(f"Unsupported Wan quantization: {quant}")
    mode = "I2V" if image_to_video else "T2V"
    repo = I2V_REPO if image_to_video else T2V_REPO
    files = {
        "high": (repo, f"HighNoise/Wan2.2-{mode}-A14B-HighNoise-{quant}.gguf"),
        "low": (repo, f"LowNoise/Wan2.2-{mode}-A14B-LowNoise-{quant}.gguf"),
        "encoder": ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q8_0.gguf"),
        "vae": (
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "split_files/vae/wan_2.1_vae.safetensors",
        ),
    }
    return {
        name: hf_hub_download(
            model_repo,
            filename=filename,
            revision=MODEL_REVISIONS[model_repo],
            cache_dir=os.getenv("HF_HUB_CACHE", "models"),
        )
        for name, (model_repo, filename) in files.items()
    }
