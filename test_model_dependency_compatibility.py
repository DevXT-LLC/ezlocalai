"""Small offline checks for the Transformers/Accelerate/Hub dependency upgrade."""

import inspect
from pathlib import Path
import tempfile
import unittest

import torch
from accelerate import cpu_offload
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import (
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    QuantoConfig,
)


class ModelDependencyCompatibilityTests(unittest.TestCase):
    def test_gemma_checkpoint_load_device_map_and_generation(self):
        # The LTX text encoder uses Gemma 3 and Accelerate's device-map loader.
        # Use tiny random text weights to test those APIs without gated downloads.
        config = Gemma3TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            sliding_window=32,
            layer_types=["full_attention"],
            eos_token_id=2,
            pad_token_id=0,
        )
        model = Gemma3ForCausalLM(config).eval()
        inputs = torch.tensor([[1, 3, 4]])
        with torch.no_grad():
            expected = model(inputs).logits
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder, safe_serialization=True, max_shard_size="5KB")
            self.assertTrue((Path(folder) / "model.safetensors.index.json").is_file())
            loaded = Gemma3ForCausalLM.from_pretrained(
                folder, device_map="cpu", dtype=torch.float32
            ).eval()
            with torch.no_grad():
                torch.testing.assert_close(loaded(inputs).logits, expected)
                generated = loaded.generate(inputs, max_new_tokens=2, do_sample=False)
            self.assertGreater(generated.shape[1], inputs.shape[1])
            # Exercise the CPU offload hook used by diffusion pipelines.
            cpu_offload(loaded, execution_device=torch.device("cpu"))
            with torch.no_grad():
                torch.testing.assert_close(loaded(inputs).logits, expected)

    def test_ltx_and_quantization_integrations_still_import(self):
        from ezlocalai import LTX_VIDEO
        from sentence_transformers import SentenceTransformer

        self.assertTrue(LTX_VIDEO.import_success)
        self.assertTrue(callable(Gemma3ForConditionalGeneration.from_pretrained))
        self.assertTrue(callable(SentenceTransformer))
        self.assertEqual(QuantoConfig(weights="int8").weights, "int8")
        # Config parsing remains available even in CPU-only builds without BNB.
        self.assertFalse(BitsAndBytesConfig().load_in_4bit)

    def test_hub_download_calls_accept_existing_arguments(self):
        # Bind the actual Hub signatures used by native GGUF/TTS/video downloads.
        inspect.signature(hf_hub_download).bind(
            "org/model",
            "model.gguf",
            revision="main",
            local_dir="models",
            cache_dir="models",
        )
        inspect.signature(snapshot_download).bind(
            "org/model", revision="main", cache_dir="models", allow_patterns=["*.json"]
        )


if __name__ == "__main__":
    unittest.main()
