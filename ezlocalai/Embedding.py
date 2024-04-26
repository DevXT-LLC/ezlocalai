from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from ezlocalai.Helpers import get_tokens, chunk_content
from huggingface_hub import hf_hub_download

import torch
import os


class Embedding:
    def __init__(self):
        model_dir = os.path.join(
            os.getcwd(),
            "models",
            "models--hooman650--bge-m3-onnx-o4",
            "snapshots",
            "848e8bc2408aad2c8849c7be9475f7aec3ee781a",
        )
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        # if model_optimized.onnx.data does not exist, download the model.
        files = [
            "config.json",
            "model_optimized.onnx",
            "model_optimized.onnx.data",
            "ort_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        for file in files:
            if not os.path.exists(os.path.join(model_dir, file)):
                hf_hub_download(
                    repo_id="hooman650/bge-m3-onnx-o4",
                    filename=file,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                )
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            "hooman650/bge-m3-onnx-o4",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(os.getcwd(), "models"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hooman650/bge-m3-onnx-o4",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            cache_dir=os.path.join(os.getcwd(), "models"),
        )

    def get_embeddings(self, input):
        tokens = get_tokens(input)
        sentences = chunk_content(input)
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        out = self.model(**encoded_input, return_dict=True).last_hidden_state
        dense_vecs = torch.nn.functional.normalize(out[:, 0], dim=-1)
        embeddings = dense_vecs.cpu().detach().numpy().tolist()
        data = []
        for i, embedding in enumerate(embeddings):
            data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                }
            )
        return {
            "object": "list",
            "data": data,
            "model": "bge-m3",
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
