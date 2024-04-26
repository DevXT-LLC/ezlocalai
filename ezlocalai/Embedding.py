from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch
from typing import List
import spacy
import tiktoken
import os


def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


class Embedding:
    def __init__(self):
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            "bge-m3-onnx",
            provider=(
                "CUDAExecutionProvider"
                if torch.cuda.is_available()
                else "CPUExecutionProvider"
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hooman650/bge-m3-onnx-o4",
            cache_dir=os.path.join(os.getcwd(), "models"),
        )

    def chunk_content(self, text: str) -> List[str]:
        try:
            sp = spacy.load("en_core_web_sm")
        except:
            spacy.cli.download("en_core_web_sm")
            sp = spacy.load("en_core_web_sm")
        sp.max_length = 99999999999999999999999
        doc = sp(text)
        sentences = list(doc.sents)
        content_chunks = [str(sentence).strip() for sentence in sentences]
        return content_chunks

    def get_embeddings(self, input):
        tokens = get_tokens(input)
        sentences = self.chunk_content(input)
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        out = self.model(**encoded_input, return_dict=True).last_hidden_state
        dense_vecs = torch.nn.functional.normalize(out[:, 0], dim=-1)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": dense_vecs[i].tolist()}
                for i in range(tokens)
            ],
            "model": "bge-m3",
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
