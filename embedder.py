import os
import tarfile
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import List, Union, Sequence


def embed_text(text) -> List[Union[Sequence[float], Sequence[int]]]:
    texts = [text]
    model_path = os.path.join(os.getcwd(), "models")
    onnx_path = os.path.join(model_path, "onnx")
    if not all(
        os.path.exists(os.path.join(onnx_path, f))
        for f in [
            "config.json",
            "model.onnx",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
        ]
    ):
        with tarfile.open(
            name=os.path.join(os.getcwd(), "onnx.tar.gz"), mode="r:gz"
        ) as tar:
            tar.extractall(path=model_path)
    tokenizer = Tokenizer.from_file(os.path.join(onnx_path, "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
    model = ort.InferenceSession(
        os.path.join(onnx_path, "model.onnx"), providers=ort.get_available_providers()
    )
    all_embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        encoded = [tokenizer.encode(d) for d in batch]
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        onnx_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }
        last_hidden_state = model.run(None, onnx_input)[0]
        input_mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1), last_hidden_state.shape
        )
        sum_hidden_state = np.sum(last_hidden_state * input_mask_expanded, 1)
        embeddings = sum_hidden_state / np.clip(
            input_mask_expanded.sum(1), a_min=1e-9, a_max=None
        )
        norm = np.linalg.norm(embeddings, axis=1)
        norm[norm == 0] = 1e-12
        embedding = embeddings / norm[:, np.newaxis]
        all_embeddings.append(embedding.astype(np.float32))
    return np.concatenate(all_embeddings).tolist()
