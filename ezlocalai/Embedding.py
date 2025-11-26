import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os


def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # First element is last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class Embedding:
    def __init__(self):
        self.model_name = "BAAI/bge-m3"
        cache_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(cache_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model.half()  # Use fp16 for faster inference on GPU

    def get_embeddings(self, input):
        # Handle both string and list inputs
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input

        # Tokenize inputs
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        # Count tokens for usage
        total_tokens = sum(batch_dict["attention_mask"].sum(dim=1).tolist())

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = mean_pooling(outputs, batch_dict["attention_mask"])
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings_list = embeddings.float().cpu().numpy().tolist()

        data = []
        for i, embedding in enumerate(embeddings_list):
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
            "usage": {"prompt_tokens": int(total_tokens), "total_tokens": int(total_tokens)},
        }
