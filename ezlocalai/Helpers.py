from typing import List
import spacy
import tiktoken


def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def chunk_content(text: str) -> List[str]:
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
