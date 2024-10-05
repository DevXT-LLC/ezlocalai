try:
    import spacy
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy
from typing import List
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


# Export chunks of paragraphs up to 2000 tokens
def chunk_content_by_tokens(text: str, max_tokens: int = 2000) -> List[str]:
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    encoding = tiktoken.get_encoding("cl100k_base")
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    def add_to_chunk(content: str):
        nonlocal current_chunk, current_chunk_tokens, chunks
        content_tokens = encoding.encode(content)
        if current_chunk_tokens + len(content_tokens) > max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_chunk_tokens = 0
        current_chunk.append(content)
        current_chunk_tokens += len(content_tokens)

    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)
        if len(paragraph_tokens) <= max_tokens:
            add_to_chunk(paragraph)
        else:
            # Split long paragraph into sentences using spaCy
            doc = nlp(paragraph)
            sentences = [sent.text for sent in doc.sents]
            current_sentence_group = []
            current_group_tokens = 0

            for sentence in sentences:
                sentence_tokens = encoding.encode(sentence)
                if current_group_tokens + len(sentence_tokens) <= max_tokens:
                    current_sentence_group.append(sentence)
                    current_group_tokens += len(sentence_tokens)
                else:
                    # Add the current group of sentences as a chunk
                    if current_sentence_group:
                        add_to_chunk(" ".join(current_sentence_group))
                    current_sentence_group = [sentence]
                    current_group_tokens = len(sentence_tokens)

            # Add any remaining sentences
            if current_sentence_group:
                add_to_chunk(" ".join(current_sentence_group))

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
