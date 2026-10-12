"""Keep wake-word names and artifacts inside their configured storage roots."""

import os
import re
from pathlib import Path


def wakeword_name(word: str) -> str:
    """Preserve existing case/space normalization, rejecting path components."""
    name = word.lower().replace(" ", "_")
    if name in {".", ".."} or not re.fullmatch(r"[^/\\\x00-\x1f\x7f]+", name):
        raise ValueError("Wake word must be a nonempty name without path separators")
    return name


def contained_path(root: Path, *parts: str) -> Path:
    """Resolve symlinks and require a strict descendant, including for new files."""
    root = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root, *parts))
    if not candidate.startswith(root.rstrip(os.sep) + os.sep) or candidate == root:
        raise ValueError("Wake word path must remain inside its storage directory")
    return Path(candidate)


def wakeword_directory(root: Path, word: str) -> Path:
    return contained_path(root, wakeword_name(word))
