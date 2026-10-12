#!/usr/bin/env python3
"""Persist tested GB10 replica counts without altering other worker settings.

Run as the repository owner, then recreate the GB10 compose service. A private
backup is saved alongside the environment file before the first change.
"""

import argparse
import os
from pathlib import Path
import re
import tempfile


POOL_KEYS = ("TTS_N_PARALLEL", "STT_N_PARALLEL", "EMBEDDING_N_PARALLEL")


def configure_pools(path: Path, replicas: int):
    if not 1 <= replicas <= 16:
        raise ValueError("replicas must be between 1 and 16")
    original = path.read_text()
    backup = path.with_name(path.name + ".pre-gb10-pools")
    if not backup.exists():
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            output.write(original)
    updated = original
    for key in POOL_KEYS:
        pattern = rf"(?m)^{key}=.*$"
        if re.search(pattern, updated):
            updated = re.sub(pattern, f"{key}={replicas}", updated)
        else:
            updated = updated.rstrip("\n") + f"\n{key}={replicas}\n"
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".gb10-pools-")
    try:
        with os.fdopen(fd, "w") as output:
            output.write(updated)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--replicas", type=int, required=True)
    args = parser.parse_args()
    configure_pools(args.env_file, args.replicas)
    print(f"Configured {args.replicas} replicas for embedding, STT and TTS.")


if __name__ == "__main__":
    main()
