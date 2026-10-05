#!/usr/bin/env python3
"""Prefetch Wan 2.2 T2V/I2V GGUF assets without loading them onto a GPU."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ezlocalai.WAN import download_wan_models, T2V_REPO


def activate_wan(env_path):
    """Explicitly migrate the selected deployment's model, retaining a backup."""
    env_path = Path(env_path)
    original = env_path.read_text()
    backup = env_path.with_name(env_path.name + ".pre-wan2.2")
    if not backup.exists():
        # .env can contain credentials; never put its contents in command output.
        import os

        with os.fdopen(
            os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as out:
            out.write(original)
    lines = original.splitlines()
    lines = [line for line in lines if not line.lstrip().startswith("VIDEO_MODEL=")]
    lines.append(f"VIDEO_MODEL={T2V_REPO}")
    env_path.write_text("\n".join(lines) + "\n")
    print(f"Configured VIDEO_MODEL={T2V_REPO}; restart the worker to activate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("t2v", "i2v", "both"), default="both")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument(
        "--activate",
        type=Path,
        metavar="ENV_FILE",
        help="After downloads succeed, update VIDEO_MODEL in this .env file (with a private backup)",
    )
    args = parser.parse_args()
    for mode in ("t2v", "i2v"):
        if args.mode in (mode, "both"):
            print(
                json.dumps({mode: download_wan_models(mode == "i2v", args.quant)}),
                flush=True,
            )
    if args.activate:
        activate_wan(args.activate)


if __name__ == "__main__":
    main()
