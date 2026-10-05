#!/usr/bin/env python3
"""Prefetch Wan 2.2 T2V/I2V GGUF assets without loading them onto a GPU."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ezlocalai.WAN import download_wan_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("t2v", "i2v", "both"), default="both")
    parser.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args()
    for mode in ("t2v", "i2v"):
        if args.mode in (mode, "both"):
            print(
                json.dumps({mode: download_wan_models(mode == "i2v", args.quant)}),
                flush=True,
            )


if __name__ == "__main__":
    main()
