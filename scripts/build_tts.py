#!/usr/bin/env python3
"""Build the pinned native TTS worker (no Python TTS model dependencies).

Example: python scripts/build_tts.py --cuda --build-dir /opt/ezlocalai-tts/build
Set QWEN_TTS_BIN to <build-dir>/bin/ezlocalai-tts for native installations.
"""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--hip", action="store_true")
    parser.add_argument("--build-dir", default="native/tts/build")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("cmake_args", nargs="*")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "native" / "tts"
    build = Path(args.build_dir).resolve()
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={build / 'bin'}",
    ]
    if args.cuda:
        command += ["-DGGML_CUDA=ON"]
    if args.hip:
        command += ["-DGGML_HIP=ON"]
    subprocess.run(command + args.cmake_args, check=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "ezlocalai-tts",
            "--config",
            "Release",
            "--parallel",
            str(max(1, args.jobs)),
        ],
        check=True,
    )
    print(f"QWEN_TTS_BIN={build / 'bin' / 'ezlocalai-tts'}")


if __name__ == "__main__":
    main()
