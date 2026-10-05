#!/usr/bin/env python3
"""Compare opt-in CUDA graph scheduling in fresh native LLM processes.

Run inside the worker image with an idle GPU. Keeps the configured model,
quantization, context and parallel slots. No router registration or .env edits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("0", "1"))
    args = parser.parse_args()
    if args.candidate is None:
        for candidate in ("0", "1", "0"):
            subprocess.run(
                [sys.executable, __file__, "--candidate", candidate], check=True
            )
        return
    os.environ["GGML_CUDA_GRAPH_OPT"] = args.candidate
    from ezlocalai.LLM import LLM
    from Globals import getenv

    llm = LLM(model=getenv("DEFAULT_MODEL"))
    for trial in range(5):
        started = time.perf_counter()
        result = llm.server.handle_chat_completions(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Explain why the sky looks blue in clear language. Trial {trial}.",
                    }
                ],
                "max_tokens": 128,
                "temperature": 0,
                "seed": 42,
            }
        )
        assert result.get("choices"), result
        message = result["choices"][0]["message"]
        # Reasoning models may spend the whole token budget in reasoning.
        # Include it when comparing deterministic output between candidates.
        output = json.dumps(message, sort_keys=True, ensure_ascii=False)
        print(
            json.dumps(
                {
                    "graph_opt": args.candidate,
                    "trial": trial,
                    "seconds": time.perf_counter() - started,
                    "usage": result.get("usage"),
                    "timings": result.get("timings"),
                    "message": message,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
