"""Isolated greedy correctness/speed smoke test for MTP and DFlash2.

Run only on an idle worker with enough VRAM for the target + draft. This loads
models directly, not through the router, and does not modify server settings.
Use the production API separately to benchmark realistic sampling/workloads.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument("--quant", default="Q3_K_XL")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--child", choices=["none", "mtp", "dflash2"])
    args = parser.parse_args()
    if not args.child:
        results = []
        for backend in ("none", "mtp", "dflash2"):
            cmd = [sys.executable, __file__, *sys.argv[1:], "--child", backend]
            env = dict(os.environ, LLM_SPECULATIVE_TYPE=backend)
            result = subprocess.run(
                cmd, env=env, stdout=subprocess.PIPE, text=True, check=True
            )
            record = next(
                line.removeprefix("RESULT ")
                for line in result.stdout.splitlines()
                if line.startswith("RESULT ")
            )
            results.append(json.loads(record))
        baseline = [p["sha256"] for p in results[0]["prompts"]]
        for result in results:
            result["matches_greedy_baseline"] = [
                p["sha256"] for p in result["prompts"]
            ] == baseline
        print(json.dumps(results, indent=2))
        return

    os.environ["LLM_SPECULATIVE_TYPE"] = args.child
    from ezlocalai.LLM import LLM

    llm = LLM(
        model=args.model,
        quant_type=args.quant,
        max_tokens=args.context,
        gpu_layers=-1,
        main_gpu=args.gpu,
        n_parallel=1,
        batch_size=1024,
        ubatch_size=512,
    )
    results = []
    for prompt in (
        "Write a Python function that returns all primes up to n.\n",
        "Explain why the sky appears blue in three concise sentences.\n",
    ):
        # Call the native completion API to keep effective greedy sampling
        # independent of ezlocalai's enforced Qwen production sampling profile.
        response = llm.server.handle_completions(
            {
                "prompt": prompt,
                "max_tokens": args.tokens,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": False,
            }
        )
        text = response["choices"][0]["text"]
        results.append(
            {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "timings": response.get("timings"),
                "usage": response.get("usage"),
            }
        )
    print(
        "RESULT "
        + json.dumps(
            {"backend": args.child, "context": args.context, "prompts": results}
        )
    )


if __name__ == "__main__":
    main()
