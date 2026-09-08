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
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument("--quant", default="Q3_K_XL")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--draft-lengths", default="7", help="Comma-separated DFlash lengths to compare"
    )
    parser.add_argument(
        "--prompt-chars",
        default="0",
        help="Comma-separated source-prefix character counts; actual tokens are reported",
    )
    parser.add_argument(
        "--prompt-file", default=str(Path(__file__).with_name("Pipes.py"))
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--kv-cache", choices=["q4_0", "q8_0", "f16"], default="q4_0")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--child", choices=["none", "mtp", "dflash2"])
    args = parser.parse_args()
    if not args.child:
        results = []
        variants = [("none", None), ("mtp", None)] + [
            ("dflash2", int(n)) for n in args.draft_lengths.split(",")
        ]
        for backend, draft_length in variants:
            cmd = [sys.executable, __file__, *sys.argv[1:], "--child", backend]
            env = dict(os.environ, LLM_SPECULATIVE_TYPE=backend)
            if draft_length is not None:
                env["DFLASH_SPEC_DRAFT_N_MAX"] = str(draft_length)
            print(
                f"Benchmarking {backend} n={draft_length}", file=sys.stderr, flush=True
            )
            result = subprocess.run(
                cmd, env=env, stdout=subprocess.PIPE, text=True, check=True
            )
            record = next(
                line.removeprefix("RESULT ")
                for line in result.stdout.splitlines()
                if line.startswith("RESULT ")
            )
            results.append(json.loads(record))
            print("MEASUREMENT " + record, file=sys.stderr, flush=True)
        baseline = [p["sha256"] for p in results[0]["prompts"]]
        for result in results:
            result["matches_greedy_baseline"] = (
                args.temperature == 0
                and [p["sha256"] for p in result["prompts"]] == baseline
            )
        print(json.dumps(results, indent=2))
        return

    os.environ["LLM_SPECULATIVE_TYPE"] = args.child
    os.environ["KV_CACHE_TYPE"] = args.kv_cache
    from ezlocalai.LLM import LLM

    llm = LLM(
        model=args.model,
        quant_type=args.quant,
        max_tokens=args.context,
        gpu_layers=-1,
        main_gpu=args.gpu,
        n_parallel=1,
        batch_size=args.batch,
        ubatch_size=args.ubatch,
    )
    # Exclude first-use CUDA graph setup from decode comparisons.
    llm.server.handle_completions(
        {"prompt": "Warmup: count to ten.", "max_tokens": 32, "temperature": 0}
    )
    results = []
    source = Path(args.prompt_file).read_text()
    prompts = []
    for chars in map(int, args.prompt_chars.split(",")):
        if chars < 0 or chars > len(source):
            raise ValueError(f"prompt-chars must be between 0 and {len(source)}")
        prefix = (
            ("Reference code:\n" + source[:chars] + "\nEnd of reference.\n")
            if chars
            else ""
        )
        for repeat in range(args.repeats):
            for kind, question in (
                (
                    "code",
                    "Write a complete Python implementation of a thread-safe bounded work queue, with cancellation, error handling, and tests. Be detailed.\n",
                ),
                (
                    "prose",
                    "Explain how to design a reliable concurrent inference service. Discuss backpressure, cancellation, latency, fairness, memory, and recovery in at least 500 words.\n",
                ),
            ):
                prompts.append((chars, repeat, kind, prefix + question))
    for chars, repeat, kind, prompt in prompts:
        # Call the native completion API to keep effective greedy sampling
        # independent of ezlocalai's enforced Qwen production sampling profile.
        started = time.monotonic()
        response = llm.server.handle_chat_completions(
            {
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": args.tokens,
                "temperature": args.temperature,
                "top_p": 0.95,
                "seed": 42,
                "cache_prompt": True,
            }
        )
        text = response["choices"][0]["message"]["content"]
        results.append(
            {
                "kind": kind,
                "prefix_chars": chars,
                "repeat": repeat,
                "wall_seconds": time.monotonic() - started,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "timings": response.get("timings"),
                "usage": response.get("usage"),
            }
        )
    print(
        "RESULT "
        + json.dumps(
            {
                "backend": args.child,
                "draft_n_max": (
                    os.getenv("DFLASH_SPEC_DRAFT_N_MAX")
                    if args.child == "dflash2"
                    else None
                ),
                "context": args.context,
                "kv_cache": args.kv_cache,
                "temperature": args.temperature,
                "batch": args.batch,
                "ubatch": args.ubatch,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "prompts": results,
            }
        )
    )


if __name__ == "__main__":
    main()
