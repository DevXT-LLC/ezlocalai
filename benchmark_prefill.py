"""Measure cold prefill and cache reuse at a real long prompt length.

Run inside the worker image, with the serving worker stopped to free its GPU.
Keep --prompt-file unchanged between variants; allocated context is not the
same as actual input length. Results include native evaluated/cached tokens.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def prefill_cases(
    source, prompt_chars, append_chars, guidance="", guidance_layout="trailing"
):
    prefix = (source * ((prompt_chars + len(source) - 1) // len(source)))[:prompt_chars]
    base = [
        {"role": "system", "content": "You review source code. Reply briefly."},
        {
            "role": "user",
            "content": "Reference code:\n"
            + prefix
            + "\nWhat language is this? Reply in one word.",
        },
    ]
    if guidance and guidance_layout == "prefix":
        base.append({"role": "user", "content": guidance})
    cases = [
        ("cold", base),
        ("repeat", base),
        (
            "append",
            base
            + [
                {"role": "assistant", "content": "Python."},
                {
                    "role": "user",
                    "content": "What is its purpose? Reply in one sentence.",
                },
            ],
        ),
        (
            "extend",
            base
            + [
                {"role": "assistant", "content": "Python."},
                {
                    "role": "user",
                    "content": source[:append_chars]
                    + "\nWhat language is this? Reply in one word.",
                },
            ],
        ),
    ]
    if guidance and guidance_layout == "trailing":
        cases = [
            (name, messages + [{"role": "user", "content": guidance}])
            for name, messages in cases
        ]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument("--quant", default="Q3_K_XL")
    parser.add_argument("--context", type=int, default=220000)
    parser.add_argument("--backend", choices=["none", "mtp", "dflash2"], default="mtp")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--kv-cache", choices=["q4_0", "q8_0", "f16"], default="q4_0")
    parser.add_argument("--prompt-file", default="Pipes.py")
    parser.add_argument("--prompt-chars", type=int, default=720000)
    parser.add_argument("--append-chars", type=int, default=24000)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--guidance-file",
        help="Optional fixed instructions to test prefix versus trailing placement",
    )
    parser.add_argument(
        "--guidance-layout", choices=["prefix", "trailing"], default="trailing"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.context, args.batch, args.ubatch, args.prompt_chars, args.tokens) < 1:
        parser.error(
            "Context, batch sizes, prompt length and output budget must be positive"
        )
    if args.ubatch > args.batch or args.tokens >= args.context or args.append_chars < 0:
        parser.error("Require ubatch <= batch, tokens < context and append-chars >= 0")
    source = Path(args.prompt_file).read_text()
    if not source:
        parser.error("The prompt source must not be empty")
    os.environ["LLM_SPECULATIVE_TYPE"] = args.backend
    os.environ["KV_CACHE_TYPE"] = args.kv_cache
    for family in ("3090", "4090", "5090", "T4", "A100", "H100"):
        os.environ[f"KV_CACHE_TYPE_{family}"] = ""
    from ezlocalai.LLM import LLM
    import torch
    import xllamacpp

    llm = LLM(
        model=args.model,
        quant_type=args.quant,
        max_tokens=args.context,
        gpu_layers=-1,
        main_gpu=0,
        n_parallel=1,
        batch_size=args.batch,
        ubatch_size=args.ubatch,
    )
    guidance = Path(args.guidance_file).read_text() if args.guidance_file else ""
    cases = prefill_cases(
        source, args.prompt_chars, args.append_chars, guidance, args.guidance_layout
    )
    report = {
        "settings": vars(args),
        "guidance_sha256": hashlib.sha256(guidance.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "xllamacpp": xllamacpp.__version__,
        "effective_batch": llm.xlc_params.n_batch,
        "effective_ubatch": llm.xlc_params.n_ubatch,
        "results": [],
    }
    for name, messages in cases:
        started = time.monotonic()
        response = llm.server.handle_chat_completions(
            {
                "messages": messages,
                "max_tokens": args.tokens,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        if response.get("error"):
            raise RuntimeError(response["error"])
        message = response["choices"][0]["message"]
        record = {
            "kind": name,
            "wall_seconds": time.monotonic() - started,
            "timings": response.get("timings"),
            "usage": response.get("usage"),
            "output_sha256": hashlib.sha256(
                json.dumps(message, sort_keys=True).encode()
            ).hexdigest(),
        }
        report["results"].append(record)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print("MEASUREMENT " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
