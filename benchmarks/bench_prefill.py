"""Benchmark evaluated prompt tokens and cache reuse through an HTTP endpoint.

Use a llama-server endpoint for controlled greedy sampling. Worker endpoints
may apply model presets. This script never changes model/runtime settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid


def prefill_cases(source, characters, append_characters=24000, nonce=""):
    if not source or characters < 1 or append_characters < 0:
        raise ValueError("require nonempty source, positive length and nonnegative extension")
    prefix = (source * ((characters + len(source) - 1) // len(source)))[:characters]
    base = [
        {"role": "system", "content": "You review source code. Reply briefly."},
        {"role": "user", "content": nonce + "Reference code:\n" + prefix
         + "\nWhat language is this? Reply in one word."},
    ]
    continuation = base + [{"role": "assistant", "content": "Python."}]
    return [
        ("cold", base), ("repeat", base),
        ("append", continuation + [{"role": "user", "content": "What is its purpose? Reply in one sentence."}]),
        ("extend", continuation + [{"role": "user", "content": source[:append_characters]
                                   + "\nWhat language is this? Reply in one word."}]),
    ]


def measure(base, key, messages, max_tokens=32):
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
               "seed": 42, "cache_prompt": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        value = json.load(response)
    elapsed = time.perf_counter() - started
    if value.get("error") or not value.get("choices"):
        raise RuntimeError("inference response failed; no benchmark sample recorded")
    timings = value.get("timings") or {}
    required = ("prompt_n", "cache_n", "prompt_ms", "prompt_per_second")
    if any(name not in timings for name in required):
        raise RuntimeError("native evaluated/cached prompt timings are required")
    return {"wall_seconds": elapsed, "timings": timings, "usage": value.get("usage"),
            "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(json.dumps(
                value["choices"][0]["message"], sort_keys=True).encode()).hexdigest()}


def require_cold(sample):
    if sample["timings"]["cache_n"] >= 100:
        raise RuntimeError("cold sample reused a long prefix; exclude it from cold-prefill comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-chars", type=int, nargs="+", default=[300000, 680000])
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--exact-first-prompt", action="store_true",
                        help="match the old prompt hash; requires one length and a freshly started engine")
    parser.add_argument("--api-key-env", default="EZLOCALAI_API_KEY")
    args = parser.parse_args()
    if args.cold_repeats < 1 or any(n < 1 for n in args.prompt_chars):
        parser.error("lengths and repeat count must be positive")
    if args.exact_first_prompt and len(args.prompt_chars) != 1:
        parser.error("exact-first mode requires one prompt length per fresh engine")
    source = args.source.read_text(encoding="utf-8")
    if not source:
        parser.error("source must be nonempty")
    report = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "requested_sampling": {"temperature": 0, "seed": 42, "enable_thinking": False},
              "sampling_note": "worker endpoints may override sampling through model presets",
              "warmup_excluded": True,
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    key = os.environ.get(args.api_key_env, "")
    measure(args.base_url, key, [{"role": "user", "content": "Warmup " + uuid.uuid4().hex
                                + "\n" + source[:4000] + "\nReply OK."}], 8)
    for characters in args.prompt_chars:
        for trial in range(args.cold_repeats):
            nonce = "" if args.exact_first_prompt and trial == 0 else "[cold " + uuid.uuid4().hex + "]\n"
            cases = prefill_cases(source, characters, nonce=nonce)
            for kind, messages in cases if trial == 0 else cases[:1]:
                sample = measure(args.base_url, key, messages)
                if kind == "cold":
                    require_cold(sample)
                sample.update(kind=kind, source_characters=characters, trial=trial)
                report["results"].append(sample)
                temporary = args.output.with_name(args.output.name + ".tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(args.output)
                print(json.dumps(sample), flush=True)


if __name__ == "__main__":
    main()
