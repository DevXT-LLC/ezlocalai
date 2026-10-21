"""API-level benchmark for one ezlocalai worker (Python or C++).

usage: bench.py BASE_URL API_KEY PREFIX_FILE OUT_JSON LABEL
Every request is greedy and streamed; prefill cases start with a unique nonce
so no prompt-cache reuse occurs unless the case measures reuse explicitly.
"""
import json
import statistics
import sys
import time
import urllib.request
import uuid

base, key, prefix_file, out_path, label = sys.argv[1:6]
MODEL = "unsloth/Qwen3.8-27B-GGUF"
prefix_text = open(prefix_file, encoding="utf-8").read()


def stream_chat(messages, max_tokens, timeout=1800):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    request = urllib.request.Request(base + "/v1/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    timings = None
    pieces = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        buffer = b""
        for raw in response:
            buffer += raw
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    if data.get("timings"):
                        timings = data["timings"]
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            pieces += 1
                            if first is None:
                                first = time.perf_counter()
    end = time.perf_counter()
    return {"status": status, "wall_s": end - start, "ttft_s": (first - start) if first else None,
            "decode_wall_tps": (pieces - 1) / (end - first) if first and pieces > 1 and end > first else None,
            "pieces": pieces, "timings": timings}


def nonce():
    return f"[run {uuid.uuid4().hex}]\n"


def prefill_messages(chars, question="Summarize what this code does in one sentence."):
    return [{"role": "user", "content": nonce() + prefix_text[:chars] + "\n\n" + question}]


results = {"label": label, "base": base, "started": time.time(), "cases": {}}


def record(name, runs):
    results["cases"][name] = runs
    t = [r["timings"] or {} for r in runs]
    summary = {
        "wall_s": [round(r["wall_s"], 3) for r in runs],
        "ttft_s": [round(r["ttft_s"], 3) if r["ttft_s"] else None for r in runs],
        "prompt_n": [x.get("prompt_n") for x in t], "cache_n": [x.get("cache_n") for x in t],
        "prompt_tps": [round(x.get("prompt_per_second", 0), 1) for x in t],
        "predicted_n": [x.get("predicted_n") for x in t],
        "decode_tps": [round(x.get("predicted_per_second", 0), 1) for x in t],
    }
    print(f"{label} {name}: {json.dumps(summary)}", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)


# Warmup (excluded): first-use graph capture and allocation.
stream_chat([{"role": "user", "content": nonce() + "Say hi."}], 16)
stream_chat(prefill_messages(4000), 8)

record("latency_1tok", [stream_chat([{"role": "user", "content": nonce() + "Reply with OK."}], 1) for _ in range(10)])
record("decode_256", [stream_chat([{"role": "user", "content": nonce() + "Write a detailed explanation of how a hash map works, with an example in Python."}], 256) for _ in range(3)])
for chars, name, repeats in ((32000, "prefill_8k", 3), (128000, "prefill_32k", 3), (400000, "prefill_100k", 2)):
    record(name, [stream_chat(prefill_messages(chars), 16) for _ in range(repeats)])
# Prompt-cache reuse: identical 32k prompt, then a short follow-up turn.
messages = prefill_messages(128000)
cold = stream_chat(messages, 16)
repeat = stream_chat(messages, 16)
followup = stream_chat(messages + [{"role": "assistant", "content": "It is a module."}, {"role": "user", "content": "Name one function in it."}], 16)
record("cache_32k_cold_repeat_followup", [cold, repeat, followup])
results["finished"] = time.time()
with open(out_path, "w") as f:
    json.dump(results, f, indent=1)
