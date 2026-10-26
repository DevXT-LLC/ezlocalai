"""Qualify fixed-context cold prefill and image understanding on an isolated engine.

This client never changes model/runtime settings or retries inference with a
smaller context. The caller owns reservation, immutable launch and restoration.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time
import urllib.parse
import urllib.request
import uuid
import zlib

from bench_progression import (atomic_report, digest, native_metrics,
                               prompt_cache_key_argument, validate_metadata)


def red_png(width=64, height=64):
    if width < 1 or height < 1:
        raise ValueError("positive image dimensions required")

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    pixels = (b"\0" + b"\xff\0\0" * width) * height
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels, 9)) + chunk(b"IEND", b""))


class Client:
    def __init__(self, base, key, timeout):
        self.base, self.key, self.timeout = base.rstrip("/"), key, timeout

    def __call__(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.base + path, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict) or value.get("error"):
            raise RuntimeError("engine request failed at " + path)
        return value


def request_body(messages, prompt_cache_key=None):
    body = {"messages": messages, "max_tokens": 8, "temperature": 0, "seed": 42,
            "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if prompt_cache_key is not None:
        body["prompt_cache_key"] = prompt_cache_key_argument(prompt_cache_key)
    return body


def launch_settings(metadata):
    values = dict(metadata.get("original_parameters") or {})
    values.update(metadata.get("overrides") or {})
    return {"n_ctx": metadata.get("n_ctx", values.get("-c", values.get("--ctx-size"))),
            "n_ubatch": metadata.get("n_ubatch", values.get("-ub", values.get("--ubatch-size")))}


def validate_properties(props, context, ubatch, metadata):
    settings = props.get("default_generation_settings") or {}
    observed = settings.get("n_ctx", props.get("n_ctx"))
    # llama.cpp can round the allocated FA context upward. The request budget
    # remains the exact declared launch context, and any downward fit fails.
    padded = (context + 255) // 256 * 256
    if isinstance(observed, bool) or observed not in (context, padded):
        raise RuntimeError("effective context differs from declared context/padding; no reduced-context retry allowed")
    if props.get("total_slots") != 1:
        raise RuntimeError("capacity qualification requires one isolated engine slot")
    if props.get("is_sleeping") or (props.get("modalities") or {}).get("vision") is not True:
        raise RuntimeError("engine must be awake with its vision projector loaded")
    launch = launch_settings(metadata)
    if launch["n_ctx"] is None or int(launch["n_ctx"]) != context:
        raise RuntimeError("verified launch metadata must retain the exact declared context")
    exposed_ubatch = settings.get("n_ubatch", (settings.get("params") or {}).get("n_ubatch", props.get("n_ubatch")))
    if exposed_ubatch is not None and int(exposed_ubatch) != ubatch:
        raise RuntimeError("engine physical batch differs from expected n_ubatch")
    if launch["n_ubatch"] is None or int(launch["n_ubatch"]) != ubatch:
        raise RuntimeError("verified launch metadata must match expected n_ubatch; /props does not expose it at the pinned revision")
    return {"declared_context": context, "effective_context": observed, "expected_n_ubatch": ubatch,
            "ubatch_evidence": "endpoint and launch metadata" if exposed_ubatch is not None else "caller-verified launch metadata",
            "model_identity_evidence": "caller model/projector fingerprints and unchanged engine properties",
            "props_sha256": digest(props)}


def language_messages(source, characters, run_id):
    text = (source * ((characters + len(source) - 1) // len(source)))[:characters]
    return [{"role": "system", "content": "[capacity " + run_id + "] You review source code. Reply briefly."},
            {"role": "user", "content": "Reference code:\n" + text + "\nWhat language is this? Reply in one word."}]


def count_text(client, messages, prompt_cache_key=None):
    prompt = client("/apply-template", request_body(messages, prompt_cache_key)).get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("text chat template did not return a string")
    tokens = client("/tokenize", {"content": prompt, "add_special": True, "parse_special": True}).get("tokens")
    if not isinstance(tokens, list) or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens):
        raise RuntimeError("tokenizer did not return token IDs")
    return len(tokens)


def prepare_capacity(source, context, count, run_id, fraction=.95, reserve=32):
    if not source or context < 512 or not .9 <= fraction < 1 or reserve < 8:
        raise ValueError("require nonempty source, context>=512, fraction>=0.9 and headroom>=8")
    target = math.floor(context * fraction)
    if target + reserve > context:
        raise ValueError("target and reserved headroom exceed the unchanged context")
    tolerance = max(16, context // 1000)
    seen = {}

    def measure(characters):
        if characters not in seen:
            seen[characters] = count(language_messages(source, characters, run_id))
        return seen[characters]

    base = measure(0)
    probe = min(max(len(source), 16000), 32000)
    density = (measure(probe) - base) / probe
    if density <= 0:
        raise RuntimeError("source did not add tokens to the context probe")
    estimate = max(1, int((target - base) / density))
    limit = context * 64
    low, high = 0, min(limit, max(estimate * 2, probe))
    while measure(high) < target and high < limit:
        low, high = high, min(limit, high * 2)
    if measure(high) < target - tolerance:
        raise RuntimeError("cannot construct a near-context source prompt")
    best = (0, base)
    candidate = min(estimate, high)
    for _ in range(32):
        n_tokens = measure(candidate)
        if n_tokens <= target and n_tokens > best[1]:
            best = candidate, n_tokens
        if target - tolerance <= n_tokens <= target:
            break
        if n_tokens > target:
            high = candidate - 1
        else:
            low = candidate + 1
        if low > high:
            break
        candidate = (low + high) // 2
    characters, input_tokens = best
    if not target - tolerance <= input_tokens <= target or input_tokens + reserve > context:
        raise RuntimeError("prepared prompt missed the near-context target; no inference retry allowed")
    return language_messages(source, characters, run_id), {
        "source_characters": characters, "input_tokens": input_tokens, "target_input_tokens": target,
        "declared_context": context, "input_fraction": input_tokens / context,
        "reserved_headroom": reserve, "remaining_context_tokens": context - input_tokens,
        "tokenizer_passes": len(seen)}


def validate_completion(value, expected_tokens, expected_word, context):
    if value.get("truncated") is True or not value.get("choices"):
        raise RuntimeError("inference failed or truncated its input")
    choice = value["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise RuntimeError("qualification answer did not finish normally")
    answer = (choice.get("message") or {}).get("content")
    if not isinstance(answer, str) or not re.fullmatch(r"\s*" + re.escape(expected_word) + r"[.!]?\s*", answer, re.I):
        raise RuntimeError("qualification answer must be the unambiguous word " + expected_word)
    timings = value.get("timings")
    metrics = native_metrics(timings)
    if metrics["cached_prompt_tokens"] != 0 or metrics["evaluated_prompt_tokens"] != expected_tokens:
        raise RuntimeError("inference reused/truncated prompt tokens instead of evaluating the full prepared input")
    usage = value.get("usage") or {}
    if usage.get("prompt_tokens") != expected_tokens or expected_tokens + 8 > context:
        raise RuntimeError("inference input-token accounting exceeds or differs from the fixed context budget")
    return {"answer": answer, "timings": timings, "usage": usage, **metrics,
            "output_sha256": digest(choice["message"])}


def qualify(client, source, context, ubatch, metadata, run_id, save, fraction=.95, prompt_cache_key=None):
    props = client("/props")
    runtime = validate_properties(props, context, ubatch, metadata)
    save("runtime", runtime)
    messages, preparation = prepare_capacity(source, context,
        lambda m: count_text(client, m, prompt_cache_key), run_id, fraction)
    body = request_body(messages, prompt_cache_key)
    exact = client("/v1/chat/completions/input_tokens", body).get("input_tokens")
    if exact != preparation["input_tokens"]:
        raise RuntimeError("chat-token endpoint differs from prepared tokenization; inference was not attempted")
    save("capacity_preparation", {**preparation, "request_sha256": digest(body)})
    started = time.perf_counter()
    response = client("/v1/chat/completions", body)
    wall = time.perf_counter() - started
    save("capacity_response", response)
    result = validate_completion(response, exact, "Python", context)
    result.update(wall_seconds=wall, request_sha256=digest(body), **preparation)
    save("capacity", result)

    image = red_png()
    image_url = "data:image/png;base64," + base64.b64encode(image).decode()
    messages = [{"role": "system", "content": "[vision " + run_id + "] Answer the image question in one word."},
                {"role": "user", "content": [
                    {"type": "text", "text": "What is the single solid color in this image? Reply with the color word only."},
                    {"type": "image_url", "image_url": {"url": image_url}}]}]
    body = request_body(messages, prompt_cache_key)
    n_tokens = client("/v1/chat/completions/input_tokens", body).get("input_tokens")
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 1 or n_tokens + 32 > context:
        raise RuntimeError("vision input exceeds or lacks a verified context budget")
    save("vision_preparation", {"input_tokens": n_tokens, "request_sha256": digest(body),
         "image_sha256": hashlib.sha256(image).hexdigest(), "image_width": 64, "image_height": 64})
    started = time.perf_counter()
    response = client("/v1/chat/completions", body)
    wall = time.perf_counter() - started
    save("vision_response", response)
    result = validate_completion(response, n_tokens, "red", context)
    result.update(wall_seconds=wall, request_sha256=digest(body),
                  image_sha256=hashlib.sha256(image).hexdigest(), image_width=64, image_height=64)
    save("vision", result)
    if digest(client("/props")) != runtime["props_sha256"]:
        raise RuntimeError("engine properties changed during qualification")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("source", type=Path)
    parser.add_argument("context", type=int)
    parser.add_argument("expected_ubatch", type=int)
    parser.add_argument("label")
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path, required=True, help="verified immutable launch metadata, including n_ctx/n_ubatch or original_parameters plus overrides")
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--prompt-cache-key", type=prompt_cache_key_argument,
                        help="optional named key for text and vision requests; full cold evaluation remains required")
    parser.add_argument("--fraction", type=float, default=.95)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--api-key-env", default="EZLOCALAI_API_KEY")
    args = parser.parse_args(argv)
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        parser.error("require an HTTP(S) base URL without credentials, query or fragment")
    if args.context < 512 or args.expected_ubatch < 1 or not .9 <= args.fraction < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("invalid fixed context, batch, fraction or timeout")
    return args


def main(argv=None):
    args = arguments(argv)
    source = args.source.read_text(encoding="utf-8")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    report = {"status": "running", "label": args.label, "run_id": args.run_id,
              "declared_context": args.context, "expected_n_ubatch": args.expected_ubatch,
              "target_fraction": args.fraction, "metadata": metadata, "prompt_cache_key": args.prompt_cache_key,
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "requested_sampling": {"temperature": 0, "seed": 42, "enable_thinking": False},
              "inference_retries": 0, "context_downgrade_allowed": False}

    def save(name, value):
        report[name] = value
        atomic_report(args.output, report)
        if not name.endswith("_response"):
            print(json.dumps({"stage": name, **value}), flush=True)

    atomic_report(args.output, report)
    try:
        qualify(Client(args.base_url, os.environ.get(args.api_key_env, ""), args.timeout), source,
                args.context, args.expected_ubatch, metadata, args.run_id, save, args.fraction, args.prompt_cache_key)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        atomic_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
