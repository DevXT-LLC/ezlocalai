"""Measure first-output latency, native token rates and conversation cache reuse.

Use isolated, reserved llama-server endpoints for controlled sampling. Worker
endpoints may apply model presets. The caller owns drains and restoration;
this program never changes model files, runtime settings or deployments.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
import urllib.parse
import urllib.request
import uuid

from bench_prefill import prefill_cases, require_cold


def digest(value):
    # Match bench_prefill and the saved October 18/25 message hashes.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_metadata(value, path="metadata"):
    """Reject likely credential fields before copying caller metadata to a report."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if not normalized.endswith("_sha256") and any(
                    name in normalized for name in
                    ("api_key", "authorization", "password", "secret", "access_token", "hf_token")):
                raise ValueError("credential field must be removed from " + path + "." + str(key))
            validate_metadata(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_metadata(item, path + "[" + str(index) + "]")


def sse_events(response):
    """Decode SSE data records, including CRLF, comments and multiline data."""
    parts = []
    first_line = True
    for line in response:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if first_line:
            line = line.removeprefix("\ufeff")
            first_line = False
        line = line.rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            if separator and value.startswith(" "):
                value = value[1:]
            parts.append(value)
    if parts:
        # A final unterminated record is parseable, but measure_stream still
        # requires [DONE] so a dropped connection cannot become a fast sample.
        yield "\n".join(parts)


def native_metrics(timings):
    required = ("prompt_n", "cache_n", "prompt_ms", "prompt_per_second",
                "predicted_n", "predicted_ms", "predicted_per_second")
    if not isinstance(timings, dict) or any(name not in timings for name in required):
        raise RuntimeError("native evaluated/cached prompt and decode timings are required")
    for name in required:
        value = timings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError("invalid native timing: " + name)
    for name in ("prompt_n", "cache_n", "predicted_n"):
        if int(timings[name]) != timings[name]:
            raise RuntimeError("invalid native token count: " + name)
    total = timings["prompt_n"] + timings["cache_n"]
    return {"prefill_tokens_per_second": timings["prompt_per_second"],
            "decode_tokens_per_second": timings["predicted_per_second"],
            "evaluated_prompt_tokens": timings["prompt_n"],
            "cached_prompt_tokens": timings["cache_n"],
            "cache_fraction": timings["cache_n"] / total if total else None}


def measure_stream(base, key, messages, max_tokens=32, model=None, cache_key=None,
                   timeout=1800, clock=time.perf_counter, opener=urllib.request.urlopen):
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
               "seed": 42, "stream": True, "stream_options": {"include_usage": True},
               "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        payload["model"] = model
    if cache_key:
        payload["prompt_cache_key"] = cache_key
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers)
    started = clock()
    first_event = first_output = None
    content, reasoning = [], []
    timings = usage = None
    done = False
    finish_reasons = []
    with opener(request, timeout=timeout) as response:
        status = response.status
        if status != 200:
            raise RuntimeError("stream request failed with HTTP " + str(status))
        for record in sse_events(response):
            now = clock()
            if first_event is None:
                first_event = now
            if record.strip() == "[DONE]":
                done = True
                break
            data = json.loads(record)
            if not isinstance(data, dict) or data.get("error"):
                raise RuntimeError("inference stream returned an error")
            if data.get("timings") is not None:
                timings = data["timings"]
            if data.get("usage") is not None:
                usage = data["usage"]
            for choice in data.get("choices") or []:
                if choice.get("index", 0) != 0:
                    raise RuntimeError("benchmark expects one output choice")
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if not isinstance(text, str) or not isinstance(thought, str):
                    raise RuntimeError("unsupported non-text stream delta")
                if text or thought:
                    if first_output is None:
                        first_output = now
                    content.append(text)
                    reasoning.append(thought)
                if choice.get("finish_reason") is not None:
                    finish_reasons.append(choice["finish_reason"])
    ended = clock()
    if not done:
        raise RuntimeError("incomplete inference stream: missing [DONE]")
    if first_output is None:
        raise RuntimeError("inference produced no content or reasoning delta; TTFT is unavailable")
    output = {"content": "".join(content), "reasoning_content": "".join(reasoning)}
    return {"status": status, "wall_seconds": ended - started,
            "ttft_seconds": first_output - started,
            "first_event_seconds": first_event - started,
            "timings": timings, "usage": usage, **native_metrics(timings),
            "request_sha256": digest(payload), "messages_sha256": digest(messages),
            "output_sha256": digest(output), "output": output, "finish_reasons": finish_reasons}


def conversation_messages(source, characters, run_id, branch):
    if not source or characters < 1:
        raise ValueError("conversation source and length must be nonempty")
    shared = (source * ((characters + len(source) - 1) // len(source)))[:characters]
    marker = "BRANCH_" + str(branch) + "_END"
    return [{"role": "system", "content": "Follow the final instruction. Return the branch label only."},
            {"role": "user", "content": "[cache " + run_id + "]\nReference code:\n" + shared
             + "\nBranch record:\n" + (("The exact branch label is " + marker
             + ". These records belong only to this branch.\n") * 90)
             + "\nReturn only the exact branch label from this branch record."}]


def summarize(rows):
    grouped = {}
    for row in rows:
        key = row["kind"]
        if "source_characters" in row:
            key += ":" + str(row["source_characters"])
        grouped.setdefault(key, []).append(row)
    return {name: {"samples": len(values), **{
        field + "_median": statistics.median(v[field] for v in values)
        for field in ("wall_seconds", "ttft_seconds", "prefill_tokens_per_second", "decode_tokens_per_second")}}
        for name, values in grouped.items()}


def prompt_cache_key_argument(value):
    if not value.strip() or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise argparse.ArgumentTypeError("prompt cache key must be nonempty and contain no control characters")
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model")
    parser.add_argument("--metadata", type=Path, help="sanitized host/image/engine/runtime metadata JSON")
    parser.add_argument("--run-id", default=uuid.uuid4().hex,
                        help="reuse across fresh engines for identical requests; never reuse on a warm engine")
    parser.add_argument("--suites", choices=("latency", "decode", "prefill", "cache"), nargs="+",
                        default=["latency", "decode", "prefill", "cache"])
    parser.add_argument("--prompt-chars", type=int, nargs="+", default=[32000, 128000, 300000, 680000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--prompt-cache-key", type=prompt_cache_key_argument,
                        help="optional named key for prefill requests only; preserve across fresh comparison engines")
    parser.add_argument("--cache-chars", type=int, default=100000)
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--cache-rounds", type=int, default=2)
    parser.add_argument("--require-identical-cache-output", action="store_true")
    parser.add_argument("--exact-first-prompt", action="store_true",
                        help="historical messages for first prefill; requires prefill-only, one length, fresh engine")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--api-key-env", default="EZLOCALAI_API_KEY")
    args = parser.parse_args(argv)
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        parser.error("base URL must be HTTP(S) without credentials, query or fragment")
    if any(n < 1 for n in args.prompt_chars + [args.repeats, args.latency_repeats,
            args.decode_tokens, args.prefill_tokens, args.cache_chars]) or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("lengths, repeat counts, token limits and timeout must be positive")
    if args.conversations < 2 or args.cache_rounds < 2:
        parser.error("cache replay needs at least two conversations and two rounds")
    if args.exact_first_prompt and (args.suites != ["prefill"] or len(args.prompt_chars) != 1):
        parser.error("historical mode requires prefill-only and one prompt length per fresh engine")
    if args.prompt_cache_key is not None and "prefill" not in args.suites:
        parser.error("--prompt-cache-key requires the prefill suite")
    return args


def main(argv=None):
    args = arguments(argv)
    source = args.source.read_text(encoding="utf-8")
    if not source:
        raise ValueError("source must be nonempty")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    validate_metadata(metadata)
    report = {"schema_version": 1, "label": args.label, "base_url": args.base_url,
              "run_id": args.run_id, "status": "running", "started_unix_seconds": time.time(),
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "metadata": metadata,
              "source_file_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "configuration": {name: getattr(args, name) for name in (
                  "suites", "model", "prompt_chars", "repeats", "latency_repeats", "decode_tokens",
                  "prefill_tokens", "cache_chars", "conversations", "cache_rounds", "exact_first_prompt", "prompt_cache_key")},
              "requested_sampling": {"temperature": 0, "seed": 42, "enable_thinking": False},
              "sampling_note": "worker endpoints may override sampling through model presets",
              "metrics_note": "TTFT starts at request and ends at first nonempty content/reasoning delta; token rates use native timings, not stream chunks",
              "warmup_excluded": True, "results": []}
    key = os.environ.get(args.api_key_env, "")

    def call(messages, tokens, cache_key=None):
        return measure_stream(args.base_url, key, messages, tokens, args.model, cache_key, args.timeout)

    def record(kind, messages, tokens, cold=False, cache_key=None, **fields):
        sample = call(messages, tokens, cache_key)
        if cold:
            require_cold(sample)
        sample.update(kind=kind, **fields)
        report["results"].append(sample)
        report["summary"] = summarize(report["results"])
        atomic_report(args.output, report)
        print(json.dumps({k: v for k, v in sample.items() if k != "output"}), flush=True)
        return sample

    atomic_report(args.output, report)
    try:
        call([{"role": "user", "content": "[warmup " + args.run_id + "]\n" + source[:4000]
               + "\nReply with OK."}], 8)
        if "latency" in args.suites:
            for trial in range(args.latency_repeats):
                record("latency", [{"role": "user", "content": "[latency " + args.run_id + " "
                       + str(trial) + "]\nReply with OK."}], 8, cold=True, trial=trial)
        if "decode" in args.suites:
            for trial in range(args.repeats):
                record("decode", [{"role": "user", "content": "[decode " + args.run_id + " "
                       + str(trial) + "]\nExplain in detail how a hash map handles collisions, with a Python example. Write at least 500 words."}],
                       args.decode_tokens, cold=True, trial=trial)
        if "prefill" in args.suites:
            for characters in args.prompt_chars:
                for trial in range(args.repeats):
                    nonce = "" if args.exact_first_prompt and trial == 0 else "[cold " + args.run_id + " " + str(characters) + " " + str(trial) + "]\n"
                    cases = prefill_cases(source, characters, nonce=nonce)
                    for kind, messages in cases if trial == 0 else cases[:1]:
                        record("prefill_" + kind, messages, args.prefill_tokens, cold=kind == "cold",
                               cache_key=args.prompt_cache_key, source_characters=characters, trial=trial)
        if "cache" in args.suites:
            originals = {}
            for round_number in range(args.cache_rounds):
                for branch in range(args.conversations):
                    messages = conversation_messages(source, args.cache_chars, args.run_id, branch)
                    sample = call(messages, 8, "bench:" + args.run_id + ":" + str(branch))
                    sample.update(kind="cache_first" if round_number == 0 else "cache_return",
                                  conversation=branch, round=round_number, source_characters=args.cache_chars)
                    content = sample["output"]["content"]
                    sample["expected_branch_present"] = "BRANCH_" + str(branch) + "_END" in content and all(
                        "BRANCH_" + str(other) + "_END" not in content for other in range(args.conversations) if other != branch)
                    if round_number == 0 and branch == 0:
                        require_cold(sample)
                    if round_number == 0:
                        originals[branch] = sample["output_sha256"]
                    else:
                        sample["output_matches_first"] = sample["output_sha256"] == originals[branch]
                        sample["tail_reused"] = sample["evaluated_prompt_tokens"] <= 16
                    report["results"].append(sample)
                    report["summary"] = summarize(report["results"])
                    atomic_report(args.output, report)
                    print(json.dumps({k: v for k, v in sample.items() if k != "output"}), flush=True)
                    if not sample["expected_branch_present"]:
                        raise RuntimeError("conversation replay answered with the wrong branch label")
                    if args.require_identical_cache_output and sample.get("output_matches_first") is False:
                        raise RuntimeError("returning conversation output differs from its first answer")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["finished_unix_seconds"] = time.time()
        atomic_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
