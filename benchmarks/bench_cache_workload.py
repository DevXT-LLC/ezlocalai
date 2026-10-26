"""Measure long decode after conversation switches, rewinds and cache pressure.

Use an isolated, reserved, single-slot engine. This HTTP client owns no worker
drains, launches, memory/swap settings, model changes or deployment actions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import urllib.parse
import urllib.request
import uuid

import bench_prefill
import bench_progression
from bench_prefill import require_cold
from bench_progression import atomic_report, digest, native_metrics, sse_events, validate_metadata


def validate_provenance(value):
    allowed = {"image", "production_image", "engine_revision", "engine_version", "engine_sha256",
               "engine_binary_sha256", "model_sha256", "mmproj_sha256", "source_commit", "worker_version",
               "driver_version", "cuda_version", "platform", "label", "hostname", "gpu", "gpus", "cpu",
               "original_parameters", "overrides", "environment", "environment_sha256", "overlays",
               "n_ctx", "n_batch", "n_ubatch", "context", "cache_mib", "kv_cache_type", "mtp", "threads",
               "memory_limit_bytes", "memory_swap_limit_bytes", "swap_allowance_bytes", "cache_budget_bytes",
               "memory_limit_mib", "swap_limit_mib", "cache_ram_mib", "host_ram_mib", "host_swap_mib", "cgroup_limits"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("metadata must be a sanitized provenance object with known fields")
    validate_metadata(value)
    finite_json(value)
    if "environment" in value:
        environment = value["environment"]
        if not isinstance(environment, dict) or any(not isinstance(key, str) or not key.startswith(
                ("GGML_", "LLM_", "EZLOCALAI_LLAMA_")) for key in environment):
            raise ValueError("environment metadata permits only selected performance settings, not a raw environment")
        if any(not isinstance(item, (str, int, float, bool, type(None))) for item in environment.values()):
            raise ValueError("environment performance values must be scalar")


def finite_json(value):
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("provenance/response must contain finite JSON values") from error


def branch_identity(branch):
    return "BRANCH_" + str(branch) + "_END", "CODE_" + str(910000 + 17 * branch)


def request_payload(messages, tokens, model=None, cache_key=None):
    payload = {"messages": messages, "max_tokens": tokens, "temperature": 0,
               "seed": 42, "stream": True, "stream_options": {"include_usage": True},
               "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        payload["model"] = model
    if cache_key:
        payload["prompt_cache_key"] = cache_key
    return payload


class Client:
    def __init__(self, base, key, timeout, opener=urllib.request.urlopen):
        self.base, self.key, self.timeout, self.opener = base.rstrip("/"), key, timeout, opener

    def request(self, path, payload=None, streaming=False):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        if streaming:
            headers["Accept"] = "text/event-stream"
        return urllib.request.Request(self.base + path, headers=headers,
            data=None if payload is None else json.dumps(payload).encode())

    def json(self, path, payload=None):
        with self.opener(self.request(path, payload), timeout=self.timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict) or value.get("error"):
            raise RuntimeError("engine returned an error at " + path)
        return value

    def measure(self, payload, clock=time.perf_counter):
        request = self.request("/v1/chat/completions", payload, True)
        started = clock()
        first_event = first_output = None
        content, reasoning, finishes = [], [], []
        timings = usage = None
        done = False
        with self.opener(request, timeout=self.timeout) as response:
            if response.status != 200:
                raise RuntimeError("stream request failed with HTTP " + str(response.status))
            for record in sse_events(response):
                now = clock()
                if first_event is None:
                    first_event = now
                if record.strip() == "[DONE]":
                    done = True
                    break
                value = json.loads(record)
                if not isinstance(value, dict) or value.get("error"):
                    raise RuntimeError("inference stream returned an error")
                if value.get("truncated") is True:
                    raise RuntimeError("inference reported a truncated prompt")
                if value.get("timings") is not None:
                    timings = value["timings"]
                if value.get("usage") is not None:
                    usage = value["usage"]
                choices = value.get("choices") or []
                if len(choices) > 1:
                    raise RuntimeError("benchmark expects exactly one output choice")
                for choice in choices:
                    if choice.get("index", 0) != 0:
                        raise RuntimeError("benchmark expects exactly one output choice")
                    if choice.get("truncated") is True:
                        raise RuntimeError("inference reported a truncated prompt")
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
                        finishes.append(choice["finish_reason"])
        ended = clock()
        if not done or first_output is None:
            raise RuntimeError("incomplete inference stream or missing first output")
        if not finishes or any(value not in ("stop", "length") for value in finishes):
            raise RuntimeError("inference did not finish with stop or the requested token cap")
        output = {"content": "".join(content), "reasoning_content": "".join(reasoning)}
        return {"status": 200, "wall_seconds": ended - started,
                "ttft_seconds": first_output - started, "first_event_seconds": first_event - started,
                "timings": timings, "usage": usage, **native_metrics(timings),
                "request_sha256": digest(payload), "messages_sha256": digest(payload["messages"]),
                "output_sha256": digest(output), "output": output, "finish_reasons": finishes}


def validate_runtime(props, context):
    observed = (props.get("default_generation_settings") or {}).get("n_ctx", props.get("n_ctx"))
    padded = (context + 255) // 256 * 256
    if isinstance(observed, bool) or observed not in (context, padded):
        raise RuntimeError("effective context differs from the declared context/padding")
    if type(props.get("total_slots")) is not int or props["total_slots"] != 1 or props.get("is_sleeping"):
        raise RuntimeError("benchmark requires one awake isolated engine slot")
    return {"declared_context": context, "effective_context": observed,
            "total_slots": 1, "props_sha256": digest(props)}


def input_tokens(client, payload, context, reserve=32):
    count = client.json("/v1/chat/completions/input_tokens", payload).get("input_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RuntimeError("engine did not return a positive integer input token count")
    if count + payload["max_tokens"] + reserve > context:
        raise RuntimeError("request exceeds the unchanged context with decode/headroom reserved")
    return count


def finite_control_turn(kind, round_number=0):
    if kind in ("seed", "replay"):
        return "TURN_ALPHA"
    if kind == "append" and type(round_number) is int and round_number > 0:
        return "TURN_BRAVO_r" + str(round_number)
    if kind == "rewind":
        return "TURN_CHARLIE"
    raise ValueError("unknown finite control turn")


def finite_control_content(branch, turn):
    if not re.fullmatch(r"TURN_(?:ALPHA|CHARLIE|BRAVO_r[1-9]\d*)", turn):
        raise ValueError("invalid finite control answer")
    marker, code = branch_identity(branch)
    return marker + " " + code + "\n" + turn


def conversation_messages(source, characters, run_id, branch, prefix_mode, records,
                          finite_control=False, control_turn="TURN_ALPHA"):
    if not source or characters < 1 or records < 1:
        raise ValueError("source, source length and branch records must be nonempty")
    marker, code = branch_identity(branch)
    identity = "branch " + str(branch) + "; " if prefix_mode == "independent" else ""
    system = ("[cache-workload " + identity + "run " + run_id + "]\n"
              "Use only this branch's identity record for the marker and code on your first line, "
              "separated by one space. Do not print any other branch marker or identity code. "
              "Then answer the programming question in detail using the reference code and "
              "general programming knowledge.")
    if finite_control:
        system = ("[cache-workload " + identity + "run " + run_id + "]\n"
                  "Use only this branch's identity record for the marker and code. Follow the "
                  "latest user turn's exact two-line answer instruction, even when it changes "
                  "the preceding answer. Emit only those two lines and then stop. Do not emit "
                  "reasoning, Markdown, explanations or another branch's identity.")
    shared = (source * ((characters + len(source) - 1) // len(source)))[:characters]
    record = "This branch's marker is " + marker + " and its identity code is " + code + ".\n"
    if finite_control:
        return [{"role": "system", "content": system}, {"role": "user", "content":
            "Reference code:\n" + shared + "\nBranch identity record:\n" + record * records
            + "\nFor this turn, output exactly the following two lines and then stop. "
              "Do not add any other text, code fences or reasoning. Required answer:\n"
            + finite_control_content(branch, control_turn)}]
    return [{"role": "system", "content": system}, {"role": "user", "content":
        "Reference code:\n" + shared + "\nBranch identity record:\n" + record * records
        + "\nCopy this branch's marker and code on the first line. Then explain in at least "
          "800 words how hash maps handle collisions, with a Python example and a discussion "
          "of resource management relevant to the reference code."}]


def append_messages(base, previous, run_id, branch, round_number, finite_control=False):
    marker, code = branch_identity(branch)
    if finite_control:
        return [*base, {"role": "assistant", "content": previous["output"]["content"]},
                {"role": "user", "content": "[continuation " + run_id + " " + str(round_number)
                 + "] For this new turn, replace the previous turn's second line. Output "
                   "exactly the following two lines and then stop. Do not repeat the previous "
                   "answer or add any other text, code fences or reasoning. Required answer:\n"
                 + finite_control_content(branch, finite_control_turn("append", round_number))}]
    return [*base, {"role": "assistant", "content": previous["output"]["content"]},
            {"role": "user", "content": "[continuation " + run_id + " " + str(round_number) + "] "
             "Start with " + marker + " " + code + " on the first line. In at least 800 words, "
             "extend the explanation with deletion, resizing and concurrent access examples. "
             "Retain this branch's identity and do not mention another branch."}]


def validate_finite_control(sample, branch, expected_turn):
    expected = finite_control_content(branch, expected_turn)
    if sample["output"].get("reasoning_content"):
        raise RuntimeError("finite control emitted reasoning despite thinking being disabled")
    if sample["output"]["content"] not in (expected, expected + "\n"):
        raise RuntimeError("finite control answer differs from the exact branch and turn instruction")
    finishes = sample.get("finish_reasons")
    if not finishes or any(reason != "stop" for reason in finishes):
        raise RuntimeError("finite control must finish with stop, not the token cap")
    return {"finite_control_correct": True}


def validate_sample(sample, expected_tokens, decode_tokens, branch, minimum_generated,
                    require_draft=False, expected_turn=None):
    timings = sample["timings"]
    metrics = native_metrics(timings)
    if metrics["evaluated_prompt_tokens"] + metrics["cached_prompt_tokens"] != expected_tokens:
        raise RuntimeError("native prompt counts differ from the preflight input; possible truncation")
    usage = sample.get("usage") or {}
    if usage.get("prompt_tokens") != expected_tokens:
        raise RuntimeError("usage prompt count differs from the preflight input")
    generated = timings["predicted_n"]
    if generated < minimum_generated or generated > decode_tokens:
        raise RuntimeError("generated token count is outside the requested decode bounds")
    if usage.get("completion_tokens") != generated:
        raise RuntimeError("usage and native generated token counts differ")
    if sample["output"].get("reasoning_content"):
        raise RuntimeError("reasoning was emitted despite thinking being disabled")
    content = sample["output"]["content"]
    marker, code = branch_identity(branch)
    first = next((line.strip() for line in content.splitlines() if line.strip() and not re.fullmatch(
        r"(?:`{3,}|~{3,})(?:[A-Za-z][A-Za-z0-9_+-]*)?", line.strip())), "")
    if not re.match(r"^\W*" + re.escape(marker) + r"\b", first) or not re.search(r"\b" + code + r"\b", first):
        raise RuntimeError("response first meaningful line does not preserve its branch marker and identity code")
    if any(value != marker for value in re.findall(r"\bBRANCH_\d+_END\b", content)) or any(
            value != code for value in re.findall(r"\bCODE_\d+\b", content)):
        raise RuntimeError("response contains another conversation's identity")
    control = validate_finite_control(sample, branch, expected_turn) if expected_turn is not None else {}
    draft = {}
    names = ("draft_n", "draft_n_accepted")
    present = [name in timings for name in names]
    if (any(present) or require_draft) and not all(present):
        raise RuntimeError("both native MTP draft counters are required")
    if all(present):
        for name in names:
            value = timings[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value:
                raise RuntimeError("invalid native MTP draft counter: " + name)
        if timings["draft_n_accepted"] > timings["draft_n"]:
            raise RuntimeError("accepted MTP draft tokens exceed generated draft tokens")
        draft = {"draft_tokens": timings["draft_n"], "accepted_draft_tokens": timings["draft_n_accepted"],
                 "draft_acceptance_ratio": timings["draft_n_accepted"] / timings["draft_n"] if timings["draft_n"] else None}
    cached, evaluated = metrics["cached_prompt_tokens"], metrics["evaluated_prompt_tokens"]
    outcome = "tail_reuse" if cached > 0 and evaluated <= 16 else "prefix_reuse" if cached >= 100 else "full_reevaluation"
    return {"identity_correct": True, "generated_tokens": generated,
            "generated_budget_reached": generated == decode_tokens, "cache_outcome": outcome,
            "tail_reused": outcome == "tail_reuse", **draft, **control}


def summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["kind"], []).append(row)
    result = {}
    for kind, values in grouped.items():
        item = {"samples": len(values), "cache_outcomes": {name: sum(v["cache_outcome"] == name for v in values)
                for name in ("tail_reuse", "prefix_reuse", "full_reevaluation")}}
        for name in ("wall_seconds", "ttft_seconds", "decode_tokens_per_second", "prefill_tokens_per_second", "generated_tokens"):
            item[name + "_median"] = statistics.median(v[name] for v in values)
        drafts = [v for v in values if "draft_tokens" in v]
        if drafts:
            total = sum(v["draft_tokens"] for v in drafts)
            accepted = sum(v["accepted_draft_tokens"] for v in drafts)
            item.update(draft_tokens=total, accepted_draft_tokens=accepted,
                        draft_acceptance_ratio=accepted / total if total else None)
        result[kind] = item
    return result


def run_workload(client, source, args, report, save):
    props = client.json("/props")
    report["runtime"] = validate_runtime(props, args.context)
    save()
    seeds = [conversation_messages(source, args.prompt_chars, args.run_id, branch,
             args.prefix_mode, args.branch_records, args.finite_control) for branch in range(args.conversations)]
    rewinds = [conversation_messages(source, args.prompt_chars, args.run_id, branch,
               args.prefix_mode, args.rewind_records, args.finite_control,
               "TURN_CHARLIE") for branch in range(args.conversations)]
    seed_outputs, histories = {}, {}

    def record(kind, branch, round_number, messages, dependency=None):
        cache_key = "cache-workload:" + args.run_id + ":" + str(branch)
        payload = request_payload(messages, args.decode_tokens, args.model, cache_key)
        expected = input_tokens(client, payload, args.context)
        sample_id = kind + ":r" + str(round_number) + ":b" + str(branch)
        fields = {"sample_id": sample_id, "kind": kind, "branch": branch, "round": round_number,
                  "expected_input_tokens": expected, "declared_context": args.context,
                  "remaining_context_tokens": args.context - expected, "request_sha256": digest(payload),
                  "messages_sha256": digest(messages), "model_output_dependent_request": dependency is not None}
        expected_turn = finite_control_turn(kind, round_number) if args.finite_control else None
        if expected_turn is not None:
            fields.update(expected_control_turn=expected_turn,
                          finite_control_expected_content=finite_control_content(branch, expected_turn))
        if dependency is not None:
            fields.update(history_dependency_sample_id=dependency["sample_id"],
                          history_dependency_output_sha256=dependency["output_sha256"])
        report["pending_request"] = fields
        save()
        sample = {**client.measure(payload), **fields}
        finite_json(sample)
        report["last_response"] = sample
        save()
        sample.update(validate_sample(sample, expected, args.decode_tokens, branch,
                      args.minimum_generated, args.require_draft_counters, expected_turn))
        if kind == "seed" and (branch == 0 or args.prefix_mode == "independent"):
            require_cold(sample)
        if kind == "replay":
            sample["output_matches_seed"] = sample["output_sha256"] == seed_outputs[branch]["output_sha256"]
            sample["possible_eviction_or_unavailable_state"] = sample["cache_outcome"] == "full_reevaluation"
        report["results"].append(sample)
        report["summary"] = summary(report["results"])
        report.pop("last_response", None)
        report.pop("pending_request", None)
        save()
        print(json.dumps({k: v for k, v in sample.items() if k not in ("output", "usage", "timings")}), flush=True)
        return sample

    # Preflight every deterministic prompt before any generation. Appended
    # histories are checked once their actual assistant response is available.
    for branch, messages in enumerate(seeds):
        input_tokens(client, request_payload(messages, args.decode_tokens, args.model), args.context)
        if "rewind" in args.cases:
            input_tokens(client, request_payload(rewinds[branch], args.decode_tokens, args.model), args.context)
    warmup = request_payload([{"role": "user", "content": "[cache-workload warmup " + args.run_id + "] Reply OK."}], 8, args.model)
    input_tokens(client, warmup, args.context)
    client.measure(warmup)
    for branch in range(args.conversations):
        seed_outputs[branch] = record("seed", branch, 0, seeds[branch])
        histories[branch] = (seeds[branch], seed_outputs[branch])
    for round_number in range(1, args.rounds):
        for kind in args.cases:
            for branch in range(args.conversations):
                dependency = None
                if kind == "append":
                    base, dependency = histories[branch]
                    messages = append_messages(base, dependency, args.run_id, branch, round_number,
                                               args.finite_control)
                else:
                    messages = seeds[branch] if kind == "replay" else rewinds[branch]
                sample = record(kind, branch, round_number, messages, dependency)
                histories[branch] = (messages, sample)
    if digest(client.json("/props")) != report["runtime"]["props_sha256"]:
        raise RuntimeError("engine properties changed during the workload")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--model")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--mode", choices=("resident", "pressure"), default="resident")
    parser.add_argument("--prefix-mode", choices=("common", "independent"))
    parser.add_argument("--conversations", type=int)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=("replay", "append", "rewind"))
    parser.add_argument("--prompt-chars", type=int, default=100000)
    parser.add_argument("--branch-records", type=int, default=90)
    parser.add_argument("--rewind-records", type=int, default=40)
    parser.add_argument("--decode-tokens", type=int)
    parser.add_argument("--minimum-generated", type=int)
    parser.add_argument("--finite-control", action="store_true",
                        help="verify exact finite turn instructions instead of long freeform decode")
    parser.add_argument("--require-draft-counters", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--api-key-env", default="EZLOCALAI_API_KEY")
    args = parser.parse_args(argv)
    args.decode_tokens = args.decode_tokens if args.decode_tokens is not None else (64 if args.finite_control else 256)
    args.minimum_generated = args.minimum_generated if args.minimum_generated is not None else (1 if args.finite_control else 64)
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        parser.error("require an HTTP(S) base URL without credentials, query or fragment")
    args.conversations = args.conversations if args.conversations is not None else (3 if args.mode == "resident" else 8)
    args.prefix_mode = args.prefix_mode or ("common" if args.mode == "resident" else "independent")
    args.cases = args.cases or (["replay", "append", "rewind"] if args.mode == "resident" else ["replay"])
    if args.context < 512 or args.conversations < 2 or args.rounds < 2 or args.prompt_chars < 1 or args.prompt_chars > args.context * 64:
        parser.error("require context>=512, at least two conversations/rounds and a bounded positive source length")
    if not 1 <= args.rewind_records < args.branch_records <= args.context or len(args.cases) != len(set(args.cases)):
        parser.error("require distinct cases and 1 <= rewind records < branch records <= context")
    if not 1 <= args.minimum_generated <= args.decode_tokens or args.decode_tokens + 32 >= args.context:
        parser.error("invalid decode count/minimum or insufficient fixed-context headroom")
    if args.finite_control and (args.decode_tokens not in (64, 128) or args.minimum_generated != 1):
        parser.error("finite control requires a decode budget of 64 or 128 and minimum-generated 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not args.run_id or len(args.run_id) > 128:
        parser.error("require a finite positive timeout and run ID of 1..128 characters")
    return args


def main(argv=None):
    args = arguments(argv)
    source_bytes = args.source.read_bytes()
    source = source_bytes.decode("utf-8")
    if not source:
        raise ValueError("source must be nonempty")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    validate_provenance(metadata)
    report = {"schema_version": 1, "status": "running", "label": args.label,
              "base_url": args.base_url, "run_id": args.run_id, "metadata": metadata,
              "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "harness_dependencies_sha256": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for module in (bench_prefill, bench_progression)},
              "configuration": {key: getattr(args, key) for key in ("mode", "prefix_mode", "conversations", "rounds", "cases", "context", "prompt_chars", "branch_records", "rewind_records", "decode_tokens", "minimum_generated", "finite_control", "require_draft_counters", "model")},
              "requested_sampling": {"temperature": 0, "seed": 42, "enable_thinking": False},
              "warmup_excluded": True, "inference_retries": 0, "context_downgrade_allowed": False,
              "notes": ["Native timings supply token rates; SSE chunks are never token counts.",
                        "Request encoding and token-count preflight are excluded from HTTP wall time/TTFT.",
                        "Append requests depend on previous generated text; compare request hashes before pairing variants.",
                        "Full re-evaluation under pressure is an observed miss, not proof of a specific eviction cause.",
                        "Memory and swap telemetry require the caller's external cgroup measurements."],
              "started_unix_seconds": time.time(), "results": []}

    def save():
        atomic_report(args.output, report)

    save()
    try:
        run_workload(Client(args.base_url, os.environ.get(args.api_key_env, ""), args.timeout), source, args, report, save)
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["ended_unix_seconds"] = time.time()
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
