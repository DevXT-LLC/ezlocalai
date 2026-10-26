"""Check text and vision continuations after real disk slot save/restore.

Use only an isolated single-slot test engine, with RAM prompt caching disabled
and an operator-owned /state-proof/ save directory. This client generates safe
basenames, makes HTTP requests only, and never reads or removes snapshot files.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.parse
import uuid

import bench_cache_workload
import bench_capacity
import bench_prefill
import bench_progression
from bench_cache_workload import (Client, append_messages, conversation_messages,
    finite_control_content, finite_control_turn, finite_json, input_tokens,
    request_payload, validate_provenance, validate_runtime, validate_sample)
from bench_capacity import red_png
from bench_prefill import require_cold
from bench_progression import atomic_report, digest, native_metrics


def launch_settings(metadata):
    values = dict(metadata.get("original_parameters") or {})
    values.update(metadata.get("overrides") or {})
    return values


def validate_launch(metadata, context):
    validate_provenance(metadata)
    values = launch_settings(metadata)
    cache = values.get("--cache-ram", values.get("-cram"))
    if type(cache) not in (str, int) or not re.fullmatch(r"[+]?0+", str(cache)):
        raise RuntimeError("caller-verified --cache-ram 0 is required so RAM cache cannot mask disk restoration")
    configured = values.get("-c", values.get("--ctx-size", metadata.get("n_ctx")))
    if type(configured) not in (str, int) or not re.fullmatch(r"[0-9]+", str(configured)) or int(configured) != context:
        raise RuntimeError("caller-verified launch metadata must retain the declared context")
    save_path = values.get("--slot-save-path")
    if not isinstance(save_path, str) or save_path.rstrip("/") != "/state-proof":
        raise RuntimeError("caller-verified slot save path must be the operator-owned /state-proof/ directory")
    return {"cache_ram_mib": 0, "declared_context": context, "slot_save_path": "/state-proof/",
            "evidence": "caller-verified immutable launch metadata; HTTP cannot verify host mount ownership"}


def snapshot_filename(kind):
    if kind not in ("text", "vision"):
        raise ValueError("unknown snapshot kind")
    return "ezl-state-" + kind + "-" + uuid.uuid4().hex + ".bin"


def validate_slot_response(response, filename, operation):
    finite_json(response)
    if type(response.get("id_slot")) is not int or response["id_slot"] != 0 or response.get("filename") != filename:
        raise RuntimeError("slot RPC did not return the requested slot and safe filename")
    token_name, byte_name, time_name = (("n_saved", "n_written", "save_ms") if operation == "save"
                                      else ("n_restored", "n_read", "restore_ms"))
    for name in (token_name, byte_name):
        if type(response.get(name)) is not int or response[name] <= 0:
            raise RuntimeError("slot RPC did not report positive token/byte counts: " + name)
    elapsed = (response.get("timings") or {}).get(time_name)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("slot RPC did not report a valid native duration")


def vision_messages(run_id, image_url):
    return [{"role": "system", "content": "[disk vision " + run_id + "] Answer the image question in one color word."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is the single solid color in this image? Reply with the color word only."},
                {"type": "image_url", "image_url": {"url": image_url}}]}]


def validate_vision(sample, expected_tokens, maximum_tokens, require_draft):
    metrics = native_metrics(sample["timings"])
    if metrics["evaluated_prompt_tokens"] + metrics["cached_prompt_tokens"] != expected_tokens:
        raise RuntimeError("vision native prompt counts differ from preflight; possible truncation")
    usage = sample.get("usage") or {}
    generated = sample["timings"]["predicted_n"]
    if usage.get("prompt_tokens") != expected_tokens or usage.get("completion_tokens") != generated or not 1 <= generated <= maximum_tokens:
        raise RuntimeError("vision usage/native token counts differ or exceed the requested budget")
    if sample.get("finish_reasons") != ["stop"] or sample["output"].get("reasoning_content"):
        raise RuntimeError("vision answer did not stop normally with reasoning disabled")
    if not re.fullmatch(r"\s*red[.!]?\s*", sample["output"]["content"], re.I):
        raise RuntimeError("vision response must identify the unambiguous color red")
    # Reuse the production benchmark's native MTP validation on a temporary
    # identity header; the actual red response/hash remains unchanged.
    identity = dict(sample)
    identity["output"] = {"content": "BRANCH_0_END CODE_910000\n" + sample["output"]["content"], "reasoning_content": ""}
    details = validate_sample(identity, expected_tokens, maximum_tokens, 0, 1, require_draft)
    return {key: value for key, value in details.items() if key != "identity_correct"} | {"color_correct": True}


def compare_continuation(reference, restored, seed_input_tokens):
    if reference["request_sha256"] != restored["request_sha256"]:
        raise RuntimeError("restored continuation request differs from the reference request")
    if reference["output_sha256"] != restored["output_sha256"]:
        raise RuntimeError("disk-restored continuation output differs from its exact reference")
    minimum_cached = max(1, seed_input_tokens - 16)
    if reference["cached_prompt_tokens"] < minimum_cached or restored["cached_prompt_tokens"] < minimum_cached:
        raise RuntimeError("continuation did not reuse the saved seed prefix; full recomputation could mask a broken restore")
    return {"request_identical": True, "output_identical": True, "saved_seed_prefix_reused": True,
            "minimum_cached_tokens": minimum_cached,
            "reference_cached_tokens": reference["cached_prompt_tokens"], "restored_cached_tokens": restored["cached_prompt_tokens"]}


def run_roundtrips(client, source, args, report, save):
    report["launch"] = validate_launch(report["metadata"], args.context)
    props = client.json("/props")
    report["runtime"] = validate_runtime(props, args.context)
    if (props.get("modalities") or {}).get("vision") is not True:
        raise RuntimeError("disk roundtrip qualification requires the vision projector loaded")
    save()

    def measure(kind, messages, tokens, branch=None, cold=False, expected_turn=None):
        payload = request_payload(messages, tokens, args.model, "disk-state:" + args.run_id + ":" + kind.split("_")[0])
        report["counts"]["input_token_preflights"] += 1
        expected = input_tokens(client, payload, args.context)
        pending = {"kind": kind, "expected_input_tokens": expected, "request_sha256": digest(payload),
                   "messages_sha256": digest(messages), "requested_generated_tokens": tokens}
        if expected_turn is not None:
            pending.update(expected_control_turn=expected_turn,
                           finite_control_expected_content=finite_control_content(branch, expected_turn))
        report["pending_request"] = pending
        save()
        report["counts"]["generation_calls"] += 1
        sample = {**client.measure(payload), **pending}
        finite_json(sample)
        report["last_response"] = sample
        save()
        if branch is None:
            details = validate_vision(sample, expected, tokens, args.require_draft_counters)
        else:
            minimum = 1 if expected_turn is not None else min(64, tokens)
            details = validate_sample(sample, expected, tokens, branch, minimum,
                                      args.require_draft_counters, expected_turn)
        sample.update(details)
        if cold:
            require_cold(sample)
        report["results"].append(sample)
        report["counts"]["recorded_samples"] = len(report["results"])
        report.pop("last_response", None)
        report.pop("pending_request", None)
        save()
        print(json.dumps({key: value for key, value in sample.items() if key not in ("output", "timings", "usage")}), flush=True)
        return sample

    def snapshot(kind):
        filename = snapshot_filename(kind)
        item = {"kind": kind, "filename": filename, "status": "save_requested"}
        report["snapshots"].append(item)
        save()
        started = time.perf_counter()
        report["counts"]["save_calls"] += 1
        item["save"] = client.json("/slots/0?action=save", {"filename": filename})
        item["save_wall_seconds"] = time.perf_counter() - started
        save()
        validate_slot_response(item["save"], filename, "save")
        item["status"] = "saved"
        save()
        return item

    def restore(item):
        item["status"] = "restore_requested"
        save()
        started = time.perf_counter()
        report["counts"]["restore_calls"] += 1
        item["restore"] = client.json("/slots/0?action=restore", {"filename": item["filename"]})
        item["restore_wall_seconds"] = time.perf_counter() - started
        save()
        validate_slot_response(item["restore"], item["filename"], "restore")
        if item["restore"]["n_restored"] != item["save"]["n_saved"]:
            raise RuntimeError("disk restore token count differs from the saved slot")
        if item["restore"]["n_read"] != item["save"]["n_written"]:
            raise RuntimeError("disk restore did not consume all bytes reported by the slot save")
        item["status"] = "restored"
        save()

    warmup = request_payload([{"role": "user", "content": "[disk-state warmup " + args.run_id + "] Reply OK."}], 8, args.model)
    report["counts"]["input_token_preflights"] += 1
    input_tokens(client, warmup, args.context)
    report["counts"]["generation_calls"] += 1
    report["counts"]["warmup_generation_calls"] += 1
    client.measure(warmup)
    save()

    seed_turn = finite_control_turn("seed") if args.finite_control else None
    continuation_turn = finite_control_turn("append", 1) if args.finite_control else None
    text = conversation_messages(source, args.prompt_chars, args.run_id + ":disk-text", 0,
                                 "independent", 90, args.finite_control)
    seed = measure("text_seed", text, args.decode_tokens, 0, True, seed_turn)
    text_file = snapshot("text")
    continuation = append_messages(text, seed, args.run_id + ":disk-text", 0, 1, args.finite_control)
    reference = measure("text_reference", continuation, args.decode_tokens, 0,
                        expected_turn=continuation_turn)
    stomp = conversation_messages(source, min(args.prompt_chars, 4000), args.run_id + ":stomp-text", 1,
                                  "independent", 20, args.finite_control)
    measure("text_stomp", stomp, args.decode_tokens if args.finite_control else 128, 1, True, seed_turn)
    restore(text_file)
    repeated = measure("text_restored", continuation, args.decode_tokens, 0,
                       expected_turn=continuation_turn)
    report["comparisons"]["text"] = compare_continuation(reference, repeated, seed["expected_input_tokens"])
    save()

    image = red_png()
    image_url = "data:image/png;base64," + base64.b64encode(image).decode()
    report["image"] = {"sha256": hashlib.sha256(image).hexdigest(), "width": 64, "height": 64}
    vision = vision_messages(args.run_id, image_url)
    seed = measure("vision_seed", vision, 8, cold=True)
    vision_file = snapshot("vision")
    continuation = [*vision, {"role": "assistant", "content": seed["output"]["content"]},
                    {"role": "user", "content": "What was the single solid color in the image? Reply with the same color word only."}]
    reference = measure("vision_reference", continuation, 8)
    stomp = conversation_messages(source, min(args.prompt_chars, 4000), args.run_id + ":stomp-vision", 2, "independent", 20)
    measure("vision_stomp", stomp, 128, 2, True)
    restore(vision_file)
    repeated = measure("vision_restored", continuation, 8)
    report["comparisons"]["vision"] = compare_continuation(reference, repeated, seed["expected_input_tokens"])
    save()
    if digest(client.json("/props")) != report["runtime"]["props_sha256"]:
        raise RuntimeError("engine properties changed during disk-state qualification")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("base_url")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--model")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run-id", default=uuid.uuid4().hex)
    parser.add_argument("--prompt-chars", type=int, default=100000)
    parser.add_argument("--decode-tokens", type=int)
    parser.add_argument("--finite-control", action="store_true",
                        help="require exact finite new-turn answers for the text disk roundtrip only")
    parser.add_argument("--require-draft-counters", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--api-key-env", default="EZLOCALAI_API_KEY")
    args = parser.parse_args(argv)
    args.decode_tokens = args.decode_tokens if args.decode_tokens is not None else (64 if args.finite_control else 256)
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        parser.error("require an HTTP(S) base URL without credentials, query or fragment")
    if args.context < 512 or not 1 <= args.prompt_chars <= args.context * 64 or not 64 <= args.decode_tokens < args.context - 32:
        parser.error("invalid fixed context, bounded source length or long decode budget")
    if args.finite_control and args.decode_tokens not in (64, 128):
        parser.error("finite control requires a text decode budget of 64 or 128")
    if not args.run_id or len(args.run_id) > 100 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("require a run ID of 1..100 characters and a finite positive timeout")
    return args


def main(argv=None):
    args = arguments(argv)
    source_bytes = args.source.read_bytes()
    source = source_bytes.decode("utf-8")
    if not source:
        raise ValueError("source must be nonempty")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    validate_launch(metadata, args.context)
    dependencies = (bench_cache_workload, bench_capacity, bench_prefill, bench_progression)
    report = {"schema_version": 1, "status": "running", "label": args.label, "run_id": args.run_id,
              "base_url": args.base_url, "metadata": metadata,
              "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "harness_dependencies_sha256": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in dependencies},
              "requested_sampling": {"temperature": 0, "seed": 42, "enable_thinking": False},
              "configuration": {"context": args.context, "prompt_chars": args.prompt_chars,
                                "decode_tokens": args.decode_tokens, "finite_control": args.finite_control,
                                "model": args.model},
              "warmup_excluded": True, "inference_retries": 0, "context_downgrade_allowed": False,
              "file_access": "HTTP slot RPC only; operator hashes/removes files in its owned snapshot directory",
              "counts_scope": "HTTP operation attempts; recorded_samples counts only validated inference samples",
              "counts": {"generation_calls": 0, "warmup_generation_calls": 0, "recorded_samples": 0,
                         "input_token_preflights": 0, "save_calls": 0, "restore_calls": 0},
              "results": [], "snapshots": [], "comparisons": {}, "started_unix_seconds": time.time()}

    def save():
        atomic_report(args.output, report)

    save()
    try:
        run_roundtrips(Client(args.base_url, os.environ.get(args.api_key_env, ""), args.timeout), source, args, report, save)
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
