#!/usr/bin/env python3
"""Benchmark ezlocalai model load/unload transitions through its public API.

Configure the worker normally, then alternate configured LLM names with:

    python benchmark_model_lifecycle.py \
      --models model-a,model-b --rounds 2

The worker records model-only lifecycle timings in ``/v1/resources``. Request
wall time is printed separately so generation and HTTP overhead are not confused
with model loading. Pass ``--image`` to include one image handoff when image
generation is enabled on the worker.
"""

import argparse
import os
import time
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


load_dotenv()


def _headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _json(response: requests.Response) -> Dict[str, Any]:
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8091")
    parser.add_argument("--api-key", default=os.getenv("EZLOCALAI_API_KEY", "none"))
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated configured LLMs. Defaults to /v1/models output.",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--image",
        action="store_true",
        help="Also issue one 512x512 image request to benchmark the LLM handoff.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = _headers(args.api_key)
    configured = [part.strip() for part in args.models.split(",") if part.strip()]
    if not configured:
        model_data = _json(
            requests.get(f"{base_url}/v1/models", headers=headers, timeout=30)
        )
        configured = [
            item["id"]
            for item in model_data.get("data", [])
            if item.get("id") and not item.get("capability")
        ]
    if not configured:
        raise SystemExit("No configured LLMs were found")

    started_at = time.time()
    wall_results: List[Dict[str, Any]] = []
    sequence = configured * max(1, args.rounds)
    for model in sequence:
        request_started = time.perf_counter()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": max(1, args.max_tokens),
                "temperature": 0,
                "stream": False,
            },
            timeout=1800,
        )
        elapsed = time.perf_counter() - request_started
        wall_results.append(
            {
                "kind": "chat",
                "model": model,
                "seconds": elapsed,
                "status": response.status_code,
            }
        )
        response.raise_for_status()

    if args.image:
        request_started = time.perf_counter()
        response = requests.post(
            f"{base_url}/v1/images/generations",
            headers=headers,
            json={
                "prompt": "A small brass stopwatch on a clean workbench",
                "size": "512x512",
                "n": 1,
            },
            timeout=1800,
        )
        wall_results.append(
            {
                "kind": "image",
                "model": "configured image model",
                "seconds": time.perf_counter() - request_started,
                "status": response.status_code,
            }
        )
        response.raise_for_status()

    resources = _json(
        requests.get(f"{base_url}/v1/resources", headers=headers, timeout=30)
    )
    lifecycle = resources.get("model_lifecycle", {})
    events = [
        event
        for event in lifecycle.get("recent_events", [])
        if float(event.get("timestamp", 0)) >= started_at
    ]

    print(
        "Residency:",
        lifecycle.get("llm_model_residency", "unknown"),
        "| loaded:",
        lifecycle.get("loaded_llms", []),
    )
    print("\nRequest wall time (includes inference):")
    for result in wall_results:
        print(
            f"  {result['kind']:5} {result['model']}: "
            f"{result['seconds']:.3f}s (HTTP {result['status']})"
        )
    print("\nModel lifecycle time:")
    if not events:
        print(
            "  No load/unload occurred during this run (models were already resident)."
        )
    for event in events:
        status = "ok" if event.get("success") else "failed"
        print(
            f"  {event.get('model_type'):5} {event.get('operation'):6} "
            f"{event.get('model')}: {float(event.get('duration_seconds', 0)):.3f}s "
            f"({status})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
