#!/usr/bin/env python3
"""Compare warm replica configurations using fixed-size direct-worker bursts.

Run once per deployed replica configuration. Random prefixes avoid embedding
prefix reuse; TTS suffixes avoid cached audio. Credentials are read from EZLOCALAI_API_KEY or .env and are
never included in output. Keep other GPU traffic idle for comparable results.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import statistics
import time
import uuid
import wave

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--speech-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {os.getenv('EZLOCALAI_API_KEY', '')}"}
    audio = args.speech_file.read_bytes()

    def resources():
        response = requests.get(base + "/v1/resources", headers=headers, timeout=30)
        response.raise_for_status()
        body = response.json()
        return {
            "slots": body["slots"]["cap_slots"],
            "model_pools": body.get("model_pools", {}),
            "devices": {
                key: value["device"] for key, value in body["loaded_models"].items()
            },
            "available_ram_gb": body["fallback"]["free_ram_gb"],
            "cuda_free_gb": body["free_vram_gb"],
        }

    results = {"label": args.label, "before": resources(), "bursts": []}

    def request(kind):
        suffix = uuid.uuid4().hex[:8]
        if kind == "tts":
            path = "/v1/audio/speech"
            kwargs = {
                "json": {
                    "model": "tts-1",
                    "voice": "default",
                    "response_format": "wav",
                    "input": "The weather is pleasant today. Test code " + suffix,
                }
            }
        elif kind == "stt":
            path = "/v1/audio/transcriptions"
            kwargs = {
                "files": {"file": ("sample.wav", audio, "audio/wav")},
                "data": {"model": "whisper-1"},
            }
        else:
            path = "/v1/embeddings"
            repetitions = 30 if kind == "embedding_short" else 300
            kwargs = {
                "json": {
                    "model": "Qwen3-Embedding-0.6B",
                    "input": suffix
                    + " "
                    + (
                        "Embedding speed depends on token count and available compute. "
                        * repetitions
                    ),
                }
            }
        started = time.perf_counter()
        response = requests.post(base + path, headers=headers, timeout=300, **kwargs)
        row = {
            "seconds": time.perf_counter() - started,
            "status": response.status_code,
            "input_id": suffix,
        }
        response.raise_for_status()
        if kind == "tts":
            with wave.open(io.BytesIO(response.content)) as wav:
                row["audio_seconds"] = wav.getnframes() / wav.getframerate()
        elif kind == "stt":
            row["text"] = response.json()["text"]
        else:
            row["dimensions"] = len(response.json()["data"][0]["embedding"])
        return row

    try:
        for kind in ("embedding_short", "embedding_long", "stt", "tts"):
            for concurrency in (1, 2, 4):
                for round_index in range(args.rounds):
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(pool.map(lambda _: request(kind), range(4)))
                    burst = {
                        "kind": kind,
                        "concurrency": concurrency,
                        "round": round_index,
                        "seconds": time.perf_counter() - started,
                        "requests": rows,
                    }
                    results["bursts"].append(burst)
                    print(
                        f"{kind} concurrency={concurrency} round={round_index}: burst={burst['seconds']:.3f}s median={statistics.median(row['seconds'] for row in rows):.3f}s",
                        flush=True,
                    )
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
        results["after"] = resources()
        if results["before"]["model_pools"] != results["after"]["model_pools"]:
            results["error"] = (
                "Replica count/device changed during benchmark; exclude these timings from a like-for-like comparison."
            )
            raise RuntimeError(results["error"])
    finally:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
