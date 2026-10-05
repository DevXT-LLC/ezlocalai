#!/usr/bin/env python3
"""Measure first and subsequent requests directly against one worker.

Run immediately after startup to measure first-use latency. Requests use unique
text to avoid measuring cached responses. API keys come from EZLOCALAI_API_KEY
(or the local .env), never from a command-line argument or the result file.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import time
import uuid
import wave

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--text-model", help="Also benchmark this configured LLM")
    parser.add_argument("--speech-file", type=Path, help="Also transcribe this audio")
    parser.add_argument("--tts", action="store_true", help="Also synthesize WAV audio")
    args = parser.parse_args()
    if args.rounds < 1 or args.parallel < 1:
        parser.error("rounds and parallel must be positive")
    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {os.getenv('EZLOCALAI_API_KEY', '')}"}
    results = {"base_url": base, "started_at": time.time(), "requests": []}

    def resources():
        response = requests.get(base + "/v1/resources", headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def request(kind, path, **kwargs):
        started = time.perf_counter()
        response = requests.post(base + path, headers=headers, timeout=300, **kwargs)
        row = {
            "kind": kind,
            "seconds": round(time.perf_counter() - started, 4),
            "status": response.status_code,
        }
        response.raise_for_status()
        if kind == "tts":
            with wave.open(io.BytesIO(response.content)) as audio:
                row["audio_seconds"] = audio.getnframes() / audio.getframerate()
            assert row["audio_seconds"] > 0, "Empty TTS output"
            row["real_time_factor"] = row["seconds"] / row["audio_seconds"]
        else:
            data = response.json()
            row["usage"] = data.get("usage")
            row["timings"] = data.get("timings")
            if kind.startswith("embedding"):
                vector = data["data"][0]["embedding"]
                assert vector and all(math.isfinite(x) for x in vector)
                row["dimensions"] = len(vector)
                row["norm"] = math.sqrt(sum(x * x for x in vector))
                assert abs(row["norm"] - 1) < 0.01, "Unnormalized embedding"
            elif kind == "stt":
                assert data.get("text", "").strip(), "Empty transcript"
                row["text"] = data["text"]
            else:
                assert data.get("choices"), "Empty completion"
        print(json.dumps(row), flush=True)
        return row

    def embedding(repeats, kind):
        text = f"Trial {uuid.uuid4()}. " + (
            "The quick brown fox jumps over the lazy dog. " * repeats
        )
        return request(
            kind,
            "/v1/embeddings",
            json={"model": args.embedding_model, "input": text},
        )

    try:
        results["before"] = resources()
        for repeats in (1, 30, 300, 850):
            for _ in range(args.rounds):
                results["requests"].append(embedding(repeats, f"embedding-{repeats}"))
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results["requests"].extend(
                pool.map(
                    lambda _: embedding(30, "embedding-concurrent"),
                    range(args.parallel),
                )
            )
        for iteration in range(args.rounds):
            if args.text_model:
                results["requests"].append(
                    request(
                        "text",
                        "/v1/chat/completions",
                        json={
                            "model": args.text_model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"Trial {uuid.uuid4()}. Explain why the sky looks blue.",
                                }
                            ],
                            "max_tokens": 128,
                            "temperature": 0,
                        },
                    )
                )
            if args.speech_file:
                with args.speech_file.open("rb") as audio:
                    results["requests"].append(
                        request(
                            "stt",
                            "/v1/audio/transcriptions",
                            files={"file": (args.speech_file.name, audio)},
                            data={"language": "en", "response_format": "json"},
                        )
                    )
            if args.tts:
                results["requests"].append(
                    request(
                        "tts",
                        "/v1/audio/speech",
                        json={
                            "model": "tts-1",
                            "input": f"The sky is blue on this clear morning. Test number {iteration + 1}.",
                            "voice": "default",
                            "response_format": "wav",
                        },
                    )
                )
        results["after"] = resources()
    except Exception as exc:
        results["error"] = str(exc)
        raise
    finally:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
