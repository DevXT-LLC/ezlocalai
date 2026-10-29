#!/usr/bin/env python3
"""Bounded synthetic helper load alongside LFM voice; never invokes real agents."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import threading
import time
import urllib.request
import uuid
import zlib


def synthetic_frame(index, run_id):
    """Fresh 384px red fixture; every tile varies, including across test runs."""
    size = 384
    seed = hashlib.sha256((run_id + ":" + str(index)).encode()).digest()
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            tile = (x // 16 + 7 * (y // 16)) % len(seed)
            rows.extend((255, seed[tile] & 31, seed[(tile + 13) % len(seed)] & 31))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--background-model", default="LiquidAI/LFM3-Omni-2B-GGUF#2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--voice", default="warm_american_narrator")
    parser.add_argument("--audio", default="")
    parser.add_argument(
        "--image", default="", help="Optional local PNG for helper vision requests"
    )
    parser.add_argument(
        "--synthetic-vision",
        action="store_true",
        help="Use a different synthetic red PNG on every helper request",
    )
    parser.add_argument("--interval", type=float, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-background-requests", type=int, default=60)
    parser.add_argument("--inject-context-words", type=int, default=1200)
    args = parser.parse_args()
    if args.image and args.synthetic_vision:
        parser.error("Choose --image or --synthetic-vision")
    if (
        args.interval < 0
        or not 1 <= args.max_background_requests <= 200
        or not 1 <= args.max_output_tokens <= 1024
    ):
        parser.error(
            "Use a nonnegative interval, 1..200 requests and 1..1024 output tokens"
        )
    config = json.loads(Path(args.connection).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    headers = {"Content-Type": "application/json"}
    if config.get("key"):
        headers["Authorization"] = "Bearer " + config["key"]
    image = (
        "data:image/png;base64,"
        + base64.b64encode(Path(args.image).read_bytes()).decode()
        if args.image
        else None
    )
    stopped = threading.Event()
    records = []
    run = uuid.uuid4().hex

    def load():
        while not stopped.is_set() and len(records) < args.max_background_requests:
            current_image = image
            image_hash = None
            if args.synthetic_vision:
                data = synthetic_frame(len(records), run)
                image_hash = hashlib.sha256(data).hexdigest()
                current_image = (
                    "data:image/png;base64," + base64.b64encode(data).decode()
                )
            prompt = "Synthetic helper " + run + " " + str(len(records)) + ". "
            prompt += (
                "Describe the image color in one paragraph."
                if current_image
                else "Explain a robust test and deployment workflow in detail."
            )
            content = (
                [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": current_image}},
                ]
                if current_image
                else prompt
            )
            body = {
                "model": args.background_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are reviewing Cedar. "
                        + "The review checks correctness, tests, error handling, and deployment. "
                        * 110,
                    },
                    {"role": "user", "content": content},
                ],
                "max_tokens": args.max_output_tokens,
                "temperature": 0,
                "disable_fallback": True,
            }
            if config.get("worker"):
                body["worker"] = config["worker"]
            started = time.monotonic()
            try:
                req = urllib.request.Request(
                    config["base"].rstrip("/") + "/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers=headers,
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.load(response)
                text = result["choices"][0]["message"].get("content") or ""
                records.append(
                    {
                        "status": "passed",
                        "seconds": time.monotonic() - started,
                        "usage": result.get("usage"),
                        "timings": result.get("timings"),
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "red_recognized": (
                            "red" in text.lower() if current_image else None
                        ),
                        "image_sha256": image_hash,
                    }
                )
            except Exception as error:
                records.append(
                    {
                        "status": "failed",
                        "seconds": time.monotonic() - started,
                        "error": type(error).__name__,
                    }
                )
                stopped.wait(0.5)
            stopped.wait(args.interval)

    thread = threading.Thread(target=load)
    thread.start()
    time.sleep(0.3)
    try:
        command = [
            sys.executable,
            str(Path(__file__).with_name("bench_lfm_realtime.py")),
            "--connection",
            args.connection,
            "--output",
            str(output / "voice"),
            "--rounds",
            str(args.rounds),
            "--voices",
            args.voice,
            "--tools",
            "--context-words",
            "1200",
            "--interleave-context-words",
            str(args.inject_context_words),
        ]
        if args.audio:
            command += [
                "--audio",
                args.audio,
                "--audio-context-words",
                str(args.inject_context_words),
            ]
        result = subprocess.run(command)
    finally:
        stopped.set()
        thread.join(35)
        (output / "background.json").write_text(json.dumps(records, indent=2) + "\n")
    if thread.is_alive():
        raise RuntimeError("Background request did not stop")
    if (
        result.returncode
        or not records
        or any(row["status"] != "passed" for row in records)
    ):
        raise SystemExit(
            "Concurrent benchmark failed; inspect recorded voice and background results"
        )
    print("Concurrent helper requests:", len(records), "all passed")


if __name__ == "__main__":
    main()
