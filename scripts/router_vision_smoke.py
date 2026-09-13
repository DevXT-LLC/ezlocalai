"""After deploying worker targeting, test each provider with a synthetic red PNG.

Uses EZLOCALAI_API_KEY from the environment; no third-party dependencies.
Each provider receives one ordinary and one streaming vision request.
"""

import argparse
import base64
import json
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib


def red_image_url():
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64))
    png += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def vision_payload(worker, model, stream):
    return {
        "worker": worker,
        "model": model,
        "stream": stream,
        "max_tokens": 32,
        "reasoning": {"enabled": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the main color in this image? Reply with one color word.",
                    },
                    {"type": "image_url", "image_url": {"url": red_image_url()}},
                ],
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://api.ezlocal.ai")
    parser.add_argument(
        "--worker", action="append", help="Worker label; repeat to test several"
    )
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("EZLOCALAI_API_KEY", "")
    if key and key != "none":
        headers["Authorization"] = f"Bearer {key}"
    base = args.url.rstrip("/")
    request = urllib.request.Request(base + "/v1/router/workers", headers=headers)
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        workers = json.load(response)["data"]
    failures = 0
    for label in args.worker or ["Chutes.ai", "OpenRouter.ai"]:
        matches = [w for w in workers if w["label"].casefold() == label.casefold()]
        if len(matches) != 1 or not matches[0].get("models"):
            print(f"FAIL {label}: expected one registered worker with models")
            failures += 1
            continue
        for stream in (False, True):
            mode = "streaming" if stream else "non-streaming"
            payload = vision_payload(label, matches[0]["models"][0], stream)
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    if stream:
                        answer = ""
                        for line in response:
                            if not line.startswith(b"data:"):
                                continue
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                break
                            event = json.loads(data)
                            if "error" in event:
                                raise ValueError(str(event["error"]))
                            for choice in event.get("choices", []):
                                answer += choice.get("delta", {}).get("content") or ""
                    else:
                        result = json.load(response)
                        if "error" in result:
                            raise ValueError(str(result["error"]))
                        answer = result["choices"][0]["message"]["content"]
                if "red" not in answer.lower():
                    raise ValueError(f"expected red, received {answer!r}")
                print(f"PASS {label} {mode}: {answer.strip()}")
            except urllib.error.HTTPError as exc:
                print(
                    f"FAIL {label} {mode}: HTTP {exc.code}: {exc.read().decode()[:500]}"
                )
                failures += 1
            except (
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                print(f"FAIL {label} {mode}: {exc}")
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
