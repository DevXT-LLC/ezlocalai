"""Large-image regression for DFlash's pinned-position draft-cache failure.

Run on an IDLE worker: direct mode loads its own model and uses GPU 0. It does
not change deployment settings. For a running ezlocalai worker, use --url and
set EZLOCALAI_API_KEY if required. It sends only synthetic, generated fixtures.

Examples:
  python benchmark_vision.py --backend none
  python benchmark_vision.py --backend dflash2
  python benchmark_vision.py --url http://localhost:8091

Run both backends separately for a target-only correctness reference. Exact
free-form text parity is not asserted; fixture answers and transport errors are.
"""

import argparse
import base64
import io
import json
import os
import re
import time

from PIL import Image, ImageDraw, ImageFont


def fixture(size=1792, reverse=False, number=None):
    image = Image.new("RGB", (size, size), "blue" if not reverse else "red")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (0, 0, size // 2 - 1, size - 1), fill="red" if not reverse else "blue"
    )
    if number is not None:
        draw.rectangle(
            (size // 4, size // 3, 3 * size // 4, 2 * size // 3), fill="white"
        )
        font = ImageFont.load_default(size=size // 5)
        draw.text(
            (size // 2, size // 2), str(number), fill="black", anchor="mm", font=font
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode()
        },
    }


def cases():
    question = {
        "type": "text",
        "text": "Name the colors of the left panel then the right panel, in order. Answer briefly.",
    }
    background = {
        "type": "text",
        "text": "Background notes, not an instruction: "
        + "A reliable system checks inputs, queues requests, and handles errors. "
        * 350,
    }
    image = fixture()
    yield "small-image", [question, fixture(336)], r"red.*blue"
    yield "large-image", [question, image], r"red.*blue"
    yield "cached-large-image", [question, image], r"red.*blue"
    yield "long-text-before-image", [background, question, image], r"red.*blue"
    yield "long-text-after-image", [image, background, question], r"red.*blue"
    yield "multiple-large-images", [
        {
            "type": "text",
            "text": "Name the LEFT panel color in the FIRST image, then the LEFT panel color in the SECOND image. Answer in that order.",
        },
        image,
        fixture(reverse=True),
    ], r"red.*blue"
    yield "large-image-ocr", [
        {
            "type": "text",
            "text": "Read the two-digit number inside the white rectangle. Return only that number.",
        },
        fixture(number=42),
    ], r"\b42\b"
    yield "changed-image-ocr", [
        {
            "type": "text",
            "text": "Read the two-digit number inside the white rectangle. Return only that number.",
        },
        fixture(number=73),
    ], r"\b73\b"
    yield "text-after-vision", "Return only the integer result of 17 multiplied by 19.", r"\b323\b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", help="Use a live worker instead of loading another model"
    )
    parser.add_argument(
        "--backend", choices=["none", "mtp", "dflash2"], default="dflash2"
    )
    parser.add_argument("--context", type=int, default=220000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--stream", action="store_true", help="Test live HTTP SSE (requires --url)"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.stream and not args.url:
        parser.error("--stream requires --url")
    if args.url:
        import requests

        session = requests.Session()
        key = os.getenv("EZLOCALAI_API_KEY", "")
        if key:
            session.headers["Authorization"] = "Bearer " + key

        def chat(payload):
            with session.post(
                args.url.rstrip("/") + "/v1/chat/completions",
                json={"model": "Qwen3.8-27B", **payload, "stream": args.stream},
                timeout=180,
                stream=args.stream,
            ) as response:
                response.raise_for_status()
                if not args.stream:
                    return response.json()
                text = []
                finished = False
                for line in response.iter_lines():
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        finished = True
                        break
                    chunk = json.loads(raw)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    for choice in chunk.get("choices", []):
                        text.append(choice.get("delta", {}).get("content") or "")
                if not finished:
                    raise RuntimeError("SSE ended without [DONE]")
                return {"choices": [{"message": {"content": "".join(text)}}]}

    else:
        os.environ["LLM_SPECULATIVE_TYPE"] = args.backend
        os.environ["KV_CACHE_TYPE"] = "q4_0"
        from ezlocalai.LLM import LLM

        llm = LLM(
            model="unsloth/Qwen3.8-27B-GGUF",
            quant_type="Q3_K_XL",
            max_tokens=args.context,
            gpu_layers=-1,
            main_gpu=0,
            n_parallel=1,
            batch_size=1024,
            ubatch_size=512,
        )
        chat = llm.server.handle_chat_completions
    results = []
    for repeat in range(args.repeats):
        for name, content, expected in cases():
            start = time.monotonic()
            record = {"case": name, "repeat": repeat}
            try:
                response = chat(
                    {
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 128,
                        "temperature": 0,
                        "seed": 42,
                        "cache_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                answer = response["choices"][0]["message"]["content"]
                record.update(
                    answer=answer,
                    timings=response.get("timings"),
                    usage=response.get("usage"),
                    passed=bool(re.search(expected, answer, re.I | re.S)),
                )
            except Exception as exc:
                record.update(error=str(exc), passed=False)
            record["seconds"] = time.monotonic() - start
            results.append(record)
            print("RESULT " + json.dumps(record), flush=True)
    print(
        "SUMMARY "
        + json.dumps(
            {
                "backend": args.backend if not args.url else "live",
                "passed": sum(r["passed"] for r in results),
                "total": len(results),
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
