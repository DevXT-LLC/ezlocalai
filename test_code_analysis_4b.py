#!/usr/bin/env python3
"""Benchmark 4B model with full code analysis - Dynamic Context Version"""
import requests
import time
import sys


def main():
    """Run the opt-in live 4B benchmark."""
    with open("Pipes.py", "r") as f:
        code_content = f.read()

    print(
        f"Code size: {len(code_content):,} chars, "
        f"~{len(code_content)//4:,} tokens estimate"
    )
    prompt = f"""Here is the complete Pipes.py code from ezlocalai:

```python
{code_content}
```

Analyze this code and explain:
1. How does the GPU layer auto-calibration work?
2. What is the fallback strategy if native estimation fails?
3. How does the multi-model hot-swap work?

Be concise but thorough."""

    print(f"Prompt size: {len(prompt):,} chars")
    print("\nTesting 4B model with full code analysis...")
    print("(Dynamic context - will auto-size to fit prompt)")
    print("(4B has 37/37 GPU layers at 32k context)\n")
    sys.stdout.flush()

    start = time.time()
    resp = requests.post(
        "http://localhost:8091/v1/chat/completions",
        json={
            "model": "unsloth/Qwen3-VL-4B-Instruct-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        timeout=600,
    )
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"Error: {resp.status_code}")
        print(resp.text[:1000])
        return

    data = resp.json()
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    content = data["choices"][0]["message"]["content"]
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    print("=== 4B Model with Dynamic Context ===")
    print(f"Prompt tokens: {prompt_tokens:,}")
    print(f"Completion tokens: {completion_tokens:,}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Generation speed: {completion_tokens/elapsed:.1f} tok/s overall")
    if timings:
        print("\nLLM Timings:")
        print(f"  Prompt processing: {timings.get('prompt_per_second', 0):.1f} tok/s")
        print(f"  Generation: {timings.get('predicted_per_second', 0):.1f} tok/s")
    print(f"\n{'='*60}")
    print("RESPONSE:")
    print(f"{'='*60}\n")
    print(content)


if __name__ == "__main__":
    main()
