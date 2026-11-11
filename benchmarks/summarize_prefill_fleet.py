"""Reduce completed fleet probes to reviewable evidence, without prompt content."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    provenance = next(row for row in rows if row["event"] == "provenance")
    restored = [row for row in rows if row["event"] == "restored"]
    if not restored or rows[-1]["event"] != "restored":
        raise ValueError(f"{path.name}: no final restoration receipt")
    samples = []
    for row in rows:
        if row["event"] != "sample":
            continue
        timing = row["timings"]
        samples.append(
            {
                key: row.get(key)
                for key in (
                    "block",
                    "variant",
                    "task",
                    "target_tokens",
                    "repeat",
                    "kind",
                    "ttft_seconds",
                    "wall_seconds",
                    "finish_reason",
                    "quality_pass",
                    "prompt_sha256",
                    "output_sha256",
                )
            }
            | {"timings": timing}
        )
    return {
        "source": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": provenance,
        "ready": [row for row in rows if row["event"] == "ready"],
        "restored": restored[-1],
        "sample_count": len(samples),
        "quality_passes": sum(s["quality_pass"] is True for s in samples),
        "quality_failures": sum(s["quality_pass"] is False for s in samples),
        "externally_graded_samples": sum(s["quality_pass"] is None for s in samples),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "date": "2026-11-11",
        "scope": "Controlled prefill screening, not general model-quality certification",
        "production_defaults_changed": False,
        "experiments": [summarize(path) for path in args.inputs],
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
