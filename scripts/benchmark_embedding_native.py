#!/usr/bin/env python3
"""Compare embedding microbatch/thread settings inside the GPU worker image.

Run with the serving worker idle. Each candidate gets its own subprocess so
native model memory is released before the next candidate. Does not edit .env.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Support invocation as python scripts/benchmark_embedding_native.py in /app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="512,1024,2048")
    parser.add_argument("--threads", default="4,10,20")
    parser.add_argument("--candidate", nargs=2, type=int)
    args = parser.parse_args()
    if not args.candidate:
        for batch in args.batches.split(","):
            for threads in args.threads.split(","):
                subprocess.run(
                    [sys.executable, __file__, "--candidate", batch, threads],
                    check=True,
                )
        return

    batch, threads = args.candidate
    if batch < 1 or threads < 1:
        parser.error("batch and threads must be positive")
    os.environ.update(
        EMBEDDING_BATCH_SIZE=str(batch),
        EMBEDDING_UBATCH_SIZE=str(batch),
        EMBEDDING_N_PARALLEL="1",
        EMBEDDING_KEEP_LOADED="true",
        EMBEDDING_WARMUP="true",
    )
    from ezlocalai.Embedding import Embedding

    class Candidate(Embedding):
        def _build_params(self, *args):
            params = super()._build_params(*args)
            params.cpuparams.n_threads = threads
            params.cpuparams_batch.n_threads = threads
            return params

    embedder = Candidate()
    for repeats in (1, 30, 300, 850):
        times = []
        for trial in range(4):
            text = (
                f"Trial {trial}. "
                + "The quick brown fox jumps over the lazy dog. " * repeats
            )
            start = time.perf_counter()
            result = embedder.get_embeddings(text)
            times.append(time.perf_counter() - start)
        print(
            json.dumps(
                {
                    "batch": batch,
                    "threads": threads,
                    "tokens": result["usage"]["prompt_tokens"],
                    "first_seconds": times[0],
                    "median_warm_seconds": statistics.median(times[1:]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
