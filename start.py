#!/usr/bin/env python3
"""
ezlocalai startup script with precaching.

This script:
1. Runs precache.py to download all models ONCE
2. Starts uvicorn with the configured number of workers

This ensures models are only downloaded once, not per-worker.
"""

import os
import sys
import subprocess


def main():
    # Run precache first
    print("[Startup] Running precache to download models...")
    precache_result = subprocess.run(
        [sys.executable, "precache.py"], cwd=os.path.dirname(os.path.abspath(__file__))
    )

    if precache_result.returncode != 0:
        print("[Startup] WARNING: Precache had errors, continuing anyway...")

    # Get worker count from environment
    workers = os.getenv("UVICORN_WORKERS", "1")
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8091")

    print(f"[Startup] Starting uvicorn with {workers} workers on {host}:{port}")

    # Start uvicorn
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        host,
        "--port",
        port,
        "--workers",
        workers,
        "--proxy-headers",
    ]

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
