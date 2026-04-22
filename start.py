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
import warnings

# Suppress SyntaxWarnings from third-party packages (e.g., pydub regex patterns)
warnings.filterwarnings("ignore", category=SyntaxWarning)


def main():
    router_mode = (os.getenv("ROUTER_MODE", "false") or "").strip().lower() == "true"

    if router_mode:
        # Router has no models to precache and uses a separate FastAPI app
        print("[ezlocalai] Starting in ROUTER mode (no local models)")
        app_target = "router_app:app"
    else:
        # Run precache first (quietly - it has its own logging)
        precache_result = subprocess.run(
            [sys.executable, "precache.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if precache_result.returncode != 0:
            print("[ezlocalai] Warning: Precache had errors, continuing anyway...")
        app_target = "app:app"

    # Get worker count from environment
    workers = os.getenv("UVICORN_WORKERS", "1")
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8091")

    # Start uvicorn
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app_target,
        "--host",
        host,
        "--port",
        port,
        "--workers",
        workers,
    ]

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
