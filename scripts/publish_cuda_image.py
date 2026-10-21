#!/usr/bin/env python3
"""Build and publish the ezlocalai CUDA Docker image to the LAN registry.

Called by the Deploy API pipeline after fast-forwarding the build checkout
to origin/main.  Accepts:

    --image <registry/repo:tag>   Target image reference (required)
    --variant <cuda|rocm|...>     Build variant (default: cuda)

The script builds via docker compose using the matching
docker-compose-<variant>.yml, then pushes the resulting image.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict | None = None, cwd: str | None = None) -> None:
    """Run a command, streaming output and raising on failure."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {result.returncode}: {' '.join(cmd)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and publish the ezlocalai CUDA image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Target image reference, e.g. localhost:5000/devxt/ezlocalai-cuda:latest",
    )
    parser.add_argument(
        "--variant",
        default="cuda",
        choices=["cuda", "rocm", "jetson"],
        help="Build variant (default: cuda)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    compose_file = f"docker-compose-{args.variant}.yml"
    compose_path = repo_root / compose_file

    if not compose_path.is_file():
        print(f"ERROR: {compose_path} not found", file=sys.stderr)
        sys.exit(1)

    # The compose file reads EZLOCALAI_CUDA_IMAGE (or equivalent) for the
    # service image tag.  We set it so `docker compose build` produces an
    # image with exactly the reference we want to push.
    env = os.environ.copy()
    if args.variant == "cuda":
        env["EZLOCALAI_CUDA_IMAGE"] = args.image
    elif args.variant == "rocm":
        env["EZLOCALAI_ROCM_IMAGE"] = args.image
    elif args.variant == "jetson":
        env["EZLOCALAI_JETSON_IMAGE"] = args.image

    # Build
    print(f"Building {args.variant} image: {args.image}", flush=True)
    run(
        [
            "docker",
            "compose",
            "-f",
            compose_file,
            "build",
            "ezlocalai",
        ],
        env=env,
        cwd=str(repo_root),
    )

    # Push
    print(f"Pushing {args.image}", flush=True)
    run(["docker", "push", args.image])

    # Get the image digest for deploy API verification.
    # The Deploy API (main.py) parses stdout with:
    #   re.findall(r"published and verified: .+ \(?(sha256:[a-f0-9]{64})\)?", ...)
    # so we must print in that exact format.
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", args.image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(
            f"ERROR: could not verify image after push: {args.image}",
            file=sys.stderr,
        )
        sys.exit(1)

    digest = result.stdout.strip()
    print(f"published and verified: {args.image} ({digest})", flush=True)


if __name__ == "__main__":
    main()