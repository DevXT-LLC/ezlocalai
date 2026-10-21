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

    # Get the registry manifest digest for deploy API verification.
    # The Deploy API (main.py) parses stdout with:
    #   re.findall(r"published and verified: .+ \(?(sha256:[a-f0-9]{64})\)?", ...)
    # so we must print in that exact format.
    #
    # IMPORTANT: We must use the registry's manifest digest (Docker-Content-Digest),
    # NOT the Docker image ID. The image ID (docker inspect --format {{.Id}}) is the
    # content hash of config+layers, which is different from the SHA256 of the
    # manifest JSON document that the registry actually stores and serves. Workers
    # pull by @sha256:<manifest_digest>, so using the image ID causes 500 errors.
    import re as _re

    # Parse the image reference to extract registry host, repo, and tag.
    # e.g., "localhost:5000/devxt/ezlocalai-cuda:latest"
    img = args.image
    if ":" in img.rsplit("/", 1)[0]:
        # Has explicit registry host with port (e.g., localhost:5000)
        registry, _, repo_tag = img.partition("/")
    else:
        # No explicit registry (docker.io implicit) - unlikely for our use case
        registry = "registry-1.docker.io"
        repo_tag = img

    if ":" in repo_tag.rsplit("/", 1)[-1]:
        repo, tag = repo_tag.rsplit(":", 1)
    else:
        repo = repo_tag
        tag = "latest"

    # Query the registry for the actual manifest digest.
    # Use HEAD request to avoid downloading the full manifest body.
    # For localhost or IP addresses, use http (LAN registry); otherwise https.
    if registry.startswith("localhost") or _re.match(r"^\d+\.\d+\.\d+\.\d+", registry):
        registry_url = f"http://{registry}"
    else:
        registry_url = f"https://{registry}"

    head_url = f"{registry_url}/v2/{repo}/manifests/{tag}"
    result = subprocess.run(
        ["curl", "-sI", "-H", "Accept: application/vnd.docker.distribution.manifest.v2+json", head_url],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"ERROR: could not query registry for manifest digest: {head_url}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract Docker-Content-Digest from response headers.
    digest = None
    for line in result.stdout.splitlines():
        if line.lower().startswith("docker-content-digest:"):
            digest = line.split(":", 1)[1].strip()
            break

    if not digest or not _re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        print(
            f"ERROR: could not extract manifest digest from registry response for {args.image}\n"
            f"Headers: {result.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"published and verified: {args.image} ({digest})", flush=True)


if __name__ == "__main__":
    main()