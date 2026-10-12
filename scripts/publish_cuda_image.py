#!/usr/bin/env python3
"""Build CUDA or GB10 on the registry host and retain only latest per image.

Uses Python's standard library and the host Docker CLI. Registry manifest
removal must be enabled; the registry's offline maintenance job reclaims the
unreferenced blobs. Never run garbage collection against a writable registry.
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_VARIANTS = {
    "cuda": ("cuda.Dockerfile", "linux/amd64"),
    "gb10": ("gb10.Dockerfile", "linux/arm64"),
}
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ]
)


class Registry:
    def __init__(self, image, scheme="http"):
        match = re.fullmatch(r"([^/]+)/([a-z0-9._/-]+):latest", image)
        if not match:
            raise ValueError("Image must include a registry/repository and use :latest")
        host, repository = match.groups()
        self.base = f"{scheme}://{host}/v2/{repository}"

    def request(self, suffix, method="GET", missing_ok=False):
        request = urllib.request.Request(
            self.base + suffix, method=method, headers={"Accept": MANIFEST_ACCEPT}
        )
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                exc.close()
                return None
            raise

    def digest(self, reference):
        response = self.request(
            "/manifests/" + urllib.parse.quote(reference, safe=":"),
            method="HEAD",
            missing_ok=True,
        )
        if response is None:
            return None
        with response:
            digest = response.headers.get("Docker-Content-Digest", "")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            raise RuntimeError("Registry did not return a valid manifest digest")
        return digest

    def tagged_digests(self):
        digests = set()
        # Explicit pagination avoids missing old tags in repositories with history.
        last = ""
        while True:
            response = self.request(
                "/tags/list?n=100&last=" + urllib.parse.quote(last, safe=""),
                missing_ok=True,
            )
            if response is None:
                return digests
            with response:
                tags = json.load(response).get("tags") or []
            for tag in tags:
                digest = self.digest(tag)
                if digest:
                    digests.add(digest)
            if len(tags) < 100:
                return digests
            if tags[-1] == last:
                raise RuntimeError("Registry tag pagination did not advance")
            last = tags[-1]

    def delete(self, digest):
        response = self.request("/manifests/" + digest, "DELETE", missing_ok=True)
        if response:
            with response:
                if response.status != 202:
                    raise RuntimeError("Registry did not accept manifest deletion")
        print(f"Removed superseded image manifest: {digest}", flush=True)


def write_pending(path, digests):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(sorted(digests)))
    temporary.replace(path)


def publish(image, registry, state_dir, runner=subprocess.run, variant="cuda"):
    dockerfile, platform = IMAGE_VARIANTS[variant]
    state_dir.mkdir(parents=True, exist_ok=True)
    # Serialize manual invocations as well as Deploy requests. Keep failed cleanup
    # digests so a later successful publication can retry untagged old manifests.
    with (state_dir / "publish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending_path = state_dir / "pending.json"
        pending = (
            set(json.loads(pending_path.read_text()))
            if pending_path.exists()
            else set()
        )
        pending.update(registry.tagged_digests())
        previous_latest = registry.digest("latest")
        if previous_latest:
            pending.add(previous_latest)
        write_pending(pending_path, pending)
        runner(
            [
                "docker",
                "build",
                "--platform",
                platform,
                "--file",
                dockerfile,
                "--tag",
                image,
                ".",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        built = runner(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{.Os}}/{{.Architecture}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if built.stdout.strip() != platform:
            raise RuntimeError(
                f"Refusing to publish {variant}: expected {platform}, got {built.stdout.strip()}"
            )
        # Save the push output and compare its manifest digest with registry latest
        # before deleting anything. Build/push errors leave the old image intact.
        print(f"Pushing {variant.upper()} latest: {image}", flush=True)
        try:
            pushed = runner(
                ["docker", "push", image], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            print(exc.stdout or "", end="", flush=True)
            print(exc.stderr or "", end="", flush=True)
            raise
        print(pushed.stdout, end="", flush=True)
        print(pushed.stderr, end="", flush=True)
        matches = re.findall(
            r"digest: (sha256:[a-f0-9]{64})", pushed.stdout + pushed.stderr
        )
        latest = registry.digest("latest")
        if not matches or latest != matches[-1]:
            raise RuntimeError(
                "Registry latest does not match the pushed image manifest"
            )
        # The old latest may have been an alias of the same digest (no-op build).
        # Never delete it, because digest deletion also removes all its tags.
        pending.discard(latest)
        write_pending(pending_path, pending)
        for digest in sorted(pending):
            if registry.digest("latest") != latest:
                raise RuntimeError(
                    "Registry latest changed during cleanup; refusing deletion"
                )
            registry.delete(digest)
            pending.remove(digest)
            write_pending(pending_path, pending)
        if registry.digest("latest") != latest:
            raise RuntimeError("Registry latest changed after cleanup")
        print(
            f"{variant.upper()} published and verified: {image} ({latest})", flush=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=IMAGE_VARIANTS, default="cuda")
    parser.add_argument("--image")
    parser.add_argument("--registry-scheme", choices=["http", "https"], default="http")
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    image = args.image or os.getenv(
        f"EZLOCALAI_{args.variant.upper()}_IMAGE",
        f"localhost:5000/devxt/ezlocalai-{args.variant}:latest",
    )
    state_dir = (
        args.state_dir or Path.home() / f".local/state/ezlocalai-{args.variant}-publish"
    )
    publish(
        image, Registry(image, args.registry_scheme), state_dir, variant=args.variant
    )


if __name__ == "__main__":
    main()
