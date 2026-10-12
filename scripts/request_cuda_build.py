#!/usr/bin/env python3
"""Ask Deploy to publish CUDA latest, then wait for verified completion."""

import json
import os
import re
import time
import urllib.error
import urllib.request


def request_json(url, secret, payload=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"X-API-Key": secret, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def wait_for_build(url, secret, event_id, timeout=18000):
    if not re.fullmatch(r"[0-9]+-[a-f0-9]+", event_id or ""):
        raise RuntimeError(
            "Deploy returned no valid event_id; install the Deploy CUDA update first"
        )
    deadline = time.monotonic() + timeout
    status_url = f"{url.rstrip('/')}/status/events/{event_id}"
    previous = None
    while time.monotonic() < deadline:
        try:
            event = request_json(status_url, secret)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise RuntimeError(f"Deploy status returned HTTP {exc.code}") from exc
            print(
                f"Deploy status temporarily unavailable (HTTP {exc.code}); retrying",
                flush=True,
            )
        except (urllib.error.URLError, TimeoutError):
            print("Deploy status temporarily unreachable; retrying", flush=True)
        else:
            if (
                event.get("id") != event_id
                or event.get("deploy_type") != "ezlocalai-cuda-build"
            ):
                raise RuntimeError("Deploy returned a different build event")
            status = event.get("status")
            if status != previous:
                print(f"CUDA registry build: {status}", flush=True)
                previous = status
            if status == "success":
                print(
                    "CUDA latest published, verified, and old manifests removed",
                    flush=True,
                )
                return
            if status not in {"queued", "running"}:
                raise RuntimeError(
                    f"CUDA registry build {status}: {event.get('message', '')}"
                )
        time.sleep(15)
    raise TimeoutError("CUDA publication did not complete before the workflow timeout")


def main():
    url = os.environ["DEPLOY_URI"].rstrip("/")
    secret = os.environ["DEPLOY_SECRET"]
    domain = os.environ.get("DEPLOY_DOMAIN", "api.ezlocal.ai")
    if not url or not secret or domain != "api.ezlocal.ai":
        raise RuntimeError(
            "Configure DEPLOY_URI, DEPLOY_SECRET, and DEPLOY_DOMAIN=api.ezlocal.ai"
        )
    response = request_json(
        url,
        secret,
        {
            "domain": domain,
            "repository": os.environ["GITHUB_REPOSITORY"],
            "branch": "main",
            "commit_id": os.environ["GITHUB_SHA"],
            "github_actor": os.environ["GITHUB_ACTOR"],
        },
    )
    if response.get("status") != "accepted":
        raise RuntimeError(
            "Deploy did not accept a CUDA build event; install the Deploy update first"
        )
    wait_for_build(url, secret, response.get("event_id"))


if __name__ == "__main__":
    main()
