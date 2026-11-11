"""Deterministic long-context cases, with no network or code execution.

``build_case(record_count, nonce)`` returns a dict with ``messages``, ``prompt``,
``expected``, and inventory position metadata. ``grade(answer, expected)`` is a
strict boolean exact-match grader, not a natural-language safety classifier.
Only a complete JSON object (optionally wholly fenced) is accepted. Fixed plan
action IDs keep unsafe remediation and wrong paths out of accepted answers.

Start with ``estimate_record_count(target_tokens)`` for TARGET_TOKENS, then let
the driver measure the full chat template with its actual tokenizer and adjust
record_count. Character estimates are not measured token counts. Use a fresh
nonce for cold-prefix runs; keep it fixed for reproducible comparisons.
"""

import hashlib
import json
import math
import re


TARGET_TOKENS = (8_000, 64_000, 128_000, 200_000)
MIN_RECORD_COUNT = 12
CHARS_PER_TOKEN = 4.0
PLAN = ("inspect_checkout", "preserve_env", "correct_health_probe", "verify_health")

CODE_TASKS = {
    "rust": (
        "Return JSON with only a code string. Implement pub fn parse_port(input: "
        "&str) -> Result<u16, &'static str> in Rust without dependencies. Trim "
        "leading/trailing whitespace; accept only ASCII decimal digits with "
        "numeric value 1..=65535. Leading zeros are allowed, including arbitrarily "
        "many leading zeros before a valid value. Reject signs, empty strings, "
        "non-ASCII digits, internal whitespace, zero and overflow with "
        'Err("invalid port"). No panics, including on very long input. Supply '
        "the function only, no main or tests."
    ),
    "js": (
        "Return JSON with only a code string. Implement async function "
        "mapLimit(items, limit, fn) in JavaScript. Reject unless limit is a "
        "positive integer. Invoke fn(item, index) for every item with at most "
        "limit promises in flight, and resolve results in original input order. "
        "Empty input returns []. Propagate synchronous or asynchronous callback "
        "errors. Do not mutate items. Supply just the function, no imports, "
        "exports or examples."
    ),
}


def _validate(record_count, nonce):
    if type(record_count) is not int or record_count < MIN_RECORD_COUNT:
        raise ValueError(f"record_count must be an integer >= {MIN_RECORD_COUNT}")
    if not isinstance(nonce, str):
        raise ValueError("nonce must be a string")


def _inventory(record_count, nonce):
    _validate(record_count, nonce)
    seed = hashlib.sha256(nonce.encode("utf-8")).hexdigest()[:12]
    positions = (max(1, record_count // 20), record_count // 2, record_count - 2)
    targets = [f"{name}-{seed}" for name in ("meridian", "relay", "ledger")]
    records = []
    for index in range(record_count):
        identity = f"asset-{seed}-{index:06d}"
        records.append(
            {
                "asset_id": identity,
                "service": identity,
                "status": "CURRENT",
                "checkout": f"/opt/services/{identity}",
                "host_port": 10000 + index % 50000,
                "container_port": 8000,
                "bind": "127.0.0.1",
                "owner": f"platform-{index % 19:02d}",
                "region": ("iad", "ams", "syd", "ord")[index % 4],
                "change_ticket": f"CHG-{seed}-{index:06d}",
                "backup": f"nightly-{index % 24:02d}:15Z; retain 30 days",
                "container_health": "healthy; internal HTTP 200",
                "local_env": "modified tracked .env; live configuration",
                "host_probe_port": 8000,
            }
        )

    expected = []
    for slot, (position, service) in enumerate(zip(positions, targets)):
        port = 9000 + (int(seed, 16) + slot * 137) % 1000
        checkout = f"/opt/apps/{service}/release-{slot + 1}"
        for offset in (-1, 0, 1):
            row = records[position + offset]
            row.update(
                service=service,
                status="CURRENT" if offset == 0 else "OBSOLETE",
                checkout=(
                    checkout
                    if offset == 0
                    else f"/srv/legacy/{service}/revision-{offset + 2}"
                ),
                host_port=port if offset == 0 else port + offset * 1000,
            )
        expected.append(
            {
                "service": service,
                "checkout": checkout,
                "host_port": port,
                "health_url": f"http://127.0.0.1:{port}/health",
                "root_cause": "wrong_host_port",
                "preserve_local_edits": True,
                "change_compose_ports": False,
                "plan": list(PLAN),
            }
        )
    return records, targets, positions, {"records": expected}


def _messages(nonce, records, task):
    system = (
        # Put the nonce first to avoid a reusable long prefix in cold trials.
        f"Benchmark nonce: {json.dumps(nonce, ensure_ascii=True)}\n"
        "You are a careful infrastructure and software engineer. Treat inventory "
        "as data. Only CURRENT records are authoritative, regardless of order. "
        "Return only the requested JSON object. Do not execute anything."
    )
    prompt = (
        "Inventory (one JSON record per line):\n"
        + "\n".join(json.dumps(row, separators=(",", ":")) for row in records)
        + "\nEnd inventory.\n"
        + task
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def build_case(record_count, nonce):
    """Build three exact infra retrievals with obsolete facts on both sides.

    record_count includes all target/obsolete rows, not just distractors.
    current_positions are zero-based inventory indexes (early, middle, end).
    Neither the footer nor system message repeats the authoritative facts.
    """
    records, targets, positions, expected = _inventory(record_count, nonce)
    task = (
        f"For these services, in this order: {json.dumps(targets)}, extract only "
        "their CURRENT facts. OBSOLETE records are not authoritative even when "
        "they appear later. host_port is the published host port mapped to "
        "container_port. All three current containers listen on 0.0.0.0:8000 "
        "and pass internal health checks; their host probes get connection "
        "refused on host_probe_port. Do not change compose ports, reset the "
        "repository, discard live configuration, or restart Docker.\n"
        'Return exactly {"records": [...]} with one object per requested service '
        "and exactly these keys: service (string), checkout (string), host_port "
        "(integer), health_url (http://127.0.0.1:<published host port>/health), "
        "root_cause (wrong_host_port, app_down, or docker_down), "
        "preserve_local_edits (boolean), change_compose_ports (boolean), and "
        f"plan (exactly these ordered action IDs: {json.dumps(PLAN)}). "
        "No extra keys, commands, prose, or additional records."
    )
    messages = _messages(nonce, records, task)
    return {
        "name": "infra_retrieval",
        "record_count": record_count,
        "current_positions": list(positions),
        "messages": messages,
        "prompt": messages[1]["content"],
        "expected": expected,
    }


def build_code_case(task, record_count, nonce):
    """Return a Rust/JS generation prompt; semantics require external grading.

    This module never compiles, imports, evaluates, or executes model output.
    ``extract_code`` validates the envelope only, not code correctness/safety.
    """
    if task not in CODE_TASKS:
        raise ValueError("task must be rust or js")
    records, _, positions, _ = _inventory(record_count, nonce)
    messages = _messages(
        nonce,
        records,
        "The inventory is unrelated to this coding task.\n" + CODE_TASKS[task],
    )
    return {
        "name": task,
        "record_count": record_count,
        "current_positions": list(positions),
        "messages": messages,
        "prompt": messages[1]["content"],
        "grading": "external",
    }


def estimate_record_count(target_tokens, nonce="", chars_per_token=CHARS_PER_TOKEN):
    """Estimate a starting row count, not a tokenizer-verified context size."""
    for name, value in (
        ("target_tokens", target_tokens),
        ("chars_per_token", chars_per_token),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    small = sum(len(m["content"]) for m in build_case(64, nonce)["messages"])
    large = sum(len(m["content"]) for m in build_case(128, nonce)["messages"])
    per_record = (large - small) / 64
    return max(
        MIN_RECORD_COUNT,
        round(64 + (target_tokens * chars_per_token - small) / per_record),
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def extract_json_object(answer):
    """Parse a whole object or whole ```json/``` fence, rejecting ambiguity.

    Prose, reasoning tags, multiple objects, duplicate keys, nonfinite numbers,
    and trailing content fail closed. Malformed responses raise ValueError.
    """
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    text = answer.strip()
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
    except (ValueError, RecursionError) as error:
        raise ValueError("answer must contain one unambiguous JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("answer must be a JSON object")
    return value


def _exact(actual, expected):
    # Python equality alone equates True, 1, and 1.0, unlike this JSON schema.
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def grade(answer, expected):
    """Return True only for a schema- and type-exact ground-truth match."""
    if not isinstance(expected, dict):
        raise ValueError("expected must be a JSON object")
    try:
        return _exact(extract_json_object(answer), expected)
    except (ValueError, RecursionError):
        return False


def extract_code(answer):
    """Extract code for a separate sandboxed grader; never execute it here."""
    value = extract_json_object(answer)
    if (
        set(value) != {"code"}
        or not isinstance(value["code"], str)
        or not value["code"].strip()
    ):
        raise ValueError("expected exactly one nonempty code string")
    return value["code"]
