"""Grade Rust/JS probe JSONL in disposable CPU-only Docker sandboxes.

Usage: python benchmarks/grade_prefill_code.py PROBE.jsonl GRADES.json

Only completed Rust/JS samples are tested; infra grades are left untouched.
Images must already be cached. Model code is sent on stdin to Docker, never
executed or compiled on the host. These are behavioral checks, not a proof of
correctness or a tamper-proof judge for intentionally adversarial submissions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from prefill_cases import extract_code


IMAGES = {"js": "node:22-alpine", "rust": "rust:1.93-slim-bookworm"}
DIAGNOSTIC_LIMIT = 1500
TEST_TIMEOUT = 20
IDENTIFIERS = (
    "block",
    "variant",
    "target_tokens",
    "template_tokens",
    "repeat",
    "kind",
    "task",
    "finish_reason",
    "prompt_sha256",
)

JS_TESTS = r"""
const assert = require('node:assert/strict');
(async () => {
  let active=0, peak=0; const seen=[]; const items=[0,1,2,3,4,5,6];
  const result=await mapLimit(items,3,async (n,i)=>{
    active++; peak=Math.max(peak,active); seen.push(i);
    await new Promise(r=>setTimeout(r, (7-n)*2)); active--; return n*10+i;
  });
  assert.deepEqual(result,[0,11,22,33,44,55,66]);
  assert.deepEqual(items,[0,1,2,3,4,5,6]);
  assert.deepEqual(seen.slice().sort((a,b)=>a-b),[0,1,2,3,4,5,6]);
  assert(peak<=3 && peak>1);
  assert.deepEqual(await mapLimit([],3,()=>{throw Error('called')}),[]);
  assert.deepEqual(await mapLimit([2,4],10,n=>n/2),[1,2]);
  assert.deepEqual(await mapLimit([2,4],1,n=>n/2),[1,2]);
  for(const bad of [0,-1,1.5,NaN,Infinity,'2',null,undefined,true]) {
    await assert.rejects(async()=>mapLimit([1],bad,n=>n));
    await assert.rejects(async()=>mapLimit([],bad,n=>n));
  }
  await assert.rejects(async()=>mapLimit([1,2],2,()=>{throw Error('sync')}),/sync/);
  await assert.rejects(async()=>mapLimit([1,2],2,async()=>{throw Error('async')}),/async/);
  process.stdout.write('\nPREFILL_CODE_TESTS_PASSED\n');
})().catch(e=>{console.error(e);process.exitCode=1;});
"""

RUST_TESTS = r"""
#[test] fn accepted() {
  for (s,n) in [("1",1),("65535",65535),(" 9187\n",9187),("000080",80),
                ("\u{2003}80\u{2003}",80)] {
    assert_eq!(parse_port(s),Ok(n),"{s:?}");
  }
  assert_eq!(parse_port(&("0".repeat(10000)+"42")),Ok(42));
}
#[test] fn rejected() {
  for s in [""," ","0","0000","65536","9999999999999999999999","+80","-80",
            "8 0","１２","١٢","1.0","0x50","80tcp","8\t0","8\n0","80\0"] {
    assert_eq!(parse_port(s),Err("invalid port"),"{s:?}");
  }
  assert_eq!(parse_port(&"9".repeat(10000)),Err("invalid port"));
  assert_eq!(parse_port(&("0".repeat(10000)+"65536")),Err("invalid port"));
}
"""


def sandbox_command(task, owner):
    """Only trusted shell text enters argv; model source travels over stdin."""
    if task not in IMAGES:
        raise ValueError("only rust and js can be sandbox graded")
    name = "prefill-code-grade-" + owner
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "-i",
        "--name",
        name,
        "--label",
        "xt.prefill.grader=" + owner,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--cpus",
        "1",
        "--pids-limit",
        "64",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=128m,mode=1777",
        "--workdir",
        "/tmp",
        "--env",
        "HOME=/tmp",
        "--log-driver",
        "none",
        "--entrypoint",
        "sh",
        IMAGES[task],
        "-c",
    ]
    filename = "/tmp/source.js" if task == "js" else "/tmp/source.rs"
    runner = (
        "node /tmp/source.js"
        if task == "js"
        else "rustc --edition=2021 --test /tmp/source.rs -o /tmp/tests && /tmp/tests"
    )
    # Keep potentially large compiler/test output in the bounded container tmpfs.
    command.append(
        f"cat > {filename} || exit 1; ({runner}) > /tmp/test-output 2>&1; "
        'status=$?; tail -c 6000 /tmp/test-output; exit "$status"'
    )
    return command


def evaluate(sample):
    """Return (passed, diagnostic); never mutate the input sample."""
    task = sample.get("task")
    if task not in IMAGES:
        raise ValueError("only rust and js can be sandbox graded")
    if sample.get("finish_reason") != "stop":
        return (
            False,
            ("incomplete: " + str(sample.get("finish_reason")))[:DIAGNOSTIC_LIMIT],
        )
    try:
        code = extract_code(sample.get("answer"))
    except ValueError as error:
        return False, ("invalid output: " + str(error))[:DIAGNOSTIC_LIMIT]
    owner = uuid.uuid4().hex
    command = sandbox_command(task, owner)
    tests = JS_TESTS if task == "js" else RUST_TESTS
    marker = (
        "PREFILL_CODE_TESTS_PASSED" if task == "js" else "test result: ok. 2 passed;"
    )
    passed = False
    diagnostic = ""
    try:
        result = subprocess.run(
            command,
            input=code + "\n" + tests,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=TEST_TIMEOUT,
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0 and marker in result.stdout
        if result.returncode:
            diagnostic = f"sandbox exit {result.returncode}: " + output[-1400:]
        elif not passed:
            diagnostic = "missing test completion marker: " + output[-1400:]
        else:
            diagnostic = output[-DIAGNOSTIC_LIMIT:]
    except subprocess.TimeoutExpired:
        diagnostic = "test timeout"
    except OSError as error:
        diagnostic = "sandbox unavailable: " + str(error)
    finally:
        # UUID names and ownership labels isolate cleanup from all other work.
        try:
            cleanup = subprocess.run(
                ["docker", "rm", "-f", "prefill-code-grade-" + owner],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=15,
            )
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                passed = False
                diagnostic = (
                    "sandbox cleanup failed: "
                    + cleanup.stderr[-700:]
                    + "; "
                    + diagnostic
                )
        except (OSError, subprocess.TimeoutExpired) as error:
            passed = False
            diagnostic = (
                "sandbox cleanup failed: " + type(error).__name__ + "; " + diagnostic
            )
    return passed, diagnostic[:DIAGNOSTIC_LIMIT]


def grade_file(path):
    """Hash original bytes, validate all rows, then grade only Rust/JS samples."""
    raw = Path(path).read_bytes()
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected an object")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"Invalid probe JSONL at line {number}") from error
        rows.append((number, row))
    samples = []
    skipped_infra = 0
    for number, row in rows:
        if row.get("event") != "sample":
            continue
        if row.get("task", "infra") == "infra":
            skipped_infra += 1
            continue
        if row.get("task") not in IMAGES:
            continue
        passed, diagnostic = evaluate(row)
        result = {key: row[key] for key in IDENTIFIERS if key in row}
        result.update(
            line=number,
            passed=passed,
            diagnostic=diagnostic,
            sandbox_image=IMAGES[row["task"]],
            answer_sha256=(
                hashlib.sha256(row["answer"].encode("utf-8")).hexdigest()
                if isinstance(row.get("answer"), str)
                else None
            ),
        )
        samples.append(result)
    passed = sum(sample["passed"] for sample in samples)
    return {
        "schema_version": 1,
        "input_path": str(path),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "summary": {
            "graded": len(samples),
            "passed": passed,
            "failed": len(samples) - passed,
            "skipped_infra": skipped_infra,
        },
        "samples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Grades must be written separately from probe evidence")
    if args.output.exists():
        parser.error("Grade output already exists; use a new path")
    report = grade_file(args.input)
    # Exclusive creation also protects against another grader racing this one.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=True)
        output.write("\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
