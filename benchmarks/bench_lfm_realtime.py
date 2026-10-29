#!/usr/bin/env python3
"""Benchmark LFM realtime calls; tools are deterministic mocks, never real agents.

Connection JSON accepts base/key and optional worker/resource_base/resource_key.
Keep credentials in an untracked private file. Only endpoint/worker metadata is
written to results. PCM arrival, RMS activity and hypothetical untrimmed playout
are distinct: the last assumes an 80ms starting buffer and is not human speech
onset certification. Requires websockets and, for --audio, ffmpeg.
"""
import re, argparse, base64, json, math, struct, time, urllib.request, urllib.parse, wave, statistics, uuid, subprocess
from pathlib import Path
from websockets.sync.client import connect


def main():
    MODEL = "LiquidAI/LFM3-Omni-2B-GGUF"
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--voices",
        nargs="+",
        default=["native", "british_rp_female", "warm_american_narrator"],
    )
    parser.add_argument("--audio", default="")
    parser.add_argument("--tools", action="store_true")
    parser.add_argument("--context-words", type=int, default=0)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Explicitly allow CPU-only LFM diagnostics",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.connection).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    assert set(config) <= {
        "base",
        "key",
        "worker",
        "resource_base",
        "resource_key",
    }, "Unexpected connection fields"
    assert all(
        re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", voice) and voice not in (".", "..")
        for voice in args.voices
    ), "Use a safe voice basename"
    base = config["base"].rstrip("/")
    key = config.get("key", "")
    headers = {"Authorization": "Bearer " + key} if key else {}
    label = config.get("worker")
    run = uuid.uuid4().hex[:12]
    records = []
    calls = []
    errors = []

    def http(path):
        with urllib.request.urlopen(
            urllib.request.Request(
                config.get("resource_base", base) + path,
                headers=(
                    {"Authorization": "Bearer " + config.get("resource_key", key)}
                    if config.get("resource_key", key)
                    else {}
                ),
            ),
            timeout=15,
        ) as r:
            return json.load(r)

    def event(ws, deadline):
        return json.loads(ws.recv(timeout=max(0.1, deadline - time.monotonic())))

    def open_call(voice, tools=False):
        q = {"model": MODEL, "voice": voice}
        if label:
            q["worker"] = label
        started = time.monotonic()
        ws = connect(
            base.replace("http:", "ws:").replace("https:", "wss:")
            + "/v1/realtime?"
            + urllib.parse.urlencode(q),
            additional_headers=headers,
            open_timeout=120,
            close_timeout=5,
            compression=None,
            ping_interval=None,
            max_size=4 * 1024**2,
        )
        e = event(ws, time.monotonic() + 30)
        assert e["type"] == "session.created", e
        s = e["session"]
        rate = s["audio"]["output"]["format"]["rate"]
        assert rate == 24000
        if voice != "native":
            assert (
                s.get("voice_conversion", {}).get("engine") == "meanvc2"
                and s["voice_conversion"]["voice"] == voice
            ), "Selected voice not verified"
        session = {
            "instructions": "Silent benchmark id "
            + run
            + " "
            + voice
            + ". You are a concise voice assistant. The agent project codename is Cedar. It is currently running unit tests. Answer audio questions naturally. Follow requests to say exact words, and do not read benchmark markers aloud.",
            "max_output_tokens": 128,
            "output_modalities": ["text", "audio"],
        }
        if tools:
            session.update(
                {
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_agent_progress",
                            "description": "Get the current working agent task and progress.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        }
                    ],
                    "tool_choice": "auto",
                }
            )
        ws.send(json.dumps({"type": "session.update", "session": session}))
        while True:
            e = event(ws, time.monotonic() + 30)
            if e["type"] == "error":
                raise RuntimeError(str(e))
            if e["type"] == "session.updated":
                break
        calls.append(
            {
                "voice": voice,
                "setup_seconds": time.monotonic() - started,
                "session_voice_conversion": s.get("voice_conversion"),
            }
        )
        return ws, s

    def response(ws, voice, case, cancel=False, opts=None):
        started = time.monotonic()
        ws.send(
            json.dumps(
                {"type": "response.create", **({"response": opts} if opts else {})}
            )
        )
        tool_calls = []
        pcm = bytearray()
        packet = []
        first_text = None
        text = ""
        audible = None
        deadline = started + 90
        cancelled = False
        cancel_at = None
        while time.monotonic() < deadline:
            e = event(ws, deadline)
            t = time.monotonic() - started
            kind = e["type"]
            if kind == "error":
                raise RuntimeError(str(e["error"]))
            if (
                kind == "response.output_item.added"
                and e.get("item", {}).get("type") == "function_call"
            ):
                tool_calls.append(e["item"])
            if kind == "response.output_text.delta":
                if first_text is None:
                    first_text = t
                text += e.get("delta", "")
            if kind == "response.output_audio.delta":
                b = base64.b64decode(e["delta"], validate=True)
                if not b:
                    continue
                samples = struct.unpack("<" + "h" * (len(b) // 2), b)
                packet.append({"at": t, "samples": len(samples)})
                pcm.extend(b)
                if (
                    audible is None
                    and len(samples) >= 480
                    and any(
                        math.sqrt(
                            sum(x * x for x in samples[k : k + 480])
                            / len(samples[k : k + 480])
                        )
                        > 98
                        for k in range(0, len(samples), 480)
                    )
                ):
                    audible = t
                if cancel and not cancelled:
                    ws.send(json.dumps({"type": "response.cancel"}))
                    cancelled = True
                    cancel_at = t
            if kind == "response.done":
                elapsed = time.monotonic() - started
                seconds = len(pcm) / 48000
                prior = 0
                late = []
                for p in packet:
                    late.append(
                        max(0, p["at"] - packet[0]["at"] - 0.08 - prior / 24000)
                    )
                    prior += p["samples"]
                record = {
                    "voice": voice,
                    "case": case,
                    "first_text_seconds": first_text,
                    "first_pcm_seconds": packet[0]["at"] if packet else None,
                    "first_energy_packet_seconds": audible,
                    "elapsed_seconds": elapsed,
                    "audio_seconds": seconds,
                    "generation_rtf": elapsed / seconds if seconds else None,
                    "largest_playout_lateness_with_80ms_buffer_seconds": max(
                        late, default=0
                    ),
                    "cancel_ack_seconds": (
                        elapsed - cancel_at if cancel_at is not None else None
                    ),
                    "status": e["response"].get("status"),
                    "text": text,
                    "packets": packet,
                    "tool_calls": tool_calls,
                }
                leading = None
                if pcm:
                    all_samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
                    leading = next(
                        (
                            i / 24000
                            for i in range(0, len(all_samples), 480)
                            if math.sqrt(
                                sum(x * x for x in all_samples[i : i + 480])
                                / len(all_samples[i : i + 480])
                            )
                            > 98
                        ),
                        None,
                    )
                record["leading_quiet_seconds"] = leading
                record["estimated_untrimmed_energy_playout_seconds"] = (
                    max(audible, packet[0]["at"] + 0.08 + leading)
                    if leading is not None and audible is not None
                    else None
                )
                records.append(record)
                if not cancel and case != "tool-request":
                    assert pcm and audible is not None, "No energy-bearing speech"
                elif cancel:
                    assert record["status"] == "cancelled", "Cancel not acknowledged"
                with wave.open(str(out / (voice + "-" + case + ".wav")), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(pcm)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "packets"}),
                    flush=True,
                )
                return record
        raise TimeoutError("Response deadline")

    def user(ws, text):
        ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )

    try:
        before = http("/v1/resources")
        for voice in args.voices:
            ws, s = open_call(voice)
            try:
                runtime = (
                    http("/v1/resources")
                    .get("model_lifecycle", {})
                    .get("loaded_llm_runtime", [])
                )
                selected = [row for row in runtime if row.get("model") == MODEL]
                if not args.allow_cpu and (
                    not selected or selected[0].get("gpu_layers", 0) == 0
                ):
                    raise RuntimeError(
                        "LFM GPU offload was not verified; check GPU_LAYERS or explicitly use --allow-cpu"
                    )
                for n in range(args.rounds):
                    user(
                        ws,
                        "Silent marker "
                        + run
                        + "-"
                        + str(n)
                        + ". Say exactly: Hello. The working agent is running the tests. I can help you steer it.",
                    )
                    response(ws, voice, "turn-" + str(n))
                if args.context_words:
                    background = (
                        "Completed dependency review and verified the test fixtures for the Cedar project. "
                        * ((args.context_words // 15) + 1)
                    ).split()[: args.context_words]
                    user(
                        ws,
                        "Silent working-agent context update "
                        + run
                        + ": "
                        + " ".join(background)
                        + " The current task has now changed to deploying Cedar. Tell the user its current task in one sentence.",
                    )
                    response(ws, voice, "context-update")
                user(
                    ws,
                    "Silent marker "
                    + run
                    + "-interrupt. Count slowly from one to one hundred.",
                )
                response(ws, voice, "cancel", True)
                user(
                    ws,
                    "Silent marker "
                    + run
                    + "-resume. The user interrupted. Say only: Ready.",
                )
                response(ws, voice, "after-cancel")
            finally:
                ws.close()
            if args.audio:
                ws, s = open_call(voice)
                try:
                    rate = s["audio"]["input"]["format"]["rate"]
                    pcm = subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-i",
                            args.audio,
                            "-ar",
                            str(rate),
                            "-ac",
                            "1",
                            "-f",
                            "s16le",
                            "pipe:1",
                        ],
                        stdout=subprocess.PIPE,
                        check=True,
                    ).stdout
                    step = rate // 50 * 2
                    started = time.monotonic()
                    for i in range(0, len(pcm), step):
                        ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(
                                        pcm[i : i + step]
                                    ).decode(),
                                }
                            )
                        )
                        delay = started + (i + step) / 2 / rate - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                    ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    response(ws, voice, "speech-to-speech")
                finally:
                    ws.close()
            if args.tools:
                ws, s = open_call(voice, True)
                try:
                    user(
                        ws,
                        "Call get_agent_progress now, then report what the working agent is doing.",
                    )
                    r = response(
                        ws, voice, "tool-request", opts={"tool_choice": "required"}
                    )
                    assert (
                        r["tool_calls"]
                        and r["tool_calls"][0]["name"] == "get_agent_progress"
                    ), "Missing progress tool"
                    call = r["tool_calls"][0]
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call["call_id"],
                                    "output": json.dumps(
                                        {
                                            "project": "Cedar",
                                            "task": "deploying",
                                            "status": "in progress",
                                        }
                                    ),
                                },
                            }
                        )
                    )
                    r = response(ws, voice, "tool-result", opts={"tool_choice": "none"})
                    assert "deploy" in r["text"].lower(), "Tool result not reflected"
                finally:
                    ws.close()
        after = http("/v1/resources")
    except BaseException as e:
        errors.append(type(e).__name__ + ": " + str(e))
        raise
    finally:
        summary = {}
        for voice in args.voices:
            rows = [
                r
                for r in records
                if r["voice"] == voice and r["case"].startswith("turn-")
            ]

            def stats(name):
                a = sorted(r[name] for r in rows if r[name] is not None)
                return (
                    {
                        "n": len(a),
                        "median": statistics.median(a),
                        "min": a[0],
                        "max": a[-1],
                    }
                    if a
                    else None
                )

            summary[voice] = {
                k: stats(k)
                for k in [
                    "first_pcm_seconds",
                    "first_energy_packet_seconds",
                    "leading_quiet_seconds",
                    "estimated_untrimmed_energy_playout_seconds",
                    "generation_rtf",
                    "largest_playout_lateness_with_80ms_buffer_seconds",
                ]
            }
        result = {
            "model": MODEL,
            "run_id": run,
            "connection": {
                k: v
                for k, v in config.items()
                if k in ["base", "worker", "resource_base"]
            },
            "calls": calls,
            "records": records,
            "summary": summary,
            "errors": errors,
            "resources_before": locals().get("before"),
            "resources_after": locals().get("after"),
        }
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
