# Performance measurements

Keep weights, quantization, allocated context, KV precision and speculative
settings fixed when comparing throughput. Record the engine revision and GPU
clocks, power and temperature. Different revisions or hardware settings make
a comparison unsuitable for attributing a change to a single cause.

## Cold prefill and cache reuse

`bench_prefill.py` measures native evaluated and cached token counts through
HTTP. Use a reserved, isolated engine endpoint for controlled greedy sampling;
the worker API can apply model presets that override requested sampling.
The operator manages router drain ownership, lease renewal and restoration.
The script itself does not stop workers or change runtime settings.

```sh
python benchmarks/bench_prefill.py http://127.0.0.1:28192 frozen-source.py \
  build/validation/prefill.json --prompt-chars 300000 680000 --cold-repeats 3
```

A nonce at the beginning of each new user prefix prevents a shorter request
from warming the longer request. The same prefix is retained for the identical
repeat, short follow-up and appended-source cases. Warmup is excluded. A cold
sample reusing 100 or more tokens is rejected. Token rates use the engine's
evaluated-token timings; cached input is not counted as evaluated work.

For the exact historical prompt hashes, use `--exact-first-prompt` with one
prompt length and a freshly started engine. Freeze and hash the source file
across runs. Pass API credentials through the environment named by
`--api-key-env` (default `EZLOCALAI_API_KEY`).

The older `bench_api.py` includes streamed TTFT and decode cases. Its
`decode_wall_tps` field counts content/reasoning chunks; chunks can contain
multiple tokens. Use native `timings.predicted_per_second` for token throughput.

## Worker request preparation

`bench_llm_request` isolates CPU request normalization from model inference:

```sh
cmake --build build --target bench_llm_request --parallel 2
build/bench_llm_request frozen-source.py build/validation/request-preparation
```

It reports seven samples of ten preparations at three source byte budgets.
Payload files permit before/after byte and hash comparisons, including all
messages and sampling settings. This measures request preparation, not GPU
prefill or whole-request speed. Build with the same compiler/options and compare
on the same idle host. The target is excluded from ordinary builds.

Benchmark guard checks run with
`python -m unittest discover -s benchmarks -p test_bench_prefill.py`.

## Comparative streaming and conversation measurements

`bench_progression.py` adds streamed first-output latency, native decode rates
and an alternating conversation replay. Reserve the worker/engine before
running it and capture sanitized image, engine, GPU and runtime settings in
the metadata file. The caller owns drain leases and restoration.

```sh
python benchmarks/bench_progression.py http://127.0.0.1:28192 frozen-source.py \
  build/validation/4090-baseline.json --label DevXT4090 \
  --model unsloth/Qwen3.8-27B-GGUF --metadata sanitized-runtime.json \
  --run-id comparison-20261026 --prompt-chars 300000 680000
```

Run the same command on fresh baseline and candidate engines with the same
`--run-id` to compare identical request hashes and answers. The default run ID
is random, preventing accidental reuse between unrelated runs. A fixed run ID
must not be reused on a warmed engine: cold samples that reuse 100 or more
tokens are rejected. `messages_sha256` retains the historical `bench_prefill`
serialization. Exact October 18/25 first prompts require one fresh engine per
depth and `--suites prefill --prompt-chars 680000 --exact-first-prompt`.

Suites can be selected independently with `--suites latency decode prefill
cache`. First-output latency ends at the first nonempty content or reasoning
delta; role announcements, empty deltas, usage and keepalives do not count.
Reports retain native evaluated/cached/generated token counts and rates. They
never derive token throughput from stream chunks. Inspect the generated-token
count and finish reason when comparing decode runs. Warmup is excluded.

The cache suite alternates three named conversations twice by default. It
records evaluated tokens, cached tokens, wall time, first-output latency,
branch-label correctness and whether returning answers match their first
answers. `tail_reused` means at most 16 prompt tokens were evaluated; initial
visits to later conversations may legitimately reuse their common prefix.
`--require-identical-cache-output` makes a changed returning answer fail the
run. Cache hits depend on the actual byte budget and model state size.

Completed samples and summaries are saved atomically after each request.
Failures leave a report with `status=failed`; only `status=complete` represents
a complete comparison run. Reports include input and harness fingerprints,
requested sampling, full answer text/hashes, and caller-supplied settings.
Likely credential fields in metadata are rejected; API credentials come from
`--api-key-env` and are not recorded. Engine timing fields are required, so
endpoints that filter them cannot produce misleading native throughput.

The parser, timing, cold-cache and checkpoint guards run without an inference
server: `python -m unittest discover -s benchmarks -p 'test_bench*.py'`.

## Fixed-context batch qualification

Before promoting a physical-batch increase, `bench_capacity.py` checks a cold
text prompt near 95% of the declared context and a small solid-red PNG through
the same isolated engine/projector:

```sh
python benchmarks/bench_capacity.py http://127.0.0.1:28192 frozen-source.py \
  230000 1024 DevXT3090V build/validation/3090v-capacity.json \
  --metadata sanitized-launch.json --run-id qualification-20261026
```

The metadata must describe the verified immutable launch: either `n_ctx` and
`n_ubatch`, or the existing benchmark `original_parameters` map plus
`overrides` (`-c`/`-ub`). Include model/projector SHA-256, image and engine
identities when collecting promotion evidence. The pinned engine's `/props`
exposes effective context and vision availability, but not physical batch or
projector byte identity; those values require the caller's launch provenance.

The client calibrates only the prepared text length using `/apply-template`
and `/tokenize`, then cross-checks `/v1/chat/completions/input_tokens`. It keeps
the declared context fixed, allowing only its normal upward 256-token
allocation padding. It does not retry inference or reduce context on failure.
Both actual inference requests disable prefix reuse and must evaluate every
prepared input token, stop normally and answer `Python` or `red`, respectively.
The generated image, requests, source and harness are fingerprinted. Atomic
checkpoints retain raw completion responses for diagnosing failed guards.

A `status=complete` report qualifies these two workloads at the recorded
settings; larger image/video workloads and sustained concurrency need their
own validation. This program owns no drains, deployment or restoration steps.

## Long decode after cache switches and cache pressure

`bench_cache_workload.py` exercises MTP after named conversation restores,
including an identical prompt replay, an appended assistant/user history and
a shorter deterministic branch-record prefix. It requires a reserved,
single-slot engine with `/props` and `/v1/chat/completions/input_tokens`:

```sh
python benchmarks/bench_cache_workload.py http://127.0.0.1:28192 frozen-source.py \
  build/validation/cache-resume.json --label DevXT4090 --context 230000 \
  --model unsloth/Qwen3.8-27B-GGUF --metadata sanitized-runtime.json \
  --run-id cache-resume-20261026 --require-draft-counters
```

The default resident workload fills three conversations, then alternates
replay, append and rewind requests. Each requests 256 generated tokens; use
`--decode-tokens 512` for longer decode, or select cases with `--cases replay
append`. `--rounds 2` means the seed fill followed by one round of each selected
case. With only `--cases append`, later rounds extend the preceding real
assistant answer. The default rewind shortens the branch identity record from
90 to 40 lines while retaining the source prefix; these sizes are configurable.

Reuse a run ID on fresh baseline and candidate engines with the same source
and arguments. Seed, replay and rewind prompts are deterministic. Appended
requests include the actual previous answer and record its sample ID/output
hash; their payloads may diverge if preceding answers differ. Compare request
hashes before treating two measurements as a matched pair. A branch marker
and identity code on the first response line detect cross-conversation mixups.
Freeform wording differences are retained, not automatically called a quality
regression. Warmup is excluded, and a short marker-only answer cannot become
a fast decode sample (`--minimum-generated` defaults to 64).

Pressure mode defaults to eight independent early-prefix conversations and
replay requests only:

```sh
python benchmarks/bench_cache_workload.py http://127.0.0.1:28192 frozen-source.py \
  build/validation/cache-pressure.json --label DevXT4090 --context 230000 \
  --mode pressure --conversations 8 --rounds 3 --prompt-chars 100000 \
  --metadata sanitized-runtime.json --run-id cache-pressure-20261026 \
  --require-draft-counters
```

Choose the conversation count/source size against the recorded host cache
budget. `--prefix-mode common` measures related conversations; `independent`
puts their discriminator before the long prompt, requiring cold seed fills.
Returning requests are classified as tail reuse, prefix reuse or full
re-evaluation. A miss under pressure may be a legitimate eviction, unavailable
checkpoint or skipped save; the client reports the work without asserting its
cause or requiring every returning state to fit. It never changes memory or
swap limits. Collect cgroup `memory.current`, `memory.swap.current`, events and
limits externally to establish actual swap pressure; cache misses alone are
not evidence of swap-in or eviction.

The declared context is fixed. Each prompt is token-counted before generation,
including appended histories, with the full requested decode budget and 32
tokens of headroom reserved. Native prompt/usage counts must match the
preflight count, and reported truncation, missing completions, unexpected
reasoning and invalid MTP counters fail explicitly. Draft acceptance uses
native `draft_n`/`draft_n_accepted`, and throughput uses native timing fields.
No token rate is derived from SSE chunks. Draft counters may be unavailable
for a non-speculative baseline unless `--require-draft-counters` is passed.
Token-count preflight and request JSON encoding are excluded from HTTP wall
time and TTFT, matching the earlier streaming measurement client.

Reports are atomic after each sample and retain a pending request/failed raw
sample when possible. `complete`, `failed` and interrupted `cancelled` statuses
are distinct. The source, driver and imported parser/cold-guard modules are
fingerprinted. Metadata accepts known sanitized runtime/provenance fields,
including the earlier benchmark launch maps and selected performance settings;
raw Docker inspect/environment lists and likely credential fields are rejected.
API credentials come only from the environment named by `--api-key-env`.
