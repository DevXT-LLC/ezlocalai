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
