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
