# Native C++ vs Python ezlocalai, RTX 3090 Ti, 2026-10-19

The two workers were measured on the same host, with the same `.env`, the same
model files and the same GPU. Raw per-request timings are in
[cpp-vs-python-3090ti-20261019.json](cpp-vs-python-3090ti-20261019.json).

## Setup

- **Host:** RTX 3090 Ti (24,564 MiB), i9-12900KS, 123 GiB RAM.
- **Workers:** one at a time on port 8091.
  - **Native:** image `edad6bc87dce`, commit `fd9f509`.
  - **Python:** image `c2180ffee0b1`, the local `ezlocalai-ezlocalai` build.
    Production Python workers report the same source commit, `0f7e9f4`.
- **Configuration:** the production `.env` unchanged.
  - Model: `unsloth/Qwen3.8-27B-GGUF`, Q3_K_XL.
  - Allocated context 220,000, one slot, Q4 KV, full GPU offload.
  - MTP draft n-max 3, p-min 0.1; batch 4096 / ubatch 512.
- **Router:** `ROUTER_URL` was blanked through a Compose interpolation
  override, so neither worker registered and no production traffic reached the
  card during measurement.
- **Requests:**
  - Greedy, streamed, sent over the HTTP API by
    [`bench_api.py`](bench_api.py).
  - Prompt rates and decode rates come from llama.cpp `timings`. Wall time and
    time to first token (TTFT) are measured by the client.
  - Two warmup requests per worker are excluded.
  - Each cold prefill starts with a unique nonce. Only the 42-token chat
    template prefix is reused (`cache_n=42`).
  - The prompt is Python `Pipes.py` followed by `app.py`, truncated to the
    case length. SHA-256 prefix of the full text: `1e3069548a76c26f`.
- **Sample size:** 10 runs for the latency case, 3 for decode and the shorter
  prefills, 2 for the 89.4k prefill. These samples are small, so differences
  of a few percent are within run-to-run variation.

## Inference

| Metric | Native | Python | Native vs Python |
| --- | ---: | ---: | ---: |
| 1-token request, median wall time (s) | 0.257 | 0.256 | +0.0% |
| 256-token decode, mean tok/s | 70.3 | 68.3 | +2.9% |
| 256-token decode, mean TTFT (s) | 0.393 | 0.369 | +6.4% |
| Cold prefill 7.6k tokens, tok/s | 1463 | 1453 | +0.7% |
| Cold prefill 7.6k tokens, TTFT (s) | 5.45 | 5.48 | -0.5% |
| Cold prefill 29.7k tokens, tok/s | 1347 | 1339 | +0.6% |
| Cold prefill 29.7k tokens, TTFT (s) | 22.56 | 23.43 | -3.7% |
| Cold prefill 89.4k tokens, tok/s | 1020 | 1009 | +1.0% |
| Cold prefill 89.4k tokens, TTFT (s) | 88.83 | 89.66 | -0.9% |
| Cached identical 29.7k repeat, TTFT (s) | 0.266 | 0.256 | +3.7% |
| Cached 29.7k + short follow-up, TTFT (s) | 0.589 | 0.620 | -5.0% |

Both workers run the same llama.cpp revision, so inference speed is
equivalent:

- Cold prefill is 0.6–1.0% faster natively in every case. That is consistent
  but small.
- The decode differences track speculative-draft acceptance, not the server.
  Individual 256-token runs ranged from 61.6 to 84.0 tok/s natively and from
  63.7 to 76.6 tok/s in Python.
- One Python 29.7k run took 25.1 s against about 22.6 s for the others, which
  accounts for most of that row's TTFT gap.
- Prompt-cache reuse behaves identically: the whole 29.7k prefix is reused on
  a repeat and on a follow-up turn.

## Footprint and startup

| Metric | Native | Python |
| --- | ---: | ---: |
| Image size | 19.3 GB | 22.4 GB |
| `up -d` to `/health` (warm page cache) | 5.9 s | 14.7 s |
| `up -d` to first successful chat | 6.2 s | 15.1 s |
| Container RAM after model load | 1.64 GiB | 8.46 GiB |
| Container RAM after the benchmark | 16.84 GiB | 23.54 GiB |
| GPU memory, model loaded | 20,790 MiB (llama-server 20,530 + worker 260) | 20,530 MiB |

- Native starts about 2.4× faster and uses about 6.8 GiB less host RAM. Both
  grow by a similar amount over the benchmark, which is the host-RAM prompt
  cache filling.
- Startup was measured with the model files already in the page cache. A cold
  disk read would add the same load time to both.
- **Native GPU overhead:** the native worker process holds 260 MiB of VRAM.
  That is the primary CUDA context retained by `cuda_memory_info`, which
  supplies free-VRAM telemetry for the heartbeat, admission and
  `/v1/resources`. Python's torch context lives in the same process as its
  in-process llama.cpp, so it adds no separate allocation.
- **Proposed fix:** query NVML for memory, which needs no context, and map the
  results to CUDA-visible ordinals. That would recover the 260 MiB.

## Not measured

- Concurrency: `N_PARALLEL=1`, so concurrent requests only queue.
- Thinking-mode sampling.
- Media workloads.
- Cold-disk startup.
- Repeated sessions on different days.
