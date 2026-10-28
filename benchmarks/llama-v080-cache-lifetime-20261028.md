# llama.cpp v0.8.0 and cache lifetime, 2026-10-28

The maintained engine moves from v0.7.0 (`620d2225cc436ab0b001e3200e590e3b286ef9ab`)
to [v0.8.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.8.0)
(`8642541b255bfb5dc39cd351a178be09013a55d6`). All eight existing patches apply;
a ninth preserves an early valid named-conversation checkpoint within the
existing count/byte budget. Exact `EZLOCALAI_LLAMA_PREFIX_ANCHOR=0` disables that
policy. No precision, model, sampling, context or permanent production setting
changes accompany the upgrade.

## Cache investigation

The engine already defaults to `sleep_idle_seconds=-1`; it has no default idle
expiry. WorkerRouter user affinity instead expired after ten minutes and
conversation/prefix hints after an hour. [Router PR #9](https://github.com/DevXT-LLC/WorkerRouter/pull/9)
changes both defaults to 24 hours, preserving bounded maps, explicit pins,
health checks and spillover. Both explicit `ROUTER_USER_AFFINITY_TTL` and
`ROUTER_PROMPT_AFFINITY_TTL` must be updated to `86400` when deployments already
set them. Two admission regressions exercise a 30-minute idle user lane and a
two-hour prompt home with a faster competing cold worker. Full router tests:
89 passed, two ignored.

Read-only WorkConductor metadata showed a stable 341,320-byte system message
across inspected recent main turns and changing runtime/user suffixes. Router
history included a nearby 77,530-token cold input with 28,066.638 ms prefill, but
lacked a conversation correlation identifier. Timing alone does not establish
the exact cause. No WorkConductor change or speculative issue was made; no
conversation export is committed.

A recurrent model needs an actual checkpoint before a changed suffix; matching
system tokens alone is insufficient if all retained states are later. The new
[admission policy](../patches/llama-prefix-anchor.md) keeps the earliest valid
boundary for named prompts with more than one checkpoint, including after a
same-boundary refresh moves its entry. A newly evaluated earlier boundary
replaces the anchor. Existing token-prefix invalidation still runs first.
One-slot and unnamed policies are unchanged. Protecting one early state can
sacrifice another recent checkpoint. This is neither a larger allocation nor a
24-hour KV reservation; pressure, unloads and restarts still evict caches.
The actual-header regression simulates 40 turns/120 admissions and demonstrates
old-policy loss/new-policy survival, plus refresh, invalidation and capacity
edges. It is not a 40-turn physical-model test or an overnight retention claim.

## Builds and deterministic checks

The GPU-inaccessible owned builder used image
`sha256:c9d7cd30fb6f0d75280fda34b20cd70350dc69072ffebf564f0fffc92b8b330a`,
with runtime `runc`, no GPU devices and network disabled. CPU limits rose from
two CPUs/6 GiB to four CPUs/8 GiB for the final build. CUDA 12.8.93/GCC 13.3.0
compiled SM86/89/120 shipping targets with `GGML_NATIVE=OFF`, NCCL disabled and
CURL enabled. All nine patches and 15 managed source fingerprints match.

- Final oracle-free CUDA package regex: **73 passed, 306.78 s**.
- CPU package regex: **72 passed, 311.57 s**; no CUDA binary-inventory test.
- Focused CPU engine suites: **10 passed, 15.32 s**, including strict source
  hashes, atomic/storage/cache/MTP/checkpoint policies, managed patch lifecycle,
  real tiny-model acquisition and multi-conversation restore.
- Cross-pin lifecycle also passes independently: remove the old eight patches
  at the old pin, checkout v0.8.0, install nine, remove cleanly. This fixes an
  updater migration bug that would otherwise leave old edits blocking checkout.
- Four CUDA-linked CPU-only contract/CLI/real-model/cache checks pass in 2.10 s.
  CUDA loader stubs must be on the library path; earlier missing-stub failures
  are retained as environment failures.
- Six compose profiles: **1,875 source comparisons passed**.
- Final installation/native/TTS/ACE linkage has no missing libraries or Python;
  ACE resolves its own ggml base. Initial ACE linkage without the loader-stub
  export failed and was rerun correctly. The builder reports zero OOM kills;
  reclaimable page-cache pressure did reach its memory limit.

The optional download smoke had a stale final HTTP-count assertion. Both old
and new real engines produced five catalogue/estimation calls and 40 projector
HEAD probes, with one weight GET and working generate/repeat/stream/reload. The
expectation now matches both controls and reports individual counters.

## Qwen TTS

v0.8.0 changes `gen_audio` construction to default construction plus `init`.
The native factory supports both APIs and checks initialization before reporting
readiness. A syntax-only compile against v0.7 headers passes; the final v0.8
binary is built and exercised.

Upstream also moves codec RNG state from the shared model into each utterance.
The first buffered waveform matches v0.7 exactly; subsequent stochastic hashes
change. This is recorded as an intentional seed-lifecycle correction, not
bitwise audio preservation. The unchanged hash-pinned Python wrapper records
five sequential requests/six float32 buffers against the new CPU engine; the
native wrapper and an independent direct-worker run reproduce all six hashes.
The final factory passes that exact source/model/speaker-hash test. Historical
fixtures are retained unchanged; the new `llama_tts_model_v080_cases.jsonl`
distinguishes reference and native worker source hashes. CPU fixture selection
is explicit through `EZLOCALAI_TTS_MODEL_CASES`.

Historical composed/GPU model fixtures have not been regenerated for v0.8.0 and
must not be reported as newly qualified. Physical GPU speech quality and broader
long-form audio acceptance remain open.

## Physical RTX 4090 whole-engine comparison

The existing production worker was revision `13f8f71`, image
`sha256:00c5cf54c8c1e23f214adf68c0a226edb74926192de32a6d154542c2520559a4`.
All phases used its same image, resources, environment and read-only mounts,
with its full argv except the loopback binding. Model Qwen3.8-27B UD-Q3_K_XL,
context 230,000 (padded 230,144), batch 4,096, ubatch 1,024, Q4_0 K/V, MTP draft
maximum three, one slot, 12 GiB cache, 32 checkpoints and 8,192-token minimum
step were held fixed. The complete nine-file engine bundle was replaced
together; mixing ggml 0.27 and 0.28 libraries is invalid. Actual executable,
loaded library and inherited environment hashes were checked in each phase.
The same loader directory was prepended for both versions; no host `.env` changed.

The fixed public workload made 13 requests per old/new/old phase with no
inference retries, warmup generations or context reductions: two finite cold
prompts, three 256-token short decodes, four finite cache operations, a 130K seed
and three cached 256-token returns. **39 requests completed their functional
controls**. Input/evaluated/cached/generated/finish counts match. Both old-engine
runs match all 13 full signatures exactly. The new engine matches all six
finite-answer/cache cases, but **seven greedy output hashes and their MTP
counters differ**. The original strict text-equivalence gate therefore remains
**failed**; completed execution is not a passing equivalence result.

| Native metric | Old before | v0.8.0 | Old after |
| --- | ---: | ---: | ---: |
| Cold ~4K prefill, tokens/s | 2,557.12 | 2,581.60 | 2,534.05 |
| Cold ~16K prefill, tokens/s | 3,000.47 | 3,050.86 | 2,986.26 |
| Short decode median, tokens/s | 112.81 | 109.81 | 112.75 |
| Cached ~130K decode median, tokens/s | 77.22 | 79.11 | 77.21 |
| Cached ~130K first output median, ms | 202.92 | 205.53 | 204.03 |

Changed greedy trajectories make decode timings descriptive, not matched-output
speedup evidence. This upgrade alone is not a large prefill gain.

The fixed-continuation diagnostic used three prescribed public continuations,
pre-sampling full-vocabulary probabilities and unchanged original limits
(0.001 nats/token mean-NLL difference; 0.01 maximum token logprob difference).
Both private children used `--spec-type none`, because v0.7 leaves speculative
token probabilities as `1.0` placeholders. Production MTP settings stayed intact.
The diagnostic **failed**: the 42-token hash-map continuation changed mean NLL
2.627369 to 2.636198 (+0.008829), with maximum token logprob difference 0.274055.
Arithmetic changed 0.040581 to 0.040096, but maximum token difference 0.034918
exceeded the bound. The long target string, request and counts match, but its
26-token BPE sequences differ, preventing a fair tokenwise NLL comparison.
These limits were not relaxed. This small forced-continuation test is neither
a dataset perplexity benchmark nor evidence of universal quality loss/equality.
Upstream changes low-bit MMVQ reductions, RMS normalization and graph fusions;
the individual cause has not been isolated and no numerical workaround is added.

A **separate**, predeclared finite-task check used the original production MTP
configuration, twenty arithmetic/logic/code/extraction questions and one ~130K
identity-extraction question. It used no forced-answer grammar/bias and preserved
all case grades. Both versions score **20/21**, preserving every correct baseline
answer; both answer the repeated-update Python loop with `9` instead of `33`.
All 42 requests complete with cold counts and normal stops. This supports
non-regression only on these tested answers. It does not turn either failed
text/numerical equivalence gate into a pass.

## Restoration and limits

Every execution window requires an owned fresh idle drain, two healthy alternate
workers, periodic lease renewal and a watchdog. The preparation memory guard
may credit only the measured anonymous RAM of the original process that will be
stopped. Every actual launch rechecks real available RAM against the cache budget
plus an 8 GiB reserve; unit tests reject shortfalls. Metadata exports only hashes
of inherited environments. Full original container/image/environment/mounts/
configuration/argv restoration and two fresh matching router heartbeats are
required before owned drain release.

Earlier windows failed safely during read-only preparation, pre-stop RAM checks,
metadata validation and numerical-probe preparation. The probability validator
also rejected v0.7 MTP placeholder probabilities; a draft-max-zero diagnostic
aborted in v0.7 with an output-capacity assertion while MTP remained enabled.
These failures and restoration proofs remain in ignored raw evidence. They are
not erased or counted as successful inference runs.

The original image/container/environment/mounts/full argv are restored, with
production worker version `13f8f71`. Every executed window has a saved exact
restoration proof, two fresh matching router heartbeats and a released owned
drain. Final routed chat pinned to `DevXT4090` returns `READY`; the worker is
healthy and undrained. Rollback:
`ezlocalai:rollback-llama-v080-20261028-4090`, pointing to the original image above.
Only owned inference children, twelve verified remote staging directories,
redundant nonce aliases and the completed CPU builder are removed. Other images,
containers, caches and projects are untouched. No production `.env` changes were
made. Duplicate local transfer archives/bundles are removed after saving logs;
one custom engine bundle is retained for reproducibility.

No fresh full Docker image is built/deployed by this task. Other physical GPUs,
ARM/ROCm/Jetson/Pi/GB10, multi-GPU, vision, maximum model context, overnight idle
survival and unlock causality are unqualified. The previous vector-fallback
race warnings have not been rechecked with physical sanitizer runs on v0.8.0.
No universal text/audio quality or bitwise cross-version equivalence is claimed.
Compact evidence: `tests/fixtures/llama_v080_cache_lifetime_20261028_validation.json`.
Bulky raw logs, frozen plans and downloaded engines stay under ignored
`build/validation/llama-v080-20261028/`.
