# GB10 concurrent LFM voice — 2026-10-28

The GB10 has enough RAM for independent LFM instances. The earlier slowdown
was not solved by allocating more context slots alone. Private NVIDIA MPS
improved two-process text-plus-voice generation from roughly playback speed to
about 1.65× playback speed. Fresh-image requests exposed further interference;
per-instance MPS priorities improved that workload from RTF 1.077 to about 0.39,
over 2.5× playback speed, with no measured gaps in the final bounded screen. Quantization, KV precision and
voice-conversion quality settings were preserved.

## Investigation

The deployed llama.cpp 0.8.0 backend (`8642541b2`) advances realtime generation
with its own decode loop from `server_context_impl::process_realtime_slots()`.
It bypasses the ordinary shared text batch. With two slots on one engine,
normal text/vision and realtime operations still take turns on the server
thread. Two independent engine processes avoid that particular serialization
and retain a full context per instance, but ordinary separate CUDA contexts
still suffered contention. MPS enables cooperative CUDA execution across
processes; see [NVIDIA's description](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).
The configuration A/B demonstrates the improvement; no CUPTI/kernel trace was
collected to apportion the remaining cost between individual kernels.

Both instances used LFM Q4_K_M, 32000 context each, one slot each, full GPU
offload and f16 KV. MeanVC2 remained CPU/F32 with four threads. Automatic
batch/ubatch were 4096/1024 and 2048/512. Test containers had a 48 GiB RAM ceiling
and did not change the production worker or register as extra router workers.
The base image was `58b53ae4ad6cc06234a1775d076ef6c08b726f453804c3a2f4a85e2629b9e84b`.

The helper sent bounded synthetic text requests with a repeated roughly
1200-word system prompt, unique user markers and a 256-token output cap. Voice
turns requested the same short spoken status update. These are stress tests,
not a replay of the much shorter median MiniCPM requests in retained history.
The first normal turn includes cold-call effects; medians cover three normal
turns. RTF means generation wall time divided by produced audio duration.

| Profile | First PCM median | Normal RTF median | Worst playout lateness, 80 ms buffer |
| --- | ---: | ---: | ---: |
| Separate processes, ordinary CUDA | 414 ms | 1.021 | 153 ms |
| Separate processes, private MPS prototype | 302 ms | 0.605 | 0 ms |
| MPS prototype with context during input/output | 304 ms | 0.594 | 0 ms |

The ordinary CUDA screen completed 30 helper requests and nine voice responses.
The MPS screens completed 24/nine and 30/eleven respectively. These screens completed
without API/protocol errors. The ordinary screen reached its helper-request cap
near the end, so the final response is not evidence of sustained contention;
the three normal-turn measurements were under load. MPS screen load continued
until the voice suite finished. A later final coherent-text screen had median
RTF 0.582 but one 400 ms tool-result gap, so equal priority is not sufficient
for a smooth-call guarantee. Reusing the same image also hid the fresh-image
cost: varying actual pixels on every request produced normal RTF 1.077 and up
to 321 ms lateness before a test-container TTL ended the trial. Only its three
completed normal turns are used for that comparison; its incomplete suite is
not counted as a pass.

A priority prototype completed 27 distinct fresh-image requests (all recognized
as red) and eleven voice responses, including tools, microphone input and
context injected during output. Its median normal first PCM was 188 ms and RTF
0.385, with no measured underruns. The native implementation then repeated the
trial using the public model name, so replica preference was automatic: 27
unique frames and eleven responses passed, median first PCM 228 ms, RTF 0.390,
no measured underruns. Both engines' actual process environments confirmed the
same private MPS pipe and priorities 0/1.

Some diagnostics were excluded: an accidental overlap of two load drivers, an
unready-service startup probe and the incomplete container-TTL trial. These are
retained separately in raw artifacts. No claim is made that arbitrary numbers
of competing helper streams can run without affecting calls.

Ordinary REST text requests were also checked explicitly: the default and
`modalities:["text"]` returned text without hidden audio output. This was not
an accidental speech-generation workload disguised as text.

## Text injected into a call

The benchmark now supports `--interleave-context-words` to send a
`conversation.item.create` after the first PCM packet, while continuing to
consume every audio delta. It verifies acknowledgment and checks that a later
response uses a newly injected project/task. `--audio-context-words` injects
working-agent text halfway through paced microphone input. Context is queued
for the next response; an utterance already in progress is not rewritten.

An uncontended native/converted screen completed 12 responses with 1200-word
updates, no measured playout gaps, and 41–52 ms context acknowledgments during
speech. A subsequent next-response check identified the new project/task.
A stricter diagnostic initially returned stale status when the update repeated
1200 words of the old project/task and contradicted them in its final sentence.
An internally consistent update plus explicit latest-status instructions
correctly changed the spoken answer to integration tests for Marigold. This
separates model/prompt consistency from transport delivery; it does not establish
robustness to contradictory full-history snapshots. Final checks use the new
project/task to distinguish injection from prior session knowledge.

First PCM is not first heard speech. Leading quiet audio and the client buffer
remain part of perceived latency. Human voice quality and similarity, long-call
stability, maximum context, two simultaneous converted calls, and other GPUs
are not certified by these measurements. The existing converter capacity is
one, although two LFM instances can support a voice call plus a helper request.
MeanVC2's bandwidth remains 16 kHz with 24 kHz wire PCM.

## Implementation

`EZLOCALAI_CUDA_MPS=true` opts into a private MPS server. The default is off.
A CUDA-free native supervisor creates private 0700 pipe/log directories, launches
an owned foreground controller, checks server readiness, and starts the worker
with the private pipe directory inherited by its model processes. It forwards
shutdown signals and returns the worker's exit status. It waits for the worker
process to exit fully before stopping/reaping MPS: an earlier RAII-only attempt
exposed retained CUDA primary contexts that outlive local backend objects.
No global daemon, host compute-mode change or external MPS pipe is commandeered.
`LLM_CUDA_MPS_CLIENT_PRIORITY` supplies 0 (normal) or 1 (below normal) per
configured model. Engines receive the selected value through an argument-vector
`/usr/bin/env` launch, not mutation of the threaded worker environment. Calls
prefer available normal-priority replicas; ordinary helper requests prefer
available below-normal replicas. Explicit pins and fallback capacity are
preserved. NVIDIA calls priority a performance hint, not an execution-order
promise; see [client priority](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html).
Nonidentity numeric device selections are rejected to avoid MPS ordinal
remapping; GPU UUID selections preserve the intended devices.
External MPS can still be used with managed mode off. Unavailable MPS fails
startup explicitly with a log location.

The lifecycle test covers disabled/external setup preservation, private
permissions, readiness, startup failures, signal forwarding, exit status,
recursive-bootstrap prevention and child reaping. All platform package gates
and the `cuda_package` target include it; CPU/ROCm defaults do not require MPS.
Production instructions and the three-instance MiniCPM/LFM/LFM example are in
`VOICE_SELECTION.md`. No WorkerRouter code change is required for existing
replica slot accounting.

## Production MiniCPM adjustment

At the user's request, the actual GB10 worker was changed to one MiniCPM slot
with 32768 context. Only these environment entries changed:

```dotenv
LLM_MAX_TOKENS=32768,32000
N_PARALLEL=1,1
```

The image, model list, quants, LFM settings and mounts stayed unchanged. A
require-idle, owner-scoped router drain and independent rollback watchdog
protected the cutover. Two fresh heartbeats confirmed context and capacity,
and a routed MiniCPM smoke returned OK. Rollback image tag:
`ezlocalai:rollback-gb10-minicpm32k-20261028`; its private environment backup is
under `/home/josh/ezlocalai/build/validation/lfm-concurrency-20261028/minicpm-cutover/`.
The previous `ezlocalai:rollback-gb10-voice-20261029` also remains available.
Production does not yet run the new managed MPS binary or the second LFM copy.
After merging/deploying this change, enable MPS and add the duplicate LFM entry
as documented; do not infer that updating the environment alone updates an old
worker binary. No MiniCPM-to-LFM task-quality acceptance is claimed.

## Reproduction

Use private connection JSON as described by `bench_lfm_realtime.py`. Against a
direct two-instance worker, run:

```sh
python3 benchmarks/bench_lfm_concurrency.py --connection /tmp/worker.private.json --output build/validation/concurrent --audio /tmp/question.wav --inject-context-words 1200
```

Add `--synthetic-vision --interval 1 --max-output-tokens 64` for intermittent
fresh-frame vision. Each frame has changing pixels and a recorded distinct hash;
`--image /tmp/red.png` instead reuses a fixed local PNG. Through a router, select the public model using `--background-model
LiquidAI/LFM3-Omni-2B-GGUF` and a worker pin in connection JSON; `#2` is a
direct-worker replica selector. Raw JSON/WAVs and staging scripts are kept in
ignored `build/validation/lfm-concurrency-20261028/`. The committed validation
fixture records compact results and limitations rather than credentials or
bulky audio. Final native supervisor checks are recorded there as well.


## Final validation and deployment handoff

The final native implementation completed 42 voice responses and 98 helper
requests across continuous text, fresh vision, native voice and converted voice
screens: zero errors and zero measured playout lateness with an 80 ms buffer.
All 71 synthetic image requests were recognized as red. The last image recipe
changes pixels in every tile and includes a per-run salt; the converted screen
completed 28 such unique frames with normal-turn median PCM 167 ms and RTF
0.364. The native-voice screen completed 16 unique frames/nine responses, with
one normal-turn observation at 115 ms first PCM and RTF 0.338. Continuous text
with native priority preference produced median PCM 233 ms and RTF 0.465.
These are bounded observations, not percentiles. Coherent new task updates were
reflected in microphone and next-response captions in all four final profiles.

The complete final `cuda_package` target was rebuilt on the GB10; its 77-test
package gate passed without a Python reference checkout (326.03 seconds).
The initial package attempt inherited disabled benchmark capabilities and
failed a capacity fixture; the clean-environment rerun and final gate passed.
The standalone lifecycle/CSV test also passed on x86 and under ARM QEMU using
a mocked NVIDIA control program; the QEMU result is not GPU validation.
Real model children shared the private MPS pipe, with actual priorities 0/1;
all owned worker/controller/server children were reaped without shutdown errors.
The candidate recorded zero cgroup memory-limit/OOM events and zero throttling.

A production-ready configuration snapshot is staged privately at
`/home/josh/ezlocalai/build/validation/lfm-concurrency-20261028/proposed-qos.env.private`.
Compose expansion was checked and its mounts match production. It is **not
applied**: deploy the new implementation first, then merge the seven documented
settings into the current environment, take a fresh rollback snapshot, perform
an idle router drain, recreate, and verify actual process priorities and fresh
router capacity/context. Regenerate the snapshot if production settings change;
do not overwrite newer unrelated settings with a stale environment copy.

Remaining work: verify the actual post-merge image rather than relying on build
submission acknowledgment; enable this GB10 profile; human-listen and investigate
leading quiet audio without clipping speech; exercise long calls/context budgets
and representative MiniCPM tasks; test multiple helper streams and other GPUs.
The client must mark latest status as authoritative and avoid contradictory
snapshots. WorkConductor remains issue-only, and no WorkerRouter changes are
required for the existing replica accounting.
