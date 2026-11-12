# 5090 Cold-Prefill Profile: 2026-11-11

Evidence only; no MTP runtime patch or new GPU trace. This note is separate from
the fleet report. Machine-readable record: [profile fixture](../tests/fixtures/prefill_profile_20261111_validation.json).

## Capture

Host `josh@192.168.1.135`, report directory
`/home/josh/prefill-experiments/profiles/3374a9cd3ad3452280ad98cdd9bc09d7`.
Artifacts: `cold.nsys-rep`, `cold.sqlite`, `kernel-summary.txt`,
`transfer-api-summary.txt`, `nsys-launch.log`, and `profile-status.json`.
Artifact hashes and the immutable image/revision are in the fixture.

One fresh baseline request, Qwen Q3_K_XL, context 262144, batch 4096,
ubatch 2048, Q4 KV, draft-MTP enabled, voice resident. Tracing started after
health/calibration, excluding model startup. Capture and task quality passed;
64,183 prompt tokens took 21.74856 s; 483 generated tokens took 3.39881 s.
This instrumented run is not an uninstrumented throughput comparison.

## Inferred Phase Split

No NVTX request/MTP ranges exist. Use the first `mul_mat_vec_q` start at
21.795208927 s on the capture clock as a **kernel-regime proxy**, not an exact
TTFT boundary. Last MMQ ends at 21.786186185 s; first `cudaGraphLaunch` starts
at 21.822345514 s. Moving the split across these alternatives changes prefill
copy duration only from 449.377 to 449.897 ms. The server's prompt-evaluation
timer is a separate clock interval, not this capture timestamp.

| Measure | Prefill proxy | Decode proxy |
| --- | ---: | ---: |
| Summed kernel time | 19.8343 s | 3.0979 s |
| H2D duration / decimal GB | 372.892 ms / 15.125 | 9.425 ms / 0.237 |
| D2H duration / decimal GB | 76.917 ms / 1.785 | 20.673 ms / 0.499 |
| H2D / D2H event count | 539 / 321 | 4,604 / 1,333 |
| Summed stream-synchronization API time | 8.7132 s | 2.6290 s |

Prefill kernel shares: all MMQ **52.5%** (two dominant types: **51.9%**),
large flash attention **30.8%**, GDN `gdn_single` **4.5%**. These use the
prefill kernel sum as denominator, not wall time. Whole-capture shares were
44.9%, 26.7%, and 3.9% respectively for the two MMQ types, large FA, and GDN.

Reproduce from SQLite opened with `mode=ro`: join kernel `shortName` or
`demangledName` to `StringIds.id`; join memcpy `copyKind` to
`ENUM_CUDA_MEMCPY_OPER.id`. Split at `21795208927` ns and sum clipped durations
`max(0, min(end, phase_end) - max(start, phase_start))`. The fixture retains
integer nanoseconds, bytes, and boundary sensitivity for arithmetic checks.

## MTP Attribution Limit

All prefill copies total **449.809 ms**, approximately **2.1%** of logged prompt
evaluation. Even eliminating every such copy gives only that copy-only budget;
MTP owns an unproven subset. This is **not** an upper bound on total MTP overhead
or a predicted speedup: CPU copying, synchronization dependencies, and draft
compute are not isolated. API wait duration includes GPU execution and cannot
be added as removable handoff cost. MTP acceptance (362/363) confirms use, not
event ownership. Similar transfer sizes do not prove MTP attribution.

Across both phases, copies occupy 479.907 ms, with 467.073 ms not overlapping
kernels. The GPU-event span also contains 1.648 s of gaps. Neither occupancy
nor gaps establish a dependency critical path. Stop at this evidence; no risky
device-handoff patch is justified by the small demonstrated copy budget.

## AutoDefaultScope

Do not generalize this screen into a 5090-wide auto-cap increase. With at least
28 GiB free, `src/llm_batch.cpp` selects batch 8192 before capping long-context
Qwen ubatch to 1024. Relaxing that cap exposes **8192/2048**, whereas the measured
baseline is **4096/2048**. The cap matches a model-name substring, not the resolved
GGUF quantization, so a GPU-only exception would also affect larger quants.
Startup fallback can reduce ubatch or GPU layers, but does not establish
full-context memory safety. Retain the qualified explicit setting; any future
auto exception needs resolved-artifact eligibility and validation of its actual
batch pair, KV/context settings, and resident-voice headroom. No default change
is implemented or recommended for unqualified models or other GPUs here.

## Recovery and Fix

**Automatic restoration failed.** The owned probe was removed, but lowercase
Docker `error: no such object` failed a case-sensitive absence check, preventing
the original container's restart. The parent manually issued `docker start`.
No automatic `restored` receipt is claimed or manufactured for this run.

After that restart, the original container ID, immutable image, configuration
fingerprint, both Qwen/voice argv lists, and `/8091/health` were independently
verified. Only watchdog PID `999274`, matched to this UUID and drain owner,
was terminated, with pidfd-confirmed exit. Owner-scoped resume of
`prefill-da86526d9f744543a91832eb8a504279` succeeded; the router then reported
`draining=false`. These are retrospective manual recovery observations.

The helper and standalone watchdog now match missing-container errors
case-insensitively while retaining fail-closed behavior for daemon failures
and unowned containers. Future restoration failures retain safe diagnostics.
Offline regression coverage includes lowercase absence after removal, an
already-removed probe, strict non-optional inspection, and daemon failures.
Run: `python -m unittest discover -s benchmarks -p test_prefill_profile.py -v`.
