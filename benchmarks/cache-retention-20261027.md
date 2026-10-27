On a controlled RTX 4090, fixing checkpoint accounting improves shared-cache
return time to first token from **340 to 258 ms** across all measured returns.
The eight returns that previously lost their saved tail improve **2.206 to
0.258 seconds (88.3%)**. All **236 measured outputs match** across calibration,
ABBA pressure and separate finite/independent qualification. Decode remains
approximately 89 tokens/s. These gains come from avoiding re-evaluation within
the existing 12 GiB ceiling; no permanent settings change.

The cache retention change counts shared checkpoint allocations once and avoids
unnecessary oldest-checkpoint eviction when refreshing the same token boundary.
It addresses avoidable reprocessing when conversations branch or compete for
cache space. It does not change the model's arithmetic or raise a configured
memory ceiling.

## Retention behavior

Cached prompts already share immutable target/draft checkpoint bytes. Charging
each owner for those bytes could reject a save or evict a useful conversation
although the retained allocations fit. Admission, eviction, protected-restore
checks and reported cache size now use the allocation union. Full target/draft
snapshots, speculative CPU state, checkpoint CPU vectors and spare capacities
remain separately charged. Detached copy-on-write allocations count separately;
disabling sharing charges each new copy. A final check verifies actual cloned
capacities before publishing the saved state.

For example, two 192 KiB snapshots sharing a 1 MiB checkpoint need approximately
1.375 MiB, while the old accounting charges 2.375 MiB. The production allocator
regression verifies both branches survive a 2 MiB ceiling, along with protected
restore, independent-allocation, copy-on-write, spare and failure cases.

A second fix removes an existing checkpoint at the same token count before
making capacity room. Refreshing a full list previously could evict an unrelated
oldest checkpoint and then finish below capacity. Minimum-step cleanup and the
current-task exception retain their prior behavior, including deliberate removal
of close older boundaries. See [the patch contract](../patches/llama-cache-retention.md).

These decisions do not modify checkpoint bytes, batch splitting policy, weights,
precision, sampling or context. The budget retains its previous scope: cache
snapshots and spare buffers. Live-slot-only checkpoint allocations remain runtime
overhead. Allocation-free identity scans are quadratic in checkpoint fields;
cache-depth measurements are required to assess their CPU cost.

## Physical shared-cache ABBA and independent controls

Four fresh engines run baseline/candidate/candidate/baseline. Each phase seeds
eight conversations sharing a public-source prefix, runs two finite/new-turn
canaries and returns twice to all eight conversations. Each seed input contains
25,758 tokens; each shared freeform response has a 256-token budget. Requests,
messages, output hashes, generated counts and finishes agree across all 104
measurements; each build's evaluated/reused count signatures repeat in its pair.
Evaluated plus reused native counts match actual input usage throughout.

| Return cohort | Samples per build | Baseline median TTFT | Candidate median TTFT |
| --- | ---: | ---: | ---: |
| All returns | 32 | 339.800 ms | 258.251 ms |
| First-sweep branches that lost their saved tail | 8 | 2,205.650 ms | 258.119 ms |
| Same branches in the second sweep, already retained | 8 | 257.359 ms | 258.492 ms |

Complete tail reuse improves **24/32 to 32/32**. Total re-evaluation falls
**41,704 to 128 tokens**, while reused input rises 782,552 to 824,128 tokens.
Each affected return evaluates 5,201 tokens before the fix and four afterward.
No return is fully cold in either build. Native return decode medians remain
88.906 and 88.948 tokens/s. These timings include matched trace logging, cache
scans, snapshots/transfers, sampling and inference on the GPU host. They measure
bounded restore/TTFT behavior, with two runs per build; they do not establish
higher cold-prefill/decode kernel throughput or production-traffic averages.

Separate A/B qualification adds **96 exact finite controls**: 64- and 128-token
budgets exercise seed/replay/append/rewind and latest-turn instructions, then
eight independent-prefix conversations return twice each. All paired outputs,
requests, generated counts, finishes and evaluated/reused counts match. Both
builds retain **16/16 independent returns**, with no full re-evaluation.
The two-build calibration adds 36 matching measured outputs, giving 236 total.

All compared engines retain the actual 12,288 MiB cache limit. Docker OOM state,
cgroup OOM events and host OOM-kill deltas remain zero; observed cgroup swap peaks
are zero. The shared ABBA reports five admission evictions per baseline phase
and zero per candidate phase. Across the separate multi-suite qualification,
both builds evict six older entries while moving between suites, without losing
independent return coverage. Post-update cache charges remain below the ceiling;
they are not transient allocation or process-RSS measurements. Evidence is in
`tests/fixtures/cache_retention_cuda_20261027_validation.json`.

## Build and regression validation

All seven focused engine checks pass in 16.05 seconds. All 72 tests in the exact
CUDA Dockerfile package gate pass in 328.89 seconds without the Python reference
checkout. The isolated builder uses image
`sha256:d7321ef6403b8dbc3b692d4e989ee38bf762d2a33080f239a36c9727ec6cd8e5`,
two CPU cores, 6 GiB RAM, no network and the runc runtime with no GPU devices.
These package results do not constitute hardware inference validation.

Detailed output confirms 18 storage scenarios, 163 MTP checks and 364,393
checkpoint coverage/allocator checks. Old accounting and old refresh ordering
fail deliberate negative controls. All eleven managed source fingerprints
match the rebuilt pin plus seven patches. Managed installation checks cover
pristine and old-prefix upgrades, repeated application, metadata and complete
removal, including direct removal from the previous six-patch release. All four
platform recipes carry patch seven.

Only the server implementation library differs in the physical comparison:
baseline `2fbc02b3…`, candidate `8c918e13…`. The deployed image is
`sha256:b56ebbeb9efb1419e5d060d9df12ed5637cc683972f2ce5036d66fad28d6239c`
(worker `3412754`). The original common library remains `c9fd8582…`, and the
actual 4090 CUDA library remains `4d225f16…`. Weights (`8c2a45ff…`), projector
(`83ee4f4f…`), 230,000 context, 4096/1024 logical/physical batch, Q4_0 K/V,
MTP maximum three, one slot, 32 checkpoints and 12,288 MiB cache ceiling remain
fixed. The test child uses private loopback transport and read-only data binds;
both compared builds use the same trace verbosity five for cache accounting.
No production environment setting changes.

The preliminary two-build calibration measures 36 requests with matching
paired outputs, including eight shared-prefix seeds, finite/latest-turn
canaries and eight returns in each build. Identical eight-state snapshots charge
approximately 11,674 MiB before deduplication and 10,627 MiB after. An outgoing
canary adds about 906 MiB. Reconstructed prospective sums put the former above
the unchanged ceiling and the latter below it, explaining the observed eviction.
Those are estimates from rounded trace charges, not a new allocation counter;
actual admission/restore behavior and source budget checks provide the evidence.

The original production container is retained throughout owned idle-router
leases. Rollback tag: `ezlocalai:rollback-round5-20261027-4090`. Each completed
window verifies exact image/environment/mount/config/full argv restoration and
two fresh matching router heartbeats before releasing its own drain. A final
routed chat pinned to the restored worker returns the exact answer `READY`.
All owned CPU/GPU children and remote staging directories are removed; the
rollback tag and local evidence remain. Fresh
images, other physical GPUs/platforms, maximum context, vision and full-list
checkpoint refresh on hardware are not requalified by these checks.

## Remaining investigation

Ranking saved conversations by raw common-prefix length can overestimate reuse
on a hybrid recurrent model. The effective reusable boundary also depends on
saved positions, available checkpoints, final-token replay and draft predecessor
state. A future selector must use the same complete eligibility rules as the
restore path; a token-count-only heuristic is insufficient. This change retains
the existing selector.

The router's recent history describes mixed input depths and workloads rather
than a controlled hardware comparison. Separate diagnostic traces on the
unchanged deployed engine identify these next targets:

| Controlled case | Timed kernels | Largest categories by summed kernel duration |
| --- | ---: | --- |
| Cold 4K | 16,167 | Quantized matrix multiply: 71.9% |
| Cold 16K | 48,339 | Quantized matrix multiply: 69.3% |
| Cached 130K, 103 output tokens | 69,009 | Matrix-vector multiply: 42.6%; attention: 25.3%; Q4 KV expansion: 24.5% |

Cold matrix-multiply kernels primarily process IQ3_S/IQ4_XS weights. The deep
case includes 864 Q4-to-half launches: 27 MTP verification batches, 16 target
attention layers, and separate K/V conversions. Launch sizes follow the padded
130K context. The dispatcher converts the entire active quantized KV cache to
F16 for four-query verification. Fusing that conversion into the existing F16
attention tile loader is a concrete next investigation. The existing direct
quantized vector path also quantizes queries, so merely widening its dispatch
threshold would change arithmetic. See [the implementation handoff](q4-kv-attention-handoff-20261027.md).

These are diagnostic category shares, not speedup predictions. Missing importer
support interrupted live export before the final uninstrumented control. The
matching local importer recovered all three captures; finite request/answer
signatures agree with their preceding uninstrumented controls. All captures
warn about event completeness, and the deep trace has graph-ID annotation
errors. CUDA synchronization durations largely overlap recorded GPU execution;
they cannot be added as independent CPU overhead. No kernel change is promoted
from these traces. Compact evidence is in
`tests/fixtures/cuda_kernel_profile_20261027_validation.json`.
