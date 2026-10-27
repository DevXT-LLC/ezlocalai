The 2026-10-26 CUDA measurements support keeping 3090V at physical batch 1024. Batch 2048 was effectively flat and 4096 was slower. The unlocked 4090 currently reaches 2704 prompt tokens/s on the fixed 49,993-token workload. The J common-model run is retained only as **exploratory, contended data**: the user confirmed unrelated game build/testing on J during this session. It is not a clean locked-card baseline and cannot predict unlock gains. No production settings were promoted.

The compact results and exact input/output fingerprints are in [unlocked_prefill_20261026_validation.json](../tests/fixtures/unlocked_prefill_20261026_validation.json). [unlocked_fleet_20261026_validation.json](../tests/fixtures/unlocked_fleet_20261026_validation.json) records GPU identities, drivers, RAM, full artifact hashes and UTC unlock evidence. Native revision was `c42148868a04d05171cdfc3bca7840f227b97216`; the engine pin was `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Worker, llama-server and its recorded libraries were byte-identical across the inspected J, V, 4090 and 5090 deployments, although image IDs differed.

| Measurement | Model SHA prefix | Declared context | Batch / physical batch | Cache RAM | MTP max |
| --- | --- | ---: | --- | ---: | ---: |
| V production-weight screen | `8c2a45ff85e7` | 230,000 | 4096 / varied | 12,288 MiB | 2 |
| 4090 deployed baseline | `8c2a45ff85e7` | 230,000 | 4096 / 1024 | 12,288 MiB | 3 |
| V common-model proxy | `00cf92e666c6` | 220,000 | 4096 / 1024 | 12,288 MiB | 2 |
| J common-model exploratory run | `00cf92e666c6` | 220,000 | 4096 / 1024 | 12,288 MiB | 2 |

All measured engines used one slot, flash attention and q4_0 K/V cache. The common-model J/V launches additionally fixed prompt/decode CPU threads at six, 32 checkpoints and checkpoint step 8192. Actual child argv, model and projector hashes matched across all four launches, as did the selected performance environment fingerprint. J and V production models have the same filename but different bytes: the J/common model is 13,441,059,904 bytes; V/4090 production weights are 13,146,393,504 bytes. The projector SHA is identical. The common file was copied into an owned read-only benchmark directory; production weights stayed intact. Throughput differences between the two V suites cannot be attributed to one setting because their weights and declared contexts differ.

The V physical-batch screen ran in order 1024, 2048, 4096, 1024. Every cold request evaluated 49,993 tokens with zero cached tokens and returned the identical finite answer `Python`.

| Physical batch | Native prefill tokens/s | Difference from bracketed 1024 mean |
| ---: | ---: | ---: |
| 1024, first | 1432.95 | +1.04% |
| 2048 | 1414.95 | −0.22% |
| 4096 | 1202.75 | −15.19% |
| 1024, repeat | 1403.34 | −1.04% |

The two 1024 runs average **1418.14 tokens/s**. The 2048 result lies within their spread; 4096 loses throughput while increasing observed cgroup memory. These rejected candidates did not receive full quality or maximum-context capacity qualification. This screen establishes no benefit from raising V's physical batch, and leaves its environment unchanged.

| Current engine measurement | Cold prefill tokens/s | Decode tokens/s | Cache-return wall time | Evaluated / cached return tokens |
| --- | ---: | ---: | ---: | --- |
| 4090, deployed production engine | 2704.00 | 111.75 | 0.421 s | 4 / 49,989 |
| V, common model, two repetitions | 1455.48 | 85.42 | 0.431 s | 4 / 49,989 in each |
| J, common model, contended exploratory run | 1213.90 | 72.63 | 0.667 s | 4 / 49,989 in each |

Prefill and decode rates come from native engine timing fields. Decode uses three cold, 60-input-token requests generating 256 tokens each; J and V each repeat all three, and each displayed common-model decode rate is the median of six measurements. V's cold prefill runs were 1464.13 and 1446.83 tokens/s; J's were 1208.56 and 1219.23 tokens/s. All four cold request/messages/model hashes, finite cold/replayed answers and decode outputs by trial matched. Decode request hashes matched by trial too. The 4090 decode outputs differ from V's, with different weights and MTP controls; these bounded checks do not establish general model-quality equivalence.

The 4090 harness called the already-running llama-server through `docker exec curl`; its 0.421-second client wall time includes Docker CLI startup. J and V used the same transport to isolated engine launches at a private host-loopback endpoint. Neither path measures a complete routed worker request, and client wall times should not be treated as comparable network latency. A cold prompt here means zero prompt-cache reuse, not an uncached disk/model load. The replay checks demonstrate returning to one conversation after three short intervening requests; they do not qualify arbitrary cache depth.

All cold suites used request SHA `6b1b3caa10ee64f011072e4174f0d087f7d17e09b6ee85f8093e36414d6a2c6e`, messages SHA `06e86b02f2b0e8577d6aa744c68b157939d6f42ed0aa7101d1aaa0a08f58db09` and finite output SHA `2945b0d6616d16592538930914f4a2bc539b8d976a0b9200b3e82540f3c07c25`. Sampling was temperature zero, seed 42 and thinking disabled, with no inference retries or context downgrade. The result fixture preserves every decode request/messages/output hash and replay request hash.

Paging counters separate whole phases, including lifecycle/model loading, from the observed benchmark windows. Swap counts are pages; major-fault counts are events.

| Host / repetition | Whole-phase host swap in / out | Window host swap in / out | Window host major faults | Whole-phase child major faults |
| --- | --- | --- | ---: | ---: |
| V / 1 | 2089 / 5044 | 788 / 0 | 493 | 613 |
| V / 2 | 13 / 0 | 2 / 0 | 482 | 0 |
| J / 1 | 74 / 674 | 74 / 0 | 24,930 | 4 |
| J / 2 | 0 / 0 | 0 / 0 | 6,065 | 0 |

All four temporary cgroups reported zero swap and no OOM events. J's much larger host major-fault counts than its child phase counts demonstrate substantial activity outside the benchmark child. Host counters cover other processes, and periodic window boundaries can miss subsecond activity: zero observed window swap-out or zero cgroup swap does not establish that the host never paged. V's first-phase 18.53 GB cgroup peak includes reclaimable file cache and is not an OOM by itself.

The standalone V dense GEMM diagnostic produced raw 8192³ medians of **159.25 TFLOPS FP16→FP32**, **157.36 TFLOPS BF16→FP32** and **286.78 TOPS INT8→INT32**. All six cases, including 4096³, passed 32 sampled exact integer dot-product references. **Operator acceptance failed clock plausibility**: NVML reported P8, 210 MHz SM and 405 MHz memory even during 100% utilization and 325–462 W. These numerical results remain diagnostic observations; they support no clock-normalized or causal unlock claim, and Tensor Core instruction selection was not independently profiled. [The probe fixture](../tests/fixtures/gpu_compute_probe_20261026_validation.json) retains the full scope and cleanup evidence. J's dense run was not started because desktop/Brave GPU activity and free VRAM failed unchanged guards; no 4090 dense result is available.

J and V are both RTX 3090 Ti, but J uses driver 570.211.01 and V uses 615.71.09. J's upstream PCIe port is physically limited to x4; the actual benchmark windows observed PCIe 4 x4 on J and PCIe 4 x16 on V. J had 31% total GPU utilization before its run, with Code SM samples of 16–19% and Brave samples of 12–15%; those applications stayed running. CPU, RAM, background activity and power/temperature behavior also differ. The user also confirmed another agent was building/testing a game on J during this session. Its exact overlap and CPU/GPU contention were not isolated; this makes the J run exploratory rather than a clean locked-J baseline. Any archived J/V ratios describe these observations only and are unsuitable for predicting unlock uplift. V boot-unlock success is anchored at **2026-10-25 13:11:45.282904 UTC**; 4090 at **2026-10-26 02:18:08.390111 UTC**. Historical samples require individual timestamp classification. No causal unlock percentage or comparison against historical runs with different prompt lengths is supported.

The V screen and both J/V common-model controllers held owned router drains and temporarily stopped each retained original production container. Each isolated launch used the same immutable image with inherited binds read-only. The controllers restored the exact original container/image/configuration/environment/mounts/full engine argv, verified healthy state and two fresh heartbeats with unchanged stable metadata, then released their leases. The 4090 production engine was never stopped or reconfigured; its process/configuration identity and two fresh heartbeats were verified afterward. The dense probe left the original worker resident and removed its owned diagnostic process/container before releasing its drain. No permanent environment, model, clock, power or unlock settings were changed by these runs. A's dotenv ubatch 1024 is user-reported; its effective process setting remains unverified through the available API.

After the completed GPU runs, the exact owned V common-model copy (13,441,059,904 bytes, SHA `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2`) and its ownership marker were removed, then their unique benchmark directory was removed. Cleanup verified the marker content/hash, regular-file ownership/mode, exact copied inode, full model hash, expected two directory entries and absence of remaining benchmark child containers/binds. V's healthy original production container/configuration/full argv was checked before and after; J's production source remained intact with its verified file identity. Operator and audit directories remain available. The sanitized cleanup proof is `build/validation/unlocked-prefill-20261026/common-model-cleanup-proof.json`, SHA `66ea31f8341d807dc5992b2553ef366122892bfd3c335708982c1074c22d8cd2`; the result fixture records its compact facts.

The remaining unlock question requires a same-card, same-driver locked/unlocked comparison with controlled background load; these observations cannot resolve it. A future clean J baseline needs a fresh plan, current container identity and a quiet host window. J and 4090 dense diagnostic runs are also unmeasured in this evidence. Larger batches on V received no production promotion, and these bounded engine workloads do not replace full quality or maximum-context qualification.
