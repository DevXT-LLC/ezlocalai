# Isolated Perplexity Tool

2026-11-11: `llama-perplexity` was already configured in the existing CUDA 12.8
build. A read-only dry run confirmed only a new implementation object/library
and main object/executable were needed. The target-specific `build.make` rules
built those two objects and linked the two new artifacts in about two seconds
(17:49:56 to 17:49:58 UTC). No CMake reconfiguration or dependency rebuild ran.

The compiler used the same pinned image
`sha256:85764c1a7540b36d143faa494fdb3bac6927fe99b59c8d6a65f31261e2a6ee12`,
with runtime runc, no network, no GPU device requests, `NVIDIA_VISIBLE_DEVICES=void`,
4 CPUs, 12 GiB memory, no swap allowance, and a 240-second build deadline.
Container `experimental-prefill-perplexity-cuda128-20261111` exited 0 and is not
running. No application or GPU kernel was executed.

## Artifacts

Both new files are in:
`/home/josh/repos/xtsys/ezlocalai-prefill/build/validation/experimental-prefill-gdn-cuda128-20261111/build/bin`

| New file | SHA-256 |
| --- | --- |
| `llama-perplexity` | `8a7157529486c50ced35fa7e087f02b7c06134eeeba3ec8db6499ae1a4607428` |
| `libllama-perplexity-impl.so` | `051cec206bbde950e3a5d72170d911c8ef83bc3ddb085c934f1ec4f1e1807d28` |

Copy both files into the staged bin directory; the executable needs its new
implementation library plus the already-staged engine libraries. Existing
GDN/server/operator binaries and all 24 existing bin entries are byte-identical
to their pre-build checksums. `CMakeCache.txt` is unchanged. Existing manifests
and pinned hashes were not edited.

Evidence in the scratch root: `perplexity-impl-build.log`,
`perplexity-exe-build.log`, `perplexity-existing.sha256`,
`perplexity-existing-check.log`, `perplexity-artifact-sha256.txt`, and
`perplexity-static-relocations.log`. Static `ldd -r`, using the driver stub only
for this GPU-free inspection, found no missing dependencies or undefined symbols.
Never include that stub directory in the real evaluation loader path.

## Later Quality Check

These are source-checked invocation examples, NOT executed tests. Run only under
the parent's owned drain, separately from the speed experiment. Keep the candidate
bin first in `LD_LIBRARY_PATH` and use the same CUDA 12.8 image/runtime library
selection for all quality runs.

Freeze separate public-code and public-text corpus files, their source/revision,
and SHA-256 before comparing models. Do not grade generated responses as target
log probabilities. For each corpus, use the identical tokenizer/vocabulary/token
IDs and fixed context, logical/physical batch, attention and KV settings. The KL
path consumes the reference file's token IDs; vocabulary-size equality alone
does not prove token mapping compatibility.

A bounded initial screen can use context/batch/ubatch 512 and two chunks. Each
corpus must tokenize to at least 1,024 tokens; this evaluates 510 target tokens
(the latter 255 of each chunk). This is a smoke screen, not a guarantee of quality
at long context or across workloads. Extend the fixed evaluation set separately.

```bash
export GGML_CUDA_EXPERIMENTAL_GDN_SLICE64=0
shared=(-c 512 -b 512 -ub 512 -ngl 99 --chunks 2 -ctk f16 -ctv f16)

# Baseline model: teacher-forced reference on the frozen corpus.
timeout --kill-after=10s 300s "$BIN/llama-perplexity" \
  "${shared[@]}" -m "$BASE_MODEL" -f "$CORPUS" \
  --save-all-logits "$REFERENCE"

# Reference serialization/control floor: the baseline against its own export.
timeout --kill-after=10s 300s "$BIN/llama-perplexity" \
  "${shared[@]}" -m "$BASE_MODEL" \
  --kl-divergence --kl-divergence-base "$REFERENCE"

# Candidate model, with the same settings and exact saved target-token stream.
timeout --kill-after=10s 300s "$BIN/llama-perplexity" \
  "${shared[@]}" -m "$CANDIDATE_MODEL" \
  --kl-divergence --kl-divergence-base "$REFERENCE"
```

Use a different writable reference path for each frozen corpus. FP16 KV here is
an explicit common evaluation setting; alternatively use the same predeclared
production KV setting for every run, never a different setting per model. No MTP
or GDN schedule change should be mixed into an NVFP4-only comparison.

Record paired mean NLL change (`Mean ln(PPL(Q)/PPL(base))`), PPL ratio, mean/tail
KLD, target-token probability changes, and sample counts separately for code and
text. Set acceptance thresholds before looking at candidate results; speed is
not a substitute for this check. The reference export uses uint16-encoded log
probabilities, not lossless raw FP32 logits, so retain the baseline self-comparison.

Do not accept exit 0 alone: this tool returns from several failed evaluation paths
without propagating an error to main. Require complete nonempty reference data,
all requested chunks, finite `Final estimate: PPL` for the reference run, and
finite `Mean PPL(Q)`, `Mean PPL(base)`, `Mean ln(PPL(Q)/PPL(base))`, and `Mean KLD`
summaries for comparison runs. Reject context/vocabulary mismatch diagnostics,
short input, partial reads, failed decode, missing statistics, or timeouts.
The CLI does not enforce model-quality thresholds automatically.
