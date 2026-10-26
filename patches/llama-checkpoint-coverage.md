# Aligned checkpoint coverage experiment for named MTP conversations

This fifth managed patch applies after conversation-cache, atomic save/load,
bounded storage and MTP-resume at llama.cpp
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. It leaves model weights, precision,
physical batch sizes, sampling and context limits unchanged. Its CUDA promotion
is conditional on a new comparison; model-free tests cannot establish quality.

The two original capped offsets, `4 + n_ubatch` and `4`, always remain. For the
measured seed with 24,577 tokens, those create checkpoints at 23,549 and 24,573.
A rewind to 23,327 precedes both and falls back to the early user checkpoint.

The earlier fixed 2052-token offset was first accidentally disabled for text
chats with a loaded projector. Correcting that eligibility showed a substantial
rewind gain, but changed all generated answers in the comparison. That fixed
split is superseded by this aligned experiment; its logs remain separate.

The new split uses a boundary the original physical splitter already had.
For the observed final logical chunk starting at 20,555, the original target
and draft groups end at 21,579, 22,603 and 23,549. Splitting the logical call
at 22,603 preserves those physical groups and makes a checkpoint available at
22,603, with 1,974 input tokens remaining. It does not force an absolute 2052
boundary. MTP continues with the exact predecessor `h[22602]`.

The experiment requires all of these conditions:

- a named prompt of at least 4096 tokens with actual text only;
- an existing FULL/RS checkpoint path and at least four configured checkpoints;
- exactly one configured slot, causal attention and no future user-start split;
- only `draft-mtp`, one trained MTP head and matching actual target/draft physical
  batches of 1024; other sizes, including 2048, retain the original scheduler;
- no LoRA or shared-parent prefill boundary in the eligible last-user phase.

The trained-head count comes from `llama_get_model(ctx_dft)`. Borrowed MTP
initializers intentionally leave their owned-model pointer null, so that pointer
must not determine eligibility. A model-free accessor-seam regression exercises
the production helper with a borrowed context and null ownership, including
null context/model and zero/multiple-head paths. The actual draft-parameter
converter is also tested: default zero inherits target UB, an explicit draft
override is preserved, and the runtime guard still reads actual context UB.

An additional split must follow a complete physical batch from the actual
current logical batch start. It lies in the one-batch-wide remaining-token
window `(1028, 2052]`. The original logical chunk must still leave at least
`n_rs_seq + 1` rows after the candidate when recurrent rollback is enabled;
otherwise the original splitter would rebalance its short tail. At most one
additional near-end checkpoint is created. A natural batch start already in
that window can be snapshotted without another split. Both old checkpoints,
configured count limits, eviction rules and cache byte ceilings remain intact.

Text-only classification inspects actual media chunks. The OAI chat tokenizer
sets `server_tokens::has_mtmd` for projector-loaded text too. The helper checks
media at token zero separately because the existing next-media lookup uses
`upper_bound`. Existing media decode/checkpoint guards remain unchanged.

One extra checkpoint uses additional pageable RAM (about 150 MiB for the
currently measured Qwen model), charged by the existing cache storage budget.
A tight cache may retain fewer conversation states. Set
`EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE=0` for the original scheduler in the same
binary. The candidate default enables the experiment only after all guards;
production promotion still requires the new CUDA comparison.

At trace verbosity (`--log-verbosity 4` or higher), `checkpoint coverage:`
records numeric eligibility inputs and the selected window: actual target/draft
UB, slot/rollback counts, trained heads, borrowed/owned model presence, mode,
message phase, adapters, shared tokens and checkpoint settings. It includes
neither prompt content nor cache-key values. An aligned experiment before the
context-model accessor correction was also inactive; its equal outputs do not
qualify the now-active policy.

`test_llama_checkpoint_coverage` links the actual `server_tokens`, MTMD and
production batch allocator. It compares complete ordered token/position/output
groups from target `split_equal` with RS-tail reservation and draft
`split_simple`, tests eligibility and preservation of old checkpoints, and
checks the MTP carryover boundary. Projector text, media at zero/after text,
cloning, serialization and truncation are included. Optimized and ASan/UBSan
runs pass. Negative controls forcing an unaligned boundary or ignoring the
recurrent tail fail. The actual modified server context passes a release-flags
syntax compile. Root must provide the existing engine test libraries plus
`${EZLOCALAI_TTS_LLAMA_SOURCE}/src` in this target's include directories.

Identical cold physical groups are a stronger invariant than an unchanged
physical-batch setting. They still do not establish universal GPU bitwise
identity. A shortened prompt evaluated cold can have a different tail grouping
from the longer prompt that created a reused checkpoint, so rewind off/on output
hashes are not universally expected to match. New CUDA checks must compare seed
and replay requests, verify restored-state continuation, and assess rewind
answers and cache-pressure costs. No corrected aligned GPU result is claimed
by this patch's model-free fixture.
