# Earlier checkpoint coverage for named recurrent conversations

This fifth managed patch applies after the conversation-cache, atomic save/load,
bounded storage and MTP-resume patches at llama.cpp
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. It changes checkpoint placement, not
physical batch sizes, model weights, precision, sampling or context limits.

The original near-end checkpoints cover offsets `4 + n_ubatch` and `4` from the
prompt end, capped by the logical batch. With physical batch 1024, the first
checkpoint can fall after a roughly 2K-token branch record has diverged or been
shortened. A hybrid/recurrent target then lacks an earlier usable state and
reprocesses the entire shared prefix.

For a **named, text-only prompt of at least 4096 tokens**, the planner adds one
earlier offset `4 + max(n_ubatch, 2048)`, capped by the logical batch and
deduplicated against the two original offsets. It requires an already-enabled
FULL/RS checkpoint path and at least four configured checkpoint slots. With
batch 4096 / physical batch 1024 the offsets become `2052, 1028, 4`; with
physical batch 2048 the offset already exists and behavior stays unchanged.

Text-only classification inspects actual media chunks. A loaded projector makes
the OAI chat tokenizer set `server_tokens::has_mtmd` even when the request has
only text, so that flag cannot exclude a prompt from this policy. The helper
checks media at token zero separately because the existing next-media lookup
uses `upper_bound`. Actual image/audio chunks remain excluded.

The checkpoint is saved **before** the batch beginning at that earlier
boundary. Its expanded near-end span is accepted only for that designated
batch start; it cannot add snapshots at unrelated intermediate/user-message
splits. Both original late offsets and their near-end decisions remain.
Existing multimodal checks, maximum checkpoint count, minimum-step/eviction
rules and hard prompt-cache byte accounting remain in force. This adds at most
one near-end snapshot rather than raising the configured count limit.

One extra checkpoint consumes additional pageable RAM (about 150 MiB for the
currently measured Qwen model). Its capacity is charged by the existing cache
storage budget when the prompt is cached. A tight cache may retain fewer
conversation states; the change does not expand its byte ceiling. Set
`EZLOCALAI_LLAMA_CHECKPOINT_COVERAGE=0` for original placement in the **same
binary**; the default enables coverage only for the qualified cases above.

`test_llama_checkpoint_coverage` exercises the production planner's split and
near-end methods: eligibility, short/unnamed/non-recurrent paths, checkpoint
counts below four, opt-out, integer bounds, logical-batch caps, deduplication,
unchanged 2048 behavior, preservation of both original offsets and exactly one
additional snapshot for a long prompt. It also links actual `server_tokens` and
MTMD implementations for projector-loaded text, media at zero/after text,
cloning, packed serialization and truncation. Optimized and ASan/UBSan runs pass
494 checks. Private negative controls using the old `has_mtmd` predicate or only
the next-media lookup fail the relevant regression. The actual modified
`server-context.cpp` passes a syntax compile using
the matched four-patch engine's release flags and headers. Those source and
binary artifacts remain frozen; this validation uses a separate tiny source
tree and does not replace the four-patch binaries.

Real CUDA rewind/shared-prefix gains, core state hashes, quality, context and
vision checks are separate qualification work. Source/planner tests alone do
not establish GPU speed or output quality.

The first CUDA coverage on/off experiment had identical schedules and no rewind
gain because the former `!input_tokens.has_mtmd` eligibility condition disabled
the policy for projector-loaded text chats. The corrected media predicate needs
a new CUDA run; that earlier result is not evidence for the corrected schedule.
