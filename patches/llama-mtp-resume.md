# Position-aware MTP cache resume

This managed patch targets llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab`
after the conversation-cache, atomic restore/save and bounded cache-storage
patches. Custom engine revisions retain their own implementation.

MTP previously saved target/draft KV without its CPU hidden predecessor. A
conversation switch could therefore pair A's first resumed token with B's last
hidden row. A snapshot's final hidden row also cannot stand in for an earlier
row after the server rewinds a cached prefix or replays its final token.

The driver now stores rows by the next-input boundary they precede. Request
preflight resolves that exact boundary after the target checkpoint and
last-token decisions, before evaluating any target batch. If it is absent,
the server restores an older compatible checkpoint or clears target, draft and
CPU state and reprocesses the prompt. It never substitutes zero at a nonzero
boundary or skips an uncached draft row. Older/unrepresented boundaries may
therefore require more prompt evaluation instead of reusing inconsistent state.

CPU state follows live prompt-cache entries, ordinary checkpoints, speculative
generation rollback, slot files and copied child sequences. Context shifts and
explicit KV-chunk reuse transform the matching boundary positions too. Missing,
truncated, malformed or allocation-failed CPU restores invalidate prior
carryover. A slot file with no required draft payload cannot restore usable CPU
metadata against an empty draft prefix.

Vision uses actual media boundaries: Qwen M-RoPE can assign the same first
position to every image patch. Uninterrupted embedded batches retain their
ordered predecessor; the server seals the final row to the complete chunk's
next position. An unsealed continuation is never revived by deserialization.
Uninterrupted mixed image/text input can bind its actual next text boundary,
while new requests must pass preflight first.

The driver retains only the verification prefix selectable by the maximum
enabled draft length, plus a recent tail sized for draft/recurrent rollback
(capped at 256 rows). A 4096-token prefill no longer copies its entire hidden
matrix into CPU verification scratch. Accepted rows are kept through the actual
accepted position; rejected tail rows cannot be selected on a later resume.
Serialized CPU vectors are moved into cache entries and their allocated
capacity participates in the same hard cache budget. They are not pinned or
retained in the spare GPU-snapshot buffer pair.

The server's save wrapper also uses the atomic cache save API for every model
type. Short target/draft serializations remove the incomplete outgoing entry;
they do not publish recycled bytes or clear the active prompt's checkpoints.

`test_llama_mtp_resume` exercises the production row-selection/state helper:
A/B/restore-A, exact append, final-token and checkpoint replay, accepted/rejected
verification tails, copied sequences, position shifts, M-RoPE continuations,
bounded copy size, binary float preservation, malformed/truncated payloads and
allocation-failure invalidation. Optimized and ASan/UBSan runs pass 163 checks.
The combined CPU `llama-server` compiles with `GGML_NATIVE=OFF`.

This evidence does not establish real-model acceptance, output quality or GPU
speed. Those require the separate controlled CUDA comparisons. The patch does
not change model weights, quantization, target sampling, KV precision, context
limits, draft lengths or confidence presets. The shared engine artifact to
replace for validation is `libllama-server-impl.so` (or a complete matching bin
set), since the changes are in common/server code rather than `libllama.so`.
