This experiment is not promoted after its completed CUDA screen. It is not
part of the proposed checkpoint entry-window runtime patch stack.

The managed speculative-overhead patch retains the pinned engine's token
arithmetic, sampler settings and target/draft state format. It removes two
pieces of host work.

During speculative verification, the target always accepts at least one token,
so rollback can cover at most the number of draft tokens. A sampler snapshot
is needed only when full rollback is required, or recurrent rollback capacity
is smaller than that upper bound. The existing snapshot and copy-back path is
retained for those cases, including RNG and grammar state. Other verification
rounds no longer clone and discard the sampler's vocabulary-sized buffers.

MTP target hidden rows are unmasked and densely stored in input-token order.
The process hook now synchronizes and obtains that buffer once, then reads the
same row offsets for draft catch-up and retained CPU carryover. Draft evaluation
does not alter the target output buffer. Removing per-row public getters avoids
repeated backend synchronization; it does not change hidden values or retained
row selection, including embedded and shared-KV paths.

`llama_speculative_overhead` compiles the actual managed process method and
snapshot selector with an instrumented backend. It checks synchronization call
counts, exact serialized CPU state and predecessor rows, multiple sequences,
embedded positions, shared KV, chained heads, and every accepted-token count
at the tested rollback boundaries. Managed-stack tests also cover upgrading the
previous five patches, idempotence, metadata, and complete removal. These local
guards do not establish a GPU speedup or general model-quality equivalence;
controlled hardware results and limitations belong in the dated benchmark report.
