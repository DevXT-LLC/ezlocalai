# Batched host state transfers

The patch is qualified against llama.cpp `620d2225` and leaves state-file and
in-memory snapshot formats unchanged. Host serializers collect the same tensor
ranges in the same order. Ordinary CUDA device-buffer ranges use the backend's
async data-access API with bounded pinned staging, then synchronize before
returning to the caller.

Each context allocates at most 64 MiB of pinned working memory per CUDA backend,
lazily on the first qualifying transfer. The window is reused across snapshots;
cached state vectors remain pageable and can still be swapped by the OS. A
transfer larger than the window is split into chunks. Downloads synchronize
before copying staging bytes into pageable destinations, in global serialization
order across devices. Uploads first copy the caller's bytes into stable staging
and finish the device copies before reusing that window. Direction changes,
alias dependencies, full windows and the bounded 256-copy record list also finish
pending work. The context releases pinned workspaces after synchronization and
before scheduler or backend teardown.

CPU, HIP, Metal, split, host and unrecognized buffers retain synchronous data
access, as does a missing or failed pinned allocator. CUDA's silent fallback to
a pageable host buffer is detected and rejected; allocation is attempted only
once per backend until the pool is cleared. Synchronous fallbacks first finish
pending work so aliased host or special buffers cannot overtake a device write.
ON_DEVICE state serializers are unchanged. No weights, KV precision, context,
sampling, or numerical kernel changes are involved.

`EZLOCALAI_LLAMA_STATE_IO_BATCHING=0` opts out for matched performance and byte
comparisons using the same engine binary. The default enables the CUDA-only
path. Compose already includes the worker `.env` as an environment file, so the
opt-out passes through to owned engine processes.

`llama_state_io` is the mandatory model-free transport test, including a fake
backend that defers copies until synchronization. It covers queue completion,
views, global ordering and aliases, captured upload sources, staging capacity
and reuse, allocation failure, fallbacks and opt-out behavior. With the tiny smoke model
configured, `llama_state_io_model` compares actual serialized snapshots and
post-restore CPU logits, and checks recovery after a truncated restore. These
checks establish transport behavior; physical CUDA performance and model-output
qualification remain separate requirements before rollout.
