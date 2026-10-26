# Batched host state transfers

The patch is qualified against llama.cpp `620d2225` and leaves state-file and
in-memory snapshot formats unchanged. Host serializers collect the same tensor
ranges in the same order. Ordinary CUDA device-buffer ranges use the backend's
async data-access API and synchronize before returning to the caller. A
synchronous fallback first finishes pending work so aliased host or special
buffers cannot overtake a device write.

CPU, HIP, Metal, split, host and unrecognized buffers retain synchronous data
access. ON_DEVICE state serializers are unchanged. No weights, KV precision,
context, sampling, or numerical kernel changes are involved.

`EZLOCALAI_LLAMA_STATE_IO_BATCHING=0` opts out for matched performance and byte
comparisons using the same engine binary. The default enables the CUDA-only
path. Compose already includes the worker `.env` as an environment file, so the
opt-out passes through to owned engine processes.

`llama_state_io` is the mandatory model-free transport test, including a fake
backend that defers copies until synchronization. It covers queue completion,
views, ordering, fallbacks and opt-out behavior. With the pinned tiny smoke model
configured, `llama_state_io_model` compares actual serialized snapshots and
post-restore CPU logits, and checks recovery after a truncated restore. These
checks establish transport behavior; physical CUDA performance and model-output
qualification remain separate requirements before rollout.
