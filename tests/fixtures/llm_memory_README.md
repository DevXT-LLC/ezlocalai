# Managed LLM allocation accounting

`llm_memory_cases.jsonl` was captured from reference commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The header pins the complete
`ezlocalai/LLM.py` and `Pipes.py` file hashes.

The capture executes unchanged AST statements from `LLM.__init__`, beginning
with `gpu_free_before_load = None` and ending immediately before model-list
lookup. The native server constructor and startup completion are successful
stubs; CUDA availability is true and the two free-memory queries receive the
listed vectors or raise a telemetry error. It then executes the unchanged
`Pipes._load_llm_resilient` method with that instance, an initially empty
estimate cache, and stub lifecycle/runtime recording. The fixture records the
allocation delta, number of telemetry calls, and resulting estimate cache.
The capture driver is `/tmp/capture-llm-memory.py` in the validation environment;
no Python capture code is shipped in the native product.

The C++ test runs each case for text and vision using real managed fixture
processes and injected ResourceManager probes. It also verifies public/shared
resource records, unload/restore, retained maximum measurements, destruction,
and protection of an active inference from idle-memory reclamation.
It preloads before requesting inference, matching the captured constructor
boundary. Lazy request admission has a separate headroom probe and is covered
by `llm_headroom_reference`; it must not consume the allocation fixture's
before/after samples.

The reference measures the selected main GPU, not the sum across all devices.
Negative deltas clamp to zero. Telemetry failure does not fail model startup.
`Pipes` registers managed LLM resource records with device `cuda`, including
CPU retries; the test deliberately retains that reference behavior. Without
any measured or cached estimate, its type defaults are 8 GiB for text and
6 GiB for vision. External llama-server URLs do not establish ownership of
local GPU memory and retain their existing external metadata behavior.

This does not establish pre-load GGUF estimation or automatic multi-GPU
placement parity, nor allocation equivalence under concurrent external GPU
loads. Those remain separate requirements. Full runtime memory attribution
and process-local Torch allocator statistics also require separate evidence.
