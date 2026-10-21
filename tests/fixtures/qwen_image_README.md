# Qwen image source captures

Captured on 2026-10-18 from the unchanged Python installation at
`/home/josh/repos/xtsys/ezlocalai` (commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`) using its installed
`ezlocalai-ezlocalai` Docker image and `/opt/venv/bin/python`.
Each fixture includes the SHA256 of the source file it executes; the native
runner checks that hash before comparing results.

- `qwen_image_cases.json`: 56 calls to the real `ezlocalai.IMG.IMG.generate`.
  Only the engine subprocess and CUDA memory probes are replaced in the Python
  capture. Pillow performs real reference decoding and output validation.
  Captures include argv, ordered decoded RGB reference pixels, exact returned
  base64, and exceptions. Temporary paths are normalized to `TMP`, `MODELS` and
  `SDCLI`. The C++ runner launches a real child executable, checks process exits
  including signals, consumes the captured output bytes, and verifies cleanup.
- `image_edit_api_cases.json`: 66 calls to the unchanged `ImageEdit` Pydantic
  class extracted from `app.py`. Exact error structures and accepted normalized
  objects are captured. The native test sends the inputs through NativeServer.
- `image_policy_cases.json`: 640 calls to the unchanged
  `_image_should_unload_llm_for_generation` and `_media_keep_models_loaded`
  methods extracted from `Pipes.py`. Inputs cover explicit overrides, existing
  image/LLM residency, secondary GPUs, remote image providers, free memory,
  thresholds, invalid values, NaN, and Python numeric underscores.

The native target also tests real HTTP forwarding to isolated ephemeral peers
for image-server, router and fallback paths, URL reference downloads, separate
image-instance serialization with `flock`, successful and failed LLM handoffs,
resource reporting, idle wrapper retention and URL output files. TTS/STT pool
tests verify that media eviction refuses active leases and closes idle models.
The Python project's own `python -m unittest test_qwen_image -q` passes 18 tests.
These mocked-engine captures do not establish model inference parity.

External capture scripts were kept outside the C++ repository:

| Script | SHA256 |
| --- | --- |
| `/tmp/ezlocalai-capture-qwen-image.py` | `ef176f9e283177966ea54f4625d14a7fb17877e279af96517bff9cd0da1e7560` |
| `/tmp/ezlocalai-capture-image-api.py` | `88d0208790badaa87bcab864d35f54f7bcd3cf588839a35e475e83462c477a0a` |
| `/tmp/ezlocalai-capture-image-policy.py` | `ea68c4a72f95a2f514a7dc15feac14b00ad9c95626b923862ea8e140adba7add` |

Run with `ctest --test-dir build -R qwen_image_reference --output-on-failure`.

## Real CUDA deployment check

`qwen_image_deployment_cuda.json` records the later real-model verification on
this machine's RTX 3090 Ti. It is an audit record, not an automatically enabled
CTest target. Native and Python workers use the same four cached model files;
the record includes their content hashes and the engine/binary/image identities.
A small external C++ exec wrapper adds only `--seed 42` to both engines for the
comparison. Production generation retains the source's default random seed.

Text-to-image, primary-image editing and two-reference editing at 256x256 and
the default 20 steps produced identical RGBA pixels. The multi-reference request
also matched through the native worker WebSocket tunnel to an isolated Rust
WorkerRouter. URL publication retained all RGBA pixels and removed generation
metadata. PNG byte differences are recorded explicitly: the shallow native
engine checkout reports an unknown descriptive version instead of the Python
image's tag, and URL re-encoding uses libpng rather than Pillow compression.

A separate check loaded the original configured Qwen3.8 27B LLM at 220000
context, observed its admission capacity become zero during real image
inference, then verified restoration at the same context and approximately
20.018 GiB GPU footprint. Chat returned READY before and after the handoff.
Original production image flags remain disabled after deployment. Raw responses,
PNGs, model hashes and logs are in the ignored `build/validation/qwen-real`
directory. No Python runtime or test seed wrapper is installed in production.
