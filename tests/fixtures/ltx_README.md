# LTX policy and conditioning evidence

Scope correction (2026-10-19): the replacement must support the complete LTX
pipeline. Separate Gemma storage/quantization/residency experiments, Quanto
fallback and exact experimental offload call order are no longer acceptance
requirements. Their fixtures below document historical implementation work.
The pipeline's required text encoder and other inference assets remain internal
LTX dependencies; this is not a separate Gemma feature. See the current scope
at the top of `PARITY_PROGRESS.md`.

LTX parity is **not complete**. The current HTTP handler dispatches LTX to the
native runtime, and the CUDA Dockerfile applies both engine patches. The
sections below record the earlier engine experiments and their then-current
limits; see `PARITY_PROGRESS.md` for the latest deployment state.

`ltx_i2v_resize_cases.json` captures 64 executions of the installed Diffusers
image processor configured exactly as `LTX2ImageToVideoPipeline` (PIL BILINEAR).
It records pipeline, processor and capture-script hashes. `ltx_runtime` checks
those pixel hashes plus cached snapshot refs, complete indexed shards, real
child-process arguments and environment isolation, MP4/audio encoding,
condition priority, sequential/reduced retry, timeout cleanup and native HTTP
batching/LLM handoff. The engine process in this test produces synthetic frames
and audio; it does not establish model inference parity.

`ltx_policy_cases.json` contains 2,006 executions of the Python reference:
480 settings cases, 16 pipeline-selection cases, 128 failure/recovery cases,
734 residency decisions, 64 device/dtype/BNB decisions, and 584 environment
cases. The capture executes the actual `LTX_VIDEO.py` methods or AST-selected
policy blocks, and the actual `VIDEO_UTILS.py` residency function. Model
loaders, tensor execution, CUDA probes, exports and cleanup are substitutes;
the fixture does not establish their behavior. The C++ test verifies both
source file hashes before comparing results.

The capture script is `/tmp/ezlocalai-capture-ltx-policy.py`; its SHA-256 is
stored in the fixture. The current source default video model is Wan 2.2.
LTX's loader uses a 10 GiB short-clip offload threshold; the standalone
residency utility defaults to 12 GiB. These defaults are intentionally
different. Native settings use bounded integers; arbitrary-size Python
negative dimensions and environment limits are not fully represented.

`ltx_condition_cases.json` captures 137 cases from the installed Python
container's `diffusers/pipelines/ltx2/pipeline_ltx2_condition.py`:
80 image preprocessing cases, 48 index cases, and nine coordinate cases.
The capture AST-executes the actual methods with real CPU PyTorch, NumPy and
Pillow, without loading model weights. The installed source SHA-256 is
`3d657784b1d0c3b63e16522b3bc98416ef6ac84c89a1bc599991172541f4457f`.
The capture script `/tmp/ezlocalai-capture-ltx-conditions.py` has SHA-256
`ba2b51e2e47b138e602143deca1314b6db699234f0aecb921f28ac5c92eb3c69`.

The engine test compiles the actual patched `ltx_condition.hpp` and tensor
operations. It checks float bilinear cover-resizing/center cropping with
absolute tolerance 2e-6 after normalization, exact keyframe coordinates,
negative latent-index wrapping, skipped positive out-of-range indices, and
multiple appended conditions with independent masks and positions. These
tests cover single-image conditions exposed by the worker HTTP schema;
Python's internal multi-frame condition objects are not implemented here.

To run the engine helper test, apply
`patches/sdcpp-ltx-conditions.patch` to stable-diffusion.cpp revision
`c678dfe704a2230342376b46add9c8ca736a653d`, then configure the native build with
`-DEZLOCALAI_SDCPP_SOURCE=/absolute/path/to/patched/stable-diffusion.cpp`.
Build `test_ltx_conditions` and run CTest `ltx_conditions_reference`.
The default `ltx_policy_reference` target needs only the Python reference root.

The patched CUDA engine built successfully in the isolated
`ezlocalai-ltx-engine-validation` container. Its executable SHA-256 is
`624adeea6cf08b0ec5cb131d09d5bb9c6a77ff556a99856a453653a7f518bbf4`.
The patch is not yet applied by the production Dockerfile. The follow-up
section records subsequent component loading and CUDA execution work.
Text-encoder quantization parity, image-to-video preprocessing, multi-frame
conditions, export integration, OOM execution/reload, HTTP lifecycle integration,
and completed CPU inference remain open. A native
GGML sampler/quantizer must not be assumed numerically equivalent to the
Python Diffusers/Torch pipeline.

A read-only, isolated CPU probe of the unconverted Python cache exited 134
before inference. The native engine recognized LTXAV and Gemma 3, then its VAE
constructor aborted with `LTX VAE encoder is only implemented for version >= 2`.
The published Diffusers VAE layout is therefore not directly usable as-is.
The probe used the existing production engine, not the conditioning patch,
and did not change the cache or production worker. Its log is retained at
`build/validation/ltx-cache-probe.log`.

## Diffusers component adapter and model execution

The follow-up `sdcpp-ltx-diffusers.patch` fixes that component-layout failure.
`ltx_mapping_cases.json` covers all 170 video-VAE names, 102 audio-VAE names,
and 262 published connector names without collisions. The oracle executes
the installed single-file conversion functions and verifies connector names
and shapes against the cached GGUF and text-projection metadata. For the
fourth decoder up block absent from the legacy conversion dictionary, it
executes the installed up-block method to establish upsampler/resnet order.
Fixture provenance includes both installed source hashes and the capture
script hash. `test_ltx_mapping` compiles the actual patched adapter header.

An isolated CPU probe constructs the complete model and executes Gemma text
encoding/projection, then reaches its 180-second deadline before completing
video generation. This is not a successful CPU inference test.

A real native CUDA run with the existing model cache, published tokenizer,
BF16 Gemma weights, CPU parameter offload, two Euler steps and seed 42 produces
nine 64x64 PNG frames plus stereo 48 kHz PCM audio. The frames decode and are
nonblank; audio is non-silent. FFmpeg produces a valid MP4 with nine frames
(0.375 seconds) and 0.370 seconds of stereo audio. This is a short execution
test, not an assessment of production-resolution quality or Python numerical
equivalence. Logs, raw frames, WAV and MP4 are under
`build/validation/ltx-diffusers/cuda`.

A 17-frame request with conditions at indices 0, 1 and -1, individual strengths
1.0/0.5/0.25, and a skipped out-of-range condition timed out after 900 seconds
while loading transformer weights with the default pinned host storage. The
host showed severe memory pressure. Repeating the same request with `--mmap`
and `GGML_CUDA_NO_PINNED=1` completed in about 56 seconds and produced all 17
frames plus stereo audio. This validates the combined memory mode; it does
not isolate the effect of each flag. A repeated nine-frame text-to-video run
with that mode produced byte-identical PNG and WAV files to the earlier
successful native run. These are native-to-native comparisons, not Python
numerical equivalence. The conditioned result and memory-mode comparison are
under `cuda-conditions-mmap` and `cuda-mmap` in the same evidence directory.

The Python reference's automatic model-offload attempt raises an OOM on this
machine for the same short request. Executing its actual `Pipes.py` retry
method with real LTX models also exhausts memory after sequential reload and
the reduced-GPU retry, returning no video. The retry oracle substitutes the
resource-manager callbacks and model factory/lifetime boundary; it does not
exercise the complete Python server or LLM handoff. Evidence is retained in
`build/validation/ltx-diffusers/python` and `python-retry`. The native
BF16 encoder/offload implementation is distinct from the reference's BNB and
Quanto paths; exact memory and numerical parity remain open.

A separate fresh process with `VIDEO_GPU_RESIDENCY=sequential` completes real
Python inference in 56.21 seconds, including 37.59 seconds of loading. It
captures nine RGB uint8 frames and float32 audio of shape `[1,2,15840]`.
The initial harness omitted the `outputs` directory: the wrapper returned a
path but no MP4 existed. This is a harness error, not a verified production
export failure. Replaying the actual `_export_video` method with the captured
frames reconstructed as PIL images, the captured audio tensor and an existing
output directory produces a valid H.264/AAC MP4. FFprobe reports nine frames
at 24 fps (0.375 seconds) and stereo 24 kHz audio lasting 0.660 seconds.
The replay does not rerun the model. A discarded first replay used lists of
NumPy arrays, which the encoder does not accept; the final replay restores
the pipeline's default PIL frame type.

Native audio has 17,760 samples per channel at 48 kHz (0.370 seconds), so both
waveform length and the export sample rate differ. The shared transformer
weights and matching request dimensions do not establish numerical or export
parity. Fresh sequential evidence, raw arrays, the corrected export and
FFprobe output are retained under `python-sequential` in the same evidence
directory. Its capture and replay scripts are
`/tmp/ezlocalai-capture-ltx-real.py` and `/tmp/ezlocalai-reexport-ltx.py`.

`ltx_inference_validation.json` records the patch/script hashes, run results,
output hashes, decoded media properties, native repeat comparison and these
limitations. The four focused CTest targets pass. Production remains on the
previous CUDA image, healthy after every isolated GPU run; a final live Rust
router check confirms one registered worker with context 220000 and HTTP 200
for chat, streaming and vision. Neither LTX engine patch is deployed yet.

## Audio latent-count correction

`ltx_audio_length_cases.json` executes the actual installed `LTX2Pipeline`
assignments for duration, audio latent rate and audio frame count. Its 196
cases include exact halfway values. The native engine had used ceiling;
Python uses double-precision `round`, with ties to even. The adapter patch now
uses the latter, and `ltx_audio_reference` compiles the actual engine helper.
The C++ worker also follows the reference wrapper's 24 kHz audio export
interpretation without resampling the waveform. Prior engine clips/hashes and
the measured rate/length mismatch above describe the earlier implementation.
See the current progress document for real verification of the correction.

## Gemma3 raw-weight normalization

`ltx_gemma_norm_cases.json` captures 36 executions of the installed
`Gemma3RMSNorm.forward`, across four widths, three weight patterns and three
input scales. The capture script is `/tmp/ezlocalai-capture-ltx-norm.py`;
source and capture hashes are embedded in the fixture. The C++ test executes
the engine's actual GGML RMS block both with raw weights and with the unit
offset already folded into weights. All 72 graphs pass an absolute tolerance
of 2e-6 (observed maximum error 2.38419e-7).

This check caught a loader distinction missed by tensor-name mapping tests:
raw Hugging Face Gemma3 weights require `1 + weight`, whereas the engine's
normal GGUF weights have that offset folded in during conversion. The
adapter now enables the unit offset only for raw Diffusers Gemma3 loading.
The regression target requires the patched engine's CPU GGML libraries:
configure with `EZLOCALAI_SDCPP_SOURCE` and `EZLOCALAI_SDCPP_BUILD`, then
run `ctest -R '^ltx_norm_reference$' --output-on-failure`.

The corrected encoder's first larger HTTP run exposed a separate engine
race while loading VAE tensors. A core backtrace identified
`convert_diffusers_vae_to_original_sd1`: its lazy mutable table could be read
while another loader thread reallocated it. The adapter now initializes a
const table with a guarded static lambda. `ltx_name_concurrency` links the
actual engine source and checks 16,000 mappings from 32 simultaneous first-use
threads. The original source produced an incomplete mapping; the fix passed
20 fresh process executions. The core was removed after retaining its
backtrace at `/tmp/ezlocalai-ltx-norm-crash.log`.

## Worker, Rust router and deployment verification

`ltx_http_validation.json` separates three native runs: the original candidate
that failed visual prompt review, the normalization-only experiment whose
first VAE load crashed before a successful router retry, and the rebuilt
image containing both fixes. Do not apply the current patch hashes to either
historical binary. The final image passes all five HTTP cases without an
engine failure: direct text-to-video, routed positioned conditions, routed
first-image input, and two routed 512x320/41-frame clips. Every case drains
text admission and restores the original LLM with context 220000. URL assets,
base64 output, invalid-size failure/restoration, and before/after chat are
also checked. The short clip has stereo 24 kHz audio lasting 0.660 seconds,
matching the Python exporter timing.

Visual review remains distinct from structural checks. The corrected red-ball
clip depicts a rolling ball beneath a table; the actual Python reference
places it on a tabletop. A second prompt produces a yellow rubber duck on
blue water as requested. The final red-ball MP4 is byte-identical to the
successful normalization-only experiment, showing the concurrency fix did
not alter that output. These examples establish working generation, not
general prompt-fidelity or numerical equivalence.

Local artifacts are under `build/validation/ltx-http-corrected`, with
the actual Python normal-size reference under `ltx-diffusers/python-normal`.
The final harness is `/tmp/ezlocalai-validate-ltx-corrected-router.py`; it
rejects any engine failure hidden by a router retry and restores the original
worker after testing. The final image is deployed with the original `.env`
and unchanged effective environment. See the progress document for live checks.

## Cross-attention timestep traces (2026-10-19)

`ltx_timestep_cases.json` records forward pre-hooks on all eight timestep
modules of the original container's `LTX2VideoTransformer3DModel`. A small
zero-layer model runs the actual forward method, with six scheduler sigmas,
four video denoising masks and both `use_cross_timestep` settings. Source and
capture hashes are embedded in the fixture. The source worker uses the default
`false` setting; its 24 cases preserve each video token's mask and pass sigma
times 1000 to both modality cross-attention modules. The optional opposite
modality cases are retained for comparison, not claimed as worker coverage.

`sdcpp-ltx-timesteps.patch` corrects that path only for raw Diffusers component
loading. `test_ltx_timesteps` executes the patched engine's GGML helper and
compares all four cross-attention module inputs with the default-mode traces.
It separately verifies that legacy standalone GGUF inputs remain unscaled.
Configure `EZLOCALAI_SDCPP_SOURCE` with the patched source and
`EZLOCALAI_SDCPP_BUILD` with its static GGML CPU libraries to enable the test.

A real routed CUDA I2V experiment passes structural and restoration checks,
but the timestep correction alone does not resolve the confirmed visual
regression against Python. This experimental patch is not yet in the deployed
image. See `music_video_http_validation.json` for both runs and limitations.

`ltx_schedule_cases.json` executes the actual schedule-preparation statements
from all three installed pipelines and their real FlowMatch Euler scheduler.
It covers six step counts and four native token counts (72 cases). The source
passes the configured maximum sequence length to `calculate_shift`; it does
not use the requested clip's token count. The reference checkpoint therefore
uses mu=2.05 at every size. `ltx_video_command` now passes equal base/max shifts
to the native scheduler. `test_ltx_schedule` runs that engine's actual scheduler
with the same command arguments: maximum sigma error is 2.98023e-7; the old
size-dependent defaults fail 45 cases. Synthetic process tests verify the
arguments reach ordinary video generation and each music-video scene.

This schedule setting matches the reference LTX-2.3 checkpoint. Arbitrary
replacement scheduler configurations through `LTX2_CONFIG_REPO` are still
outside the verified scope. The corrected CUDA candidate passes routed music-video structural checks, but
deployment remains held for the text-conditioning discrepancy described below.


## CPU noise and identical-input diagnostics (2026-10-19)

`ltx_rng_cases.json` captures 44 CPU-generator cases and two consecutive draws
per case from installed PyTorch. The source worker explicitly constructs that
CPU generator with seed 42; CUDA inference does not select CUDA noise. The native
runtime now passes `--rng cpu --seed 42`. The engine's actual MT19937 generator
matches all 88 draws within 2.38419e-7, and runtime subprocess checks verify those
arguments for ordinary video and every music-video scene.

The numerical and visual comparisons are separate:

- Encoded storyboard latents correlate at 0.999895 with Python. Replaying Python
  final latents through the native decoder retains the graphics and performers;
  exported pixels still differ.
- Native positive/negative video/audio prompt contexts correlate at
  0.99656–0.99964 with Python.
- Substituting identical Python inputs to the first native transformer forward
  gives video prediction correlations of 0.999874 and 0.999841 for the positive
  and negative passes. The first two video transformer blocks exceed 0.99997.
- Correct CPU noise restores the concert scene. Replaying only Python prompt
  contexts throughout the full native pipeline additionally retains its theorem
  graphic at four seconds. Native BF16 conditioning loses that graphic.

This isolates the remaining visible discrepancy to text conditioning. Python
uses bitsandbytes 0.50.2 NF4 with nested statistics on this GPU; the initial
candidate used unquantized BF16 Gemma. A later native NF4-derived weight experiment
preserves the graphic, confirming quantization as a major source of the visible
difference. Automatic NF4 loading now reproduces that corrected diagnostic MP4
byte for byte from the original cached weights, using the selected CUDA device's
actual geometry. Complete encoder/connector arithmetic remains unverified;
see `nf4_validation.json` for historical comparisons and `nf4_README.md` for the
runtime integration. NF4 compressed storage/GPU matmul, Quanto fallback and
complete memory/residency policy remain implementation gaps.
The diagnostic replay/trace hooks are not shipped in the candidate.

Local evidence is under `build/validation/ltx-latent-diagnostics`,
`ltx-context-diagnostics`, `ltx-forward-diagnostics`, `ltx-cpu-rng-validation`,
`ltx-context-replay` and `music-video-corrected`. The portable record is
`ltx_alignment_validation.json`. Its earlier Python latent-capture runner has
an unhealthy restoration observation caused by an overlapping diagnostic run;
the inference succeeded, and the subsequent native runner restored production
healthy. This exception is preserved in the record.

## Packed residency and separate connectors (2026-10-19)

The BF16 and early NF4 records above are historical. Packed Gemma GPU residency,
distinct whole-model/sequential execution and per-session connector reuse are now
implemented. `ltx_connectors_validation.json` records two byte-identical real
generations without reload and five varied HTTP requests, including four through
Rust WorkerRouter. CPU/CUDA cache lifecycle checks and all 34 packaged tests pass.
Quanto fallback, complete Python numerical parity, quantized LLM/video handoff,
production-size presets and multiple GPUs remain open.
