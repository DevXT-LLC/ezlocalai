# Native STT speaker diarization

`stt_diarization_cases.json` records unchanged `STT.py._diarize_segments` and
its process-wide `VoicePrintStore`, plus the installed feature/clustering
operations. Reference STT.py SHA-256 is
`d0dd8340c243d7ff9ba84f4e3babe3c41b0b0ce7c7dddf2565eb886ba3f21fe2`.
The original Python image uses librosa 1.0.0, NumPy 2.4.6, SciPy 1.18.1 and
python-soxr 1.1.0. Reference capture scripts were temporary external tools;
the implementation and repository test runners are C++.

The native path decodes with libsndfile, mixes channels to float32 mono,
resamples to 16 kHz with libsoxr HQ, extracts 20 MFCC means and standard
deviations, and uses cosine distance with average-linkage clustering.
Automatic speaker counts use the source inconsistency rule. Session centroids
use the source similarity threshold, weighted updates and lazy 24-hour expiry.
One synchronized store spans STT instances and pool reloads. Speaker IDs are
random `SPK_` identifiers, not synthesized voices or verified identities.

The nearest-neighbor chain and cluster traversal follow SciPy's BSD-licensed
hierarchy implementation. Its notice is in `licenses/SciPy.txt` and is
installed with the native package. The resampler uses the public opaque libsoxr
stream API; it does not require Python or the python-soxr extension.

`stt_diarization_reference` verifies 33 feature vectors, 72 clustering cases
and 45 diarization/session cases. Assets cover 8/16/22.05/48 kHz, mono/stereo,
nonintegral resampling length, distinct synthetic voices, nonfinite audio,
short/skipped segments, missing files, slicing and session expiration.
Feature comparison allows 0.003 absolute error for different FFT/reduction
implementations. Cluster labels and complete diarization JSON must match
exactly. Sixteen concurrent requests also check persistent-ID consistency.

`stt_diarization_model_cases.json` captures six real CPU/int8 Whisper calls
using the pinned mobiuslabsgmbh/faster-whisper-large-v3-turbo snapshot
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`. The fixture records model and audio
hashes, output segments, timestamps and speaker-prefixed text. It covers
automatic/explicit speaker counts, repeated sessions, different sessions and
model reload. `test_stt_diarization_model` compares those calls and two public
multipart transcription-handler calls. Only generated speaker UUIDs are
normalized consistently across calls; all other JSON values must match.

Run the model test with these five arguments: Python reference root, model
fixture JSON, pinned model directory, native VAD directory, and
`tests/fixtures/stt_runtime_assets`. The model is intentionally not downloaded
by the ordinary CTest suite. The packaged CUDA check runs the deterministic
feature/session suite; real model tests use mounted caches separately.

These tests establish behavior for the captured inputs. They do not measure
speaker-identification accuracy on a representative multi-speaker corpus or
guarantee identical clustering for all near-tied floating-point inputs.
Diarization feature processing is CPU-based even when Whisper uses a GPU.
