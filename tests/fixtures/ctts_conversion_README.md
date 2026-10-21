# CTTS audio conversion and reference voice paths

Both fixtures execute unchanged methods from `ezlocalai/CTTS.py` at reference
commit `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Each header checks the whole
source file's SHA-256. The external capture programs are
`/tmp/ezlocalai_ctts_audio_oracle.py` and
`/tmp/ezlocalai_ctts_paths_oracle.py`. Repository implementation and runners are
C++; Python is used only outside the repository to capture source results.

## Audio

`ctts_audio_cases.jsonl.gz` contains 2,122 source cases:

| Method | Cases |
| --- | ---: |
| `_to_mono_float32` | 420 |
| `_array_to_pcm16_bytes` | 426 |
| `_array_to_wav_bytes` | 435 |
| `_tensor_to_bytes` | 399 |
| `_bytes_to_tensor` | 442 |

The current capture uses NumPy 2.4.6, Torch/torchaudio 2.9.1+cu128, soundfile 0.14.0
and libsndfile 1.2.2 on the Intel Core i9-12900KS deployment host. It pins the torchaudio functional and transforms source
hashes as provenance. It exercises scalar, empty, one/two/three-dimensional,
channel-first/channel-last, C/Fortran, transpose, negative-stride and stepped
views; signed zero, denormals, NaN/infinity, clipping and quantization boundaries;
and cancellation-sensitive sums around NumPy's eight-lane/128-element reduction
boundaries. Native arrays carry float32 storage, element strides and offsets.
Conversion from arbitrary Python objects or other array dtypes is outside this
typed input interface.

All 2,122 cases were recaptured from the unchanged installed Python class on
2026-10-18 and compared with the packaged native libraries. Relative to the
earlier NumPy 1.26.4/soundfile 0.13.1 capture, 45 resampling outputs and five NaN
payloads changed; every other output stayed identical. The native implementation
matches the refreshed capture without a production code change. No general
floating-point tolerance was added.

The previous gzip SHA-256 was
`2a67c4e17ce52888602c6eb6fdc256a4f1bed3bb9bcabfe0c1dad24dc0f06a55`.
The external refresh script SHA-256 is
`bf0fdee01181585733170f0593be30fd30c103dd67076220f097ec4536166c7a`.

Finite samples (including signed zero) are compared bit for bit. Intermediate
NaNs are compared by classification; their payload/sign bits can vary with
compiled reduction instructions and are not established by this test. Complete
WAV and PCM bytes are compared exactly, including non-finite input cases.
WAV uses soundfile's default PCM16 subtype and clipping through native
libsndfile virtual I/O. Streaming PCM instead replaces NaNs, clips to [-1,1],
multiplies in float32 by 32767 and casts to little-endian int16. They deliberately
remain separate conversion paths.

Decoding covers WAV PCM16/24/32/float/double, WAVEX, FLAC, AIFF and Ogg/Vorbis,
empty/truncated/invalid data, one/three channels, invalid rates and real speech
and silence WAV assets. The source reads float64 before converting to float32;
native decoding preserves that order and produces channel-first arrays.
Resampling uses torchaudio's default Hann-windowed sinc kernel, constructed with
matching LibTorch operations and float64/float32 promotion. Kernel caching
precision, convolution, padding, shape changes and target-length rounding follow
the source. Captures use one Torch thread on CPU. The optional LibTorch adapter
is required for rate-changing cases; builds without it report those cases as
unavailable. This is not GPU resampling verification.

The existing actual CTTS model fixture also includes complete WAV and PCM
SHA-256 values for all three CPU audio buffers, in addition to raw float32
hashes and error recovery. See `ctts_runtime_README.md` for model provenance.
No soundfile, NumPy or torchaudio Python package is required at native runtime.
The x86-64 CUDA image now packages the verified native dependencies. Other
versions/platforms remain unverified.
NumPy, PyTorch and torchaudio notices are in `licenses`.

For exact codec output, configure `EZLOCALAI_SNDFILE_LIBRARY` to the native
library used by the source soundfile installation. It defaults to the system
`libsndfile.so.1` otherwise. The tested x86-64 library's SHA-256 is
`c872fb84af2afc0e96b603e0000f0f407af86c49ad1126f2785e5722c7bb27d7`, also recorded
in the fixture header. The system and bundled libraries both report 1.2.2,
but their Vorbis builds produced a one-ULP difference in a captured decoded
sample. A version string alone is therefore insufficient for exact parity.
The optional native library selection is shared by CTTS and LlamaTts WAV reads.

## Voice paths

`ctts_paths_cases.jsonl` contains 610 real-filesystem cases for
`_contained_file_path`, `_voice_audio_path` and `_voice_ref_text`. The source
capture uses CPython 3.12.3. Both runners construct isolated files, directories
and symlinks from the same fixture instructions and normalize only the unique
temporary-root prefix in results/errors.

Cases cover name sanitization, default fallback, absent and directory-valued
voices, missing parent components, symlink escapes, dangling links, loops,
invalid extensions, root-directory containment, embedded NUL, transcript
whitespace/newlines, BOM, unreadable and directory-valued transcripts, and
malformed/truncated UTF-8. The native non-strict POSIX realpath traversal follows
CPython behavior rather than resolving missing components with different C++
filesystem semantics. Transcript reads suppress OS failures but propagate
UnicodeDecodeError, as the source does. Voice resolution preserves the source's
existence test: it does not require an existing path to be a regular WAV file.

These helpers do not introduce a stronger race-free filesystem contract than
the source's check-then-open sequence. Windows paths, arbitrary Python path-like
objects, surrogate-escaped filenames and filesystem mutation races remain
unverified. Logging is still separate unfinished work.

These components now compose with CTTS construction, cache and buffered
generation, as documented in `ctts_generate_README.md`. Stream transport,
TTS pool/voice guard and public speech endpoint
integration remain incomplete, and the speech manifest rows remain partial.

The native AudioCache component is covered by `ctts_cache_README.md` and is now composed into buffered Ctts generation.
