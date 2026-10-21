# Native AudioCache policy and filesystem behavior

`ctts_cache_cases.jsonl.gz` contains 205 scenarios and 703 operations captured
from the complete, unchanged `AudioCache` class at reference commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Its header pins AudioCache.py and the
PyDub audio/utility modules by SHA-256. The external capture program is
`/tmp/ezlocalai_ctts_cache_oracle.py`. Only data and the C++ runner are included
in this repository.

The capture uses CPython 3.12.3, NumPy 1.26.4, PyDub 0.25.1 and
FFmpeg/ffprobe 6.1.1-3ubuntu5. All filesystem
operations are real, in an isolated temporary directory. The clock is controlled
and every call is recorded. Generation callbacks return captured byte strings or
raise configured exceptions. Duration analysis uses actual PyDub; one fault case
changes directory permissions after a successful duration read to exercise
unlink failure. The native runner repeats that boundary fault with the native
duration implementation.

Each operation compares its return/error, generator arguments, clock-call count,
configuration, raw statistics and the complete directory tree. File contents are
compared byte for byte, including JSON indentation, key order, ASCII escaping,
timestamps, statistics files, partial writes and leftover temporary files.
Only the randomized temporary-root prefix is normalized in strings; audio and
metadata bytes are not normalized. Store also compares mutations to the caller's
metadata when a later write fails.

Coverage includes:

- Default and overridden configuration, disabled caches, directory conflicts,
  corrupt/absent/incorrect statistics, statistics persistence and save failures.
- Misses, hits, metadata updates, TTL equality/fractions/negative and very large
  values, invalid dates and timezone-aware metadata rejection, expiry cleanup,
  malformed metadata and read/write/unlink failures.
- Size limits, access-time ordering, orphan WAV files, missing metadata fields,
  voice-specific clearing, full clearing and symlinked cache directories.
- Cache keys using pinned Unicode lowercase/whitespace policy, JSON sorting,
  nested parameters, float formatting, escaping and extra-field overrides.
- Shortest/median-based selection, ties, default selection, sample-generation
  failures, invalid counts, failed decoding and failed temporary-file deletion.
- Explicit warming, duplicate normalized phrases, disabled warming, existing
  audio-only entries and metadata/parameter precedence.

Important source behavior is retained. Disabled construction still creates
directories and loads statistics. Reads update metadata before reading audio;
stores write audio before metadata and mutate the caller's object first. Eviction
counts all WAV bytes but only eligible metadata entries can be removed. Reading
`last_accessed` still evaluates the `created_at` default eagerly. Expiry events
are excluded from the periodic-save total, but still trigger its modulo check,
including a save when that total is zero. Voice-specific clearing does not reset
statistics. Full clearing saves reset statistics even when recording is disabled.
A sample remains eligible if its duration was read successfully but unlink fails.
The warm-on-init setting gates `warm()`; construction itself does not call it.

## Native dependencies

`CttsAudioCache` uses C++ filesystem/POSIX I/O, OpenSSL hashing and JSON handling.
It does not invoke Python, NumPy or PyDub. WAV duration analysis follows PyDub's
PCM header path, including millisecond ties-to-even rounding and 8/16/24/32-bit
widths. Other encodings use native `ffprobe`/`ffmpeg` processes. The captured
container cases cover WAV PCM16/24/32/float/double, WAVEX, FLAC, AIFF and Vorbis;
each native result is compared against real PyDub. Converter discovery beyond
the configured PATH's ffprobe/ffmpeg, exotic stream metadata and arbitrary
codec/transport errors remain unverified. PyDub and NumPy notices are retained
under `licenses`.

## Remaining scope

The interface accepts UTF-8 strings and JSON configuration/metadata. It does not
establish every arbitrary Python object, mapping key, non-finite JSON value,
unbounded integer, datetime ISO variant, locale or mutable-attribute behavior.
Concurrent mutation, filesystem races, non-Linux behavior and logging are not
fully ported or verified. Clock-call order is compared using deterministic
timestamps; the native default clock formats the system's naive local time.

The cache now composes with complete native CTTS construction/warmup and buffered
generation, as documented in `ctts_generate_README.md`. A standalone bounded
stream bridge is covered by `audio_stream_bridge_README.md` and native byte
streaming by `ctts_stream_README.md`. Async transport, TTS pools/voice guard and
public speech integration remain unfinished. Passing
this fixture does not complete any speech endpoint's parity claim.
