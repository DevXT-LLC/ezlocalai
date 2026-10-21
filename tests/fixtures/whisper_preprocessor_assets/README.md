These fixtures execute the unchanged faster-whisper 1.0.3 and 1.2.0 preprocessor,
transcription controllers, and the reference STT wrapper at revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Capture programs are external under
`/tmp`; first-party implementations and regression runners are C++.

`../whisper_preprocessor_{modern,legacy}.jsonl` contains 32 file/constructor cases
per profile: defaults, accepted keys, ignored keys, duplicate keys, Unicode
integer strings, missing files, directories, malformed JSON, BOM, non-object
JSON, invalid UTF-8, zero divisors and the accepted `self` key that subsequently
fails constructor invocation. File bytes are recorded as hexadecimal.

`../whisper_preprocessor_features_{modern,legacy}.jsonl` contains ten feature
cases per profile. They vary sample rate (8/16/22.05 kHz), hop length, FFT size,
feature bands and chunk length. `audio.f32z` is zlib-compressed little-endian
float32 input; the other compressed files hold source feature output. Each
record carries its output hash and dimensions. The legacy odd-FFT case records
its ValueError. The existing maximum absolute feature tolerance stays 2e-7.
The external generator is `/tmp/ezlocalai_whisper_preprocessor_oracle.py`.

`../whisper_preprocessor_segments_{modern,legacy}.jsonl` adds 40 standard
controller cases per profile, and `../whisper_preprocessor_batch.jsonl` adds 30
batch cases. The unchanged controller methods use real configuration metadata
and substituted inference boundaries. Complete feature hashes, engine calls,
prompts, alignments, segment/word output and retained state are compared. The
fixtures include nonintegral frame timing, clipping, multiple windows, word
rounding and anomalous alignments. The external generator is
`/tmp/ezlocalai_whisper_preprocessor_controller_oracle.py`.

`../whisper_preprocessor_model_{modern,legacy}.jsonl` adds nine constructors and
24 actual-model calls per profile. Temporary model directories contain the
pinned tiny.en weights, configuration, tokenizer and vocabulary plus each
recorded preprocessor file. The unchanged STT constructor/transcription/close
methods use real PortAudio, faster-whisper, CTranslate2, PyAV, tokenizer and VAD
operations. CUDA availability is substituted as false; inference uses four CPU
threads and one OpenBLAS thread. Standard, word and batched calls cover default,
8 kHz, nondefault hop/FFT/chunk settings, incompatible 40/128 bands, malformed
JSON and constructor errors. The legacy batch-keyword error is preserved.
Models, audio and VAD identities are verified by the runner. Exact text, JSON
structure, error categories/messages and file cleanup are compared; timestamps
retain the existing 1e-12 allowance. Audio/model provenance and dependencies are
in `../stt_runtime_assets/README.md`. The external generator is
`/tmp/ezlocalai_whisper_preprocessor_model_oracle.py`.

This coverage does not establish arbitrary preprocessor metadata compatibility.
Float-valued timing fields, negative/oversized integers, nonstandard JSON
numbers, malformed field types, all I/O failures, persistent direct-backend
chunk overrides, other models and hardware remain unverified or incomplete.
The speech API still uses the previous whisper.cpp adapter.
