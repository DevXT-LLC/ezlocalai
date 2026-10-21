These fixtures were captured using tokenizers 0.22.1 and the unchanged faster-whisper 1.2.0 tokenizer wrapper. The 1.0.3 wrapper has the same encode/decode/word-splitting behavior; its separate prompt and suppression functions are executed alongside the 1.2.0 functions. Source file hashes appear in `../whisper_tokenizer_cases.tsv`.

The bundled compressed tokenizer JSON files preserve the exact source bytes:

- English: Systran/faster-whisper-tiny.en revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba`, tokenizer.json SHA-256 `929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df`.
- Multilingual: Systran/faster-whisper-large-v3 revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`, tokenizer.json SHA-256 `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca`.

The C++ test verifies hashes before loading. All `.json.z` files are zlib-compressed UTF-8 JSON. The `*-decode.json.z` arrays contain ordinary and timestamp-aware decoding results for every token ID through five IDs beyond each vocabulary. This covers 102,239 single-token comparisons, including byte fragments, special tokens and unknown timestamp IDs.

The TSV contains 1,741 ByteLevel splitting cases, 3,482 complete BPE encodings, 800 multi-token decode cases, 7,100 Unicode/space/language word-splitting cases, 1,260 constructor/task/language cases, 640 prompt cases, 28 version-specific suppression cases, two non-speech token lists, two exhaustive decode arrays and two model identities. Inputs include NUL/control bytes expressed as Unicode text, contractions, non-ASCII whitespace, combining marks, emoji, mixed scripts, special-token strings, long repeated text, timestamp tokens and deterministic random text/token sequences.

`whisper_unicode.hpp` is generated reference data: tokenizers.Regex classifies every Unicode scalar value for `\p{L}`, `\p{N}` and `\s`. These produce 677 letter ranges, 144 number ranges and 10 whitespace ranges. The C++ ByteLevel splitter uses these tables to avoid silently substituting the system Unicode version. The tokenizer implementation is C++; no Rust or Python runtime is required. The test does not establish equivalence for arbitrary user-defined tokenizer configurations or other tokenizers dependency versions.

External capture programs: `/tmp/ezlocalai_whisper_tokenizer_oracle.py` and `/tmp/ezlocalai_whisper_unicode_oracle.py`. Upstream algorithm license notices are in `licenses/tokenizers.txt` and `licenses/faster-whisper.txt`.
