# Current CTTS text policy

`ctts_text_cases.jsonl.gz` contains 44,831 results from unchanged CTTS helpers at
ezlocalai revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The header pins the
whole `ezlocalai/CTTS.py` file and the native Unicode table by SHA-256. The
external capture program is `/tmp/ezlocalai_ctts_text_oracle.py`; the repository
contains only the compressed data and C++ implementation/tests. The test reads
gzip directly through native zlib.

The capture extracts the complete original helper function AST nodes and their
constants, plus the unchanged CTTS `_resolve_language` and `_prepare_text`
methods. It executes their bodies without rewriting statements. Model/Torch
imports are unnecessary for these pure helpers. The reference is CPython 3.12.3
with Unicode 15.0.0.

| Operation | Source comparisons |
| --- | ---: |
| Integer spelling | 2,963 |
| English time/date/number normalization | 4,974 |
| Multilingual cleanup | 4,974 |
| Buffered chunking | 12,960 |
| Streaming chunking | 5,400 |
| Language aliases and fallback title casing | 11,715 |
| Language-dependent text preparation | 585 |
| Safe filename stem | 1,260 |

The runner additionally compares 9,662 representable cases through the existing
C ABI, independently of the direct C++ result. Number cases include the minimum
and maximum signed `long`, thousand/million transitions, random integers and
large decimal integers. The native decimal-string entry point avoids machine
integer truncation; the C ABI remains limited to its declared `long` argument.
The helper spelling deliberately retains source forms such as `twenty onest`.

The previous native helpers counted UTF-8 bytes, supported only ASCII sentence
boundaries and used an older comma algorithm. The replacement counts Unicode
code points and follows both current CTTS chunkers, including their treatment
of punctuation as captured split fields. It preserves the source's short-text
bypass, empty text, zero/negative limit clamp, whitespace-only fallback,
unbroken words, long intermediate subchunks and shorter first streaming chunk.
Tests include fractional numeric thresholds and randomized multilingual input.
The old C ABI comma fixture was corrected to the independently captured source
result; the new algorithm can emit spaces around punctuation and punctuation
at the beginning of a chunk, exactly as the current source does.

Normalization uses Python's Unicode decimal digits and word boundaries. Cases
cover every decimal digit block, boundaries of all word-character ranges,
Unicode whitespace, mixed-case AM/PM, greedy whitespace after times, invalid
months, partial dates/years and numeric identifiers. Cleanup retains the order
of source substitutions: punctuation collapse precedes control removal and
`#` removal follows whitespace collapse. These details can leave repeated
punctuation or doubled spaces; they are intentionally not cleaned a second time.

`tts_unicode.hpp` freezes Unicode 15.0 properties and full per-character lower/
title mappings rather than inheriting the host ICU version. Its data was
captured across all Unicode code points from CPython's `str` operations and
Unicode cased/case-ignorable predicates. The source comparisons exercise every
nonidentity casing mapping, context around case-ignorable ranges, Greek final
sigma and title-case expansions. No Python runtime is linked or invoked.
See `licenses/CPython.txt` for the capture runtime's license.

These are strongly typed C++ text helpers for valid UTF-8. They do not establish
all Python object/type-error or unpaired-surrogate behavior. C ABI strings stop
at NUL; embedded-NUL cases are checked through the direct C++ interface.
Language resolution tests use string languages, and filename tests use string
or absent values with string defaults. JSON/type coercion belongs to the future
CTTS wrapper and the existing public request validator.

This is shared text policy, not an end-to-end CTTS service. Model selection,
constructor settings and CPU fallback are now covered separately in
`ctts_runtime_README.md`. Reference-audio filesystem containment and audio
conversion now have separate implementations in `ctts_conversion_README.md`.
Full composition, cache behavior, streaming transport, model pools and public
speech integration remain unfinished. Actual Qwen worker/wrapper verification is in
`llama_tts_README.md`; these text tests neither replace nor broaden that CPU
model equivalence claim.
