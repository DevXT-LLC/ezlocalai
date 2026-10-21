# Native gTTS fallback evidence

The native `gtts` module implements the local fallback called by the source's
`_generate_gtts_wav_sync`, and `tts_gtts_operations` supplies its callback to
`TtsService`. Public speech HTTP/WebSocket integration is still pending.

Reference: `ezlocalai` revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`,
`app.py` SHA-256 `fab365c963725d80c81063b544d5b18b4cf931263573bdcb236e537519c7d365`.
The reference environment uses gTTS **2.5.3**. Its installed Python files and
the frozen Unicode table are hashed in `gtts_cases.jsonl.gz`'s first record.
The source Dockerfiles allow `gTTS>=2.4.0`; these captures establish behavior
for the installed version, not every allowed dependency version.

`gtts_cases.jsonl.gz` contains 3,230 comparisons: 180 language choices, 2,580
request preparations, and 470 response/error cases. They cover English/Russian
aliases and Cyrillic range boundaries; Unicode length/whitespace, abbreviations,
punctuation, all ASCII characters, long words and randomized text; exact prepared
headers and nested JSON/form bodies; greedy RPC extraction, ignored response
lines, CR/LF splitting, invalid UTF-8/ASCII/base64 and HTTP status errors.
Uncompressed fixture SHA-256:
`e8767f2d1ad310dc0c7c3757735da0884eaf80e7e1a0e552a49ff70b30074103`.

The fixture also records real source FFmpeg conversion of zero, one, and two
MP3 buffers. `gtts_audio.mp3` is derived from the existing speech test audio
using FFmpeg 6.1.1-3ubuntu5. Its SHA-256 is
`79695a1d4fbd2aab2c6fb84f745960fa21ff497b1e5b6e387a7229efb6676fca`.
The nonempty conversions compare every output WAV byte by SHA-256 and size;
empty input compares the actual `CalledProcessError`. Both implementations use
`pipe:0`/`pipe:1`, 24 kHz, one channel, WAV output and the same native FFmpeg.

`gtts_http_cases.jsonl` captures 14 actual reference loopback scenarios / 55 HTTP
requests. The C++ test compares request methods, paths, every header and body,
and the final result/error. It covers 301/302/303/307/308, mixed redirects,
redirect cookies, a fresh session for each text part, redirect limits, gzip,
empty/malformed audio and errors stopping subsequent requests. The transport
uses native curl subprocesses, private temporary files and no shell. Curl's
user configuration is disabled. Source POST-to-GET/header removal is explicit;
curl's automatic redirect policy alone does not match requests here.

`gtts_process_cases.jsonl` captures the unchanged app helper with real converter
child processes returning four nonzero statuses and eight terminating signals.
The native test also covers unavailable executables and connection refusal.
FFmpeg spawning restores SIGPIPE/SIGXFZ/SIGXFSZ defaults like Python subprocess;
the process comparisons run with SIGPIPE ignored in the native parent.
All child processes are reaped and private temporary workspaces are removed.

Capture programs are external reference-only tools at
`/tmp/ezlocalai_gtts_oracle.py`, `/tmp/ezlocalai_gtts_http_oracle.py`, and
`/tmp/ezlocalai_gtts_process_oracle.py`; no Python implementation is shipped.
The reference source is unmodified. The first response mock lacked requests'
prepared-request and consumed-content metadata; those harness errors were
corrected before accepting the fixtures. No request was sent to Google.

Limits: public routing and asynchronous request/disconnect behavior remain
unfinished. External Google availability/content, exhaustive requests-versus-curl
proxy, cookie, TLS, redirect/auth/URL edge behavior, subprocess/resource exhaustion,
Python recursion limits, arbitrary non-string/private-helper inputs, and other
FFmpeg/dependency versions are not established by these comparisons. File-backed
subprocess I/O has different resource limits from the source's memory buffers.
The manifest remains partial; this is not a full equivalence claim.
