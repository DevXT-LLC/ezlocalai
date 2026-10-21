# Voice management source comparisons

The native public GET/POST `/v1/audio/voices` routes now preserve the current
source's sorted `Path.glob("*.wav")` stems, including `.wav`, hidden entries,
matching directories and symlinks. Uploads preserve the supplied bytes, apply
the ASCII voice-name allowlist and 100-character limit, resolve the destination
directory with CPython's non-strict POSIX realpath behavior, and select a unique
name using a following stat rather than file readability. Filesystem failures
produce the source plain HTTP 500. Validation precedes the exact lowercase
`TTS_ENABLED == "false"` gate; surrounding whitespace is not removed.

Listing uses the same retained native voice client as public speech. The client
preserves its URL/key, caches positive and negative availability for 30 seconds,
performs separate five-second health and ten-second listing GETs, decodes the
JSON response using its media type/charset, and falls back locally for falsey
JSON or ordinary client errors. Truthy values are serialized by the API rather
than forwarding upstream headers. Non-finite numbers and lone surrogates fail
API serialization as in the source. The native aiohttp transport now retries
an idempotent request once after a pre-response disconnect, using the original
deadline and redirect state; response-body errors do not trigger that retry.

Reference checkout: `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.

- `app.py`: `fab365c963725d80c81063b544d5b18b4cf931263573bdcb236e537519c7d365`
- `Pipes.py`: `8238cef03d4a1db6e775d7f62234bcf998d8f08bd9fc15f1d8a36ea0d3280005`

| Fixture | Captured evidence | SHA-256 |
| --- | --- | --- |
| `voice_files_cases.jsonl` | 247 scenarios / 886 endpoint calls | `97218f9ec055d906b43fa3c846e636f1b4035af4f2051c0f8632400f93281ea2` |
| `voice_remote_cases.jsonl` | 97 scenarios / 110 endpoint calls / 207 upstream requests | `ca399dfe7b21b7f34558a85224578b8305a7a2d4747d75a07078a534d005adca` |

External reference tools are `/tmp/ezlocalai_voice_files_oracle.py` and
`/tmp/ezlocalai_voice_remote_oracle.py`. They compile the unchanged source
functions/classes and execute the original FastAPI routes through ASGI. No
Python implementation or executable test is added to the repository. Captured
versions are FastAPI 0.128.5, Starlette 0.49.3, Pydantic 2.12.5,
python-multipart 0.0.22 and aiohttp 3.13.2.

The filesystem capture uses real temporary files, directories, links and
multipart requests. It checks complete filesystem contents after every call,
including default-name collisions, dangling/cyclic directory links, Unicode
sanitization, repeated requests, binary data, duplicate form fields, missing
files and selected UploadFile validation errors. Local listing disables the
remote client boundary. The remote capture uses the real VoiceServerClient and
singleton functions against a loopback TCP peer, with controlled clock values.
It covers typed JSON, serialization errors, charset aliases, content types,
compression, incomplete bodies/headers, redirects, retry recovery/exhaustion,
keys/modes, expiry boundaries, backward time, retained configuration and shared
speech/listing availability. Native tests compare status/content type, JSON
values or error/audio bytes, and every captured upstream request.

The first native comparison exposed incorrect dangling-link resolution. Later
comparisons exposed missing body reads, lone-surrogate serialization, the
`latin-1` codec alias and aiohttp's idempotent disconnect retry. These were
corrected against source behavior; expectations were not changed to fit native
results. The existing HTTP test's unsorted local-list expectation was replaced
by the current source's sorted order.

These tests do not prove arbitrary malformed multipart parsing, all charset/
filename escapes, spilled UploadFile validation serialization, permission/race
timing, all invalid filesystem-name encodings, or every HTTP/TLS/DNS/cancellation
schedule. JSON object ordering, whitespace and numeric spelling are not compared.
The source's global voice client is shared more broadly than these public
speech/listing routes; complete startup/STT/configuration transitions remain
unverified. Router-mode aggregation is a separate route implementation. PDF
speech, model discovery, GPU TTS and deployment packaging remain incomplete.
The voice and speech parity rows therefore stay partial.

Verification: the four focused targets passed in 35.88 seconds. The complete
warning-free C++ Debug build then passed all 118/118 tests in 863.35 seconds,
including the new voice suites, shared transport comparisons, GPU STT and real
speech-model output hashes. No implementation changed during the full run.
