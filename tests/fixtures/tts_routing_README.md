# Public TTS provider routing and remote HTTP

`route_tts` implements the public buffered and streaming endpoint selection
from `app.py`, pinned revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.
`NativeServer` uses it before entering the local TTS service. Voice-server
forwarding precedes router forwarding for an unavailable local capability;
general fallback is buffered-only. Disabled TTS can still forward to the router.
A successful empty router/general response is returned, while empty voice audio
falls through. Streaming selects a provider before headers and starts its POST
only when the response writer runs. Voice-server stream errors finish the
transfer; router stream errors truncate it, preserving bytes already delivered.

`tts_routing_cases.jsonl` captures 2,124 scenarios from the unchanged full source
endpoints and `_router_client_for_unserved_capability`, running through FastAPI
ASGI with controlled provider clients and local models. It covers provider
priority, configuration/availability, dedicated mode, disabled/local capability
combinations, empty output, errors, partial streams, nullable voice/language,
and data URI selection/title defaults. The C++ test compares response metadata,
bytes, completion and the complete ordered provider/model operation trace.
It also checks lazy stream ownership and cancellation categories. PDF/audio
pipeline implementations are controlled callbacks in this comparison, not
certified by it. SHA-256:
`254823e430a2fc11e13b7d6753427d921ed1ff0fc19e346708639a73fa889f4a`.

`tts_remote_cases.jsonl` captures 94 scenarios using the unchanged
`VoiceServerClient` and `EzlocalaiClient` classes as well as the original public
endpoints. A local TCP server returns scripted responses; no external service
is called. The C++ test runs each scenario through `NativeServer` in buffered
and deferred modes, for 188 comparisons. Besides downstream status, headers,
binary output, completion and local-model calls, it compares actual upstream
methods, paths, normalized header maps and exact request bytes. Only the dynamic
Host port is normalized. Header order and TCP packet boundaries are not compared.
The source runtime's default User-Agent is retained for these requests.
SHA-256: `1c97838be7e5b7d98c383ab2781f5d611fdac585b2e0d7fa0104174b81664b77`.

Remote coverage includes binary and empty success, HTTP errors, early close,
Content-Length and chunked truncation, delayed audio, gzip (including split
compressed input), invalid gzip, unsupported zstd, 302/307 redirects, redirect
limits, failed/malformed/typed health responses, and disabled router requests.
The native transport shares existing URL, redirect, cookie, deadline and payload
decoding code, with incremental draining for successful streamed responses.
Eight additional native tests cancel blocked header/body reads through downstream
disconnect or application destruction, checking upstream connection closure.
Body-phase cases wait for a complete downstream audio chunk before cancelling;
application destruction must truncate the transfer even for a voice server.
Deferred tests also assert that selection performs only the health probe.

The first remote capture closed redirect connections without advertising
`Connection: close`, producing an aiohttp connection-reuse race. The scripted
responses now explicitly close connections; the fixture was recaptured before
acceptance. Keep-alive reuse/connection-close races remain outside this evidence.
The legacy HTTPS integration test exposed a separate voice health cache; public
speech now shares that cache with existing voice routes. Legacy scenarios that
change voice URL/key now create a new application, matching the source singleton
settings lifetime, as their fallback scenarios already did. Their public speech
request expectations now use the source JSON whitespace, and stream clients
keep their request connection open while consuming output.

The captures are generated externally by `/tmp/ezlocalai_tts_routing_oracle.py`
and `/tmp/ezlocalai_tts_remote_oracle.py`. Their fixture headers pin app/Pipes
SHA-256 and package versions (FastAPI 0.128.5, Starlette 0.49.3, Pydantic 2.12.5,
and aiohttp 3.13.2 for real remote requests). Shipped code and tests are C++.

Remaining scope includes complete data-URI PDF/audio pipelines, internal
WebSocket/voice forwarding, all client settings/cache transitions across routes,
keep-alive connection reuse, cancellation during DNS/connect/TLS, exhaustive
network/parser failures, GPU TTS and deployment packaging. General fallback
selection is covered by the routing fixture and existing fallback tests, but
this remote fixture exercises voice/router clients. Exact 1:1 parity is not
claimed; speech manifest rows remain partial.
