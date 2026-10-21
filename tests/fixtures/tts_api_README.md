# Native local speech HTTP integration

The native executable now supplies a real TTS pool factory to `NativeServer`.
When built with the LibTorch adapter, it supplies the same CTTS hardware and
inference hooks as the real-model service tests. `TTS_N_PARALLEL`, preload policy,
dedicated voice mode, shared voice admission, resource accounting and the native
gTTS fallback compose through `TtsService`. Set `QWEN_TTS_BIN` and the model paths
as described in the native README; deployment discovery remains a separate task.

`tts_api_cases.jsonl` captures 912 scenarios from the unchanged full
`TextToSpeech`, `text_to_speech` and `text_to_speech_stream` definitions in
`app.py`, revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The first record
pins the app SHA-256 and FastAPI/Starlette/Pydantic versions. The external capture
script is `/tmp/ezlocalai_tts_api_oracle.py`; no Python runtime is shipped.
The versions are FastAPI 0.128.5, Starlette 0.49.3 and Pydantic 2.12.5. Fixture
SHA-256: `a84345b4219fa4425a86e7b45297f195651a3e0e4a0747ddd621963e59427e27`.

The capture executes application/json requests through a real FastAPI ASGI application, retaining response-start
and body messages even when streaming raises after headers. Authentication and
remote provider availability are controlled; the model returns deterministic
binary packets or selected errors, and the fallback returns controlled bytes.
The original source request model, validation, endpoint branches, fallback
eligibility and response objects run unchanged. Pool/guard behavior has separate
unchanged-source coverage in `tts_service_README.md`.

The C++ comparison runs each case through `NativeServer::handle_request` in both
buffered and deferred modes: 1,824 comparisons of status, response content type
and audio headers, binary body or parsed validation JSON, transfer completion,
model calls, released permits/leases and resource usage. JSON content lengths
and key ordering are not compared; binary response lengths are. Deferred mode
uses real socket pairs and parses HTTP chunks. Coverage includes missing/null
bodies, malformed JSON, field types/defaults, empty/Unicode text, explicit null
voice/language, disabled flag whitespace/casing, model-load/generation errors,
gTTS eligibility, invalid base64, empty streams and failures at different yields.

`tts_api_process` uses the native CTTS/LlamaTts stack and C++ worker fixture to
block actual generation. Disconnecting the HTTP writer's peer cancels/reaps that
worker, releases admission and lets a queued empty stream finish. Four cases
cover transient, preloaded, dedicated and LLM-handoff modes.

`tts_api_model_reference` starts the actual `ezlocalai-native` executable and
performs requests over TCP using the real Qwen worker/models. It reuses the
pinned source/model/environment evidence in `tts_service_model_cases.jsonl`:
two exact WAV hashes (including a cache hit) and all 65 HTTP stream packet hashes
match the source service capture. This tests main-program factory wiring,
startup/preload, public endpoint dispatch, native inference, HTTP metadata and
chunk framing together. CUDA is hidden for this CPU comparison. The server exits
normally on SIGTERM. The source service/model test independently checks model
and source file hashes; the HTTP test directly verifies the speaker hash.

Public provider selection and voice/router HTTP transport now have separate
source-backed coverage in `tts_routing_README.md`, including unserved/disabled
router dispatch and streaming failures.

Known remaining scope: complete data-URI pipelines, exhaustive remote client
configuration/cache/transport edges, local voices routes, model-management
and cross-modality residency integration, exhaustive cancellation and scheduling,
shutdown of queued requests in all server-stop modes, GPU synthesis, deployment
discovery/packaging, non-JSON request content types and unsupported Python object boundaries. The existing
data-URI and internal forwarding adapters remain partial; these local fixtures do not certify those
branches. Speech rows remain partial and the overall parity goal is incomplete.

The WebSocket route now has separate source-backed coverage in
`tts_websocket_README.md`. The process target now contains 12 HTTP/WebSocket
disconnect/shutdown cases, and the real executable comparison also checks
67 WebSocket frame hashes after the HTTP sequence.

The 2026-10-20 CUDA acceptance extends these CPU/source comparisons:
`fleet_hardware_20261020_validation.json` records actual Qwen WAV and PCM through
Rust WorkerRouter in direct and tunnel modes, and
`stream_cancellation_20261020_validation.json` records active PCM interruption
in both modes, cleared speech admission/model registration, LLM restoration
and successful follow-up chat. Active GPU process allocations are observed.
Temporary settings are removed and production chat/SSE/vision pass afterward.
These records supersede the GPU synthesis and packaging gaps above for the
tested x86 CUDA image; they do not establish identical GPU waveforms or all
handoff, transport and cancellation schedules. PDF-to-speech is out of scope.
