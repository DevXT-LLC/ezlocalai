# Native transcription/translation HTTP reference captures

The source is ezlocalai revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`.
Fixture headers pin `app.py`, `Pipes.py` and, for real-model cases, `STT.py`.
The capture executes unchanged transcription/translation endpoint bodies,
subtitle formatters, STT pool methods, resource-count methods and `_VoiceSlotGuard`.
It uses FastAPI 0.128.5/Starlette 0.49.3 and real multipart HTTP requests through
TestClient. Authentication, delegation and router boundaries are disabled for
this local-inference comparison. Earlier forwarding/authentication tests remain
separate.

`stt_api_cases.jsonl` contains 194 scenarios and 392 HTTP requests. Model loading
and inference are substituted. Cases cover transient models, resident preload,
voice-server lease retention, explicit shared-GPU handoff, repeated requests,
all transcription response formats, translation's format-independent response,
load/inference exceptions, empty scalar defaults, repeated form fields, decimal
integer coercion, bracketed timestamp precedence and signed subtitle timestamps.
The native runner invokes the real NativeServer request handler and compares
HTTP status, MIME, complete response, inference kwargs, load/close/handoff events,
resident/transient/available/leased model identities, pending cleanup and resource
usage. Source transcription inference ValueErrors become HTTP 400; translation
and model-construction failures remain HTTP 500. The local model parameter is
ignored in favor of the captured WHISPER_MODEL, as in the source.

`stt_api_model_{modern,legacy}.jsonl` add 56 HTTP requests using pinned tiny.en
weights, actual STT wrappers, PortAudio and faster-whisper 1.2.0/1.0.3. Both use
four inference threads and one BLAS thread. Headers pin model/audio hashes and
dependency versions. Each profile exercises all four pool/handoff modes with
JSON, text, word-bearing verbose JSON, SRT, VTT, translation and invalid audio.
Bodies use an `<AUDIO>` placeholder for the pinned `stt_runtime_assets` bytes;
recorded base64 inference arguments use the same placeholder. All other fields,
responses and lifecycle data are retained. Native tests additionally verify
real temporary-audio cleanup.

`stt_api_main_reference` launches the compiled ezlocalai-native executable with
CUDA hidden, the pinned local model/VAD assets and no configured whisper.cpp
executable. It sends actual TCP requests, checks transcription, subtitles and HTTP 400,
then checks clean shutdown and audio cleanup. This verifies executable factory
selection as well as the in-process handler.

The external capture programs are `/tmp/ezlocalai_stt_api_oracle.py` and
`/tmp/ezlocalai_stt_api_model_oracle.py`. An initial tracing-argument collision
made those first captures invalid; they were replaced and normal-request success
assertions added before verification. All first-party runtime/test code is C++;
Python is not linked or invoked by native inference or tests.

This does not establish full speech parity. The subtitle endpoint now has separate
native coverage described in subtitles_README.md. Native LLM audio/video input coverage is described in stt_pipeline_README.md; the separate speech audio-to-audio path still uses whisper.cpp. Native diarization, microphone streams, HTTP
cancellation, live GPU API/model handoffs, all malformed/oversized/nonfinite form
values, alternate runtime profiles in the executable and complete container/
dependency packaging remain incomplete or unverified. The executable selects
native transcription/translation/subtitles only when the CTranslate2/VAD adapter is built.

Subsequent CUDA acceptance is recorded in `fleet_hardware_20261020_validation.json`
and `subtitles_cuda_20261020_validation.json`. It verifies actual GPU
transcription through Rust in direct/tunnel modes, the local translation
endpoint on English audio, and local subtitles translated by the real LLM into
Spanish/French. Active allocations, speech cleanup, LLM unload/reload and
follow-up inference pass on the packaged RTX 3090 Ti image. The original
production settings are restored. This supersedes the GPU handoff and x86 CUDA
packaging gaps above for these cases; it does not certify non-English Whisper
translation quality, longer recordings or exhaustive cancellation behavior.
