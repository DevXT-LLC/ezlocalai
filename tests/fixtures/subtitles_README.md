# Native subtitle reference captures

The fixtures execute unchanged functions from ezlocalai revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`; headers pin app.py and Pipes.py.
External capture programs are `/tmp/ezlocalai_subtitles_oracle.py` and
`/tmp/ezlocalai_subtitles_model_oracle.py`. All native runtime and test code is C++.

`subtitles_api_cases.jsonl` contains 81 scenarios and 163 multipart HTTP requests
through FastAPI 0.128.5/Starlette 0.49.3 TestClient. The unchanged subtitle endpoint,
translation helper, SRT/VTT formatters, STT pool, resource bookkeeping and voice
slot guard run with model loading/inference and `Pipes.get_response` substituted.
Cases cover transient/preloaded/voice-server/handoff lifecycle, inference and
construction errors, JSON/SRT/VTT/unknown formats, fast-mode decode arguments,
empty/repeated fields, duplicate/blank/Unicode-padded target languages, missing
language/segments, malformed translation responses and translation exceptions.
The native runner compares status, MIME, full output, STT arguments, lifecycle,
pool identities/resource counts and LLM request bodies. Independent languages'
chat call order is not compared; the multiset of their complete requests is.

`subtitle_translation_cases.jsonl` captures 70 direct helper calls. It verifies
exact sequential batch requests and results at word limits 0, 1, 2, 4 and 4000,
including empty strings, Python Unicode whitespace, oversized single segments,
numbering reset per batch, duplicate numbers, Unicode decimal numbers, incomplete
translations, response type failures and preservation of only id/start/end/text
for translated segments. A native barrier check additionally requires different
target languages to enter translation concurrently.

`subtitles_api_model_{modern,legacy}.jsonl` uses the unchanged STT wrapper,
actual pinned tiny.en model, PortAudio and faster-whisper 1.2.0/1.0.3. Each profile
contains 16 HTTP requests across four pool modes: JSON source segments, fast-mode
SRT with source/translated languages, VTT translation and invalid audio. LLM
translation is substituted at the get_response boundary. Model, audio and VAD
hashes are verified; complete speech output and temporary-file cleanup are
checked. The same four-inference-thread, one-BLAS-thread profiles as the earlier
STT API captures are used.

Native subtitle translation invokes the existing LLM pipeline directly, without
chat API defaults, queue admission/timeouts or API response conversion. The
pipeline still performs fallback checks, model-slot acquisition and cleanup.
Source-language JSON output retains the original segment dictionaries; translated
output retains precisely the four fields selected by the source helper. Unknown
formats still run translations and return an empty subtitles object. Duplicate
languages still trigger duplicate work and the last result replaces the value.

Scope remains partial: these cases do not prove all Python object shapes,
arbitrary nonfinite/oversized values, HTTP cancellation or failure-time scheduling
of outstanding language tasks. Native futures join outstanding work before
returning a translation error; Python asyncio.gather can propagate an error while
other tasks are still running. Live weighted-LLM translation, GPU API handoff,
diarization, microphone streams, speech audio-to-audio input and complete container/dependency
packaging remain separate incomplete areas. LLM audio/video input now has native
coverage described in stt_pipeline_README.md.

The 2026-10-20 CUDA acceptance supersedes the live weighted-LLM/GPU handoff and
x86 CUDA packaging gaps above for the tested image. The deployed RTX 3090 Ti
worker runs Whisper on the English audio fixture, translates its segments with
the real Qwen3.8-27B LLM into Spanish and French, and preserves IDs and timestamps.
Fast-mode SRT/VTT output also passes. Active STT and LLM CUDA allocations,
four unload/reload cycles, released speech ownership and subsequent routed
chat/SSE/vision are verified. Reporting is checked after heartbeat convergence.
These are local worker APIs; neither router exposes the subtitle route.
See `subtitles_cuda_20261020_validation.json`. The fixture is short and has one
segment; this does not certify longer recordings, all languages, translation
quality in general or the failure/cancellation schedules described above.
