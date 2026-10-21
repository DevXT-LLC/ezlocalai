# Voice startup and configuration lifetime

The 219 scenarios retain the original fixture's configuration and action lists.
Expected outcomes were independently recaptured on 2026-10-18 from reference
commit `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. The fixture header pins
Pipes.py, app.py, Globals.py, ModelSettings.py, ezlocalai/LlamaTTS.py and
ezlocalai/Speculative.py. It also records the capture driver's SHA-256.

The capture executes the source Pipes constructor, voice pool methods,
_VoiceSlotGuard, ResourceManager, retained voice-client configuration, and
app.py's text_to_speech route using unchanged AST function/class definitions.
Every request runs in a distinct asyncio task. STT/TTS model stand-ins record
load, inference and close; they return `transcript` and `YWJj`, or raise the
requested inference failure. CPU-only probes and 16 GiB RAM are injected.
LLM loading, handoff/restoration and cleanup scheduling are controlled
stand-ins; source voice policy, lease ownership and pool bookkeeping execute
normally. The precache file check is injected for each scenario. No model
weights or remote servers are used.

Each action records both pool states, resource records, pending cleanup and
the cumulative event trace. Responses use Starlette JSON/plain response
serialization. Unexpected source exceptions abort capture rather than being
silently converted into a golden HTTP 500. The fixture covers 1,854 actions,
2,073 state/event snapshots and 843 responses: 788 successes, 37 deliberate
runtime failures (including CPU admission refusal), and 18 disabled responses.
Final event traces include 1,631 actual stand-in inference events.

The previous fixture SHA-256 was
`8142cd46ee7a14eac572e5e48c9ba9948b0b0d357f031708c38f63c35c438646`.
Its responses predominantly failed before inference and it recorded no
inference events. The independent capture yields successful requests for
those scenarios. The original capture driver was unavailable, so its precise
failure cause is not established. Do not infer production behavior from
those old errors.

The C++ comparison now clears every scenario-specific environment override
between cases, including flags absent from the built-in defaults table.
Without that isolation, per-service unload flags from an earlier case changed
later startup and pool behavior. All 219 independently recaptured scenarios
match after this test correction; no production voice policy was changed to
obtain the result. The capture driver is retained at
`/tmp/recapture-voice-lifetime.py` in the validation environment.

This fixture verifies the listed startup/configuration transitions and
sequential data-URI audio requests. It does not prove actual STT/TTS inference,
CUDA packaging, arbitrary concurrent environment mutation, remote routing,
PDF synthesis or every initialization failure path.
