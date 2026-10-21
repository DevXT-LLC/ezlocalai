# Native local speech service comparisons

`tts_service_cases.jsonl.gz` contains 1,036 scenarios and 5,628 operations from
reference revision `0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Its header pins
`app.py` and `Pipes.py`. The external generator is
`/tmp/ezlocalai_tts_service_oracle.py`; native implementation and tests do not
execute Python.

The capture executes unchanged AST statements from the local portion of
`text_to_speech` and `text_to_speech_stream`, the original gTTS eligibility
helpers, nine original TTS pool methods, ResourceManager registration/usage
methods, and `_VoiceSlotGuard`. Provider routing before the local branches,
Pydantic validation and HTTP response transport are excluded. Response wrappers
expose source bytes/generators directly. Model/system callbacks, environment,
clock, request identity and deferred cleanup are controlled. Permit callbacks
fail on unexpected blocking in the policy fixture; native process tests below
exercise actual waiting and cancellation instead.

Each action compares returned bytes or exact error category/message, callback
order, model close state, stream position/close count, resource usage,
resident/transient/available collections, lease identities, pending cleanup,
voice/context stacks, permits and shared handoff state. Coverage includes
transient, preload, dedicated voice and handoff modes; repeated buffered
requests; null models; gTTS message/configuration eligibility; bad base64;
pre-header stream loading; consumer-task transfer; completion; abandonment;
cancellation; ordinary errors and selected BaseExceptions; and failures in
resource marking, stream close, model cleanup, LLM restoration or permit release.

Source ordering is preserved even for unusual states: buffered generation
performs an explicit resource decrement before optional pool destruction;
dedicated voice mode retains its buffered lease. Streaming does not use gTTS,
releases its lease in dedicated mode too, and delegates handoff cleanup to the
guard. Closing an unstarted generator does not run its cleanup. A failed
explicit streaming resource decrement prevents entry into the later release
finally. A BaseException during pre-header model loading bypasses the source's
Exception-only slot release. Later finally failures can replace earlier errors.

`tts_service_process` covers twelve cases with real native worker processes,
Ctts, ResourceManager, the pool and VoiceSlotGuard. Each of four modes exercises
buffered decoding, completed streaming and blocked-stream cancellation. A
second stream waits for the first slot and then completes. It verifies model
lease retention/removal, queued versus synchronous destruction, handoff
restoration, resource removal and child reaping. Additional checks cover
canceling queued admission, a stream retaining the service after its public
handle is dropped, idempotent finished-stream close, repeated automatic consumer identity and safe rejection of a
missing fallback callback. The process fixture supplies deterministic audio;
it does not run GGUF inference.

`tts_service_model_cases.jsonl` adds 77 operations using complete unchanged CTTS,
AudioCache, AudioStreaming and LlamaTTS with the pinned Qwen CPU worker/model and
speaker asset. It includes buffered synthesis and a cache hit, a two-chunk
stream, abandonment after a chunk length, replacement of the closed model and
final unload. The native comparison checks 70 exact WAV/packet hashes along
with all pool/slot/resource/model state and wrapper event order. Headers pin
source, worker, GGUF, speaker and source dependency identities. Generation uses
four threads, seed 23, eight generation tokens and four warmup tokens. The
external capture is `/tmp/ezlocalai_tts_service_model_oracle.py`.

TtsService requires a caller-provided fallback transport callback. It implements
source eligibility and fallback invocation/lifetime, not the gTTS network
client. Native gTTS/ffmpeg fallback, public HTTP/WebSocket wiring, remote routing,
complete shared-GPU residency orchestration and disconnect behavior remain open.
Typed string results/request IDs do not establish arbitrary Python object
boundaries. Exhaustive scheduler/reentrant interleavings, logging, GPU synthesis
and native dependency packaging also remain incomplete or unverified. The
public speech manifest rows remain partial.
