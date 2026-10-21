# Native STT inside LLM input preparation

These fixtures pin Pipes.py at ezlocalai revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Capture programs remain outside the
repository: `/tmp/ezlocalai_stt_pipeline_oracle.py`,
`/tmp/ezlocalai_stt_pipeline_http_oracle.py` and
`/tmp/ezlocalai_stt_pipeline_model_oracle.py`. Native implementation and tests
are C++.

The capture extracts all five unchanged `async with self._stt_lock` blocks from
`Pipes._get_response_internal`. It supplies the original local input variables,
then executes each complete block with the unchanged STT pool, ResourceManager
methods and voice-slot guard. Constructors, inference and idle/unload/restore
boundaries are controlled; no statement inside a captured block is rewritten.

`stt_pipeline_cases.jsonl` contains 160 scenarios/640 calls: five source sites,
four pool/handoff modes, capacities one/two, and success, ValueError, RuntimeError
or constructor failure. Calls include failure recovery and repeated task keys.
Comparisons cover returned value/error, inference kwargs, usage count during
inference, lifecycle order, resident/transient/available/leased identities,
pending cleanup and registered resource state. LLM calls add a second usage mark;
audio failures retain their leases, while video calls destroy in finally.
Successful audio calls destroy even in voice-server mode.

`stt_pipeline_http_cases.jsonl` adds 24 source scenarios/96 requests with unique
request keys. The C++ runner uses NativeServer, the managed LlamaCpp backend and
a native process fixture for the LLM. The STT factory is substituted. It checks
source state/events for inline audio, chat/completion audio_format and video,
including inference/construction failures and recovery. The two legacy blocks
remain covered in isolation above; they are unreachable for ordinary JSON after
list content is flattened. Additional unchanged full-prefix captures verify that
audio_url/video_url placed on a message with string content are ignored. Video uses real frame/audio extraction from the earlier pinned media
fixture. Base64 payloads are normalized to `<AUDIO>` in the event comparison;
prior video/audio tests independently compare decoded bytes. Additional checks
verify STT versus LLM load order, preprocessing-error HTTP 500, nonfatal video STT
errors, model-child cleanup and subsequent text requests. Two zero-timeout shared
GPU handoff cases check the source's already-active LLM lease: audio fails before
STT loads, video omits its transcript, and LLM admission recovers afterward.

`stt_pipeline_model_{modern,legacy}.jsonl` contains 48 calls using the unchanged
STT wrapper, actual pinned tiny.en weights, PortAudio and faster-whisper 1.2.0 or
1.0.3. Each profile exercises audio/video cleanup in transient/preloaded/voice/
handoff mode, with speech, invalid audio and recovery. Complete speech output,
errors, lifecycle, usage and temporary-file cleanup are compared. Headers pin
STT/Pipes source, model/audio hashes and dependency versions; native runners
also verify VAD model hashes. The profiles use four inference threads and one
BLAS thread. Handoff at this layer uses controlled idle/unload/restore boundaries.
The executable TCP smoke additionally sends real speech to chat's audio_format
path with whisper.cpp unavailable and verifies the transcript reaches the echo
LLM backend.

This does not establish all speech parity. HTTP/native inference cancellation,
live weighted-LLM/GPU handoffs, every concurrent request-key interleaving,
diarization, microphone streams, arbitrary configuration edges and complete
container/dependency packaging remain incomplete. The speech endpoint's separate
audio-to-audio/PDF-to-audio path is not migrated here. Non-adapter builds retain
the existing whisper.cpp path. Subtitle failure-time background scheduling is
also still partial, as documented in subtitles_README.md.
