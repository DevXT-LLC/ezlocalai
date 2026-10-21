# Music-video planning, orchestration and stitching

The local HTTP music-video route now connects native planning/orchestration,
ACE-Step, storyboard rendering, video generation and FFmpeg stitching when
`EZLOCALAI_STORYBOARD_DEPS` is configured (included in the CUDA image). Builds
without those dependencies return 501. The production image has not yet been
updated for this adapter. Three real CUDA requests pass, including newly
generated ACE-Step audio and Rust router URL/base64 responses; deployment is
held for a remaining text-conditioning discrepancy against the original Python
path. CPU noise, scheduler shift and cross-attention timestep corrections restore
the concert scene; replaying Python prompt embeddings also preserves its graphic.

`music_video_cases.json` contains 1,040 executions of current reference code:

- 740 scene-planning cases, including environment overrides and invalid values;
- 70 LTX-aligned frame-count cases;
- 120 lyric selection cases, including Unicode line separators and ties;
- 28 theorem-detection cases;
- 82 executions of the actual `Pipes.generate_music_video` method.

The orchestration capture substitutes music/video inference, graphics,
filesystem resolution and stitching with recording callbacks. It verifies
payload/default precedence, audio reuse, scene prompts/images, condition
precedence, storyboard/overlay decisions, one video-session scope across all
scenes, restoration-before-stitch ordering, final metadata, and failures in
music, scene generation, output resolution, storyboard creation and stitching.
It does not establish actual model execution, pixel rendering or server memory
management. Source and capture hashes are embedded in the fixture. Capture:
`/tmp/ezlocalai-capture-music-video.py`.

`test_music_video` also invokes real FFmpeg/FFprobe with synthetic red/blue
clips and a short audio track. It checks scene ordering, trimming, audio,
overlay composition, base64 output and failed-command propagation. Local
output resolution covers worker/router URLs, quoted filenames, contained
symlinks, escaping/dangling symlinks and URLs whose query merely resembles an
output path. The resolver reuses the native non-strict POSIX realpath helper.
Complete urllib URL parsing/error equivalence remains to be audited.

`music_video_stitch_validation.json` records six direct comparisons between
the actual Python `_stitch_music_video` method and C++ `stitch_music_video`.
Both use the same host FFmpeg 6.1.1 and synthetic input media. All six MP4s are
byte-identical, covering short/long duration, audio padding, RGB and translucent
RGBA overlays, and URL/base64 responses. These hashes are evidence for that
toolchain; they are not portable expectations for other FFmpeg builds.
Artifacts: `build/validation/music-video-stitch`. Comparator:
`/tmp/ezlocalai-compare-music-video-stitch.py`.

Build `test_music_video`, then run:

```
ctest --test-dir build -R '^music_video_reference$' --output-on-failure
```

`MusicVideoOperations::video_session` is deliberately one callback around the
whole scene loop. The server adapter must hold its video lock and displacement
state through that callback, restore models on success or failure, and release
them before stitching. Calling the public video endpoint independently for
each scene would change the Python lifecycle.

The C++ storyboard and theorem-overlay renderers now match 630 installed
Pillow raster/text/composition cases; see `storyboard_README.md`.

`test_ltx_runtime` now exercises both HTTP aliases with synthetic video engine
processes and real FFmpeg. It checks default storyboards, theorem overlays,
reused audio, filtered scene images, positioned-condition precedence, optional
scene audio, URL/base64 output, one LLM handoff across two scenes, and failure
cleanup before generation, in the second scene and during final stitching.
The ordinary video route retains its per-output handoff. This substitutes
model execution and does not establish actual CUDA composition.

`music_video_http_validation.json` records the real combined CUDA and Rust
router cases, 106 passing native tests, 89 unchanged WorkerRouter wire tests,
additional HTTP edge tests, input-image equality and audio/MP4 hashes.

The follow-up numerical and packaged CUDA evidence is in
`ltx_alignment_validation.json`; the complete native suite passes 111 tests,
followed by two passing NF4 regressions. `nf4_validation.json` records native
NF4-derived weight experiments that preserve the graphic. The subsequent native
loader now selects NF4 automatically from selected-device properties and free
VRAM. A real routed storyboard case reproduces the corrected diagnostic MP4
byte for byte using the original cached weights; ordinary routed video also
passes. Both restore the 220000-token context. See `nf4_README.md` for current
runtime validation and the earlier diagnostic provenance. The candidate is now
deployed; `nf4_runtime_validation.json` records the successful production
router/chat/SSE/vision checks and unchanged deployment configuration.

Remaining work includes compressed NF4 storage/GPU matmul, Quanto fallback,
complete residency policy, persistent video-engine weights, model numerical
equivalence and complete image-decoder/URL error equivalence.
Native integer/timeline limits also remain narrower than Python arbitrary-size
integers; infinite-duration planning is rejected instead of looping forever.
