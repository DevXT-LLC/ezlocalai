# Native STT PortAudio initialization

`stt_audio_cases.jsonl` contains 120 captures through the unchanged PyAudio
0.2.14 Python class and compiled extension. This is the version pinned by all
three reference requirements files. The native implementation and test runners
are C++; Python is used only by the external reference capture.

Source package: `PyAudio-0.2.14.tar.gz`, SHA-256
`78dfff3879b4994d1f4fc6485646a57755c6ee3c19647a491f790a0895bd2f87`.
The unmodified `src/pyaudio/init.c` hash is
`afa4b513f087fa8da55aa2d92f931e43536df0253f18bc39001c05bdd4df3065`;
`src/pyaudio/__init__.py` is
`ed5e998cd26b6b9c2329c22892599253ce1498532b166db70d3d5db56b715c65`.
The source MIT notice is in `licenses/pyaudio-LICENSE`.

The reference extension was compiled against the Ubuntu 24.04
`portaudio19-dev` 19.6.0-1.2build3 headers and the host's `libportaudio.so.2`.
The latter's SHA-256 is
`62eed4dd6a8e105e57bc34f7b1fa252123aec7e4c27b55ceece4d20aa6f9622b`.
For controlled failures, `LD_PRELOAD` supplies the three PortAudio ABI functions
used by initialization/termination from `tests/portaudio_fixture.cpp`. It does
not replace the PyAudio class or extension. The same library supplies these
functions to the native runner. The fixtures compare all calls in order and
exact exception categories/messages, including error-text lookup after the
termination on a failed initialization.

Cases cover all 29 declared PortAudio errors, two unrecognized error codes,
successful and failed termination, repeated direct termination, two simultaneous
instances, dropped objects, reopening, and empty/Unicode/multiline error text.
No stream is opened by these operations. Dropping an object without explicitly
terminating it leaves PortAudio initialized, matching the source; the native
library remains resident across object lifetimes. The STT wrapper's repeated
close behavior is additionally checked with the same native library boundary.

`stt_audio_live.json` was captured separately without `LD_PRELOAD`, using the
actual PyAudio extension and PortAudio library. It checks uninitialized state,
two successful initializations, one termination retaining initialized state,
and final termination restoring the uninitialized error. Nonnegative device
counts are normalized to initialized status, so hardware inventory does not
change the expected trace. The native live test performs the same operations
on real PortAudio. The 30 actual-model STT wrapper captures were also regenerated
with real PyAudio initialization and now run against the default native audio
operations; their inference outputs and errors did not change.

External capture scripts are `/tmp/ezlocalai_portaudio_oracle.py` and
`/tmp/ezlocalai_stt_runtime_model_oracle.py`. The source package and build
artifacts stay under `/tmp/ezlocalai-portaudio-reference`; the oracle environment
is `/tmp/ezlocalai-portaudio-oracle-env`.

Coverage is initialization and termination used by uploaded-audio transcription.
Recording streams, microphone listening, wake detection, device selection,
platform-specific import/load errors and other PyAudio methods are not ported
by this change. Native GPU/cuDNN operations, speech API integration and the other
remaining STT gaps still prevent full parity.
