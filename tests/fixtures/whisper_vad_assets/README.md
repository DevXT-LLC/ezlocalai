These fixtures execute the unchanged VAD implementations from faster-whisper 1.2.0 and 1.0.3, using NumPy 1.26.4 and ONNX Runtime 1.24.1 with CPUExecutionProvider. Their source hashes are recorded in `../whisper_vad_cases.tsv`:

- faster-whisper 1.2.0 vad.py: `807390d31eac395388692b08cb268aaffd2130a4c4e1ebedeb0dbea80d8352ac`
- faster-whisper 1.0.3 vad.py: `af7bf3eb3f140222004f41185856cd3bb5aeb822588aef6fa41c4fdcf15ffd24`

The policy fixtures cover segmentation with controlled probability sequences, audio chunk collection and original timestamp mapping. The source model boundary is substituted for the policy-only capture; segmentation and mapping functions are unchanged. Real model captures separately exercise the unchanged ONNX calls and preprocessing. Each `.f32z` contains zlib-compressed little-endian float32 samples or probabilities. `../whisper_vad_model_cases.tsv` records model hashes, audio/probability array lengths and expected speech chunks under both library defaults and ezlocalai's STT parameters (500 ms minimum silence, 200 ms speech padding).

Model files come from the corresponding faster-whisper distributions and remain external:

- `silero_encoder_v5.onnx`: `0e9fc8f56407693d283f99045fb95c2b90fdca3433ce5f83e2c121fb4e615075`
- `silero_decoder_v5.onnx`: `8c20344f509846a07ccd85827e8857017fae67fabd58689bec1af79e1d488307`
- `silero_vad.onnx`: `6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396`

The native ONNX Runtime shared library used for this capture has SHA-256 `94ac52f650e195b784587e1cb7d25b4bf074048b50fa9a7127795b2ed34db22e`. Its C++ headers are from the official v1.24.1 release. The model fixtures include silence, generated sinusoids, sample counts around the 448/512 boundaries, real speech, speech separated by silence and the 10,000-window encoder batch boundary. Speech comes from the same public JFK sample documented in `../whisper_kernel_assets/README.md`; combined fixtures insert zero samples into that decoded signal.

`../whisper_speech_cases.tsv` captures the combined VAD, collection, feature extraction, tokenization/prompt/suppression and CTranslate2 generation path for both deployment profiles, with speech, inserted silence and all-silence inputs at beam sizes 1/3, plus initial-prompt cases. The expanded 16-case capture uses each version's actual suppression helper (the earlier 12-case preprocessing capture supplied the modern explicit suppression list for both profiles). It checks filtered audio hashes, exact output token IDs and decoded text using the pinned tiny.en model from the kernel fixtures. This is a decoder-path comparison, not the complete faster-whisper transcription controller. In particular, raw generation on silence still produces tokens; the controller's no-speech rejection remains separate work.

External capture programs: `/tmp/ezlocalai_whisper_vad_oracle.py` and `/tmp/ezlocalai_whisper_speech_oracle.py`. Repository test runners are C++ and do not invoke Python.
