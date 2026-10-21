These fixtures cover the missing-tokenizer branch of the unchanged
faster-whisper 1.0.3/1.2.0 constructors and the reference STT wrapper at revision
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Implementations and test runners are
C++; Python capture programs remain outside the repository under `/tmp`.

`../whisper_tokenizer_loader_{modern,legacy}.jsonl` contains 108 cases per
profile. The real constructor executes with its model engine substituted.
Tokenizer parsing is real; the Hub download boundary records calls or raises
controlled errors. Cases vary the model's multilingual flag, regular/missing
files, directories, file/directory/dangling/looping symlinks, a regular-file
parent and a NUL-containing directory. Captures compare selection and exact
download arguments, including the absence of the model-specific cache root,
and error propagation. The native runner checks the resolver independently.
The generator is `/tmp/ezlocalai_whisper_tokenizer_loader_oracle.py`.

The relevant Python binding of tokenizers 0.22.1 delegates `from_pretrained`
to Python `huggingface_hub.hf_hub_download`; its Rust-only implementation uses
a different path. Source:
https://github.com/huggingface/tokenizers/blob/v0.22.1/bindings/python/src/tokenizer.rs.
The native implementation uses the existing Hub 1.5.0 single-file downloader.
Its 27 new HTTP/cache comparisons are documented in `../hub_snapshot_README.md`.

`../whisper_tokenizer_fallback_model_{modern,legacy}.jsonl` contains four real
constructors and 16 wrapper calls per profile. Each constructor uses the pinned
English/multilingual tiny model with a missing tokenizer file or a directory
at that filename. The default Hub cache contains the pinned OpenAI tokenizer;
the model download root is separate. Source and native runs use real PortAudio,
CPU int8 CTranslate2 inference with four threads, and one OpenBLAS thread.
Cases exercise ordinary segments, words, batch handling, automatic language
detection and beam/prompt settings. Output/error JSON and cleanup are compared
with the established 1e-12 timestamp tolerance. The legacy batch-keyword error
is preserved. Model/audio/VAD provenance is in `../stt_runtime_assets/README.md`.
The generator is `/tmp/ezlocalai_whisper_tokenizer_fallback_model_oracle.py`.

The compressed files contain unchanged tokenizer JSON from the public Hub:

| File | Repository | Revision | Decompressed SHA-256 |
| --- | --- | --- | --- |
| en.jsonz | openai/whisper-tiny.en | 87c7102498dcde7456f24cfd30239ca606ed9063 | 5eb60cec1e77aeeb6869a2bb5a8e01a84c3fe5d072d75369343021fe6f5310d0 |
| multi.jsonz | openai/whisper-tiny | 169d4a4341b33bc18d8881c4b69c2e104e1cc0af | 27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566 |

The assets were fetched with `hf download`. The native downloader independently
fetched both pinned files into `/tmp/ezlocalai-openai-tokenizer-native-cache`;
their hashes match. Public network availability is not required by CTest.
The pinned Hugging Face model card declares Apache-2.0; the upstream Whisper
repository carries the OpenAI MIT notice. Both are retained under `licenses`
as `openai-whisper-Apache-2.0.txt` and `openai-whisper-MIT.txt`.

This does not establish all tokenizer-file parsing/error behavior, all Hub
transport or filesystem errors, alternate tokenizer configurations, all
environment changes after imports, or complete service integration. The public
speech API still uses the earlier whisper.cpp adapter.
