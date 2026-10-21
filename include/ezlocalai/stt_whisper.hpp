#pragma once
#include "ezlocalai/stt_runtime.hpp"
#include "ezlocalai/stt_pool.hpp"
#include "ezlocalai/whisper_transcription.hpp"
namespace ezlocalai {
struct SttWhisperConfiguration {
    WhisperTranscriptionProfile profile = WhisperTranscriptionProfile::faster_whisper_1_2_0;
    std::string vad_directory;
    // Optional override; the default resolves local paths and source aliases
    // through the native Hub snapshot cache.
    std::function<std::string(const std::string &,const std::string &)> resolve_model;
};
// Audio defaults to PortAudio. GPU operations default to native Torch when the
// optional LibTorch target is configured. Diarization defaults to the shared
// native MFCC/clustering implementation and may be overridden for testing.
// Model resolution
// returns a CTranslate2 directory; the inference implementation is entirely C++.
SttRuntimeOperations stt_whisper_operations(SttWhisperConfiguration configuration,SttRuntimeOperations platform);
SttPoolOperations stt_whisper_pool_operations(SttWhisperConfiguration configuration,
    std::shared_ptr<ResourceManager> resources,SttRuntimeOperations platform = {});
}
