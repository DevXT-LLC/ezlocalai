#pragma once
#include <functional>
#include <string>
namespace ezlocalai {
struct TtsAudioPipelineInput {std::string format,bytes;};
// Pipes.audio_to_audio parses and decodes before acquiring STT. The resulting
// bytes still pass through STT's own base64 decoder, as in the source.
TtsAudioPipelineInput tts_audio_pipeline_input(const std::string &audio_json);
struct TtsAudioPipelineOperations {
    // Returns the JSON-encoded STT result, preserving its type for CTTS.
    std::function<std::string(const std::string &,const std::string &)> transcribe;
    // Returns the CTTS URL/base64 string unchanged.
    std::function<std::string(const std::string &)> generate;
};
std::string tts_audio_to_audio(const std::string &request,const std::string &local_uri_json,const TtsAudioPipelineOperations &operations);
}
