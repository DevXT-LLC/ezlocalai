#pragma once
#include <string>
#include <vector>

namespace ezlocalai {
struct SttAudioInput { std::string extension, bytes; };
std::string stt_audio_extension(const std::string &format);
std::string stt_base64_decode(const std::string &audio);
std::string stt_base64_decode_bytes(const std::string &audio);
// STT._transcribe_audio_sync's format normalization and base64 decode phase.
// Arguments are JSON values; failures use LlmResponseError.
SttAudioInput stt_audio_input(const std::string &audio, const std::string &format);
bool llm_audio_handoff(const std::string &generic_mode, const std::string &service_mode,
    const std::string &keep_loaded, const std::string &voice_server,
    const std::string &text_server, const std::vector<std::string> &models);
}
