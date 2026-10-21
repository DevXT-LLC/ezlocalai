#pragma once
#include "ezlocalai/tts_service.hpp"
#include <map>
namespace ezlocalai {
// app.py's gTTS fallback, pinned to gTTS 2.5.3's default tokenizer and RPC.
struct GttsRequest {
    std::string url,body;
    std::map<std::string,std::string> headers;
};
struct GttsResponse { int status=200; std::string reason="OK",body; };
struct GttsOperations {
    std::function<GttsResponse(const GttsRequest &)> post;
    std::function<std::string(const std::string &)> to_wav;
};
std::string gtts_language(const std::string &language_json,const std::string &text);
std::vector<std::string> gtts_tokenize(const std::string &text);
std::vector<GttsRequest> gtts_prepare(const std::string &text,const std::string &language_json="null");
std::string gtts_response_audio(const GttsResponse &response);
std::string gtts_generate_wav(const std::string &text,const std::string &language_json="null",GttsOperations operations={});
// Native curl and ffmpeg subprocess adapters; no Python runtime is used.
GttsOperations gtts_native_operations(GttsOperations operations={});
TtsRequestOperations tts_gtts_operations(TtsRequestOperations operations={},GttsOperations fallback={});
}
