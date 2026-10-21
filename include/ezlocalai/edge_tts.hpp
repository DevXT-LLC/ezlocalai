#pragma once
#include "ezlocalai/tts_service.hpp"
#include <map>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace ezlocalai {
std::vector<std::string> edge_tts_chunks(const std::string &text);
std::string edge_tts_ssml(const std::string &escaped_text, const std::string &voice);
std::string edge_tts_token(double corrected_unix_time);
struct EdgeTtsRequest {
    std::string url;
    std::map<std::string, std::string> headers;
    std::vector<std::string> messages;
};
EdgeTtsRequest edge_tts_request(const std::string &escaped_text, const std::string &voice,
    double unix_time, double clock_skew, const std::string &connection_id,
    const std::string &request_id, const std::string &muid);
class EdgeTtsStream {
    std::string audio_;
    bool received_ = false;
public:
    // Returns false at turn.end. Frame types are text=1 and binary=2.
    bool accept(int type, const std::string &data);
    std::string finish() const;
};
struct EdgeTtsResponse { int status = 101; std::map<std::string, std::string> headers; };
struct EdgeTtsOperations {
    std::function<double()> time;
    std::function<std::string(bool uuid)> identifier;
    std::function<EdgeTtsResponse(const EdgeTtsRequest &, EdgeTtsStream &)> exchange;
    std::shared_ptr<std::atomic<double>> clock_skew;
};
EdgeTtsResponse edge_tts_exchange(const EdgeTtsRequest &request, EdgeTtsStream &stream);
std::string edge_tts_mp3(const std::string &text, const std::string &voice, EdgeTtsOperations operations = {});
}
