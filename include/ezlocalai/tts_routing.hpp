#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace ezlocalai {

using TtsEmit = std::function<bool(const std::string &)>;
using TtsInterrupt = std::function<void(std::function<void()>)>;
using TtsProducer = std::function<void(const TtsEmit &, const TtsInterrupt &)>;

enum class TtsProvider { Voice, Router, General };

// Client methods retain their own source error policy: VoiceServerClient
// returns None/ends its stream on ordinary failures; EzlocalaiClient throws.
struct TtsRemoteClient {
    bool configured = false;
    std::function<bool()> available;
    // JSON returned by VoiceServerClient.get_voices; ordinary failures are None.
    std::function<std::optional<std::string>()> voices;
    std::function<std::optional<std::string>(const std::string &)> generate;
    std::function<void(const std::string &, const TtsEmit &, const TtsInterrupt &)> stream;
};

struct TtsRoutingOperations {
    std::function<bool()> disabled, voice_server_mode, local_capability, general_policy;
    std::function<std::shared_ptr<TtsRemoteClient>(TtsProvider)> client;
    std::function<std::string()> uuid;
};

enum class TtsRouteKind { Local, Pdf, Audio, Disabled, Buffered, Stream };
struct TtsRoute {
    TtsRouteKind kind = TtsRouteKind::Local;
    // JSON keyword arguments for the selected local/media operation.
    std::string request;
    std::string bytes;
    // Selection and health probes happen now; the upstream streaming POST
    // starts only when this producer runs, after downstream response headers.
    TtsProducer produce;
};

// Input has already passed TextToSpeech validation. This implements the two
// public endpoint branches; internal PDF/audio/WebSocket pipelines differ.
TtsRoute route_tts(const std::string &input, bool streaming, const TtsRoutingOperations &operations);
// aiohttp's json= payload: keyword order, ASCII escaping and JSON whitespace.
std::string tts_remote_payload(const std::string &request);

}
