#include "ezlocalai/tts_routing.hpp"
#include "ezlocalai/llama_tts.hpp"
#include "ezlocalai/voice_slots.hpp"
#include <nlohmann/json.hpp>

namespace ezlocalai {
std::string tts_remote_payload(const std::string &request) {
    const auto data = nlohmann::json::parse(request);
    return "{\"input\": " + data.at("text").dump(-1, ' ', true)
        + ", \"voice\": " + data.at("voice").dump(-1, ' ', true)
        + ", \"language\": " + data.at("language").dump(-1, ' ', true) + '}';
}
namespace {
// Python's `except Exception` must not consume cancellation or BaseException.
bool base_exception(const std::exception &error) {
    if (dynamic_cast<const VoiceCancelled *>(&error)) return true;
    const auto *tts = dynamic_cast<const TtsError *>(&error);
    return tts && (tts->category == "CancelledError" || tts->category == "GeneratorExit"
        || tts->category == "KeyboardInterrupt" || tts->category == "SystemExit");
}
TtsRoute remote_stream(std::shared_ptr<TtsRemoteClient> client, const std::string &request) {
    TtsRoute result;
    result.kind = TtsRouteKind::Stream;
    result.produce = [client = std::move(client), request](const TtsEmit &emit, const TtsInterrupt &interrupt) {
        client->stream(request, emit, interrupt);
    };
    return result;
}
std::optional<TtsRoute> remote_buffer(const std::shared_ptr<TtsRemoteClient> &client, const std::string &request) {
    try {
        TtsRoute result;
        result.kind = TtsRouteKind::Buffered;
        result.bytes = client->generate(request).value_or("");
        return result;
    } catch (const std::exception &error) {
        if (base_exception(error)) throw;
        return {};
    }
}
}

TtsRoute route_tts(const std::string &input, bool streaming, const TtsRoutingOperations &ops) {
    using Json = nlohmann::json;
    const auto data = Json::parse(input);
    const auto text = data.at("input").get<std::string>();
    const auto voice = data.value("voice", Json("default"));
    const auto language = data.value("language", Json("auto"));
    const auto request = Json{{"text", text}, {"voice", voice}, {"language", language}}.dump();
    const auto router = [&]() -> std::shared_ptr<TtsRemoteClient> {
        if (ops.local_capability()) return {};
        auto client = ops.client(TtsProvider::Router);
        return client->configured && client->available() ? client : nullptr;
    };
    if (ops.disabled()) {
        if (auto client = router()) {
            if (streaming) return remote_stream(std::move(client), request);
            if (auto result = remote_buffer(client, request)) return std::move(*result);
        }
        TtsRoute result;
        result.kind = TtsRouteKind::Disabled;
        return result;
    }
    if (!streaming && text.compare(0, 5, "data:") == 0) {
        if (text.find("pdf") != std::string::npos) {
            const auto user = data.value("user", Json());
            const auto title = user.is_string() && !user.get_ref<const std::string &>().empty() ? user.get<std::string>() : ops.uuid();
            TtsRoute result;
            result.kind = TtsRouteKind::Pdf;
            result.request = Json{{"title", title}, {"voice", voice}, {"pdf", text}, {"chunk_size", 200}}.dump();
            return result;
        }
        if (text.find("audio/") != std::string::npos) {
            TtsRoute result;
            result.kind = TtsRouteKind::Audio;
            result.request = Json{{"voice", voice}, {"audio", text}}.dump();
            return result;
        }
    }
    if (!ops.voice_server_mode()) {
        const auto client = ops.client(TtsProvider::Voice);
        if (client->configured && client->available()) {
            if (streaming) return remote_stream(client, request);
            const auto bytes = client->generate(request);
            if (bytes && !bytes->empty()) {
                TtsRoute result;
                result.kind = TtsRouteKind::Buffered;
                result.bytes = *bytes;
                return result;
            }
        }
    }
    if (auto client = router()) {
        if (streaming) return remote_stream(std::move(client), request);
        if (auto result = remote_buffer(client, request)) return std::move(*result);
    }
    if (!streaming && ops.general_policy()) {
        const auto client = ops.client(TtsProvider::General);
        if (client->configured && client->available()) {
            if (auto result = remote_buffer(client, request)) return std::move(*result);
        }
    }
    TtsRoute result;
    result.request = request;
    return result;
}
}
