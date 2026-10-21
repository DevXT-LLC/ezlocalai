#include "ezlocalai/tts_routing.hpp"
#include "ezlocalai/llama_tts.hpp"
#include "ezlocalai/voice_slots.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;

std::string bytes_hex(const std::string &value) {
    std::ostringstream out;
    for (unsigned char c : value) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
    return out.str();
}
std::string name(TtsProvider provider) {
    return provider == TtsProvider::Voice ? "voice" : provider == TtsProvider::Router ? "router" : "general";
}

int main(int argc, char **argv) { try {
    expect(argc == 3, "source and routing fixture paths");
    const auto cases = records(argv[2]);
    expect(hash(read(fs::path(argv[1]) / "app.py")) == cases[0].at("source_sha256"), "unchanged source routing endpoints");
    expect(cases.size() - 1 == cases[0].at("cases"), "all routing cases");
    size_t lazy_streams = 0;
    for (size_t index = 1; index < cases.size(); ++index) {
        const auto &row = cases[index];
        const auto config = row.at("config");
        Json calls = Json::array();
        TtsRoutingOperations ops;
        ops.disabled = [&] { return config.value("environment", Json::object()).value("TTS_ENABLED", "true") == "false"; };
        ops.voice_server_mode = [&] { return config.value("dedicated", false); };
        ops.local_capability = [&] { return config.value("local", true); };
        ops.general_policy = [&] { calls.push_back({{"op", "general.policy"}}); return config.value("use_general", false); };
        ops.uuid = [] { return "fixed-uuid"; };
        ops.client = [&](TtsProvider provider) {
            const auto id = name(provider);
            const auto settings = config.value(id, Json::object());
            calls.push_back({{"op", id + ".client"}});
            auto client = std::make_shared<TtsRemoteClient>();
            client->configured = settings.value("configured", false);
            client->available = [&, id, settings] { calls.push_back({{"op", id + ".check"}}); return settings.value("available", true); };
            client->generate = [&, id, settings](const std::string &request) -> std::optional<std::string> {
                calls.push_back({{"op", id + ".generate"}, {"request", Json::parse(request)}});
                if (settings.value("error", false)) {
                    if (id == "voice") return {};
                    throw TtsError("RuntimeError", id + " failure");
                }
                return unhex(settings.value("hex", "72656d6f7465"));
            };
            client->stream = [&, id, settings](const std::string &request, const TtsEmit &emit, const TtsInterrupt &) {
                calls.push_back({{"op", id + ".stream"}, {"request", Json::parse(request)}});
                if (settings.value("error", false)) {
                    if (id == "voice") return;
                    throw TtsError("RuntimeError", id + " failure");
                }
                const auto packets = settings.value("packets", Json::array({"6162", "6364"}));
                for (size_t at = 0; at <= packets.size(); ++at) {
                    if (int(at) == settings.value("stream_error_at", -1)) {
                        if (id == "voice") return;
                        throw TtsError("RuntimeError", id + " stream failure");
                    }
                    if (at == packets.size()) break;
                    if (!emit(unhex(packets[at]))) break;
                }
            };
            return client;
        };
        const bool streaming = row.at("path") == "/v1/audio/speech/stream";
        auto route = route_tts(row.at("body"), streaming, ops);
        Json headers = {{"content-type", "audio/wav"}};
        int status = 200;
        bool aborted = false;
        std::string body;
        if (route.kind == TtsRouteKind::Disabled) {
            status = 404;
            headers["content-type"] = "application/json";
            body = "{\"detail\":\"Text to speech is disabled.\"}";
        } else if (route.kind == TtsRouteKind::Pdf || route.kind == TtsRouteKind::Audio) {
            const std::string kind = route.kind == TtsRouteKind::Pdf ? "pdf" : "audio";
            calls.push_back({{"op", kind}, {"request", Json::parse(route.request)}});
            headers["content-type"] = "application/json";
            body = Json{{"kind", kind}}.dump();
        } else if (route.kind == TtsRouteKind::Buffered) body = route.bytes;
        else if (route.kind == TtsRouteKind::Stream) {
            // A source StreamingResponse selects the provider without starting
            // its async generator. This also proves the route owns its client.
            for (const auto &call : calls) expect(call.at("op").get<std::string>().find(".stream") == std::string::npos, "no upstream POST during selection");
            ops.client = {};
            try { route.produce([&](const auto &packet) { body += packet; return true; }, {}); }
            catch (const TtsError &) { aborted = true; }
            ++lazy_streams;
        } else if (!streaming) {
            calls.push_back({{"op", "generate"}, {"request", Json::parse(route.request)}});
            body = std::string("RIFF\0\1", 6);
        } else {
            calls.push_back({{"op", "stream"}, {"request", Json::parse(route.request)}});
            for (const auto &packet : {"00000000", "410042", "00000000"}) {
                calls.push_back({{"op", "next"}});
                body += unhex(packet);
            }
            calls.push_back({{"op", "next"}});
            calls.push_back({{"op", "close"}});
        }
        if (streaming && status == 200) headers = {{"content-type", "application/octet-stream"}, {"x-audio-format", "pcm"}, {"x-sample-rate", "24000"}, {"x-bits-per-sample", "16"}, {"x-channels", "1"}};
        else headers["content-length"] = std::to_string(body.size());
        const auto label = "routing " + std::to_string(index);
        compare_json(status, row.at("status"), label + " status");
        compare_json(headers, row.at("headers"), label + " headers");
        compare_json(bytes_hex(body), row.at("hex"), label + " bytes");
        compare_json(aborted, row.at("aborted"), label + " stream completion");
        compare_json(calls, row.at("calls"), label + " provider and model operations");
    }
    // Ordinary router errors fall through, but Python BaseException must escape.
    for (const auto *category : {"CancelledError", "GeneratorExit", "KeyboardInterrupt", "SystemExit"}) {
        auto client = std::make_shared<TtsRemoteClient>();
        client->configured = true;
        client->available = [] { return true; };
        client->generate = [category](const auto &) -> std::optional<std::string> { throw TtsError(category, "cancelled"); };
        TtsRoutingOperations ops;
        ops.disabled = [] { return true; };
        ops.local_capability = [] { return false; };
        ops.client = [client](auto) { return client; };
        bool escaped = false;
        try { route_tts("{\"input\":\"cancel\"}", false, ops); }
        catch (const TtsError &error) { escaped = error.category == category; }
        expect(escaped, std::string(category) + " bypasses router fallback");
    }
    std::cout << cases.size() - 1 << " source routing scenarios match; " << lazy_streams << " deferred remote streams\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
