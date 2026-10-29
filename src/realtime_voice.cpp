#include "ezlocalai/http_deadline.hpp"
#include "ezlocalai/realtime.hpp"
#include "ezlocalai/tunnel_websocket.hpp"
#include "ezlocalai/voice_audio.hpp"
#include "ezlocalai/voice_conversion.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <deque>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <stdexcept>
namespace ezlocalai {
namespace {
using Json = nlohmann::json;
std::string wire_frame(unsigned char op, const std::string &s) {
    std::string out(1, char(128 | op));
    const uint64_t n = s.size();
    if (n < 126)
        out += char(n);
    else if (n <= 65535) {
        out += char(126);
        out += char(n >> 8);
        out += char(n);
    } else {
        out += char(127);
        for (int i = 56; i >= 0; i -= 8)
            out += char(n >> i);
    }
    out += s;
    return out;
}
struct Outbound {
    std::string wire;
    size_t sent = 0;
    bool response = false;
};
} // namespace
void relay_realtime_voice(int client, TunnelWebSocketConnection &engine, std::string from_client,
                          std::string from_engine, VoiceConversion &converter, const std::string &voice,
                          const std::function<bool()> &stopped, double idle_seconds) {
    constexpr size_t budget = 1024 * 1024;
    TunnelWebSocketReader frontend(4 * budget, 1), upstream(4 * budget, 0);
    std::deque<Outbound> output;
    size_t output_bytes = 0, held_bytes = 0;
    std::deque<Json> held;
    std::string response_id, audio_metadata;
    bool audio_started = false, finish_queued = false, cancelled = false, server_done = true,
         conversion_done = false;
    uint64_t event_number = 0;
    auto last = std::chrono::steady_clock::now();
    bool closing = false, awaiting_created = false, cancel_before_created = false;
    std::deque<std::string> engine_pongs;
    const auto queue_wire = [&](unsigned char op, const std::string &bytes, bool response = false) {
        auto frame = wire_frame(op, bytes);
        if (output_bytes + frame.size() > budget)
            throw std::runtime_error("Converted voice playback queue exceeds limit");
        output_bytes += frame.size();
        output.push_back({std::move(frame), 0, response});
    };
    const auto send_event = [&](const Json &event) {
        queue_wire(1, event.dump(), event.value("type", std::string()).rfind("response.", 0) == 0);
    };
    const auto event_id = [&] { return "event_meanvc2_" + std::to_string(++event_number); };
    const auto error = [&](const std::string &message) {
        send_event({{"type", "error"},
                    {"event_id", event_id()},
                    {"error",
                     {{"type", "invalid_request_error"},
                      {"code", "voice_conversion_error"},
                      {"message", message}}}});
    };
    const auto flush = [&] {
        if (output.empty())
            return true;
        auto &front = output.front();
        HttpDeadline slice(.005, false, stopped);
        auto n = http_write(client, nullptr, front.wire.data() + front.sent, front.wire.size() - front.sent,
                            slice);
        if (n > 0) {
            front.sent += size_t(n);
            last = std::chrono::steady_clock::now();
            if (front.sent == front.wire.size()) {
                output_bytes -= front.wire.size();
                output.pop_front();
            }
            return true;
        }
        return n < 0 && errno == ETIMEDOUT;
    };
    const auto complete_conversion = [&] {
        if (!conversion_done)
            return;
        while (!held.empty()) {
            send_event(held.front());
            held.pop_front();
        }
        held_bytes = 0;
        finish_queued = false;
        audio_started = false;
        audio_metadata.clear();
    };
    const auto start_finish = [&] {
        if (audio_started && !finish_queued) {
            converter.finish(audio_metadata);
            finish_queued = true;
        }
    };
    const auto cancel = [&] {
        if (awaiting_created)
            cancel_before_created = true;
        converter.cancel();
        cancelled = true;
        audio_started = false;
        finish_queued = false;
        conversion_done = false;
        // A partly written frame must finish to preserve RFC6455 framing. All
        // other unsent response frames are discarded; clients clear playback.
        for (auto i = output.begin(); i != output.end();)
            if (i->response && i->sent == 0) {
                output_bytes -= i->wire.size();
                i = output.erase(i);
            } else
                ++i;
        Json done;
        for (const auto &e : held)
            if (e.value("type", std::string()) == "response.done")
                done = e;
        held.clear();
        held_bytes = 0;
        audio_metadata.clear();
        if (server_done && !response_id.empty()) {
            if (done.is_null())
                done = {{"type", "response.done"}, {"response", {{"id", response_id}}}};
            done["event_id"] = event_id();
            done["response"]["status"] = "cancelled";
            send_event(done);
        }
    };
    const auto handle_upstream = [&](Json event) {
        const auto type = event.at("type").get<std::string>();
        if (type == "session.created" || type == "session.updated") {
            if (event.contains("session")) {
                auto &s = event["session"];
                if (s.contains("audio") && s["audio"].contains("output")) {
                    const auto &format = s["audio"]["output"]["format"];
                    if (format.value("rate", 0) != 24000 ||
                        format.value("type", std::string()) != "audio/pcm")
                        throw std::runtime_error("MeanVC2 requires native 24kHz PCM output");
                    s["audio"]["output"]["voice"] = voice;
                }
                s["voice_conversion"] = {{"engine", "meanvc2"}, {"voice", voice}, {"bandwidth_rate", 16000}};
            }
            send_event(event);
            return;
        }
        if (type == "error" && awaiting_created) {
            awaiting_created = false;
            server_done = true;
            cancel_before_created = false;
        }
        if (type == "response.created") {
            if (audio_started || finish_queued)
                throw std::runtime_error("Overlapping converted voice responses");
            response_id = event.at("response").at("id").get<std::string>();
            cancelled = cancel_before_created;
            cancel_before_created = false;
            awaiting_created = false;
            server_done = false;
            conversion_done = false;
            send_event(event);
            return;
        }
        if (type == "response.output_audio.delta") {
            if (cancelled)
                return;
            if (event.value("response_id", std::string()) != response_id)
                throw std::runtime_error("Converted audio response identity changed");
            const auto pcm = voice_base64_decode(event.at("delta").get<std::string>());
            if (pcm.size() % 2 || pcm.size() > 48000)
                throw std::runtime_error("Invalid or oversized native audio packet");
            if (pcm.empty())
                return;
            if (finish_queued)
                throw std::runtime_error("Native audio arrived after audio completion");
            if (!audio_started) {
                audio_metadata = event.dump();
                audio_started = true;
            } else {
                const auto first = Json::parse(audio_metadata);
                if (first.value("item_id", std::string()) != event.value("item_id", std::string()) ||
                    first.value("content_index", -1) != event.value("content_index", -1))
                    throw std::runtime_error(
                        "Multiple audio parts in one converted response are unsupported");
            }
            for (size_t pos = 0; pos < pcm.size(); pos += 7680)
                converter.audio(pcm.substr(pos, std::min<size_t>(7680, pcm.size() - pos)), audio_metadata);
            return;
        }
        if (type == "response.output_audio.done" && !cancelled)
            start_finish();
        if (type == "response.done") {
            server_done = true;
            const auto status = event.at("response").value("status", std::string());
            if (status != "completed") {
                cancel();
                held.clear();
                held_bytes = 0;
                event["response"]["status"] = status;
                // cancel() emits locally only when the server was already done;
                // this actual upstream terminal event is sufficient here.
                if (!output.empty() && output.back().response) {
                    output_bytes -= output.back().wire.size();
                    output.pop_back();
                }
                send_event(event);
                return;
            }
            if (cancelled) {
                event["response"]["status"] = "cancelled";
                send_event(event);
                return;
            }
            start_finish();
        }
        if (cancelled && type.rfind("response.", 0) == 0)
            return;
        if (finish_queued && !conversion_done && type.rfind("response.", 0) == 0) {
            held_bytes += event.dump().size();
            if (held_bytes > budget || held.size() >= 128)
                throw std::runtime_error("Voice completion queue exceeds limit");
            held.push_back(std::move(event));
        } else
            send_event(event);
    };
    try {
        while (!stopped()) {
            if (idle_seconds > 0 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - last).count() >=
                    idle_seconds)
                break;
            if (!flush())
                break;
            if (closing) {
                if (output.empty())
                    break;
                continue;
            }
            if (engine.writing()) {
                if (engine.flush_frame() < 0)
                    break;
            }
            if (!engine.writing() && !engine_pongs.empty()) {
                engine.queue_frame(10, engine_pongs.front());
                engine_pongs.pop_front();
            }
            VoiceConversionResult converted;
            if (output_bytes < budget / 2 && converter.pop(converted)) {
                if (converted.generation == converter.generation() && !cancelled) {
                    if (!converted.pcm.empty()) {
                        auto event = Json::parse(converted.metadata);
                        event["event_id"] = event_id();
                        event["delta"] = voice_base64_encode(converted.pcm);
                        send_event(event);
                        last = std::chrono::steady_clock::now();
                    }
                    if (converted.finished) {
                        conversion_done = true;
                        complete_conversion();
                    }
                }
            }
            // Read only a bounded amount from either side, retaining writable
            // engine frames and frontend backpressure while controls stay live.
            if (!engine.writing()) {
                TunnelWebSocketMessage msg;
                if (frontend.pop(from_client, msg)) {
                    last = std::chrono::steady_clock::now();
                    if (msg.opcode == 8) {
                        converter.cancel();
                        queue_wire(8, msg.payload);
                        closing = true;
                        continue;
                    }
                    if (msg.opcode == 9) {
                        queue_wire(10, msg.payload);
                        continue;
                    }
                    if (msg.opcode == 10)
                        continue;
                    if (msg.opcode != 1)
                        throw std::runtime_error("Converted realtime accepts JSON events only");
                    auto event = Json::parse(msg.payload);
                    const auto type = event.at("type").get<std::string>();
                    if (type == "response.cancel") {
                        const bool native_active = !server_done;
                        cancel();
                        if (native_active)
                            engine.queue_frame(1, msg.payload);
                    } else if (type == "response.create" &&
                               (awaiting_created || audio_started || finish_queued ||
                                (!response_id.empty() && !server_done)))
                        error("Cancel or finish the current converted response before starting another");
                    else if (type == "session.update" && event.contains("session") &&
                             event["session"].contains("audio") &&
                             event["session"]["audio"].contains("output") &&
                             event["session"]["audio"]["output"].contains("voice")) {
                        auto &selected = event["session"]["audio"]["output"]["voice"];
                        if (!selected.is_string() || selected.get<std::string>() != voice)
                            error("Choose the reference voice in the initial voice query parameter");
                        else {
                            event["session"]["audio"]["output"].erase("voice");
                            engine.queue_frame(1, event.dump());
                        }
                    } else {
                        if (type == "response.create") {
                            awaiting_created = true;
                            server_done = false;
                        }
                        engine.queue_frame(1, msg.payload);
                    }
                    last = std::chrono::steady_clock::now();
                }
                if (frontend.error_code())
                    throw std::runtime_error("Invalid client WebSocket frame");
                if (from_client.size() < 4 * budget) {
                    pollfd p{client, POLLIN, 0};
                    if (poll(&p, 1, 0) > 0) {
                        char bytes[16384];
                        HttpDeadline slice(.005, false, stopped);
                        auto n = http_read(client, nullptr, bytes, sizeof(bytes), slice);
                        if (n > 0)
                            from_client.append(bytes, size_t(n));
                        else if (!n)
                            break;
                        else if (errno != ETIMEDOUT)
                            break;
                    }
                }
            }
            if (output_bytes < budget / 2 && converter.queued_bytes() < budget / 2) {
                TunnelWebSocketMessage msg;
                if (upstream.pop(from_engine, msg)) {
                    last = std::chrono::steady_clock::now();
                    if (msg.opcode == 8) {
                        converter.cancel();
                        queue_wire(8, msg.payload);
                        closing = true;
                        continue;
                    }
                    if (msg.opcode == 9) {
                        if (engine_pongs.size() >= 8)
                            throw std::runtime_error("Too many pending engine pings");
                        engine_pongs.push_back(msg.payload);
                    } else if (msg.opcode != 10) {
                        if (msg.opcode != 1)
                            throw std::runtime_error("Unexpected binary native realtime output");
                        handle_upstream(Json::parse(msg.payload));
                    }
                    last = std::chrono::steady_clock::now();
                }
                if (upstream.error_code())
                    throw std::runtime_error("Invalid engine WebSocket frame");
                if (from_engine.size() < 4 * budget && engine.readable(0)) {
                    char bytes[16384];
                    auto n = engine.read_some(bytes, sizeof(bytes));
                    if (n > 0)
                        from_engine.append(bytes, size_t(n));
                    else if (!n)
                        break;
                }
            }
            pollfd waiting{client, short(engine.writing() ? 0 : POLLIN), 0};
            poll(&waiting, 1, 2);
        }
    } catch (const std::exception &e) {
        converter.cancel();
        try {
            error(e.what());
            queue_wire(8, std::string("\x03\xf3", 2));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
            while (!output.empty() && !stopped() && std::chrono::steady_clock::now() < deadline)
                if (!flush())
                    break;
        } catch (...) {
        }
    }
}
} // namespace ezlocalai
