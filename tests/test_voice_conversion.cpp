#include "ezlocalai/http_socket.hpp"
#include "ezlocalai/realtime.hpp"
#include "ezlocalai/tunnel_websocket.hpp"
#include "ezlocalai/voice_audio.hpp"
#include "ezlocalai/voice_conversion.hpp"
#include "ezlocalai/voice_resample.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
using namespace ezlocalai;
using Json = nlohmann::json;
using namespace std::chrono_literals;
namespace {
void check(bool v, const char *s) {
    if (!v)
        throw std::runtime_error(s);
}
int fixture() {
    std::cout << Json{{"ready", true}, {"sample_rate", 24000}}.dump() << std::endl;
    for (std::string line; std::getline(std::cin, line);) {
        auto j = Json::parse(line);
        auto pcm = voice_base64_decode(j.value("pcm_s16", std::string()));
        if (j["op"] == "audio") {
            std::this_thread::sleep_for(80ms);
            for (auto &c : pcm)
                c ^= 0x55;
        }
        std::cout << Json{{"ok", true}, {"sample_rate", 24000}, {"pcm_s16", voice_base64_encode(pcm)}}.dump()
                  << std::endl;
    }
    return 0;
}
VoiceConversionConfig config(const char *self) {
    VoiceConversionConfig c;
    c.executable = std::filesystem::absolute(self);
    c.model = "fixture";
    c.chunk_timeout = 2;
    c.startup_timeout = 2;
    return c;
}
void resampling() {
    bool bad = false;
    try {
        VoiceResampler invalid(0, 0);
    } catch (const std::invalid_argument &) {
        bad = true;
    }
    check(bad, "invalid sample rate rejected");
    for (auto rates : std::vector<std::pair<int, int>>{{24000, 16000}, {16000, 24000}}) {
        std::vector<float> input(24003);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(.4 * std::sin(2 * std::acos(-1.) * 1000 * i / rates.first));
        VoiceResampler whole(rates.first, rates.second), split(rates.first, rates.second);
        auto expected = whole.push(input);
        auto tail = whole.finish();
        expected.insert(expected.end(), tail.begin(), tail.end());
        std::vector<float> actual;
        for (size_t i = 0; i < input.size(); i += 137) {
            auto out = split.push({input.begin() + i, input.begin() + std::min(input.size(), i + 137)});
            actual.insert(actual.end(), out.begin(), out.end());
        }
        tail = split.finish();
        actual.insert(actual.end(), tail.begin(), tail.end());
        check(expected == actual, "resampling is independent of chunk boundaries");
        check(actual.size() == (input.size() * rates.second + rates.first - 1) / rates.first,
              "sample count has no cumulative drift");
        for (auto v : actual)
            check(std::isfinite(v) && std::abs(v) < .5, "finite bounded resampled tone");
    }
}
void paths() {
    const auto dir =
        std::filesystem::temp_directory_path() / ("ezlocalai-voice-path-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir / "voices");
    std::ofstream(dir / "voices/safe.wav") << "fixture";
    std::ofstream(dir / "outside.wav") << "outside";
    std::filesystem::create_symlink(dir / "outside.wav", dir / "voices/escape.wav");
    VoiceConversionConfig c;
    c.voices = (dir / "voices").string();
    check(voice_conversion_reference(c, "safe") == std::filesystem::canonical(dir / "voices/safe.wav"),
          "named reference resolved");
    for (auto name : {"../outside", "escape", "missing", "/outside"}) {
        bool rejected = false;
        try {
            voice_conversion_reference(c, name);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "reference traversal or unknown name rejected");
    }
    std::filesystem::remove_all(dir);
}
void model_blocks() {
    std::vector<float> pcm(10003);
    for (size_t i = 0; i < pcm.size(); ++i)
        pcm[i] = float(i);
    auto collect = [&](size_t step) {
        VoiceBlockBuffer b;
        std::vector<float> out;
        auto sink = [&](const std::vector<float> &v, uint64_t at) {
            check(at == out.size(), "aligned model offsets");
            out.insert(out.end(), v.begin(), v.end());
        };
        for (size_t i = 0; i < pcm.size(); i += step)
            b.push(std::vector<float>(pcm.begin() + i, pcm.begin() + std::min(pcm.size(), i + step)), sink);
        b.finish(sink);
        return out;
    };
    auto a = collect(2560);
    check(a == collect(160) && a == collect(1), "model input independent of tiny transport chunks");
    check(a.size() == 10240 && std::equal(pcm.begin(), pcm.end(), a.begin()),
          "final partial model block retained");
    VoiceBlockBuffer b;
    std::vector<float> out;
    auto sink = [&](const auto &v, uint64_t) { out.insert(out.end(), v.begin(), v.end()); };
    b.finish(sink);
    check(out.empty(), "no fabricated empty speech");
    b.push({.5f}, sink);
    b.finish(sink);
    check(out.size() == 5120 && out[0] == .5f, "short utterance padded for native frontend");
    b.reset();
    out.clear();
    b.finish(sink);
    check(out.empty(), "cancel clears model fragment");
}
void memory_budget() {
    const auto root =
        std::filesystem::temp_directory_path() / ("ezlocalai-voice-memory-" + std::to_string(getpid()));
    std::filesystem::create_directories(root);
    std::ofstream(root / "memory.max") << "2147483648";
    std::ofstream(root / "memory.current") << "1073741824";
    std::ofstream(root / "memory.stat") << "inactive_file 536870912\n";
    check(voice_conversion_free_ram(128, root.string()) == 1.5,
          "container budget includes only reclaimable cache");
    check(voice_conversion_free_ram(1, root.string()) == 1, "host availability remains an upper bound");
    std::ofstream(root / "memory.current") << "2147483648";
    std::ofstream(root / "memory.stat") << "inactive_file 0\n";
    check(voice_conversion_free_ram(128, root.string()) == 0, "full container blocks converter startup");
    std::ofstream(root / "memory.max") << "max";
    check(voice_conversion_free_ram(7, root.string()) == 7, "unlimited cgroup uses host availability");
    std::filesystem::remove_all(root);
}
void process(const char *self) {
    std::atomic<bool> stop{false};
    {
        VoiceConversion v(config(self), "reference", [&] { return stop.load(); });
        v.audio("old!", "old-response");
        std::this_thread::sleep_for(20ms);
        const auto start = std::chrono::steady_clock::now();
        v.cancel();
        check(std::chrono::steady_clock::now() - start < 30ms, "cancel does not wait on inference");
        v.audio("new!", "new-response");
        v.finish("new-response");
        VoiceConversionResult result;
        bool done = false, got = false;
        auto deadline = std::chrono::steady_clock::now() + 3s;
        while (std::chrono::steady_clock::now() < deadline && !done) {
            if (v.pop(result)) {
                check(result.metadata == "new-response", "old generation output discarded");
                if (!result.pcm.empty()) {
                    std::string expected = "new!";
                    for (auto &c : expected)
                        c ^= 0x55;
                    check(result.pcm == expected, "fresh audio preserved after cancel");
                    got = true;
                }
                done = result.finished;
            } else
                std::this_thread::sleep_for(1ms);
        }
        check(done && got, "next turn completed after cancel");
    }
    int status;
    check(waitpid(-1, &status, WNOHANG) < 0 && errno == ECHILD, "owned child reaped");
    {
        VoiceConversion v(config(self), "reference", [] { return false; });
        bool full = false;
        try {
            for (int i = 0; i < 300; ++i)
                v.audio(std::string(7680, 'x'), "response");
        } catch (const std::runtime_error &) {
            full = true;
        }
        check(full, "input queue bounded under overload");
    }
}
std::string frame(unsigned char op, const std::string &s, bool masked) {
    std::string o(1, char(128 | op));
    auto n = s.size();
    if (n < 126)
        o += char((masked ? 128 : 0) | n);
    else {
        o += char((masked ? 128 : 0) | 126);
        o += char(n >> 8);
        o += char(n);
    }
    const unsigned char key[] = {1, 2, 3, 4};
    if (masked)
        o.append(reinterpret_cast<const char *>(key), 4);
    for (size_t i = 0; i < n; ++i)
        o += char(uint8_t(s[i]) ^ (masked ? key[i % 4] : 0));
    return o;
}
Json read_message(int fd, TunnelWebSocketReader &reader, std::string &pending) {
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
        TunnelWebSocketMessage msg;
        if (reader.pop(pending, msg)) {
            if (msg.opcode == 1)
                return Json::parse(msg.payload);
            throw std::runtime_error("Unexpected close/control frame");
        }
        pollfd p{fd, POLLIN, 0};
        if (poll(&p, 1, 10) > 0) {
            char buf[16384];
            auto n = recv(fd, buf, sizeof(buf), 0);
            if (n <= 0)
                break;
            pending.append(buf, size_t(n));
        }
    }
    throw std::runtime_error("Voice event deadline");
}
void relay(const char *self, bool cancel_late, bool early = false, bool overlap = false) {
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 &&
              listen(listener, 1) == 0,
          "engine listen");
    socklen_t size = sizeof(address);
    getsockname(listener, reinterpret_cast<sockaddr *>(&address), &size);
    std::atomic<bool> stop{false};
    auto backend = std::async(std::launch::async, [&] {
        int peer = accept(listener, nullptr, nullptr);
        check(peer >= 0, "engine accept");
        std::string head;
        while (head.find("\r\n\r\n") == std::string::npos) {
            char b;
            check(recv(peer, &b, 1, 0) == 1, "upgrade read");
            head += b;
        }
        auto pos = head.find("Sec-WebSocket-Key: ") + 19;
        const auto key = head.substr(pos, head.find("\r\n", pos) - pos);
        http_socket::send_all(peer, "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: "
                                    "websocket\r\nSec-WebSocket-Accept: " +
                                        tunnel_websocket_accept(key) + "\r\n\r\n");
        TunnelWebSocketReader reader(1024 * 1024, 1);
        std::string input;
        auto send = [&](const Json &j) {
            check(http_socket::send_all(peer, frame(1, j.dump(), false)), "engine event send");
        };
        send(
            {{"type", "session.created"},
             {"session", {{"audio", {{"output", {{"format", {{"type", "audio/pcm"}, {"rate", 24000}}}}}}}}}});
        for (int turn = 0; turn < (cancel_late ? 2 : 1); ++turn) {
            Json command;
            do {
                command = read_message(peer, reader, input);
                if (command.value("fixture_reject", false)) {
                    send({{"type", "error"}, {"error", {{"message", "fixture rejection"}}}});
                    command = Json::object();
                }
            } while (command["type"] != "response.create");
            auto id = "response-" + std::to_string(turn);
            send({{"type", "response.created"}, {"response", {{"id", id}}}});
            if (overlap)
                std::this_thread::sleep_for(35ms);
            send({{"type", "response.output_audio.delta"},
                  {"response_id", id},
                  {"item_id", "item-" + id},
                  {"content_index", 0},
                  {"delta", voice_base64_encode(turn ? "next" : "first!")}});
            send({{"type", "response.output_audio.done"}, {"response_id", id}});
            send({{"type", "response.done"}, {"response", {{"id", id}, {"status", "completed"}}}});
        }
        while (!stop.load())
            std::this_thread::sleep_for(1ms);
        close(peer);
    });
    TunnelWebSocketConnection engine([&] { return stop.load(); });
    std::string initial, error;
    check(engine.connect_endpoint("http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)) +
                                      "/v1/realtime",
                                  "fixture", &initial, &error),
          error.c_str());
    int pair[2];
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "front socketpair");
    auto relaying = std::async(std::launch::async, [&] {
        VoiceConversion converter(config(self), "reference", [&] { return stop.load(); });
        relay_realtime_voice(pair[0], engine, "", initial, converter, "safe", [&] { return stop.load(); }, 5);
    });
    try {
        TunnelWebSocketReader reader(1024 * 1024, 0);
        std::string pending;
        auto created = read_message(pair[1], reader, pending);
        check(created["session"]["audio"]["output"]["voice"] == "safe", "selected voice returned");
        auto send = [&](const Json &j) {
            check(http_socket::send_all(pair[1], frame(1, j.dump(), true)), "client event send");
        };
        if (overlap) {
            send({{"type", "response.create"}, {"fixture_reject", true}});
            check(read_message(pair[1], reader, pending)["type"] == "error", "rejected create forwarded");
        }
        send({{"type", "response.create"}});
        if (overlap) {
            check(read_message(pair[1], reader, pending)["type"] == "response.created",
                  "retry after rejected create");
            send({{"type", "response.create"}});
            check(read_message(pair[1], reader, pending)["type"] == "error",
                  "overlap before first audio rejected locally");
        }
        if (cancel_late) {
            Json e;
            if (!early) {
                e = read_message(pair[1], reader, pending);
                check(e["type"] == "response.created", "created before cancellation");
                std::this_thread::sleep_for(15ms);
            }
            send({{"type", "response.cancel"}});
            bool done = false;
            while (!done) {
                e = read_message(pair[1], reader, pending);
                check(e["type"] != "response.output_audio.delta", "pending audio discarded on cancel");
                if (e["type"] == "response.done") {
                    check(e["response"]["status"] == "cancelled", "cancelled terminal status");
                    done = true;
                }
            }
            send({{"type", "response.create"}});
        }
        bool audio = false, audio_done = false, done = false;
        while (!done) {
            auto e = read_message(pair[1], reader, pending);
            check(e["type"] != "error", "no conversion protocol error");
            if (e["type"] == "response.output_audio.delta") {
                check(!audio_done, "audio precedes audio.done");
                auto expected = cancel_late ? std::string("next") : std::string("first!");
                for (auto &c : expected)
                    c ^= 0x55;
                check(voice_base64_decode(e["delta"]) == expected, "only converted audio delivered");
                audio = true;
            }
            if (e["type"] == "response.output_audio.done") {
                check(audio, "audio.done after converted PCM");
                audio_done = true;
            }
            if (e["type"] == "response.done") {
                check(audio_done, "response.done after converted tail");
                done = true;
            }
        }
        check(audio, "converted audio received");
    } catch (...) {
        stop = true;
        shutdown(pair[1], SHUT_RDWR);
        relaying.get();
        backend.get();
        close(pair[0]);
        close(pair[1]);
        close(listener);
        throw;
    }
    stop = true;
    shutdown(pair[1], SHUT_RDWR);
    relaying.get();
    backend.get();
    close(pair[0]);
    close(pair[1]);
    close(listener);
}
} // namespace
int main(int argc, char **argv) {
    if (argc == 6)
        return fixture();
    try {
        std::cerr << "resampling\n";
        resampling();
        model_blocks();
        memory_budget();
        std::cerr << "paths\n";
        paths();
        std::cerr << "process\n";
        process(argv[0]);
        std::cerr << "relay normal\n";
        relay(argv[0], false);
        std::cerr << "relay cancel\n";
        relay(argv[0], true);
        relay(argv[0], true, true);
        relay(argv[0], false, false, true);
        std::cout << "Voice conversion resampling, bounded IPC, cancellation/recovery and realtime event "
                     "ordering passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
