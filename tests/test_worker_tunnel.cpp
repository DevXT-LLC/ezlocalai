#include "ezlocalai/native_server.hpp"
#include "ezlocalai/tunnel_connection.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <poll.h>
#include <csignal>
#include <pthread.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using Json = nlohmann::json;
using namespace std::chrono_literals;
static std::atomic<int> observed_local_port{0}, observed_read_timeout{0};
// Observe the actual worker-to-local-server socket budget without waiting
// ten minutes for the former timeout. Keep the real kernel socket operation.
extern "C" int setsockopt(int fd, int level, int option, const void *value, socklen_t length) noexcept {
    if (level == SOL_SOCKET && option == SO_RCVTIMEO && length == sizeof(timeval)) {
        sockaddr_in peer{}; socklen_t size = sizeof(peer);
        if (getpeername(fd, reinterpret_cast<sockaddr *>(&peer), &size) == 0
            && peer.sin_family == AF_INET && ntohs(peer.sin_port) == observed_local_port)
            observed_read_timeout = static_cast<const timeval *>(value)->tv_sec;
    }
    return static_cast<int>(syscall(SYS_setsockopt, fd, level, option, value, length));
}
static void check(bool value, const char *message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
static void wait_for(const std::function<bool()> &ready, const char *message) {
    const auto end = std::chrono::steady_clock::now() + 20s;
    while (!ready() && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(5ms);
    check(ready(), message);
}
static void send_all(int fd, const std::string &bytes) {
    size_t at = 0;
    while (at < bytes.size()) {
        auto n = send(fd, bytes.data() + at, bytes.size() - at, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        check(n > 0, "send"); at += n;
    }
}
static std::atomic<int> interrupted_receives{0};
static ssize_t receive(int fd, void *data, size_t size, int flags) {
    ssize_t count;
    do { count = recv(fd, data, size, flags); if (count < 0 && errno == EINTR) ++interrupted_receives; } while (count < 0 && errno == EINTR);
    return count;
}
static thread_local const char *read_context = "WebSocket frame";
static std::string read_exact(int fd, size_t count) {
    std::string result(count, '\0'); size_t at = 0;
    while (at < count) { auto n = receive(fd, result.data() + at, count - at, 0); if (n <= 0) std::cerr << read_context << ": fd=" << fd << " expected=" << count << " received=" << at << " result=" << n << " errno=" << errno << "\n"; check(n > 0, "receive"); at += n; }
    return result;
}
static std::string read_request(int fd) {
    read_context = "HTTP request";
    std::string result;
    while (result.find("\r\n\r\n") == std::string::npos) result += read_exact(fd, 1);
    auto pos = result.find("Content-Length: ");
    if (pos != std::string::npos) result += read_exact(fd, std::stoul(result.substr(pos + 16)));
    read_context = "WebSocket frame";
    return result;
}
static void send_frame(int fd, const std::string &value, unsigned char opcode = 1, bool final = true) {
    std::string frame(1, static_cast<char>((final ? 0x80 : 0) | opcode));
    if (value.size() < 126) frame += static_cast<char>(value.size());
    else if (value.size() < 65536) { frame += static_cast<char>(126); frame += static_cast<char>(value.size() >> 8); frame += static_cast<char>(value.size()); }
    else { frame += static_cast<char>(127); for (int i = 7; i >= 0; --i) frame += static_cast<char>(static_cast<uint64_t>(value.size()) >> (8 * i)); }
    send_all(fd, frame + value);
}
static std::pair<unsigned char, std::string> receive_frame(int fd) {
    auto h = read_exact(fd, 2); uint64_t count = static_cast<unsigned char>(h[1]) & 127;
    check((h[1] & 128) != 0, "client frames masked");
    if (count == 126 || count == 127) {
        auto sizes = read_exact(fd, count == 126 ? 2 : 8); count = 0;
        for (unsigned char c : sizes) count = (count << 8) | c;
    }
    check(count < 1024 * 1024, "bounded test frame");
    auto mask = read_exact(fd, 4), bytes = read_exact(fd, count);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] ^= mask[i % 4];
    return {static_cast<unsigned char>(h[0] & 15), bytes};
}
static Json receive_json(int fd) { auto frame = receive_frame(fd); check(frame.first == 1, "text frame"); return Json::parse(frame.second); }
static std::string encode(const std::string &value) {
    std::string out(4 * ((value.size() + 2) / 3), '\0');
    EVP_EncodeBlock(reinterpret_cast<unsigned char *>(out.data()), reinterpret_cast<const unsigned char *>(value.data()), value.size());
    return out;
}
struct Listener {
    int fd, port;
    bool ipv6;
    explicit Listener(bool use_ipv6 = false) : ipv6(use_ipv6) {
        fd = socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, 0); check(fd >= 0, "listener socket");
        if (ipv6) {
            sockaddr_in6 address{}; address.sin6_family = AF_INET6; address.sin6_addr = in6addr_loopback;
            check(bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "IPv6 bind");
            socklen_t length = sizeof(address); getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length); port = ntohs(address.sin6_port);
        } else {
            sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            check(bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "bind");
            socklen_t length = sizeof(address); getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length); port = ntohs(address.sin_port);
        }
        check(listen(fd, 32) == 0, "listen");
    }
    int accept_one() {
        int client; do { client = accept(fd, nullptr, nullptr); } while (client < 0 && errno == EINTR); check(client >= 0, "accept");
        timeval timeout{5, 0}; setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)); return client;
    }
    ~Listener() { close(fd); }
    std::string url() const { return std::string(ipv6 ? "http://[::1]:" : "http://127.0.0.1:") + std::to_string(port); }
};
struct MetadataBackend : ezlocalai::InferenceBackend {
    std::string chat(const std::string &) override { return "{}"; }
    std::string completion(const std::string &) override { return "{}"; }
    std::string embedding(const std::string &) override { return "{}"; }
    std::map<std::string, int64_t> loaded_model_contexts() const override { return {{"first", 512}}; }
};
static volatile sig_atomic_t fixture_signal_seen = 0;
static void fixture_signal(int) { fixture_signal_seen = 1; }
static void check_interrupted_read() {
    int pair[2]; check(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair");
    struct sigaction handler{}, previous{};
    handler.sa_handler = fixture_signal; sigemptyset(&handler.sa_mask);
    check(sigaction(SIGUSR1, &handler, &previous) == 0, "install fixture signal");
    const auto reader = pthread_self();
    std::thread writer([&] {
        const auto end = std::chrono::steady_clock::now() + 2s;
        while (!interrupted_receives && std::chrono::steady_clock::now() < end) {
            check(pthread_kill(reader, SIGUSR1) == 0, "interrupt fixture read");
            std::this_thread::sleep_for(5ms);
        }
        send_all(pair[1], "x");
    });
    check(read_exact(pair[0], 1) == "x" && fixture_signal_seen && interrupted_receives > 0, "fixture retries interrupted receive");
    writer.join(); close(pair[0]); close(pair[1]); sigaction(SIGUSR1, &previous, nullptr);
}
int main() {
    check_interrupted_read();
    Listener router(true), local;
    observed_local_port = local.port;
    setenv("REQUEST_TIMEOUT", "120", 1);
    unsetenv("VIDEO_GENERATION_TIMEOUT"); unsetenv("ACE_STEP_TIMEOUT");
    const std::string large_body = "payload" + std::string(13 * 1024 * 1024, 'p');
    std::atomic<bool> large_body_ok{false}, backpressure_started{false}, backpressure_closed{false};
    std::atomic<int> tunnel{-1}, connections{0}, registrations{0}, heartbeats{0}, deregistrations{0};
    std::atomic<bool> stop{false}, slow_started{false}, slow_closed{false}, headers_ok{false};
    std::atomic<int> slow_count{0}, handshake_attempts{0};
    std::vector<std::thread> handlers;
    std::thread router_thread([&] {
        while (!stop) {
            pollfd p{router.fd, POLLIN, 0}; if (poll(&p, 1, 20) <= 0) continue;
            int fd = router.accept_one(); auto request = read_request(fd);
            check(request.find("Authorization: Bearer router-key") != std::string::npos, "router authorization");
            if (request.rfind("GET ", 0) == 0) {
                check(request.find("Host: [::1]:" + std::to_string(router.port)) != std::string::npos, "IPv6 Host header");
                const std::string prefix = "Sec-WebSocket-Key: ";
                const auto begin = request.find(prefix) + prefix.size();
                const auto key = request.substr(begin, request.find("\r\n", begin) - begin);
                const bool invalid = ++handshake_attempts == 1;
                send_all(fd, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "
                    + (invalid ? "wrong" : ezlocalai::tunnel_websocket_accept(key)) + "\r\n\r\n"
                    + (invalid ? "\x81\x0c{\"t\":\"ping\"}" : ""));
                if (invalid) {
                    char byte; const auto count = receive(fd, &byte, 1, 0);
                    check(count == 0 || (count < 0 && errno == ECONNRESET), "invalid challenge closes without dispatch"); close(fd);
                } else { tunnel = fd; ++connections; }
            } else {
                if (request.find("/register ") != std::string::npos || request.find("/heartbeat ") != std::string::npos) {
                    const auto state = Json::parse(request.substr(request.find("\r\n\r\n") + 4));
                    check(state.at("cap_slots").at("tts").at("capacity") == 3, "router receives independent TTS capacity");
                    check(state.at("cap_slots").at("stt").at("capacity") == 1, "router receives shared STT capacity");
                    check(state.at("extra").at("llm_unload_dependent_capabilities") == Json::array({"stt"}), "registration and heartbeat retain speech sharing dependency");
                }
                if (request.find("/register ") != std::string::npos) {
                    const auto state = Json::parse(request.substr(request.find("\r\n\r\n") + 4));
                    check(state.at("total_vram_gb") == 10.75, "registration preserves visible-device VRAM precision");
                    check(state.at("free_vram_gb") == 4.25, "registration uses resource-manager free VRAM");
                    check(state.at("model_context").at("first") == 512, "loaded context overrides configured maximum");
                    check(state.at("model_context").at("second") == 8192, "per-model CSV context and parallel slots");
                    ++registrations;
                }
                if (request.find("/heartbeat ") != std::string::npos) ++heartbeats;
                if (request.find("/deregister ") != std::string::npos) ++deregistrations;
                send_all(fd, "HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"ok\":true}"); close(fd);
            }
        }
    });
    std::thread local_thread([&] {
        while (!stop) {
            pollfd p{local.fd, POLLIN, 0}; if (poll(&p, 1, 20) <= 0) continue;
            int fd = local.accept_one();
            handlers.emplace_back([&, fd] {
                const auto request = read_request(fd);
                if (request.find(" /slow ") != std::string::npos) {
                    ++slow_count; slow_started = true;
                    // Deliberately never send headers. Cancellation must interrupt
                    // this phase too, without waiting for the 600-second timeout.
                    char byte; check(receive(fd, &byte, 1, 0) == 0, "cancellation closes local request"); slow_closed = true;
                } else if (request.find(" /stream ") != std::string::npos) {
                    send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nfirst\r\n");
                    char byte; check(receive(fd, &byte, 1, 0) == 0, "stream cancellation closes local socket"); slow_closed = true;
                } else if (request.find(" /backpressure ") != std::string::npos) {
                    send_all(fd, "HTTP/1.1 200 OK\r\nContent-Length: 67108864\r\n\r\n");
                    backpressure_started = true;
                    const std::string chunk(65536, 'x'); size_t sent = 0;
                    while (sent < 64*1024*1024) {
                        const auto count = send(fd, chunk.data(), chunk.size(), MSG_NOSIGNAL);
                        if (count < 0 && errno == EINTR) continue;
                        if (count <= 0) break;
                        sent += count;
                    }
                    check(sent < 64*1024*1024, "backpressure cancellation interrupts local upload");
                    backpressure_closed = true;
                } else if (request.find(" /broken ") != std::string::npos) {
                    send_all(fd, "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort");
                } else {
                    if (request.find(" /large ") != std::string::npos)
                        large_body_ok = request.substr(request.find("\r\n\r\n") + 4) == large_body;
                    headers_ok = request.find("Authorization: Bearer forwarded-key") != std::string::npos
                        && request.find("X-Request-ID: trace-123") != std::string::npos
                        && request.find("payload") != std::string::npos;
                    send_all(fd, "HTTP/1.1 201 Created\r\nContent-Type: application/octet-stream\r\nContent-Length: 3\r\nX-Request-ID: reply-123\r\n\r\nabc");
                    // The worker must finish by Content-Length without waiting
                    // for a peer using keep-alive to close its connection.
                    char byte; check(receive(fd, &byte, 1, 0) == 0, "fixed-length response released");
                }
                close(fd);
            });
        }
    });
    setenv("ROUTER_URL", router.url().c_str(), 1); setenv("ROUTER_API_KEY", "router-key", 1);
    setenv("EZLOCALAI_URL", local.url().c_str(), 1); setenv("WORKER_TUNNEL", "true", 1);
    setenv("WORKER_LABEL", "tunnel-test", 1); setenv("WORKER_HEARTBEAT_INTERVAL", "2", 1);
    setenv("DEFAULT_MODEL", "first,second", 1);
    setenv("LLM_MAX_TOKENS", "8192,24576", 1);
    setenv("N_PARALLEL", "2,3", 1);
    for (auto name : {"TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED"}) setenv(name, "false", 1);
    setenv("TTS_ENABLED", "true", 1); setenv("STT_ENABLED", "true", 1);
    setenv("VOICE_SERVER", "", 1);
    setenv("TTS_N_PARALLEL", "3", 1); setenv("STT_N_PARALLEL", "4", 1);
    setenv("TTS_UNLOAD_LLM_DURING_GENERATION", "false", 1);
    setenv("STT_UNLOAD_LLM_DURING_GENERATION", "true", 1);
    setenv("MUSIC_UNLOAD_LLM_DURING_GENERATION", "false", 1);
    ezl_env_config config{}; ezl_load_config(&config);
    std::snprintf(config.api_key, sizeof(config.api_key), "%s", "different-local-key");
    auto probes = ezlocalai::native_resource_probes();
    probes.gpu_count = [] { return 2; };
    probes.total_per_gpu = [] { return std::vector<double>{3.5, 7.25}; };
    probes.free_per_gpu = [] { return std::vector<double>{1.5, 2.75}; };
    auto resources = std::make_shared<ezlocalai::ResourceManager>(probes);
    ezlocalai::NativeServer worker(config, std::make_unique<MetadataBackend>(), resources);
    std::string error; check(worker.start("127.0.0.1", 0, &error), error.c_str());
    wait_for([&] { return tunnel >= 0 && registrations && heartbeats; }, "registration and tunnel");
    std::cerr << "stage: registered and upgraded\n";
    int ws = tunnel;
    // A complete JSON value inside a non-final frame must not dispatch early.
    send_frame(ws, "{\"t\":\"ping\"}", 1, false);
    send_frame(ws, "between-fragments", 9);
    auto control = receive_frame(ws);
    check(control.first == 10 && control.second == "between-fragments", "control frame interleaved with unfinished JSON");
    send_frame(ws, "", 0);
    check(receive_json(ws).at("t") == "pong", "application dispatch after final continuation");
    std::cerr << "stage: fragmented ping complete\n";
    const auto expect_reply = [&](const std::string &id) {
        auto reply = receive_json(ws);
        check(reply.at("id") == id && reply.at("status") == 201, "framed request dispatch");
        check(receive_json(ws).at("data_b64") == encode("abc"), "framed response bytes");
        check(receive_json(ws).at("t") == "resp_end", "framed response complete");
    };
    Json framed{{"t", "req"}, {"id", "fragmented"}, {"method", "POST"}, {"path", "/fast"},
        {"headers", {{"Authorization", "Bearer forwarded-key"}, {"X-Request-ID", "trace-123"}}},
        {"body_b64", encode("payload")}, {"note", "💩"}};
    const auto fragmented = framed.dump();
    const auto split = fragmented.find("💩") + 2;
    send_frame(ws, fragmented.substr(0, split), 1, false);
    send_frame(ws, fragmented.substr(split), 0);
    expect_reply("fragmented");
    std::cerr << "stage: fragmented request complete\n";
    framed["id"] = "large"; framed["path"] = "/large"; framed["body_b64"] = encode(large_body);
    send_frame(ws, framed.dump());
    expect_reply("large");
    std::cerr << "stage: large request complete\n";
    check(large_body_ok, "request larger than 16 MiB preserves full decoded body");
    check(observed_read_timeout == 600, "ordinary tunnel requests retain their default budget");
    int timeout_case = 0;
    const auto expect_timeout = [&](const char *path, int seconds) {
        const auto id = "timeout-policy-" + std::to_string(++timeout_case);
        framed["id"] = id; framed["path"] = path; framed["body_b64"] = encode("payload");
        send_frame(ws, framed.dump()); expect_reply(id);
        check(observed_read_timeout == seconds, "actual local socket honors media/request timeout");
    };
    expect_timeout("/v1/videos/generations", 3600);
    expect_timeout("/v1/videos/music?test=1", 3600);
    expect_timeout("/v1/videos/music/generations", 3600);
    expect_timeout("/v1/audio/music", 1800);
    expect_timeout("/v1/audio/music/generations", 1800);
    setenv("REQUEST_TIMEOUT", "4_000.2", 1);
    expect_timeout("/fast", 4001);
    expect_timeout("/v1/videos/music", 4001);
    setenv("VIDEO_GENERATION_TIMEOUT", "4500", 1);
    expect_timeout("/v1/videos/generations", 4500);
    setenv("ACE_STEP_TIMEOUT", "5000", 1);
    expect_timeout("/v1/videos/music", 5000);
    expect_timeout("/v1/audio/music", 5000);
    setenv("REQUEST_TIMEOUT", "nan", 1);
    setenv("VIDEO_GENERATION_TIMEOUT", "invalid", 1);
    setenv("ACE_STEP_TIMEOUT", "-1", 1);
    expect_timeout("/fast", 600);
    expect_timeout("/v1/videos/generations", 3600);
    expect_timeout("/v1/audio/music", 1800);
    setenv("REQUEST_TIMEOUT", "120", 1);
    unsetenv("VIDEO_GENERATION_TIMEOUT"); unsetenv("ACE_STEP_TIMEOUT");
    double expected_reconnect = 3.0; // The initial rejected handshake already waited two seconds.
    const auto expect_close = [&](const std::string &wire, int code) {
        const auto disconnected_at = std::chrono::steady_clock::now();
        const int previous_connections = connections;
        send_all(ws, wire);
        auto reply = receive_frame(ws);
        check(reply.first == 8 && reply.second.size() >= 2, "protocol error sends close frame");
        check(((static_cast<unsigned char>(reply.second[0]) << 8) | static_cast<unsigned char>(reply.second[1])) == code,
            "source-compatible protocol close code");
        close(ws);
        wait_for([&] { return connections > previous_connections; }, "reconnect after protocol error");
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - disconnected_at).count();
        check(elapsed >= expected_reconnect - 0.1, "short-lived connection retains growing backoff");
        std::cout << "reconnect waited " << elapsed << " seconds (expected " << expected_reconnect << ")\n";
        expected_reconnect *= 1.5;
        ws = tunnel;
        send_frame(ws, "{\"t\":\"ping\"}");
        check(receive_json(ws).at("t") == "pong", "fresh reader after protocol error");
    };
    expect_close(std::string("\x80\0", 2), 1002); // continuation without initial frame
    expect_close(std::string("\x81\x01\xff", 3), 1007); // invalid completed UTF-8
    expect_close(std::string("\x81\x7f\0\0\0\0\x04\0\0\0", 10), 1009); // declared 64 MiB
    const auto request = [&](const std::string &id, const std::string &path) {
        send_frame(ws, Json{{"t", "req"}, {"id", id}, {"method", "POST"}, {"path", path},
            {"headers", {{"Authorization", "Bearer forwarded-key"}, {"Content-Type", "application/json"}, {"X-Request-ID", "trace-123"}}},
            {"body_b64", encode("payload")}, {"stream", true}}.dump());
    };
    request("slow", "/slow"); wait_for([&] { return slow_started.load(); }, "local request started");
    request("slow", "/slow"); // duplicate in-flight ID is ignored
    send_frame(ws, "{\"t\":\"ping\"}");
    check(receive_json(ws).at("t") == "pong", "application pong during blocked inference");
    send_frame(ws, "ping-payload", 9);
    const auto pong = receive_frame(ws); check(pong.first == 10 && pong.second == "ping-payload", "WebSocket pong payload");
    request("fast", "/fast");
    auto start = receive_json(ws);
    check(start.at("id") == "fast" && start.at("status") == 201, "concurrent dispatch");
    check(start.at("headers").at("x-request-id") == "reply-123", "response headers forwarded");
    check(receive_json(ws).at("data_b64") == encode("abc"), "binary response bytes");
    check(receive_json(ws).at("t") == "resp_end", "fixed-length response ends promptly");
    check(headers_ok && slow_count == 1, "request headers/body and duplicate suppression");
    send_frame(ws, "{\"t\":\"cancel\",\"id\":\"slow\"}");
    wait_for([&] { return slow_closed.load(); }, "cancel before response headers");
    slow_closed = false;
    request("stream", "/stream"); check(receive_json(ws).at("t") == "resp_start", "stream starts");
    check(receive_json(ws).at("data_b64") == encode("first"), "incremental chunk");
    send_frame(ws, "{\"t\":\"cancel\",\"id\":\"stream\"}");
    wait_for([&] { return slow_closed.load(); }, "cancel during body");
    // Keep the router's receive window full longer than the former one-second
    // write timeout. Incoming cancellation must still reach both local calls.
    slow_started = false; slow_closed = false;
    request("backpressure", "/backpressure"); wait_for([&] { return backpressure_started.load(); }, "large response starts");
    request("beside-backpressure", "/slow"); wait_for([&] { return slow_started.load(); }, "request admitted during large upload");
    std::this_thread::sleep_for(1300ms);
    send_frame(ws, "{\"t\":\"cancel\",\"id\":\"backpressure\"}");
    send_frame(ws, "{\"t\":\"cancel\",\"id\":\"beside-backpressure\"}");
    wait_for([&] { return slow_closed.load(); }, "second request cancellation received while router is not reading");
    std::cerr << "stage: second request cancelled during backpressure\n";
    wait_for([&] { return backpressure_closed.load(); }, "large response cancellation received while router is not reading");
    send_frame(ws, "after-backpressure", 9);
    bool control_after_backpressure = false;
    while (!control_after_backpressure) {
        const auto frame = receive_frame(ws);
        if (frame.first == 10) control_after_backpressure = frame.second == "after-backpressure";
        else check(frame.first == 1 && Json::parse(frame.second).at("id") == "backpressure", "only already-sent response frames precede control pong");
    }
    request("after-backpressure", "/fast"); expect_reply("after-backpressure");
    std::cerr << "stage: duplex backpressure and cancellation complete\n";
    request("broken", "/broken"); check(receive_json(ws).at("t") == "resp_start", "broken start");
    check(receive_json(ws).at("t") == "resp_chunk", "broken partial data");
    check(receive_json(ws).at("t") == "resp_err", "truncated body is an error, not successful end");
    slow_started = false; slow_closed = false;
    request("disconnect", "/slow"); wait_for([&] { return slow_started.load(); }, "disconnect request starts");
    const int before_disconnect = connections;
    shutdown(ws, SHUT_RDWR); close(ws);
    wait_for([&] { return slow_closed.load(); }, "disconnect cancels active request");
    wait_for([&] { return connections > before_disconnect; }, "reconnect"); ws = tunnel;
    slow_started = false; slow_closed = false;
    request("shutdown", "/slow"); wait_for([&] { return slow_started.load(); }, "shutdown request starts");
    const auto before = std::chrono::steady_clock::now(); worker.stop();
    check(std::chrono::steady_clock::now() - before < 2s, "prompt shutdown during inference");
    check(slow_closed && deregistrations == 1, "shutdown cancels and deregisters"); close(ws);
    stop = true; router_thread.join(); local_thread.join(); for (auto &thread : handlers) thread.join();
    std::cout << "worker tunnel concurrency, cancellation, headers, framing and reconnect passed\n";
}
