#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/native_server.hpp"
#include "fixture_ports.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <zlib.h>

namespace {
using Json = nlohmann::json;
constexpr size_t limit = 16 * 1024 * 1024;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
struct Socket {
    int fd;
    ~Socket() { if (fd >= 0) close(fd); }
};
void send_all(int fd, const std::string &bytes) {
    size_t sent = 0;
    while (sent < bytes.size()) {
        const auto count = send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
        check(count > 0, "fixture TCP send failed");
        sent += static_cast<size_t>(count);
    }
}
std::string gzip(const std::string &body) {
    z_stream stream{};
    check(deflateInit2(&stream, 1, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) == Z_OK, "gzip setup");
    struct End { z_stream &stream; ~End() { deflateEnd(&stream); } } end{stream};
    stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(body.data()));
    stream.avail_in = static_cast<uInt>(body.size());
    std::string output;
    int status;
    do {
        char block[65536];
        stream.next_out = reinterpret_cast<Bytef *>(block);
        stream.avail_out = sizeof(block);
        status = deflate(&stream, Z_FINISH);
        check(status == Z_OK || status == Z_STREAM_END, "gzip encode");
        output.append(block, sizeof(block) - stream.avail_out);
    } while (status != Z_STREAM_END);
    return output;
}
struct Response { int status; std::string body, headers; };
Response request(int port, const std::string &path, const std::string &body,
    const std::string &coding = "", const std::string &authorization = "Bearer gzip-worker-key",
    size_t declared_size = std::string::npos, const std::string &method = "POST", bool chunked = false) {
    Socket socket{::socket(AF_INET, SOCK_STREAM, 0)};
    check(socket.fd >= 0, "fixture TCP socket");
    timeval timeout{20, 0};
    setsockopt(socket.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    sockaddr_in address{}; address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
    check(!connect(socket.fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "fixture TCP connect");
    std::string wire = method + ' ' + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\n";
    if (!authorization.empty()) wire += "Authorization: " + authorization + "\r\n";
    if (!coding.empty()) wire += "Content-Encoding: " + coding + "\r\n";
    if (chunked) {
        wire += "Transfer-Encoding: chunked\r\n\r\n";
        for (size_t at = 0; at < body.size(); at += 7) {
            const auto bytes = std::min<size_t>(7, body.size() - at);
            std::ostringstream frame; frame << std::hex << bytes << "\r\n";
            wire += frame.str() + body.substr(at, bytes) + "\r\n";
        }
        wire += "0\r\n\r\n";
        send_all(socket.fd, wire);
    } else {
        wire += "Content-Length: " + std::to_string(declared_size == std::string::npos ? body.size() : declared_size) + "\r\n\r\n";
        send_all(socket.fd, wire);
        send_all(socket.fd, body);
    }
    shutdown(socket.fd, SHUT_WR);
    std::string response;
    char buffer[4096];
    for (;;) {
        const auto count = recv(socket.fd, buffer, sizeof(buffer), 0);
        check(count >= 0, "fixture TCP response timed out");
        if (!count) break;
        response.append(buffer, static_cast<size_t>(count));
    }
    const auto end = response.find("\r\n\r\n"), space = response.find(' ');
    check(end != std::string::npos && space != std::string::npos, "fixture HTTP response framing");
    return {std::stoi(response.substr(space + 1, 3)), response.substr(end + 4), response.substr(0, end)};
}

class CaptureRouter {
    Socket socket_{::socket(AF_INET, SOCK_STREAM, 0)};
    std::atomic<bool> stopping_{false};
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::pair<std::string, Json>> calls_;
    std::vector<std::string> forwarded_;
    std::string failure_;
public:
    int port = 0;
    std::atomic<int> alive_workers{0};
    CaptureRouter() {
        check(socket_.fd >= 0, "capture router socket");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        check(!bind(socket_.fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) && !listen(socket_.fd, 8), "capture router bind");
        socklen_t size = sizeof(address);
        check(!getsockname(socket_.fd, reinterpret_cast<sockaddr *>(&address), &size), "capture router port");
        port = ntohs(address.sin_port);
        thread_ = std::thread([this] {
            try {
                while (!stopping_) {
                    pollfd ready{socket_.fd, POLLIN, 0};
                    if (poll(&ready, 1, 20) <= 0) continue;
                    Socket peer{accept(socket_.fd, nullptr, nullptr)};
                    if (peer.fd < 0) { if (stopping_) break; throw std::runtime_error("capture router accept"); }
                    timeval timeout{3, 0}; setsockopt(peer.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
                    std::string wire; char buffer[4096]; size_t body_start = std::string::npos, expected = 0;
                    for (;;) {
                        const auto count = recv(peer.fd, buffer, sizeof(buffer), 0);
                        check(count > 0, "capture router body receive");
                        wire.append(buffer, static_cast<size_t>(count));
                        const auto header_end = wire.find("\r\n\r\n");
                        if (header_end == std::string::npos) continue;
                        body_start = header_end + 4;
                        const auto marker = wire.find("Content-Length: ");
                        check((marker != std::string::npos && marker < header_end) || wire.rfind("GET ", 0) == 0,
                            "capture router length");
                        expected = marker != std::string::npos && marker < header_end ? std::stoull(wire.substr(marker + 16)) : 0;
                        if (wire.size() >= body_start + expected) break;
                    }
                    const auto begin = wire.find(' '), end = wire.find(' ', begin + 1);
                    check(begin != std::string::npos && end != std::string::npos, "capture router request line");
                    const auto method = wire.substr(0, begin), path = wire.substr(begin + 1, end - begin - 1);
                    const bool health = method == "GET" && path == "/v1/router/health";
                    const bool forward = method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions");
                    check(health || forward || (method == "POST" && (path == "/v1/router/register"
                        || path == "/v1/router/heartbeat" || path == "/v1/router/deregister")), "unexpected capture router request");
                    auto headers = wire.substr(0, body_start);
                    std::transform(headers.begin(), headers.end(), headers.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                    const auto key = health || forward ? "gzip-client-key" : "gzip-router-key";
                    check(headers.find(std::string("\r\nauthorization: bearer ") + key + "\r\n") != std::string::npos,
                        "worker registration or fallback authentication changed");
                    check(wire.size() == body_start + expected, "capture router stale content length");
                    const auto parsed = expected ? Json::parse(wire.substr(body_start, expected)) : Json();
                    if (forward) {
                        check(headers.find("\r\ncontent-encoding:") == std::string::npos
                            && headers.find("\r\ntransfer-encoding:") == std::string::npos,
                            "decoded fallback retained compressed request headers");
                        check(parsed.at("disable_fallback") == true, "decoded fallback allowed a second hop");
                    }
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        calls_.emplace_back(path, parsed);
                        if (forward) forwarded_.push_back(wire);
                        changed_.notify_all();
                    }
                    const auto response = health ? Json{{"status", "healthy"}, {"alive_workers", alive_workers.load()}}.dump()
                        : forward ? std::string(R"({"id":"routed","object":"chat.completion","model":"gzip-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"from router"},"finish_reason":"stop"}]})") : std::string("{}");
                    send_all(peer.fd, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                        + std::to_string(response.size()) + "\r\nConnection: close\r\n\r\n" + response);
                }
            } catch (const std::exception &error) {
                std::lock_guard<std::mutex> lock(mutex_); failure_ = error.what(); changed_.notify_all();
            }
        });
    }
    ~CaptureRouter() {
        stopping_ = true;
        if (thread_.joinable()) thread_.join();
    }
    std::vector<std::string> forwarded() {
        std::lock_guard<std::mutex> lock(mutex_); check(failure_.empty(), failure_); return forwarded_;
    }
    void check_advertisements() {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto complete = [&] {
            bool registration = false, heartbeat = false;
            for (const auto &call : calls_) {
                registration |= call.first == "/v1/router/register";
                heartbeat |= call.first == "/v1/router/heartbeat";
            }
            return registration && heartbeat;
        };
        check(changed_.wait_for(lock, std::chrono::seconds(8), [&] { return complete() || !failure_.empty(); }), "worker registration/heartbeat timed out");
        check(failure_.empty(), failure_);
        for (const auto &call : calls_) if (call.first == "/v1/router/register" || call.first == "/v1/router/heartbeat")
            check(call.second.at("extra").at("http_request_encodings") == Json::array(), "unset gzip policy advertised compression");
    }
};

struct Recorded {
    std::atomic<bool> fail{false};
    mutable std::mutex mutex;
    std::vector<std::pair<std::string, std::string>> requests;
    size_t size() const { std::lock_guard<std::mutex> lock(mutex); return requests.size(); }
    std::pair<std::string, std::string> last() const {
        std::lock_guard<std::mutex> lock(mutex); check(!requests.empty(), "recorded inference missing"); return requests.back();
    }
};
class Backend final : public ezlocalai::InferenceBackend {
    std::shared_ptr<Recorded> recorded_;
    std::string answer(const std::string &kind, const std::string &body) {
        std::lock_guard<std::mutex> lock(recorded_->mutex);
        recorded_->requests.emplace_back(kind, body);
        return R"({"id":"gzip-fixture","object":"chat.completion","created":0,"model":"gzip-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"fixture"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})";
    }
public:
    explicit Backend(std::shared_ptr<Recorded> recorded) : recorded_(std::move(recorded)) {}
    std::string loaded_models_json() const override { return R"({"llm":{"name":"gzip-fixture"}})"; }
    std::string chat(const std::string &body) override { return answer("chat", body); }
    std::string completion(const std::string &body) override { return answer("completion", body); }
    std::string embedding(const std::string &body) override { return answer("embedding", body); }
    ezlocalai::InferenceResult text_result(const std::string &body, const std::string &replica, bool chat_mode) override {
        if (!recorded_->fail) return InferenceBackend::text_result(body, replica, chat_mode);
        answer(chat_mode ? "chat" : "completion", body);
        return {R"({"error":{"message":"local inference failed"}})", {"Exception", "local inference failed"}};
    }
};
std::shared_ptr<ezlocalai::ResourceManager> resources() {
    auto probes = ezlocalai::native_resource_probes();
    probes.gpu_count = [] { return 0; };
    probes.total_per_gpu = probes.free_per_gpu = [] { return std::vector<double>{}; };
    probes.free_vram = [](auto) { return 0.; };
    probes.allocator_stats = [] { return std::vector<ezlocalai::CudaAllocatorStats>{}; };
    probes.system_ram = probes.available_ram = [] { return 64.; };
    return std::make_shared<ezlocalai::ResourceManager>(probes);
}
void environment(const CaptureRouter &router) {
    for (const auto &[key, value] : ezlocalai::environment_defaults) unsetenv(std::string(key).c_str());
    unsetenv("EZLOCALAI_ROUTER_REQUEST_GZIP");
    for (const auto *key : {"TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED", "WORKER_TUNNEL", "ROUTER_MODE"}) setenv(key, "false", 1);
    for (const auto *key : {"EZLOCALAI_DOTENV_PATH", "TEXT_SERVER", "VOICE_SERVER", "FALLBACK_SERVER", "FALLBACK_API_KEY", "IMAGE_SERVER"}) setenv(key, "", 1);
    setenv("DEFAULT_MODEL", "gzip-fixture", 1); setenv("EZLOCALAI_API_KEY", "gzip-worker-key", 1);
    setenv("MODELS_PATH", "./models", 1); setenv("USAGE_FILE", "./usage.json", 1);
    setenv("ROUTER_DRAIN_FILE", "./router-drains.json", 1); setenv("ROUTER_OUTPUTS_DIR", "./outputs", 1);
    setenv("HF_HUB_OFFLINE", "1", 1); setenv("TRANSFORMERS_OFFLINE", "1", 1);
    setenv("N_PARALLEL", "1", 1); setenv("GPU_LAYERS", "0", 1); setenv("LLM_SPECULATIVE_TYPE", "none", 1);
    setenv("ROUTER_API_KEY", "gzip-router-key", 1); setenv("WORKER_LABEL", "gzip-fixture", 1);
    setenv("ROUTER_CLIENT_API_KEY", "gzip-client-key", 1);
    setenv("WORKER_HEARTBEAT_INTERVAL", "2", 1); setenv("WORKER_HEARTBEAT_TIMEOUT", "1", 1);
    setenv("ROUTER_URL", ("http://127.0.0.1:" + std::to_string(router.port)).c_str(), 1);
}
struct Workspace {
    std::filesystem::path previous = std::filesystem::current_path(), root;
    Workspace() {
        char pattern[] = "/tmp/ezlocalai-request-gzip-XXXXXX";
        const auto path = mkdtemp(pattern); check(path != nullptr, "owned temporary workspace");
        root = path; std::filesystem::current_path(root);
    }
    ~Workspace() { std::filesystem::current_path(previous); std::filesystem::remove_all(root); }
};
}

int main() {
    try {
        Workspace workspace;
        CaptureRouter router;
        environment(router);
        auto recorded = std::make_shared<Recorded>();
        ezl_env_config config{}; ezl_load_config(&config);
        config.request_timeout_ms = 20000; config.max_concurrent_requests = 1; config.max_queue_size = 8;
        ezlocalai::NativeServer server(config, std::make_unique<Backend>(recorded), resources());
        const int port = fixture_ports(); std::string failure;
        check(server.start("127.0.0.1", port, &failure), failure);
        // Advertisement defaults off; explicit gzip clients and requests
        // already in flight during policy revocation still decode normally.
        router.check_advertisements();
        const std::vector<std::pair<std::string, std::string>> bodies{
            {"/v1/chat/completions", R"({"model":"gzip-fixture","messages":[{"role":"user","content":"é \\ quoted \" text"}],"stream":false,"temperature":0,"prompt_cache_key":"separate-history","disable_fallback":true,"metadata":{"unchanged":true}})"},
            {"/v1/completions", R"({"model":"gzip-fixture","prompt":"é \\ quoted \" text","stream":false,"temperature":0,"prompt_cache_key":"separate-history","disable_fallback":true,"metadata":{"unchanged":true}})"},
            {"/v1/chat/completions?gzip_stream=1", R"({"model":"gzip-fixture","messages":[{"content":"stream fixture"}],"stream":true,"prompt_cache_key":"separate-history","disable_fallback":true,"metadata":{"unchanged":true}})"}
        };
        for (const auto &[path, body] : bodies) {
            const auto before = recorded->size();
            check(request(port, path, body).status == 200, "plain request failed");
            check(recorded->size() == before + 1, "plain request inferred more than once");
            const auto plain = recorded->last();
            const auto packed = gzip(body);
            check(packed.find('\0') != std::string::npos, "fixture must exercise binary gzip bytes");
            check(request(port, path, packed, " GZip\t").status == 200, "gzip request failed");
            check(recorded->size() == before + 2 && recorded->last() == plain, "decompressed request changed backend bytes or inference count");
            const auto decoded = Json::parse(plain.second);
            check(decoded.at("prompt_cache_key") == "separate-history" && decoded.at("metadata").at("unchanged") == true,
                "cache hint or metadata changed during compression");
            for (const auto &coding : {"identity", "br", "gzip, identity"}) {
                const auto attempts = recorded->size();
                check(request(port, path, body, coding).status == 200 && recorded->size() == attempts + 1 && recorded->last() == plain,
                    "unsupported coding changed the legacy raw-body path");
            }
            const auto before_chunked = recorded->size();
            const auto chunked_response = request(port, path, packed, "gzip", "Bearer gzip-worker-key", std::string::npos, "POST", true);
            check(chunked_response.status == 200 && recorded->size() == before_chunked + 1 && recorded->last() == plain,
                "chunked HTTP framing changed gzip payload or inference count");
            if (decoded.at("stream") == true)
                check(chunked_response.body.find("data: [DONE]") != std::string::npos && chunked_response.body.find("fixture") != std::string::npos,
                    "gzip changed streamed chat content or normal completion");
        }
        const auto body = bodies.front().second, packed = gzip(body);
        const auto rejected = [&](const std::string &bytes, int status, const std::string &detail,
                                  const std::string &authorization = "Bearer gzip-worker-key", size_t declared = std::string::npos) {
            const auto before = recorded->size();
            const auto response = request(port, bodies.front().first, bytes, "gzip", authorization, declared);
            check(response.status == status, "gzip rejection status changed");
            if (!detail.empty()) {
                check(Json::parse(response.body) == Json{{"detail", detail}}, "gzip rejection detail changed");
                check(response.headers.find("Content-Type: application/json\r\n") != std::string::npos,
                    "gzip rejection content type changed");
            }
            check(recorded->size() == before, "rejected request reached inference or retried");
        };
        for (const auto &bad : std::vector<std::string>{"", "not gzip", packed.substr(0, packed.size() - 1), packed + "trailing", packed + packed})
            rejected(bad, 400, "Malformed gzip request body");
        auto crc_bad = packed; crc_bad[crc_bad.size() - 8] ^= 1;
        rejected(crc_bad, 400, "Malformed gzip request body");
        rejected("not gzip", 401, "Invalid API Key", "Bearer wrong");
        rejected("not gzip", 401, "Invalid API Key", "");
        rejected(packed, 413, "Gzip request body exceeds 16 MiB", "Bearer gzip-worker-key", limit + 1);
        rejected(packed, 401, "Invalid API Key", "Bearer wrong", limit + 1);
        rejected(gzip("not JSON"), 422, "");
        rejected(gzip(""), 422, "");
        const auto completion_before = recorded->size();
        check(request(port, "/v1/completions", packed + packed, "gzip").status == 400
            && request(port, "/v1/completions", "not gzip", "gzip", "Bearer wrong").status == 401
            && request(port, "/v1/completions", packed, "gzip", "Bearer gzip-worker-key", limit + 1).status == 413
            && recorded->size() == completion_before, "completion gzip errors reached inference or changed authentication priority");
        const auto nontext = R"({"input":"fixture"})";
        check(request(port, "/v1/embeddings", nontext).status == 404 && request(port, "/v1/embeddings", nontext, "gzip").status == 404,
            "non-text raw requests changed behavior");
        check(request(port, "/v1/embeddings", gzip(nontext), "gzip").status == 422,
            "non-text requests unexpectedly decoded gzip");
        const auto before_method = recorded->size();
        check(request(port, bodies.front().first, "not gzip", "gzip", "Bearer wrong", std::string::npos, "GET").status == 405,
            "method validation moved after gzip/authentication");
        check(recorded->size() == before_method, "invalid method reached inference");
        const std::string prefix = R"({"model":"gzip-fixture","messages":[{"content":"boundary"}],"disable_fallback":true,"padding":")";
        const std::string suffix = R"("})";
        auto at_limit = prefix + std::string(limit - prefix.size() - suffix.size(), 'x') + suffix;
        check(at_limit.size() == limit, "decoded boundary fixture size");
        const auto before_limit = recorded->size();
        check(request(port, bodies.front().first, gzip(at_limit), "gzip").status == 200,
            "exact decoded 16 MiB body rejected");
        check(recorded->size() == before_limit + 1, "decoded boundary inference count");
        check(Json::parse(recorded->last().second).at("padding").get<std::string>().size() == limit - prefix.size() - suffix.size(),
            "decoded boundary body was truncated");
        at_limit.insert(at_limit.size() - suffix.size(), 1, 'x');
        rejected(gzip(at_limit), 413, "Gzip request body exceeds 16 MiB");
        check(request(port, bodies.front().first, body).status == 200, "plain recovery request failed");
        const auto plain_recovery = recorded->last();
        const auto before_direct = recorded->size();
        const auto direct = server.handle_request("POST", bodies.front().first, "Bearer gzip-worker-key", packed,
            "application/json", "127.0.0.1", "localhost",
            {{"CoNtEnT-EnCoDiNg", "GZip"}, {"cOnTeNt-LeNgTh", std::to_string(packed.size())}});
        check(direct.status == 200 && recorded->size() == before_direct + 1 && recorded->last() == plain_recovery,
            "direct handler callers lost case-insensitive gzip header behavior");
        const auto status = request(port, "/v1/queue/status", "", "", "Bearer gzip-worker-key", std::string::npos, "GET");
        check(status.status == 200, "queue status failed after gzip rejections");
        const auto queue = Json::parse(status.body);
        check(queue.at("active_requests") == 0 && queue.at("total_processed") == recorded->size() && queue.at("total_failed") == 0,
            "gzip failures leaked queue admissions or created inference retries");
        // Public requests may use the existing router fallback. Decode before
        // that handoff, preserving the one-hop guard and rebuilding plain HTTP
        // framing rather than forwarding the public client's compressed headers.
        router.alive_workers = 1;
        recorded->fail = true;
        for (size_t item = 0; item < 2; ++item) {
            auto public_body = Json::parse(bodies[item].second);
            public_body.erase("disable_fallback");
            const auto original = public_body.dump();
            const auto forwarded_before = router.forwarded().size();
            std::string plain_forward;
            for (bool compressed : {false, true}) {
                const auto attempts = recorded->size();
                const auto response = request(port, bodies[item].first, compressed ? gzip(original) : original,
                    compressed ? "gzip" : "");
                check(response.status == 200 && response.body.find("from router") != std::string::npos,
                    "public request failed to use the existing router fallback");
                check(recorded->size() == attempts + 1, "fallback retried local inference");
                const auto forwarded = router.forwarded();
                check(forwarded.size() == forwarded_before + (compressed ? 2 : 1), "fallback did not make exactly one router POST");
                const auto &wire = forwarded.back();
                check(wire.rfind("POST " + bodies[item].first + " HTTP/1.1\r\n", 0) == 0, "fallback changed endpoint");
                const auto forwarded_body = wire.substr(wire.find("\r\n\r\n") + 4);
                const auto fields = Json::parse(forwarded_body);
                check(fields.at("disable_fallback") == true && fields.at("model") == public_body.at("model")
                    && fields.at("prompt_cache_key") == public_body.at("prompt_cache_key")
                    && fields.at("metadata") == public_body.at("metadata"), "decoded fallback changed hints or the one-hop guard");
                const auto content = item ? "prompt" : "messages";
                check(fields.at(content) == public_body.at(content), "decoded fallback changed public request content");
                if (compressed) check(forwarded_body == plain_forward, "gzip fallback changed the forwarded JSON bytes");
                else plain_forward = forwarded_body;
            }
            public_body["disable_fallback"] = true;
            const auto attempts = recorded->size(), forwards = router.forwarded().size();
            const auto blocked = request(port, bodies[item].first, gzip(public_body.dump()), "gzip");
            check(blocked.status == 503 && recorded->size() == attempts + 1 && router.forwarded().size() == forwards,
                "gzip disable_fallback:true did not block the router handoff");
        }
        recorded->fail = false;
        check(request(port, bodies.front().first, body).status == 200, "worker did not recover after fallback failures");
        const auto after_fallback = request(port, "/v1/queue/status", "", "", "Bearer gzip-worker-key", std::string::npos, "GET");
        check(after_fallback.status == 200 && Json::parse(after_fallback.body).at("active_requests") == 0,
            "gzip router handoff leaked a queue admission");
        router.check_advertisements();
        server.stop();
        std::cout << "Request gzip TCP: byte identity, negotiation, strict framing/caps, auth, decoded one-hop fallback and queue cleanup passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
