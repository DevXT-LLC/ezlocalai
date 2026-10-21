#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <nlohmann/json.hpp>
// Text requests that cannot be served locally go to ROUTER_URL when
// FALLBACK_SERVER is unset, with one router hop at most.
namespace {
using Json = nlohmann::json;
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
struct Seen { std::string method, path, auth; Json body; };
class FakeRouter {
public:
    std::string url;
    int alive = 1, chat_status = 200;
    std::string chat_body = R"({"id":"routed","object":"chat.completion","model":"other-worker","choices":[{"index":0,"message":{"role":"assistant","content":"from router"},"finish_reason":"stop"}]})";
    std::vector<std::string> stream_events = {
        R"({"choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}],"model":"other-worker","object":"chat.completion.chunk"})",
        R"({"choices":[{"index":0,"delta":{"content":"routed"},"finish_reason":null}],"model":"other-worker","object":"chat.completion.chunk"})",
        R"({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"model":"other-worker","object":"chat.completion.chunk"})"};
    FakeRouter() {
        fd_ = socket(AF_INET, SOCK_STREAM, 0); int on = 1; setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(!bind(fd_, reinterpret_cast<sockaddr *>(&a), sizeof(a)), "router bind");
        socklen_t size = sizeof(a); getsockname(fd_, reinterpret_cast<sockaddr *>(&a), &size);
        url = "http://127.0.0.1:" + std::to_string(ntohs(a.sin_port)); expect(!listen(fd_, 8), "router listen");
        thread_ = std::thread([this] { while (!stopped_) { int peer = accept(fd_, nullptr, nullptr); if (peer < 0) continue; serve(peer); shutdown(peer, SHUT_RDWR); close(peer); } });
    }
    ~FakeRouter() { stopped_ = true; shutdown(fd_, SHUT_RDWR); thread_.join(); close(fd_); }
    std::vector<Seen> seen() { std::lock_guard<std::mutex> lock(mutex_); return seen_; }
private:
    int fd_; std::atomic<bool> stopped_{false}; std::thread thread_; std::mutex mutex_; std::vector<Seen> seen_;
    static void send_all(int peer, const std::string &data) { for (size_t at = 0; at < data.size();) { const auto n = send(peer, data.data() + at, data.size() - at, MSG_NOSIGNAL); if (n <= 0) return; at += n; } }
    void serve(int peer) {
        timeval timeout{5, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        std::string request; char buffer[8192]; size_t end = std::string::npos, length = 0; Seen item;
        for (;;) {
            const auto n = recv(peer, buffer, sizeof(buffer), 0); if (n <= 0) return; request.append(buffer, n);
            if (end == std::string::npos) {
                end = request.find("\r\n\r\n"); if (end == std::string::npos) continue;
                std::istringstream lines(request.substr(0, end)); std::string line; std::getline(lines, line);
                std::istringstream first(line); first >> item.method >> item.path;
                while (std::getline(lines, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    const auto colon = line.find(':'); if (colon == std::string::npos) continue;
                    auto key = line.substr(0, colon); for (auto &c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    auto value = line.substr(colon + 1); while (!value.empty() && value.front() == ' ') value.erase(value.begin());
                    if (key == "content-length") length = std::stoul(value);
                    if (key == "authorization") item.auth = value;
                }
            }
            if (request.size() >= end + 4 + length) break;
        }
        const auto body = request.substr(end + 4, length);
        item.body = body.empty() ? Json() : Json::parse(body, nullptr, false);
        { std::lock_guard<std::mutex> lock(mutex_); seen_.push_back(item); }
        if (item.path == "/v1/router/health") {
            const auto payload = Json{{"status", "healthy"}, {"alive_workers", alive}}.dump();
            send_all(peer, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(payload.size()) + "\r\nConnection: close\r\n\r\n" + payload);
        } else if (item.body.is_object() && item.body.value("stream", false)) {
            send_all(peer, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
            const auto chunk = [&](const std::string &data) { std::ostringstream size; size << std::hex << data.size(); send_all(peer, size.str() + "\r\n" + data + "\r\n"); };
            chunk(": keepalive\n\n");
            for (const auto &event : stream_events) chunk("data: " + event + "\n\n");
            chunk("data: [DONE]\n\n"); send_all(peer, "0\r\n\r\n");
        } else {
            send_all(peer, "HTTP/1.1 " + std::to_string(chat_status) + " Fixture\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(chat_body.size()) + "\r\nConnection: close\r\n\r\n" + chat_body);
        }
    }
};
class Backend final : public ezlocalai::InferenceBackend {
public:
    unsigned calls = 0;
    std::string chat(const std::string &) override { return "{}"; }
    std::string completion(const std::string &) override { return "{}"; }
    std::string embedding(const std::string &) override { return "{}"; }
    ezlocalai::InferenceResult text_result(const std::string &, const std::string &, bool) override {
        ++calls; return {R"({"error":{"message":"local inference failed"}})", {"Exception", "local inference failed"}};
    }
};
void environment(const std::string &router) {
    for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("EZLOCALAI_API_KEY", "worker-key", 1); setenv("DEFAULT_MODEL", "default-model", 1);
    setenv("REQUEST_TIMEOUT", "5", 1); setenv("ROUTER_URL", (router + "/").c_str(), 1); setenv("ROUTER_CLIENT_API_KEY", "router-key", 1);
}
struct Outcome { ezlocalai::HttpResponse response; unsigned calls; };
Outcome request(const std::string &body) {
    auto backend = std::make_unique<Backend>(); auto *local = backend.get();
    ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 10000;
    ezlocalai::NativeServer server(config, std::move(backend));
    auto response = server.handle_request("POST", "/v1/chat/completions", "Bearer worker-key", body, "application/json");
    return {std::move(response), local->calls};
}
const std::string chat = R"({"model":"default-model","messages":[{"role":"user","content":"hi"}]})";
}

int main() { try {
    {
        FakeRouter router; environment(router.url);
        const auto outcome = request(chat);
        expect(outcome.response.status == 200 && Json::parse(outcome.response.body) == Json::parse(router.chat_body), "failed inference returns the router response: " + std::to_string(outcome.response.status) + " " + outcome.response.body);
        expect(outcome.calls == 1, "one local attempt before router fallback");
        const auto seen = router.seen();
        expect(seen.size() == 2 && seen[0].method == "GET" && seen[0].path == "/v1/router/health" && seen[1].method == "POST" && seen[1].path == "/v1/chat/completions", "health probe then forward");
        expect(seen[0].auth == "Bearer router-key" && seen[1].auth == "Bearer router-key", "router client key");
        expect(seen[1].body.value("disable_fallback", false) && seen[1].body["messages"] == Json::parse(chat)["messages"] && seen[1].body["model"] == "default-model", "forwarded request disables a second hop: " + seen[1].body.dump());
    }
    {
        FakeRouter router; environment(router.url);
        const auto outcome = request(R"({"model":"default-model","messages":[{"role":"user","content":"hi"}],"disable_fallback":true})");
        expect(outcome.response.status == 503 && router.seen().empty(), "disable_fallback requests are not forwarded: " + outcome.response.body);
    }
    {
        FakeRouter router; router.alive = 0; environment(router.url);
        const auto outcome = request(chat);
        const auto seen = router.seen();
        expect(outcome.response.status == 503 && seen.size() == 1 && seen[0].path == "/v1/router/health", "router without live workers is not used: " + outcome.response.body);
    }
    {
        FakeRouter router; router.chat_status = 413;
        router.chat_body = R"({"detail":{"type":"context_capacity_error","message":"too long","n_prompt_tokens":9,"n_ctx":4,"compact_and_retry":true}})";
        environment(router.url);
        const auto outcome = request(chat);
        expect(outcome.response.status == 413 && Json::parse(outcome.response.body) == Json::parse(router.chat_body), "router error status passes through: " + outcome.response.body);
    }
    {
        FakeRouter router; environment(router.url); setenv("FALLBACK_MEMORY_THRESHOLD", "1000000", 1);
        const auto outcome = request(R"({"model":"default-model","stream":true,"messages":[{"role":"user","content":"hi"}]})");
        std::string expected;
        for (const auto &event : router.stream_events) expected += "data: " + Json::parse(event).dump() + "\n\n";
        std::string actual;
        std::istringstream lines(outcome.response.body);
        for (std::string line; std::getline(lines, line);) if (line.rfind("data: {", 0) == 0) actual += "data: " + Json::parse(line.substr(6)).dump() + "\n\n";
        expect(outcome.response.status == 200 && outcome.calls == 0, "pre-check streams from the router without local inference: " + outcome.response.body);
        expect(actual == expected && outcome.response.body.find("data: [DONE]") != std::string::npos, "router stream chunks are relayed unchanged: " + outcome.response.body);
        const auto seen = router.seen();
        expect(seen.size() == 2 && seen[1].body.value("disable_fallback", false) && seen[1].body.value("stream", false), "streamed forward disables a second hop");
    }
    {
        FakeRouter router; environment(router.url); setenv("FALLBACK_SERVER", "http://127.0.0.1:9", 1);
        request(chat);
        expect(router.seen().empty(), "FALLBACK_SERVER takes precedence over the router");
    }
    std::cout << "router fallback cases passed\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
