#include <arpa/inet.h>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
int main(int argc, char **argv) {
    int port = 0, layers = 0, ubatch = 0, context = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--port") port = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "-ngl") layers = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "-ub") ubatch = std::atoi(argv[i + 1]);
        if (std::string(argv[i]) == "-c") context = std::atoi(argv[i + 1]);
    }
    if (!port) return 2;
    const char *root = std::getenv("EMBEDDING_FIXTURE_DIR");
    if (!root) return 3;
    const auto directory = fs::path(root); fs::create_directories(directory);
    const auto suffix = std::to_string(port);
    { std::ofstream out(directory / ("launch-" + suffix), std::ios::app); out << getpid() << '\n'; }
    { std::ofstream out(directory / ("args-" + suffix)); for (int i = 1; i < argc; ++i) out << argv[i] << '\n'; }
    { std::ofstream out(directory / ("attempts-" + suffix), std::ios::app); out << layers << '\n'; }
    { std::ofstream out(directory / ("load-plan-" + suffix), std::ios::app); out << layers << '\t' << ubatch << '\t' << context << '\n'; }
    if (fs::exists(directory / ("fail-launch-" + suffix))) { std::cerr << "fixture model load unavailable\n"; return 4; }
    if (const auto failure = std::getenv("LLM_FIXTURE_FAIL_LAUNCH")) {
        std::ifstream launches(directory / ("launch-" + suffix)); int count = 0;
        for (int pid; launches >> pid;) ++count;
        if (count == std::atoi(failure)) {
            if (const auto replacement = std::getenv("LLM_FIXTURE_CUDA_AFTER_FAILURE"))
                if (const auto devices = std::getenv("EZLOCALAI_TEST_CUDA_DEVICES")) std::ofstream(devices) << replacement;
            std::cerr << "CUDA final reload failure\n"; return 4;
        }
    }
    if (const char *failed = std::getenv("EMBEDDING_FIXTURE_FAIL_PORT")) if (port == std::atoi(failed)) return 4;
    if (const char *limit = std::getenv("EMBEDDING_FIXTURE_MAX_LAYERS")) if (layers < 0 || layers > std::atoi(limit)) {
        const auto message = std::getenv("EMBEDDING_FIXTURE_FAILURE");
        std::cerr << (message ? message : "CUDA out of memory") << '\n'; return 4;
    }
    std::signal(SIGTERM, stop); std::signal(SIGINT, stop);
    if (fs::exists(directory / ("hold-startup-" + suffix))) {
        std::ofstream(directory / ("waiting-startup-" + suffix)) << getpid();
        while (!stopped && fs::exists(directory / ("hold-startup-" + suffix)))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (stopped) return 0;
    }
    if (std::getenv("EMBEDDING_FIXTURE_TIMEOUT")) {
        while (!stopped) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return 0;
    }
    const int listener = socket(AF_INET, SOCK_STREAM, 0); if (listener < 0) return 5;
    int reuse = 1; setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
    if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) || listen(listener, 8)) {
        std::cerr << "fixture listen port " << port << ": " << std::strerror(errno) << '\n'; return 6;
    }
    std::signal(SIGTERM, stop); std::signal(SIGINT, stop);
    while (!stopped) {
        pollfd ready{listener, POLLIN, 0}; if (poll(&ready, 1, 100) <= 0) continue;
        const int peer = accept(listener, nullptr, nullptr); if (peer < 0) continue;
        timeval timeout{2, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        std::string request; char buffer[4096]; size_t expected = std::string::npos;
        while (request.size() < expected) {
            const auto count = recv(peer, buffer, sizeof(buffer), 0); if (count <= 0) break;
            request.append(buffer, count);
            const auto end = request.find("\r\n\r\n");
            if (end != std::string::npos) { const auto length = request.find("Content-Length: "); expected = end + 4 + (length == std::string::npos ? 0 : std::stoul(request.substr(length + 16))); }
        }
        const bool text = request.find("POST /v1/chat/completions ") == 0 || request.find("POST /v1/completions ") == 0;
        const bool health = request.find("POST /v1/embeddings ") != 0 && !text;
        const auto body_start = request.find("\r\n\r\n");
        const bool startup = text && body_start != std::string::npos && request.substr(body_start + 4) == R"({"prompt":"Hi","max_tokens":1})";
        if (startup) {
            std::ofstream(directory / ("probe-" + suffix), std::ios::app) << request << '\n';
            if (std::getenv("LLM_FIXTURE_PROBE_BLOCK")) while (!stopped && !fs::exists(directory / ("release-probe-" + suffix))) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (std::getenv("LLM_FIXTURE_PROBE_DROP")) { close(peer); continue; }
        }
        if (!health && !startup) {
            std::ofstream(directory / ("seen-" + suffix), std::ios::app) << request << '\n';
            if (request.find("block") != std::string::npos) while (!stopped && !fs::exists(directory / ("release-" + suffix))) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (text && request.find("fixture-stream-cancel") != std::string::npos) {
            setsockopt(peer, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            const auto send_all = [&](const std::string &bytes) {
                size_t sent = 0;
                while (sent < bytes.size()) {
                    const auto count = send(peer, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
                    if (count < 0 && errno == EINTR && !stopped) continue;
                    if (count <= 0) return false;
                    sent += static_cast<size_t>(count);
                }
                return true;
            };
            bool disconnected = !send_all("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
            const std::string event = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"fixture-stream-cancel-output\"},\"finish_reason\":null}]}\n\n";
            std::ostringstream chunk; chunk << std::hex << event.size() << "\r\n" << event << "\r\n";
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (!stopped && !disconnected && std::chrono::steady_clock::now() < deadline) {
                disconnected = !send_all(chunk.str());
                char byte;
                const auto count = recv(peer, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
                disconnected |= count == 0 || (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::ofstream(directory / ((disconnected ? "cancelled-stream-" : "stream-timeout-") + suffix)) << "finished";
            close(peer);
            continue;
        }
        const bool error = request.find("fixture-error") != std::string::npos || (!health && std::getenv("EMBEDDING_FIXTURE_WARMUP_ERROR"));
        std::string body = health ? "{}" : error ? R"({"error":{"message":"fixture inference failed","code":400}})"
            : text ? R"({"choices":[{"message":{"content":"fixture"}}]})"
            : "{\"data\":[{\"embedding\":[" + suffix + "]}]}";
        int status = error ? 400 : 200;
        if (text && !startup) {
            if (const auto result = std::getenv("LLM_FIXTURE_INFERENCE_RESULT")) body = result;
            if (const auto path = std::getenv("LLM_FIXTURE_INFERENCE_FILE")) {
                std::ifstream input(path, std::ios::binary);
                body.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
            }
            if (const auto code = std::getenv("LLM_FIXTURE_INFERENCE_STATUS")) status = std::atoi(code);
        }
        if (startup) {
            if (const auto result = std::getenv("LLM_FIXTURE_PROBE_RESULT")) body = result;
            if (const auto code = std::getenv("LLM_FIXTURE_PROBE_STATUS")) status = std::atoi(code);
            if (const auto limit = std::getenv("LLM_FIXTURE_PROBE_MAX_LAYERS")) if (layers < 0 || layers > std::atoi(limit)) body = R"({"error":{"message":"CUDA out of memory during startup inference"}})";
            if (const auto limit = std::getenv("LLM_FIXTURE_MAX_UBATCH")) if (ubatch > std::atoi(limit)) body = R"({"error":{"message":"CUDA compute allocation failed"}})";
        }
        const std::string response = "HTTP/1.1 " + std::to_string(status) + " " + (status == 200 ? "OK" : "Error") + "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        size_t sent = 0;
        while (sent < response.size()) { const auto count = send(peer, response.data() + sent, response.size() - sent, MSG_NOSIGNAL); if (count <= 0) break; sent += count; }
        close(peer);
    }
    close(listener); return 0;
}
