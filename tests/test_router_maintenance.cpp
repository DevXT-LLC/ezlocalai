#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"

#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <future>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <openssl/sha.h>

namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    expect(bool(input), "could not read " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
std::vector<std::string> split(const std::string &value, char separator) {
    std::vector<std::string> parts;
    size_t begin = 0;
    do {
        const auto end = value.find(separator, begin);
        parts.push_back(value.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    } while (true);
    return parts;
}
std::string unhex(const std::string &value) {
    expect(value.size() % 2 == 0, "invalid hex fixture");
    std::string result;
    for (size_t i = 0; i < value.size(); i += 2) result += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16));
    return result;
}
std::string sha256(const std::string &value) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    std::ostringstream result;
    for (const auto byte : digest) result << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return result.str();
}

// Keep JSON field order, strings and types exact. Normalize numeric spelling and
// the four clock-dependent fields; lease duration/persistence are checked below.
std::string stable_json(const std::string &input) {
    if (input == "Internal Server Error") return input;
    std::string result, key;
    for (size_t i = 0; i < input.size();) {
        const char ch = input[i];
        if (std::isspace(static_cast<unsigned char>(ch))) { ++i; continue; }
        if (ch == '"') {
            const auto begin = i++;
            while (i < input.size() && input[i] != '"') {
                if (input[i] == '\\') ++i;
                ++i;
            }
            expect(i < input.size(), "unterminated JSON string");
            ++i;
            const auto value = input.substr(begin, i - begin);
            result += value;
            auto next = input.find_first_not_of(" \t\r\n", i);
            key = next != std::string::npos && input[next] == ':' ? value : "";
        } else if (ch == '-' || (ch >= '0' && ch <= '9')) {
            const auto begin = i++;
            while (i < input.size() && std::string("0123456789.eE+-").find(input[i]) != std::string::npos) ++i;
            if (key == "\"age_seconds\"" || key == "\"last_heartbeat_age\"" || key == "\"expires_at\"" || key == "\"drain_until\"") result += '0';
            else {
                std::ostringstream number;
                number << std::setprecision(17) << std::stod(input.substr(begin, i - begin));
                result += number.str();
            }
            key.clear();
        } else { result += ch; ++i; }
    }
    return result;
}

std::unique_ptr<ezlocalai::NativeServer> make_server() {
    ezl_env_config config{};
    ezl_load_config(&config);
    std::snprintf(config.api_key, sizeof(config.api_key), "%s", "test-secret");
    return std::make_unique<ezlocalai::NativeServer>(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
}
ezlocalai::HttpResponse call(ezlocalai::NativeServer &server, const std::string &path, const std::string &body, const std::string &method = "POST") {
    return server.handle_request(method, path, "Bearer test-secret", body, "application/json", "testclient");
}

const std::string registration = R"({"worker_id":"w1","label":"alpha","url":"http://worker:8091","capabilities":["text"],"models":["model"]})";
const std::string drain_body = R"({"label":"alpha","owner":"rollout"})";
const std::string idle_drain_body = R"({"label":"alpha","owner":"rollout","require_idle":true})";
void status(const ezlocalai::HttpResponse &response, int expected, const std::string &message) {
    expect(response.status == expected, message + ": " + std::to_string(response.status) + " " + response.body);
}
void public_contains(ezlocalai::NativeServer &server, const std::string &field) {
    const auto response = call(server, "/v1/router/workers", "", "GET");
    status(response, 200, "list workers");
    expect(response.body.find(field) != std::string::npos, "missing public field " + field + " in " + response.body);
}
void write_file(const std::filesystem::path &path, const std::string &body) {
    std::ofstream file(path, std::ios::binary);
    file << body;
    expect(bool(file), "write " + path.string());
}

void persistence_cases(const std::filesystem::path &directory) {
    const auto path = directory / "nested" / "drains.json";
    setenv("ROUTER_DRAIN_FILE", path.c_str(), 1);
    auto server = make_server();
    status(call(*server, "/v1/router/register", registration), 200, "register persistent worker");
    const auto before = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    status(call(*server, "/v1/router/drain", drain_body), 200, "persist drain");
    const auto data = read_file(path);
    const auto until_key = data.find("\"until\":");
    expect(until_key != std::string::npos, "persisted expiration");
    const auto until = std::stod(data.substr(until_key + 8));
    expect(until >= before + 120 && until < before + 122, "default lease duration is 120 seconds");
    server = make_server();
    status(call(*server, "/v1/router/register", R"({"worker_id":"new-process","label":"alpha","url":"http://worker:8091"})"), 200, "register after router and worker restart");
    public_contains(*server, "\"draining\":true");
    status(call(*server, "/v1/router/resume", R"({"label":"alpha","owner":"other"})"), 409, "owner persists across restart");
    status(call(*server, "/v1/router/resume", drain_body), 200, "persist resume");
    expect(read_file(path) == "{}", "resume removes disk lease");
    server = make_server();
    status(call(*server, "/v1/router/register", registration), 200, "register after persisted resume");
    public_contains(*server, "\"draining\":false");

    write_file(path, R"({"alpha":{"owner":"expired-owner","until":1}})");
    server = make_server();
    status(call(*server, "/v1/router/register", registration), 200, "register expired lease");
    public_contains(*server, "\"draining\":false");
    public_contains(*server, "\"drain_until\":null");
    status(call(*server, "/v1/router/resume", drain_body), 409, "expired lease still requires its owner to resume");
    status(call(*server, "/v1/router/drain", idle_drain_body), 200, "new drain replaces expired lease");
    expect(read_file(path).find("expired-owner") == std::string::npos, "expired lease pruned on write");

    write_file(path, "[]");
    server = make_server();
    status(call(*server, "/v1/router/workers", "", "GET"), 500, "invalid persisted root rejected");
    write_file(path, "{}");
    status(call(*server, "/v1/router/register", registration), 200, "registry retries after invalid file repaired");
    std::filesystem::remove(path);
    std::filesystem::create_directory(path);
    status(call(*server, "/v1/router/drain", drain_body), 500, "failed atomic save");
    public_contains(*server, "\"draining\":false");
    expect(std::distance(std::filesystem::directory_iterator(path.parent_path()), std::filesystem::directory_iterator()) == 1, "failed save cleans temporary file");
    std::filesystem::remove(path);
    status(call(*server, "/v1/router/drain", drain_body), 200, "drain after write failure");
    std::filesystem::remove(path);
    std::filesystem::create_directory(path);
    status(call(*server, "/v1/router/resume", drain_body), 500, "failed resume save");
    public_contains(*server, "\"draining\":true");
    std::filesystem::remove(path);
    status(call(*server, "/v1/router/resume", drain_body), 200, "resume after write failure");
    setenv("ROUTER_DRAIN_FILE", "", 1);
}

void authorization_and_contention_cases() {
    auto server = make_server();
    for (const auto *path : {"/v1/router/drain", "/v1/router/resume"}) {
        status(server->handle_request("POST", path, "", drain_body, "application/json"), 401, "maintenance requires worker auth");
        status(server->handle_request("POST", path, "Bearer wrong", drain_body, "application/json"), 401, "maintenance rejects wrong key");
        status(call(*server, path, "", "GET"), 405, "maintenance requires POST");
    }
    status(call(*server, "/v1/router/register", registration), 200, "register contention worker");
    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::vector<std::future<ezlocalai::HttpResponse>> responses;
    for (int i = 0; i < 16; ++i) responses.push_back(std::async(std::launch::async, [&, i] {
        ready.wait();
        return call(*server, "/v1/router/drain", "{\"label\":\"alpha\",\"owner\":\"owner-" + std::to_string(i) + "\",\"require_idle\":true}");
    }));
    start.set_value();
    int winners = 0;
    for (auto &future : responses) {
        const auto response = future.get();
        if (response.status == 200) ++winners;
        else status(response, 409, "contending rollout must conflict");
    }
    expect(winners == 1, "exactly one concurrent rollout acquires lease");
}

// One controllable HTTP request, using an ephemeral port and bounded waits.
// Headers arrive immediately; the response body stays blocked until finish().
class HeldUpstream {
public:
    explicit HeldUpstream(bool streaming) {
        listener_ = socket(AF_INET, SOCK_STREAM, 0);
        expect(listener_ >= 0, "upstream socket");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(bind(listener_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "bind upstream");
        socklen_t size = sizeof(address);
        expect(getsockname(listener_, reinterpret_cast<sockaddr *>(&address), &size) == 0, "upstream port");
        port = ntohs(address.sin_port);
        expect(listen(listener_, 1) == 0, "listen upstream");
        thread_ = std::thread([this, streaming] {
            pollfd descriptor{listener_, POLLIN, 0};
            if (poll(&descriptor, 1, 5000) <= 0) return;
            const int peer = accept(listener_, nullptr, nullptr);
            if (peer < 0) return;
            timeval timeout{5, 0};
            setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            std::string request;
            char buffer[4096];
            size_t expected = std::string::npos;
            while (expected == std::string::npos || request.size() < expected) {
                const auto n = recv(peer, buffer, sizeof(buffer), 0);
                if (n <= 0) { close(peer); return; }
                request.append(buffer, static_cast<size_t>(n));
                const auto headers = request.find("\r\n\r\n");
                const auto length = request.find("Content-Length: ");
                if (headers != std::string::npos) expected = headers + 4 + (length == std::string::npos ? 0 : std::stoul(request.substr(length + 16)));
            }
            const std::string body = streaming ? "data: [DONE]\n\n" : "{}";
            const auto headers = "HTTP/1.1 200 OK\r\nContent-Type: " + std::string(streaming ? "text/event-stream" : "application/json") + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
            send(peer, headers.data(), headers.size(), MSG_NOSIGNAL);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                entered_ = true;
                changed_.notify_all();
                changed_.wait_for(lock, std::chrono::seconds(5), [&] { return finished_; });
            }
            send(peer, body.data(), body.size(), MSG_NOSIGNAL);
            shutdown(peer, SHUT_RDWR);
            close(peer);
        });
    }
    ~HeldUpstream() { finish(); if (thread_.joinable()) thread_.join(); close(listener_); }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex_);
        expect(changed_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; }), "request reached upstream");
    }
    void finish() { std::lock_guard<std::mutex> lock(mutex_); finished_ = true; changed_.notify_all(); }
    int port = 0;
private:
    int listener_ = -1;
    bool entered_ = false, finished_ = false;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread thread_;
};

void active_request_cases() {
    setenv("ROUTER_RESERVATION_TTL", "0.25", 1);
    setenv("ROUTER_WAIT_TIMEOUT", "0", 1);
    setenv("ROUTER_MAX_RETRIES", "0", 1);
    for (const int mode : {0, 1, 2, 3, 4}) {
        const bool speech = mode >= 3;
        const bool streaming = mode == 1 || mode == 2 || mode == 4;
        const std::string capability = speech ? "tts" : "text";
        const std::string path = speech ? streaming ? "/v1/audio/speech/stream" : "/v1/audio/speech" : "/v1/chat/completions";
        auto server = make_server();
        HeldUpstream upstream(streaming);
        const auto register_body = "{\"worker_id\":\"w1\",\"label\":\"alpha\",\"url\":\"http://127.0.0.1\",\"port\":" + std::to_string(upstream.port) + ",\"capabilities\":[\"" + capability + "\"],\"models\":[\"model\"],\"cap_slots\":{\"" + capability + "\":{\"capacity\":1}},\"model_slots\":{\"model\":{\"capacity\":1}}}";
        auto register_worker = [&] { return server->handle_request("POST", "/v1/router/register", "Bearer test-secret", register_body, "application/json", "127.0.0.1"); };
        status(register_worker(), 200, "register held upstream");
        const auto request = speech ? R"({"model":"model","input":"hello","voice":"alloy"})" : "{\"model\":\"model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":" + std::string(streaming ? "true" : "false") + "}";
        auto result = std::async(std::launch::async, [&] { return server->handle_request("POST", path, "Bearer test-secret", request, "application/json", "testclient", "", {}, true); });
        upstream.wait_entered();
        public_contains(*server, "\"active_requests\":1");
        public_contains(*server, "\"reported_in_flight\":0");
        public_contains(*server, "\"router_in_flight\":1");
        public_contains(*server, "\"router_cap_in_flight\":{\"" + capability + "\":1}");
        public_contains(*server, "\"router_model_in_flight\":{\"model\":1}");
        status(call(*server, "/v1/router/heartbeat", R"({"worker_id":"w1","in_flight":0,"queue_depth":0})"), 200, "heartbeat during active request");
        status(register_worker(), 200, "reregister preserves request token");
        std::this_thread::sleep_for(std::chrono::milliseconds(350));
        public_contains(*server, speech ? "\"router_in_flight\":1" : "\"router_in_flight\":0");
        public_contains(*server, "\"active_requests\":1");
        public_contains(*server, "\"maintenance_busy\":1");
        status(call(*server, "/v1/router/drain", idle_drain_body), 409, "expired capacity lease does not imply maintenance idle");
        const auto drain = call(*server, "/v1/router/drain", drain_body);
        status(drain, 200, "drain active worker");
        expect(drain.body.find("\"busy\":1") != std::string::npos, "drain reports active request");
        status(call(*server, path, request), 503, "draining worker rejects new dispatch");
        upstream.finish();
        auto response = result.get();
        status(response, 200, "held response succeeds");
        if (mode == 1 || mode == 4) {
            expect(bool(response.stream_writer), "stream remains deferred");
            public_contains(*server, "\"active_requests\":1");
            int sockets[2];
            expect(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "stream client socketpair");
            response.stream_writer(sockets[0]);
            close(sockets[0]);
            char buffer[1024];
            const auto n = recv(sockets[1], buffer, sizeof(buffer), 0);
            close(sockets[1]);
            expect(n > 0 && std::string(buffer, static_cast<size_t>(n)).find("[DONE]") != std::string::npos, "stream body delivered");
        } else if (mode == 2) {
            expect(bool(response.stream_writer), "abandoned response has stream writer");
            public_contains(*server, "\"active_requests\":1");
            response = {};
        }
        public_contains(*server, "\"active_requests\":0");
        status(call(*server, "/v1/router/drain", idle_drain_body), 200, "idle lease succeeds after request cleanup");
    }
}

size_t reference_cases(const char *root, const char *fixture) {
    size_t count = 0;
    std::unique_ptr<ezlocalai::NativeServer> server;
    std::string scenario;
    std::istringstream lines(read_file(fixture));
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') {
            const auto marker = line.find(" sha256=");
            if (marker != std::string::npos) {
                const auto path = std::string(root) + "/" + line.substr(2, marker - 2);
                expect(sha256(read_file(path)) == line.substr(marker + 8), "reference changed; refresh router maintenance oracle: " + path);
            }
            continue;
        }
        const auto columns = split(line, '\t');
        expect(columns.size() == 7, "malformed maintenance fixture");
        if (scenario != columns[0]) {
            scenario = columns[0];
            server = make_server();
        }
        const auto response = call(*server, columns[3], unhex(columns[4]), columns[2]);
        const auto context = columns[0] + "/" + columns[1];
        expect(response.status == std::stoi(columns[5]), context + " status: " + std::to_string(response.status) + " expected " + columns[5] + " body: " + response.body);
        expect(stable_json(response.body) == stable_json(unhex(columns[6])), context + " response: " + response.body + " expected " + unhex(columns[6]));
        ++count;
    }
    expect(count == 219, "all maintenance reference cases executed");
    return count;
}
} // namespace

int main(int argc, char **argv) {
    try {
        expect(argc == 3, "reference checkout and maintenance fixture required");
        for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
        setenv("EZLOCALAI_DOTENV_PATH", "", 1);
        setenv("ROUTER_MODE", "true", 1);
        setenv("ROUTER_DRAIN_FILE", "", 1);
        setenv("ROUTER_RESERVATION_TTL", "15", 1);
        setenv("ROUTER_API_KEY", "test-secret", 1);
        setenv("ROUTER_WORKER_TTL", "30", 1);
        const auto count = reference_cases(argv[1], argv[2]);
        char directory[] = "/tmp/ezlocalai-maintenance-XXXXXX";
        expect(mkdtemp(directory) != nullptr, "maintenance temporary directory");
        const auto previous = std::filesystem::current_path();
        std::filesystem::current_path(directory);
        setenv("USAGE_FILE", "usage.json", 1);
        persistence_cases(directory);
        authorization_and_contention_cases();
        active_request_cases();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(directory);
        std::cout << "Verified " << count << " router maintenance reference cases, persistence rollback, concurrent leases, request and stream lifetime\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
