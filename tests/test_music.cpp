#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
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
            if (key == "\"created\"" || key == "\"age_seconds\"" || key == "\"last_heartbeat_age\"" || key == "\"expires_at\"" || key == "\"drain_until\"") result += '0';
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

std::string stable_response(const std::string &body) {
    static const std::regex uuid("outputs/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-");
    return stable_json(std::regex_replace(body, uuid, "outputs/UUID-"));
}
struct Exchange { std::string method, path, request, content_type, body; int status; std::string authorization; };
class ScriptedServer {
public:
    explicit ScriptedServer(std::vector<Exchange> script) : script_(std::move(script)) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        expect(fd_ >= 0, "upstream socket");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "bind upstream");
        socklen_t size = sizeof(address);
        expect(getsockname(fd_, reinterpret_cast<sockaddr *>(&address), &size) == 0, "upstream port");
        url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port));
        expect(listen(fd_, 8) == 0, "listen upstream");
        thread_ = std::thread([this] { serve(); });
    }
    ~ScriptedServer() { finish(); }
    void finish() {
        if (fd_ < 0) return;
        stopped_ = true;
        shutdown(fd_, SHUT_RDWR);
        if (thread_.joinable()) thread_.join();
        close(fd_); fd_ = -1;
    }
    void verify(const std::string &name) {
        finish();
        expect(error_.empty(), name + ": " + error_);
        expect(index_ == script_.size(), name + ": expected " + std::to_string(script_.size()) + " upstream requests, saw " + std::to_string(index_));
    }
    std::string url;
private:
    void serve() {
        try {
            while (!stopped_) {
                pollfd descriptor{fd_, POLLIN, 0};
                if (poll(&descriptor, 1, 100) <= 0) continue;
                const int peer = accept(fd_, nullptr, nullptr);
                if (peer < 0) continue;
                timeval timeout{3, 0};
                setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
                std::string raw;
                char buffer[4096];
                size_t expected = std::string::npos, header_end = std::string::npos;
                while (expected == std::string::npos || raw.size() < expected) {
                    const auto n = recv(peer, buffer, sizeof(buffer), 0);
                    if (n <= 0) break;
                    raw.append(buffer, static_cast<size_t>(n));
                    header_end = raw.find("\r\n\r\n");
                    const auto length = raw.find("Content-Length: ");
                    if (header_end != std::string::npos) expected = header_end + 4 + (length == std::string::npos ? 0 : std::stoul(raw.substr(length + 16)));
                }
                std::istringstream line(raw);
                std::string method, path;
                line >> method >> path;
                Exchange exchange{"", "", "", "text/plain", "unexpected upstream request", 500};
                if (index_ >= script_.size()) error_ = "unexpected " + method + " " + path;
                else {
                    exchange = script_[index_++];
                    const auto body = header_end == std::string::npos ? "" : raw.substr(header_end + 4);
                    if (method != exchange.method || path != exchange.path) error_ = "request " + method + " " + path + " expected " + exchange.method + " " + exchange.path;
                    if (stable_json(body) != stable_json(exchange.request)) error_ = "request body " + body + " expected " + exchange.request;
                    if (!exchange.authorization.empty() && raw.find("Authorization: " + exchange.authorization + "\r\n") == std::string::npos) error_ = "missing upstream authorization " + exchange.authorization;
                }
                const auto response = "HTTP/1.1 " + std::to_string(exchange.status) + " Response\r\nContent-Type: " + exchange.content_type + "\r\nContent-Length: " + std::to_string(exchange.body.size()) + "\r\nConnection: close\r\n\r\n" + exchange.body;
                size_t sent = 0;
                while (sent < response.size()) {
                    const auto n = send(peer, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
                    if (n <= 0) break;
                    sent += static_cast<size_t>(n);
                }
                shutdown(peer, SHUT_RDWR); close(peer);
            }
        } catch (const std::exception &e) { error_ = e.what(); }
    }
    int fd_ = -1;
    std::thread thread_;
    std::atomic<bool> stopped_{false};
    std::vector<Exchange> script_;
    size_t index_ = 0;
    std::string error_;
};

void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const char *key : {"ACE_STEP_SERVER_URL", "ACE_STEP_TIMEOUT", "ACE_STEP_POLL_INTERVAL", "ROUTER_WAIT_TIMEOUT", "ROUTER_MODE"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1);
    setenv("ROUTER_DRAIN_FILE", "", 1);
    setenv("ROUTER_API_KEY", "test-secret", 1);
    setenv("USAGE_FILE", "usage.json", 1);
    setenv("ACE_STEP_AUTO_START", "false", 1);
}
std::unique_ptr<ezlocalai::NativeServer> make_server() {
    ezl_env_config config{};
    ezl_load_config(&config);
    std::snprintf(config.api_key, sizeof(config.api_key), "%s", "test-secret");
    return std::make_unique<ezlocalai::NativeServer>(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
}
size_t reference_cases(const char *root, const char *fixture) {
    size_t count = 0, failures = 0;
    std::istringstream lines(read_file(fixture));
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') {
            const auto marker = line.find(" sha256=");
            if (marker != std::string::npos) {
                const auto path = std::string(root) + "/" + line.substr(2, marker - 2);
                expect(sha256(read_file(path)) == line.substr(marker + 8), "reference changed; refresh music oracle: " + path);
            }
            continue;
        }
        const auto columns = split(line, '\t');
        expect(columns.size() == 7, "malformed music fixture");
        ++count;
        try {
        environment();
        for (const auto &entry : split(columns[1], ';')) {
            const auto equals = entry.find('=');
            setenv(entry.substr(0, equals).c_str(), unhex(entry.substr(equals + 1)).c_str(), 1);
        }
        std::vector<Exchange> script;
        for (const auto &step : split(unhex(columns[6]), '\n')) {
            if (step.empty()) continue;
            const auto fields = split(step, '\t');
            expect(fields.size() == 6, "malformed HTTP transcript");
            script.push_back({fields[0], fields[1], unhex(fields[2]), unhex(fields[4]), unhex(fields[5]), std::stoi(fields[3])});
        }
        const auto last_audio = !script.empty() && script.back().content_type.find("audio/") == 0 ? script.back().body : "";
        ScriptedServer upstream(std::move(script));
        setenv("ACE_STEP_SERVER_URL", upstream.url.c_str(), 1);
        for (const char *key : {"ROUTER_URL", "FALLBACK_SERVER"}) {
            const char *value = getenv(key);
            if (value && std::string(value) == "<UPSTREAM>") setenv(key, upstream.url.c_str(), 1);
        }
        auto server = make_server();
        const auto response = server->handle_request("POST", columns[2], "Bearer test-secret", unhex(columns[3]), "application/json");
        upstream.verify(columns[0]);
        expect(response.status == std::stoi(columns[4]), columns[0] + " status " + std::to_string(response.status) + " expected " + columns[4] + " body " + response.body);
        expect(stable_response(response.body) == stable_response(unhex(columns[5])), columns[0] + " response " + response.body + " expected " + unhex(columns[5]));
        const auto output = response.body.find("outputs/");
        if (!last_audio.empty() && output != std::string::npos) {
            const auto end = response.body.find('"', output);
            const auto filename = response.body.substr(output, end - output);
            expect(read_file(filename) == last_audio, columns[0] + " persisted original audio bytes");
            const auto download = server->handle_request("GET", "/" + filename, "", "");
            expect(download.status == 200 && download.body == last_audio, columns[0] + " generated audio is downloadable");
        }
        if (std::string(getenv("MUSIC_ENABLED")) == "true" && response.status == 200) {
            const auto resources = server->handle_request("GET", "/v1/resources", "Bearer test-secret", "");
            expect(resources.body.find("\"music\":{\"capacity\":1,\"in_flight\":0") != std::string::npos, columns[0] + " releases local music slot");
            expect(resources.body.find("\"device\":\"external\"") != std::string::npos, columns[0] + " retains initialized ACE client metadata");
            const auto queue = server->handle_request("GET", "/v1/queue", "Bearer test-secret", "");
            expect(queue.body.find("\"total_queued\":0") != std::string::npos, columns[0] + " music does not enter LLM queue");
        }
        } catch (const std::exception &error) {
            if (++failures <= 12) std::cerr << error.what() << "\n";
        }
    }
    expect(failures == 0, std::to_string(failures) + " music fixture cases differed");
    expect(count == 568, "all music reference cases executed");
    return count;
}

void router_cases() {
    for (const bool video : {false, true}) for (const bool alias : {false, true}) {
        environment();
        setenv("ROUTER_MODE", "true", 1);
        setenv("ROUTER_WAIT_TIMEOUT", "0", 1);
        const std::string capability = video ? "music_video" : "music";
        const std::string canonical = video ? "/v1/videos/music" : "/v1/audio/music";
        const std::string path = canonical + (alias ? "/generations" : "");
        const std::string request = R"({"model":"audio-hint","video_model":"video-hint","arbitrary_field":{"x":1}})";
        const auto asset_path = "/outputs/" + capability + (alias ? "-alias.wav" : ".wav");
        ScriptedServer upstream({
            {"POST", canonical, request, "application/json", "{\"data\":[{\"url\":\"http://worker" + asset_path + "\"}]}", 200, "Bearer worker-key"},
            {"GET", asset_path, "", "audio/wav", "RIFF audio", 200, "Bearer worker-key"}
        });
        auto server = make_server();
        const auto port = upstream.url.substr(upstream.url.rfind(':') + 1);
        const auto registration = "{\"worker_id\":\"music-worker\",\"label\":\"Music Worker\",\"url\":\"" + upstream.url + "\",\"port\":" + port + ",\"api_key\":\"worker-key\",\"capabilities\":[\"" + capability + "\"],\"models\":[]}";
        auto response = server->handle_request("POST", "/v1/router/register", "Bearer test-secret", registration, "application/json", "127.0.0.1");
        expect(response.status == 200, "register router music worker");
        response = server->handle_request("POST", path, "Bearer test-secret", request, "application/json", "127.0.0.1", "router.example:8091");
        expect(response.status == 200, "router music forwarding: " + response.body);
        expect(response.body == "{\"data\":[{\"url\":\"http://router.example:8091" + asset_path + "\"}]}", "persisted music URL: " + response.body);
        expect(read_file("." + asset_path) == "RIFF audio", "persisted music bytes");
        const auto workers = server->handle_request("GET", "/v1/router/workers", "Bearer test-secret", "");
        expect(workers.body.find("\"active_requests\":0") != std::string::npos, "music proxy releases request token");
        const auto usage = server->handle_request("GET", "/v1/router/usage", "Bearer test-secret", "");
        expect(usage.body.find("\"" + capability + "\":{\"requests\":") != std::string::npos, "music usage counted");
        response = server->handle_request("POST", path, "Bearer test-secret", "[]", "application/json");
        expect(response.status == 422, "router music accepts dict bodies only");
        response = server->handle_request("GET", path, "Bearer test-secret", "");
        expect(response.status == 405, "music routes require POST");
        response = server->handle_request("POST", path, "Bearer wrong", request, "application/json");
        expect(response.status == 401, "music router authentication");
        upstream.verify(path);
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        expect(argc == 3, "reference root and fixture required");
        const auto root = std::filesystem::absolute(argv[1]).string();
        const auto fixture = std::filesystem::absolute(argv[2]).string();
        char temporary[] = "/tmp/ezlocalai-music-XXXXXX";
        expect(mkdtemp(temporary) != nullptr, "music temporary directory");
        const auto previous = std::filesystem::current_path();
        std::filesystem::current_path(temporary);
        const auto count = reference_cases(root.c_str(), fixture.c_str());
        router_cases();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(temporary);
        std::cout << "Verified " << count << " music and music-video API/ACE-Step reference cases\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
