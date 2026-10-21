#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
using namespace std::chrono_literals;
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream input(path, std::ios::binary); return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()}; }
std::vector<int> pids(const fs::path &path) { std::vector<int> result; std::istringstream input(read(path)); for (int pid; input >> pid;) result.push_back(pid); return result; }
template<class Predicate> bool wait(Predicate predicate, std::chrono::milliseconds timeout = 2000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do { if (predicate()) return true; std::this_thread::sleep_for(5ms); } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}
int available_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "port bind"); socklen_t size = sizeof(address);
    getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size); close(fd); return ntohs(address.sin_port);
}
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const auto *key : {"LLAMA_CPP_CONTEXT", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_MMPROJ_PATH",
        "LLM_FIXTURE_INFERENCE_RESULT", "LLM_FIXTURE_INFERENCE_FILE", "WHISPER_CPP_EXTRA_ARGS", "WHISPER_CPP_CLI", "LLM_AUDIO_FIXTURE_DELAY_MS",
        "LLM_AUDIO_FIXTURE_EXIT", "LLM_AUDIO_FIXTURE_RESULT", "LLM_AUDIO_FIXTURE_BLOCK", "VOICE_WAIT_FOR_LLM_IDLE_TIMEOUT", "STT_WAIT_FOR_LLM_IDLE_TIMEOUT"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("EZLOCALAI_API_KEY", "none", 1); setenv("GPU_LAYERS", "0", 1); setenv("DEFAULT_MODEL", "ordinary", 1);
    setenv("LLM_MAX_TOKENS", "512", 1); setenv("LLM_PROMPT_CACHE_MIB", "0", 1); setenv("SPECULATIVE_TYPE", "none", 1); setenv("N_PARALLEL", "2", 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("WHISPER_CPP_BIN", TEST_LLM_AUDIO_FIXTURE_BIN, 1); setenv("STT_UNLOAD_LLM_DURING_GENERATION", "false", 1);
}
struct Scenario {
    fs::path root;
    std::string suffix;
    explicit Scenario(const fs::path &path) : root(path), suffix('-' + std::to_string(available_port())) {
        environment(); fs::create_directories(root); std::ofstream(root / "model.gguf") << "fixture"; std::ofstream(root / "whisper.bin") << "fixture";
        setenv("LLAMA_CPP_MODEL_PATH", (root / "model.gguf").c_str(), 1); setenv("WHISPER_MODEL_PATH", (root / "whisper.bin").c_str(), 1);
        setenv("EMBEDDING_FIXTURE_DIR", root.c_str(), 1); setenv("LLM_AUDIO_FIXTURE_DIR", root.c_str(), 1);
        setenv("LLAMA_CPP_SERVER_PORT", suffix.substr(1).c_str(), 1); setenv("ROUTER_OUTPUTS_DIR", (root / "outputs").c_str(), 1);
    }
    std::unique_ptr<ezlocalai::NativeServer> server() {
        ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 5000;
        return std::make_unique<ezlocalai::NativeServer>(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
    }
    bool loaded() const { return fs::exists(root / ("launch" + suffix)); }
    size_t audio_count() const { return pids(root / "audio-launches").size(); }
    std::string seen() const { return read(root / ("seen" + suffix)); }
    void verify_cleanup() const {
        for (const auto &file : {root / ("launch" + suffix), root / "audio-launches"})
            for (const auto pid : pids(file)) expect(kill(pid, 0) != 0, "audio/model child reaped");
        expect(!fs::exists(root / "outputs") || fs::is_empty(root / "outputs"), "temporary STT audio removed");
    }
};
const std::string inline_body = R"({"messages":[{"role":"user","content":["before",{"audio_url":"data:audio/wav;base64,YQ=="},"after"]}]})";
const std::string format_body = R"({"messages":[{"role":"user","content":"YQ=="}],"audio_format":"audio/x-wav"})";
const std::string completion_body = R"({"prompt":"YQ==","audio_format":"audio/mpeg"})";
const std::string recovery_body = R"({"messages":[{"role":"user","content":"recovered"}]})";
ezlocalai::HttpResponse request(ezlocalai::NativeServer &server, const std::string &body, bool completion = false) {
    return server.handle_request("POST", completion ? "/v1/completions" : "/v1/chat/completions", "", body);
}
void phases(const fs::path &root) {
    for (int kind = 0; kind < 3; ++kind) {
        Scenario scenario(root / std::to_string(kind)); setenv("LLM_AUDIO_FIXTURE_BLOCK", "1", 1);
        auto server = scenario.server(); const auto body = kind == 0 ? inline_body : kind == 1 ? format_body : completion_body;
        auto pending = std::async(std::launch::async, [&] { return request(*server, body, kind == 2); });
        const bool started = wait([&] { return scenario.audio_count() == 1; });
        const bool loaded = scenario.loaded();
        std::ofstream(scenario.root / "audio-release") << "release";
        const auto response = pending.get();
        expect(started && loaded == (kind != 0), "inline STT precedes LLM load; audio_format follows it");
        expect(response.status == 200, "managed audio status " + response.body);
        const auto seen = scenario.seen();
        expect(seen.find(kind == 0 ? R"("content":"Transcribed Audio:  first second\n\n\nbeforeafter")"
            : kind == 1 ? R"("content":" first second\n")" : R"("prompt":" first second\n")") != std::string::npos,
            "raw joined transcript reaches LLM " + seen);
        const auto pid = std::to_string(pids(scenario.root / "audio-launches").front());
        expect(read(scenario.root / ("audio-bytes-" + pid)) == "a", "decoded bytes reach STT");
        const auto args = read(scenario.root / ("audio-args-" + pid));
        const auto start = args.find("\n-f\n") + 4, end = args.find('\n', start);
        const auto audio_path = fs::path(args.substr(start, end - start));
        expect(audio_path.stem().string().size() == 32 && audio_path.extension() == (kind == 0 ? ".YQ==" : kind == 1 ? ".wav" : ".mp3"), "source UUID filename and extension policy " + audio_path.string());
        server.reset(); scenario.verify_cleanup();
    }
}
void errors(const fs::path &root) {
    struct Case { std::string body; bool loaded, launched, completion = false; };
    const std::vector<Case> cases = {
        {R"({"messages":[{"content":[{"audio_url":"data:audio/wav;base64,a"}]}]})", false, false},
        {R"({"messages":[{"content":"a"}],"audio_format":"wav"})", true, false},
        {R"({"messages":[{"content":null}],"audio_format":"wav"})", true, false},
        {R"({"messages":[],"audio_format":"wav"})", false, false},
        {R"({"messages":[{"content":"YQ=="}],"audio_format":true})", true, false},
        {R"({"prompt":"a","audio_format":"wav"})", true, false, true},
        {inline_body, false, true}, {format_body, true, true}, {completion_body, true, true, true},
    };
    size_t count = 0;
    for (const auto &item : cases) {
        Scenario scenario(root / std::to_string(count++));
        if (item.launched) setenv("LLM_AUDIO_FIXTURE_EXIT", "7", 1);
        auto server = scenario.server(); const auto response = request(*server, item.body, item.completion);
        expect(response.status == 500 && response.body == R"({"detail":"Internal server error"})", "audio prep error contract " + response.body);
        expect(scenario.loaded() == item.loaded && scenario.audio_count() == (item.launched ? 1 : 0), "failed audio phase " + std::to_string(count));
        expect(scenario.seen().empty(), "failed preparation skips inference");
        unsetenv("LLM_AUDIO_FIXTURE_EXIT");
        expect(request(*server, inline_body).status == 200, "STT failure releases admission and LLM slots");
        server.reset(); scenario.verify_cleanup();
    }
}
class Remote {
    int listener_;
    std::atomic<bool> running_{true};
    std::thread thread_;
    std::mutex mutex_;
    std::string requests_;
public:
    std::string url;
    Remote() {
        listener_ = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(!bind(listener_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) && !listen(listener_, 8), "remote bind");
        socklen_t size = sizeof(address); getsockname(listener_, reinterpret_cast<sockaddr *>(&address), &size); url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port));
        thread_ = std::thread([this] {
            while (running_) {
                int peer = accept(listener_, nullptr, nullptr); if (peer < 0) break;
                timeval timeout{2, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
                std::string wire; char buffer[4096]; size_t expected = std::string::npos;
                while (wire.size() < expected) {
                    const auto count = recv(peer, buffer, sizeof(buffer), 0); if (count <= 0) break; wire.append(buffer, count);
                    const auto end = wire.find("\r\n\r\n"); if (end != std::string::npos) { const auto length = wire.find("Content-Length: "); expected = end + 4 + (length == std::string::npos ? 0 : std::stoul(wire.substr(length + 16))); }
                }
                { std::lock_guard<std::mutex> lock(mutex_); requests_ += wire; }
                const bool audio = wire.rfind("GET /audio ", 0) == 0;
                const auto body = audio ? "a" : wire.rfind("GET /v1/resources ", 0) == 0 ? R"({"free_vram_gb":64})"
                    : wire.rfind("GET ", 0) == 0 ? R"({"data":[{"id":"ordinary"}]})" : R"({"choices":[{"message":{"content":"forwarded"}}]})";
                const auto response = "HTTP/1.1 " + std::string(audio ? "500 Error" : "200 OK") + "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(std::string(body).size()) + "\r\nConnection: close\r\n\r\n" + body;
                for (size_t at = 0; at < response.size();) { const auto n = send(peer, response.data() + at, response.size() - at, MSG_NOSIGNAL); if (n <= 0) break; at += n; }
                close(peer);
            }
        });
    }
    ~Remote() { running_ = false; shutdown(listener_, SHUT_RDWR); close(listener_); thread_.join(); }
    std::string requests() { std::lock_guard<std::mutex> lock(mutex_); return requests_; }
};
void networking(const fs::path &root) {
    Remote remote;
    for (int fallback = 0; fallback < 2; ++fallback) {
        Scenario scenario(root / (fallback ? "fallback" : "text"));
        setenv(fallback ? "FALLBACK_SERVER" : "TEXT_SERVER", remote.url.c_str(), 1);
        if (fallback) setenv("FALLBACK_MEMORY_THRESHOLD", "999999999999", 1);
        auto server = scenario.server();
        for (const auto &body : {inline_body, format_body}) {
            const auto response = request(*server, body);
            expect(response.status == 200 && response.body.find("forwarded") != std::string::npos, "remote audio forwarding " + response.body);
        }
        expect(!scenario.loaded() && scenario.audio_count() == 0, "forwarding precedes local model and STT load");
        expect(remote.requests().find("data:audio/wav;base64,YQ==") != std::string::npos && remote.requests().find(R"("audio_format":"audio/x-wav")") != std::string::npos, "raw audio fields forwarded");
        server.reset(); scenario.verify_cleanup();
    }
    Scenario scenario(root / "download"); auto server = scenario.server();
    const auto response = request(*server, "{\"messages\":[{\"content\":[{\"audio_url\":{\"url\":\"" + remote.url + "/audio\"}}]}]}");
    expect(response.status == 200, "HTTP error audio body still transcribed " + response.body);
    const auto pid = std::to_string(pids(scenario.root / "audio-launches").front());
    expect(read(scenario.root / ("audio-bytes-" + pid)) == "a" && read(scenario.root / ("audio-args-" + pid)).find(".wav\n") != std::string::npos, "remote audio body and default wav");
    expect(remote.requests().find("User-Agent: python-requests/2.32.5") != std::string::npos, "source requests user agent");
    server.reset(); scenario.verify_cleanup();
}
void admission(const fs::path &root) {
    for (int capacity : {1, 2}) {
        Scenario scenario(root / std::to_string(capacity)); setenv("STT_N_PARALLEL", std::to_string(capacity).c_str(), 1); setenv("LLM_AUDIO_FIXTURE_BLOCK", "1", 1);
        auto server = scenario.server(); setenv("STT_N_PARALLEL", std::to_string(3 - capacity).c_str(), 1);
        auto first = std::async(std::launch::async, [&] { return request(*server, inline_body); });
        const bool started = wait([&] { return scenario.audio_count() == 1; });
        auto second = std::async(std::launch::async, [&] { return request(*server, inline_body); });
        const bool parallel = wait([&] { return scenario.audio_count() == 2; }, capacity == 1 ? 150ms : 2000ms);
        std::ofstream(scenario.root / "audio-release") << "release";
        const auto a = first.get(), b = second.get();
        expect(started && parallel == (capacity == 2), "STT capacity is captured when Pipes is constructed");
        expect(a.status == 200 && b.status == 200 && scenario.audio_count() == 2, "queued STT requests finish");
        server.reset(); scenario.verify_cleanup();
    }
}
void handoff(const fs::path &root, const std::vector<std::vector<std::string>> &guards) {
    size_t count = 0;
    for (const auto &guard : guards) for (bool top_level : {false, true}) {
        Scenario scenario(root / std::to_string(count++));
        unsetenv("STT_UNLOAD_LLM_DURING_GENERATION"); setenv("VOICE_WAIT_FOR_LLM_IDLE_TIMEOUT", guard[0].c_str(), 1); setenv("STT_WAIT_FOR_LLM_IDLE_TIMEOUT", guard[1].c_str(), 1);
        auto server = scenario.server(); const auto start = std::chrono::steady_clock::now();
        const auto response = request(*server, top_level ? format_body : inline_body);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        expect(response.status == std::stoi(guard[4]) && response.body == R"({"detail":"Internal server error"})", "nested handoff fails on its own LLM lease " + response.body);
        expect(scenario.audio_count() == 0 && scenario.loaded() == top_level, "handoff timeout happens before STT load");
        expect(elapsed >= std::chrono::milliseconds(std::stoi(guard[8])), "source handoff polls every 100ms");
        expect(request(*server, recovery_body).status == 200, "handoff failure clears unavailable flag and releases slots");
        server.reset(); scenario.verify_cleanup();
    }
}
}
void llm_audio_runtime(const std::vector<std::vector<std::string>> &guards) {
    char pattern[] = "/tmp/ezlocalai-audio-XXXXXX"; const auto path = mkdtemp(pattern); expect(path, "audio test directory"); const fs::path root(path);
    phases(root / "phases"); errors(root / "errors"); networking(root / "network"); admission(root / "admission"); handoff(root / "handoff", guards);
    fs::remove_all(root);
    std::cout << "Managed audio load order, transcript bytes, failures, forwarding, HTTP fetch, admission and handoff recovery passed\n";
}
