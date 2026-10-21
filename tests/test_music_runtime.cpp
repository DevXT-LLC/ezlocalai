#include "ezlocalai/music_runtime.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "process_fixture.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <csignal>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read_file(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::vector<std::string> split(const std::string &text, char delimiter) {
    std::vector<std::string> values;
    size_t start = 0;
    do {
        const auto end = text.find(delimiter, start);
        values.push_back(text.substr(start, end == std::string::npos ? end : end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    return values;
}
std::string unhex(const std::string &text) {
    std::string value;
    for (size_t i = 0; i < text.size(); i += 2) value += static_cast<char>(std::stoi(text.substr(i, 2), nullptr, 16));
    return value;
}
std::vector<std::string> vector_value(const std::string &text) {
    std::vector<std::string> values;
    if (text != "-") for (const auto &item : split(text, ',')) values.push_back(unhex(item));
    return values;
}
std::string sha256(const std::string &value) {
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned length = 0;
    expect(EVP_Digest(value.data(), value.size(), digest, &length, EVP_sha256(), nullptr) == 1, "SHA256");
    std::ostringstream out;
    for (unsigned i = 0; i < length; ++i) out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    return out.str();
}
void environment() {
    unsetenv("LLAMA_CPP_EMBEDDING_MODEL_PATH");
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const char *key : {"ACE_STEP_EXTRA_ARGS", "ACE_STEP_KEEP_LOADED", "ACE_STEP_ADAPTERS_DIR", "ACE_STEP_MAX_BATCH", "ACE_STEP_MAX_SEQ", "ACE_STEP_VAE_CHUNK", "ACE_STEP_VAE_OVERLAP", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_MODEL_PATH", "LLAMA_CPP_SERVER_BIN", "LLAMA_CPP_SERVER_PORT", "LLAMA_CPP_START_TIMEOUT", "LLAMA_CPP_EMBEDDING_SERVER_URL", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_CONTEXT", "LLAMA_CPP_EMBEDDING_SERVER_PORT"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1);
    setenv("ROUTER_DRAIN_FILE", "", 1);
    setenv("USAGE_FILE", "usage.json", 1);
    setenv("DEFAULT_MODEL", "test-model", 1);
    setenv("MAX_CONCURRENT_REQUESTS", "2", 1);
    setenv("N_PARALLEL", "1", 1);
    setenv("MUSIC_ENABLED", "true", 1);
}
size_t reference_cases(const std::string &root, const std::string &fixture) {
    std::istringstream input(read_file(fixture));
    size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto marker = line.find(" sha256=");
            if (marker != std::string::npos) expect(sha256(read_file(root + "/" + line.substr(2, marker - 2))) == line.substr(marker + 8), "reference changed: " + line);
            continue;
        }
        const auto c = split(line, '\t');
        expect(c.size() == 10, "runtime fixture columns");
        environment();
        if (c[1] != "-") for (const auto &entry : split(c[1], ';')) {
            const auto equal = entry.find('=');
            setenv(entry.substr(0, equal).c_str(), unhex(entry.substr(equal + 1)).c_str(), 1);
        }
        expect(ezlocalai::ace_step_internal_url() == unhex(c[2]), c[0] + " internal URL");
        expect(ezlocalai::ace_step_server_url() == unhex(c[3]), c[0] + " server URL");
        expect(ezlocalai::ace_step_models_dir() == unhex(c[4]), c[0] + " models directory");
        expect(ezlocalai::ace_step_binary_path() == unhex(c[5]), c[0] + " binary path");
        expect(ezlocalai::selected_ace_step_model_files() == vector_value(c[6]), c[0] + " model files");
        std::vector<std::string> command; std::string error;
        try { command = ezlocalai::build_ace_step_server_command(); } catch (const std::invalid_argument &e) { error = e.what(); }
        if (c[7].rfind("ERROR:", 0) == 0) expect(error == unhex(c[7].substr(6)), c[0] + " command error");
        else expect(error.empty() && command == vector_value(c[7]), c[0] + " command argv");
        expect(ezlocalai::music_should_unload_llm() == (c[8] == "1"), c[0] + " unload policy");
        expect(ezlocalai::music_should_reload_llm() == (c[9] == "1"), c[0] + " reload policy");
        ++count;
    }
    expect(count == 681, "all music runtime reference cases");
    return count;
}

void precache_cases(const std::string &root, const std::string &fixture) {
    using Json = nlohmann::json;
    const auto captured = Json::parse(read_file(fixture));
    for (const auto &[path, digest] : captured.at("sources").items())
        expect(sha256(read_file(root + "/" + path)) == digest.get<std::string>(), "precache reference changed: " + path);
    expect(captured.at("cases").size() == 45, "all precache source cases");
    for (const auto &item : captured.at("cases")) {
        environment();
        for (const auto &[key, value] : item.at("environment").items()) setenv(key.c_str(), value.get<std::string>().c_str(), 1);
        Json trace = Json::array();
        const std::string failure = item.at("failure").is_null() ? "" : item.at("failure").get<std::string>();
        int downloads = 0;
        ezlocalai::MusicPrecacheOperations operations;
        operations.mkdir = [&](const auto &path) {
            trace.push_back({"mkdir", path});
            if (failure == "mkdir") throw std::runtime_error("oracle mkdir failure");
        };
        operations.exists = [&](const auto &path) {
            trace.push_back({"exists", path});
            if (failure == "exists") throw std::runtime_error("oracle exists failure");
            const auto &cached = item.at("cached");
            return std::find(cached.begin(), cached.end(), Json(path)) != cached.end();
        };
        operations.download = [&](const auto &repo, const auto &file, const auto &dir) {
            trace.push_back({"download", repo, file, dir});
            if (failure == "download-" + std::to_string(++downloads)) throw std::runtime_error("oracle download failure");
        };
        ezlocalai::precache_music_models(std::move(operations));
        expect(trace == item.at("trace"), item.at("name").get<std::string>() + " precache calls: " + trace.dump());
    }
}

class Loopback {
public:
    using Handler = std::function<ezlocalai::HttpResponse(const std::string &, const std::string &)>;
    explicit Loopback(Handler handler) : handler_(std::move(handler)) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        expect(fd_ >= 0, "socket");
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(bind(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "bind");
        socklen_t size = sizeof(addr);
        expect(getsockname(fd_, reinterpret_cast<sockaddr *>(&addr), &size) == 0, "port");
        port = ntohs(addr.sin_port);
        url = "http://127.0.0.1:" + std::to_string(port);
        expect(listen(fd_, 8) == 0, "listen");
        thread_ = std::thread([this] { serve(); });
    }
    ~Loopback() { stopped_ = true; shutdown(fd_, SHUT_RDWR); thread_.join(); close(fd_); }
    int port;
    std::string url;
private:
    void serve() {
        while (!stopped_) {
            pollfd poller{fd_, POLLIN, 0};
            if (poll(&poller, 1, 100) <= 0) continue;
            const int peer = accept(fd_, nullptr, nullptr);
            if (peer < 0) continue;
            timeval timeout{2, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            std::string raw; char buffer[4096]; size_t expected = std::string::npos;
            while (raw.size() < expected) {
                const auto count = recv(peer, buffer, sizeof(buffer), 0);
                if (count <= 0) break;
                raw.append(buffer, static_cast<size_t>(count));
                const auto header = raw.find("\r\n\r\n");
                if (header != std::string::npos) {
                    const auto length = raw.find("Content-Length: ");
                    expected = header + 4 + (length == std::string::npos ? 0 : std::stoul(raw.substr(length + 16)));
                }
            }
            std::istringstream line(raw); std::string method, path; line >> method >> path;
            const auto response = handler_(path, raw);
            const auto bytes = "HTTP/1.1 " + std::to_string(response.status) + " Response\r\nContent-Type: " + response.content_type + "\r\nContent-Length: " + std::to_string(response.body.size()) + "\r\nConnection: close\r\n\r\n" + response.body;
            size_t sent = 0;
            while (sent < bytes.size()) {
                const auto count = send(peer, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
                if (count <= 0) break;
                sent += static_cast<size_t>(count);
            }
            shutdown(peer, SHUT_RDWR); close(peer);
        }
    }
    int fd_ = -1;
    Handler handler_;
    std::thread thread_;
    std::atomic<bool> stopped_{false};
};

template<class Predicate> void wait_until(Predicate predicate, const std::string &message) {
    const auto start = std::chrono::steady_clock::now();
    while (!predicate()) {
        expect(std::chrono::steady_clock::now() - start < std::chrono::seconds(5), message);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
struct ChildCleanup {
    ezlocalai::AceStepRuntime runtime;
    ~ChildCleanup() {
        const int pid = runtime.process_id();
        if (pid > 0) { kill(pid, SIGTERM); int status; while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} }
    }
};
void startup_cases() {
    environment();
    const auto binary = std::filesystem::absolute("ace fixture");
    const auto arguments = std::filesystem::absolute("ace-argv");
    write_process_fixture(binary, {arguments.string(), true, 10000, 0, false, {}, {}});
    setenv("ACE_STEP_BIN", binary.c_str(), 1);
    setenv("ACE_STEP_MODELS_DIR", "models with spaces; $(literal)", 1);
    setenv("ACE_STEP_EXTRA_ARGS", "--example 'two words' '' '$literal'", 1);
    ChildCleanup child;
    int probes = 0, timeouts = 0;
    const auto healthy = [&](const std::string &url) { ++probes; expect(url == "http://127.0.0.1:8085", "health URL"); return std::filesystem::exists(arguments); };
    const auto timeout = [&] { ++timeouts; return 2.0; };
    setenv("MUSIC_ENABLED", "false", 1);
    expect(child.runtime.start(healthy, timeout) < 0 && probes == 0, "disabled music skips startup");
    setenv("MUSIC_ENABLED", "yes", 1);
    setenv("ACE_STEP_SERVER_URL", " http://external ", 1);
    expect(child.runtime.start(healthy, timeout) < 0 && probes == 0, "external server skips startup");
    unsetenv("ACE_STEP_SERVER_URL"); setenv("ACE_STEP_AUTO_START", "off", 1);
    expect(child.runtime.start(healthy, timeout) < 0 && probes == 0, "auto start disabled");
    setenv("ACE_STEP_AUTO_START", "", 1);
    expect(child.runtime.start([](auto &) { return true; }, timeout) < 0, "healthy server skips spawn");
    const int pid = child.runtime.start(healthy, timeout);
    expect(pid > 0 && timeouts == 1, "spawn with health wait despite missing models");
    expect(read_file(arguments).find("two words\n\n$literal\n") != std::string::npos, "literal argv and quoted empty argument");
    expect(child.runtime.start([](auto &) { return false; }, timeout) == pid && timeouts == 1, "running child reused without wait");
    expect(child.runtime.start(healthy, timeout) < 0, "health precedes existing child check");
    kill(pid, SIGTERM);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(child.runtime.start([](auto &) { return false; }, [] { return 0.0; }) != pid, "exited child restarted; zero timeout leaves new child running");
    expect(kill(child.runtime.process_id(), 0) == 0, "startup timeout preserves child");
    {
        ChildCleanup invalid;
        bool caught = false;
        try { invalid.runtime.start([](auto &) { return false; }, []() -> double { throw std::invalid_argument("invalid timeout"); }); }
        catch (const std::invalid_argument &) { caught = true; }
        expect(caught && invalid.runtime.process_id() > 0, "timeout parsed after spawn");
    }
    setenv("ACE_STEP_BIN", "/missing/ace-server", 1);
    ChildCleanup missing;
    expect(missing.runtime.start([](auto &) { return false; }, timeout) < 0, "missing binary is nonfatal");
    const auto denied = std::filesystem::absolute("not executable");
    { std::ofstream out(denied); out << "not a program"; }
    setenv("ACE_STEP_BIN", denied.c_str(), 1);
    expect(missing.runtime.start([](auto &) { return false; }, timeout) < 0, "spawn error is nonfatal");
    const auto quick = std::filesystem::absolute("exits immediately");
    write_process_fixture(quick, {{}, false, 0, 9, false, {}, {}});
    setenv("ACE_STEP_BIN", quick.c_str(), 1);
    expect(missing.runtime.start([](auto &) { return false; }, timeout) > 0, "early process exit returns process");
}

struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
    void block() { std::unique_lock<std::mutex> lock(mutex); entered = true; cv.notify_all(); cv.wait_for(lock, std::chrono::seconds(5), [&] { return released; }); }
    void wait() { std::unique_lock<std::mutex> lock(mutex); expect(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "gate entered"); }
    void release() { std::lock_guard<std::mutex> lock(mutex); released = true; cv.notify_all(); }
};
class TrackedBackend : public ezlocalai::InferenceBackend {
public:
    std::atomic<int> chats{0}, unloads{0}, restores{0};
    std::atomic<bool> loaded{true}, reloaded{false};
    Gate *chat_gate = nullptr;
    bool unload_error = false, restore_error = false;
    std::string last_body;
    std::string chat(const std::string &body) override {
        ++chats; loaded = true; last_body = body;
        if (chat_gate) chat_gate->block();
        return R"({"choices":[{"message":{"role":"assistant","content":"local"}}]})";
    }
    std::string completion(const std::string &body) override { return chat(body); }
    std::string embedding(const std::string &) override { return "{}"; }
    std::string loaded_models_json() const override { return loaded ? R"({"llm":{"name":"test-model"}})" : "{}"; }
    std::function<void(bool)> unload_llms_for_media() override {
        ++unloads;
        if (unload_error) throw std::runtime_error("unload failed");
        loaded = false;
        return [this](bool reload) { ++restores; reloaded = reload; loaded = reload; if (restore_error) throw std::runtime_error("restore failed"); };
    }
};
std::unique_ptr<ezlocalai::NativeServer> make_server(std::unique_ptr<ezlocalai::InferenceBackend> backend) {
    ezl_env_config config{}; ezl_load_config(&config);
    std::snprintf(config.api_key, sizeof(config.api_key), "%s", "test-secret");
    return std::make_unique<ezlocalai::NativeServer>(config, std::move(backend));
}
ezlocalai::HttpResponse request(ezlocalai::NativeServer &server, const std::string &path, const std::string &body = "") {
    return server.handle_request(body.empty() ? "GET" : "POST", path, "Bearer test-secret", body);
}
const std::string music_body = R"({"prompt":"test","lyrics":"[Instrumental]","keyscale":"C major","duration":10,"audio_codes":"codes","response_format":"b64_json"})";
const std::string chat_body = R"({"messages":[{"role":"user","content":"test"}]})";
ezlocalai::HttpResponse ace_response(const std::string &path) {
    if (path == "/synth") return {200, "application/json", "{\"id\":\"song\"}"};
    if (path == "/job?id=song") return {200, "application/json", "{\"status\":\"done\"}"};
    if (path == "/job?id=song&result=1") return {200, "audio/wav", "RIFF audio"};
    return {200, "application/json", "{}"};
}
void handoff_cases() {
    for (int scenario = 0; scenario < 6; ++scenario) {
        environment();
        setenv("MUSIC_UNLOAD_LLM_DURING_GENERATION", "true", 1);
        if (scenario == 1) setenv("MUSIC_RELOAD_LLM_AFTER_GENERATION", "false", 1);
        if (scenario == 4) setenv("MUSIC_UNLOAD_LLM_DURING_GENERATION", "false", 1);
        Gate music_gate;
        Loopback ace([&](const std::string &path, const std::string &) {
            if (path == "/synth") { music_gate.block(); if (scenario == 2) return ezlocalai::HttpResponse{503, "text/plain", "synthesis failed"}; }
            return ace_response(path);
        });
        setenv("ACE_STEP_SERVER_URL", ace.url.c_str(), 1);
        std::atomic<int> forwarded{0};
        std::atomic<bool> control_leaked{false};
        Loopback fallback([&](const std::string &path, const std::string &raw) {
            if (path == "/v1/resources") return ezlocalai::HttpResponse{200, "application/json", "{\"free_vram_gb\":100}"};
            ++forwarded; control_leaked = raw.find("disable_fallback") != std::string::npos;
            if(raw.find("_fixture_remote_error")!=std::string::npos)return ezlocalai::HttpResponse{200,"application/json",R"({"error":{"code":503,"message":"remote result"}})"};
            return ezlocalai::HttpResponse{200, "application/json", "{\"choices\":[{\"text\":\"fallback\"}]}"};
        });
        // Configure before the first client use. Temporarily disabling fallback
        // on each request lets the same captured client exercise both paths.
        setenv("FALLBACK_SERVER", fallback.url.c_str(), 1);
        auto backend = std::make_unique<TrackedBackend>(); auto *tracked = backend.get();
        tracked->restore_error = scenario == 3;
        tracked->unload_error = scenario == 5;
        auto server = make_server(std::move(backend));
        auto future = std::async(std::launch::async, [&] { return request(*server, "/v1/audio/music", music_body); });
        if (scenario != 5) {
            if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
                const auto early = future.get();
                throw std::runtime_error("music returned before synthesis in scenario " + std::to_string(scenario) + ": " + early.body);
            }
            music_gate.wait();
            const auto resources = request(*server, "/v1/resources").body;
            const bool unavailable = scenario != 4;
            expect(resources.find(unavailable ? "\"text\":{\"capacity\":0" : "\"text\":{\"capacity\":1") != std::string::npos, "text capacity during music: " + resources);
            expect(resources.find("\"music\":{\"capacity\":1,\"in_flight\":1") != std::string::npos, "music activity");
            auto text = request(*server, "/v1/chat/completions", R"({"messages":[],"disable_fallback":true})");
            expect(text.status == (unavailable ? 503 : 200) && (text.body.find("temporarily_unavailable") != std::string::npos) == unavailable, "text admission during music: " + text.body);
            if (unavailable) {
                text = request(*server, "/v1/chat/completions", R"({"messages":[],"stream":true,"disable_fallback":true})");
                expect(text.body == "data: \"error\"\n\ndata: [DONE]\n\n", "reference handoff chat stream");
                text = request(*server, "/v1/completions", R"({"prompt":"test","stream":true,"disable_fallback":true})");
                expect(text.status == 500, "reference handoff completion stream");
                text = request(*server, "/v1/chat/completions", chat_body);
                expect(text.body.find("fallback") != std::string::npos && forwarded == 1, "handoff fallback");
                text = request(*server, "/v1/chat/completions", R"({"messages":[],"stream":true})");
                expect(text.status==200 && text.content_type=="text/event-stream; charset=utf-8" && text.body=="data: {\"error\": {\"message\": \"'async_generator' object is not iterable\", \"type\": \"streaming_error\"}}\n\n" && forwarded==2, "source handoff fallback chat stream");
                text = request(*server, "/v1/completions", R"({"prompt":"test","stream":true})");
                expect(text.status==500 && text.content_type=="text/plain; charset=utf-8" && text.body=="Internal Server Error" && forwarded==3, "source handoff fallback completion stream");
                text = request(*server, "/v1/chat/completions", R"({"messages":[],"_fixture_remote_error":true})");
                expect(text.status==200 && text.body==R"({"error":{"code":503,"message":"remote result"}})" && forwarded==4, "remote error-valued JSON remains a successful response through the queue");
                text = request(*server, "/v1/chat/completions", R"({"messages":[],"disable_fallback":true})");
                expect(text.body.find("temporarily_unavailable") != std::string::npos && forwarded == 4 && !control_leaked, "disable fallback during handoff");
            }
            music_gate.release();
        }
        const auto response = future.get();
        expect(response.status == (scenario == 2 || scenario == 3 || scenario == 5 ? 502 : 200), "music final status: " + response.body);
        expect(tracked->unloads == (scenario == 4 ? 0 : 1), "unload count");
        expect(tracked->restores == (scenario == 4 || scenario == 5 ? 0 : 1), "restore count");
        if (scenario == 1) expect(!tracked->loaded && !tracked->reloaded, "reload disabled keeps LLM unloaded");
        const auto text = request(*server, "/v1/chat/completions", R"({"messages":[],"disable_fallback":true})");
        expect(text.status == 200 && text.body.find("local") != std::string::npos, "admission resumes after success/failure");
        expect(tracked->last_body.find("disable_fallback") == std::string::npos, "control consumed before local inference");
    }
    environment();
    setenv("MUSIC_UNLOAD_LLM_DURING_GENERATION", "true", 1);
    setenv("MUSIC_WAIT_FOR_LLM_IDLE_TIMEOUT", "0", 1);
    std::atomic<int> ace_requests{0};
    Loopback ace([&](const std::string &path, const std::string &) { ++ace_requests; return ace_response(path); });
    setenv("ACE_STEP_SERVER_URL", ace.url.c_str(), 1);
    Gate chat_gate;
    auto backend = std::make_unique<TrackedBackend>(); auto *tracked = backend.get(); tracked->chat_gate = &chat_gate;
    auto server = make_server(std::move(backend));
    auto chat = std::async(std::launch::async, [&] { return request(*server, "/v1/chat/completions", chat_body); });
    chat_gate.wait();
    const auto busy = request(*server, "/v1/resources").body;
    expect(busy.find("\"music\":{\"capacity\":1,\"in_flight\":1") != std::string::npos, "LLM reserves dependent music capacity");
    const auto failed = request(*server, "/v1/audio/music", music_body);
    expect(failed.status == 502 && failed.body.find("LLM inference is active; music generation cannot unload the LLM yet") != std::string::npos, "busy LLM timeout");
    expect(ace_requests == 0 && tracked->unloads == 0, "busy LLM does not initialize ACE or unload");
    setenv("MUSIC_WAIT_FOR_LLM_IDLE_TIMEOUT", "2", 1);
    auto music = std::async(std::launch::async, [&] { return request(*server, "/v1/audio/music", music_body); });
    expect(music.wait_for(std::chrono::milliseconds(60)) == std::future_status::timeout && tracked->unloads == 0, "music waits for active inference");
    chat_gate.release();
    expect(chat.get().status == 200 && music.get().status == 200 && tracked->restores == 1, "idle transition unloads and restores");
}

void owned_llama_case() {
    environment();
    setenv("DEFAULT_MODEL", "test-model,unused-model", 1);
    const auto binary = std::filesystem::absolute("llama fixture");
    const auto arguments = std::filesystem::absolute("llama-argv");
    write_llama_fixture(binary, arguments, true);
    std::ofstream("owned-model.gguf") << "GGUF fixture";
    setenv("LLAMA_CPP_MODEL_PATH", "owned-model.gguf", 1);
    std::atomic<bool> health_ok{true};
    Loopback llama([&](const std::string &path, const std::string &) {
        if (!health_ok) return ezlocalai::HttpResponse{503, "application/json", "{}"};
        if (path == "/health") return ezlocalai::HttpResponse{200, "application/json", "{}"};
        return ezlocalai::HttpResponse{200, "application/json", "{\"choices\":[{\"text\":\"local\"}]}"};
    });
    Loopback embedding([](const std::string &, const std::string &) { return ezlocalai::HttpResponse{200, "application/json", "{\"data\":[{\"embedding\":[1.0]}]}"}; });
    setenv("LLAMA_CPP_SERVER_BIN", binary.c_str(), 1);
    setenv("LLAMA_CPP_SERVER_PORT", std::to_string(llama.port).c_str(), 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    setenv("LLAMA_CPP_CONTEXT", "4096", 1);
    setenv("GPU_LAYERS", "7", 1);
    setenv("LLAMA_CPP_EMBEDDING_SERVER_URL", embedding.url.c_str(), 1);
    ezlocalai::LlamaCppInferenceBackend backend("test-model");
    expect(backend.chat(chat_body).find("local") != std::string::npos, "load only the requested owned LLM");
    backend.embedding("{\"input\":\"text\"}");
    wait_until([&] { return !read_file(arguments).empty(); }, "llama fixture arguments");
    const auto original = read_file(arguments);
    expect(original.find("unused-model") == std::string::npos, "second model remains lazy");
    auto restore = backend.unload_llms_for_media();
    const auto unloaded = backend.loaded_models_json();
    expect(unloaded.find("\"llm\"") == std::string::npos && unloaded.find("\"embedding\"") != std::string::npos, "unload only text engines");
    expect(backend.llm_context_size() == 0, "detached context unavailable");
    restore(true);
    wait_until([&] { return read_file(arguments).size() > original.size(); }, "restored llama fixture");
    expect(read_file(arguments) == original + original, "restoration preserves argv including context and placement");
    expect(backend.llm_context_size() == 4096, "restored context");
    setenv("LLAMA_CPP_CONTEXT", "2048", 1);
    auto video_restore = backend.unload_llms_for_video(); video_restore(true);
    expect(backend.llm_context_size() == 4096, "first video restoration preserves prior runtime context");
    video_restore = backend.unload_llms_for_video(); video_restore(false);
    expect(backend.llm_context_size() == 0, "video initial reload policy can defer restoration");
    const auto before_retry = read_file(arguments).size();
    video_restore(true);
    wait_until([&] { return read_file(arguments).size() > before_retry; }, "video retry process arguments");
    expect(backend.llm_context_size() == 2048, "video fallback retry uses configured context");
    expect(read_file(arguments).substr(before_retry).find("-ngl\n7\n") != std::string::npos, "video fallback retry uses configured GPU layers");
    setenv("LLAMA_CPP_CONTEXT", "4096", 1);
    restore = backend.unload_llms_for_media(); restore(false);
    expect(backend.loaded_models_json().find("\"llm\"") == std::string::npos, "no immediate reload");
    expect(backend.chat(chat_body).find("local") != std::string::npos, "lazy text reload after music");
    restore = backend.unload_llms_for_media();
    health_ok = false;
    std::filesystem::rename(binary, binary.string() + ".saved");
    restore(true);
    expect(backend.loaded_models_json().find("\"llm\"") == std::string::npos, "failed restore leaves model unloaded");
    std::filesystem::rename(binary.string() + ".saved", binary);
    health_ok = true;
    expect(backend.chat(chat_body).find("local") != std::string::npos, "retry succeeds after failed restore");
}

void owned_embedding_handoff_case() {
    environment(); setenv("DEFAULT_MODEL", "none", 1);
    std::ofstream("embedding-model.gguf") << "GGUF";
    setenv("LLAMA_CPP_EMBEDDING_MODEL_PATH", "embedding-model.gguf", 1);
    setenv("EMBEDDING_GPU_LAYERS", "0", 1);
    const auto binary = std::filesystem::absolute("embedding fixture");
    const auto arguments = std::filesystem::absolute("embedding-argv");
    write_llama_fixture(binary, arguments, true);
    Gate gate; std::atomic<bool> block{false}, health_ok{true};
    Loopback embedding([&](const std::string &path, const std::string &) {
        if (!health_ok) return ezlocalai::HttpResponse{503, "application/json", "{}"};
        if (path != "/health" && block) gate.block();
        return ezlocalai::HttpResponse{200, "application/json", "{\"data\":[{\"embedding\":[1.0]}]}"};
    });
    setenv("LLAMA_CPP_SERVER_BIN", binary.c_str(), 1);
    setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT", std::to_string(embedding.port).c_str(), 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    ezlocalai::LlamaCppInferenceBackend backend("none");
    expect(!backend.unload_aux_model("embedding"), "absent embedding stays absent");
    expect(backend.embedding("{\"input\":\"first\"}").find("embedding") != std::string::npos, "owned embedding starts");
    wait_until([&] { return !read_file(arguments).empty(); }, "embedding fixture arguments");
    const auto original = read_file(arguments);
    expect(backend.unload_aux_model("embedding") && backend.loaded_models_json() == "{}", "idle owned embedding unloads");
    backend.warm_aux_model("embedding");
    wait_until([&] { return read_file(arguments).size() > original.size(); }, "embedding process restarts");
    expect(read_file(arguments) == original + original && backend.loaded_models_json().find("embedding") != std::string::npos, "embedding warm restore retains configured arguments");
    block = true;
    auto request = std::async(std::launch::async, [&] { return backend.embedding("{\"input\":\"held\"}"); });
    wait_until([&] { return backend.loaded_models_json().find("\"active_requests\":1") != std::string::npos; }, "active embedding registered");
    const bool unloaded_active = backend.unload_aux_model("embedding");
    gate.release(); block = false;
    expect(request.get().find("embedding") != std::string::npos && !unloaded_active, "active embedding survives handoff race");
    expect(read_file(arguments) == original + original, "active embedding was not restarted");
    expect(backend.unload_aux_model("embedding"), "embedding unloads after inference finishes");
    health_ok = false;
    std::filesystem::rename(binary, binary.string() + ".saved");
    backend.warm_aux_model("embedding");
    expect(backend.loaded_models_json() == "{}", "failed embedding restoration leaves no phantom residency");
    std::filesystem::rename(binary.string() + ".saved", binary); health_ok = true;
    expect(backend.embedding("{\"input\":\"retry\"}").find("embedding") != std::string::npos, "later embedding request retries failed restore");
}

void lazy_start_integration() {
    environment();
    const auto binary = std::filesystem::absolute("lazy ace fixture");
    const auto arguments = std::filesystem::absolute("lazy-ace-argv");
    write_process_fixture(binary, {arguments.string(), true, 10000, 0, false, {}, {}});
    std::atomic<int> health{0};
    Loopback ace([&](const std::string &path, const std::string &) {
        if (path == "/health") { ++health; return ezlocalai::HttpResponse{std::filesystem::exists(arguments) ? 204 : 503, "text/plain", ""}; }
        return ace_response(path);
    });
    setenv("ACE_STEP_BIN", binary.c_str(), 1);
    setenv("ACE_STEP_PORT", std::to_string(ace.port).c_str(), 1);
    setenv("ACE_STEP_STARTUP_TIMEOUT", "2", 1);
    auto server = make_server(std::make_unique<TrackedBackend>());
    const auto first = request(*server, "/v1/audio/music", music_body);
    const auto before = health.load();
    const auto second = request(*server, "/v1/audio/music", music_body);
    const int pid = ezlocalai::ace_step_runtime().process_id();
    if (pid > 0) { kill(pid, SIGTERM); int status; waitpid(pid, &status, 0); }
    expect(first.status == 200 && second.status == 200, "lazy native ACE startup and synthesis");
    expect(before >= 2 && health == before, "ACE initialized once per client");
}
} // namespace

int main(int argc, char **argv) {
    try {
        expect(argc == 4, "reference, runtime fixture and precache fixture required");
        const auto root = std::filesystem::absolute(argv[1]).string(), fixture = std::filesystem::absolute(argv[2]).string();
        const auto precache = std::filesystem::absolute(argv[3]).string();
        const auto previous = std::filesystem::current_path();
        char temporary[] = "/tmp/ezlocalai-music-runtime-XXXXXX";
        expect(mkdtemp(temporary) != nullptr, "temporary directory");
        std::filesystem::current_path(temporary);
        const auto count = reference_cases(root, fixture);
        precache_cases(root, precache);
        startup_cases(); handoff_cases(); owned_llama_case(); owned_embedding_handoff_case(); lazy_start_integration();
        std::filesystem::current_path(previous); std::filesystem::remove_all(temporary);
        std::cout << "Verified " << count << " music runtime reference cases, 45 precache source cases and native lifecycle integration\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
