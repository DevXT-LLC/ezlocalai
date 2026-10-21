#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <nlohmann/json.hpp>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <unistd.h>

namespace {
using namespace ezlocalai;
using Json = nlohmann::json;
using namespace std::chrono_literals;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned entered = 0;
    bool open = false;
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered; changed.notify_all();
        check(changed.wait_for(lock, 10s, [&] { return open; }), "fixture gate timed out");
    }
    void ready(unsigned count = 1) {
        std::unique_lock<std::mutex> lock(mutex);
        check(changed.wait_for(lock, 4s, [&] { return entered >= count; }), "fixture gate not entered");
    }
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; changed.notify_all(); }
};
struct Backend : InferenceBackend {
    Gate *llm = nullptr, *unload = nullptr, *restore = nullptr;
    bool fail_restore = false;
    std::string chat(const std::string &) override { if (llm) llm->wait(); return "{}"; }
    std::string completion(const std::string &body) override { return chat(body); }
    std::string embedding(const std::string &) override { return "{}"; }
    std::function<void(bool)> unload_llms_for_media() override {
        if (unload) unload->wait();
        return [this](bool) { if (restore) restore->wait(); if (fail_restore) throw std::runtime_error("fixture restore failure"); };
    }
};
void environment() {
    for (const auto &entry : environment_defaults) unsetenv(std::string(entry.first).c_str());
    for (const auto &[name, value] : std::map<std::string, std::string>{
        {"EZLOCALAI_API_KEY", ""}, {"DEFAULT_MODEL", "none"}, {"VOICE_SERVER", ""},
        {"ROUTER_URL", ""}, {"TTS_ENABLED", "false"}, {"STT_ENABLED", "false"},
        {"EMBEDDING_ENABLED", "false"}, {"IMAGE_ENABLED", "false"}, {"VIDEO_ENABLED", "false"},
        {"MUSIC_ENABLED", "false"}, {"LAZY_LOAD_VOICE", "true"}, {"TTS_N_PARALLEL", "1"},
        {"STT_N_PARALLEL", "1"}, {"QWEN_TTS_ENABLE_GTTS_FALLBACK", "false"}})
        setenv(name.c_str(), value.c_str(), 1);
}
std::shared_ptr<ResourceManager> manager() {
    auto probes = native_resource_probes();
    probes.gpu_count = [] { return 0; };
    probes.total_per_gpu = probes.free_per_gpu = [] { return std::vector<double>{}; };
    probes.free_vram = [](auto) { return 0.; };
    probes.allocator_stats = [] { return std::vector<CudaAllocatorStats>{}; };
    probes.system_ram = probes.available_ram = [] { return 16.; };
    return std::make_shared<ResourceManager>(probes);
}
TtsPoolFactory tts_factory(Gate *synthesis = nullptr) {
    return [synthesis](auto resources) {
        auto ops = tts_pool_resources({}, resources);
        ops.can_load = [] { return ModelAdmission{true, "cpu", "fixture"}; };
        ops.schedule_cleanup = [](auto work) { work(); };
        ops.load = [synthesis](const auto &) {
            auto model = std::make_shared<TtsPoolModel>();
            model->device = [] { return "cpu"; };
            model->closed = [] { return false; };
            model->close = [] {};
            model->generate = [synthesis](const auto &) { if (synthesis) synthesis->wait(); return "UklGRg=="; };
            return model;
        };
        return ops;
    };
}
SttPoolFactory stt_factory() {
    return [](auto resources) {
        auto ops = stt_pool_resources({}, resources);
        ops.can_load = [] {};
        ops.schedule_cleanup = [](auto work) { work(); };
        ops.load = [](const auto &) {
            auto model = std::make_shared<SttPoolModel>();
            model->device = [] { return "cpu"; };
            model->close = [] {};
            return model;
        };
        return ops;
    };
}
Json slots(NativeServer &server) {
    const auto response = server.handle_request("GET", "/v1/resources", "", "");
    check(response.status == 200, "resources status");
    return Json::parse(response.body).at("slots");
}
void compare_speech(NativeServer &server, const Json &row) {
    auto actual = slots(server).at("cap_slots");
    actual.erase("text");
    check(actual == row.at("slots").at("cap_slots"), row.at("name").get<std::string>() + ": " + actual.dump());
}
void static_cases(const Json &cases) {
    for (const auto &row : cases) {
        environment();
        const auto config_env = row.at("environment");
        // The client URL is captured, while parallel limits are read live by the snapshot.
        setenv("VOICE_SERVER", config_env.at("VOICE_SERVER").get<std::string>().c_str(), 1);
        ezl_env_config config{}; ezl_load_config(&config);
        auto resources = manager();
        NativeServer server(config, std::make_unique<Backend>(), resources, stt_factory(), tts_factory(), {}, true);
        for (const auto &[key, value] : config_env.items()) setenv(key.c_str(), value.get<std::string>().c_str(), 1);
        for (const auto &[kind, key] : std::vector<std::pair<ModelType, std::string>>{{ModelType::Tts, "tts_active"}, {ModelType::Stt, "stt_active"}}) {
            resources->register_model(kind, key, "cpu", 0);
            for (int i = 0; i < row.at(key).get<int>(); ++i) resources->mark_model_in_use(kind, true);
        }
        const auto actual = slots(server);
        check(actual == row.at("slots"), row.at("name").get<std::string>() + ": " + actual.dump());
    }
}
const Json &named(const Json &rows, const std::string &name) {
    for (const auto &row : rows) if (row.at("name") == name) return row;
    throw std::runtime_error("missing transition " + name);
}
void handoff(const Json &rows, bool fail_restore = false) {
    environment();
    for (const auto &[key, value] : named(rows, "idle").at("environment").items()) setenv(key.c_str(), value.get<std::string>().c_str(), 1);
    setenv("DEFAULT_MODEL", "fixture-text", 1);
    Gate llm, unload, synthesis, restore;
    auto backend = std::make_unique<Backend>();
    backend->llm = &llm; backend->unload = &unload; backend->restore = &restore;
    backend->fail_restore = fail_restore;
    ezl_env_config config{}; ezl_load_config(&config);
    NativeServer server(config, std::move(backend), manager(), stt_factory(), tts_factory(&synthesis), {}, true);
    std::vector<std::future<HttpResponse>> pending;
    struct Release { Gate &a, &b, &c, &d; ~Release() { a.release(); b.release(); c.release(); d.release(); } } release{llm, unload, synthesis, restore};
    compare_speech(server, named(rows, "idle"));
    pending.push_back(std::async(std::launch::async, [&] { return server.handle_request("POST", "/v1/chat/completions", "", R"({"model":"fixture-text","messages":[]})"); }));
    llm.ready(); compare_speech(server, named(rows, "llm"));
    pending.push_back(std::async(std::launch::async, [&] { return server.handle_request("POST", "/v1/audio/speech", "", R"({"input":"hello"})", "application/json"); }));
    const auto deadline = std::chrono::steady_clock::now() + 4s;
    while (slots(server).at("cap_slots").at("text").at("capacity") != 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(5ms);
    check(slots(server).at("cap_slots").at("text").at("capacity") == 0, "voice waits for LLM while excluding new LLM work");
    compare_speech(server, named(rows, "waiting"));
    llm.release(); check(pending[0].get().status == 200, "LLM request completed");
    unload.ready(); compare_speech(server, named(rows, "unloading"));
    unload.release(); synthesis.ready(); compare_speech(server, named(rows, "synthesis"));
    synthesis.release(); restore.ready(); compare_speech(server, named(rows, "restoring"));
    restore.release(); check(pending[1].get().status == (fail_restore ? 500 : 200), "speech request reports restoration outcome");
    compare_speech(server, named(rows, "idle"));
    check(slots(server).at("cap_slots").at("text").at("capacity") == 1, "LLM restored capacity");
}
void parallel(const Json &row) {
    environment();
    for (const auto &[key, value] : row.at("environment").items()) setenv(key.c_str(), value.get<std::string>().c_str(), 1);
    setenv("LAZY_LOAD_VOICE", "false", 1);
    Gate synthesis;
    ezl_env_config config{}; ezl_load_config(&config);
    NativeServer server(config, std::make_unique<Backend>(), manager(), stt_factory(), tts_factory(&synthesis), {}, true);
    std::vector<std::future<HttpResponse>> pending;
    struct Release { Gate &gate; ~Release() { gate.release(); } } release{synthesis};
    for (int i = 0; i < 3; ++i) pending.push_back(std::async(std::launch::async, [&] { return server.handle_request("POST", "/v1/audio/speech", "", R"({"input":"hello"})", "application/json"); }));
    synthesis.ready(2);
    const auto deadline = std::chrono::steady_clock::now() + 4s;
    while (server.tts_permits()->waiters.empty() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(5ms);
    check(server.tts_permits()->waiters.size() == 1, "third speech request waits without occupying a model slot");
    compare_speech(server, row);
    synthesis.release();
    for (auto &future : pending) check(future.get().status == 200, "parallel speech completes");
    check(slots(server).at("cap_slots").at("tts").at("in_flight") == 0, "parallel activity returns to zero");
}
}
int main(int argc, char **argv) { try {
    check(argc == 2, "source fixture required");
    std::ifstream input(argv[1]); Json fixture; input >> fixture;
    char pattern[] = "/tmp/ezlocalai-speech-slots-XXXXXX";
    check(mkdtemp(pattern), "temporary directory");
    const auto previous = std::filesystem::current_path(); std::filesystem::current_path(pattern);
    struct Cleanup { std::filesystem::path previous, path; ~Cleanup() { std::filesystem::current_path(previous); std::error_code ec; std::filesystem::remove_all(path, ec); } } cleanup{previous, pattern};
    static_cases(fixture.at("static_cases"));
    handoff(fixture.at("transitions"));
    handoff(fixture.at("transitions"), true);
    parallel(named(fixture.at("transitions"), "parallel"));
    parallel(named(fixture.at("transitions"), "cross-modal"));
    std::cout << fixture.at("static_cases").size() << " Python slot snapshots, parallel admission and LLM handoff transitions match\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
