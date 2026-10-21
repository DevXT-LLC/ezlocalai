#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/native_server.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace ezlocalai;
using Json = nlohmann::json;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
double now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
int main() {
    try {
        for (const auto &[key, value] : environment_defaults) setenv(std::string(key).c_str(), std::string(value).c_str(), 1);
        for (const char *key : {"TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED"}) setenv(key, "false", 1);
        for (const char *key : {"EZLOCALAI_API_KEY", "ROUTER_URL", "FALLBACK_SERVER", "TEXT_SERVER", "VOICE_SERVER", "IMAGE_SERVER", "EZLOCALAI_DOTENV_PATH"}) setenv(key, "", 1);
        setenv("DEFAULT_MODEL", "queue-fixture", 1);
        auto probes = native_resource_probes();
        probes.gpu_count = [] { return 0; };
        probes.total_per_gpu = probes.free_per_gpu = [] { return std::vector<double>{}; };
        probes.free_vram = [](auto) { return 0.; };
        probes.system_ram = probes.available_ram = [] { return 64.; };
        auto resources = std::make_shared<ResourceManager>(probes);
        ezl_env_config config{};
        ezl_load_config(&config);
        NativeServer server(config, std::make_unique<EchoInferenceBackend>("queue-fixture"), resources, {}, {}, {}, true);
        const double before = now();
        for (int i = 0; i < 1105; ++i) {
            auto response = server.handle_request("POST", "/v1/chat/completions", "", R"({"model":"queue-fixture","messages":[{"role":"user","content":"ok"}]})");
            check(response.status == 200, "chat failed");
        }
        const double after = now();
        auto old = server.handle_request("GET", "/v1/queue/request/ezl-1", "", "");
        check(old.status == 404 && Json::parse(old.body).at("detail") == "Request ezl-1 not found", "expired request HTTP contract");
        auto recent = server.handle_request("GET", "/v1/queue/request/ezl-1105", "", "");
        check(recent.status == 200, "recent request missing");
        const auto info = Json::parse(recent.body);
        check(info.at("status") == "completed" && info.at("completion_type") == "chat" && info.at("error").is_null(), "recent request metadata");
        const double timestamp = info.at("timestamp");
        check(timestamp >= before && timestamp <= after, "queue timestamp lost precision during JSON serialization");
        const auto status = Json::parse(server.handle_request("GET", "/v1/queue/status", "", "").body);
        check(status.at("total_processed") == 1105 && status.at("total_failed") == 0 && status.at("active_requests") == 0, "queue status accounting");
        std::cout << "queue HTTP: history expiration, status accounting and timestamp precision passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
