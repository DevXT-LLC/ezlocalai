#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::vector<std::string> events;
class Model final : public ezlocalai::WakewordModel {
    double score_;
public:
    explicit Model(double score) : score_(score) {}
    std::pair<bool, double> predict(const std::string &audio) override {
        std::string hex; const char *digits = "0123456789abcdef";
        for (unsigned char c : audio) { hex += digits[c >> 4]; hex += digits[c & 15]; }
        events.push_back("predict:" + hex);
        if (audio == "error") throw std::runtime_error("invalid audio");
        const auto score = audio == "nan" ? std::numeric_limits<double>::quiet_NaN() : score_;
        return {score > .5, score};
    }
};
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "Cache fixture required");
        std::ifstream input(argv[1]); Json fixture; input >> fixture;
        const auto previous = fs::current_path();
        char pattern[] = "/tmp/ezlocalai-wakeword-cache-XXXXXX"; const char *path = mkdtemp(pattern); check(path, "temporary directory");
        const fs::path root(path); fs::current_path(root);
        struct Cleanup { fs::path previous, root; ~Cleanup() { fs::current_path(previous); fs::remove_all(root); } } cleanup{previous, root};
        for (const auto &entry : ezlocalai::environment_defaults) unsetenv(std::string(entry.first).c_str());
        setenv("EZLOCALAI_API_KEY", "", 1); setenv("DEFAULT_MODEL", "model", 1); setenv("VOICE_SERVER", "false", 1);
        for (const auto *name : {"EZLOCALAI_PRELOAD_MODEL_ON_START", "EZLOCALAI_MODEL_STATUS_ON_START", "TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED"}) setenv(name, "false", 1);
        ezl_env_config config{}; ezl_load_config(&config);
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
        server.set_wakeword_factory([](const std::string &directory) {
            events.push_back("load:" + fs::path(directory).filename().string());
            std::ifstream file(fs::path(directory) / "model.pt"); std::string score; file >> score;
            return std::make_shared<Model>(std::stod(score));
        });
        setenv("STT_ENABLED", "true", 1);
        size_t index = 0;
        for (const auto &row : fixture.at("steps")) {
            events.clear(); const auto label = "step " + std::to_string(index++);
            const auto action = row.at("action").get<std::string>();
            if (action == "write") {
                const auto file = root / "wakeword_models" / row.at("name").get<std::string>(); fs::create_directories(file.parent_path());
                std::ofstream out(file); out << row.at("value").get<std::string>();
            } else if (action == "remove") fs::remove(root / "wakeword_models" / row.at("name").get<std::string>());
            else {
                const auto response = server.handle_request(row.at("method"), row.at("path"), "", row.at("body"), row.at("content_type"));
                check(response.status == row.at("status"), label + " status: " + std::to_string(response.status));
                const auto expected = row.at("response").get<std::string>();
                if (Json::accept(expected)) check(Json::parse(response.body) == Json::parse(expected), label + " JSON: " + response.body);
                else check(response.body == expected, label + " body");
            }
            check(Json(events) == row.at("events"), label + " cache/load event order: " + Json(events).dump());
        }
        std::cout << index << " Python manager cache and HTTP steps matched\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
