#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/qwen_image.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace ezlocalai;
using Json = nlohmann::json;
namespace fs = std::filesystem;
namespace {
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
void env(const char *key, const std::string &value) { setenv(key, value.c_str(), 1); }
struct Backend : InferenceBackend {
    bool resident, warmed = false;
    std::shared_ptr<ResourceManager> resources;
    Backend(bool loaded, std::shared_ptr<ResourceManager> r) : resident(loaded), resources(std::move(r)) {}
    std::string chat(const std::string &) override { return "{}"; }
    std::string completion(const std::string &) override { return "{}"; }
    std::string embedding(const std::string &) override { return "{}"; }
    std::string loaded_models_json() const override { return resident ? R"({"llm":{"name":"fixture"}})" : "{}"; }
    void warm_aux_model(const std::string &kind) override {
        check(kind == "embedding", "embedding startup stage");
        check(resources->loaded_models().empty(), "media must follow embedding preload");
        warmed = true;
    }
};
void scenario(const Json &row, const fs::path &directory, const fs::path &binary) {
    for (const auto &[key, value] : environment_defaults) unsetenv(std::string(key).c_str());
    for (const auto *key : {"CUDA_VISIBLE_DEVICES", "EZLOCALAI_TEST_CUDA_FAIL"}) unsetenv(key);
    env("EZLOCALAI_DOTENV_PATH", ""); env("EZLOCALAI_API_KEY", ""); env("DEFAULT_MODEL", "none");
    env("TTS_ENABLED", "false"); env("STT_ENABLED", "false"); env("EMBEDDING_ENABLED", "true"); env("EMBEDDING_KEEP_LOADED", "true");
    env("EZLOCALAI_PRELOAD_MODEL_ON_START", "false"); env("SDCPP_MODELS_DIR", (directory / "models").string());
    env("SDCPP_BIN", row.at("missing_binary").get<bool>() ? (directory / "missing").string() : binary.string());
    for (const auto &[key, value] : row.at("environment").items()) env(key.c_str(), value.get<std::string>());
    const bool secondary = !row.at("secondary").is_null();
    { std::ofstream devices(directory / "devices");
      if (secondary) {
          const bool zero = row.at("secondary") == 0;
          devices << (zero ? "NVIDIA RTX 3080" : "NVIDIA RTX 4090") << "\t8\t6\t25769803776\n";
          devices << (zero ? "NVIDIA RTX 4090" : "NVIDIA RTX 3080") << "\t8\t6\t25769803776\n";
      } else devices << "NVIDIA RTX 3090\t8\t6\t25769803776\n";
    }
    env("EZLOCALAI_TEST_CUDA_DEVICES", (directory / "devices").string());
    auto probes = native_resource_probes(); const double free = row.at("free_gb");
    probes.gpu_count = [=] { return secondary ? 2 : 1; };
    probes.total_per_gpu = [=] { return std::vector<double>(secondary ? 2 : 1, 24); };
    probes.free_per_gpu = [=] { return std::vector<double>(secondary ? 2 : 1, free / (secondary ? 2 : 1)); };
    probes.free_vram = [=](auto) { return free; };
    probes.system_ram = probes.available_ram = [] { return 64.; };
    auto resources = std::make_shared<ResourceManager>(probes);
    auto backend = std::make_unique<Backend>(row.at("resident_llm"), resources); auto *model = backend.get();
    ezl_env_config config{}; ezl_load_config(&config);
    auto server = create_native_server(config, std::move(backend), resources);
    check(model->warmed, "embedding preload must run");
    Json actual = Json::array();
    for (const auto &item : resources->loaded_models()) actual.push_back({{"type", model_type_name(item.type)}, {"name", item.name}, {"device", item.device}, {"vram_gb", item.vram_gb}});
    check(actual == row.at("registered"), row.at("name").get<std::string>() + ": " + actual.dump() + " expected " + row.at("registered").dump());
    if (row.at("name") == "missing-QuantStack/Wan2.2-T2V-A14B-GGUF") {
        // A null frame count reaches the generation scope, then fails before
        // downloading weights. Exercise retry and finally cleanup there.
        const std::string request = R"({"prompt":"fixture","num_frames":null})";
        auto reply = server->handle_request("POST", "/v1/videos/generations", "", request);
        check(reply.status == 500 && !resources->is_model_loaded(ModelType::Video), "missing Wan binary remains retryable");
        env("SDCPP_BIN", binary.string());
        reply = server->handle_request("POST", "/v1/videos/generations", "", request);
        check(reply.status == 400 && resources->is_model_loaded(ModelType::Video), "Wan lazy retry registers wrapper after binary repair");
        env("VIDEO_UNLOAD_LLM_DURING_GENERATION", "true");
        reply = server->handle_request("POST", "/v1/videos/generations", "", request);
        check(reply.status == 400 && !resources->is_model_loaded(ModelType::Video), "handoff removes Wan registration on generation error");
    }
    check(!fs::exists(directory / "engine-launched"), "startup must not launch a media engine");
}
}
int main(int argc, char **argv) {
    if (argc > 2) { std::ofstream("engine-launched") << "unexpected"; return 99; }
    const auto old = fs::current_path(); fs::path directory;
    try {
        check(argc == 2, "media startup fixture required"); std::ifstream file(argv[1]); Json fixture; file >> fixture;
        const auto binary = fs::canonical("/proc/self/exe");
        directory = fs::temp_directory_path() / ("ezlocalai-media-startup-" + std::to_string(getpid()));
        fs::create_directories(directory); fs::current_path(directory);
        for (const auto &asset : qwen_image_assets()) { const auto path = directory / "models" / asset.filename; fs::create_directories(path.parent_path()); std::ofstream(path) << "cached fixture"; }
        for (const auto &row : fixture.at("cases")) scenario(row, directory, binary);
        fs::current_path(old); fs::remove_all(directory);
        std::cout << fixture.at("cases").size() << " Python media startup cases, embedding-before-media order, resource estimates and no engine launch passed\n";
        return 0;
    } catch (const std::exception &error) { fs::current_path(old); std::cerr << error.what() << "\nArtifacts: " << directory << '\n'; return 1; }
}
