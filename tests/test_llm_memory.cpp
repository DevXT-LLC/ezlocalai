#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace std::chrono_literals;
static void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
static std::string read(const fs::path &path) { std::ifstream file(path); return {std::istreambuf_iterator<char>(file), {}}; }
static std::string hash(const std::string &value) {
    unsigned char bytes[32]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), bytes);
    std::ostringstream out; for (auto byte : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return out.str();
}
static void env(const char *name, const std::string &value) { setenv(name, value.c_str(), 1); }
struct Fixture {
    fs::path dir; std::string port;
    Fixture() : port(std::to_string(fixture_ports())) {
        std::string pattern = (fs::temp_directory_path() / "ezl-llm-memory-XXXXXX").string();
        const auto path = mkdtemp(pattern.data()); require(path, "temporary directory"); dir = path;
        for (const auto &entry : ezlocalai::environment_defaults) env(entry.first.data(), std::string(entry.second));
        for (const auto key : {"LLAMA_CPP_SERVER_URL", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_MMPROJ_PATH", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_DEFAULT_CONTEXT"}) unsetenv(key);
        env("EMBEDDING_ENABLED", "false"); env("LLM_SPECULATIVE_TYPE", "none"); env("GPU_LAYERS", "0");
        env("LLAMA_CPP_CONTEXT", "128"); env("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN);
        env("LLAMA_CPP_START_TIMEOUT", "2"); env("LLAMA_CPP_SERVER_PORT", port); env("EMBEDDING_FIXTURE_DIR", dir.string());
        std::ofstream(dir / "model.gguf") << "GGUF fixture"; env("LLAMA_CPP_MODEL_PATH", (dir / "model.gguf").string());
    }
    fs::path path(const std::string &prefix) const { return dir / (prefix + '-' + port); }
    ~Fixture() { fs::remove_all(dir); }
};
static void check_case(const Json &row, bool vision) {
    Fixture fixture; env("MAIN_GPU", std::to_string(row.at("main_gpu").get<int>()));
    if (vision) { std::ofstream(fixture.dir / "mmproj.gguf") << "projector"; env("LLAMA_CPP_MMPROJ_PATH", (fixture.dir / "mmproj.gguf").string()); }
    int calls = 0;
    bool restored = false;
    auto probes = ezlocalai::native_resource_probes();
    probes.gpu_count = [] { return 2; }; probes.total_per_gpu = [] { return std::vector<double>{24,24}; };
    probes.free_per_gpu = [&] {
        if (restored) return std::vector<double>(16, 24.0);
        const auto &value = row.at(calls++ % 2 ? "after" : "before");
        if (value.is_string()) throw std::runtime_error("fixture telemetry failure");
        return value.get<std::vector<double>>();
    };
    auto resources = std::make_shared<ezlocalai::ResourceManager>(probes);
    const auto type = vision ? ezlocalai::ModelType::VisionLlm : ezlocalai::ModelType::Llm;
    const std::string key = ezlocalai::model_type_name(type);
    {
        ezlocalai::LlamaCppInferenceBackend backend("model", resources);
        // The capture covers LLM construction/_load_llm_resilient. A lazy
        // request also probes admission headroom before that boundary, which
        // must not consume the fixture's before/after allocation samples.
        std::string startup_error;
        require(backend.warmup(&startup_error) && startup_error.empty(), "preload succeeds");
        require(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "startup and inference succeed");
        require(calls == row.at("calls"), "allocation probes follow source ordering");
        require(Json::parse(backend.model_lifecycle_json()).at("vram_estimates_gb") == row.at("estimates"), "source retained allocation estimate");
        const auto records = resources->loaded_models(); require(records.size() == 1 && records[0].type == type, "managed engine registered in shared tracker");
        const double expected = row.at("delta").is_null() ? ezlocalai::model_vram_estimate(type) : row.at("delta").get<double>();
        require(records[0].device == "cuda" && records[0].vram_gb == expected, "source device and allocation/fallback");
        require(Json::parse(backend.loaded_models_json()).at(key).at("vram_gb") == expected, "public resource record shares tracker");
        require(!records[0].in_use && records[0].active_requests == 0, "completed inference releases resource activity");
        auto restore = backend.unload_llms_for_media();
        require(!resources->is_model_loaded(type) && backend.loaded_models_json() == "{}", "handoff releases shared resource record");
        // A smaller later allocation must not shrink a known residency estimate.
        restored = true;
        restore(true);
        require(resources->is_model_loaded(type), "restore registers loaded model");
        require(resources->loaded_models()[0].vram_gb == (row.at("delta").is_null() ? 0.0 : expected), "restoration retains measured estimate");
    }
    require(!resources->is_model_loaded(type), "destruction unregisters owned engine");
}
static void active_request() {
    Fixture fixture;
    auto probes = ezlocalai::native_resource_probes(); probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; };
    auto resources = std::make_shared<ezlocalai::ResourceManager>(probes);
    ezlocalai::LlamaCppInferenceBackend backend("model", resources);
    auto response = std::async(std::launch::async, [&] { return backend.chat(R"({"messages":[{"role":"user","content":"block"}]})"); });
    struct Release { fs::path path; ~Release() { std::ofstream(path) << "release"; } } release{fixture.path("release")};
    for (int i=0; i<500 && !fs::exists(fixture.path("seen")); ++i) std::this_thread::sleep_for(10ms);
    const bool active = resources->get_model_active_count(ezlocalai::ModelType::Llm) == 1;
    const auto candidates = resources->get_models_to_unload(1);
    std::ofstream(release.path) << "release";
    require(response.get().find("choices") != std::string::npos, "held inference completed");
    require(active && candidates.empty(), "shared resource tracker protects active LLM from eviction");
    require(resources->get_model_active_count(ezlocalai::ModelType::Llm) == 0, "held inference released shared activity");
}
int main(int argc, char **argv) { try {
    require(argc == 3, "reference root and fixture required"); std::istringstream input(read(argv[2])); std::string line; std::getline(input,line); const auto header=Json::parse(line);
    for (const auto &[file,sha] : header.at("sources").items()) require(hash(read(fs::path(argv[1])/file)) == sha, "source hash: " + file);
    size_t count=0; while (std::getline(input,line)) if (!line.empty()) { const auto row=Json::parse(line); check_case(row,false); check_case(row,true); ++count; }
    require(count==header.at("cases"), "all allocation cases"); active_request();
    std::cout << count << " source allocation cases for text/vision, shared activity, handoff and destruction verified\n";
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; } }
