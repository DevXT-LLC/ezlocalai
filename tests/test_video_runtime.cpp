#include "ezlocalai/video_runtime.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/wan_video.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>

namespace fs = std::filesystem;
namespace {
std::vector<std::string> restoration_cases;
std::vector<std::string> auxiliary_api_cases;
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream file(path, std::ios::binary); expect(bool(file), "Cannot read " + path.string()); return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &value, char separator = '\t') {
    std::vector<std::string> out; size_t start = 0;
    do { const auto end = value.find(separator, start); out.push_back(value.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) break; start = end + 1; } while (true);
    return out;
}
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &value) {
    unsigned char hash[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), hash);
    std::ostringstream out; for (auto byte : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return out.str();
}
std::optional<int> optional_gpu(const std::string &value) { const int number = std::stoi(value); return number < 0 ? std::nullopt : std::optional<int>(number); }
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("ROUTER_DRAIN_FILE", "", 1); setenv("USAGE_FILE", "usage.json", 1);
}
void set_environment(const std::string &values) {
    if (values == "-") return;
    for (const auto &item : split(values, ';')) { const auto equals = item.find('='); setenv(item.substr(0, equals).c_str(), unhex(item.substr(equals + 1)).c_str(), 1); }
}
std::string mask_text(const ezlocalai::VideoAuxiliaryMask &values) {
    std::string result; for (const bool value : values) result += value ? '1' : '0'; return result;
}
size_t auxiliary_reference_cases(const fs::path &root, const fs::path &fixture) {
    std::istringstream lines(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); if (marker != std::string::npos) expect(sha256(read(root / line.substr(2, marker - 2))) == line.substr(marker + 8), "Auxiliary reference changed"); continue; }
        ++count; const auto c = split(line); environment(); set_environment(c[1]);
        if (c[0] == "unload") {
            std::array<ezlocalai::AuxiliaryResidency, 4> models{}; const auto active = split(c[3], ',');
            for (size_t i = 0; i < models.size(); ++i) models[i] = {c[2][i] == '1', std::stoi(active[i])};
            expect(mask_text(ezlocalai::video_auxiliary_unload(models, c[4] == "1")) == c[5], "auxiliary unload policy: " + line);
        } else if (c[0] == "restore") {
            std::vector<std::string> sources; if (c[2] != "-") for (const auto &source : split(c[2], ',')) sources.push_back(unhex(source));
            ezlocalai::VideoAuxiliaryMask unloaded{}; for (size_t i = 0; i < unloaded.size(); ++i) unloaded[i] = c[3][i] == '1';
            expect(mask_text(ezlocalai::video_auxiliary_restore(unloaded, sources, c[4] == "1")) == c[5], "auxiliary restore policy: " + line);
        } else if (c[0] == "api") auxiliary_api_cases.push_back(line);
        else throw std::runtime_error("Unknown auxiliary case");
    }
    expect(count == 3057, "all auxiliary fixture cases"); return count;
}
size_t reference_cases(const fs::path &root, const fs::path &fixture) {
    std::istringstream lines(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); if (marker != std::string::npos) expect(sha256(read(root / line.substr(2, marker - 2))) == line.substr(marker + 8), "Video runtime reference changed"); continue; }
        ++count; const auto c = split(line);
        if (c[0] == "score") {
            const auto score = ezlocalai::gpu_generation_score(unhex(c[1]));
            expect(score == std::array<int64_t, 3>{std::stoll(c[2]), std::stoll(c[3]), std::stoll(c[4])}, "GPU name score: " + unhex(c[1]));
        } else if (c[0] == "ranking") {
            std::vector<ezlocalai::CudaDevice> devices;
            if (!c[1].empty()) for (const auto &row : split(unhex(c[1]), '\n')) {
                const auto p = split(row); devices.push_back({static_cast<int>(devices.size()), unhex(p[0]), std::stoi(p[1]), std::stoi(p[2]), std::stoull(p[3])});
            }
            const auto ranked = ezlocalai::rank_cuda_devices(devices);
            const auto ordinals = split(c[2], ','), scores = split(c[3], ',');
            expect(ranked.size() == devices.size(), "rank count");
            for (size_t i = 0; i < ranked.size(); ++i) expect(ranked[i].device.ordinal == std::stoi(ordinals[i]) && ranked[i].score == std::stod(scores[i]), "rank ordinal and exact capability score");
            expect((ranked.empty() ? 0 : ranked.front().device.ordinal) == std::stoi(c[4]), "primary GPU");
            expect(ezlocalai::secondary_cuda_device(devices) == optional_gpu(c[5]), "secondary GPU");
        } else if (c[0] == "placement") {
            expect(ezlocalai::video_device(optional_gpu(c[1]), c[2] == "1", c[3] == "1", c[4]) == c[5], "video placement");
        } else if (c[0] == "unload") {
            environment();
            for (const auto &item : split(c[1], ';')) { const auto equals = item.find('='); setenv(item.substr(0, equals).c_str(), unhex(item.substr(equals + 1)).c_str(), 1); }
            expect(ezlocalai::video_should_unload_llm(c[2] == "1", optional_gpu(c[3])) == (c[4] == "1"), "video unload policy");
        } else if (c[0] == "reload") {
            environment(); setenv("VIDEO_RELOAD_LLM_AFTER_GENERATION", unhex(c[1]).c_str(), 1);
            expect(ezlocalai::video_should_reload_llm() == (c[2] == "1"), "video initial reload policy");
        } else if (c[0] == "restore") {
            restoration_cases.push_back(line);
        } else throw std::runtime_error("Unknown runtime case");
    }
    expect(count == 2769, "all runtime fixture cases"); return count;
}
void probe() {
    for (const auto &item : ezlocalai::rank_cuda_devices(ezlocalai::visible_cuda_devices())) {
        const auto &device = item.device;
        std::cout << device.ordinal << '\t' << device.name << '\t' << device.major << '\t' << device.minor << '\t' << device.total_memory << '\t' << std::setprecision(17) << item.score << '\n';
    }
}
class Backend : public ezlocalai::InferenceBackend {
public:
    bool loaded = true; int unloads = 0, restores = 0, loads = 0;
    std::string outcome = "success", actions;
    ezlocalai::VideoAuxiliaryMask auxiliary{};
    int auxiliary_active = 0;
    std::string auxiliary_actions;
    bool check_auxiliary_order = false;
    bool fail_auxiliary_restore = false;
    std::string chat(const std::string &) override { return "{}"; }
    std::string completion(const std::string &) override { return "{}"; }
    std::string embedding(const std::string &) override { return "{}"; }
    std::string loaded_models_json() const override {
        std::string result = loaded ? R"({"llm":{"name":"fixture"})" : "{";
        for (size_t i = 0; i < auxiliary.size(); ++i) if (auxiliary[i]) {
            if (result.size() > 1) result += ',';
            result += '"' + std::string(ezlocalai::video_auxiliary_models[i]) + "\":{\"active_requests\":" + std::to_string(auxiliary_active) + "}";
        }
        return result + '}';
    }
    bool unload_aux_model(const std::string &model) override {
        if (check_auxiliary_order) expect(!loaded, "LLM unload precedes auxiliary unload");
        for (size_t i = 0; i < auxiliary.size(); ++i) if (model == ezlocalai::video_auxiliary_models[i]) {
            expect(auxiliary[i] && auxiliary_active == 0, "only idle resident auxiliary models unload");
            auxiliary[i] = false;
            if (!auxiliary_actions.empty()) auxiliary_actions += ',';
            auxiliary_actions += "unload:" + model; return true;
        }
        return false;
    }
    void warm_aux_model(const std::string &model) override {
        if (check_auxiliary_order) expect(loaded, "LLM restore precedes auxiliary restore");
        if (!auxiliary_actions.empty()) auxiliary_actions += ',';
        auxiliary_actions += "restore:" + model;
        if (fail_auxiliary_restore) throw std::runtime_error("fixture auxiliary warm-load failure");
        for (size_t i = 0; i < auxiliary.size(); ++i) if (model == ezlocalai::video_auxiliary_models[i]) auxiliary[i] = true;
    }
    std::function<void(bool)> unload_llms_for_video() override {
        const bool prior = loaded; ++unloads; loaded = false;
        return [this, prior, attempt = 0](bool reload) mutable {
            ++restores; ++attempt;
            if (!reload || !prior) return;
            ++loads; if (!actions.empty()) actions += ',';
            actions += attempt == 1 ? "runtime" : "configured";
            if (outcome == "throw_first" && loads == 1) throw std::runtime_error("fixture restoration exception");
            if (outcome == "fail_all" || (outcome == "fail_first" && loads == 1)) return;
            loaded = true;
        };
    }
};
void api_child(int ordinal, bool should_unload, const std::vector<std::string> &restoration = {}, const std::vector<std::string> &auxiliary = {}) {
    environment(); setenv("DEFAULT_MODEL", "fixture", 1); setenv("VIDEO_ENABLED", "true", 1); setenv("VIDEO_MODEL", "wan2.2", 1);
    setenv("SDCPP_BIN", TEST_WAN_FIXTURE_BIN, 1); setenv("HF_HUB_OFFLINE", "1", 1); setenv("WAN_FIXTURE_ARGUMENTS", "sd-argv", 1);
    setenv("HF_HUB_CACHE", (fs::current_path() / "models").c_str(), 1);
    for (const auto &asset : ezlocalai::wan_model_assets(false, "Q4_K_M")) {
        auto repository = asset.repository; size_t pos = 0; while ((pos = repository.find('/', pos)) != std::string::npos) { repository.replace(pos, 1, "--"); pos += 2; }
        const auto path = fs::path("models") / ("models--" + repository) / "snapshots" / asset.revision / asset.filename;
        fs::create_directories(path.parent_path()); std::ofstream(path) << "fixture model";
    }
    fs::remove("sd-argv");
    if (!auxiliary.empty()) set_environment(auxiliary[1]);
    ezl_env_config config{}; ezl_load_config(&config);
    auto backend = std::make_unique<Backend>(); auto *tracked = backend.get();
    const bool invalid_video = (!restoration.empty() && restoration[4] == "1") || (!auxiliary.empty() && auxiliary[4] == "1");
    if (!auxiliary.empty()) { tracked->auxiliary.fill(true); tracked->auxiliary_active = std::stoi(auxiliary[3]); tracked->check_auxiliary_order = auxiliary[2] == "1"; }
    if (!restoration.empty()) {
        setenv("VIDEO_UNLOAD_LLM_DURING_GENERATION", "true", 1);
        setenv("VIDEO_RELOAD_LLM_AFTER_GENERATION", unhex(restoration[1]).c_str(), 1);
        tracked->loaded = restoration[2] == "1"; tracked->outcome = restoration[3];
    }
    ezlocalai::NativeServer server(config, std::move(backend));
    const auto response = server.handle_request("POST", "/v1/videos/generations", "", invalid_video
        ? R"({"prompt":null,"size":"64x64","num_frames":5})" : R"({"prompt":"test","size":"64x64","num_frames":5})");
    if (!auxiliary.empty()) {
        expect(response.status == (invalid_video ? 500 : 200), "auxiliary generation response: " + response.body);
        expect((tracked->auxiliary_actions.empty() ? "-" : tracked->auxiliary_actions) == auxiliary[5], "source-derived auxiliary sequence: " + tracked->auxiliary_actions);
        expect(tracked->unloads == std::stoi(auxiliary[2]) && tracked->restores == std::stoi(auxiliary[2]), "auxiliary and LLM policy independence");
        if (!invalid_video && tracked->auxiliary_active == 0 && auxiliary[5].find("restore:") != std::string::npos) {
            tracked->auxiliary.fill(true); tracked->auxiliary_actions.clear(); tracked->fail_auxiliary_restore = true;
            const auto failed_restore = server.handle_request("POST", "/v1/videos/generations", "", R"({"prompt":"test","size":"64x64","num_frames":5})");
            expect(failed_restore.status == 200 && tracked->auxiliary_actions == auxiliary[5], "failed auxiliary warm loads preserve video and continue restoring other models");
        }
        return;
    }
    if (!restoration.empty()) {
        expect(response.status == std::stoi(restoration[8]), "restoration response: " + response.body);
        expect(tracked->unloads == 1 && tracked->restores == std::stoi(restoration[5]), "reference restoration attempt count");
        expect((tracked->actions.empty() ? "-" : tracked->actions) == restoration[6], "reference runtime/configured restoration sequence");
        expect(tracked->loaded == (restoration[7] == "1"), "reference final LLM residency");
        return;
    }
    expect(response.status == 200, "video generation with driver inventory: " + response.body);
    expect(read("sd-argv").find("\ncuda" + std::to_string(ordinal) + "\n") != std::string::npos, "sd-cli device follows CUDA capability ranking");
    expect(tracked->unloads == int(should_unload) && tracked->restores == int(should_unload) && tracked->loaded, "auto handoff follows visible GPU count");
    if (!should_unload) {
        const auto resources = server.handle_request("GET", "/v1/resources", "", "");
        expect(resources.body.find("\"device\":\"cuda:" + std::to_string(ordinal) + "\"") != std::string::npos, "resident video metadata retains selected ordinal");
    }
}
void driver_cases(const std::string &self) {
    environment();
    const auto previous_path = std::getenv("LD_LIBRARY_PATH"); const std::string old_path = previous_path ? previous_path : "";
    setenv("LD_LIBRARY_PATH", (std::string(TEST_CUDA_DRIVER_DIRECTORY) + (old_path.empty() ? "" : ":" + old_path)).c_str(), 1);
    setenv("EZLOCALAI_TEST_CUDA_DEVICES", (fs::current_path() / "devices.tsv").c_str(), 1);
    std::ofstream("devices.tsv") << "NVIDIA GeForce RTX 3090\t8\t6\t25769803776\nNVIDIA GeForce RTX 5090\t12\t0\t34359738368\n";
    auto child = [](const std::vector<std::string> &arguments) {
        const auto execution = ezlocalai::run_process_to_files(arguments, "child.log", "", 10, true);
        expect(!execution.timed_out, "runtime child timed out");
        return ezlocalai::ProcessResult{execution.status, read("child.log")};
    };
    unsetenv("CUDA_VISIBLE_DEVICES"); unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    auto result = child({self, "--probe"});
    expect(result.status == 0 && result.output.rfind("1\tNVIDIA GeForce RTX 5090\t12\t0\t34359738368", 0) == 0, "driver query and rank");
    result = child({self, "--geometry", "0"});
    expect(result.status == 0 && result.output == "80 1536\n", "selected CUDA device geometry");
    result = child({self, "--api", "0", "0"}); expect(result.status == 0, "secondary ordinal zero API: " + result.output);
    setenv("CUDA_VISIBLE_DEVICES", "1,0", 1);
    result = child({self, "--geometry", "0"});
    expect(result.status == 0 && result.output == "120 1536\n", "geometry follows remapped CUDA ordinals");
    result = child({self, "--api", "1", "0"}); expect(result.status == 0, "reordered visibility API: " + result.output);
    setenv("CUDA_VISIBLE_DEVICES", "1", 1);
    result = child({self, "--api", "0", "1"}); expect(result.status == 0, "single visible GPU API: " + result.output);
    for (const auto &row : restoration_cases) {
        result = child({self, "--restore", row});
        expect(result.status == 0, "source-derived restoration case " + row + ": " + result.output);
    }
    for (const auto &row : auxiliary_api_cases) {
        result = child({self, "--aux", row});
        expect(result.status == 0, "source-derived auxiliary API " + row + ": " + result.output);
    }
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    result = child({self, "--probe"}); expect(result.status == 0 && result.output.empty(), "no visible CUDA devices");
    unsetenv("CUDA_VISIBLE_DEVICES");
    for (const char *operation : {"init", "count", "get", "name", "compute", "memory"}) {
        setenv("EZLOCALAI_TEST_CUDA_FAIL", operation, 1);
        result = child({self, "--probe"});
        const bool availability = std::string(operation) == "init" || std::string(operation) == "count";
        expect(availability ? result.status == 0 && result.output.empty() : result.status != 0 && result.output.find("CUDA error") != std::string::npos, std::string("driver failure: ") + operation);
    }
    unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    setenv("EZLOCALAI_TEST_CUDA_FAIL", "attribute", 1);
    result = child({self, "--geometry", "0"});
    expect(result.status != 0 && result.output.find("CUDA error") != std::string::npos, "attribute errors propagate");
    unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    if (previous_path) setenv("LD_LIBRARY_PATH", old_path.c_str(), 1); else unsetenv("LD_LIBRARY_PATH");
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--probe") { probe(); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--geometry") {
            const auto geometry = ezlocalai::cuda_multiprocessor_geometry(std::stoi(argv[2]));
            std::cout << geometry.first << ' ' << geometry.second << '\n';
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--restore") { api_child(0, true, split(argv[2])); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--aux") { api_child(0, true, {}, split(argv[2])); return 0; }
        if (argc == 4 && std::string(argv[1]) == "--api") { api_child(std::stoi(argv[2]), std::string(argv[3]) == "1"); return 0; }
        expect(argc == 4, "reference and fixtures required");
        const auto self = fs::absolute(argv[0]).string(), root = fs::absolute(argv[1]).string(), fixture = fs::absolute(argv[2]).string(), auxiliary_fixture = fs::absolute(argv[3]).string();
        const auto previous = fs::current_path(); char path[] = "/tmp/ezlocalai-video-runtime-XXXXXX";
        expect(mkdtemp(path), "temporary directory"); fs::current_path(path);
        const auto count = reference_cases(root, fixture), auxiliary_count = auxiliary_reference_cases(root, auxiliary_fixture); driver_cases(self);
        fs::current_path(previous); fs::remove_all(path);
        std::cout << "Verified " << count << " reference GPU/placement/policy cases, " << auxiliary_count << " auxiliary lifecycle cases and CUDA driver/API integration\n"; return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
