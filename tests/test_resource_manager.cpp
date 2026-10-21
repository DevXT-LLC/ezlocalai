#include "ezlocalai/resource_manager.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace ezlocalai;
namespace {
void expect(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &p) { std::ifstream in(p, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &value, char delimiter = '\t') {
    std::vector<std::string> out; size_t start = 0;
    do { const auto end = value.find(delimiter, start); out.push_back(value.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) break; start = end + 1; } while (true);
    return out;
}
std::string hex(const std::string &value) { std::ostringstream out; for (const unsigned char c : value) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &value) { unsigned char hash[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), hash); return hex(std::string(reinterpret_cast<char *>(hash), sizeof(hash))); }
std::string number(double value) { std::ostringstream out; out << std::setprecision(17) << value; return out.str(); }
std::string join(const std::vector<std::string> &values, char delimiter) { std::string out; bool first = true; for (const auto &v : values) { if (!first) out += delimiter; first = false; out += v; } return out; }
std::vector<double> numbers(const std::string &value) { std::vector<double> out; if (value != "-") for (const auto &v : split(value, ',')) out.push_back(std::stod(v)); return out; }
std::string numbers(const std::vector<double> &values) { std::vector<std::string> out; for (auto v : values) out.push_back(number(v)); return out.empty() ? "-" : join(out, ','); }
ModelType type(const std::string &value) { for (int i = 0; i < 8; ++i) if (value == model_type_name(static_cast<ModelType>(i))) return static_cast<ModelType>(i); throw std::runtime_error("Unknown type"); }
std::optional<double> optional(const std::string &value) { return value == "N" ? std::nullopt : std::optional<double>(std::stod(value)); }
std::string models(const std::vector<ModelResource> &values) {
    std::vector<std::string> out;
    for (const auto &m : values) out.push_back(join({model_type_name(m.type),hex(m.name),hex(m.device),number(m.vram_gb),std::to_string(m.in_use),std::to_string(m.active_requests),number(m.last_used)}, ','));
    return join(out, ';');
}
size_t reference_cases(const fs::path &root, const fs::path &fixture) {
    std::istringstream input(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); expect(sha256(read(root / line.substr(2, marker - 2))) == line.substr(marker + 8), "ResourceManager reference changed"); continue; }
        ++count; const auto c = split(line); std::vector<std::string> events, results; size_t index = 0; double clock = 0;
        std::vector<std::vector<double>> frees; for (const auto &value : split(c[2], '/')) frees.push_back(numbers(value));
        auto probes = native_resource_probes();
        probes.gpu_count = [&] { events.push_back("count"); return std::stoi(c[0]); };
        probes.total_per_gpu = [&] { events.push_back("total"); return numbers(c[1]); };
        probes.system_ram = [&] { events.push_back("ram"); return 127.25; };
        probes.free_per_gpu = [&] { events.push_back("free"); return frees.at(std::min(index++, frees.size() - 1)); };
        probes.primary_gpu = [&] { events.push_back("primary"); return 0; };
        probes.secondary_gpu = [&]() -> std::optional<int> { events.push_back("secondary"); return c[3] == "N" ? std::nullopt : std::optional<int>(std::stoi(c[3])); };
        probes.fallback_server = [&] { events.push_back("fallback"); return unhex(c[4]); };
        probes.now = [&] { events.push_back("clock"); return clock; };
        probes.allocator_stats = [&] { events.push_back("stats"); return std::vector<CudaAllocatorStats>{}; };
        ResourceManager manager(probes);
        for (const auto &op : split(unhex(c[5]), '\n')) {
            const auto f = split(op); std::string value = "ok";
            try {
                if (f[0] == "t") { clock = std::stod(f[1]); continue; }
                else if (f[0] == "r") manager.register_model(type(f[1]), unhex(f[2]), unhex(f[3]), optional(f[4]));
                else if (f[0] == "m") manager.mark_model_in_use(type(f[1]), f[2] == "1");
                else if (f[0] == "u") manager.unregister_model(type(f[1]));
                else if (f[0] == "c") { const auto a = manager.can_load_model(type(f[1]), optional(f[2])); value = std::to_string(a.can_load) + ',' + a.device + ',' + hex(a.reason); }
                else if (f[0] == "q") {
                    std::vector<ModelType> exclude; if (f[2] != "-") for (const auto &v : split(f[2], ',')) exclude.push_back(type(v));
                    std::vector<std::string> selected; for (auto mt : manager.get_models_to_unload(std::stod(f[1]), exclude)) selected.push_back(model_type_name(mt)); value = join(selected, ',');
                } else if (f[0] == "l") value = models(manager.loaded_models());
                else if (f[0] == "i") { const auto mt = type(f[1]); value = join({std::to_string(manager.is_model_loaded(mt)),std::to_string(manager.is_model_in_use(mt)),std::to_string(manager.get_model_active_count(mt))}, ','); }
                else if (f[0] == "s") {
                    const auto s = manager.get_status(); expect(s.torch_cuda.empty(), "fixture allocator stats");
                    value = join({std::to_string(s.gpu_count),number(s.total_vram_gb),number(s.free_vram_gb),number(s.system_ram_gb)}, ',')+'|'+numbers(s.free_per_gpu_gb)+'|'+models(s.loaded_models);
                } else throw std::runtime_error("Unknown operation");
            } catch (...) { value = "error"; }
            results.push_back(value);
        }
        const auto expected = split(unhex(c[6]), '\n'); expect(results.size() == expected.size(), "operation count");
        for (size_t i = 0; i < results.size(); ++i) expect(results[i] == expected[i], "resource case " + std::to_string(count) + " result " + std::to_string(i) + ": " + results[i] + " != " + expected[i]);
        expect(join(events, ',') == c[7], "probe order case " + std::to_string(count) + ": " + join(events, ',') + " != " + c[7]);
    }
    expect(count == 4449, "complete resource oracle coverage"); return count;
}
void native_driver_case() {
    for (const auto &entry : environment_defaults) unsetenv(entry.first.data());
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("DEFAULT_MODEL", "none", 1); setenv("API_KEY", "", 1);
    auto probes = native_resource_probes(); ResourceManager manager(probes);
    auto status = manager.get_status();
    expect(status.gpu_count == 2 && status.total_vram_gb == 40, "CUDA-visible cached inventory and floored capacity");
    expect(status.free_per_gpu_gb == std::vector<double>({8.125, 10.0625}) && status.free_vram_gb == 18.1875, "CUDA free memory");
    expect(status.torch_cuda.size() == 2 && status.torch_cuda[0].free_gb == 8.125 && status.torch_cuda[1].free_gb == 10.062, "allocator stats Python ties-to-even rounding");
    expect(status.torch_cuda[0].total_gb == 16.5 && status.torch_cuda[1].total_gb == 24.75, "allocator unrounded capacity basis");
    expect(status.torch_cuda[0].allocated_gb == 0 && status.torch_cuda[1].reserved_gb == 0 && status.torch_cuda[1].max_allocated_gb == 0, "no in-process PyTorch allocations");
    expect(manager.can_load_model(ModelType::Embedding).device == "cuda:0", "weaker GPU chosen even with less free memory");
    expect(manager.can_load_model(ModelType::Llm).device == "cuda:1", "LLM chooses most free memory");
    expect(manager.get_free_vram() == 18.1875 && manager.get_free_vram(1) == 10.0625 && manager.get_free_vram(99) == 0, "free memory accessors");
    auto shared = std::make_shared<ResourceManager>(probes); ezl_env_config config{}; ezl_load_config(&config);
    NativeServer server(config, std::make_unique<EchoInferenceBackend>("none"), shared);
    const auto response = server.handle_request("GET", "/v1/resources", "", "");
    expect(response.status == 200, "resource route status");
    for (const auto &part : {"\"gpu_count\":2", "\"total_vram_gb\":40", "\"free_vram_gb\":18.1875", "\"free_per_gpu_gb\":[8.125,10.0625]", "\"torch_cuda\":[{\"index\":0,\"allocated_gb\":0.0", "\"free_gb\":10.062"}) expect(response.body.find(part) != std::string::npos, std::string("resource route: ") + part + " in " + response.body);
    setenv("FALLBACK_SERVER", "http://unused", 1); setenv("FALLBACK_MEMORY_THRESHOLD", "1000000000.125", 1);
    NativeServer configured_server(config, std::make_unique<EchoInferenceBackend>("none"), shared);
    const auto configured = configured_server.handle_request("GET", "/v1/resources", "", "");
    expect(configured.status == 200 && configured.body.find("Low combined memory (18.2GB VRAM + ") != std::string::npos && configured.body.find("GB < 1000000000.125GB threshold)") != std::string::npos, "fallback decision uses shared CUDA-visible memory and full threshold precision");
    unsetenv("FALLBACK_SERVER"); unsetenv("FALLBACK_MEMORY_THRESHOLD");
    setenv("EZLOCALAI_TEST_CUDA_FAIL", "free", 1);
    bool threw = false; try { manager.can_load_model(ModelType::Embedding); } catch (...) { threw = true; }
    expect(threw, "admission propagates memory-probe errors");
    const auto disabled = server.handle_request("GET", "/v1/fallback/status", "", "");
    expect(disabled.status == 200 && disabled.body.find("No fallback server configured") != std::string::npos, "disabled fallback skips CUDA probes");
    for (const auto &stats : probes.allocator_stats()) expect(stats.error == "CUDA memory statistics unavailable", "per-device stats errors");
    unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    status = manager.get_status(); expect(status.gpu_count == 2 && status.total_vram_gb == 40 && status.free_vram_gb == 0 && status.torch_cuda.empty(), "cached capacity but live memory after topology change");
    ResourceManager hidden; expect(hidden.get_status().gpu_count == 0 && hidden.get_status().free_per_gpu_gb.empty(), "CUDA hidden at construction");
}
void concurrent_activity() {
    auto probes = native_resource_probes(); probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; }; probes.system_ram = [] { return 1; };
    ResourceManager manager(probes); manager.register_model(ModelType::Embedding, "model", "cuda", 2);
    std::atomic<int> ready{0}; std::atomic<bool> release{false}; std::vector<std::thread> workers;
    for (int i = 0; i < 12; ++i) workers.emplace_back([&] { manager.mark_model_in_use(ModelType::Embedding); ++ready; while (!release) std::this_thread::yield(); manager.mark_model_in_use(ModelType::Embedding, false); });
    while (ready != 12) std::this_thread::yield();
    const bool counted = manager.get_model_active_count(ModelType::Embedding) == 12;
    manager.register_model(ModelType::Embedding, "reloaded aggregate", "cuda", 4);
    const bool protected_model = manager.get_models_to_unload(1).empty() && manager.get_model_active_count(ModelType::Embedding) == 12;
    release = true; for (auto &worker : workers) worker.join();
    expect(counted && protected_model && !manager.is_model_in_use(ModelType::Embedding), "concurrent nested activity and registration");
    expect(manager.get_models_to_unload(1) == std::vector<ModelType>{ModelType::Embedding}, "idle after all completions");
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--cuda") { native_driver_case(); return 0; }
        expect(argc == 3, "reference and fixture required");
        const auto root = fs::absolute(argv[1]), fixture = fs::absolute(argv[2]), self = fs::absolute(argv[0]);
        const auto count = reference_cases(root, fixture); concurrent_activity();
        const auto temporary = fs::temp_directory_path() / ("ezlocalai-resources-" + std::to_string(getpid())); fs::create_directories(temporary);
        struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{temporary};
        std::ofstream(temporary / "devices.tsv") << "Fixture A\t8\t6\t17716740096\t8724152320\nFixture B\t12\t0\t26575110144\t10804527104\n";
        setenv("EZLOCALAI_TEST_CUDA_DEVICES", (temporary / "devices.tsv").c_str(), 1); unsetenv("CUDA_VISIBLE_DEVICES"); unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
        setenv("LD_LIBRARY_PATH", TEST_CUDA_DRIVER_DIRECTORY, 1);
        const auto result = run_process_to_files({self.string(), "--cuda"}, (temporary / "child.log").string(), "", 15, true);
        expect(!result.timed_out && result.status == 0, "native resource driver/API: " + read(temporary / "child.log"));
        std::cout << "Verified " << count << " source resource cases, concurrent activity, native CUDA probes and resource API\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
