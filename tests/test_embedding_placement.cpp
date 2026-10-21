#include "ezlocalai/embedding_placement.hpp"
#include "ezlocalai/video_runtime.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include <arpa/inet.h>
#include <cmath>
#include <csignal>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &value, char delimiter = '\t') {
    std::vector<std::string> out; size_t start = 0;
    do { const auto end = value.find(delimiter, start); out.push_back(value.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) break; start = end + 1; } while (true);
    return out;
}
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &value) {
    unsigned char hash[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), hash);
    std::ostringstream out; for (const auto c : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str();
}
std::string join(const std::vector<std::string> &values, char separator = '\n') { std::string out; for (const auto &value : values) { if (!out.empty()) out += separator; out += value; } return out; }
std::string integers(const std::vector<int> &values) { std::vector<std::string> text; for (const auto value : values) text.push_back(std::to_string(value)); return join(text, ','); }
void near(double actual, double expected, const std::string &message) { expect(std::abs(actual - expected) <= 1e-12 * std::max(1.0, std::abs(expected)), message + ": " + std::to_string(actual) + " != " + std::to_string(expected)); }
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const auto *key : {"LLAMA_CPP_SERVER_URL", "LLAMA_CPP_EMBEDDING_SERVER_URL", "LLAMA_CPP_CONTEXT", "LLAMA_CPP_EMBEDDING_MODEL_PATH", "LLAMA_CPP_EMBEDDING_UBATCH", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_EMBEDDING_EXTRA_ARGS", "EMBEDDING_FIXTURE_MAX_LAYERS", "EMBEDDING_FIXTURE_FAILURE", "EMBEDDING_FIXTURE_TIMEOUT", "EMBEDDING_FIXTURE_WARMUP_ERROR", "EMBEDDING_FIXTURE_FAIL_PORT", "EZLOCALAI_TEST_CUDA_FAIL"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("DEFAULT_MODEL", "none", 1); setenv("EMBEDDING_N_PARALLEL", "1", 1);
}
size_t reference_cases(const fs::path &root, const fs::path &fixture) {
    std::istringstream input(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); if (marker != std::string::npos) expect(sha256(read(root / line.substr(2, marker - 2))) == line.substr(marker + 8), "Embedding.py reference changed"); continue; }
        ++count; const auto c = split(line); const auto label = "reference case " + std::to_string(count);
        if (c[0] == "metadata") {
            const auto path = fixture.parent_path() / "embedding_gguf" / c[1]; expect(sha256(read(path)) == c[2], "GGUF fixture changed: " + c[1]);
            const auto m = ezlocalai::embedding_model_metadata(path.string());
            expect(m.layers == std::stoll(c[3]) && m.kv_heads == std::stoll(c[4]) && m.key_length == std::stoll(c[5]) && m.value_length == std::stoll(c[6]), "metadata " + c[1] + ": " + std::to_string(m.layers) + ',' + std::to_string(m.kv_heads) + ',' + std::to_string(m.key_length) + ',' + std::to_string(m.value_length));
        } else if (c[0] == "estimate") {
            const ezlocalai::EmbeddingMetadata metadata{std::stoll(c[1]), std::stoll(c[2]), std::stoll(c[3]), std::stoll(c[4])};
            const auto m = ezlocalai::estimate_embedding_memory(metadata, std::stoull(c[5]), std::stoi(c[6]), std::stoi(c[7]), std::stoi(c[8]), unhex(c[9]));
            expect(m.layers == std::stoll(c[10]), label + " layers");
            const double values[]{m.total_gb, m.fixed_gb, m.per_layer_gb, m.weights_gb, m.kv_gb, m.compute_gb};
            for (size_t i = 0; i < 6; ++i) near(values[i], std::stod(c[11 + i]), label + " estimate " + std::to_string(i));
        } else if (c[0] == "resolve") {
            ezlocalai::EmbeddingMemory m; m.layers = std::stoll(c[1]); m.total_gb = std::stod(c[2]); m.fixed_gb = std::stod(c[3]); m.per_layer_gb = std::stod(c[4]);
            std::optional<int> requested; if (c[5] != "None") requested = std::stoi(c[5]);
            std::string actual;
            try { actual = std::to_string(ezlocalai::resolve_embedding_gpu_layers(m, requested, c[6] == "1", c[7] == "1", std::stod(c[8]), std::stod(c[9]))); } catch (const std::runtime_error &) { actual = "error"; }
            expect(actual == c[10], label + " selected " + actual + " != " + c[10]);
        } else if (c[0] == "fallback") {
            ezlocalai::EmbeddingMemory m; m.layers = std::stoll(c[1]); m.fixed_gb = std::stod(c[2]); m.per_layer_gb = std::stod(c[3]);
            const auto selected = std::stoi(c[4]);
            expect(integers(ezlocalai::embedding_fallback_layers(selected, m.layers)) == c[5], label + " retry sequence");
            near(ezlocalai::embedding_gpu_memory(m, selected), std::stod(c[6]), label + " GPU estimate");
        } else if (c[0] == "error") expect(ezlocalai::embedding_memory_error(unhex(c[1])) == (c[2] == "1"), label + " memory error");
        else if (c[0] == "warm") {
            setenv("EMBEDDING_KEEP_LOADED", unhex(c[1]).c_str(), 1); setenv("EMBEDDING_WARMUP", unhex(c[2]).c_str(), 1);
            expect(join(ezlocalai::embedding_warmup_inputs(c[3] == "1", std::stoi(c[4]), std::stoi(c[5]))) == unhex(c[6]), label + " warmup input sequence");
        } else throw std::runtime_error("Unknown reference case");
    }
    expect(count == 2731, "complete embedding placement reference coverage"); return count;
}
int free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); expect(fd >= 0, "socket");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "bind"); socklen_t size = sizeof(address);
    getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size); close(fd); return ntohs(address.sin_port);
}
void managed_case(const std::string &mode, const fs::path &model) {
    environment(); const int port = free_port(); const auto path = fs::absolute(mode); fs::create_directories(path);
    setenv("EMBEDDING_FIXTURE_DIR", path.c_str(), 1); setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1);
    setenv("LLAMA_CPP_EMBEDDING_MODEL_PATH", model.c_str(), 1); setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT", std::to_string(port).c_str(), 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1); setenv("EMBEDDING_MAIN_GPU", "1", 1); setenv("EMBEDDING_VRAM_SAFETY_MARGIN", "0.1", 1);
    std::string attempts = "-1\n"; int selected = -1; bool failed = false, warm = true;
    if (mode == "partial") { setenv("EMBEDDING_MAIN_GPU", "0", 1); setenv("CUDA_VISIBLE_DEVICES", "0", 1); selected = 4; attempts = "4\n"; }
    if (mode == "best_gpu") setenv("EMBEDDING_MAIN_GPU", "0", 1);
    if (mode == "secondary") {
        std::ofstream(path / "devices.tsv") << "Fixture A\t8\t6\t8589934592\t3221225472\nFixture B\t12\t0\t17179869184\t5368709120\n";
        setenv("EZLOCALAI_TEST_CUDA_DEVICES", (path / "devices.tsv").c_str(), 1);
    }
    if (mode == "explicit") { setenv("EMBEDDING_GPU_LAYERS", "2", 1); selected = 2; attempts = "2\n"; }
    if (mode == "cpu") { setenv("CUDA_VISIBLE_DEVICES", "", 1); setenv("EMBEDDING_GPU_LAYERS", "100", 1); selected = 0; attempts = "0\n"; warm = false; }
    if (mode == "probe_failure") { setenv("EZLOCALAI_TEST_CUDA_FAIL", "free", 1); attempts = ""; failed = true; }
    if (mode == "retry") { setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "6", 1); selected = 6; attempts = "-1\n9\n6\n"; }
    if (mode == "retry_cpu") { setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "0", 1); selected = 0; attempts = "-1\n9\n6\n3\n0\n"; warm = false; }
    if (mode == "non_memory") { setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "0", 1); setenv("EMBEDDING_FIXTURE_FAILURE", "invalid model header", 1); failed = true; }
    if (mode == "warm_failure") { setenv("EMBEDDING_FIXTURE_WARMUP_ERROR", "1", 1); failed = true; }
    if (mode == "dimensions") { setenv("EMBEDDING_DIMENSIONS", "bad", 1); failed = true; }
    if (mode == "timeout") { setenv("EMBEDDING_FIXTURE_TIMEOUT", "1", 1); failed = true; }
    if (mode == "warm_disabled") { setenv("EMBEDDING_WARMUP", "false", 1); warm = false; }
    if (mode == "transient") { setenv("EMBEDDING_KEEP_LOADED", "false", 1); warm = false; }
    {
        ezlocalai::LlamaCppInferenceBackend backend("none");
        if (mode == "transient") expect(backend.embedding("{\"input\":\"request\"}").find("embedding") != std::string::npos, "transient inference succeeds");
        else backend.warm_aux_model("embedding");
        expect(read(path / ("attempts-" + std::to_string(port))) == attempts, mode + " startup retry order: " + read(path / ("attempts-" + std::to_string(port))));
        if (failed || mode == "transient") expect(backend.loaded_models_json() == "{}", mode + " unpublished/temporary instance removed");
        else {
            const auto resources = backend.loaded_models_json();
            const auto registered = backend.resource_manager()->loaded_models();
            expect(registered.size() == 1 && registered[0].type == ezlocalai::ModelType::Embedding && registered[0].name == "Qwen/Qwen3-Embedding-0.6B-GGUF", "embedding resource uses repository name rather than served alias");
            expect(resources.find(selected ? "\"device\":\"cuda\"" : "\"device\":\"cpu\"") != std::string::npos, "published device");
            const auto marker = resources.find("\"vram_gb\":"); expect(marker != std::string::npos, "published estimate");
            const auto memory = ezlocalai::estimate_embedding_memory(model.string(), 10000, 1, 512, "f16");
            // Resource serialization rounds the API's displayed values.
            expect(std::abs(std::stod(resources.substr(marker + 10)) - ezlocalai::embedding_gpu_memory(memory, selected)) < .00001, "published memory estimate");
            const auto args = read(path / ("args-" + std::to_string(port)));
            const int expected_gpu = mode == "secondary" || mode == "partial" ? 0 : 1;
            expect(args.find("--main-gpu\n" + std::to_string(expected_gpu) + "\n") != std::string::npos, "resource recommendation precedes embedding GPU environment override");
            for (const auto *option : {"--fit\noff\n", "--cache-ram\n0\n", "--no-cache-prompt\n", "--embd-normalize\n2\n"}) expect(args.find(option) != std::string::npos, "reference engine option");
            expect(args.find(selected ? "-fa\non\n" : "-fa\noff\n") != std::string::npos, "flash attention policy");
            if (!selected) expect(args.find("--no-kv-offload\n") != std::string::npos && args.find("--no-op-offload\n") != std::string::npos, "CPU excludes operation and KV offload");
            const auto seen = read(path / ("seen-" + std::to_string(port)));
            expect((seen.find("Warm up the embedding model.") != std::string::npos) == warm, "text warmup policy");
            if (warm) {
                const auto inputs = ezlocalai::embedding_warmup_inputs(true, 10000, 512);
                expect(seen.find("\"input\":" + inputs.at(1)) != std::string::npos, "full microbatch token warmup");
            }
            expect(backend.unload_aux_model("embedding"), "resident can unload after warmup");
            expect(!backend.resource_manager()->is_model_loaded(ezlocalai::ModelType::Embedding), "pool unload unregisters resource");
        }
    }
    std::istringstream pids(read(path / ("launch-" + std::to_string(port))));
    for (int pid; pids >> pid;) expect(kill(pid, 0) != 0, mode + " owned startup process is reaped");
}
void context_case() {
    void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL); expect(library != nullptr, "fixture CUDA library");
    const auto retain = reinterpret_cast<int(*)(void **, int)>(dlsym(library, "cuDevicePrimaryCtxRetain"));
    const auto push = reinterpret_cast<int(*)(void *)>(dlsym(library, "cuCtxPushCurrent_v2"));
    const auto pop = reinterpret_cast<int(*)(void **)>(dlsym(library, "cuCtxPopCurrent_v2"));
    const auto memory = reinterpret_cast<int(*)(size_t *, size_t *)>(dlsym(library, "cuMemGetInfo_v2"));
    auto check = [&] {
        void *context = nullptr; expect(!retain(&context, 101) && !push(context), "existing caller CUDA context");
        for (int i = 0; i < 20; ++i) {
            near(ezlocalai::cuda_free_memory_gb(0), .75, "selected CUDA free memory");
            size_t free = 0, total = 0; expect(!memory(&free, &total) && free == 5ull * 1024 * 1024 * 1024, "probe restores calling thread's CUDA context");
        }
        expect(!pop(&context), "caller context popped");
    };
    auto first = std::async(std::launch::async, check), second = std::async(std::launch::async, check); first.get(); second.get(); dlclose(library);
}
void driver_cases(const std::string &self, const fs::path &model) {
    environment();
    const char *old = std::getenv("LD_LIBRARY_PATH"); const std::string saved = old ? old : "";
    setenv("LD_LIBRARY_PATH", (std::string(TEST_CUDA_DRIVER_DIRECTORY) + ':' + saved).c_str(), 1);
    setenv("EZLOCALAI_TEST_CUDA_DEVICES", (fs::current_path() / "devices.tsv").c_str(), 1);
    std::ofstream("devices.tsv") << "Fixture A\t8\t6\t8589934592\t805306368\nFixture B\t12\t0\t17179869184\t5368709120\n";
    auto child = [&](const std::vector<std::string> &args) {
        const auto result = ezlocalai::run_process_to_files(args, "child.log", "", 20, true);
        expect(!result.timed_out && result.status == 0, "embedding placement child: " + read("child.log")); return read("child.log");
    };
    unsetenv("CUDA_VISIBLE_DEVICES"); unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    expect(child({self, "--probe"}) == "2 0.75 5 0.75 0.75\n", "free memory and invalid ordinals");
    child({self, "--contexts"});
    setenv("CUDA_VISIBLE_DEVICES", "1,0", 1); expect(child({self, "--probe"}) == "2 5 0.75 5 5\n", "free memory uses visible CUDA ordinals");
    setenv("CUDA_VISIBLE_DEVICES", "", 1); expect(child({self, "--probe"}) == "0 0 0 0 0\n", "CUDA hidden");
    unsetenv("CUDA_VISIBLE_DEVICES");
    for (const auto *failure : {"init", "count", "get", "retain", "push", "pop", "free"}) {
        setenv("EZLOCALAI_TEST_CUDA_FAIL", failure, 1);
        const bool absent = std::string(failure) == "init" || std::string(failure) == "count";
        expect(child({self, "--probe"}) == (absent ? "0 0 0 0 0\n" : "2 0 0 0 0\n"), std::string("probe failure returns zero: ") + failure);
    }
    unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    for (const auto *mode : {"full", "partial", "best_gpu", "secondary", "explicit", "cpu", "probe_failure", "retry", "retry_cpu", "non_memory", "warm_failure", "dimensions", "timeout", "warm_disabled", "transient"}) child({self, "--managed", mode, model.string()});
    if (old) setenv("LD_LIBRARY_PATH", saved.c_str(), 1); else unsetenv("LD_LIBRARY_PATH");
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--probe") { std::cout << ezlocalai::visible_cuda_device_count(); for (int ordinal : {0, 1, -1, 99}) std::cout << ' ' << std::setprecision(17) << ezlocalai::cuda_free_memory_gb(ordinal); std::cout << '\n'; return 0; }
        if (argc == 2 && std::string(argv[1]) == "--contexts") { context_case(); return 0; }
        if (argc == 4 && std::string(argv[1]) == "--managed") { managed_case(argv[2], argv[3]); return 0; }
        expect(argc == 3, "reference checkout and fixture required");
        const auto root = fs::absolute(argv[1]), fixture = fs::absolute(argv[2]), self = fs::absolute(argv[0]);
        const auto initial = fs::current_path(), temporary = fs::temp_directory_path() / ("ezlocalai-embedding-placement-" + std::to_string(getpid()));
        struct Cleanup { fs::path initial, temporary; ~Cleanup() { fs::current_path(initial); fs::remove_all(temporary); } } cleanup{initial, temporary};
        fs::create_directories(temporary); fs::current_path(temporary);
        const auto count = reference_cases(root, fixture);
        driver_cases(self.string(), fixture.parent_path() / "embedding_gguf/normal_le3.gguf");
        std::cout << "Verified " << count << " embedding metadata/placement/retry/warmup reference cases, CUDA context probes and 15 engine lifecycle cases\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
