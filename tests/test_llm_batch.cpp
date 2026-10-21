#include "ezlocalai/llm_batch.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include <arpa/inet.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &s, char delimiter = '\t') {
    std::vector<std::string> out;
    for (size_t begin = 0;;) { const auto end = s.find(delimiter, begin); out.push_back(s.substr(begin, end == std::string::npos ? end : end - begin)); if (end == std::string::npos) return out; begin = end + 1; }
}
std::string hex(const std::string &s) { std::ostringstream out; for (unsigned char c : s) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(c); return out.str(); }
std::string unhex(const std::string &s) { std::string out; for (size_t i = 0; i < s.size(); i += 2) out += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &s) { unsigned char hash[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), hash); return hex({reinterpret_cast<char *>(hash), 32}); }
std::optional<int64_t> optional(const std::string &s) { return s == "N" ? std::nullopt : std::optional<int64_t>(std::stoll(s)); }
template<class Callback> size_t cases(const fs::path &reference, const fs::path &path, Callback callback) {
    std::istringstream input(read(path)); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto at = line.find(" sha256="); expect(sha256(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "batch source hash"); continue; }
        callback(split(line)); ++count;
    }
    return count;
}
struct Probes {
    std::vector<ezlocalai::CudaDevice> devices;
    std::vector<ezlocalai::CudaMemoryInfo> memory;
    bool available; int count; std::string failure, trace;
    Probes(const std::string &encoded, bool a, int c, const std::string &f) : available(a), count(c), failure(f) {
        const auto raw = unhex(encoded);
        if (!raw.empty()) for (const auto &line : split(raw, '\n')) {
            const auto d = split(line, ','); const auto total = std::stoull(d[3]);
            devices.push_back({static_cast<int>(devices.size()), unhex(d[0]), std::stoi(d[1]), std::stoi(d[2]), total});
            memory.push_back({std::stoull(d[4]), total});
        }
    }
    void event(const std::string &name) {
        if (!trace.empty()) trace += '\n'; trace += name;
        if (failure == name.substr(0, 1)) throw std::runtime_error("fixture probe failed");
    }
    ezlocalai::LlmBatchProbes operations() {
        return {[&] { event("A"); return available; }, [&] { event("C"); return count; },
            [&](int i) { event("M:" + std::to_string(i)); return memory.at(i); },
            [&](int i) { event("K:" + std::to_string(i)); const auto &d = devices.at(i); return std::make_pair(d.major, d.minor); },
            [&](int i) { event("P:" + std::to_string(i)); return devices.at(i); }};
    }
};
size_t reference_cases(const fs::path &reference, const fs::path &fixtures) {
    const auto automatic = cases(reference, fixtures / "llm_batch_auto_cases.tsv", [](const auto &c) {
        Probes p(c[6], c[4] == "1", std::stoi(c[5]), c[7]);
        const auto result = ezlocalai::calculate_llm_auto_batch(std::stoll(c[1]), std::stoll(c[2]), unhex(c[3]), p.operations());
        expect(result.batch == std::stoll(c[8]) && result.ubatch == std::stoll(c[9]) && result.reason == unhex(c[10]), "auto batch case " + c[0] + ": " + std::to_string(result.batch) + '/' + std::to_string(result.ubatch) + ' ' + result.reason);
        expect(p.trace == unhex(c[11]), "auto batch probe order " + c[0] + ": " + p.trace);
    });
    const auto profiles = cases(reference, fixtures / "llm_gpu_profile_cases.tsv", [](const auto &c) {
        Probes p(c[4], c[2] == "1", std::stoi(c[3]), c[5]);
        const auto result = ezlocalai::gpu_inference_profile(p.operations(), optional(c[1]));
        expect(result.family == unhex(c[6]) && result.capacity_gb == std::stod(c[7]), "GPU profile case " + c[0]);
        expect(p.trace == unhex(c[8]), "GPU profile probe order " + c[0] + ": " + p.trace);
    });
    const auto settings = cases(reference, fixtures / "llm_batch_env_cases.tsv", [](const auto &c) {
        const ezlocalai::LlmAutoBatch automatic{std::stoll(c[1]), std::stoll(c[2]), "fixture"};
        const auto result = ezlocalai::resolve_llm_batch_parameters(automatic, unhex(c[4]), unhex(c[5]), optional(c[6]), optional(c[7]));
        expect(result.batch == c[8] && result.ubatch == c[9], "constructor batch case " + c[0]);
        expect(ezlocalai::configured_llm_ubatch(unhex(c[5]), automatic.ubatch, std::stoll(c[3])) == std::stoll(c[10]), "retry micro-batch case " + c[0]);
    });
    expect(automatic == 3482 && profiles == 148 && settings == 3612, "all batch reference cases");
    return automatic + profiles + settings;
}
void environment() {
    for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("GPU_LAYERS", "0", 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
}
int available_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)), "fixture port"); socklen_t n = sizeof(a); getsockname(fd, reinterpret_cast<sockaddr *>(&a), &n); close(fd); return ntohs(a.sin_port);
}
void managed_case(const fs::path &root, const std::string &name, const std::string &model, int context,
    const std::string &main_gpu, const std::string &batch, const std::string &ubatch, int expected_batch, int expected_ubatch,
    const std::string &expected_plan = "") {
    environment(); const auto directory = root / name; fs::create_directories(directory);
    const auto path = directory / "model.gguf"; std::ofstream(path) << "fixture GGUF";
    const auto port = std::to_string(available_port());
    setenv("LLAMA_CPP_MODEL_PATH", path.c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", directory.c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", port.c_str(), 1);
    setenv("LLAMA_CPP_CONTEXT", std::to_string(context).c_str(), 1); setenv("MAIN_GPU", main_gpu.c_str(), 1);
    setenv("LLM_BATCH_SIZE", batch.c_str(), 1); setenv("LLM_UBATCH_SIZE", ubatch.c_str(), 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend(model);
        expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "managed batch " + name);
        const auto args = read(directory / ("args-" + port));
        expect(args.find("-b\n" + std::to_string(expected_batch) + '\n') != std::string::npos && args.find("-ub\n" + std::to_string(expected_ubatch) + '\n') != std::string::npos, "managed batch flags " + name + ": " + args);
        const auto runtime = backend.model_lifecycle_json();
        expect(runtime.find("\"n_batch\":" + std::to_string(expected_batch)) != std::string::npos && runtime.find("\"n_ubatch\":" + std::to_string(expected_ubatch)) != std::string::npos, "managed batch runtime " + name);
        if (!expected_plan.empty()) expect(read(directory / ("load-plan-" + port)) == expected_plan, "retry sizing uses post-failure free memory");
    }
    std::istringstream pids(read(directory / ("launch-" + port))); for (int pid; pids >> pid;) expect(kill(pid, 0) != 0, "batch fixture child reaped");
}
void native_driver_cases(const fs::path &root) {
    unsetenv("CUDA_VISIBLE_DEVICES"); unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    const auto check = [](int64_t main, int64_t context, const std::string &model, int64_t b, int64_t u, const std::string &reason = "") {
        const auto result = ezlocalai::calculate_llm_auto_batch(main, context, model);
        expect(result.batch == b && result.ubatch == u && (reason.empty() || reason == result.reason), "native CUDA auto batch: " + result.reason);
    };
    check(0, 200000, "Qwen3.8-27B", 512, 128);
    check(1, 65536, "plain", 8192, 2048);
    check(2, 262144, "Qwen3.8-27B", 8192, 1024);
    check(2, 262144, "other-MTP", 8192, 256);
    check(9, 65536, "plain", 512, 128);
    const auto profile = ezlocalai::gpu_inference_profile(ezlocalai::native_llm_batch_probes(), std::nullopt);
    expect(profile.family == "5090" && profile.capacity_gb == 32, "5090 profile priority over larger A100");
    for (const auto failure : {"free", "compute", "name", "memory"}) { setenv("EZLOCALAI_TEST_CUDA_FAIL", failure, 1); check(1, 128, "plain", 2048, 512, "probe-failed/default"); }
    unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    managed_case(root, "t4", "plain", 65536, "0", "auto", "auto", 512, 128);
    managed_case(root, "a100", "plain", 65536, "1", "", "", 8192, 2048);
    managed_case(root, "mtp", "Qwen3.8-27B", 262144, "2", "auto", "auto", 8192, 1024);
    managed_case(root, "invalid", "plain", 65536, "1", "bad", "bad", 8192, 256);
    managed_case(root, "unicode", "plain", 65536, "1", "１２８", "٢_٥٦", 128, 128);
    managed_case(root, "negative", "plain", 65536, "1", "-4", "512", -4, -4);
    const auto devices = root / "devices.tsv";
    const auto original = read(devices);
    auto reduced = original;
    reduced.replace(reduced.find("32212254720"), 11, "8589934592");
    setenv("LLM_FIXTURE_FAIL_LAUNCH", "1", 1); setenv("LLM_FIXTURE_CUDA_AFTER_FAILURE", reduced.c_str(), 1);
    managed_case(root, "refresh", "plain", 65536, "2", "auto", "auto", 1024, 256,
        "0\t2048\t65536\n0\t256\t65536\n");
    unsetenv("LLM_FIXTURE_FAIL_LAUNCH"); unsetenv("LLM_FIXTURE_CUDA_AFTER_FAILURE");
    std::ofstream(devices) << original;
    setenv("CUDA_VISIBLE_DEVICES", "1,0", 1);
    check(0, 65536, "plain", 8192, 2048);
    check(1, 65536, "plain", 512, 128);
    managed_case(root, "visible-order", "plain", 65536, "0", "auto", "auto", 8192, 2048);
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    check(0, 64, "plain", 1024, 256, "cpu/default");
    managed_case(root, "cpu", "plain", 64, "0", "auto", "auto", 1024, 256);
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--cuda") { native_driver_cases(argv[2]); return 0; }
        expect(argc == 3, "reference root and fixtures required");
        const auto count = reference_cases(argv[1], argv[2]);
        const auto directory = fs::temp_directory_path() / ("ezl-batch-" + std::to_string(getpid())); fs::create_directories(directory);
        struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{directory};
        std::ofstream(directory / "devices.tsv") << "Tesla T4\t7\t5\t17179869184\t12884901888\nNVIDIA A100\t8\t0\t85899345920\t68719476736\nNVIDIA GeForce RTX 5090\t12\t0\t34359738368\t32212254720\n";
        setenv("EZLOCALAI_TEST_CUDA_DEVICES", (directory / "devices.tsv").c_str(), 1); setenv("LD_LIBRARY_PATH", TEST_CUDA_DRIVER_DIRECTORY, 1);
        const auto result = ezlocalai::run_process_to_files({fs::absolute(argv[0]).string(), "--cuda", directory.string()}, (directory / "child.log").string(), "", 30, true);
        expect(!result.timed_out && result.status == 0, "native GPU batch integration: " + read(directory / "child.log"));
        std::cout << "Verified " << count << " source batch/profile/configuration cases and CUDA/managed-engine integration\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
