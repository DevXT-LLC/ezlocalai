#include "ezlocalai/llm_load.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &s, char delimiter = '\t') {
    std::vector<std::string> out;
    for (size_t begin = 0;;) {
        const auto end = s.find(delimiter, begin);
        out.push_back(s.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) return out;
        begin = end + 1;
    }
}
std::string hex(const std::string &s) { std::ostringstream out; for (unsigned char c : s) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(c); return out.str(); }
std::string unhex(const std::string &s) { std::string out; for (size_t i = 0; i < s.size(); i += 2) out += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &s) { unsigned char hash[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), hash); return hex({reinterpret_cast<char *>(hash), 32}); }
std::optional<int64_t> optional(const std::string &s) { return s == "N" ? std::nullopt : std::optional<int64_t>(std::stoll(s)); }
std::string format(std::optional<int64_t> n) { return n ? std::to_string(*n) : "N"; }

size_t reference_cases(const fs::path &root, const fs::path &fixture) {
    size_t count = 0;
    std::istringstream in(read(fixture));
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto at = line.find(" sha256=");
            if (at != std::string::npos) expect(sha256(read(root / line.substr(2, at - 2))) == line.substr(at + 8), "load source hash");
            continue;
        }
        ++count;
        const auto c = split(line);
        const auto script = split(unhex(c[6]), '\n');
        const auto probe = unhex(c[7]);
        std::string trace;
        const auto record = [&](const std::string &event) { if (!trace.empty()) trace += '\n'; trace += event; };
        size_t load_index = 0, probe_index = 0;
        ezlocalai::LlmLoadConfiguration configuration;
        configuration.model = unhex(c[1]); configuration.context = std::stoll(c[2]);
        configuration.initial_ubatch = std::stoll(c[3]); configuration.total_layers = std::stoll(c[4]); configuration.gpu_layers = optional(c[5]);
        ezlocalai::LlmLoadOperations operations;
        operations.load = [&](auto layers, auto ubatch) {
            record("L\t" + format(layers) + '\t' + format(ubatch));
            const auto &result = script[std::min(load_index++, script.size() - 1)];
            if (result[0] == 'E') throw std::runtime_error(unhex(result.substr(1)));
        };
        operations.cleanup = [&] { record("C"); };
        operations.configured_ubatch = [&] { record("B"); return configuration.initial_ubatch; };
        operations.probe = [&](int64_t layers, int64_t ubatch) {
            record("P\t" + std::to_string(layers) + '\t' + std::to_string(ubatch));
            if (probe[0] == 'X') throw std::runtime_error(unhex(probe.substr(1)));
            if (probe[0] == 'T') return layers <= std::stoll(probe.substr(1));
            return probe[1 + std::min(probe_index++, probe.size() - 2)] == '1';
        };
        std::string actual;
        try { const auto selection = ezlocalai::load_llm_resilient(configuration, operations); actual = "S\t" + format(selection.gpu_layers) + '\t' + format(selection.ubatch); }
        catch (const std::exception &error) { actual = "E\tRuntimeError\t" + hex(error.what()); }
        expect(actual == unhex(c[9]), "load result case " + c[0] + ": " + actual + " expected " + unhex(c[9]));
        expect(trace == unhex(c[8]), "load trace case " + c[0] + ": " + trace + " expected " + unhex(c[8]));
    }
    expect(count == 3335, "all resilient-load source cases");
    return count;
}

void environment() {
    for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
    for (const auto key : {"LLM_FIXTURE_PROBE_RESULT", "LLM_FIXTURE_PROBE_BLOCK", "LLM_FIXTURE_PROBE_MAX_LAYERS", "LLM_FIXTURE_MAX_UBATCH", "LLM_FIXTURE_FAIL_LAUNCH", "EMBEDDING_FIXTURE_MAX_LAYERS", "EMBEDDING_FIXTURE_FAILURE"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1);
    setenv("GPU_LAYERS", "12", 1); setenv("LLAMA_CPP_TOTAL_LAYERS", "12", 1);
    setenv("LLAMA_CPP_CONTEXT", "4096", 1); setenv("LLM_BATCH_SIZE", "1024", 1); setenv("LLM_UBATCH_SIZE", "512", 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
}
int available_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)), "fixture port");
    socklen_t n = sizeof(a); getsockname(fd, reinterpret_cast<sockaddr *>(&a), &n); close(fd); return ntohs(a.sin_port);
}
struct Fixture {
    fs::path dir; int port;
    Fixture(const std::string &name) : dir(fs::temp_directory_path() / ("ezl-load-" + std::to_string(getpid()) + "-" + name)), port(available_port()) {
        fs::create_directories(dir);
        const auto model = dir / "model.gguf"; std::ofstream(model) << "GGUF fixture";
        setenv("LLAMA_CPP_MODEL_PATH", model.c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", dir.c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
    }
    fs::path path(const std::string &prefix) const { return dir / (prefix + "-" + std::to_string(port)); }
    void reaped(bool surviving = false) const {
        std::istringstream input(read(path("launch"))); std::vector<int> pids;
        for (int pid; input >> pid;) pids.push_back(pid);
        expect(!pids.empty(), "managed children launched");
        for (size_t i = 0; i < pids.size(); ++i) expect((kill(pids[i], 0) == 0) == (surviving && i + 1 == pids.size()), "probe child lifetime");
    }
    void clean() { reaped(); fs::remove_all(dir); }
};
void micro_batch_recovery() {
    environment(); Fixture fixture("ubatch");
    setenv("LLM_UBATCH_SIZE", "1024", 1); setenv("LLM_FIXTURE_MAX_UBATCH", "128", 1);
    const std::string plan = "12\t1024\t4096\n12\t512\t4096\n12\t256\t4096\n12\t128\t4096\n";
    {
        ezlocalai::LlamaCppInferenceBackend backend("org/model");
        expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "micro-batch recovery");
        expect(read(fixture.path("load-plan")) == plan, "reduce micro-batches without reducing GPU layers");
        fixture.reaped(true);
        expect(backend.model_lifecycle_json().find("\"n_ubatch\":128") != std::string::npos, "selected runtime micro-batch");
        auto restore = backend.unload_llms_for_media(); restore(true);
        expect(read(fixture.path("load-plan")) == plan + plan, "known residency reuses layers but recalculates micro-batch");
        expect(backend.model_lifecycle_json().find("\"count\":2,\"failures\":0") != std::string::npos, "one lifecycle event per complete load");
        fixture.reaped(true);
    }
    fixture.clean();
}
void layer_search(bool final_failure) {
    environment(); Fixture fixture(final_failure ? "reload-failure" : "layers");
    setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "6", 1);
    if (final_failure) setenv("LLM_FIXTURE_FAIL_LAUNCH", "8", 1);
    const std::string plan = "12\t512\t4096\n12\t256\t4096\n12\t128\t4096\n5\t128\t4096\n8\t128\t4096\n6\t128\t4096\n7\t128\t4096\n6\t128\t4096\n";
    {
        ezlocalai::LlamaCppInferenceBackend backend("org/model");
        const auto result = backend.completion(R"({"prompt":"ordinary"})");
        expect(read(fixture.path("load-plan")) == plan, "binary search maximizes residency and reloads winning configuration once");
        if (final_failure) {
            expect(result.find("CUDA final reload failure") != std::string::npos && result.find("Failed to allocate inference capacity") == std::string::npos, "final reload failure propagates without another search");
            expect(backend.loaded_models_json() == "{}" && !fs::exists(fixture.path("seen")), "failed reload is never published or used");
            expect(backend.model_lifecycle_json().find("\"count\":1,\"failures\":1") != std::string::npos, "whole failed load recorded once");
        } else {
            expect(result.find("choices") != std::string::npos, "maximum fitting GPU layers recover generation");
            expect(backend.model_lifecycle_json().find("\"gpu_layers\":6") != std::string::npos, "winning GPU placement published");
            expect(backend.model_lifecycle_json().find("\"count\":1,\"failures\":0") != std::string::npos, "probes do not add lifecycle load events");
        }
        fixture.reaped(!final_failure);
    }
    fixture.clean();
}
void nonretryable_failure() {
    environment(); Fixture fixture("nonretryable");
    setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "0", 1); setenv("EMBEDDING_FIXTURE_FAILURE", "missing model weights", 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("org/model");
        expect(backend.chat(R"({"messages":[]})").find("missing model weights") != std::string::npos, "nonretryable startup error preserved");
        expect(read(fixture.path("load-plan")) == "12\t512\t4096\n", "nonretryable failure does not launch probes");
    }
    fixture.clean();
}
void probe_timeout() {
    environment(); Fixture fixture("timeout");
    setenv("GPU_LAYERS", "1", 1); setenv("LLAMA_CPP_TOTAL_LAYERS", "1", 1);
    setenv("LLAMA_CPP_CONTEXT", "128", 1); setenv("LLM_UBATCH_SIZE", "128", 1);
    setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "0", 1); setenv("LLM_FIXTURE_PROBE_BLOCK", "1", 1); setenv("LLM_LOAD_PROBE_TIMEOUT", "0", 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("org/model");
        const auto start = std::chrono::steady_clock::now();
        const auto result = backend.chat(R"({"messages":[]})");
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        expect(elapsed >= 30 && elapsed < 36, "isolated fit probe uses one unrounded timeout, minimum 30 seconds");
        expect(result.find("Failed to allocate inference capacity for org/model at n_ctx=128") != std::string::npos, "timed-out sole candidate reports exhausted capacity");
        expect(read(fixture.path("load-plan")) == "1\t128\t128\n0\t128\t128\n", "fit probe alone reaches blocked native inference");
        expect(!fs::exists(fixture.path("seen")) && backend.loaded_models_json() == "{}", "timeout prevents user dispatch and residency");
        fixture.reaped();
    }
    fixture.clean();
}
}
int main(int argc, char **argv) {
    try {
        expect(argc == 3, "reference root and fixture required");
        const auto count = reference_cases(argv[1], argv[2]);
        micro_batch_recovery(); layer_search(false); layer_search(true); nonretryable_failure(); probe_timeout();
        std::cout << "Verified " << count << " source resilient-load cases and managed retry/probe lifetimes\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
