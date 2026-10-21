#include "ezlocalai/llm_parameters.hpp"
#include "ezlocalai/engine_revision.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
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
void expect(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &s, char separator = '\t') {
    std::vector<std::string> result;
    for (size_t start = 0;;) { const auto end = s.find(separator, start); result.push_back(s.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) return result; start = end + 1; }
}
std::string unhex(const std::string &s) { std::string out; for (size_t i = 0; i < s.size(); i += 2) out += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &s) { unsigned char hash[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), hash); std::ostringstream out; for (auto c : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
size_t reference_cases(const fs::path &reference, const fs::path &fixture) {
    std::istringstream in(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("# ezlocalai/", 0) == 0) { const auto at = line.find(" sha256="); expect(sha256(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "parameter source hash"); }
            if (line.rfind("# upstream ", 0) == 0) expect(line.find(ezlocalai::xllamacpp_reference_revision) != std::string::npos, "binding revision provenance");
            continue;
        }
        const auto c = split(line); ++count;
        if (c[0] == "R") {
            std::string status = "S", message;
            try { ezlocalai::validate_llm_reasoning_budget(c[2] == "N" ? "-1" : unhex(c[2].substr(1))); }
            catch (const ezlocalai::LlmParameterError &error) { status = error.category; message = error.message; }
            expect(status == c[3] && message == unhex(c[4]), "reasoning case " + c[1] + ": " + message);
        } else if (c[0] == "P") {
            std::string status = "S", result;
            try { result = std::to_string(ezlocalai::llm_parallel_parameter(std::stoll(c[2]), std::stoll(c[1]), c[3] != "none")); }
            catch (const ezlocalai::LlmParameterError &error) { status = error.category; result = error.message; }
            expect(status == c[4] && result == (status == "S" ? c[5] : unhex(c[5])), "parallel parameter case " + std::to_string(count));
        } else if (c[0] == "T") {
            std::vector<double> values; if (!c[3].empty()) for (const auto &s : split(c[3], ',')) values.push_back(std::strtod(s.c_str(), nullptr));
            const auto result = ezlocalai::llm_tensor_parameters(values, std::stoi(c[1]), c[2] == "1");
            expect((result.split_none ? "0" : "1") == c[4] && (result.fit ? "1" : "0") == c[5], "tensor parameter case " + std::to_string(count));
        } else expect(c[0] == "I" && std::to_string(ezlocalai::llm_image_min_tokens(unhex(c[1]))) == c[2], "image token parameter");
    }
    expect(count == 2794, "all parameter source/binding cases"); return count;
}
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const auto key : {"LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_MMPROJ_PATH", "TENSOR_SPLIT", "CUDA_VISIBLE_DEVICES", "KV_CACHE_TYPE_5090", "KV_CACHE_TYPE_3090"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("GPU_LAYERS", "0", 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1); setenv("LLM_PROMPT_CACHE_ALLOW_UNSAFE", "true", 1);
}
int available_ports() { return fixture_ports(2); }
struct Fixture {
    fs::path directory; int port;
    Fixture(const fs::path &root, const std::string &name) : directory(root / name), port(available_ports()) {
        fs::create_directories(directory); std::ofstream(directory / "model.gguf") << "fixture";
        setenv("LLAMA_CPP_MODEL_PATH", (directory / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", directory.c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
    }
    fs::path path(const std::string &name, int offset = 0) const { return directory / (name + '-' + std::to_string(port + offset)); }
    void reaped() const { for (int offset : {0, 1}) { std::istringstream in(read(path("launch", offset))); for (int pid; in >> pid;) expect(kill(pid, 0) != 0, "parameter fixture reaped"); } }
};
void managed_cases(const fs::path &root) {
    environment(); Fixture defaults(root, "defaults");
    { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "default parameter startup");
      const auto args = read(defaults.path("args")); for (const auto flag : {"--warmup\n", "--cont-batching\n", "--ctx-checkpoints\n32\n", "--checkpoint-min-step\n8192\n"}) expect(args.find(flag) != std::string::npos, "pinned engine defaults");
      auto runtime = backend.model_lifecycle_json(); expect(runtime.find("\"checkpoint_min_step\":8192") != std::string::npos && runtime.find("\"n_ctx_checkpoints\":32") != std::string::npos, "checkpoint runtime shape");
      auto restore = backend.unload_llms_for_media(); setenv("LLAMA_CPP_EXTRA_ARGS", "-ctxcp 7 -cms 2048 --no-warmup --no-cont-batching", 1); restore(true);
      runtime = backend.model_lifecycle_json(); expect(runtime.find("\"checkpoint_min_step\":2048") != std::string::npos && runtime.find("\"n_ctx_checkpoints\":7") != std::string::npos, "checkpoint overrides on restoration");
    } defaults.reaped();
    environment(); Fixture worker(root, "worker"); setenv("MAIN_GPU", "0,1", 1);
    { ezlocalai::LlamaCppInferenceBackend backend("first,second"); const auto response = backend.chat(R"({"model":"second","messages":[]})");
      expect(response.find("choices") != std::string::npos, "assigned worker startup on port " + std::to_string(worker.port + 1) + ": " + response);
      const auto args = read(worker.path("args", 1)); for (const auto flag : {"--main-gpu\n1\n", "--split-mode\nnone\n", "--fit\noff\n"}) expect(args.find(flag) != std::string::npos, "single GPU source placement");
      expect(args.find("--tensor-split") == std::string::npos, "source single GPU config uses split mode");
      const auto runtime = backend.model_lifecycle_json(); const auto at = runtime.find("\"tensor_split\":[0.0,1.0,0.0,");
      expect(at != std::string::npos, "source tensor array retained in runtime metadata");
      expect(std::count(runtime.begin() + at, runtime.begin() + runtime.find(']', at), ',') == 127, "all 128 source tensor entries retained");
    } worker.reaped();
    for (const auto model : {"Qwen3.8-27B", "qwen3.8-27banana", "ordinary"}) {
        environment(); Fixture run(root, model); setenv("LLM_SPECULATIVE_TYPE", "none", 1); setenv("LLAMA_CPP_MMPROJ_PATH", "projector.gguf", 1);
        { ezlocalai::LlamaCppInferenceBackend backend(model); expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "vision token startup"); expect((read(run.path("args")).find("--image-min-tokens\n1024\n") != std::string::npos) == (std::string(model) != "ordinary"), "model-specific image minimum"); } run.reaped();
    }
    int index = 0;
    for (const auto value : {"0", "1", "２_５６", "", "auto", "-1", "-2", "-999999999999999999999"}) {
        environment(); Fixture run(root, "reasoning-" + std::to_string(index++)); setenv("REASONING_BUDGET", value, 1);
        { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); const auto response = backend.chat(R"({"messages":[]})");
          if (value[0] == '-') expect(response.find("choices") != std::string::npos && read(run.path("args")).find("--reasoning-budget") == std::string::npos, "negative reasoning budget retains unlimited default");
          else expect(response.find(value[0] == '\0' || std::string(value) == "auto" ? "invalid literal for int()" : "has no attribute 'reasoning_budget'") != std::string::npos && !fs::exists(run.path("launch")), "source reasoning configuration failure precedes engine launch");
        } run.reaped();
    }
    environment(); Fixture order(root, "validation-order"); setenv("REASONING_BUDGET", "0", 1); setenv("KV_CACHE_TYPE", "bad", 1);
    { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); setenv("LLM_SPECULATIVE_TYPE", "bad", 1); expect(backend.chat(R"({"messages":[]})").find("has no attribute 'reasoning_budget'") != std::string::npos && !fs::exists(order.path("launch")), "reasoning failure precedes KV and live speculative validation"); }
}
void build_defaults(const fs::path &reference) {
    const fs::path native(TEST_NATIVE_SOURCE_DIRECTORY); const auto source = read(reference / "scripts/build_xllamacpp.py");
    expect(source.find(std::string("XLLAMACPP_REVISION = \"") + ezlocalai::xllamacpp_reference_revision + '"') != std::string::npos, "binding pin matches reference");
    for (const auto flag : {"-DGGML_CUDA_NCCL=OFF", "86-real;89-real;120-real"}) {
        expect(source.find(flag) != std::string::npos && read(native / "cuda.Dockerfile").find(flag) != std::string::npos && read(native / "docker-compose-cuda.yml").find(flag) != std::string::npos, "CUDA build defaults match reference");
    }
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--cuda") { managed_cases(argv[2]); return 0; }
        expect(argc == 3, "reference root and fixture required"); const auto count = reference_cases(argv[1], argv[2]); build_defaults(argv[1]);
        const auto directory = fs::temp_directory_path() / ("ezl-parameters-" + std::to_string(getpid())); fs::create_directories(directory);
        struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{directory};
        std::ofstream(directory / "devices.tsv") << "NVIDIA GeForce RTX 5090\t12\t0\t34359738368\t32212254720\nNVIDIA GeForce RTX 3090\t8\t6\t25769803776\t21474836480\n";
        setenv("EZLOCALAI_TEST_CUDA_DEVICES", (directory / "devices.tsv").c_str(), 1); setenv("LD_LIBRARY_PATH", TEST_CUDA_DRIVER_DIRECTORY, 1);
        const auto result = ezlocalai::run_process_to_files({fs::absolute(argv[0]).string(), "--cuda", directory.string()}, (directory / "child.log").string(), "", 40, true);
        expect(!result.timed_out && result.status == 0, "managed parameter integration: " + read(directory / "child.log"));
        std::cout << "Verified " << count << " source/compiled-binding parameter cases and managed configuration\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
