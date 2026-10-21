#include "ezlocalai/llm_context.hpp"
#include "ezlocalai/llm_parameters.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
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
std::string hash(const std::string &s) { unsigned char digest[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), digest); std::ostringstream out; for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::vector<std::string> split(const std::string &s) { std::vector<std::string> out; for (size_t begin = 0;;) { const auto end = s.find('\t', begin); out.push_back(s.substr(begin, end == std::string::npos ? end : end - begin)); if (end == std::string::npos) return out; begin = end + 1; } }
std::string unhex(const std::string &s) { std::string out; for (size_t i = 0; i < s.size(); i += 2) out += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)); return out; }
size_t policy_cases(const fs::path &reference, const fs::path &fixture) {
    std::istringstream in(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto at = line.find(" sha256="); if (at != std::string::npos) expect(hash(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "context reference hash"); continue; }
        const auto c = split(line); ++count; std::string status = "S", result;
        try {
            if (c[0] == "O") result = ezlocalai::llm_optimal_context(unhex(c[1]));
            else if (c[0] == "L") result = ezlocalai::llm_configured_context_limit(c[1] == "N" ? std::nullopt : std::optional<std::string>(unhex(c[1].substr(1))), c[2]);
            else if (c[0] == "P") result = std::to_string(ezlocalai::llm_context_parameter(ezlocalai::llm_effective_context(c[1], unhex(c[2]))));
            else if (c[0] == "R") result = std::to_string(ezlocalai::llm_inference_reload_allowed(std::stoul(c[1]), std::stoll(c[2])));
            else throw std::runtime_error("unknown context case");
        } catch (const ezlocalai::LlmParameterError &error) { status = error.category; result = error.message; }
        expect(status == c[c.size() - 2] && result == unhex(c.back()), "context case " + std::to_string(count) + ": " + status + ": " + result + " expected " + c[c.size() - 2] + ": " + unhex(c.back()));
    }
    expect(count == 1066, "all source and compiled binding context cases"); return count;
}
void environment() {
    for (const auto &item : ezlocalai::environment_defaults) unsetenv(item.first.data());
    for (const auto *key : {"LLAMA_CPP_CONTEXT", "LLAMA_CPP_DEFAULT_CONTEXT", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_MMPROJ_PATH", "LLM_FIXTURE_INFERENCE_RESULT", "LLM_FIXTURE_INFERENCE_STATUS", "EMBEDDING_FIXTURE_MAX_LAYERS", "LLAMA_CPP_TOTAL_LAYERS"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("GPU_LAYERS", "0", 1); setenv("LLM_PROMPT_CACHE_MIB", "0", 1);
}
int available_port() { return fixture_ports(3); }
struct Fixture {
    fs::path root; int port;
    Fixture(const fs::path &parent, const std::string &name) : root(parent / name), port(available_port()) {
        fs::create_directories(root); std::ofstream(root / "model.gguf") << "fixture";
        setenv("LLAMA_CPP_MODEL_PATH", (root / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", root.c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
    }
    fs::path path(const std::string &name, int index = 0) const { return root / (name + '-' + std::to_string(port + index)); }
    int launches(int index = 0) const { std::istringstream in(read(path("launch", index))); int count = 0; for (int pid; in >> pid;) ++count; return count; }
    void reaped() const { for (int i = 0; i < 3; ++i) { std::istringstream in(read(path("launch", i))); for (int pid; in >> pid;) expect(kill(pid, 0) != 0, "context fixture children reaped"); } }
};
void managed_cases(const fs::path &root) {
    const std::vector<std::pair<std::string, int>> settings{{"65536",65536},{"512",512},{"0",16384},{"-1",1},{"0,4096",4096},{"١٢_٨",128}};
    int number = 0;
    for (const auto &setting : settings) {
        environment(); Fixture run(root, "request-" + std::to_string(number++)); setenv("LLM_MAX_TOKENS", setting.first.c_str(), 1);
        { ezlocalai::LlamaCppInferenceBackend backend("ordinary");
          const auto body = std::string("{\"messages\":[{\"role\":\"user\",\"content\":\"") + std::string(90000, 'x') + "\"}]}";
          expect(backend.chat(body).find("choices") != std::string::npos, "context initial generation");
          expect(backend.llm_context_size() == setting.second, "captured context ceiling " + setting.first);
          expect(read(run.path("args")).find("-c\n" + std::to_string(setting.second) + "\n") != std::string::npos, "source context reaches engine");
          expect(backend.model_lifecycle_json().find("\"context\":" + std::to_string(setting.second)) != std::string::npos, "runtime records selected context");
          setenv("LLM_MAX_TOKENS", "1024", 1);
          backend.chat(body.substr(0, body.size() - 1) + ",\"stream\":true}");
          expect(run.launches() == 1 && backend.llm_context_size() == setting.second, "streaming estimate cannot enlarge captured context");
          auto restore = backend.unload_llms_for_media(); restore(true);
          expect(backend.llm_context_size() == setting.second && run.launches() == 2, "restoration preserves runtime context despite environment changes");
        } run.reaped();
    }
    environment(); Fixture csv(root,"csv"); setenv("LLM_MAX_TOKENS","512,2048",1);
    { ezlocalai::LlamaCppInferenceBackend backend("first,second");
      const auto second = backend.completion(R"({"model":"second","prompt":"test"})");
      expect(second.find("choices") != std::string::npos && backend.llm_context_size() == 2048, "second model configured context: " + std::to_string(backend.llm_context_size()) + "; response=" + second);
      expect(backend.chat(R"({"model":"first","messages":[]})").find("choices") != std::string::npos && backend.llm_context_size() == 512, "first model configured context");
      expect(csv.launches() == 1 && csv.launches(1) == 1, "switching models preserves independent contexts");
    } csv.reaped();
    environment(); Fixture preload(root,"preload-zero"); setenv("LLM_MAX_TOKENS","0",1);
    { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); std::string error;
      expect(backend.warmup(&error) && read(preload.path("args")).find("-c\n0\n") != std::string::npos, "preload passes source zero context to binding: " + error);
      expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos && backend.llm_context_size() == 16384 && preload.launches() == 2, "request replaces insufficient preloaded zero context");
    } preload.reaped();
    environment(); Fixture overflow(root,"overflow"); setenv("LLM_MAX_TOKENS","2147483648",1); setenv("REASONING_BUDGET","0",1);
    { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); const auto response = backend.chat(R"({"messages":[]})");
      expect(response.find("value too large to convert to int32_t") != std::string::npos && overflow.launches() == 0 && backend.loaded_models_json() == "{}", "context binding overflow precedes later parameter validation and launch");
    }
    for (const auto &mode : {"one", "parallel", "auto", "models", "context-error"}) {
        environment(); Fixture run(root, mode); setenv("GPU_LAYERS", "16", 1); setenv("LLM_FIXTURE_INFERENCE_STATUS", "500", 1);
        const std::string kind(mode); const bool context_error = kind == "context-error";
        setenv("LLM_FIXTURE_INFERENCE_RESULT", context_error ? R"({"error":{"message":"prompt exceeds context [n_prompt_tokens=90000, n_ctx=512]","code":500}})" : R"({"error":{"message":"CUDA out of memory","code":500}})", 1);
        if (kind == "parallel") setenv("N_PARALLEL", "2", 1);
        if (kind == "auto") setenv("N_PARALLEL", "0", 1);
        { ezlocalai::LlamaCppInferenceBackend backend(kind == "models" ? "first,second" : "ordinary");
          const auto response = backend.chat(R"({"messages":[]})"); expect(response.find("error") != std::string::npos, "inference fixture failure preserved");
          expect(kind == "one" ? run.launches() > 1 : run.launches() == 1, "reload guard for " + kind);
          expect(backend.llm_context_size() == 65536, "failure never exceeds configured context");
        } run.reaped();
    }
}
}
int main(int argc, char **argv) {
    const auto root = fs::temp_directory_path() / ("ezlocalai-context-" + std::to_string(getpid()));
    try {
        expect(argc == 3, "reference root and context fixture required"); const auto count = policy_cases(argv[1], argv[2]);
        fs::create_directories(root);
        managed_cases(root); fs::remove_all(root); std::cout << "Verified " << count << " source/compiled-binding context cases and managed ceiling/reload behavior\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << "\nArtifacts: " << root << '\n'; return 1; }
}
