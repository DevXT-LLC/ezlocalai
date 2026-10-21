#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
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
std::string read(const fs::path &path) { std::ifstream input(path); return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()}; }
std::string hash(const std::string &value) {
    unsigned char digest[32]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    std::ostringstream out; for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str();
}
std::vector<std::string> split(const std::string &value) {
    std::vector<std::string> fields;
    for (size_t at = 0;;) { const auto end = value.find('\t', at); fields.push_back(value.substr(at, end == std::string::npos ? end : end - at)); if (end == std::string::npos) return fields; at = end + 1; }
}
std::string unhex(const std::string &value) {
    std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out;
}
size_t policies(const fs::path &reference, const fs::path &fixture) {
    std::istringstream input(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') {
            const auto at = line.find(" sha256=");
            if (at != std::string::npos) expect(hash(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "LLM output reference changed");
            continue;
        }
        const auto c = split(line); expect(c.size() == 6, "output fixture fields"); ++count;
        const auto result = ezlocalai::llm_buffered_response(unhex(c[2]), unhex(c[3]), c[0] == "chat", unhex(c[1]) != "false");
        const auto category = result.failure ? result.failure->category : "OK";
        const auto value = result.failure ? result.failure->message : result.body;
        expect(category == c[4] && value == unhex(c[5]), "output case " + std::to_string(count) + ": " + category + ": " + value + " expected " + c[4] + ": " + unhex(c[5]));
    }
    expect(count == 2632, "all output cases"); return count;
}
void environment() {
    for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
    for (const auto *key : {"LLAMA_CPP_SERVER_URL", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_CONTEXT", "LLAMA_CPP_EXTRA_ARGS", "LLM_FIXTURE_INFERENCE_RESULT", "NATIVE_QUEUE_RESULT_BYTES"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("EZLOCALAI_API_KEY", "none", 1);
    setenv("DEFAULT_MODEL", "org/model-GGUF", 1); setenv("GPU_LAYERS", "0", 1); setenv("LLM_MAX_TOKENS", "512", 1);
    setenv("LLM_PROMPT_CACHE_MIB", "0", 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1);
}
int available_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "output fixture bind"); socklen_t size = sizeof(address);
    getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size); close(fd); return ntohs(address.sin_port);
}
void api_cases(const fs::path &root) {
    struct Case { std::string name, body, output, expected; bool chat = true; int status = 200; std::string type = "application/json"; };
    const std::string large(5 * 1024 * 1024, 'x');
    const std::vector<Case> cases {
        {"chat", R"({"messages":[{"role":"user","content":""}],"stop":["custom"],"disable_fallback":true})",
            R"({"choices":[{"message":{"content":"  before custom </s>after  ","role":"assistant"}},{"message":{"content":" second </s> "}}],"model":"engine","usage":{"completion_tokens":12}})",
            R"({"choices":[{"message":{"content":"before custom","role":"assistant"}},{"message":{"content":" second </s> "}}],"model":"org/model-GGUF","usage":{"completion_tokens":12}})"},
        {"completion", R"({"prompt":"hi","disable_fallback":true})",
            R"({"choices":[{"text":"","message":{"content":"  before [Insert name] after </s>tail "}}],"model":"engine"})",
            R"({"choices":[{"text":"before  after","message":{"content":"  before [Insert name] after </s>tail "}}],"model":"org/model-GGUF"})", false},
        {"null", R"({"messages":[{"role":"user","content":""}],"disable_fallback":true})", R"({"choices":[{"message":{"content":null}}]})",
            R"({"choices":[{"message":{"content":null}}],"model":"org/model-GGUF"})"},
        {"no-choices", R"({"messages":[{"role":"user","content":""}],"disable_fallback":true})", R"({"error":false})", R"({"error":false,"model":"org/model-GGUF"})"},
        {"stream-completion-dictionary", R"({"prompt":"hi","stream":true,"disable_fallback":true})",
            R"({"choices":[{"text":" raw </s> "}]})", R"({"detail":{"type":"inference_error","message":"Server.handle_completions requires a callback for streaming or a non dict prompt."}})", false, 503},
        {"raw-stream-false", R"({"prompt":"hi","stream":"false","disable_fallback":true})",
            R"({"choices":[{"text":" raw </s> "}]})", R"({"detail":{"type":"inference_error","message":"Server.handle_completions requires a callback for streaming or a non dict prompt."}})", false, 503},
        {"missing-message", R"({"messages":[{"role":"user","content":""}],"disable_fallback":true})", R"({"choices":[{}]})",
            R"({"detail":{"type":"inference_error","message":"'message'"}})", true, 503},
        {"large", R"({"messages":[{"role":"user","content":""}],"disable_fallback":true})", "{\"choices\":[{\"message\":{\"content\":\" " + large + " </s>discard\"}}]}",
            "{\"choices\":[{\"message\":{\"content\":\"" + large + "\"}}],\"model\":\"org/model-GGUF\"}"},
    };
    for (const auto &test : cases) {
        environment(); const auto directory = root / test.name; fs::create_directories(directory); const int port = available_port();
        std::ofstream(directory / "model.gguf") << "fixture"; std::ofstream(directory / "result.json") << test.output;
        setenv("LLAMA_CPP_MODEL_PATH", (directory / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", directory.c_str(), 1);
        setenv("LLM_FIXTURE_INFERENCE_FILE", (directory / "result.json").c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
        if (test.name == "large") setenv("NATIVE_QUEUE_RESULT_BYTES", "65536", 1);
        {
            ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 15000;
            ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
            const auto response = server.handle_request("POST", test.chat ? "/v1/chat/completions" : "/v1/completions", "", test.body);
            expect(response.status == test.status && response.content_type == test.type && response.body == test.expected,
                test.name + " API: " + std::to_string(response.status) + " " + response.body.substr(0, 500));
        }
        std::istringstream launches(read(directory / ("launch-" + std::to_string(port)))); int count = 0;
        for (int pid; launches >> pid;) { ++count; expect(kill(pid, 0) != 0, "output fixture reaped"); }
        expect(count == 1, test.name + " keeps model loaded after response cleaning");
    }
    unsetenv("LLM_FIXTURE_INFERENCE_FILE"); unsetenv("NATIVE_QUEUE_RESULT_BYTES");
}
}
int main(int argc, char **argv) {
    try {
        expect(argc == 3, "reference and fixture paths"); const auto count = policies(argv[1], argv[2]);
        char path[] = "/tmp/ezlocalai-output-XXXXXX"; const char *directory = mkdtemp(path); expect(directory, "output fixture directory");
        api_cases(directory); fs::remove_all(directory);
        std::cout << count << " source output cases and managed API/large-response cases passed\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
