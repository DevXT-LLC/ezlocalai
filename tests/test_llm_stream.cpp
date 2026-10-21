#include "ezlocalai/llm_stream.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/engine_revision.hpp"
#include <arpa/inet.h>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <regex>
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
            if (at != std::string::npos) expect(hash(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "LLM stream reference changed");
            continue;
        }
        auto c = split(line); ++count;
        for (size_t i = 1; i < c.size(); ++i) c[i] = unhex(c[i]);
        const auto label = "stream case " + std::to_string(count);
        if (c[0] == "H") {
            expect(c.size() == 5, "stream helper fields");
            if (c[2] != "null") expect(ezlocalai::llm_stream_delta(c[1]) == c[2], label + " delta: " + ezlocalai::llm_stream_delta(c[1]) + " expected " + c[2]);
            expect(ezlocalai::llm_stream_has_assistant_text(c[1]) == (c[3] == "true"), label + " assistant text " + c[1]);
            expect(ezlocalai::llm_stream_finish_reason(c[1]) == c[4], label + " finish reason: " + ezlocalai::llm_stream_finish_reason(c[1]) + " expected " + c[4]);
        } else {
            expect(c[0] == "G" && c.size() >= 7, "stream generator fields");
            ezlocalai::LlmChatStreamNormalizer stream(c[1], c[2], c[3], 1700000000, 1700000000);
            std::string actual;
            const auto append = [&](const std::vector<std::string> &values) { for (const auto &value : values) { if (!actual.empty()) actual += '\n'; actual += value; } };
            for (size_t i = 7; i < c.size(); ++i) append(stream.callback(c[i]));
            append(stream.finish());
            expect((stream.failure() ? stream.failure()->category : "OK") == c[4] && (stream.failure() ? stream.failure()->message : "") == c[5], label + " exception");
            expect(actual == c[6], label + " result:\n" + actual + "\nexpected:\n" + c[6]);
            expect(stream.finish().empty() && stream.callback(R"({"content":"late"})").empty(), label + " terminal state");
        }
    }
    expect(count == 4720, "all stream reference cases"); return count;
}

void api_cases(const fs::path &reference, const fs::path &fixture, const fs::path &directory, int upstream_status) {
    fs::create_directories(directory);
    for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
    for (const auto *key : {"LLAMA_CPP_SERVER_URL", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_CONTEXT", "LLAMA_CPP_EXTRA_ARGS", "LLM_FIXTURE_INFERENCE_RESULT", "LLM_FIXTURE_INFERENCE_STATUS"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("EZLOCALAI_API_KEY", "none", 1);
    setenv("DEFAULT_MODEL", "ordinary", 1); setenv("GPU_LAYERS", "0", 1); setenv("LLM_MAX_TOKENS", "512", 1);
    setenv("LLM_PROMPT_CACHE_MIB", "0", 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("LLM_SPECULATIVE_TYPE", "none", 1); setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1);
    setenv("LLM_FIXTURE_INFERENCE_STATUS", std::to_string(upstream_status).c_str(), 1);
    int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "stream fixture bind"); socklen_t size = sizeof(address);
    getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size); close(fd); const int port = ntohs(address.sin_port);
    std::ofstream(directory / "model.gguf") << "fixture";
    setenv("LLAMA_CPP_MODEL_PATH", (directory / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", directory.c_str(), 1);
    setenv("LLM_FIXTURE_INFERENCE_FILE", (directory / "result.json").c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
    size_t count = 0;
    {
        ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 15000;
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
        std::istringstream input(read(fixture));
        for (std::string line; std::getline(input, line);) {
            if (line.empty()) continue;
            if (line.front() == '#') {
                if (line.rfind("# binding", 0) == 0) { expect(line.find(ezlocalai::xllamacpp_reference_revision) != std::string::npos, "callback binding revision changed"); continue; }
                const auto at = line.find(" sha256=");
                if (at != std::string::npos) expect(hash(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "stream API reference changed");
                continue;
            }
            auto c = split(line); expect(c.size() == 5, "stream API fixture fields"); for (auto &field : c) field = unhex(field); ++count;
            std::ofstream(directory / "result.json") << c[1];
            const auto seconds = [] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); };
            const auto start = seconds();
            const auto response = server.handle_request("POST", "/v1/chat/completions", "", "{\"messages\":[{\"role\":\"user\",\"content\":\"\"}],\"disable_fallback\":true,\"stream\":" + c[0] + '}');
            const auto end = seconds();
            // Engine chunks keep their identifiers. Only generated IDs and
            // timestamps vary from the oracle's deliberately frozen clock.
            const std::regex id_pattern("chatcmpl-([0-9]+)"), created_pattern("\"created\": ?([0-9]+)");
            for (const auto *pattern : {&id_pattern, &created_pattern})
                for (auto it = std::sregex_iterator(response.body.begin(), response.body.end(), *pattern); it != std::sregex_iterator(); ++it) {
                    const auto timestamp = std::stoll((*it)[1]);
                    expect(timestamp >= start && timestamp <= end, "generated stream timestamp " + std::to_string(timestamp) + " outside " + std::to_string(start) + ".." + std::to_string(end));
                }
            auto actual = std::regex_replace(response.body, id_pattern, "chatcmpl-1700000000");
            actual = std::regex_replace(actual, std::regex("\"created\": [0-9]+"), "\"created\": 1700000000");
            actual = std::regex_replace(actual, std::regex("\"created\":[0-9]+"), "\"created\":1700000000");
            expect(response.status == std::stoi(c[2]) && response.content_type == c[3] && actual == c[4],
                "stream API case " + std::to_string(count) + ": " + std::to_string(response.status) + " " + actual + "\nexpected " + c[4]);
        }
    }
    expect(count == 48, "all source stream API cases");
    std::istringstream launches(read(directory / ("launch-" + std::to_string(port)))); int count_launches = 0;
    for (int pid; launches >> pid;) { ++count_launches; expect(kill(pid, 0) != 0, "stream fixture reaped"); }
    expect(count_launches == 1, "stream callbacks retain resident engine without retry");
    unsetenv("LLM_FIXTURE_INFERENCE_FILE");
}
}
int main(int argc, char **argv) {
    const auto directory = fs::temp_directory_path() / ("ezlocalai-stream-" + std::to_string(getpid()));
    try {
        expect(argc == 3, "reference and fixture paths");
        const auto count = policies(argv[1], argv[2]);
        for (int status : {200, 500}) api_cases(argv[1], fs::path(argv[2]).parent_path() / "llm_stream_api_cases.tsv", directory / std::to_string(status), status);
        fs::remove_all(directory);
        std::cout << count << " source stream helper/generator cases and 96 managed API cases passed\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << "\nArtifacts: " << directory << '\n'; return 1; }
}
