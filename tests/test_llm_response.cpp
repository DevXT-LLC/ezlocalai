#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/llm_parameters.hpp"
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
std::string read(const fs::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::string hash(const std::string &s) { unsigned char digest[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), digest); std::ostringstream out; for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::vector<std::string> split(const std::string &s) { std::vector<std::string> out; for (size_t begin = 0;;) { const auto end = s.find('\t', begin); out.push_back(s.substr(begin, end == std::string::npos ? end : end - begin)); if (end == std::string::npos) return out; begin = end + 1; } }
std::string unhex(const std::string &s) { std::string out; for (size_t i = 0; i < s.size(); i += 2) out += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)); return out; }
size_t policy_cases(const fs::path &reference, const fs::path &fixture) {
    std::istringstream in(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto at = line.find(" sha256="); if (at != std::string::npos) expect(hash(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "response reference hash"); continue; }
        const auto c = split(line); ++count;
        expect(c.size() == 4, "response fixture fields");
        if (c[0] == "A") { expect(ezlocalai::llm_stream_error_event(unhex(c[1])) == unhex(c[3]), "source API stream event " + std::to_string(count)); continue; }
        const auto result = ezlocalai::llm_response_error(unhex(c[1]), c[0] == "S");
        const auto category = result ? result->category : "OK";
        const auto message = result ? result->message : "";
        expect(category == c[2] && message == unhex(c[3]), "response case " + std::to_string(count) + ": " + category + ": " + message + " expected " + c[2] + ": " + unhex(c[3]));
    }
    expect(count == 2250, "all source response cases"); return count;
}
void environment() {
    for (const auto &item : ezlocalai::environment_defaults) unsetenv(item.first.data());
    for (const auto *key : {"LLAMA_CPP_CONTEXT", "LLAMA_CPP_DEFAULT_CONTEXT", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_MMPROJ_PATH", "LLM_FIXTURE_INFERENCE_RESULT", "LLM_FIXTURE_INFERENCE_STATUS", "EMBEDDING_FIXTURE_MAX_LAYERS", "LLAMA_CPP_TOTAL_LAYERS"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("GPU_LAYERS", "0", 1); setenv("LLM_PROMPT_CACHE_MIB", "0", 1);
}
int available_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)), "context fixture bind"); socklen_t size = sizeof(a); getsockname(fd, reinterpret_cast<sockaddr *>(&a), &size); close(fd); return ntohs(a.sin_port);
}
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
    struct Case { std::string name, result, request, message; int status, launches; bool failure, lazy; };
    const std::string memory = R"({"error":{"message":"CUDA out of memory"}})";
    const std::string prefix = "data: {\"choices\": [{\"delta\": {\"content\": \"before\"}}]}\n\n";
    const std::vector<Case> cases{
        {"buffered-200", memory, R"({"messages":[]})", "CUDA out of memory",200,4,true,false},
        {"buffered-500", memory, R"({"messages":[]})", "CUDA out of memory",500,4,true,false},
        {"stream-memory", memory, R"({"messages":[],"stream":true})", "CUDA out of memory",200,1,true,true},
        {"stream-false-string", memory, R"({"messages":[],"stream":"false"})", "CUDA out of memory",200,1,true,true},
        {"context-counts", R"({"code":400,"message":"too many tokens","n_prompt_tokens":90000,"n_ctx":512})", R"({"messages":[]})", "too many tokens [n_prompt_tokens=90000, n_ctx=512]",200,1,true,false},
        {"buffered-falsey", R"({"error":false})", R"({"messages":[]})", "",200,1,false,false},
        {"stream-falsey", R"({"error":false})", R"({"messages":[],"stream":true})", "False",200,1,true,true},
        {"stream-prefix", prefix + "data: " + memory + "\n\ndata: [DONE]\n\n", R"({"messages":[],"stream":true})", "CUDA out of memory",200,1,true,true},
        {"stream-success", prefix + "data: [DONE]\n\n", R"({"messages":[],"stream":true})", "",200,1,false,false},
        {"completion-stream", memory, R"({"prompt":"hello","stream":true})", "Server.handle_completions requires a callback for streaming or a non dict prompt.",200,1,true,false},
    };
    for (const auto &test : cases) {
        environment(); Fixture run(root,test.name); setenv("GPU_LAYERS","41",1);
        setenv("LLM_FIXTURE_INFERENCE_RESULT",test.result.c_str(),1); setenv("LLM_FIXTURE_INFERENCE_STATUS",std::to_string(test.status).c_str(),1);
        { ezlocalai::LlamaCppInferenceBackend backend("ordinary");
          auto result = backend.text_result(test.request,"",test.name != "completion-stream");
          const auto expected_body = test.lazy ? (test.name == "stream-prefix" ? prefix : "") + ezlocalai::llm_stream_error_event(test.message)
              : test.name == "buffered-falsey" ? std::string(R"({"error":false,"model":"ordinary"})") : test.result;
          expect(result.from_engine && (test.name == "completion-stream" || test.name == "stream-success" || result.body == expected_body), test.name + " preserves errors and normalizes buffered success");
          if (test.name == "stream-success") expect(result.body.rfind(prefix, 0) == 0 && result.body.find("\"finish_reason\": \"stop\"") != std::string::npos && result.body.find("data: [DONE]\n\n") != std::string::npos, "stream normalizes final stop; full output covered by generator oracle");
          expect(result.failure.has_value() == test.failure && result.lazy_chat_stream == test.lazy, test.name + " failure phase");
          if (test.failure) expect(result.failure->category == (test.name == "completion-stream" ? "ValueError" : "Exception") && result.failure->message == test.message, test.name + " exception");
          expect(run.launches() == test.launches, test.name + " memory retry count: " + std::to_string(run.launches()));
          expect(result.stream_prefix == (test.name == "stream-prefix" ? prefix : ""), test.name + " prefix before error");
        } run.reaped();
    }
    environment(); Fixture startup(root,"startup"); setenv("LLM_MAX_TOKENS","2147483648",1);
    { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); const auto result=backend.text_result(R"({"messages":[]})","",true);
      expect(!result.from_engine && !result.failure && !result.lazy_chat_stream && startup.launches() == 0 && result.body.find("value too large") != std::string::npos, "startup errors remain distinct from inference failures");
    }
}
void api_cases(const fs::path &root) {
    for (const std::string stream : {"true", "\"false\"", "\"off\""}) {
        environment(); Fixture run(root,"api-" + stream); setenv("GPU_LAYERS","16",1);
        setenv("DEFAULT_MODEL","ordinary",1); setenv("EZLOCALAI_API_KEY","none",1);
        const std::string prefix = "data: {\"choices\": [{\"delta\": {\"content\": \"before\"}}]}\n\n";
        const std::string failure=prefix+"data: {\"error\":{\"message\":\"CUDA out of memory\"}}\n\ndata: [DONE]\n\n";
        setenv("LLM_FIXTURE_INFERENCE_RESULT",failure.c_str(),1);
        { ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms=5000;
          ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
          const auto response=server.handle_request("POST","/v1/chat/completions","","{\"messages\":[{\"role\":\"user\",\"content\":\"\"}],\"disable_fallback\":true,\"stream\":"+stream+"}");
          if (stream == "true") expect(response.status == 200 && response.chunked && response.content_type == "text/event-stream; charset=utf-8"
              && response.body == prefix + ezlocalai::llm_stream_error_event("CUDA out of memory"), "managed API yields prefix and error without DONE");
          else expect(response.status == 500 && response.body == "Internal Server Error", "raw truthy validated false lazy failure");
          expect(run.launches() == 1,"lazy API errors do not reload");
        } run.reaped();
    }
}
}
int main(int argc,char **argv) {
    const auto root=fs::temp_directory_path()/("ezlocalai-response-"+std::to_string(getpid()));
    try {
        expect(argc==3,"reference and fixture arguments"); const auto count=policy_cases(argv[1],argv[2]);
        fs::create_directories(root); managed_cases(root); api_cases(root); fs::remove_all(root);
        std::cout << count << " source error cases and managed response phase/retry cases passed\n"; return 0;
    } catch(const std::exception &error) { std::cerr << error.what() << "\nArtifacts: " << root << '\n'; return 1; }
}
