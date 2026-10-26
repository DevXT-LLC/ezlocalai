#include "ezlocalai/llm_input.hpp"
#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <openssl/sha.h>
#include <optional>
#include <sstream>
#include <thread>
#include <unistd.h>
namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::string hex(const std::string &value) { std::ostringstream out; for (unsigned char c : value) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::string hash(const std::string &value) { unsigned char bytes[32]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), bytes); return hex({reinterpret_cast<const char *>(bytes), 32}); }
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::vector<std::string> split(const std::string &value) { std::vector<std::string> out; for (size_t at = 0;;) { const auto end = value.find('\t', at); out.push_back(unhex(value.substr(at, end == std::string::npos ? end : end - at))); if (end == std::string::npos) return out; at = end + 1; } }
std::string quote(const std::string &value) {
    std::string out = "\"";
    for (unsigned char c : value) { if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n"; else if (c == '\r') out += "\\r"; else if (c == '\t') out += "\\t";
        else if (c < 32) out += "\\u00" + hex({static_cast<char>(c)}); else out += c; }
    return out + '"';
}
std::string array(const std::vector<std::string> &values) { std::string out = "["; for (const auto &value : values) { if (out.size() > 1) out += ','; out += value; } return out + ']'; }
size_t policy(const std::optional<fs::path> &reference, const fs::path &fixture) {
    const auto captured = read(fixture);
    expect(!captured.empty(), "input fixture is missing or empty: " + fixture.string());
    std::istringstream input(captured); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') { const auto at = line.find(" sha256="); if (reference && at != std::string::npos) expect(hash(read(*reference / line.substr(2, at - 2))) == line.substr(at + 8), "input reference changed"); continue; }
        const auto c = split(line); expect(c.size() == 8, "input fixture fields"); ++count;
        std::vector<std::string> events;
        ezlocalai::LlmInputOperations operations;
        operations.local_uri = "http://local";
        operations.load = [&] { events.push_back("[\"load\"]"); return ezlocalai::LlmInputModel{"source", c[2] == "true"}; };
        operations.describe = [&](const auto &images, const auto &text) {
            events.push_back("[\"describe\"," + array(images) + ',' + quote(text) + ']');
            if (c[3] != "\"__helper__\"") return c[3];
            events.push_back("[\"load\"]");
            try {
                auto prepared = ezlocalai::llm_prepare_images(images, operations, true);
                if (prepared.empty()) return std::string("null");
                prepared.insert(prepared.begin(), "{\"type\":\"text\",\"text\":" + quote("Describe the contents of the image(s) in detail. The user's question about the image(s) is: " + text) + '}');
                events.push_back("[\"chat\",{\"messages\":[{\"role\":\"user\",\"content\":" + array(prepared) + "}],\"local_uri\":\"http://local\",\"max_tokens\":2048,\"temperature\":0.3}]");
                return std::string("\"a view\"");
            } catch (...) { return std::string("null"); }
        };
        operations.fetch_image = [&](const auto &url, const auto &agent) {
            events.push_back("[\"fetch\"," + quote(url) + ',' + quote(agent) + ']');
            const auto suffix = url.substr(url.rfind('/') + 1);
            if (suffix == "fail") throw std::runtime_error("fetch failed");
            const auto type = suffix == "missing" ? "image/jpeg" : suffix == "text" ? "text/plain" : suffix == "empty" ? ""
                : suffix == "caps" ? "Image/WEBP" : suffix == "parameter" ? "image/webp; q=1" : "image/" + suffix;
            return ezlocalai::LlmInputImage{url.find("/bad/") == std::string::npos ? "image bytes" : "bad image", type};
        };
        operations.png = [&](const std::string &bytes) {
            events.push_back("[\"png\"," + quote(hex(bytes)) + ']');
            if (bytes.rfind("bad", 0) == 0) throw std::runtime_error("invalid image");
            return "PNG:" + bytes;
        };
        std::string result = "null", category = "OK", message;
        try { expect(ezlocalai::llm_text_image_input_supported(c[0]), "fixture in text/image scope"); result = ezlocalai::llm_prepare_text_image_input(c[0], c[1] == "true", operations); }
        catch (const ezlocalai::LlmResponseError &error) { category = error.category; message = error.message; }
        const auto label = "input " + std::to_string(count) + " " + c[0];
        expect(quote(category) == c[4] && quote(message) == c[5], label + " error: " + quote(category) + ' ' + quote(message) + " expected " + c[4] + ' ' + c[5]);
        expect(result == c[6], label + " output: " + result + " expected " + c[6]);
        expect(array(events) == c[7], label + " order: " + array(events) + " expected " + c[7]);
    }
    expect(count == 3399, "all source input cases");
    for (const auto &value : {R"({"audio_format":null})", R"({"messages":[{"content":[{"audio_url":"data:audio/wav;base64,eA=="}]}]})"})
        expect(ezlocalai::llm_text_image_input_supported(value), "audio uses queued preparation");
    expect(ezlocalai::llm_text_image_input_supported(R"({"messages":[{"content":[{"type":"video"}]}]})"), "video uses queued preparation");
    return count;
}
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const auto *key : {"LLAMA_CPP_CONTEXT", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_MMPROJ_PATH", "LLM_FIXTURE_INFERENCE_RESULT", "LLM_FIXTURE_INFERENCE_FILE"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("EZLOCALAI_API_KEY", "none", 1); setenv("GPU_LAYERS", "0", 1); setenv("DEFAULT_MODEL", "ordinary", 1);
    setenv("LLM_MAX_TOKENS", "512", 1); setenv("LLM_PROMPT_CACHE_MIB", "0", 1); setenv("SPECULATIVE_TYPE", "none", 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
}
int available_port() { return fixture_ports(); }
class ImageServer {
    int fd_ = -1;
    std::thread thread_;
    std::atomic<bool> running_{true};
    std::mutex mutex_;
    std::string requests_;
public:
    std::string url;
    ImageServer() {
        fd_ = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(!bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) && !listen(fd_, 8), "image server bind");
        socklen_t size = sizeof(address); getsockname(fd_, reinterpret_cast<sockaddr *>(&address), &size); url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port));
        thread_ = std::thread([this] {
            while (running_) {
                int peer = accept(fd_, nullptr, nullptr); if (peer < 0) break;
                timeval timeout{2, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
                std::string request; char buffer[1024]; size_t expected = std::string::npos;
                while (request.size() < expected) {
                    auto n = recv(peer, buffer, sizeof(buffer), 0); if (n <= 0) break; request.append(buffer, n);
                    const auto end = request.find("\r\n\r\n");
                    if (end != std::string::npos) { const auto length = request.find("Content-Length: "); expected = end + 4 + (length == std::string::npos ? 0 : std::stoul(request.substr(length + 16))); }
                }
                { std::lock_guard<std::mutex> lock(mutex_); requests_ += request; }
                const bool fail = request.rfind("GET /fail ", 0) == 0, bad = request.rfind("GET /bad ", 0) == 0, redirect = request.rfind("GET /redirect ", 0) == 0;
                const bool api = request.find(" /v1/") != std::string::npos;
                const std::string body = api ? (request.rfind("GET ", 0) == 0 ? R"({"data":[{"id":"ordinary"}]})" : R"({"choices":[{"message":{"content":"fallback"}}]})") : bad ? "bad" : "image bytes";
                const auto wire = "HTTP/1.1 " + std::string(fail ? "404 Missing" : redirect ? "302 Found" : "200 OK")
                    + "\r\nContent-Type: " + (api ? "application/json" : bad ? "image/webp" : "text/plain") + "\r\n"
                    + (redirect ? "Location: /good\r\n" : "") + "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
                for (size_t at = 0; at < wire.size();) { auto n = send(peer, wire.data() + at, wire.size() - at, MSG_NOSIGNAL); if (n <= 0) break; at += n; }
                close(peer);
            }
        });
    }
    ~ImageServer() { running_ = false; shutdown(fd_, SHUT_RDWR); close(fd_); thread_.join(); }
    std::string requests() { std::lock_guard<std::mutex> lock(mutex_); return requests_; }
};
void managed(const fs::path &root) {
    struct Case { std::string body, expected; bool load = true, vision = false; };
    const std::vector<Case> cases = {
        {R"({"messages":[]})", "", false}, {R"({"messages":[],"stream":true})", "", false},
        {R"({"messages":[{}]})", "", false}, {R"({"messages":[{"content":[{"type":"text","text":null}]}]})", "", false},
        {R"({})", ""}, {R"({"messages":[{"content":[{"image_url":"data:image/png;base64,eA=="}]},{}]})", ""},
        {R"({"messages":[{"content":["a",{"type":"text","text":"b"},7,{},"c"]}]})", R"("messages":[{"content":"abc"}])"},
        {R"({"messages":[{"content":[{"image_url":"data:image/png;base64,eA=="},"before"]},{"content":["after"]}]})", R"("messages":[{"content":"before"},{"content":"after"}])"},
        {R"({"messages":[{"content":[{"image_url":"data:image/png;base64,eA=="},"before"]},{"content":["after"]}]})", R"("messages":[{"content":"before"},{"content":[{"type":"text","text":"after"},{"image_url":"data:image/png;base64,eA=="}]}])", true, true},
        {R"({"messages":[{"content":[{"image_url":false}]}]})", "", true, true},
    };
    size_t count = 0;
    for (const auto &item : cases) {
        environment(); const auto folder = root / std::to_string(count++); fs::create_directories(folder);
        const auto port = available_port(); const auto suffix = '-' + std::to_string(port);
        std::ofstream(folder / "model.gguf") << "fixture";
        setenv("LLAMA_CPP_MODEL_PATH", (folder / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", folder.c_str(), 1);
        setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
        if (item.vision) setenv("LLAMA_CPP_MMPROJ_PATH", "projector.gguf", 1);
        {
            ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 5000;
            ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
            const auto response = server.handle_request("POST", "/v1/chat/completions", "", item.body);
            expect(response.status == (item.expected.empty() ? 500 : 200), "managed input status " + std::to_string(count) + ' ' + response.body);
            expect(fs::exists(folder / ("launch" + suffix)) == item.load, "managed input load order " + std::to_string(count));
            const auto seen = read(folder / ("seen" + suffix));
            if (item.expected.empty()) expect(response.body == R"({"detail":"Internal server error"})" && seen.find("POST /v1/chat/completions") == std::string::npos, "preprocessing error bypasses engine and post-inference fallback");
            else expect(seen.find(item.expected) != std::string::npos, "managed prepared payload " + std::to_string(count) + ' ' + seen);
            if (item.expected.empty()) {
                const auto retry = server.handle_request("POST", "/v1/chat/completions", "", R"({"messages":[{"content":"recovered"}]})");
                expect(retry.status == 200, "preprocessing failure releases model and replica for the next request");
            }
        }
        std::istringstream pids(read(folder / ("launch" + suffix))); for (int pid; pids >> pid;) expect(kill(pid, 0) != 0, "input fixture reaped");
    }
}
void images(const fs::path &root) {
    environment(); fs::create_directories(root); ImageServer remote;
    const int port = available_port(); const auto suffix = '-' + std::to_string(port);
    std::ofstream(root / "model.gguf") << "fixture";
    setenv("LLAMA_CPP_MODEL_PATH", (root / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", root.c_str(), 1);
    setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1); setenv("LLAMA_CPP_MMPROJ_PATH", "projector.gguf", 1);
    {
        ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 5000;
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
        for (const std::string path : {"/good", "/redirect", "/fail", "/bad"}) {
            std::ofstream(root / ("seen" + suffix));
            const auto body = "{\"messages\":[{\"content\":[{\"type\":\"text\",\"text\":\"look\"},{\"image_url\":{\"url\":" + quote(remote.url + path) + ",\"detail\":\"high\"}}]}]}";
            const auto result = server.handle_request("POST", "/v1/chat/completions", "", body);
            expect(result.status == 200, "image fetch API status");
            const auto seen = read(root / ("seen" + suffix));
            if (path == "/fail") expect(seen.find(R"("messages":[{"content":"look"}])") != std::string::npos, "failed remote image falls back to text");
            else expect(seen.find(path == "/bad" ? "data:image/webp;base64,YmFk" : "data:image/jpeg;base64,aW1hZ2UgYnl0ZXM=") != std::string::npos && seen.find("detail") == std::string::npos, "remote image normalization and conversion-failure fallback");
        }
    }
    expect(remote.requests().find("User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)") != std::string::npos, "direct vision fetch user agent");
}
void description(const fs::path &root) {
    environment(); fs::create_directories(root / "models/text"); fs::create_directories(root / "models/vision");
    std::ofstream(root / "models/text/model-Q4_K_XL.gguf") << "fixture";
    std::ofstream(root / "models/vision/model-Q4_K_XL.gguf") << "fixture";
    std::ofstream(root / "models/vision/mmproj.gguf") << "projector";
    struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); unsetenv("HF_HUB_OFFLINE"); } } restore;
    fs::current_path(root); ImageServer remote;
    unsetenv("LLAMA_CPP_MODEL_PATH"); setenv("DEFAULT_MODEL", "org/text,org/vision", 1); setenv("MODELS_PATH", "models", 1);
    setenv("HF_HUB_OFFLINE", "1", 1); setenv("EMBEDDING_FIXTURE_DIR", root.c_str(), 1);
    setenv("LLM_MAX_TOKENS", "512,32768", 1);
    const int port = fixture_ports(2); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
    const auto text_seen = root / ("seen-" + std::to_string(port)), vision_seen = root / ("seen-" + std::to_string(port + 1));
    {
        ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 5000;
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
        for (const std::string path : {"/good", "/bad"}) {
            std::ofstream{text_seen}; std::ofstream{vision_seen};
            const auto body = "{\"messages\":[{\"role\":\"user\",\"content\":[\"question\",{\"image_url\":" + quote(remote.url + path) + "}]}]}";
            const auto result = server.handle_request("POST", "/v1/chat/completions", "", body);
            expect(result.status == 200, "description API response " + result.body);
            if (path == "/good") {
                expect(read(vision_seen).find("Describe the contents of the image(s) in detail. The user's question about the image(s) is: question") != std::string::npos, "description prompt reaches selected vision engine");
                expect(read(vision_seen).find(R"("max_tokens":2048,"temperature":0.3,"top_p":0.95)") != std::string::npos, "description uses wrapper defaults without normal model profiles");
                expect(read(text_seen).find(R"([Image Description: fixture]\n\nUser's question: question)") != std::string::npos, "description enriches captured text model request");
            } else expect(read(vision_seen).empty() && read(text_seen).find(R"("content":"question")") != std::string::npos, "description conversion failure drops image and skips vision inference");
        }
        expect(read(root / ("args-" + std::to_string(port + 1))).find("-c\n16384\n") != std::string::npos, "description context is 16384");
    }
    expect(remote.requests().find("User-Agent: Mozilla/5.0\r\n") != std::string::npos, "description fetch user agent");
    for (const int child_port : {port, port + 1}) { std::istringstream pids(read(root / ("launch-" + std::to_string(child_port)))); for (int pid; pids >> pid;) expect(kill(pid, 0) != 0, "vision/text children reaped"); }
}
void prepared_fallback(const fs::path &root) {
    environment(); fs::create_directories(root); ImageServer remote;
    std::ofstream(root / "model.gguf") << "fixture";
    setenv("LLAMA_CPP_MODEL_PATH", (root / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", root.c_str(), 1);
    setenv("LLAMA_CPP_SERVER_PORT", std::to_string(available_port()).c_str(), 1);
    setenv("LLM_FIXTURE_INFERENCE_RESULT", R"({"error":{"message":"fixture inference failed"}})", 1);
    setenv("FALLBACK_SERVER", remote.url.c_str(), 1);
    ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 5000;
    ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
    const auto result = server.handle_request("POST", "/v1/chat/completions", "", R"({"messages":[{"content":["before",{"type":"text","text":"after"},{"image_url":"data:image/png;base64,eA=="}]}],"disable_fallback":true})");
    expect(result.status == 200 && result.body.find("fallback") != std::string::npos, "prepared request post-inference fallback");
    expect(remote.requests().find(R"("messages":[{"content":"beforeafter"}])") != std::string::npos, "post-inference fallback receives prepared messages");
}
void duplicate_outer_fields() {
    std::vector<std::string> events;
    ezlocalai::LlmInputOperations operations;
    operations.local_uri = "http://local";
    operations.load = [&] { events.push_back("load"); return ezlocalai::LlmInputModel{"source", false, R"(["BASE"])"}; };
    operations.transcribe = [&](const std::string &audio, const std::string &format) {
        events.push_back("transcribe:" + audio + ':' + format);
        return std::string(R"("transcribed")");
    };
    // Python dict assignment keeps the first decoded-key position and the last
    // value. The later model/local URI assignments also preserve that position.
    const auto prepared = ezlocalai::llm_prepare_text_image_input(
        R"({"x":0,"\u0078":2,"messages":[{"content":"old"}],"\u006dessages":[{"content":"last"}],"local_uri":"old","\u006cocal_uri":"new","model":"old","\u006dodel":"new","stop":["A"],"\u0073top":["B"]})", true, operations);
    expect(prepared == R"({"x":2,"messages":[{"content":"last"}],"local_uri":"http://local","model":"source","stop":["BASE","B"]})",
        "duplicate and escaped outer keys retain values and output order");
    expect(events == std::vector<std::string>{"load"}, "absent audio format does not transcribe");
    events.clear();
    const auto audio = ezlocalai::llm_prepare_text_image_input(
        R"({"audio_format":"wav","\u0061udio_format":null,"messages":[{"content":"eA=="}]})", true, operations);
    expect(audio == R"({"audio_format":null,"messages":[{"content":"transcribed"}],"local_uri":"http://local","model":"source"})",
        "last duplicate audio format is preserved, including null");
    expect(events == std::vector<std::string>{"load", "transcribe:\"eA==\":null"},
        "present null audio format transcribes after model load");
}
}
int main(int argc, char **argv) {
    try { expect(argc == 3, "usage: test_llm_input REFERENCE_ROOT|--fixtures FIXTURE_FILE");
        // Packaged images carry captures, not the Python checkout. Only this
        // explicit mode skips source hash checks; every captured expectation,
        // case-count guard and managed regression still runs.
        const auto reference = std::string(argv[1]) == "--fixtures" ? std::optional<fs::path>{} : fs::path(argv[1]);
        const auto count = policy(reference, argv[2]); duplicate_outer_fields(); char path[] = "/tmp/ezlocalai-input-XXXXXX"; const auto root = mkdtemp(path); expect(root, "temporary directory"); managed(root); images(fs::path(root) / "remote"); description(fs::path(root) / "description"); prepared_fallback(fs::path(root) / "fallback"); fs::remove_all(root); std::cout << count << " recorded input cases and managed preparation/load/error/image/description/fallback checks passed\n"; return 0; }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
