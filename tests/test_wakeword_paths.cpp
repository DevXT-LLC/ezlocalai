#include "ezlocalai/wakeword_paths.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); check(bool(in), "read " + path.string()); return {std::istreambuf_iterator<char>(in), {}}; }
void write(const fs::path &path, const std::string &value) { std::ofstream out(path, std::ios::binary); out << value; check(bool(out), "write fixture"); }
std::string sha(const std::string &value) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    std::ostringstream out; for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str();
}
void setup(const fs::path &root, const std::string &scenario) {
    fs::create_directories(root / "wakeword_models/safe"); fs::create_directory(root / "outside");
    write(root / "outside/model.pt", "PRIVATE"); write(root / "outside/config.json", R"({"word":"outside"})");
    const auto models = root / "wakeword_models", safe = models / "safe", outside = root / "outside";
    write(safe / "model.pt", "PTMODEL"); write(safe / "config.json", R"({"word":"Safe"})"); write(safe / "model.onnx", "ONNX");
    if (scenario == "unicode") fs::rename(safe, models / "héllo-xt!");
    if (scenario == "internal_alias") fs::create_directory_symlink("safe", models / "alias");
    if (scenario == "external_alias") fs::create_directory_symlink(outside, models / "alias");
    if (scenario == "root_alias") fs::create_directory_symlink(".", models / "alias");
    if (scenario == "sibling_alias") { const auto other = root / "wakeword_models-private"; fs::create_directory(other); write(other / "model.pt", "PRIVATE"); fs::create_directory_symlink(other, models / "alias"); }
    if (scenario == "dangling_alias") fs::create_directory_symlink(outside / "missing", models / "alias");
    if (scenario == "loop_alias") fs::create_symlink("alias", models / "alias");
    if (scenario == "case_collision") fs::rename(safe, models / "SAFE");
    if (scenario == "null_config") write(safe / "config.json", "null");
    if (scenario == "list_config") write(safe / "config.json", "[]");
    if (scenario == "malformed_config") write(safe / "config.json", "{");
    if (scenario.size() > 9 && scenario.substr(scenario.size() - 9) == "_external") {
        const auto name = scenario.substr(0, scenario.size() - 9); fs::remove(safe / name); fs::create_symlink(outside / name, safe / name);
    }
    if (scenario.size() > 9 && scenario.substr(scenario.size() - 9) == "_internal") {
        const auto name = scenario.substr(0, scenario.size() - 9); fs::rename(safe / name, safe / ("inner-" + name)); fs::create_symlink("inner-" + name, safe / name);
    }
}
std::string normalize(std::string value, const std::string &root) {
    size_t pos = 0; while ((pos = value.find(root, pos)) != std::string::npos) { value.replace(pos, root.size(), "<root>"); pos += 6; } return value;
}
int free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); check(fd >= 0, "port socket");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "port bind");
    socklen_t size = sizeof(address); check(getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size) == 0, "port name");
    close(fd); return ntohs(address.sin_port);
}
ezlocalai::HttpResponse request(int port, const Json &row) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); check(fd >= 0, "HTTP socket");
    struct Close { int fd; ~Close() { close(fd); } } closer{fd};
    timeval timeout{5, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
    check(connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "HTTP connect");
    const auto body = row.at("body").get<std::string>();
    const auto raw = row.at("method").get<std::string>() + " " + row.at("path").get<std::string>() + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: " + row.at("content_type").get<std::string>() + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    for (size_t offset = 0; offset < raw.size();) { const auto n = send(fd, raw.data() + offset, raw.size() - offset, MSG_NOSIGNAL); check(n > 0, "HTTP send"); offset += n; }
    std::string response; char buffer[8192];
    for (;;) { const auto n = recv(fd, buffer, sizeof(buffer), 0); check(n >= 0, "HTTP receive"); if (!n) break; response.append(buffer, n); }
    const auto separator = response.find("\r\n\r\n"); check(separator != std::string::npos, "HTTP headers");
    ezlocalai::HttpResponse result; result.status = std::stoi(response.substr(response.find(' ') + 1)); result.body = response.substr(separator + 4);
    const std::string key = "Content-Disposition: "; const auto start = response.find(key);
    if (start != std::string::npos && start < separator) { const auto pos = start + key.size(); result.headers.push_back({"Content-Disposition", response.substr(pos, response.find("\r\n", pos) - pos)}); }
    return result;
}
}
int main(int argc, char **argv) {
    const auto cwd = fs::current_path();
    const auto temporary = fs::temp_directory_path() / ("native-wakeword-paths-" + std::to_string(getpid()));
    try {
        check(argc == 3, "fixture and source root required");
        const auto fixture = Json::parse(read(argv[1]));
        for (const auto &entry : fixture.at("source_sha256").items())
            check(sha(read(fs::path(argv[2]) / entry.key())) == entry.value(), "source changed: " + entry.key());
        for (const auto &row : fixture.at("names")) {
            try { const auto name = ezlocalai::wakeword_name(row.at("word")); check(row.contains("value") && row.at("value") == name, "name value " + row.dump()); }
            catch (const std::invalid_argument &error) { check(row.contains("error") && row.at("error").at("message") == error.what(), "name error " + row.dump()); }
        }
        for (const auto &entry : ezlocalai::environment_defaults) unsetenv(std::string(entry.first).c_str());
        setenv("EZLOCALAI_API_KEY", "", 1); setenv("DEFAULT_MODEL", "model", 1); setenv("VOICE_SERVER", "false", 1);
        setenv("EZLOCALAI_PRELOAD_MODEL_ON_START", "false", 1); setenv("EZLOCALAI_MODEL_STATUS_ON_START", "false", 1);
        for (const auto *cap : {"TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED"}) setenv(cap, "false", 1);
        ezl_env_config config{}; ezl_load_config(&config);
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
        const auto port = free_port(); std::string error;
        check(server.start("127.0.0.1", port, &error), "HTTP listener: " + error);
        size_t index = 0;
        for (const auto &row : fixture.at("http")) {
            fs::current_path(cwd); fs::remove_all(temporary); setup(temporary, row.at("scenario")); fs::current_path(temporary);
            setenv("STT_ENABLED", row.at("enabled").get<bool>() ? "true" : "false", 1);
            const auto response = request(port, row);
            const auto label = "case " + std::to_string(index++) + " " + row.at("scenario").get<std::string>() + " " + row.at("path").get<std::string>();
            check(response.status == row.at("status"), label + " status " + std::to_string(response.status) + " expected " + row.at("status").dump() + " " + response.body);
            const auto &expected = row.at("response");
            if (expected.contains("json")) {
                auto actual = Json::parse(normalize(response.body, temporary.string()));
                if (actual.is_object() && actual.contains("models")) for (auto &model : actual["models"]) model["created_at"] = "<timestamp>";
                check(actual == expected.at("json"), label + " body " + actual.dump() + " expected " + expected.at("json").dump());
            } else check(normalize(response.body, temporary.string()) == expected.at("body"), label + " response body");
            Json disposition = nullptr;
            for (const auto &header : response.headers) if (header.first == "Content-Disposition") disposition = header.second;
            check(disposition == row.at("content_disposition"), label + " content disposition " + disposition.dump());
            check(read(temporary / "outside/model.pt") == "PRIVATE", label + " outside data changed");
            check(fs::exists(temporary / "wakeword_models/safe/model.pt") == row.at("safe_exists"), label + " deletion side effect");
        }
        server.stop();
        size_t enablement_count = 0;
        auto assign = [](const char *name, const Json &value) {
            if (value.is_null()) unsetenv(name);
            else setenv(name, value.get<std::string>().c_str(), 1);
        };
        for (const auto &scenario : fixture.at("enablement")) {
            assign("VOICE_SERVER", scenario[0].at("voice"));
            // Keep unrelated startup warmups disabled. The first endpoint
            // access captures the source's lazy voice-client configuration.
            setenv("STT_ENABLED", "false", 1);
            ezlocalai::NativeServer instance(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
            for (const auto &step : scenario) {
                assign("VOICE_SERVER", step.at("voice")); assign("STT_ENABLED", step.at("stt"));
                const auto response = instance.handle_request("GET", "/v1/wakeword/models", "", "");
                check(response.status == (step.at("enabled").get<bool>() ? 200 : 404),
                    "enablement observation " + std::to_string(enablement_count) + " " + step.dump());
                ++enablement_count;
            }
        }
        fs::current_path(cwd); fs::remove_all(temporary);
        std::cout << fixture.at("names").size() << " Python name cases and " << index << " HTTP cases match\n";
        std::cout << enablement_count << " cached-client and dynamic STT enablement observations match\n";
        return 0;
    } catch (const std::exception &e) { fs::current_path(cwd); std::cerr << e.what() << '\n'; return 1; }
}
