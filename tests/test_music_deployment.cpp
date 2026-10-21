#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/music_runtime.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/stt_input.hpp"

#include <arpa/inet.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) {
    std::ifstream in(path, std::ios::binary); check(bool(in), "read " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::string digest(std::istream &input) {
    auto *context = EVP_MD_CTX_new(); check(context != nullptr, "digest context");
    EVP_DigestInit_ex(context, EVP_sha256(), nullptr);
    char buffer[1024 * 1024];
    while (input) { input.read(buffer, sizeof(buffer)); EVP_DigestUpdate(context, buffer, input.gcount()); }
    unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned count = 0;
    EVP_DigestFinal_ex(context, bytes, &count); EVP_MD_CTX_free(context);
    std::ostringstream result;
    for (unsigned i = 0; i < count; ++i) result << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
    return result.str();
}
std::string hash(const std::string &bytes) { std::istringstream input(bytes); return digest(input); }
void stop_engine() {
    const int pid = ezlocalai::ace_step_runtime().process_id();
    if (pid > 0) { kill(pid, SIGTERM); int status; while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} }
}
struct Cleanup {
    fs::path previous = fs::current_path(), directory;
    ~Cleanup() { stop_engine(); fs::current_path(previous); if (!directory.empty()) fs::remove_all(directory); }
};
int unused_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); check(fd >= 0, "socket");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "bind");
    socklen_t length = sizeof(address); check(getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) == 0, "port");
    close(fd); return ntohs(address.sin_port);
}
}

int main(int argc, char **argv) {
    try {
        check(argc == 4, "usage: test_music_deployment fixture models-directory ace-server");
        const auto fixture = Json::parse(read(fs::absolute(argv[1])));
        const auto models = fs::absolute(argv[2]).string(), engine = fs::absolute(argv[3]).string();
        for (const auto &[file, expected] : fixture.at("models").items()) {
            std::ifstream input(fs::path(models) / file, std::ios::binary); check(bool(input), "missing model " + file);
            check(digest(input) == expected.get<std::string>(), "model checksum " + file);
        }
        Cleanup cleanup;
        char temporary[] = "/tmp/ezlocalai-music-model-XXXXXX";
        check(mkdtemp(temporary) != nullptr, "temporary directory"); cleanup.directory = temporary; fs::current_path(temporary);
        for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
        for (const char *key : {"ACE_STEP_EXTRA_ARGS", "ACE_STEP_KEEP_LOADED", "ACE_STEP_ADAPTERS_DIR", "ACE_STEP_MAX_BATCH", "ACE_STEP_MAX_SEQ", "ACE_STEP_VAE_CHUNK", "ACE_STEP_VAE_OVERLAP", "ROUTER_MODE"}) unsetenv(key);
        for (const auto &[key, value] : std::vector<std::pair<const char *, const char *>>{
            {"EZLOCALAI_DOTENV_PATH", ""}, {"EZLOCALAI_API_KEY", ""}, {"EZLOCALAI_URL", ""}, {"ROUTER_DRAIN_FILE", ""}, {"DEFAULT_MODEL", "none"},
            {"MUSIC_ENABLED", "true"}, {"STT_ENABLED", "false"}, {"TTS_ENABLED", "false"}, {"EMBEDDING_ENABLED", "false"}, {"ACE_STEP_POLL_INTERVAL", "0.1"}}) setenv(key, value, 1);
        setenv("ACE_STEP_MODELS_DIR", models.c_str(), 1); setenv("ACE_STEP_BIN", engine.c_str(), 1);
        setenv("ACE_STEP_PORT", std::to_string(unused_port()).c_str(), 1);
        ezlocalai::precache_music_models();
        Json previous_environment = Json::object();
        size_t buffers = 0;
        for (const auto &item : fixture.at("cases")) {
            const auto &environment = item.at("server_environment");
            if (environment != previous_environment) {
                stop_engine();
                for (const auto &[key, value] : previous_environment.items()) unsetenv(key.c_str());
                for (const auto &[key, value] : environment.items()) setenv(key.c_str(), value.get<std::string>().c_str(), 1);
                previous_environment = environment;
            }
            ezl_env_config config{}; ezl_load_config(&config);
            ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>("none"));
            auto request = item.at("request");
            if (request.contains("extra")) { request["extra_parameters"] = request.at("extra"); request.erase("extra"); }
            const auto response = server.handle_request("POST", "/v1/audio/music", "", request.dump(), "application/json");
            check(response.status == 200, "music status " + std::to_string(response.status) + ": " + response.body.substr(0, 1000));
            auto actual = Json::parse(response.body), expected = item.at("response");
            actual.erase("created"); expected.erase("created");
            check(actual.at("data").size() == item.at("audio").size(), "audio count");
            for (size_t i = 0; i < actual.at("data").size(); ++i) {
                auto &audio = actual["data"][i]; std::string bytes;
                if (audio.contains("b64_json")) { bytes = ezlocalai::stt_base64_decode_bytes(audio.at("b64_json").get<std::string>()); audio.erase("b64_json"); }
                else {
                    const auto url = audio.at("url").get<std::string>(); const auto offset = url.find("outputs/");
                    check(offset != std::string::npos, "output URL"); bytes = read(url.substr(offset));
                    audio.erase("url"); expected["data"][i].erase("url");
                }
                check(bytes.size() == item["audio"][i]["bytes"].get<size_t>(), "audio byte count");
                check(hash(bytes) == item["audio"][i]["sha256"].get<std::string>(), "audio checksum for buffer " + std::to_string(buffers) + ": " + hash(bytes));
                ++buffers;
            }
            check(actual == expected, "music metadata: " + actual.dump());
        }
        std::cout << "Verified " << fixture.at("cases").size() << " real ACE-Step requests and " << buffers << " exact audio buffers\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
