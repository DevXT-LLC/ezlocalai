#include "ezlocalai/music_runtime.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ezlocalai_c.h"
#include "ezlocalai/hub_assets.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>

extern char **environ;

namespace ezlocalai {
namespace {
std::string setting(const char *name, const char *fallback = "") {
    const std::string value = ezl_getenv_fallback(name, fallback);
    return value.empty() ? fallback : value;
}
std::string lower(std::string value) {
    for (auto &c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}
bool truthy(const char *name, const char *fallback = "false") {
    const auto value = lower(strip_config_value(setting(name, fallback)));
    return value == "1" || value == "true" || value == "yes" || value == "on";
}
bool false_token(const std::string &value) {
    return value == "0" || value == "false" || value == "no" || value == "off";
}

// shlex.split(posix=True, comments=False): empty quoted words survive and a
// backslash in double quotes only escapes a quote or another backslash.
std::vector<std::string> shell_words(const std::string &value) {
    std::vector<std::string> words;
    std::string word;
    char quote = 0;
    bool started = false;
    for (size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];
        if (quote) {
            if (c == quote) { quote = 0; continue; }
            if (c == '\\' && quote == '"') {
                if (++i == value.size()) throw std::invalid_argument("No escaped character");
                if (value[i] != '"' && value[i] != '\\') word += '\\';
                word += value[i];
            } else word += c;
        } else if (c == '\'' || c == '"') {
            quote = c;
            started = true;
        } else if (c == '\\') {
            if (++i == value.size()) throw std::invalid_argument("No escaped character");
            word += value[i];
            started = true;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (started) { words.push_back(word); word.clear(); started = false; }
        } else { word += c; started = true; }
    }
    if (quote) throw std::invalid_argument("No closing quotation");
    if (started) words.push_back(word);
    return words;
}
} // namespace

std::string ace_step_internal_url() {
    return "http://" + strip_config_value(setting("ACE_STEP_HOST", "127.0.0.1"))
        + ":" + strip_config_value(setting("ACE_STEP_PORT", "8085"));
}
std::string ace_step_server_url() {
    auto url = strip_config_value(setting("ACE_STEP_SERVER_URL"));
    if (url.empty()) return ace_step_internal_url();
    while (!url.empty() && url.back() == '/') url.pop_back();
    return url;
}
std::string ace_step_models_dir() { return strip_config_value(setting("ACE_STEP_MODELS_DIR", "models/ace-step")); }
std::string ace_step_binary_path() { return strip_config_value(setting("ACE_STEP_BIN", "/opt/acestep.cpp/build/ace-server")); }
std::vector<std::string> selected_ace_step_model_files() {
    std::vector<std::string> result;
    for (const auto &entry : {std::pair{"ACE_STEP_TEXT_ENCODER_MODEL", "Qwen3-Embedding-0.6B-Q8_0.gguf"},
             {"ACE_STEP_LM_MODEL", "acestep-5Hz-lm-4B-Q8_0.gguf"}, {"ACE_STEP_DIT_MODEL", "acestep-v15-turbo-Q4_K_M.gguf"},
             {"ACE_STEP_VAE_MODEL", "vae-BF16.gguf"}}) {
        const auto value = strip_config_value(ezl_getenv_fallback(entry.first, entry.second));
        if (!value.empty() && std::find(result.begin(), result.end(), value) == result.end()) result.push_back(value);
    }
    return result;
}
std::vector<std::string> build_ace_step_server_command() {
    std::vector<std::string> command = {ace_step_binary_path(), "--models", ace_step_models_dir(),
        "--host", setting("ACE_STEP_HOST", "127.0.0.1"), "--port", setting("ACE_STEP_PORT", "8085")};
    const auto adapters = strip_config_value(setting("ACE_STEP_ADAPTERS_DIR"));
    if (!adapters.empty()) { command.push_back("--adapters"); command.push_back(adapters); }
    if (truthy("ACE_STEP_KEEP_LOADED")) command.push_back("--keep-loaded");
    for (const auto &entry : {std::pair{"--max-batch", "ACE_STEP_MAX_BATCH"}, {"--max-seq", "ACE_STEP_MAX_SEQ"},
             {"--vae-chunk", "ACE_STEP_VAE_CHUNK"}, {"--vae-overlap", "ACE_STEP_VAE_OVERLAP"}}) {
        const auto value = strip_config_value(setting(entry.second));
        if (!value.empty()) { command.push_back(entry.first); command.push_back(value); }
    }
    const auto extra = shell_words(strip_config_value(setting("ACE_STEP_EXTRA_ARGS")));
    command.insert(command.end(), extra.begin(), extra.end());
    return command;
}
void precache_music_models(MusicPrecacheOperations operations) {
    if (lower(strip_config_value(setting("MUSIC_ENABLED", "false"))) != "true"
        || false_token(lower(strip_config_value(setting("ACE_STEP_PRECACHE", "true"))))
        || !strip_config_value(setting("ACE_STEP_SERVER_URL")).empty()) return;
    try {
        const auto repository = strip_config_value(setting("ACE_STEP_MODEL_REPO", "Serveurperso/ACE-Step-1.5-GGUF"));
        const auto directory = ace_step_models_dir();
        const auto files = selected_ace_step_model_files();
        if (repository.empty() || files.empty()) return;
        if (!operations.mkdir) operations.mkdir = [](const auto &path) { std::filesystem::create_directories(path); };
        if (!operations.exists) operations.exists = [](const auto &path) { return std::filesystem::exists(path); };
        if (!operations.download) operations.download = [](const auto &repo, const auto &file, const auto &dir) { download_hub_file(repo, file, dir); };
        operations.mkdir(directory);
        size_t cached = 0, downloaded = 0;
        for (const auto &file : files) {
            if (operations.exists((std::filesystem::path(directory) / file).string())) ++cached;
            else { operations.download(repository, file, directory); ++downloaded; }
        }
        std::cerr << "[MUSIC] ACE-Step models: " << downloaded << " downloaded, " << cached << " cached\n";
    } catch (const std::exception &error) {
        std::cerr << "[MUSIC] Model preparation failed: " << error.what() << '\n';
    }
}
bool music_should_unload_llm() {
    const auto mode = lower(strip_config_value(setting("MUSIC_UNLOAD_LLM_DURING_GENERATION", "auto")));
    if (false_token(mode)) return false;
    if (mode == "1" || mode == "true" || mode == "yes" || mode == "on") return true;
    if (truthy("MEDIA_KEEP_MODELS_LOADED")) return false;
    return strip_config_value(setting("ACE_STEP_SERVER_URL")).empty();
}
bool music_should_reload_llm() {
    return !false_token(lower(strip_config_value(setting("MUSIC_RELOAD_LLM_AFTER_GENERATION", "true"))));
}
bool AceStepRuntime::running_locked() {
    if (pid_ <= 0 || reaped_) return false;
    int status = 0;
    pid_t result;
    do { result = waitpid(pid_, &status, WNOHANG); } while (result < 0 && errno == EINTR);
    if (result == 0) return true;
    reaped_ = true;
    return false;
}
int AceStepRuntime::process_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pid_;
}
int AceStepRuntime::start(const std::function<bool(const std::string &)> &healthy,
                          const std::function<double()> &startup_timeout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!truthy("MUSIC_ENABLED") || !strip_config_value(setting("ACE_STEP_SERVER_URL")).empty()
        || !truthy("ACE_STEP_AUTO_START", "true")) return -1;
    const auto url = ace_step_internal_url();
    if (healthy(url)) return -1;
    const auto binary = ace_step_binary_path();
    std::error_code error;
    if (!std::filesystem::exists(binary, error)) {
        std::cerr << "[MUSIC] ACE-Step binary not found at " << binary << '\n';
        return -1;
    }
    for (const auto &file : selected_ace_step_model_files()) {
        if (!std::filesystem::exists(std::filesystem::path(ace_step_models_dir()) / file, error))
            std::cerr << "[MUSIC] ACE-Step model file missing: " << file << '\n';
    }
    if (running_locked()) return pid_;
    const auto command = build_ace_step_server_command();
    std::vector<char *> argv;
    for (const auto &argument : command) argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);
    pid_t child = -1;
    const int result = posix_spawnp(&child, argv[0], nullptr, nullptr, argv.data(), environ);
    if (result) {
        std::cerr << "[MUSIC] Failed to start ACE-Step server: " << std::strerror(result) << '\n';
        pid_ = -1;
        return -1;
    }
    pid_ = child;
    reaped_ = false;
    // The reference parses this after Popen; invalid settings leave the child alive.
    const double timeout = startup_timeout();
    const auto started = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < timeout) {
        if (!running_locked()) {
            std::cerr << "[MUSIC] ACE-Step server exited before becoming healthy\n";
            return pid_;
        }
        if (healthy(url)) return pid_;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    std::cerr << "[MUSIC] ACE-Step startup health timeout\n";
    return pid_;
}
AceStepRuntime &ace_step_runtime() {
    static AceStepRuntime runtime;
    return runtime;
}
} // namespace ezlocalai
