#include "ezlocalai/process.hpp"
#include "ezlocalai/ezlocalai_c.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {
std::string default_env(const char *name, const std::string &fallback) {
    const char *value = std::getenv(name);
    if (value && *value) return value;
    if (setenv(name, fallback.c_str(), 1) != 0) throw std::system_error(errno, std::generic_category(), name);
    return fallback;
}
}

int main(int argc, char **argv) {
    try {
        if (const int error = ezl_load_dotenv()) throw std::system_error(error, std::generic_category(), "load .env");
        const auto host = default_env("HOST", "0.0.0.0");
        const char *legacy_port = std::getenv("EZLOCALAI_PORT");
        const auto port = default_env("PORT", legacy_port && *legacy_port ? legacy_port : "8091");
        default_env("EZLOCALAI_PORT", port);
        for (const char *name : {"MODELS_PATH", "HF_HOME", "HF_HUB_CACHE"}) {
            std::filesystem::create_directories(default_env(name, "/app/models"));
        }
        std::filesystem::create_directories(default_env("LLAMA_CACHE", "/app/models/.llama.cpp"));
        default_env("EZLOCALAI_NATIVE_BACKEND", "llama.cpp");
        default_env("LLAMA_CPP_SERVER_BIN", "/usr/local/bin/llama-server");
        for (const char *path : {"/app/outputs", "/app/voices", "/app/whispercpp/models"}) {
            std::filesystem::create_directories(path);
        }
        if (default_env("EZLOCALAI_MODEL_STATUS_ON_START", "true") == "true") {
            for (const char *command : {"status", "model-status"}) {
                try { ezlocalai::run_process({"/usr/local/bin/llama_cpp_tool", command}); }
                catch (const std::exception &error) { std::cerr << error.what() << '\n'; }
            }
        }
        if (argc > 1) ezlocalai::exec_process(std::vector<std::string>(argv + 1, argv + argc));
        ezlocalai::exec_process({"/usr/local/bin/ezlocalai-native", "--host", host, "--port", port});
    } catch (const std::exception &error) {
        std::cerr << "ezlocalai-native-entrypoint: " << error.what() << '\n';
        return 1;
    }
}
