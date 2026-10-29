#include "ezlocalai/native_server.hpp"
#include "ezlocalai/music_runtime.hpp"
#include "ezlocalai/cuda_mps.hpp"
#ifdef EZLOCALAI_CTTS_TORCH
#include "ezlocalai/ctts_torch.hpp"
#endif
#ifdef EZLOCALAI_NATIVE_STT
#include "ezlocalai/stt_whisper.hpp"
#endif

#include <csignal>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <nlohmann/json.hpp>

namespace {

volatile std::sig_atomic_t keep_running = 1;

void handle_signal(int) {
    keep_running = 0;
}

int parse_port(const char *value, int fallback) {
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > 65535) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

} // namespace

int main(int argc, char **argv) {
    // Globals.py loads dotenv before application configuration is read.
    if (const int error = ezl_load_dotenv()) {
        std::cerr << "Failed to load .env: " << std::strerror(error) << '\n';
        return 1;
    }
    (void)ezl_getenv_default("HOST");
    const char *host_env = std::getenv("HOST");
    std::string host = host_env && host_env[0] != '\0' ? host_env : "0.0.0.0";
    int port = parse_port(std::getenv("PORT"), 8091);
    port = parse_port(std::getenv("EZLOCALAI_PORT"), port);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = parse_port(argv[++i], port);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ezlocalai-native [--host 0.0.0.0] [--port 8091]\n";
            return 0;
        }
    }

    // A CUDA-free supervisor owns MPS. The child must exit completely before
    // shutdown: driver/Torch contexts can outlive the worker's local objects.
    std::unique_ptr<ezlocalai::CudaMpsService> cuda_mps;
    try { cuda_mps = std::make_unique<ezlocalai::CudaMpsService>(); }
    catch (const std::exception &error) {
        std::cerr << "Failed to initialize CUDA MPS: " << error.what() << '\n';
        return 1;
    }
    if (cuda_mps->active()) {
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::vector<std::string> arguments{"/proc/self/exe"};
        for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
        try { return cuda_mps->supervise(arguments, [] { return !keep_running; }); }
        catch (const std::exception &error) {
            std::cerr << "MPS worker supervisor failed: " << error.what() << '\n';
            return 1;
        }
    }
    ezl_env_config config{};
    ezl_load_config(&config);
    ezlocalai::precache_music_models();

    auto resource_probes = ezlocalai::native_resource_probes();
#ifdef EZLOCALAI_CTTS_TORCH
    resource_probes = ezlocalai::torch_resource_probes(std::move(resource_probes));
#endif
    auto resources = std::make_shared<ezlocalai::ResourceManager>(std::move(resource_probes));
    std::unique_ptr<ezlocalai::InferenceBackend> backend;
    if (ezlocalai::LlamaCppInferenceBackend::should_use_from_env(config)) {
        backend = std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model, resources);
    } else {
        backend = std::make_unique<ezlocalai::EchoInferenceBackend>(config.default_model);
    }
    ezlocalai::SttPoolFactory stt_factory;
#ifdef EZLOCALAI_NATIVE_STT
    ezlocalai::SttWhisperConfiguration speech;
    const char *vad = std::getenv("WHISPER_VAD_DIRECTORY");
    speech.vad_directory = vad && *vad ? vad : EZLOCALAI_WHISPER_VAD_DIRECTORY;
    stt_factory = [speech](auto resources) { return ezlocalai::stt_whisper_pool_operations(speech,std::move(resources)); };
#endif
    ezlocalai::TtsPoolFactory tts_factory=[](auto resources){
        ezlocalai::CttsServiceOperations operations;
#ifdef EZLOCALAI_CTTS_TORCH
        operations.runtime=ezlocalai::ctts_torch_operations();
#endif
        return ezlocalai::tts_pool_resources(ezlocalai::tts_pool_ctts_operations(std::move(operations)),std::move(resources));
    };
    std::unique_ptr<ezlocalai::NativeServer> server;
    try {
        server = ezlocalai::create_native_server(config, std::move(backend), resources, std::move(stt_factory), std::move(tts_factory));
    } catch (const std::exception &error) {
        std::cerr << "Failed to initialize ezlocalai-native: " << error.what() << '\n';
        return 1;
    }
#ifdef EZLOCALAI_CTTS_TORCH
    server->set_wakeword_factory(ezlocalai::wakeword_torch_factory());
    server->set_wakeword_trainer([](const std::string &word){return ezlocalai::wakeword_torch_create(nlohmann::json{{"word",word}}.dump());});
#endif

    std::string error;
    if (!server->start(host, port, &error)) {
        std::cerr << "Failed to start ezlocalai-native on " << host << ":" << port << ": " << error << "\n";
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::cout << "ezlocalai-native listening on http://" << host << ":" << port << "\n";
    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    server->stop();
    return 0;
}
