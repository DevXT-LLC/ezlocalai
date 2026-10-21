#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/qwen_image.hpp"
#include "ezlocalai/wan_video.hpp"
#include "fixture_ports.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <csignal>
using namespace ezlocalai;
using Json = nlohmann::json;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
namespace {
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
void env(const char *name, const std::string &value) { setenv(name, value.c_str(), 1); }
void wait_file(const fs::path &path) {
    for (int n = 0; n < 2000 && !fs::exists(path); ++n) std::this_thread::sleep_for(5ms);
    check(fs::exists(path), "fixture did not enter: " + path.string());
}
struct Gate {
    std::mutex mutex; std::condition_variable cv; bool entered = false, open = false;
    void wait() { std::unique_lock<std::mutex> lock(mutex); entered = true; cv.notify_all(); check(cv.wait_for(lock, 15s, [&]{return open;}), "speech gate timeout"); }
    void ready() { std::unique_lock<std::mutex> lock(mutex); check(cv.wait_for(lock, 5s, [&]{return entered;}), "speech gate not entered"); }
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; cv.notify_all(); }
};
int engine(int argc, char **argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; ++i) args[argv[i]] = argv[i + 1];
    const bool video = args.count("--video-frames");
    const auto gate = std::string(video ? "video" : "image");
    std::ofstream(gate + "-entered") << getpid();
    wait_file(gate + "-release");
    if (video && fs::exists("fail-video")) return 83;
    const int count = video ? std::stoi(args.at("--video-frames")) : 1;
    RgbImage frame{64, 64, std::vector<uint8_t>(64 * 64 * 3, 127)};
    for (int i = 0; i < count; ++i) {
        auto path = args.at("-o");
        if (video) { auto n = std::to_string(i); path.replace(path.find("%06d"), 4, std::string(6 - n.size(), '0') + n); }
        std::ofstream(path, std::ios::binary) << encode_rgb_png(frame);
    }
    return 0;
}
void scenario(const std::string &mode, const fs::path &binary) {
    for (const auto &[name, value] : environment_defaults) unsetenv(std::string(name).c_str());
    for (const auto *name : {"LLAMA_CPP_EMBEDDING_MODEL_PATH", "LLAMA_CPP_MODEL_PATH", "EMBEDDING_FIXTURE_DIR"}) unsetenv(name);
    for (const auto &[name, value] : std::map<std::string, std::string>{
        {"EZLOCALAI_DOTENV_PATH", ""}, {"EZLOCALAI_API_KEY", ""}, {"DEFAULT_MODEL", "fixture"},
        {"LLM_MODEL_RESIDENCY", "resident"}, {"LLAMA_CPP_CONTEXT", "4096"}, {"LLM_MAX_TOKENS", "4096"},
        {"N_PARALLEL", "1"}, {"GPU_LAYERS", "0"}, {"LLM_SPECULATIVE_TYPE", "none"},
        {"CUDA_VISIBLE_DEVICES", ""}, {"LLAMA_CPP_START_TIMEOUT", "5"}, {"LLAMA_CPP_SERVER_URL", ""},
        {"LLAMA_CPP_EMBEDDING_SERVER_URL", ""}, {"TTS_ENABLED", "true"}, {"STT_ENABLED", "true"},
        {"TTS_N_PARALLEL", "1"}, {"STT_N_PARALLEL", "1"}, {"EMBEDDING_N_PARALLEL", "1"},
        {"EMBEDDING_ENABLED", "true"}, {"EMBEDDING_KEEP_LOADED", "true"}, {"EMBEDDING_GPU_LAYERS", "0"},
        {"LAZY_LOAD_VOICE", "false"}, {"TTS_UNLOAD_LLM_DURING_GENERATION", "false"},
        {"STT_UNLOAD_LLM_DURING_GENERATION", "false"}, {"VOICE_SERVER", ""}, {"TEXT_SERVER", ""},
        {"IMAGE_SERVER", ""}, {"EMBEDDING_SERVER", ""}, {"FALLBACK_SERVER", ""}, {"ROUTER_URL", ""},
        {"IMAGE_ENABLED", "true"}, {"IMG_MODEL", "Qwen/Qwen-Image-2.1"}, {"IMAGE_UNLOAD_LLM_DURING_GENERATION", "false"},
        {"VIDEO_ENABLED", "true"}, {"VIDEO_MODEL", "wan2.2"}, {"MUSIC_ENABLED", "false"},
        {"VIDEO_UNLOAD_LLM_DURING_GENERATION", "true"}, {"VIDEO_RELOAD_LLM_AFTER_GENERATION", "true"},
        {"MEDIA_KEEP_MODELS_LOADED", "false"}, {"HF_HUB_OFFLINE", "1"}}) env(name.c_str(), value);
    if (mode == "keep") { env("MEDIA_KEEP_MODELS_LOADED", "true"); env("VIDEO_UNLOAD_LLM_DURING_GENERATION", "false"); }
    if (mode == "busy-image") env("MEDIA_KEEP_MODELS_LOADED", "true");
    const auto cwd = fs::current_path(), dir = cwd / mode; fs::create_directories(dir); fs::current_path(dir);
    struct Directory { fs::path cwd; ~Directory() {fs::current_path(cwd);} } directory{cwd};
    env("SDCPP_BIN", binary); env("SDCPP_MODELS_DIR", dir / "image-models"); env("HF_HUB_CACHE", dir / "models");
    for (const auto &asset : qwen_image_assets()) { auto p = dir / "image-models" / asset.filename; fs::create_directories(p.parent_path()); std::ofstream(p) << "fixture"; }
    for (const auto &asset : wan_model_assets(false, "Q4_K_M")) {
        auto repo = asset.repository; size_t pos;
        while ((pos = repo.find('/')) != std::string::npos) repo.replace(pos, 1, "--");
        auto p = dir / "models" / ("models--" + repo) / "snapshots" / asset.revision / asset.filename;
        fs::create_directories(p.parent_path()); std::ofstream(p) << "fixture";
    }
    const int port = fixture_ports(2); std::ofstream("model.gguf") << "GGUF fixture";
    env("LLAMA_CPP_MODEL_PATH", dir / "model.gguf"); env("LLAMA_CPP_EMBEDDING_MODEL_PATH", dir / "model.gguf");
    env("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN); env("LLAMA_CPP_SERVER_PORT", std::to_string(port));
    env("LLAMA_CPP_EMBEDDING_SERVER_PORT", std::to_string(port + 1)); env("EMBEDDING_FIXTURE_DIR", dir);
    std::atomic<int> tts_loads{0}, stt_loads{0}, tts_closed{0}, stt_closed{0}, cache_clears{0};
    auto probes = native_resource_probes(); probes.gpu_count = []{return 0;};
    probes.total_per_gpu = probes.free_per_gpu = []{return std::vector<double>{};}; probes.free_vram = [](auto){return 0.;};
    probes.allocator_stats = []{return std::vector<CudaAllocatorStats>{};}; probes.system_ram = probes.available_ram = []{return 64.;};
    probes.release_unused_cuda_memory = [&]{++cache_clears;}; auto resources = std::make_shared<ResourceManager>(probes);
    Gate speech; if (mode != "busy-tts" && mode != "busy-stt") speech.release();
    TtsPoolFactory tts = [&](auto r) { auto ops = tts_pool_resources({}, r); ops.can_load = []{return ModelAdmission{true,"cpu","fixture"};}; ops.schedule_cleanup = [](auto f){f();};
        ops.load = [&](const auto &) { ++tts_loads; if (mode == "restore-failure" && tts_loads > 1) throw std::runtime_error("fixture TTS restore failure");
            auto m = std::make_shared<TtsPoolModel>(); m->device = []{return "cpu";}; m->closed = []{return false;}; m->close = [&]{++tts_closed;};
            m->generate = [&](const auto &){speech.wait(); return "UklGRg==";}; return m; }; return ops; };
    SttPoolFactory stt = [&](auto r) { auto ops = stt_pool_resources({}, r); ops.can_load = []{}; ops.schedule_cleanup = [](auto f){f();};
        ops.load = [&](const auto &) { ++stt_loads; auto m = std::make_shared<SttPoolModel>(); m->device = []{return "cpu";}; m->close = [&]{++stt_closed;};
            m->transcribe = [&](const auto &){speech.wait(); return R"("fixture transcription")";}; return m; }; return ops; };
    auto backend = std::make_unique<LlamaCppInferenceBackend>("fixture", resources); auto *model = backend.get();
    std::string error; check(model->warmup(&error), "LLM warmup: " + error); model->warm_aux_model("embedding");
    ezl_env_config config{}; ezl_load_config(&config); config.max_concurrent_requests = 3;
    NativeServer server(config, std::move(backend), resources, stt, tts, {}, true); server.preload_media();
    for (auto type : {ModelType::Tts, ModelType::Stt, ModelType::Embedding, ModelType::Image}) check(resources->is_model_loaded(type), std::string("auxiliary initially loaded: ") + model_type_name(type));
    std::future<HttpResponse> active;
    struct Release { Gate &speech; ~Release(){speech.release(); std::ofstream("video-release") << "go"; std::ofstream("image-release") << "go";} } release{speech};
    if (mode == "busy-tts") active = std::async(std::launch::async, [&]{return server.handle_request("POST", "/v1/audio/speech", "", R"({"input":"fixture"})");});
    if (mode == "busy-stt") active = std::async(std::launch::async, [&]{return server.handle_request("POST", "/v1/audio/transcriptions", "", "--fixture\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\nfixture\r\n--fixture--\r\n", "multipart/form-data; boundary=fixture");});
    if (mode == "busy-tts" || mode == "busy-stt") speech.ready();
    if (mode == "busy-image") { active = std::async(std::launch::async, [&]{return server.handle_request("POST", "/v1/images/generations", "", R"({"prompt":"fixture","size":"64x64","response_format":"b64_json"})");}); wait_file("image-entered"); }
    if (mode == "error") std::ofstream("fail-video") << "fail";
    auto video = std::async(std::launch::async, [&]{return server.handle_request("POST", "/v1/videos/generations", "", R"({"prompt":"fixture","size":"64x64","num_frames":5,"num_inference_steps":2,"response_format":"b64_json"})");});
    // Release gates before joining futures even if a pre-fix assertion fails.
    struct ReleaseVideo { ~ReleaseVideo(){std::ofstream("video-release") << "go";} } release_video;
    wait_file("video-entered");
    const bool keep = mode == "keep";
    check(tts_closed == (keep || mode == "busy-tts" ? 0 : 1), "video handoff must close the actual idle TTS pool");
    check(stt_closed == (keep || mode == "busy-stt" ? 0 : 1), "video handoff must close the actual idle STT pool");
    check(resources->is_model_loaded(ModelType::Embedding) == keep, "embedding handoff");
    check(resources->is_model_loaded(ModelType::Image) == (keep || mode == "busy-image"), "idle image owner released; active image preserved");
    check(cache_clears == (keep ? 0 : 1), "video handoff clears unused allocator memory once");
    std::ofstream("video-release") << "go";
    check(video.wait_for(5s) == std::future_status::ready, "video restore must not wait on active speech/image");
    check(video.get().status == (mode == "error" ? 500 : 200), "video result");
    check(!Json::parse(model->model_lifecycle_json())["loaded_llms"].empty(), "LLM restored");
    check(resources->is_model_loaded(ModelType::Tts) == (mode != "restore-failure"), "TTS restore policy or failed warm load");
    for (auto type : {ModelType::Stt, ModelType::Embedding, ModelType::Image}) check(resources->is_model_loaded(type), "other auxiliaries restored even after TTS warm failure");
    speech.release(); std::ofstream("image-release") << "go"; if (active.valid()) check(active.get().status == 200, "active auxiliary completes");
    check(tts_loads == (keep || mode == "busy-tts" ? 1 : 2), "TTS restored through native owner exactly once");
    check(stt_loads == (keep || mode == "busy-stt" ? 1 : 2), "STT restored through native owner exactly once");
    std::cout << mode << ": actual pools, image owner, embeddings, LLM restoration and cleanup passed\n";
}
}
int main(int argc, char **argv) {
    try {
        if (argc > 1) return engine(argc, argv);
        const auto binary = fs::canonical("/proc/self/exe"), cwd = fs::current_path();
        const auto directory = fs::temp_directory_path() / ("ezl-video-auxiliary-" + std::to_string(getpid())); fs::create_directories(directory); fs::current_path(directory);
        struct Cleanup {fs::path cwd, directory; ~Cleanup(){fs::current_path(cwd);fs::remove_all(directory);}} cleanup{cwd,directory};
        for (const auto *mode : {"idle", "error", "busy-tts", "busy-stt", "busy-image", "keep", "restore-failure"}) scenario(mode, binary);
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
