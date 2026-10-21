#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>

int main(int argc, char **argv) {
    namespace fs = std::filesystem;
    const auto directory = std::getenv("LLM_AUDIO_FIXTURE_DIR");
    if (!directory) return 2;
    fs::create_directories(directory);
    const auto root = fs::path(directory);
    const auto pid = std::to_string(getpid());
    std::string input, output;
    {
        std::ofstream arguments(root / ("audio-args-" + pid));
        for (int i = 1; i < argc; ++i) {
            arguments << argv[i] << '\n';
            if (std::string(argv[i]) == "-f" && i + 1 < argc) input = argv[i + 1];
            if (std::string(argv[i]) == "-of" && i + 1 < argc) output = argv[i + 1];
        }
    }
    if (input.empty() || output.empty()) return 3;
    fs::copy_file(input, root / ("audio-bytes-" + pid));
    bool llm_loaded = false;
    for (const auto &file : fs::directory_iterator(root)) if (file.path().filename().string().rfind("launch-", 0) == 0) llm_loaded = true;
    std::ofstream(root / ("audio-llm-state-" + pid)) << (llm_loaded ? '1' : '0');
    { std::ofstream launches(root / "audio-launches", std::ios::app); launches << pid << '\n'; }
    if (std::getenv("LLM_AUDIO_FIXTURE_BLOCK")) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!fs::exists(root / "audio-release")) {
            if (std::chrono::steady_clock::now() >= deadline) return 4;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    if (const auto delay = std::getenv("LLM_AUDIO_FIXTURE_DELAY_MS"))
        std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(delay)));
    if (const auto failure = std::getenv("LLM_AUDIO_FIXTURE_EXIT")) return std::atoi(failure);
    const auto result = std::getenv("LLM_AUDIO_FIXTURE_RESULT");
    std::ofstream(output + ".json") << (result ? result : R"({"transcription":[{"text":" first"},{"text":" second\n"}]})");
    return 0;
}
