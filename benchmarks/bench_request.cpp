#include "ezlocalai/llm_request.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

// Measures worker request preparation without GPU inference. Saved payloads
// allow before/after byte comparisons, including prompts and sampling settings.
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: bench_llm_request SOURCE_FILE OUTPUT_DIRECTORY");
        std::ifstream source_file(argv[1]);
        const std::string source{std::istreambuf_iterator<char>(source_file), {}};
        if (source.empty()) throw std::runtime_error("source must be a readable, nonempty file");
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        nlohmann::json report = nlohmann::json::array();
        volatile size_t observed = 0;
        for (const size_t requested_bytes : {32000u, 128000u, 680000u}) {
            std::string text;
            while (text.size() <= requested_bytes) text += source;
            auto bytes = requested_bytes;
            // A byte budget may end inside a UTF-8 code point.
            while (bytes && (static_cast<unsigned char>(text[bytes]) & 0xc0) == 0x80) --bytes;
            text.resize(bytes);
            const auto body = nlohmann::json{
                {"messages", {{{"role", "user"}, {"content", text}}}},
                {"max_tokens", 32}, {"temperature", 0}, {"top_p", 0.95},
                {"chat_template_kwargs", {{"enable_thinking", false}}}
            }.dump();
            const auto prepare = [&] {
                return ezlocalai::llm_engine_request(ezlocalai::llm_model_request_settings(
                    body, "unsloth/Qwen3.8-27B-GGUF"), true, "32", "");
            };
            const auto payload = prepare();
            std::ofstream(output / ("payload-" + std::to_string(requested_bytes) + ".json")) << payload;
            std::vector<double> samples;
            constexpr int iterations = 10;
            for (int sample = 0; sample < 7; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < iterations; ++i) observed += prepare().size();
                samples.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count() / iterations);
            }
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            report.push_back({{"requested_source_bytes", requested_bytes}, {"source_bytes", bytes}, {"request_bytes", body.size()},
                {"payload_bytes", payload.size()}, {"iterations_per_sample", iterations},
                {"samples_ms", samples}, {"median_ms", sorted[sorted.size() / 2]},
                {"mean_ms", std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()}});
            std::cout << report.back().dump() << std::endl;
        }
        std::ofstream(output / "timings.json") << report.dump(2) << '\n';
        return observed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
