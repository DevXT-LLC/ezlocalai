#include "ezlocalai/llm_input.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using Json = nlohmann::ordered_json;
std::string sha256(const std::string &value) {
    unsigned char bytes[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), bytes);
    std::ostringstream out;
    for (const auto byte : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return out.str();
}
std::string text(size_t bytes) {
    const std::string sentence = "Reference text with a quoted \"value\", slash \\ and Unicode café.\n";
    std::string value;
    while (value.size() < bytes) value += sentence;
    while (bytes && bytes < value.size() && (static_cast<unsigned char>(value[bytes]) & 0xc0) == 0x80) --bytes;
    value.resize(bytes);
    return value;
}
struct Case { std::string name, request; bool chat = true, vision = false; };
Json request(const std::string &content) {
    return {{"messages", Json::array({{{"role", "user"}, {"content", content}}})},
        {"prompt_cache_key", "input-benchmark"}, {"max_tokens", 32},
        {"temperature", 0}, {"stop", Json::array({"STOP"})}};
}
}

// Exercises actual queued input preparation, without an engine or network.
// Output and helper-event hashes permit byte/order checks across builds.
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: bench_llm_input OUTPUT_DIRECTORY");
        const std::filesystem::path output(argv[1]);
        std::filesystem::create_directories(output);
        std::vector<Case> cases;
        for (const size_t bytes : {1024u, 102400u, 1048576u})
            cases.push_back({"text-" + std::to_string(bytes), request(text(bytes)).dump()});
        auto many = request("");
        many["messages"] = Json::array();
        for (int index = 0; index < 255; ++index)
            many["messages"].push_back({{"role", index % 2 ? "assistant" : "user"}, {"content", text(4096)}});
        many["messages"].push_back({{"role", "user"}, {"content", "current turn"}});
        cases.push_back({"messages-256", many.dump()});
        auto media = many;
        media["messages"].back()["content"] = Json::array({{{"type", "text"}, {"text", "current image"}},
            {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,eA=="}}}}});
        cases.push_back({"multimodal-history", media.dump(), true, true});
        auto audio = many;
        audio["messages"].back()["content"] = "eA==";
        audio["audio_format"] = "wav";
        cases.push_back({"audio-format-history", audio.dump()});
        cases.push_back({"completion-1048576", Json({{"prompt", text(1048576)}, {"max_tokens", 32}}).dump(), false});
        Json report = Json::array();
        volatile size_t observed = 0;
        for (const auto &item : cases) {
            std::vector<std::string> events;
            ezlocalai::LlmInputOperations operations;
            operations.local_uri = "http://local-benchmark";
            operations.load = [&] { events.push_back("load"); return ezlocalai::LlmInputModel{"source", item.vision}; };
            operations.transcribe = [&](const std::string &audio, const std::string &format) {
                events.push_back("transcribe:" + audio + ':' + format); return std::string("\"transcribed\"");
            };
            const auto prepare = [&] {
                events.clear();
                return ezlocalai::llm_prepare_text_image_input(item.request, item.chat, operations);
            };
            const auto prepared = prepare();
            const auto helper_events = Json(events).dump();
            std::ofstream(output / (item.name + ".json")) << prepared;
            std::vector<double> samples;
            const int iterations = item.request.size() < 10000 ? 100 : item.request.size() < 200000 ? 10 : 3;
            for (int sample = 0; sample < 7; ++sample) {
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < iterations; ++iteration) observed += prepare().size();
                samples.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count() / iterations);
                if (Json(events).dump() != helper_events) throw std::runtime_error("helper order changed during benchmark");
            }
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            report.push_back({{"case", item.name}, {"request_bytes", item.request.size()},
                {"prepared_bytes", prepared.size()}, {"request_sha256", sha256(item.request)},
                {"prepared_sha256", sha256(prepared)}, {"helper_events", Json(events)},
                {"helper_events_sha256", sha256(helper_events)}, {"iterations_per_sample", iterations},
                {"samples_ms", samples}, {"median_ms", sorted[sorted.size()/2]}});
            std::cout << report.back().dump() << std::endl;
        }
        std::ofstream(output / "timings.json") << report.dump(2) << '\n';
        return observed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
