#include "ezlocalai/wakeword_tts.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream file(path, std::ios::binary); check(bool(file), "Missing evidence: " + path.string()); return {std::istreambuf_iterator<char>(file), {}}; }
std::string hash(const std::string &bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest);
    std::ostringstream output; for (auto byte : digest) output << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return output.str();
}
ezlocalai::WakewordTtsOperations operations(const Json &engines, const Json &failures, Json &calls, std::mutex &mutex) {
    ezlocalai::WakewordTtsOperations result;
    const auto invoke = [&, failures](const std::string &engine, const std::string &word, const std::string &voice) {
        { std::lock_guard<std::mutex> lock(mutex); calls.push_back(Json::array({engine, word, voice})); }
        if (std::find(failures.begin(), failures.end(), Json(voice)) != failures.end()) throw std::runtime_error("fixture failure");
        return engine + "|" + word + "|" + voice;
    };
    for (const auto &engine : engines) {
        if (engine == "gtts") result.gtts_mp3 = [invoke](const auto &word, const auto &voice) { return invoke("gtts", word, voice); };
        if (engine == "edge") result.edge_mp3 = [invoke](const auto &word, const auto &voice) { return invoke("edge", word, voice); };
    }
    result.decode_mp3 = [](const std::string &bytes) { return bytes; }; return result;
}
void compare_calls(Json actual, Json expected) {
    // Edge transport completion/start order is concurrent; the request set and
    // returned sample order are the observable contracts tested separately.
    std::sort(actual.begin(), actual.end()); std::sort(expected.begin(), expected.end()); check(actual == expected, "Engine request set");
}
}
int main(int argc, char **argv) {
    try {
        check(argc >= 2, "Wake-word reference root required"); const auto root = fs::path(argv[1]) / "tts";
        if (argc == 3 && std::string(argv[2]) == "live-gtts") {
            const auto mp3 = ezlocalai::wakeword_gtts_mp3("hello ezlocalai", "en");
            const auto wav = ezlocalai::wakeword_decode_mp3(mp3);
            std::ofstream(root / "native_live.mp3", std::ios::binary).write(mp3.data(), mp3.size());
            std::ofstream(root / "native_live.wav", std::ios::binary).write(wav.data(), wav.size());
            std::cout << "Live gTTS produced " << mp3.size() << " MP3 bytes and " << wav.size() << " WAV bytes\n"; return 0;
        }
        const auto fixture = Json::parse(read(root / "reference.json"));
        for (const auto &row : fixture.at("generation")) {
            Json calls = Json::array(), events = Json::array(), samples = Json::array(); std::mutex mutex;
            ezlocalai::WakewordSampleGenerator generator((root / "native_samples").string(), operations(row.at("engines"), row.at("failures"), calls, mutex));
            std::string error;
            try { for (const auto &sample : generator.generate("hello", row.at("count"), [&](size_t current, size_t total, const std::string &stage) { events.push_back(Json::array({current, total, stage})); }))
                samples.push_back(Json{{"audio", sample.audio}, {"sample_rate", sample.sample_rate}, {"engine", sample.engine}, {"voice", sample.voice}, {"word", sample.word}, {"variation", sample.variation}}); }
            catch (const std::exception &e) { error = e.what(); }
            check(error == row.value("error", ""), "Generator error mismatch");
            if (error.empty()) check(samples == row.at("samples"), "Generated samples/order mismatch");
            check(events == row.at("events"), "Generation progress mismatch"); compare_calls(calls, row.at("calls"));
        }
        for (const auto &row : fixture.at("similar")) {
            Json calls = Json::array(), samples = Json::array(); std::mutex mutex;
            ezlocalai::WakewordSampleGenerator generator((root / "native_samples").string(), operations(Json::array({"gtts", "edge"}), Json::array(), calls, mutex));
            for (const auto &sample : ezlocalai::wakeword_similar_samples(row.at("word"), generator, row.at("count"))) samples.push_back(Json{{"audio", sample.audio}, {"metadata", sample.metadata}});
            check(samples == row.at("samples"), "Similar-word samples/order mismatch"); compare_calls(calls, row.at("calls"));
        }
        for (const auto &row : fixture.at("gtts")) {
            Json requests = Json::array();
            for (const auto &request : ezlocalai::wakeword_gtts_requests(row.at("word"), row.at("language"))) requests.push_back(Json{{"url", request.url}, {"headers", request.headers}, {"body", request.body}});
            check(requests == row.at("requests"), "gTTS request mismatch: " + row.at("word").get<std::string>());
        }
        for (const auto &row : fixture.at("decode")) {
            const auto mp3 = read(root / row.at("mp3").get<std::string>()), expected = read(root / row.at("wav").get<std::string>());
            check(hash(mp3) == row.at("mp3_sha256") && hash(expected) == row.at("wav_sha256"), "Decode reference hashes");
            const auto actual = ezlocalai::wakeword_decode_mp3(mp3);
            if (actual != expected) { std::ofstream(root / (row.at("wav").get<std::string>() + ".native"), std::ios::binary).write(actual.data(), actual.size()); throw std::runtime_error("MP3 decode mismatch: " + row.at("mp3").get<std::string>()); }
        }
        std::cout << fixture.at("generation").size() << " generation, " << fixture.at("similar").size() << " similar-word, " << fixture.at("gtts").size() << " gTTS request and " << fixture.at("decode").size() << " exact MP3/WAV cases passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
