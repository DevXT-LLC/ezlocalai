#include "ezlocalai/wakeword_samples.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
size_t comparisons = 0;
void compare(const fs::path &root, const Json &expected, const std::string &actual) {
    const auto path = root / expected.at("path").get<std::string>(); const auto bytes = read(path);
    check(hash(bytes) == expected.at("sha256"), "Reference hash mismatch: " + path.string());
    if (actual != bytes) {
        size_t index = 0; while (index < std::min(actual.size(), bytes.size()) && actual[index] == bytes[index]) ++index;
        std::ofstream(path.string() + ".native", std::ios::binary).write(actual.data(), actual.size());
        throw std::runtime_error("WAV mismatch " + path.string() + " at byte " + std::to_string(index) + ", lengths " + std::to_string(actual.size()) + "/" + std::to_string(bytes.size()));
    }
    ++comparisons;
}
void tail(ezlocalai::WakewordRandom &random, const Json &expected) {
    for (const auto &value : expected) check(random.uniform() == value.get<double>(), "RandomState continuation mismatch");
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "Wake-word reference root required"); const auto root = fs::path(argv[1]) / "samples";
        const auto fixture = Json::parse(read(root / "reference.json"));
        for (const auto &row : fixture.at("augment")) {
            ezlocalai::WakewordRandom random(row.at("seed"));
            const auto actual = ezlocalai::wakeword_augment_sample(read(root / row.at("input").at("path").get<std::string>()), random);
            check(actual.size() == row.at("outputs").size(), "Augmentation count");
            for (size_t i = 0; i < actual.size(); ++i) { check(actual[i].second == row.at("outputs")[i].at("variation"), "Augmentation order"); compare(root, row.at("outputs")[i], actual[i].first); }
            tail(random, row.at("rng_tail"));
            compare(root, row.at("normalized"), ezlocalai::wakeword_prepare_tts_wav(read(root / row.at("input").at("path").get<std::string>())));
        }
        for (const auto &row : fixture.at("noise")) {
            ezlocalai::WakewordRandom random(row.at("seed"));
            const auto actual = ezlocalai::wakeword_noise_samples(row.at("count"), random, row.at("rate"), row.at("duration"));
            check(actual.size() == row.at("outputs").size(), "Noise count");
            for (size_t i = 0; i < actual.size(); ++i) { check(Json(actual[i].metadata) == row.at("outputs")[i].at("metadata"), "Noise metadata"); compare(root, row.at("outputs")[i], actual[i].audio); }
            tail(random, row.at("rng_tail"));
        }
        for (const auto &row : fixture.at("datasets")) {
            ezlocalai::WakewordRandom random(row.at("seed")); std::vector<ezlocalai::WakewordTtsSample> samples;
            for (size_t i = 0; i < row.at("inputs").size(); ++i) samples.push_back({read(root / row.at("inputs")[i].at("path").get<std::string>()), "gtts", "en", "sample" + std::to_string(i), "normal", 16000});
            Json events = Json::array();
            const auto actual = ezlocalai::wakeword_augment_dataset(samples, random, row.at("selected"), [&](size_t current, size_t total, const std::string &stage) { events.push_back(Json::array({current, total, stage})); });
            check(actual.size() == row.at("outputs").size(), "Dataset size"); check(events == row.at("events"), "Dataset progress callbacks");
            for (size_t i = 0; i < actual.size(); ++i) { check(Json(actual[i].metadata) == row.at("outputs")[i].at("metadata"), "Dataset metadata/order"); compare(root, row.at("outputs")[i], actual[i].audio); }
            tail(random, row.at("rng_tail"));
        }
        for (const auto &row : fixture.at("chains")) {
            ezlocalai::WakewordRandom random(row.at("seed")); std::vector<ezlocalai::WakewordTtsSample> samples;
            for (size_t i = 0; i < row.at("inputs").size(); ++i) samples.push_back({read(root / row.at("inputs")[i].at("path").get<std::string>()), "gtts", "en", "sample" + std::to_string(i), "normal", 16000});
            const auto positive = ezlocalai::wakeword_augment_dataset(samples, random, row.at("selected"));
            check(positive.size() == row.at("positive_count"), "Chained positive count");
            const auto negative = ezlocalai::wakeword_noise_samples(positive.size() / 4, random);
            check(negative.size() == row.at("outputs").size(), "Chained negative count");
            for (size_t i = 0; i < negative.size(); ++i) { check(Json(negative[i].metadata) == row.at("outputs")[i].at("metadata"), "Chained noise metadata"); compare(root, row.at("outputs")[i], negative[i].audio); }
            tail(random, row.at("rng_tail"));
        }
        std::cout << comparisons << " WAV artifacts exactly matched; augmentation/noise RNG, metadata, ordering and progress matched\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
