#include "ezlocalai/wakeword_model.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/whisper_features.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) {
    std::ifstream input(path, std::ios::binary); check(bool(input), "Missing evidence: " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
std::string sha(const std::string &bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest);
    std::ostringstream out; for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str();
}
std::string base64(const std::string &bytes) {
    const char *table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t value = uint32_t(uint8_t(bytes[i])) << 16 |
            (i + 1 < bytes.size() ? uint32_t(uint8_t(bytes[i + 1])) << 8 : 0) |
            (i + 2 < bytes.size() ? uint8_t(bytes[i + 2]) : 0);
        result += table[(value >> 18) & 63]; result += table[(value >> 12) & 63];
        result += i + 1 < bytes.size() ? table[(value >> 6) & 63] : '=';
        result += i + 2 < bytes.size() ? table[value & 63] : '=';
    }
    return result;
}
void http_predictions(const fs::path &root, const Json &fixture, const std::string &device) {
    const auto previous = fs::current_path();
    char pattern[] = "/tmp/ezlocalai-wakeword-model-XXXXXX"; const char *path = mkdtemp(pattern); check(path, "temporary model directory");
    const fs::path work(path); fs::current_path(work);
    struct Cleanup { fs::path previous, work; ~Cleanup() { fs::current_path(previous); fs::remove_all(work); } } cleanup{previous, work};
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(std::string(entry.first).c_str());
    setenv("EZLOCALAI_API_KEY", "", 1); setenv("DEFAULT_MODEL", "model", 1); setenv("VOICE_SERVER", "false", 1);
    for (const auto *name : {"EZLOCALAI_PRELOAD_MODEL_ON_START", "EZLOCALAI_MODEL_STATUS_ON_START", "TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED"}) setenv(name, "false", 1);
    ezl_env_config config{}; ezl_load_config(&config);
    ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
    server.set_wakeword_factory(ezlocalai::wakeword_torch_factory(device)); setenv("STT_ENABLED", "true", 1);
    size_t requests = 0;
    for (const auto &model : fixture.at("models")) {
        const auto word = model.at("directory").get<std::string>();
        const auto directory = work / "wakeword_models" / word; fs::create_directories(directory);
        for (const auto *file : {"config.json", "model.pt"}) fs::copy_file(root / word / file, directory / file);
        for (const auto &row : model.at("cases")) {
            const auto audio = read(root / row.at("audio").get<std::string>());
            for (bool multipart : {false, true}) {
                const auto body = multipart ? "--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n" + audio + "\r\n--boundary--\r\n" : Json{{"audio_base64", base64(audio)}}.dump();
                const auto response = server.handle_request("POST", "/v1/wakeword/predict/" + word + (multipart ? "/file" : ""), "", body,
                    multipart ? "multipart/form-data; boundary=boundary" : "application/json");
                const auto label = word + "/" + row.at("audio").get<std::string>();
                if (row.contains("error")) check(response.status == 500 && Json::parse(response.body).at("detail") == "Prediction failed", label + " error response");
                else if (row.at("confidence").is_null()) check(response.status == 500 && response.body == "Internal Server Error", label + " nonfinite response");
                else {
                    check(response.status == 200, label + " response: " + response.body);
                    const auto value = Json::parse(response.body);
                    check(value.at("word") == word && value.at("detected") == row.at("detected") &&
                        std::abs(value.at("confidence").get<double>() - row.at("confidence").get<double>()) < 2e-6, label + " HTTP prediction");
                }
                ++requests;
            }
        }
    }
    std::cout << requests << " JSON and multipart predictions through native HTTP handlers\n";
}
}
int main(int argc, char **argv) {
    try {
        check(argc >= 2, "Wake-word evidence directory required");
        const fs::path root = fs::absolute(argv[1]); const std::string device = argc > 2 ? argv[2] : "cpu";
        // Exercise the same OpenBLAS/Torch coexistence as the full worker.
        // Linking OpenBLAS globally previously changed Torch MFCC arithmetic.
        const auto whisper_audio = ezlocalai::soundfile_mono_float32(read(root / "sine.wav")).first;
        (void)ezlocalai::whisper_log_mel(whisper_audio.samples);
        const auto reference = Json::parse(read(root / "reference.json"));
        size_t predictions = 0;
        for (const auto &row : reference.at("models")) {
            const auto name = row.at("kind").get<std::string>();
            check(sha(read(root / name / "model.pt")) == row.at("model_sha256"), "Checkpoint hash " + name);
            if (device != "cpu") continue; // CUDA has its own oracle below.
            auto model = ezlocalai::wakeword_torch_factory(device)((root / name).string());
            const auto result = model->predict(read(root / "sine.wav"));
            check(result.first == row.at("detected") && std::abs(result.second - row.at("confidence").get<double>()) < 2e-6,
                  name + " confidence " + std::to_string(result.second));
            std::cout << name << " " << std::setprecision(17) << result.second << '\n'; ++predictions;
        }
        if (fs::exists(root / "prediction_cases.json")) {
            const auto fixture = Json::parse(read(root / (device == "cpu" ? "prediction_cases.json" : "prediction_cases_cuda.json")));
            for (const auto &model_row : fixture.at("models")) {
                const auto name = model_row.at("directory").get<std::string>();
                check(sha(read(root / name / "model.pt")) == model_row.at("sha256"), "Checkpoint hash " + name);
                auto model = ezlocalai::wakeword_torch_factory(device)((root / name).string());
                for (const auto &row : model_row.at("cases")) {
                    const auto label = name + "/" + row.at("audio").get<std::string>();
                    const auto bytes = read(root / row.at("audio").get<std::string>());
                    std::pair<bool, double> result; bool failed = false;
                    try { result = model->predict(bytes); } catch (const std::exception &) { failed = true; }
                    check(failed == row.contains("error"), label + " error mismatch");
                    if (!failed) {
                        if (row.at("confidence").is_null()) check(!std::isfinite(result.second), label + " nonfinite result");
                        else check(result.first == row.at("detected") && std::abs(result.second - row.at("confidence").get<double>()) < 2e-6,
                                   label + " confidence " + std::to_string(result.second));
                    }
                    ++predictions;
                }
            }
            double largest_feature_error = 0;
            for (const auto &row : fixture.at("features")) {
                const auto bytes = read(root / row.at("audio").get<std::string>());
                const auto features = ezlocalai::wakeword_audio_features(bytes, row.at("sample_rate"), row.at("n_mfcc"));
                check(Json(features.shape) == row.at("shape"), "MFCC shape");
                const auto raw = read(root / row.at("file").get<std::string>());
                check(sha(raw) == row.at("sha256") && raw.size() == features.samples.size() * sizeof(float), "MFCC reference integrity");
                double max_error = 0;
                for (size_t i = 0; i < features.samples.size(); ++i) {
                    float expected; std::memcpy(&expected, raw.data() + i * sizeof(float), sizeof(float));
                    check(std::isfinite(features.samples[i]), "Nonfinite MFCC feature");
                    max_error = std::max(max_error, double(std::abs(expected - features.samples[i])));
                }
                check(max_error <= 0.0002, "MFCC mismatch: " + row.at("audio").get<std::string>() + " " + std::to_string(max_error));
                largest_feature_error = std::max(largest_feature_error, max_error);
            }
            std::cout << fixture.at("features").size() << " MFCC arrays: maximum absolute error " << largest_feature_error << '\n';
            http_predictions(root, fixture, device);
        }
        std::cout << predictions << " native " << device << " predictions matched Python checkpoints\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
