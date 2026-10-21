#include "ezlocalai/stt_whisper.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>

using namespace transcription_test;

std::string digest_file(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    expect(bool(input), "open " + path.string());
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    expect(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1, "initialize digest");
    char buffer[65536];
    while (input) {
        input.read(buffer, sizeof(buffer));
        expect(EVP_DigestUpdate(context.get(), buffer, input.gcount()) == 1, "update digest");
    }
    expect(input.eof(), "read " + path.string());
    unsigned char digest[32]; unsigned size = 0;
    expect(EVP_DigestFinal_ex(context.get(), digest, &size) == 1 && size == 32, "finish digest");
    std::ostringstream result;
    for (auto byte : digest) result << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return result.str();
}

int main(int argc, char **argv) {
    try {
        expect(argc == 6, "reference root, capture JSON, Whisper model, VAD directory, speech WAV");
        const auto reference = fs::absolute(argv[1]);
        const auto capture = Json::parse(read(argv[2]));
        const auto model = fs::absolute(argv[3]);
        const auto vad = fs::absolute(argv[4]);
        const auto audio = fs::absolute(argv[5]);
        expect(digest_file(reference / "ezlocalai/STT.py") == capture.at("source_sha256"), "source identity");
        expect(digest_file(audio) == capture.at("audio_sha256"), "speech identity");
        for (const auto &file : capture.at("model_hashes").items())
            expect(digest_file(model / file.key()) == file.value(), "model identity: " + file.key());

        Temporary temporary;
        fs::create_directory(temporary.path / "outputs");
        SttWhisperConfiguration configuration;
        configuration.vad_directory = vad.string();
        SttRuntimeOperations platform;
        platform.working_directory = [&] { return temporary.path.string(); };
        SttRuntime runtime(model.string(), stt_whisper_operations(configuration, platform));
        expect(runtime.device() == capture.at("device"), "actual inference device");
        expect(runtime.compute_type() == capture.at("compute_type"), "actual compute type");
        const auto bytes = read(audio);
        std::string encoded(4 * ((bytes.size() + 2) / 3) + 1, '\0');
        const auto encoded_size = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),
            reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
        expect(encoded_size >= 0, "encode source wrapper's base64 audio input");
        encoded.resize(encoded_size);
        size_t count = 0;
        for (const auto &row : capture.at("cases")) {
            auto request = row.at("kwargs");
            request["base64_audio"] = encoded;
            auto actual = Json::parse(runtime.transcribe(request.dump()));
            // The reference's eight-thread CPU int8 alignment can choose
            // adjacent timestamp frames between runs. Accept only word times
            // actually observed in independent reference executions; text,
            // structure and every other field still compare exactly.
            if (row.contains("reference_variants")) {
                const auto flattened = actual.flatten();
                for (const auto &field : flattened.items()) {
                    const auto &key = field.key();
                    if (!field.value().is_number() || key.find("/words/") == std::string::npos ||
                        !((key.size() >= 6 && key.compare(key.size() - 6, 6, "/start") == 0) ||
                          (key.size() >= 4 && key.compare(key.size() - 4, 4, "/end") == 0))) continue;
                    const Json::json_pointer pointer(key);
                    for (const auto &variant : row.at("reference_variants")) {
                        if (variant.contains(pointer) && variant.at(pointer).is_number() &&
                            std::abs(field.value().get<double>() - variant.at(pointer).get<double>()) <= 1e-12) {
                            actual[pointer] = row.at("output").at(pointer);
                            break;
                        }
                    }
                }
            }
            compare_json(actual, row.at("output"), "deployment case " + std::to_string(count));
            expect(runtime.device() == capture.at("device"), "no unintended CPU fallback");
            expect(fs::is_empty(temporary.path / "outputs"), "temporary audio cleanup");
            std::cout << "deployment case " << count++ << " matches\n" << std::flush;
        }
        runtime.close();
        expect(!runtime.has_model() && !runtime.has_batch() && !runtime.has_audio(), "closed native handles");
        std::cout << count << " installed-version STT calls match on " << runtime.device() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
