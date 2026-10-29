// Isolated native MeanVC2 engine. Only the duplicated protocol fd carries JSON;
// upstream diagnostics (including stdio) go to stderr, never the audio pipe.
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/registry.h"
#include "ezlocalai/process.hpp"
#include "ezlocalai/voice_audio.hpp"
#include "ezlocalai/voice_resample.hpp"
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sstream>
#include <sys/file.h>
#include <unistd.h>
using Json = nlohmann::json;
namespace rt = engine::runtime;
namespace {
void write_all(int fd, const std::string &s) {
    size_t at = 0;
    while (at < s.size()) {
        auto n = write(fd, s.data() + at, s.size() - at);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("Voice protocol pipe closed");
        at += size_t(n);
    }
}
constexpr const char *model_sha = "340d5dd63cdd9d44045cdc4b394cccf677ae870a9ce4ed18f0007c8788e65245";
constexpr const char *model_url =
    "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/874fcb41ececda20bcbc0782d6b4ea7f7d069ea8/"
    "MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf";
void verify_model(const std::string &path) {
    if (std::filesystem::file_size(path) != 1629326784)
        throw std::runtime_error("MeanVC2 F32 model size is invalid");
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("MeanVC2 F32 model is missing; provision it with --download-model");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || !EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr))
        throw std::runtime_error("Cannot initialize model verification");
    std::vector<char> buffer(1024 * 1024);
    uint64_t bytes = 0;
    while (in) {
        in.read(buffer.data(), buffer.size());
        auto n = in.gcount();
        bytes += n;
        if (!EVP_DigestUpdate(ctx.get(), buffer.data(), size_t(n)))
            throw std::runtime_error("Cannot verify model");
    }
    unsigned char digest[32];
    unsigned int size = 0;
    if (!in.eof() || !EVP_DigestFinal_ex(ctx.get(), digest, &size))
        throw std::runtime_error("Cannot read model for verification");
    std::ostringstream hex;
    for (unsigned char c : digest)
        hex << std::hex << std::setfill('0') << std::setw(2) << unsigned(c);
    if (bytes != 1629326784 || hex.str() != model_sha)
        throw std::runtime_error("MeanVC2 requires the pinned F32 model; integrity verification failed");
}
void download_model(const std::string &destination) {
    namespace fs = std::filesystem;
    auto path = fs::absolute(destination);
    fs::create_directories(path.parent_path());
    int lock = open((path.string() + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock < 0)
        throw std::runtime_error("Cannot lock MeanVC2 model directory");
    struct Lock {
        int fd;
        ~Lock() { close(fd); }
    } guard{lock};
    if (flock(lock, LOCK_EX))
        throw std::runtime_error("Cannot lock MeanVC2 model");
    if (fs::exists(path)) {
        verify_model(path.string());
        return;
    }
    std::string temp = path.string() + ".part-XXXXXX";
    int fd = mkstemp(temp.data());
    if (fd < 0)
        throw std::runtime_error("Cannot create temporary model");
    close(fd);
    try {
        const auto result =
            ezlocalai::run_process({"curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "20",
                                    "--max-time", "600", "--output", temp, model_url},
                                   true);
        if (result.status != 0)
            throw std::runtime_error("Pinned MeanVC2 model download failed");
        verify_model(temp);
        fs::rename(temp, path);
    } catch (...) {
        fs::remove(temp);
        throw;
    }
}
} // namespace
int main(int argc, char **argv) {
    const int protocol = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
    if (protocol < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
        return 1;
    const auto emit = [&](const Json &j) { write_all(protocol, j.dump() + "\n"); };
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            emit({{"engine", "meanvc2"},
                  {"source", "cd770076152085af5b166521185bec76f9d47808"},
                  {"speaker_cache", "reference-v1"},
                  {"model_sha256", model_sha}});
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--download-model") {
            download_model(argv[2]);
            emit({{"ok", true}, {"model_sha256", model_sha}});
            return 0;
        }
        if (argc != 6)
            throw std::runtime_error("usage: ezlocalai-meanvc2 MODEL VOICE_WAV BACKEND DEVICE THREADS");
        verify_model(argv[1]);
        auto registry = rt::make_default_registry();
        rt::ModelLoadRequest load;
        load.model_path = argv[1];
        load.family_hint = "meanvc2";
        auto model = registry.load(load);
        rt::SessionOptions options;
        const std::string backend = argv[3];
        if (backend == "cuda")
            options.backend.type = engine::core::BackendType::Cuda;
        else if (backend == "hip")
            options.backend.type = engine::core::BackendType::Hip;
        else if (backend != "cpu")
            throw std::runtime_error("MeanVC2 backend must be cpu, cuda or hip");
        options.backend.device = std::stoi(argv[4]);
        options.backend.threads = std::stoi(argv[5]);
        auto owner =
            model->create_task_session({rt::VoiceTaskKind::VoiceConversion, rt::RunMode::Streaming}, options);
        auto *session = dynamic_cast<rt::IStreamingVoiceTaskSession *>(owner.get());
        if (!session)
            throw std::runtime_error("MeanVC2 streaming interface unavailable");
        const auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
        if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty() ||
            wav.samples.size() > size_t(wav.sample_rate) * wav.channels * 60)
            throw std::runtime_error("Reference voice must contain 0-60 seconds of PCM audio");
        for (float value : wav.samples)
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite reference voice audio");
        rt::TaskRequest request;
        request.voice = rt::VoiceCondition{};
        request.voice->speaker = rt::VoiceReference{};
        request.voice->speaker->audio = rt::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
        request.options["seed"] = "42";
        rt::SessionPreparationRequest prep;
        prep.voice = request.voice;
        prep.audio = rt::AudioPreparationContract{16000, 1, 2560};
        session->prepare(prep);
        session->start_stream(request);
        size_t emitted = 0;
        ezlocalai::VoiceResampler down(24000, 16000), up(16000, 24000);
        emit({{"ready", true}, {"sample_rate", 24000}, {"bandwidth_rate", 16000}, {"engine", "meanvc2"}});
        ezlocalai::VoiceBlockBuffer blocks;
        uint64_t source_samples = 0, wire_emitted = 0;
        std::vector<char> line(256 * 1024);
        while (std::cin.getline(line.data(), line.size())) {
            const auto msg = Json::parse(line.data());
            const auto op = msg.at("op").get<std::string>();
            rt::AudioBuffer audio;
            audio.sample_rate = 16000;
            if (op == "audio") {
                const auto bytes = ezlocalai::voice_base64_decode(msg.at("pcm_s16").get<std::string>());
                if (bytes.size() % 2 || bytes.size() > 3840 * 2)
                    throw std::runtime_error("MeanVC2 accepts at most 160ms of 24kHz PCM16 per block");
                std::vector<float> source(bytes.size() / 2);
                for (size_t i = 0; i < source.size(); ++i) {
                    const auto bits =
                        uint16_t(uint8_t(bytes[2 * i])) | (uint16_t(uint8_t(bytes[2 * i + 1])) << 8);
                    source[i] = int16_t(bits) / 32768.f;
                }
                source_samples += source.size();
                if (source_samples > 24000 * 120)
                    throw std::runtime_error("MeanVC2 utterance exceeds 120 seconds");
                blocks.push(down.push(source), [&](const std::vector<float> &samples, uint64_t offset) {
                    rt::AudioChunk chunk;
                    chunk.sample_rate = 16000;
                    chunk.channels = 1;
                    chunk.start_sample = offset;
                    chunk.samples = samples;
                    auto event = session->process_audio_chunk(chunk);
                    if (event.audio_output)
                        audio.samples.insert(audio.samples.end(), event.audio_output->samples.begin(),
                                             event.audio_output->samples.end());
                });
            } else if (op == "finish" && source_samples) {
                auto feed = [&](const std::vector<float> &samples, uint64_t offset) {
                    rt::AudioChunk chunk;
                    chunk.sample_rate = 16000;
                    chunk.channels = 1;
                    chunk.start_sample = offset;
                    chunk.samples = samples;
                    auto event = session->process_audio_chunk(chunk);
                    if (event.audio_output)
                        audio.samples.insert(audio.samples.end(), event.audio_output->samples.begin(),
                                             event.audio_output->samples.end());
                };
                blocks.push(down.finish(), feed);
                blocks.finish(feed);
                const auto already = emitted + audio.samples.size();
                auto final = session->finish_stream();
                if (final.audio_output) {
                    if (final.audio_output->samples.size() < already)
                        throw std::runtime_error("MeanVC2 final output shrank");
                    audio.samples.insert(audio.samples.end(), final.audio_output->samples.begin() + already,
                                         final.audio_output->samples.end());
                }
            } else if (op != "reset" && op != "finish")
                throw std::runtime_error("Unknown MeanVC2 operation");
            for (float v : audio.samples)
                if (!std::isfinite(v))
                    throw std::runtime_error("Nonfinite MeanVC2 output");
            emitted += audio.samples.size();
            if (emitted > 16000 * 125)
                throw std::runtime_error("MeanVC2 output exceeds utterance limit");
            auto output = up.push(audio.samples);
            if (op == "finish") {
                auto tail = up.finish();
                output.insert(output.end(), tail.begin(), tail.end());
            }
            if (output.size() > source_samples - wire_emitted)
                output.resize(size_t(source_samples - wire_emitted));
            wire_emitted += output.size();
            std::string pcm(output.size() * 2, '\0');
            for (size_t i = 0; i < output.size(); ++i) {
                const auto v =
                    int16_t(std::max(-32768., std::min(32767., std::round(double(output[i]) * 32768))));
                pcm[2 * i] = char(uint16_t(v) & 255);
                pcm[2 * i + 1] = char(uint16_t(v) >> 8);
            }
            emit({{"ok", true}, {"pcm_s16", ezlocalai::voice_base64_encode(pcm)}, {"sample_rate", 24000}});
            if (op == "finish" || op == "reset") {
                session->reset();
                session->start_stream(request);
                emitted = 0;
                source_samples = wire_emitted = 0;
                blocks.reset();
                down.reset();
                up.reset();
            }
        }
        if (!std::cin.eof())
            throw std::runtime_error("MeanVC2 protocol line exceeds limit");
        return 0;
    } catch (const std::exception &e) {
        try {
            emit({{"error", e.what()}});
        } catch (...) {
        }
        return 1;
    }
}
