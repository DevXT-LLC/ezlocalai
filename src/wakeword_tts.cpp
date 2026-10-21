#include "ezlocalai/wakeword_tts.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/edge_tts.hpp"
#include "ezlocalai/tts_service.hpp"
#include <nlohmann/json.hpp>
#include <unicode/unistr.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace ezlocalai {
namespace {
// pydub conversion blocks the Python event loop. Keep one decoder active even
// while Edge network requests are concurrent, rather than spawning 24 FFmpegs.
std::mutex decode_mutex;
const std::vector<std::string> languages{"en", "en-au", "en-uk", "en-us", "en-ca", "en-in"};
const std::vector<std::string> voices{
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural", "en-US-DavisNeural",
    "en-US-AmberNeural", "en-US-AndrewNeural", "en-US-BrandonNeural", "en-US-ChristopherNeural",
    "en-US-CoraNeural", "en-US-ElizabethNeural", "en-US-EricNeural", "en-US-JacobNeural",
    "en-US-MichelleNeural", "en-US-MonicaNeural", "en-US-RogerNeural", "en-US-SaraNeural",
    "en-US-SteffanNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"};
size_t slice(size_t size, int64_t count) {
    if (count < 0) return uint64_t(-(count + 1)) >= size ? 0 : size - uint64_t(-(count + 1)) - 1;
    return std::min(size, size_t(count));
}
struct Workspace {
    std::filesystem::path path;
    Workspace() {
        auto name = (std::filesystem::temp_directory_path() / "ezlocalai-wakeword-mp3-XXXXXX").string();
        if (!mkdtemp(name.data())) throw std::system_error(errno, std::generic_category(), "Wake-word MP3 temporary directory"); path = name;
    }
    ~Workspace() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
std::string read(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("Cannot read decoded wake-word audio");
    return {std::istreambuf_iterator<char>(input), {}};
}
}
std::vector<GttsRequest> wakeword_gtts_requests(const std::string &word, const std::string &language) {
    if (std::find(languages.begin(), languages.end(), language) == languages.end())
        throw std::invalid_argument("Unsupported wake-word gTTS language: " + language);
    // gTTS 2.5.4 maps all five configured regional language codes to en.
    // Do not apply the ordinary speech endpoint's automatic text-language choice.
    return gtts_prepare(word, "\"en\"");
}
std::string wakeword_gtts_mp3(const std::string &word, const std::string &language, GttsOperations operations) {
    const auto requests = wakeword_gtts_requests(word, language); operations = gtts_native_operations(std::move(operations));
    std::string audio; for (const auto &request : requests) audio += gtts_response_audio(operations.post(request)); return audio;
}
std::string wakeword_decode_mp3(const std::string &mp3) {
    Workspace work; const auto input = work.path / "input.mp3", output = work.path / "output.wav";
    { std::ofstream file(input, std::ios::binary); file.write(mp3.data(), mp3.size()); if (!file) throw std::runtime_error("Cannot write wake-word MP3 input"); }
    // Match pydub's seekable cache input and MP3 float-decoder -> PCM16 choice.
    // Resampling/downmix happen afterwards through audioop-compatible routines.
    const auto result = run_process_to_files({"ffmpeg", "-y", "-v", "error", "-f", "mp3", "-read_ahead_limit", "-1", "-i", "cache:pipe:0",
        "-acodec", "pcm_s16le", "-vn", "-f", "wav", output.string()}, (work.path / "stdout").string(), (work.path / "stderr").string(), -1, false, input.string(), true);
    if (result.status) throw std::runtime_error("Wake-word MP3 decoding failed: " + read(work.path / "stderr"));
    return wakeword_prepare_tts_wav(read(output));
}
WakewordTtsOperations wakeword_native_tts_operations() {
    return {[](const std::string &word, const std::string &language) { return wakeword_gtts_mp3(word, language); },
        [](const std::string &word, const std::string &voice) { return edge_tts_mp3(word, voice); }, wakeword_decode_mp3};
}
WakewordTtsOperations wakeword_qwen_tts_operations(TtsService &service, std::vector<std::string> voices, const VoiceCancellation &cancellation) {
    WakewordTtsOperations result;
    result.qwen_voices = std::move(voices); result.cancellation = cancellation;
    result.qwen_wav = [&service, cancellation](const std::string &word, const std::string &voice) {
        return service.generate_local(nlohmann::json{{"text", word}, {"voice", voice},
            {"language", "auto"}, {"use_cache", false}, {"local_uri", nullptr}}.dump(), cancellation);
    };
    return result;
}
WakewordSampleGenerator::WakewordSampleGenerator(const std::string &directory, WakewordTtsOperations operations, bool enable_gtts, bool enable_edge)
    : operations_(std::move(operations)), gtts_(enable_gtts && bool(operations_.gtts_mp3)), edge_(enable_edge && bool(operations_.edge_mp3)) {
    std::filesystem::create_directories(directory);
    if (!operations_.decode_mp3) operations_.decode_mp3 = wakeword_decode_mp3;
}
std::optional<WakewordTtsSample> WakewordSampleGenerator::generate_one(const std::string &word, const std::string &voice, bool edge) const {
    try {
        const auto mp3 = edge ? operations_.edge_mp3(word, voice) : operations_.gtts_mp3(word, voice);
        std::lock_guard<std::mutex> decode_lock(decode_mutex);
        return WakewordTtsSample{operations_.decode_mp3(mp3), edge ? "edge" : "gtts", voice, word, "normal", 16000};
    } catch (const std::exception &error) { std::cerr << "Wake-word TTS sample failed: " << error.what() << '\n'; return {}; }
}
std::vector<WakewordTtsSample> WakewordSampleGenerator::generate(const std::string &word, int64_t target, const WakewordSampleProgress &progress) const {
    if (operations_.qwen_wav) {
        if (operations_.qwen_voices.empty()) throw std::runtime_error("No Qwen voices available for wake word training");
        std::vector<WakewordTtsSample> samples;
        for (int64_t i = 0; i < target; ++i) {
            if (operations_.cancellation.requested()) throw VoiceCancelled("");
            const auto &voice = operations_.qwen_voices[size_t(i) % operations_.qwen_voices.size()];
            try {
                auto audio = wakeword_prepare_tts_wav(operations_.qwen_wav(word, voice));
                if (operations_.cancellation.requested()) throw VoiceCancelled("");
                samples.push_back({std::move(audio), "qwen", voice, word, "normal", 16000});
            } catch (const VoiceCancelled &) { throw; }
            catch (const TtsError &error) {
                if (error.category == "CancelledError" || error.category == "GeneratorExit" || error.category == "KeyboardInterrupt" || error.category == "SystemExit") throw;
                std::cerr << "Wake-word Qwen sample failed: " << error.what() << '\n';
            } catch (const std::exception &error) { std::cerr << "Wake-word Qwen sample failed: " << error.what() << '\n'; }
            if (progress) progress(size_t(i + 1), size_t(target), "generating");
        }
        return samples;
    }
    const int enabled = int(gtts_) + int(edge_); if (!enabled) throw std::runtime_error("No TTS engines available for wake word training");
    const int64_t per_engine = target / enabled - (target < 0 && target % enabled != 0 ? 1 : 0);
    std::vector<std::optional<WakewordTtsSample>> results;
    // Python gTTS coroutines contain no await: these complete sequentially
    // before Edge's asynchronous requests. Gather preserves creation order.
    if (gtts_) for (size_t i = 0; i < slice(languages.size(), per_engine); ++i) results.push_back(generate_one(word, languages[i], false));
    std::vector<std::future<std::optional<WakewordTtsSample>>> pending;
    if (edge_) for (size_t i = 0; i < slice(voices.size(), per_engine); ++i)
        pending.push_back(std::async(std::launch::async, [&, i] { return generate_one(word, voices[i], true); }));
    for (auto &future : pending) results.push_back(future.get());
    std::vector<WakewordTtsSample> samples;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i]) samples.push_back(std::move(*results[i]));
        if (progress && (i + 1) % 10 == 0) progress(i + 1, results.size(), "generating");
    }
    return samples;
}
std::vector<WakewordTrainingSample> wakeword_similar_samples(const std::string &word, const WakewordSampleGenerator &generator, int64_t count) {
    std::vector<std::string> patterns{"hey", "hay", "hi", "he", "huh", "ok", "okay", "oh", "ow", "yeah", "yes", "yep", "no", "nope", "the", "a", "is", "it", "to", "and", "what", "that", "this", "there", "one", "two", "three", "four", "five"};
    const auto unicode = icu::UnicodeString::fromUTF8(word);
    if (unicode.countChar32() > 2) {
        std::string prefix, suffix;
        unicode.tempSubStringBetween(0, unicode.moveIndex32(unicode.length(), -1)).toUTF8String(prefix);
        unicode.tempSubString(unicode.moveIndex32(0, 1)).toUTF8String(suffix);
        patterns.insert(patterns.end(), {prefix, suffix, word + "s", word + "ing"});
    }
    std::vector<WakewordTrainingSample> result;
    for (size_t i = 0; i < slice(patterns.size(), count); ++i) {
        try { for (const auto &sample : generator.generate(patterns[i], 2)) result.push_back({sample.audio, {{"type", "similar_word"}, {"word", patterns[i]}}}); }
        catch (const VoiceCancelled &) { throw; }
        catch (const TtsError &error) {
            if (error.category == "CancelledError" || error.category == "GeneratorExit" || error.category == "KeyboardInterrupt" || error.category == "SystemExit") throw;
            std::cerr << "Could not generate similar-word sample: " << error.what() << '\n';
        }
        catch (const std::exception &error) { std::cerr << "Could not generate similar-word sample: " << error.what() << '\n'; }
    }
    return result;
}
}
