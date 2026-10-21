#pragma once
#include "ezlocalai/wakeword_samples.hpp"
#include "ezlocalai/gtts.hpp"
#include "ezlocalai/voice_slots.hpp"
#include <optional>

namespace ezlocalai {
class TtsService;
struct WakewordTtsOperations {
    std::function<std::string(const std::string &word, const std::string &language)> gtts_mp3;
    std::function<std::string(const std::string &word, const std::string &voice)> edge_mp3;
    std::function<std::string(const std::string &mp3)> decode_mp3;
    // When provided, Qwen is the sole engine; no historical provider is called.
    std::function<std::string(const std::string &word, const std::string &voice)> qwen_wav;
    std::vector<std::string> qwen_voices{"default"};
    VoiceCancellation cancellation;
};
std::vector<GttsRequest> wakeword_gtts_requests(const std::string &word, const std::string &language);
std::string wakeword_gtts_mp3(const std::string &word, const std::string &language, GttsOperations operations = {});
std::string wakeword_decode_mp3(const std::string &mp3);
WakewordTtsOperations wakeword_native_tts_operations();
// Service must outlive the generator. Each request bypasses the audio cache
// and uses shared Qwen admission, device selection and lease cleanup.
WakewordTtsOperations wakeword_qwen_tts_operations(TtsService &service,
    std::vector<std::string> voices={"default"},const VoiceCancellation &cancellation={});
class WakewordSampleGenerator {
    WakewordTtsOperations operations_;
    bool gtts_, edge_;
    std::optional<WakewordTtsSample> generate_one(const std::string &word, const std::string &voice, bool edge) const;
public:
    // Explicit adapters keep engine availability observable. Native Edge
    // transport is supplied by its adapter, never replaced by another engine.
    WakewordSampleGenerator(const std::string &output_directory, WakewordTtsOperations operations,
        bool enable_gtts = true, bool enable_edge = true);
    std::vector<WakewordTtsSample> generate(const std::string &word, int64_t target_count = 500,
        const WakewordSampleProgress &progress = {}) const;
};
std::vector<WakewordTrainingSample> wakeword_similar_samples(const std::string &word,
    const WakewordSampleGenerator &generator, int64_t count = 50);
}
