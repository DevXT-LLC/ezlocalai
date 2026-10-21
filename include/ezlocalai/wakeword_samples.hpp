#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ezlocalai {
// NumPy RandomState's scalar-seeded MT19937 and cached legacy normal stream.
// Keep one instance across augmentation and negative sample generation.
class WakewordRandom {
    std::mt19937 engine_;
    bool has_normal_ = false;
    double normal_ = 0;
public:
    WakewordRandom();
    explicit WakewordRandom(uint32_t seed);
    double uniform();
    double normal();
    unsigned noise_kind();
};
struct WakewordTtsSample {
    std::string audio, engine, voice, word, variation = "normal";
    int sample_rate = 16000;
};
struct WakewordTrainingSample {
    std::string audio;
    std::map<std::string, std::string> metadata;
};
using WakewordSampleProgress = std::function<void(size_t current, size_t total, const std::string &stage)>;
// Normalize already-decoded integer PCM WAV like the source's TTS adapters:
// pydub.set_frame_rate(16000), then set_channels(1), then WAV export.
std::string wakeword_prepare_tts_wav(const std::string &decoded_wav);
// pydub/audioop-compatible PCM WAV preparation, preserving the WAV's rate,
// channel count and sample width (24-bit input becomes 32-bit like pydub).
std::vector<std::pair<std::string, std::string>> wakeword_augment_sample(const std::string &wav, WakewordRandom &random);
std::vector<WakewordTrainingSample> wakeword_augment_dataset(const std::vector<WakewordTtsSample> &samples,
    WakewordRandom &random, int64_t augmentations_per_sample = 5, const WakewordSampleProgress &progress = {},
    const std::function<void()> &check_cancel = {});
std::vector<WakewordTrainingSample> wakeword_noise_samples(int64_t count, WakewordRandom &random,
    int sample_rate = 16000, double duration = 1.0, const std::function<void()> &check_cancel = {});
}
