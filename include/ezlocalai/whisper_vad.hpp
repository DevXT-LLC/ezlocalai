#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ezlocalai {
enum class WhisperVadProfile { faster_whisper_1_2_0, faster_whisper_1_0_3 };
struct WhisperVadOptions {
    double threshold = 0.5;
    std::optional<double> neg_threshold;
    double min_speech_duration_ms = 0;
    double max_speech_duration_s = std::numeric_limits<double>::infinity();
    double min_silence_duration_ms = 2000;
    double speech_pad_ms = 400;
};
WhisperVadOptions whisper_vad_options(WhisperVadProfile profile);
struct WhisperSpeechChunk { int64_t start = 0, end = 0; };
struct WhisperVadAudio {
    std::vector<float> audio;
    std::vector<WhisperSpeechChunk> chunks;
};
std::vector<WhisperSpeechChunk> whisper_speech_timestamps(const std::vector<float> &probabilities,
    int64_t audio_samples, const WhisperVadOptions &options,
    WhisperVadProfile profile = WhisperVadProfile::faster_whisper_1_2_0, int sampling_rate = 16000);
struct WhisperAudioChunk {
    std::vector<float> audio;
    double offset = 0, duration = 0;
    std::vector<WhisperSpeechChunk> segments;
};
std::vector<WhisperAudioChunk> whisper_collect_chunks(const std::vector<float> &audio,
    const std::vector<WhisperSpeechChunk> &chunks, int sampling_rate = 16000,
    double max_duration = std::numeric_limits<double>::infinity());
std::vector<float> whisper_collect_chunks_legacy(const std::vector<float> &audio,
    const std::vector<WhisperSpeechChunk> &chunks);
class WhisperSpeechTimestampsMap {
public:
    WhisperSpeechTimestampsMap(const std::vector<WhisperSpeechChunk> &chunks, int sampling_rate,
        int time_precision = 2, WhisperVadProfile profile = WhisperVadProfile::faster_whisper_1_2_0);
    int64_t chunk_index(double time, bool is_end = false) const;
    double original_time(double time, std::optional<int64_t> index = {}, bool is_end = false) const;
private:
    int sampling_rate_, time_precision_;
    WhisperVadProfile profile_;
    std::vector<int64_t> ends_;
    std::vector<double> silence_;
};

// Available when the optional native ONNX Runtime target is enabled. Sessions
// are reusable, while recurrent state is local to each call.
class WhisperVadModel {
public:
    WhisperVadModel(WhisperVadProfile profile, const std::string &model_directory);
    ~WhisperVadModel();
    WhisperVadModel(WhisperVadModel &&) noexcept;
    WhisperVadModel &operator=(WhisperVadModel &&) noexcept;
    std::vector<float> probabilities(const std::vector<float> &audio) const;
    std::vector<WhisperSpeechChunk> speech_timestamps(const std::vector<float> &audio,
        const WhisperVadOptions &options) const;
    WhisperVadAudio filter(const std::vector<float> &audio, const WhisperVadOptions &options) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
