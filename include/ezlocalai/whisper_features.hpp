#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace ezlocalai {
enum class WhisperFeatureProfile { faster_whisper_1_2_0, faster_whisper_1_0_3 };
struct WhisperFeatureConfiguration {
    int64_t feature_size = 80, sampling_rate = 16000, hop_length = 160, chunk_length = 30, n_fft = 400;
    int64_t maximum_frames() const { return chunk_length * sampling_rate / hop_length; }
    int64_t frames_per_second() const { return sampling_rate / hop_length; }
    int64_t tokens_per_second() const { return sampling_rate / (hop_length * 2); }
    double time_per_frame() const { return double(hop_length) / sampling_rate; }
};
struct WhisperFeatures {
    size_t bands = 80, frames = 0;
    std::vector<float> values; // Row-major [mel band, frame].
};
// faster-whisper/PyAV: first audio stream, signed 16-bit mono, then /32768.
std::vector<float> whisper_decode_audio(const std::string &path, int sample_rate = 16000);
WhisperFeatures whisper_log_mel(const std::vector<float> &audio, size_t bands = 80,
    size_t padding = 160, int sample_rate = 16000, size_t n_fft = 400, size_t hop = 160,
    WhisperFeatureProfile profile = WhisperFeatureProfile::faster_whisper_1_2_0);
WhisperFeatures whisper_log_mel_legacy(const std::vector<float> &audio,size_t bands = 80,
    bool padding = true,size_t chunk_seconds = 30);
WhisperFeatures whisper_feature_window(const WhisperFeatures &input, size_t start,
    size_t count, size_t padded_frames = 3000);
}
