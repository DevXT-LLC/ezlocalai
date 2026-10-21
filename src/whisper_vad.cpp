// Ported from faster-whisper 1.0.3/1.2.0 (MIT), adapted there from Silero VAD.
// See licenses/faster-whisper.txt and licenses/silero-vad.txt.
#include "ezlocalai/whisper_vad.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>

namespace ezlocalai {
WhisperVadOptions whisper_vad_options(WhisperVadProfile profile) {
    WhisperVadOptions options;
    if (profile == WhisperVadProfile::faster_whisper_1_0_3) options.min_speech_duration_ms = 250;
    return options;
}
std::vector<WhisperSpeechChunk> whisper_speech_timestamps(const std::vector<float> &probabilities,
    int64_t audio_samples, const WhisperVadOptions &options, WhisperVadProfile profile, int sampling_rate) {
    const bool legacy = profile == WhisperVadProfile::faster_whisper_1_0_3;
    if (legacy) sampling_rate = 16000;
    if (sampling_rate <= 0 || audio_samples < 0) throw std::invalid_argument("Invalid VAD audio dimensions");
    const auto min_speech = sampling_rate * options.min_speech_duration_ms / 1000;
    const auto padding = sampling_rate * options.speech_pad_ms / 1000;
    const auto max_speech = sampling_rate * options.max_speech_duration_s - 512 - 2 * padding;
    const auto min_silence = sampling_rate * options.min_silence_duration_ms / 1000;
    const auto max_silence = sampling_rate * 98 / 1000.0;
    const double threshold = legacy ? static_cast<float>(options.threshold) : options.threshold;
    const auto negative = legacy ? static_cast<double>(static_cast<float>(options.threshold - 0.15)) :
        options.neg_threshold.value_or(std::max(options.threshold - 0.15, 0.01));
    bool triggered = false;
    std::optional<int64_t> start;
    int64_t temp_end = 0, prev_end = 0, next_start = 0;
    std::vector<WhisperSpeechChunk> speeches;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        const double probability = probabilities[i];
        const auto position = static_cast<int64_t>(i) * 512;
        if (probability >= threshold && temp_end) {
            temp_end = 0;
            if (next_start < prev_end) next_start = position;
        }
        if (probability >= threshold && !triggered) {
            triggered = true; start = position; continue;
        }
        if (triggered && position - start.value() > max_speech) {
            if (prev_end) {
                speeches.push_back({start.value(),prev_end}); start.reset();
                if (next_start < prev_end) triggered = false;
                else start = next_start;
                prev_end = next_start = temp_end = 0;
            } else {
                speeches.push_back({start.value(),position}); start.reset();
                prev_end = next_start = temp_end = 0; triggered = false; continue;
            }
        }
        if (probability < negative && triggered) {
            if (!temp_end) temp_end = position;
            if (position - temp_end > max_silence) prev_end = temp_end;
            if (position - temp_end < min_silence) continue;
            if (temp_end - start.value() > min_speech) speeches.push_back({start.value(),temp_end});
            start.reset(); prev_end = next_start = temp_end = 0; triggered = false;
        }
    }
    if (start && audio_samples - *start > min_speech) speeches.push_back({*start,audio_samples});
    for (size_t i = 0; i < speeches.size(); ++i) {
        auto &speech = speeches[i];
        if (!i) speech.start = static_cast<int64_t>(std::max(0.0,speech.start - padding));
        if (i + 1 < speeches.size()) {
            auto &next = speeches[i + 1]; const auto silence = next.start - speech.end;
            if (silence < 2 * padding) {
                const auto half = static_cast<int64_t>(std::floor(silence / 2.0));
                speech.end += half; next.start = std::max<int64_t>(0,next.start - half);
            } else {
                speech.end = static_cast<int64_t>(std::min<double>(audio_samples,speech.end + padding));
                next.start = static_cast<int64_t>(std::max(0.0,next.start - padding));
            }
        } else speech.end = static_cast<int64_t>(std::min<double>(audio_samples,speech.end + padding));
    }
    return speeches;
}
namespace {
void append_slice(std::vector<float> &output, const std::vector<float> &audio, WhisperSpeechChunk chunk) {
    const auto size = static_cast<int64_t>(audio.size());
    const auto bound = [size](int64_t value) { return std::clamp(value < 0 ? value + size : value,int64_t{0},size); };
    const auto begin = bound(chunk.start), end = std::max(begin,bound(chunk.end));
    output.insert(output.end(),audio.begin() + begin,audio.begin() + end);
}
}
std::vector<WhisperAudioChunk> whisper_collect_chunks(const std::vector<float> &audio,
    const std::vector<WhisperSpeechChunk> &chunks, int sampling_rate, double max_duration) {
    if (sampling_rate <= 0) throw std::invalid_argument("Invalid VAD sample rate");
    std::vector<WhisperAudioChunk> result;
    WhisperAudioChunk current;
    int64_t duration = 0, total = 0;
    const auto finish = [&] {
        current.offset = static_cast<double>(total) / sampling_rate;
        current.duration = static_cast<double>(duration) / sampling_rate;
        result.push_back(std::move(current)); current = {};
        total += duration;
    };
    for (const auto &chunk : chunks) {
        const auto length = chunk.end - chunk.start;
        if (duration + length > max_duration * sampling_rate) {
            finish(); append_slice(current.audio,audio,chunk); duration = length;
            // The source omits this first segment from the new group's metadata.
        } else {
            current.segments.push_back(chunk); append_slice(current.audio,audio,chunk); duration += length;
        }
    }
    finish(); return result;
}
std::vector<float> whisper_collect_chunks_legacy(const std::vector<float> &audio,
    const std::vector<WhisperSpeechChunk> &chunks) {
    std::vector<float> output; for (const auto &chunk : chunks) append_slice(output,audio,chunk); return output;
}
WhisperSpeechTimestampsMap::WhisperSpeechTimestampsMap(const std::vector<WhisperSpeechChunk> &chunks,
    int sampling_rate, int time_precision, WhisperVadProfile profile)
    : sampling_rate_(sampling_rate), time_precision_(time_precision), profile_(profile) {
    if (sampling_rate <= 0) throw std::invalid_argument("Invalid VAD timestamp sample rate");
    int64_t previous = 0, silent = 0;
    for (const auto &chunk : chunks) {
        silent += chunk.start - previous; previous = chunk.end;
        ends_.push_back(chunk.end - silent); silence_.push_back(static_cast<double>(silent) / sampling_rate);
    }
}
int64_t WhisperSpeechTimestampsMap::chunk_index(double time, bool is_end) const {
    const auto scaled = time * sampling_rate_;
    if (!std::isfinite(scaled) || scaled >= static_cast<double>(INT64_MAX) || scaled < static_cast<double>(INT64_MIN))
        throw std::invalid_argument("Invalid VAD timestamp");
    const auto sample = static_cast<int64_t>(scaled);
    if (is_end && profile_ == WhisperVadProfile::faster_whisper_1_2_0) {
        const auto found = std::find(ends_.begin(),ends_.end(),sample);
        if (found != ends_.end()) return found - ends_.begin();
    }
    return std::min<int64_t>(std::upper_bound(ends_.begin(),ends_.end(),sample) - ends_.begin(),static_cast<int64_t>(ends_.size()) - 1);
}
double WhisperSpeechTimestampsMap::original_time(double time, std::optional<int64_t> index, bool is_end) const {
    auto selected = index ? *index : chunk_index(time,is_end);
    if (selected < 0) selected += static_cast<int64_t>(silence_.size());
    const auto value = silence_.at(static_cast<size_t>(selected)) + time;
    if (!std::isfinite(value)) return value;
    if (time_precision_ < 0 || time_precision_ > 15) throw std::invalid_argument("Unsupported VAD timestamp precision");
    char output[512]; const auto result = std::to_chars(output,output + sizeof(output),value,std::chars_format::fixed,time_precision_);
    if (result.ec != std::errc{}) throw std::runtime_error("Cannot round VAD timestamp");
    return std::stod(std::string(output,result.ptr));
}
}
