// Algorithms follow NumPy RandomState, CPython audioop, pydub and SciPy lfilter.
// See licenses/{NumPy,CPython,pydub,SciPy}.txt.
#include "ezlocalai/wakeword_samples.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ezlocalai {
WakewordRandom::WakewordRandom() : WakewordRandom(std::random_device{}()) {}
WakewordRandom::WakewordRandom(uint32_t seed) : engine_(seed) {}
double WakewordRandom::uniform() {
    const uint32_t a = engine_() >> 5, b = engine_() >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
double WakewordRandom::normal() {
    if (has_normal_) { has_normal_ = false; const auto value = normal_; normal_ = 0; return value; }
    double x, y, radius;
    do { x = 2 * uniform() - 1; y = 2 * uniform() - 1; radius = x * x + y * y; } while (radius >= 1 || radius == 0);
    const auto factor = std::sqrt(-2 * std::log(radius) / radius);
    normal_ = factor * x; has_normal_ = true; return factor * y;
}
unsigned WakewordRandom::noise_kind() { return engine_() & 3u; }
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
uint32_t read_le(const std::string &bytes, size_t offset, int width) {
    require(offset <= bytes.size() && size_t(width) <= bytes.size() - offset, "Truncated WAV data");
    uint32_t value = 0; for (int i = 0; i < width; ++i) value |= uint32_t(uint8_t(bytes[offset + i])) << (8 * i); return value;
}
void append_le(std::string &bytes, uint32_t value, int width) { for (int i = 0; i < width; ++i) bytes += char((value >> (8 * i)) & 255); }
struct Pcm {
    std::string bytes;
    int rate = 16000, channels = 1, width = 2;
    int64_t sample(size_t index) const {
        const auto value = read_le(bytes, index * width, width);
        const int bits = width * 8;
        return value & (uint32_t(1) << (bits - 1)) ? int64_t(value) - (int64_t(1) << bits) : value;
    }
    size_t count() const { return bytes.size() / width; }
    std::string wav() const {
        require(bytes.size() <= UINT32_MAX - 36 && rate > 0 && channels > 0 && channels <= UINT16_MAX, "Invalid WAV dimensions");
        require(uint64_t(rate) * channels * width <= UINT32_MAX && channels * width <= UINT16_MAX, "WAV header dimensions overflow");
        std::string data = bytes;
        if (width == 1) for (auto &byte : data) byte = char(uint8_t(byte) + 128);
        std::string result = "RIFF"; append_le(result, uint32_t(data.size() + 36), 4); result += "WAVEfmt "; append_le(result, 16, 4);
        append_le(result, 1, 2); append_le(result, channels, 2); append_le(result, rate, 4);
        append_le(result, uint32_t(uint64_t(rate) * channels * width), 4); append_le(result, channels * width, 2); append_le(result, width * 8, 2);
        result += "data"; append_le(result, uint32_t(data.size()), 4); result += data; return result;
    }
};
Pcm read_wav(const std::string &wav) {
    require(wav.size() >= 12 && wav.compare(0, 4, "RIFF") == 0 && wav.compare(8, 4, "WAVE") == 0, "Invalid PCM WAV");
    Pcm audio; bool format = false, data = false;
    for (size_t offset = 12; offset + 8 <= wav.size();) {
        const auto length = read_le(wav, offset + 4, 4); const size_t start = offset + 8;
        require(length <= wav.size() - start, "Truncated WAV chunk");
        if (wav.compare(offset, 4, "fmt ") == 0) {
            require(length >= 16 && read_le(wav, start, 2) == 1, "Only integer PCM WAV is supported for augmentation");
            audio.channels = read_le(wav, start + 2, 2); audio.rate = read_le(wav, start + 4, 4);
            const auto bits = read_le(wav, start + 14, 2); require(bits == 8 || bits == 16 || bits == 24 || bits == 32, "Unsupported WAV sample width");
            audio.width = bits / 8; format = true;
        } else if (wav.compare(offset, 4, "data") == 0) { audio.bytes = wav.substr(start, length); data = true; }
        offset = start + length + (length & 1u);
    }
    require(format && data && audio.rate > 0 && audio.channels > 0, "Invalid PCM WAV parameters");
    require(audio.bytes.size() % (audio.width * audio.channels) == 0, "Not a whole number of PCM frames");
    if (audio.width == 1) for (auto &byte : audio.bytes) byte = char(uint8_t(byte) - 128);
    if (audio.width == 3) {
        std::string converted; converted.reserve(audio.bytes.size() / 3 * 4);
        for (size_t i = 0; i < audio.bytes.size(); i += 3) { converted += uint8_t(audio.bytes[i + 2]) > 127 ? char(255) : char(0); converted.append(audio.bytes, i, 3); }
        audio.bytes = std::move(converted); audio.width = 4;
    }
    return audio;
}
Pcm change_rate(const Pcm &audio, int inrate) {
    require(inrate > 0, "Sampling rate not greater than zero");
    if (inrate == audio.rate) return audio;
    Pcm result = audio; result.bytes.clear();
    const auto divisor = std::gcd(inrate, audio.rate); inrate /= divisor; const auto outrate = audio.rate / divisor;
    int64_t phase = -outrate; size_t frame = 0; const size_t frames = audio.count() / audio.channels;
    std::vector<int64_t> previous(audio.channels), current(audio.channels);
    const int64_t scale = int64_t(1) << (32 - audio.width * 8);
    while (true) {
        while (phase < 0) {
            if (frame == frames) return result;
            for (int channel = 0; channel < audio.channels; ++channel) { previous[channel] = current[channel]; current[channel] = audio.sample(frame * audio.channels + channel) * scale; }
            ++frame; phase += outrate;
        }
        while (phase >= 0) {
            for (int channel = 0; channel < audio.channels; ++channel) {
                const auto value = int64_t((double(previous[channel]) * double(phase) + double(current[channel]) * double(outrate - phase)) / double(outrate));
                // audioop stores the upper bits of its signed 32-bit result.
                const auto narrowed = int64_t(std::floor(double(value) / double(scale)));
                append_le(result.bytes, uint32_t(narrowed), audio.width);
            }
            phase -= inrate;
        }
    }
}
int64_t clipped(double value, int width, bool floor) {
    const auto high = double((int64_t(1) << (width * 8 - 1)) - 1), low = -high - 1;
    value = std::max(low, std::min(high, value)); return int64_t(floor ? std::floor(value) : value);
}
Pcm volume(const Pcm &audio, int db) {
    Pcm result = audio; result.bytes.clear(); const auto factor = std::pow(10.0, double(db) / 20);
    for (size_t i = 0; i < audio.count(); ++i) append_le(result.bytes, uint32_t(clipped(double(audio.sample(i)) * factor, audio.width, true)), audio.width);
    return result;
}
Pcm noise(const Pcm &audio, double level, WakewordRandom &random) {
    Pcm result = audio; result.bytes.clear();
    for (size_t i = 0; i < audio.count(); ++i) append_le(result.bytes, uint32_t(clipped(double(float(audio.sample(i))) + (level * 32767) * random.normal(), 2, false)), 2);
    return result;
}
Pcm reverb(const Pcm &audio) {
    Pcm result = audio; result.bytes.clear(); const size_t delay = size_t(0.03 * audio.rate);
    for (size_t i = 0; i < audio.count(); ++i) {
        double value = float(audio.sample(i));
        if (i >= delay) value += float(float(audio.sample(i - delay)) * 0.3f);
        append_le(result.bytes, uint32_t(clipped(value, 2, false)), 2);
    }
    return result;
}
size_t slice_size(size_t size, int64_t count) {
    if (count < 0) return uint64_t(-(count + 1)) >= size ? 0 : size - uint64_t(-(count + 1)) - 1;
    return std::min(size, size_t(count));
}
}
std::string wakeword_prepare_tts_wav(const std::string &decoded_wav) {
    auto audio = read_wav(decoded_wav); const int source_rate = audio.rate;
    audio.rate = 16000; audio = change_rate(audio, source_rate);
    if (audio.channels != 1) {
        Pcm mono = audio; mono.bytes.clear(); mono.channels = 1;
        for (size_t frame = 0; frame < audio.count() / audio.channels; ++frame) {
            int64_t value = 0;
            if (audio.channels == 2) value = clipped(double(audio.sample(frame * 2)) * 0.5 + double(audio.sample(frame * 2 + 1)) * 0.5, audio.width, true);
            else for (int channel = 0; channel < audio.channels; ++channel) {
                value += int64_t(std::floor(double(audio.sample(frame * audio.channels + channel)) / audio.channels));
                const auto low = -(int64_t(1) << (audio.width * 8 - 1)), high = -low - 1;
                require(value >= low && value <= high, "Mono channel accumulation overflow");
            }
            append_le(mono.bytes, uint32_t(value), audio.width);
        }
        audio = std::move(mono);
    }
    return audio.wav();
}
std::vector<std::pair<std::string, std::string>> wakeword_augment_sample(const std::string &wav, WakewordRandom &random) {
    const auto audio = read_wav(wav); std::vector<std::pair<std::string, std::string>> result;
    const auto add = [&](const Pcm &value, const std::string &name) { result.emplace_back(value.wav(), name); };
    for (const auto &[speed, name] : std::array<std::pair<double, const char *>, 4>{{{0.85, "speed_0.85"}, {0.9, "speed_0.9"}, {1.1, "speed_1.1"}, {1.15, "speed_1.15"}}}) add(change_rate(audio, int(audio.rate * speed)), name);
    for (int semitones : {-2, -1, 1, 2}) add(change_rate(audio, int(audio.rate * std::pow(2.0, double(semitones) / 12))), "pitch_" + std::to_string(semitones));
    for (int db : {-6, -3, 3, 6}) add(volume(audio, db), "volume_" + std::to_string(db) + "db");
    for (const auto &[level, name] : std::array<std::pair<double, const char *>, 3>{{{0.005, "noise_0.005"}, {0.01, "noise_0.01"}, {0.02, "noise_0.02"}}}) add(noise(audio, level, random), name);
    add(reverb(audio), "reverb"); return result;
}
std::vector<WakewordTrainingSample> wakeword_augment_dataset(const std::vector<WakewordTtsSample> &samples, WakewordRandom &random,
    int64_t selected, const WakewordSampleProgress &progress, const std::function<void()> &check_cancel) {
    std::vector<WakewordTrainingSample> result;
    const auto add = [&](const WakewordTtsSample &sample, const std::string &bytes, const std::string &name) {
        result.push_back({bytes, {{"engine", sample.engine}, {"voice", sample.voice}, {"word", sample.word}, {"augmentation", name}}});
    };
    for (const auto &sample : samples) {
        if (check_cancel) check_cancel();
        add(sample, sample.audio, "original");
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        if (check_cancel) check_cancel();
        const auto variants = wakeword_augment_sample(samples[i].audio, random);
        for (size_t j = 0; j < slice_size(variants.size(), selected); ++j) add(samples[i], variants[j].first, variants[j].second);
        if (progress && (i + 1) % 10 == 0) progress(i + 1, samples.size(), "augmenting");
    }
    return result;
}
std::vector<WakewordTrainingSample> wakeword_noise_samples(int64_t count, WakewordRandom &random, int sample_rate, double duration,
    const std::function<void()> &check_cancel) {
    std::vector<WakewordTrainingSample> result;
    for (int64_t i = 0; i < count; ++i) {
        if (check_cancel) check_cancel();
        const auto kind = random.noise_kind();
        require(std::isfinite(duration) && duration * sample_rate >= 0 && duration * sample_rate < double(INT64_MAX), "Invalid noise sample duration");
        const auto length = size_t(duration * sample_rate); std::vector<double> values(length);
        for (auto &value : values) value = random.normal();
        if (kind == 1) {
            constexpr std::array<double, 4> b{0.049922035, -0.095993537, 0.050612699, -0.004408786}, a{1, -2.494956002, 2.017265875, -0.522189400};
            std::array<double, 3> state{};
            for (auto &value : values) {
                const auto y = state[0] + b[0] * value;
                for (size_t j = 0; j < 2; ++j) state[j] = state[j + 1] + value * b[j + 1] - y * a[j + 1];
                state[2] = value * b[3] - y * a[3]; value = y * 0.3;
            }
        } else if (kind == 2) {
            require(!values.empty(), "zero-size array to reduction operation maximum which has no identity");
            double sum = 0, maximum = 0;
            for (auto &value : values) { sum += value; value = sum * 0.001; maximum = std::max(maximum, std::abs(value)); }
            for (auto &value : values) value = value / (maximum + 1e-8) * 0.2;
        } else for (auto &value : values) value *= kind == 0 ? 0.1 : 0.001;
        Pcm audio; audio.rate = sample_rate;
        for (double value : values) append_le(audio.bytes, uint32_t(int64_t(value * 32767)), 2);
        const std::array<const char *, 4> names{"white", "pink", "brown", "silence"};
        result.push_back({audio.wav(), {{"type", "noise"}, {"noise_type", names[kind]}}});
    }
    return result;
}
}
