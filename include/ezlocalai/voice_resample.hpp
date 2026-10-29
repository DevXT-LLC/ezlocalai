#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace ezlocalai {
// Stateful windowed-sinc polyphase resampling. The 16-sample lookahead is
// retained across chunks; neither chunk boundaries nor pauses reset the filter.
class VoiceResampler {
    int source_, target_, phases_;
    int64_t count_ = 0, next_ = 0, base_ = 0;
    std::vector<float> pending_;
    std::vector<std::vector<double>> weights_;
    std::vector<float> produce(bool final) {
        std::vector<float> out;
        const int64_t wanted = (count_ * target_ + source_ - 1) / source_;
        while (next_ < wanted) {
            const int64_t numerator = next_ * source_, center = numerator / target_;
            if (!final && center + 16 >= count_)
                break;
            const auto &weights = weights_[size_t(next_ % phases_)];
            double value = 0;
            for (int j = 0; j < 32; ++j) {
                const auto index = center + j - 15;
                if (index >= 0 && index < count_) {
                    if (index < base_)
                        throw std::logic_error("Resampler history lost");
                    value += pending_.at(size_t(index - base_)) * weights[j];
                }
            }
            out.push_back(float(value));
            ++next_;
        }
        const auto keep = std::max<int64_t>(base_, std::min(count_, next_ * source_ / target_ - 15));
        pending_.erase(pending_.begin(), pending_.begin() + size_t(keep - base_));
        base_ = keep;
        return out;
    }

  public:
    VoiceResampler(int source, int target) : source_(source), target_(target), phases_(1) {
        if (source <= 0 || target <= 0 || source > 192000 || target > 192000)
            throw std::invalid_argument("Unsupported voice sample rate");
        phases_ = target / std::gcd(source, target);
        if (phases_ > 4096)
            throw std::invalid_argument("Voice resampler ratio exceeds limit");
        const double pi = std::acos(-1.), cutoff = .95 * std::min(1., double(target) / source);
        for (int phase = 0; phase < phases_; ++phase) {
            std::vector<double> w(32);
            double sum = 0;
            const double position = double((int64_t(phase) * source) % target) / target;
            for (int j = 0; j < 32; ++j) {
                const double d = j - 15 - position, u = d * cutoff;
                w[j] = std::abs(d) >= 16 ? 0
                                         : cutoff * (std::abs(u) < 1e-12 ? 1 : std::sin(pi * u) / (pi * u)) *
                                               .5 * (1 + std::cos(pi * d / 16));
                sum += w[j];
            }
            for (auto &v : w)
                v /= sum;
            weights_.push_back(std::move(w));
        }
    }
    std::vector<float> push(const std::vector<float> &input) {
        if (input.size() > 48000)
            throw std::runtime_error("Voice resampler input block too large");
        pending_.insert(pending_.end(), input.begin(), input.end());
        count_ += input.size();
        return produce(false);
    }
    std::vector<float> finish() { return produce(true); }
    void reset() {
        pending_.clear();
        count_ = next_ = base_ = 0;
    }
};
} // namespace ezlocalai

namespace ezlocalai {
// MeanVC2's frontend needs complete 160 ms windows. Transport fragments are not
// inference windows: handing it 10 ms packets can discard every feature frame.
class VoiceBlockBuffer {
    std::vector<float> pending_;
    uint64_t blocks_ = 0, samples_ = 0;
    template <class Sink> void emit(Sink &sink) {
        std::vector<float> block(2560, 0.f);
        const auto n = std::min(block.size(), pending_.size());
        std::copy_n(pending_.begin(), n, block.begin());
        sink(block, blocks_ * 2560);
        pending_.erase(pending_.begin(), pending_.begin() + n);
        ++blocks_;
    }

  public:
    template <class Sink> void push(const std::vector<float> &input, Sink sink) {
        if (input.size() > 48000)
            throw std::runtime_error("Voice input block exceeds limit");
        samples_ += input.size();
        pending_.insert(pending_.end(), input.begin(), input.end());
        while (pending_.size() >= 2560)
            emit(sink);
    }
    template <class Sink> void finish(Sink sink) {
        if (!samples_)
            return;
        if (!pending_.empty())
            emit(sink);
        // A very short acknowledgement still needs enough frontend context to
        // produce a condition frame. The caller trims padded output to input.
        while (blocks_ < 2)
            emit(sink);
    }
    void reset() {
        pending_.clear();
        blocks_ = samples_ = 0;
    }
};
} // namespace ezlocalai
