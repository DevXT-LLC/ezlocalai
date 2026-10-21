#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

// Input identities for two request-local connector results (positive/negative
// CFG). Exact owned bytes avoid stale results after buffer reuse or mutation.
class LtxConditionCache {
    struct Entry {
        std::vector<int64_t> shape;
        std::vector<float> values;
        bool audio = false, valid = false;
        float linear_scale = 0, attention_scale = 0;
    };
    std::array<Entry, 2> entries_;
    size_t next_ = 0;
public:
    std::optional<size_t> find(const std::vector<int64_t>& shape, const std::vector<float>& values,
                              bool audio, float linear_scale, float attention_scale) const {
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& entry = entries_[i];
            if (entry.valid && entry.audio == audio && entry.shape == shape &&
                entry.values.size() == values.size() && entry.linear_scale == linear_scale &&
                entry.attention_scale == attention_scale &&
                (values.empty() || std::memcmp(entry.values.data(), values.data(), values.size() * sizeof(float)) == 0))
                return i;
        }
        return std::nullopt;
    }
    size_t replacement_slot() const { return next_; }
    void commit(const std::vector<int64_t>& shape, const std::vector<float>& values,
                bool audio, float linear_scale, float attention_scale, bool reusable = true) {
        // Construct first so allocation failure leaves the previous identity intact.
        entries_[next_] = Entry{shape, values, audio, reusable, linear_scale, attention_scale};
        next_ = (next_ + 1) % entries_.size();
    }
    void clear() { entries_ = {}; next_ = 0; }
};
