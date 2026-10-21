#pragma once
#include "ezlocalai/nf4.hpp"
#include <limits>

namespace ezlocalai {
// Native-endian in-memory storage shared by CPU and CUDA. Projection values and
// their nested statistics stay compressed, alongside one scalar offset.
struct Nf4StorageLayout {
    size_t count, blocks, groups, statistics, maxima, offset, bytes;
    explicit Nf4StorageLayout(size_t elements) : count(elements) {
        if (!count || count > (std::numeric_limits<size_t>::max() - 32) / 2)
            throw std::invalid_argument("Invalid NF4 element count");
        blocks = count / 64 + (count % 64 != 0);
        groups = blocks / 256 + (blocks % 256 != 0);
        statistics = count / 2 + count % 2;
        maxima = (statistics + blocks + 3) & ~size_t(3);
        offset = maxima + groups * sizeof(float);
        bytes = offset + sizeof(float);
    }
};

inline std::vector<uint8_t> pack_nf4_storage(const Nf4Weights &weights) {
    const Nf4StorageLayout layout(weights.count);
    if (weights.packed.size() != layout.statistics || weights.nested.size() != layout.blocks
        || weights.nested_absmax.size() != layout.groups || !std::isfinite(weights.offset))
        throw std::invalid_argument("Invalid nested NF4 weights");
    for (float maximum : weights.nested_absmax)
        if (!std::isfinite(maximum) || maximum < 0) throw std::invalid_argument("Invalid NF4 nested maximum");
    std::vector<uint8_t> storage(layout.bytes);
    std::memcpy(storage.data(), weights.packed.data(), weights.packed.size());
    std::memcpy(storage.data() + layout.statistics, weights.nested.data(), weights.nested.size());
    std::memcpy(storage.data() + layout.maxima, weights.nested_absmax.data(), layout.groups * sizeof(float));
    std::memcpy(storage.data() + layout.offset, &weights.offset, sizeof(float));
    return storage;
}

inline float nf4_storage_value(const uint8_t *storage, const Nf4StorageLayout &layout, size_t index) {
    const size_t block = index / 64;
    float maximum, offset;
    std::memcpy(&maximum, storage + layout.maxima + (block / 256) * sizeof(float), sizeof(float));
    std::memcpy(&offset, storage + layout.offset, sizeof(float));
    // BNB performs dequantization and offset addition separately; do not fuse.
    const volatile float centered = nf4_detail::dynamic_code[storage[layout.statistics + block]] * maximum;
    const float scale = centered + offset;
    const auto code = (storage[index / 2] >> (index % 2 ? 0 : 4)) & 15;
    return nf4_detail::nf4_dequantization_lut[code] * scale;
}
} // namespace ezlocalai
