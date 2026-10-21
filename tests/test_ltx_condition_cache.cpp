#include "ltx_condition_cache.hpp"
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() try {
    LtxConditionCache cache;
    const std::vector<int64_t> shape{2, 2};
    std::vector<float> positive{1, 2, 3, 4}, negative{0, 0, 0, 0};
    auto lookup = [&](const std::vector<float>& values) { return cache.find(shape, values, true, 1, 1); };
    check(!lookup(positive), "unprepared input must miss");
    cache.commit(shape, positive, true, 1, 1);
    cache.commit(shape, negative, true, 1, 1);
    check(lookup(positive) == 0 && lookup(negative) == 1, "both CFG conditions survive alternating lookups");
    const auto copied = positive;
    check(lookup(copied) == 0, "equal values in a different buffer reuse results");
    positive[0] = 9;
    check(!lookup(positive) && lookup(copied) == 0, "buffer mutation cannot change owned identity");
    check(!cache.find({4, 1}, copied, true, 1, 1), "shape participates in identity");
    check(!cache.find(shape, copied, false, 1, 1), "video-only and audio-video are distinct");
    check(!cache.find(shape, copied, true, 2, 1), "linear scale participates in identity");
    check(!cache.find(shape, copied, true, 1, 2), "attention scale participates in identity");
    check(cache.replacement_slot() == 0 && lookup(copied) == 0, "selecting a replacement does not commit failed work");
    cache.commit(shape, positive, true, 1, 1);
    check(!lookup(copied) && lookup(positive) == 0 && lookup(negative) == 1, "third condition replaces a single bounded slot");
    cache.commit(shape, negative, true, 1, 1, false);
    check(!lookup(negative), "mutable adapter results cannot be reused");
    cache.clear();
    check(!lookup(positive) && cache.replacement_slot() == 0, "session cleanup discards identities");
    uint32_t nan_bits = 0x7fc00001;
    std::memcpy(&positive[0], &nan_bits, sizeof(nan_bits));
    cache.commit(shape, positive, true, 1, 1);
    check(lookup(positive) == 0, "identical NaN bytes compare exactly");
    ++nan_bits;
    std::memcpy(&positive[0], &nan_bits, sizeof(nan_bits));
    check(!lookup(positive), "different NaN payload cannot reuse results");
    std::cout << "LTX conditioning cache identity, capacity, failure and session cleanup checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
