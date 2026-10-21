#include "ezlocalai/llm_batch.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_slots.hpp"
#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <unicode/uchar.h>
#include <unicode/utf8.h>

namespace ezlocalai {
namespace {
bool word(UChar32 c) {
    const auto category = u_charType(c);
    return c == '_' || (category >= U_UPPERCASE_LETTER && category <= U_OTHER_LETTER)
        || category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER || category == U_OTHER_NUMBER;
}
std::string family(const std::string &name) {
    const auto lower = llm_unicode_lower(name);
    for (size_t i = 0; i < lower.size(); ++i) {
        if (i) { auto offset = static_cast<int32_t>(i); UChar32 c; U8_PREV(lower.data(), 0, offset, c); if (word(c)) continue; }
        for (const auto token : {"3090", "4090", "5090", "t4", "a100", "h100"}) {
            const std::string value(token);
            if (lower.compare(i, value.size(), value)) continue;
            auto offset = static_cast<int32_t>(i + value.size());
            if (offset < static_cast<int32_t>(lower.size())) { UChar32 c; U8_NEXT(lower.data(), offset, static_cast<int32_t>(lower.size()), c); if (word(c)) continue; }
            auto result = value;
            for (auto &c : result) if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
            return result;
        }
    }
    return {};
}
std::string one_decimal(double value) {
    char buffer[128]; const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed, 1);
    if (result.ec != std::errc{}) throw std::runtime_error("GPU capacity formatting failed");
    return {buffer, result.ptr};
}
std::string normalized_setting(const std::string &value) { return llm_unicode_lower(strip_config_value(value)); }
bool automatic_setting(const std::string &value) { return value.empty() || value == "auto" || value == "0"; }
std::string minimum(const std::string &a, const std::string &b) { return configuration_integer_less(a, b) ? a : b; }
}
LlmBatchProbes native_llm_batch_probes() {
    return {[] { return visible_cuda_device_count() > 0; }, visible_cuda_device_count,
        cuda_memory_info, cuda_compute_capability, [](int64_t ordinal) {
            if (ordinal < std::numeric_limits<int>::min() || ordinal > std::numeric_limits<int>::max()) throw std::out_of_range("Invalid CUDA ordinal");
            return cuda_device_properties(static_cast<int>(ordinal));
        }};
}
GpuInferenceProfile gpu_inference_profile(const LlmBatchProbes &probes, std::optional<int64_t> main_gpu) {
    try {
        if (!probes.available()) return {};
        const auto count = main_gpu ? 1 : probes.count();
        GpuInferenceProfile best; bool have_best = false;
        for (int i = 0; i < count; ++i) {
            const auto index = main_gpu.value_or(i);
            const auto properties = probes.properties(index);
            GpuInferenceProfile candidate{family(properties.name), properties.total_memory / 1073741824.0};
            if (!have_best || std::make_pair(candidate.family == "5090", candidate.capacity_gb) > std::make_pair(best.family == "5090", best.capacity_gb)) best = candidate;
            have_best = true;
        }
        return best;
    } catch (const std::exception &) { return {}; }
}
LlmAutoBatch calculate_llm_auto_batch(int64_t main_gpu, int64_t context, const std::string &model, const LlmBatchProbes &probes) {
    if (!probes.available()) return {1024, 256, "cpu/default"};
    int gpu = 0, cc = 0; double free = 0, total = 0;
    try {
        const auto count = probes.count();
        gpu = main_gpu >= 0 && main_gpu < count ? static_cast<int>(main_gpu) : 0;
        const auto memory = probes.memory(gpu);
        free = memory.free_bytes / 1073741824.0; total = memory.total_bytes / 1073741824.0;
        const auto capability = probes.capability(gpu); cc = capability.first * 10 + capability.second;
    } catch (const std::exception &) { return {2048, 512, "probe-failed/default"}; }
    const auto profile = gpu_inference_profile(probes, gpu);
    int64_t batch = 1024, ubatch = 256;
    if (cc >= 89) {
        if (free >= 28) { batch = 8192; ubatch = 2048; }
        else if (free >= 20) { batch = 4096; ubatch = 1024; }
        else if (free >= 12) { batch = 2048; ubatch = 512; }
        else if (free >= 8) { batch = 1024; ubatch = 512; }
    } else if (cc >= 80) {
        if (profile.family == "A100" && free >= 60) { batch = 8192; ubatch = 2048; }
        else if (free >= 22 && context && context <= 300000) { batch = 4096; ubatch = 1024; }
        else if (free >= 12) { batch = 2048; ubatch = 512; }
        else if (free >= 8) { batch = 1024; ubatch = 512; }
    } else if (free >= 16) { batch = 2048; ubatch = 512; }
    if (profile.family == "T4") { batch = free >= 12 ? 512 : 256; ubatch = free >= 12 ? 128 : 64; }
    if (context) { batch = std::min(batch, std::max<int64_t>(1, context)); ubatch = std::min(ubatch, batch); }
    bool capped = false;
    const auto name = llm_unicode_lower(model);
    const bool qwen = name.find("qwen3.8-27b") != std::string::npos;
    if ((qwen || name.find("-mtp") != std::string::npos) && context >= 131072 && ubatch > 256) {
        const int64_t cap = qwen ? (total >= 28 || context <= 200000 ? 1024 : 512) : 256;
        if (ubatch > cap) { ubatch = cap; capped = true; }
    }
    std::string reason = "GPU " + std::to_string(gpu) + ", free=" + one_decimal(free) + "GB/" + one_decimal(total)
        + "GB, CC " + std::to_string(cc / 10) + '.' + std::to_string(cc % 10);
    if (!profile.family.empty()) reason += ", " + profile.family + " profile";
    if (capped) reason += ", MTP long-context cap";
    return {batch, ubatch, reason};
}
LlmBatchParameters resolve_llm_batch_parameters(const LlmAutoBatch &automatic, const std::string &batch_env,
    const std::string &ubatch_env, std::optional<int64_t> batch_override, std::optional<int64_t> ubatch_override) {
    const auto batch_raw = normalized_setting(batch_env), ubatch_raw = normalized_setting(ubatch_env);
    const auto batch = batch_override ? std::to_string(*batch_override) : automatic_setting(batch_raw)
        ? std::to_string(automatic.batch) : parse_configuration_integer(batch_raw).value_or(std::to_string(automatic.batch));
    const auto ubatch = ubatch_override ? std::to_string(std::max<int64_t>(1, *ubatch_override)) : automatic_setting(ubatch_raw)
        ? std::to_string(automatic.ubatch) : parse_configuration_integer(ubatch_raw).value_or("256");
    return {batch, minimum(ubatch, batch)};
}
int64_t configured_llm_ubatch(const std::string &raw, int64_t automatic, int64_t context) {
    const auto value = normalized_setting(raw);
    if (!automatic_setting(value)) if (const auto parsed = parse_configuration_integer(value)) {
        const auto positive = configuration_integer_less(*parsed, "1") ? std::string("1") : *parsed;
        return std::stoll(minimum(std::to_string(context), positive));
    }
    return std::min(context, automatic);
}
}
