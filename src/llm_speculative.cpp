#include "ezlocalai/llm_speculative.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <unicode/uchar.h>
#include <unicode/utf8.h>

namespace ezlocalai {
namespace {
bool qwen(const std::string &model) { return llm_unicode_lower(model).find("qwen3.8-27b") != std::string::npos; }
bool ascii_alphanumeric(UChar32 c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); }
bool decimal(UChar32 c) { return u_charType(c) == U_DECIMAL_DIGIT_NUMBER; }

}
double llm_model_size_billions(const std::string &model) {
    const auto text = llm_unicode_lower(model);
    std::vector<UChar32> chars;
    for (int32_t offset = 0; offset < static_cast<int32_t>(text.size());) { UChar32 c; U8_NEXT(text.data(), offset, static_cast<int32_t>(text.size()), c); chars.push_back(c); }
    double result = 0;
    for (size_t i = 0; i < chars.size(); ++i) {
        if ((i && ascii_alphanumeric(chars[i-1])) || !decimal(chars[i])) continue;
        size_t end = i; std::string number;
        while (end < chars.size() && decimal(chars[end])) number += static_cast<char>('0' + u_charDigitValue(chars[end++]));
        if (end + 1 < chars.size() && chars[end] == '.' && decimal(chars[end+1])) {
            number += '.'; ++end;
            while (end < chars.size() && decimal(chars[end])) number += static_cast<char>('0' + u_charDigitValue(chars[end++]));
        }
        if (end == chars.size() || chars[end] != 'b' || (end + 1 < chars.size() && ascii_alphanumeric(chars[end+1]))) continue;
        double value = 0; parse_configuration_float(number, &value); result = std::max(result, value);
        // The reference regex consumes its trailing non-alphanumeric separator.
        i = end + (end + 1 < chars.size());
    }
    return result;
}
std::pair<int, double> llm_mtp_draft_length(int64_t main_gpu, const std::string &model,
    const LlmSpeculativeEnvironment &environment, const LlmBatchProbes &probes) {
    std::vector<double> totals;
    if (probes.available()) {
        const auto count = probes.count();
        for (int i = 0; i < count; ++i) totals.push_back(probes.properties(i).total_memory / 1073741824.0);
    }
    const auto profile = gpu_inference_profile(probes, main_gpu);
    auto raw = profile.family.empty() ? std::string() : strip_config_value(environment("MTP_SPEC_DRAFT_N_MAX_" + profile.family, ""));
    if (raw.empty()) raw = environment("MTP_SPEC_DRAFT_N_MAX", "auto");
    if (raw.empty()) raw = "auto";
    raw = strip_config_value(raw);
    const double vram = totals.empty() ? 0 : totals[main_gpu >= 0 && static_cast<uint64_t>(main_gpu) < totals.size() ? main_gpu : 0];
    if (!raw.empty() && llm_unicode_lower(raw) != "auto" && raw != "0") {
        if (const auto value = parse_configuration_integer(raw)) {
            if (configuration_integer_less(*value, "1")) return {1, vram};
            if (configuration_integer_less("16", *value)) return {16, vram};
            return {std::stoi(*value), vram};
        }
    }
    const auto size = llm_model_size_billions(model);
    if (qwen(model)) return {profile.family == "T4" ? 2 : 3, vram};
    return {size > 0 && size <= 4 && vram >= 20 ? 4 : 2, vram};
}
double llm_mtp_draft_probability(const std::string &model, const LlmSpeculativeEnvironment &environment) {
    const double fallback = qwen(model) ? 0.1 : 0.25;
    const auto raw = environment("MTP_SPEC_DRAFT_P_MIN", "auto");
    const auto value = llm_unicode_lower(strip_config_value(raw));
    double probability;
    if (value.empty() || value == "auto" || !parse_configuration_float(raw, &probability)) return fallback;
    if (probability < 0 || probability > 1) return std::min(std::max(probability, 0.0), 1.0);
    return probability;
}
LlmDraftSettings llm_dflash_settings(int64_t main_gpu, const LlmSpeculativeEnvironment &environment, const LlmBatchProbes &probes) {
    const auto profile = gpu_inference_profile(probes, main_gpu);
    auto raw = profile.family.empty() ? std::string() : environment("DFLASH_SPEC_DRAFT_N_MAX_" + profile.family, "");
    if (raw.empty()) raw = environment("DFLASH_SPEC_DRAFT_N_MAX", "auto");
    raw = llm_unicode_lower(strip_config_value(raw));
    const auto length = raw.empty() || raw == "auto" ? std::to_string(profile.family == "T4" ? 2 : profile.family == "3090" ? 3 : 4) : llm_parameter_integer(raw);
    const auto probability = environment("DFLASH_SPEC_DRAFT_P_MIN", "0.0"); double p_min;
    if (!parse_configuration_float(probability, &p_min)) throw LlmSpeculativeError("ValueError", "could not convert string to float: " + python_string_repr(probability));
    if (configuration_integer_less(length, "1") || configuration_integer_less("7", length)) throw LlmSpeculativeError("ValueError", "DFLASH_SPEC_DRAFT_N_MAX must be between 1 and 7");
    if (!std::isfinite(p_min) || p_min < 0 || p_min > 1) throw LlmSpeculativeError("ValueError", "DFLASH_SPEC_DRAFT_P_MIN must be between 0 and 1");
    return {std::stoi(length), p_min, 0};
}
DflashDownloadHooks native_dflash_download_hooks() {
    return {[](const std::string &path) { std::error_code error; return path.find('\0') == std::string::npos && std::filesystem::is_regular_file(path, error); },
        [](const HubAsset &asset) { return download_hub_asset(asset, default_hub_cache_directory()); }};
}
std::string download_dflash_model(const LlmSpeculativeEnvironment &environment, const DflashDownloadHooks &hooks) {
    const auto path = strip_config_value(environment("DFLASH_MODEL_PATH", ""));
    if (!path.empty()) {
        if (!hooks.is_file(path)) throw LlmSpeculativeError("FileNotFoundError", "DFLASH_MODEL_PATH does not exist: " + path);
        return path;
    }
    auto filename = strip_config_value(environment("DFLASH_MODEL_FILE", dflash_filename));
    if (filename.empty()) filename = dflash_filename;
    return hooks.download({"dflash", dflash_repository, filename, dflash_revision});
}
}
