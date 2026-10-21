#include "ezlocalai/llm_parameters.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <algorithm>
#include <limits>
#include <unicode/utf8.h>

namespace ezlocalai {
std::string llm_parameter_integer(const std::string &raw) {
    if (const auto value = parse_configuration_integer(raw)) return *value;
    // Python validates syntax before reporting its decimal conversion limit.
    const auto normalized = normalize_decimal_digits(strip_config_value(raw));
    size_t digits = 0; bool previous_digit = false, valid = !normalized.empty();
    for (size_t i = !normalized.empty() && (normalized[0] == '+' || normalized[0] == '-'); i < normalized.size(); ++i) {
        if (normalized[i] >= '0' && normalized[i] <= '9') { ++digits; previous_digit = true; }
        else if (normalized[i] == '_' && previous_digit) previous_digit = false;
        else { valid = false; break; }
    }
    for (unsigned char c : raw) if (c >= 0x1c && c <= 0x1f) valid = false;
    if (valid && previous_digit && digits > 4300) throw LlmParameterError("ValueError", "Exceeds the limit (4300 digits) for integer string conversion: value has " + std::to_string(digits) + " digits; use sys.set_int_max_str_digits() to increase the limit");
    auto repr = python_string_repr(raw); int32_t offset = 0;
    for (int i = 0; i < 200 && offset < static_cast<int32_t>(repr.size()); ++i) { UChar32 c; U8_NEXT(repr.data(), offset, static_cast<int32_t>(repr.size()), c); }
    repr.resize(offset);
    throw LlmParameterError("ValueError", "invalid literal for int() with base 10: " + repr);
}
void validate_llm_reasoning_budget(const std::string &raw) {
    const auto value = llm_parameter_integer(raw);
    // Match the pinned reference's attempted top-level assignment. Its binding
    // only exposes sampling.reasoning_budget_tokens, not reasoning_budget.
    if (!configuration_integer_less(value, "0"))
        throw LlmParameterError("AttributeError", "'xllamacpp.xllamacpp.CommonParams' object has no attribute 'reasoning_budget'");
}
int32_t llm_parallel_parameter(int64_t requested, int64_t context, bool speculative) {
    if (!requested) requested = std::min<int64_t>(16, std::max<int64_t>(1, context / 32768));
    if (speculative) requested = 1;
    if (requested < std::numeric_limits<int32_t>::min() || requested > std::numeric_limits<int32_t>::max())
        throw LlmParameterError("OverflowError", "value too large to convert to int32_t");
    return static_cast<int32_t>(requested);
}
int llm_image_min_tokens(const std::string &model) { return llm_unicode_lower(model).find("qwen3.8-27b") != std::string::npos ? 1024 : -1; }
LlmTensorParameters llm_tensor_parameters(const std::vector<double> &split, int gpu_count, bool initial_fit) {
    LlmTensorParameters result{false, initial_fit};
    if (split.empty() || gpu_count <= 1 || static_cast<size_t>(gpu_count) > split.size()) return result;
    const auto active = std::count_if(split.begin(), split.begin() + gpu_count, [](double value) { return value > 0; });
    if (active <= 1) { result.split_none = true; result.fit = false; }
    else if (split.size() <= 128) result.fit = false;
    return result;
}
}
