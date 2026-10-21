#include "ezlocalai/llm_context.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_parameters.hpp"

namespace ezlocalai {
std::string llm_optimal_context(const std::string &raw) {
    std::optional<std::string> maximum;
    try {
        size_t start = 0;
        for (;;) {
            const auto comma = raw.find(',', start);
            const auto value = strip_config_value(raw.substr(start, comma == std::string::npos ? comma : comma - start));
            if (!value.empty()) {
                const auto number = llm_parameter_integer(value);
                if (!maximum || configuration_integer_less(*maximum, number)) maximum = number;
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    } catch (const LlmParameterError &) { return "40000"; }
    return maximum.value_or("40000");
}
std::string llm_configured_context_limit(const std::optional<std::string> &configured, const std::string &optimal) {
    std::string value;
    try { value = llm_parameter_integer(configured.value_or(optimal).empty() ? "0" : configured.value_or(optimal)); }
    catch (const LlmParameterError &) { value = llm_parameter_integer(optimal.empty() ? "0" : optimal); }
    if (value == "0") value = optimal.empty() || optimal == "0" ? "16384" : llm_parameter_integer(optimal);
    return configuration_integer_less(value, "1") ? "1" : value;
}
std::string llm_effective_context(const std::string &requested, const std::string &environment) {
    return configuration_integer_less("0", requested) ? requested : llm_parameter_integer(environment);
}
int32_t llm_context_parameter(const std::string &effective) {
    if (configuration_integer_less(effective, "-9223372036854775808") || configuration_integer_less("9223372036854775807", effective))
        throw LlmParameterError("OverflowError", "Python int too large to convert to C long");
    if (configuration_integer_less(effective, "-2147483648") || configuration_integer_less("2147483647", effective))
        throw LlmParameterError("OverflowError", "value too large to convert to int32_t");
    return static_cast<int32_t>(std::stoll(effective));
}
bool llm_inference_reload_allowed(size_t model_count, int64_t capacity) { return model_count <= 1 && capacity <= 1; }
}
