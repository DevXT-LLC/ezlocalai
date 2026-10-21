#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace ezlocalai {
// Decimal strings retain Python's integer range until the binding boundary.
std::string llm_optimal_context(const std::string &raw);
std::string llm_configured_context_limit(const std::optional<std::string> &configured, const std::string &optimal);
std::string llm_effective_context(const std::string &requested, const std::string &environment);
int32_t llm_context_parameter(const std::string &effective);
bool llm_inference_reload_allowed(size_t model_count, int64_t capacity);
}
