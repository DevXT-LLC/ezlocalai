#pragma once
#include <optional>
#include <string>

namespace ezlocalai {
inline constexpr char llm_startup_request[] = "{\"prompt\":\"Hi\",\"max_tokens\":1}";
std::string python_string_repr(const std::string &value);
// The result is a JSON representation of the native completion result. An
// exception takes precedence and uses the reference constructor's wrapper.
std::optional<std::string> llm_startup_failure(const std::string &model, const std::string &result,
    const std::optional<std::string> &exception = std::nullopt);
}
