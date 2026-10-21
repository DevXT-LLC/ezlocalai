#pragma once
#include <cstddef>
#include <optional>
#include <string>

namespace ezlocalai {
struct JsonDecodeError { size_t position; std::string message; };
// JSON syntax diagnostics use Python's character offsets, including UTF-8 text.
std::optional<JsonDecodeError> json_decode_error(const std::string &input);
}
