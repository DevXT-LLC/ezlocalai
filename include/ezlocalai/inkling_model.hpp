#pragma once
#include <algorithm>
#include <string>
#include <string_view>

namespace ezlocalai {
inline constexpr char inkling2_mini_repository[] = "unsloth/Inkling-2-Mini-GGUF";
inline constexpr char inkling2_mini_quant[] = "UD-Q4_K_M";
inline constexpr int inkling2_mini_context = 32768;

inline bool inkling2_mini_model(std::string_view model) {
    const auto slash = model.find_last_of("/\\");
    auto name = std::string(model.substr(slash == std::string_view::npos ? 0 : slash + 1));
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return c >= 'A' && c <= 'Z' ? char(c + ('a' - 'A')) : char(c);
    });
    return name == "inkling-2-mini" || name == "inkling-2-mini-gguf";
}
}
