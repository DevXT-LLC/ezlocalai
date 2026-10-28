#pragma once
#include <algorithm>
#include <string>

namespace ezlocalai {
// Match the family, not arbitrary names containing "omni". Repository owners,
// GGUF quantization suffixes and local paths do not change model capabilities.
inline bool lfm3_family(const std::string &model, const std::string &family) {
    auto name = model.substr(model.find_last_of("/\\") == std::string::npos ? 0 : model.find_last_of("/\\") + 1);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return c >= 'A' && c <= 'Z' ? char(c + 32) : char(c); });
    const auto prefix = "lfm3-" + family + "-2b";
    return name.compare(0, prefix.size(), prefix) == 0 &&
        (name.size() == prefix.size() || std::string("-.:#").find(name[prefix.size()]) != std::string::npos);
}
inline bool lfm3_omni_model(const std::string &model) { return lfm3_family(model, "omni"); }
inline bool realtime_audio_model(const std::string &model) { return lfm3_omni_model(model); }
inline constexpr int lfm3_context_limit = 128000;
}
