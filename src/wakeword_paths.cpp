#include "ezlocalai/wakeword_paths.hpp"
#include "ezlocalai/ctts_paths.hpp"
#include "ezlocalai/llm_slots.hpp"
#include <algorithm>
#include <stdexcept>

namespace ezlocalai {
std::string wakeword_name(const std::string &word) {
    auto name = llm_unicode_lower(word);
    std::replace(name.begin(), name.end(), ' ', '_');
    if (name.empty() || name == "." || name == ".." ||
        std::any_of(name.begin(), name.end(), [](unsigned char c) {
            return c == '/' || c == '\\' || c < 32 || c == 127;
        }))
        throw std::invalid_argument("Wake word must be a nonempty name without path separators");
    return name;
}

std::filesystem::path wakeword_contained_path(const std::filesystem::path &root,
                                            const std::filesystem::path &part) {
    const auto parent = ctts_real_path(root.string());
    const auto candidate = ctts_real_path((std::filesystem::path(parent) / part).string());
    const auto prefix = parent == "/" ? parent : parent + "/";
    if (candidate == parent || candidate.rfind(prefix, 0) != 0)
        throw std::invalid_argument("Wake word path must remain inside its storage directory");
    return candidate;
}

std::optional<std::filesystem::path> wakeword_model_directory(
    const std::filesystem::path &root, const std::string &word) {
    const auto directory = wakeword_contained_path(root, wakeword_name(word));
    std::error_code ec;
    if (!std::filesystem::exists(directory, ec)) return {};
    if (!std::filesystem::exists(wakeword_contained_path(directory, "model.pt"), ec)) return {};
    // Check every artifact before a read or directory deletion, including
    // dangling links. A missing optional artifact itself is allowed.
    for (const auto *name : {"config.json", "model.onnx", "model.espdl"})
        wakeword_contained_path(directory, name);
    return directory;
}
}
