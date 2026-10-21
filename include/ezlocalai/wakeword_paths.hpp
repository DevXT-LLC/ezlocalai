#pragma once
#include <filesystem>
#include <optional>
#include <string>

namespace ezlocalai {
std::string wakeword_name(const std::string &word);
std::filesystem::path wakeword_contained_path(const std::filesystem::path &root,
                                            const std::filesystem::path &part);
std::optional<std::filesystem::path> wakeword_model_directory(
    const std::filesystem::path &root, const std::string &word);
}
