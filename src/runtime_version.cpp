#include "ezlocalai/runtime_version.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/process.hpp"
#include "runtime_build_version.hpp"
#include <unicode/unistr.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
namespace ezlocalai {
namespace {
namespace fs = std::filesystem;
std::string utf8(const std::string &value) {
    std::string out;
    icu::UnicodeString::fromUTF8(value).toUTF8String(out);
    return out;
}
std::string first_line(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string line; char c;
    while (in.get(c) && c != '\r' && c != '\n') line += c;
    return strip_config_value(utf8(line));
}
std::string metadata_version(const fs::path &repository) {
    try {
        auto git = repository / ".git";
        if (fs::is_regular_file(git)) {
            const auto file = first_line(git);
            if (file.rfind("gitdir:", 0) == 0) {
                git = strip_config_value(file.substr(7));
                if (git.is_relative()) git = (repository / git).lexically_normal();
            }
        }
        if (!fs::is_directory(git)) return {};
        const auto head = first_line(git / "HEAD");
        auto sha = head;
        if (head.rfind("ref:", 0) == 0) {
            const auto ref = strip_config_value(head.substr(4));
            sha = first_line((git / ref).lexically_normal());
            if (sha.empty()) {
                std::ifstream packed(git / "packed-refs");
                std::string line;
                while (std::getline(packed, line)) {
                    line = strip_config_value(utf8(line));
                    if (line.empty() || line[0] == '#' || line[0] == '^') continue;
                    std::istringstream words(line); std::string value, name;
                    if (words >> value >> name && name == ref) { sha = value; break; }
                }
            }
        }
        if (sha.size() < 7 || sha.size() > 40 || sha.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) return {};
        sha.resize(7); for (auto &c : sha) c = std::tolower(static_cast<unsigned char>(c));
        return sha;
    } catch (const std::exception &) { return {}; }
}
std::string git_version(const std::string &repository) {
    std::string pattern = (fs::temp_directory_path() / "ezlocalai-version-XXXXXX").string();
    if (!mkdtemp(pattern.data())) return {};
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{pattern};
    try {
        const auto output = (cleanup.path / "stdout").string();
        const auto result = run_process_to_files({"git", "-C", repository, "rev-parse", "--short", "HEAD"}, output, "/dev/null", 3);
        if (result.status || result.timed_out) return {};
        std::ifstream in(output, std::ios::binary);
        return utf8({std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()});
    } catch (const std::exception &) { return {}; }
}
}
std::string clean_runtime_version(const std::string &raw) {
    const auto value = strip_config_value(raw);
    std::string lower = value;
    for (auto &c : lower) c = std::tolower(static_cast<unsigned char>(c));
    if (lower.empty() || lower == "none" || lower == "null" || lower == "unknown" || lower == "undefined") return {};
    auto unicode = icu::UnicodeString::fromUTF8(value);
    unicode.truncate(unicode.moveIndex32(0, 80));
    std::string out; unicode.toUTF8String(out); return out;
}
std::string runtime_version_at(const std::string &repository, const std::string &package_version, const RuntimeGitVersion &git_probe) {
    for (auto name : {"EZLOCALAI_VERSION", "EZLOCALAI_COMMIT", "GIT_COMMIT"}) {
        const auto raw = std::getenv(name);
        const auto value = clean_runtime_version(raw ? raw : "");
        if (!value.empty()) return value;
    }
    for (auto name : {".ezlocalai-version", "VERSION"}) {
        const auto value = clean_runtime_version(first_line(fs::path(repository) / name));
        if (!value.empty()) return value;
    }
    try {
        const auto value = clean_runtime_version(git_probe ? git_probe(repository) : git_version(repository));
        if (!value.empty()) return value;
    } catch (const std::exception &) {}
    const auto value = metadata_version(repository);
    return value.empty() ? clean_runtime_version(package_version) : value;
}
const std::string &native_build_version() { static const std::string value = EZLOCALAI_BUILD_VERSION; return value; }
const std::string &runtime_version() {
    static const std::string value = runtime_version_at(EZLOCALAI_RUNTIME_ROOT, native_build_version());
    return value;
}
}
