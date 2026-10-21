#pragma once
#include <functional>
#include <string>
namespace ezlocalai {
using RuntimeGitVersion = std::function<std::string(const std::string &)>;
std::string clean_runtime_version(const std::string &value);
std::string runtime_version_at(const std::string &repository,
    const std::string &package_version, const RuntimeGitVersion &git_probe = {});
// Cached once per process, matching the worker heartbeat's Python provider.
const std::string &runtime_version();
const std::string &native_build_version();
}
