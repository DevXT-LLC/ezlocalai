#pragma once
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
extern char **environ;
// Only the synthetic engine reads this file. A persistent child cannot observe
// parent setenv changes; atomically publish fault controls between test requests.
inline void ltx_fixture_controls() {
    nlohmann::json controls = nlohmann::json::object();
    for (char **p = environ; *p; ++p) {
        const std::string entry = *p; const auto at = entry.find('=');
        const auto key = entry.substr(0, at);
        if (key.rfind("WAN_FIXTURE_", 0) == 0 || key.rfind("LTX_FIXTURE_", 0) == 0) controls[key] = entry.substr(at + 1);
    }
    { std::ofstream out(".ltx-fixture-controls.tmp"); out << controls; }
    std::filesystem::rename(".ltx-fixture-controls.tmp", ".ltx-fixture-controls.json");
}
inline int ltx_fixture_setenv(const char *key, const char *value, int overwrite) { const int status = ::setenv(key, value, overwrite); ltx_fixture_controls(); return status; }
inline int ltx_fixture_unsetenv(const char *key) { const int status = ::unsetenv(key); ltx_fixture_controls(); return status; }
