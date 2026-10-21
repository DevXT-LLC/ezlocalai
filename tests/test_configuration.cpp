#include "ezlocalai/configuration.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/ezlocalai_c.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <openssl/sha.h>

namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    expect(bool(in), "could not read " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::vector<std::string> split(const std::string &text, char separator) {
    std::vector<std::string> result;
    size_t begin = 0;
    do {
        const auto end = text.find(separator, begin);
        result.push_back(text.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    } while (true);
    return result;
}
std::string unhex(const std::string &value) {
    expect(value.size() % 2 == 0, "invalid hex fixture");
    std::string result;
    for (size_t i = 0; i < value.size(); i += 2) result += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16));
    return result;
}
std::string hex(const std::string &value) {
    std::ostringstream out;
    for (const unsigned char ch : value) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(ch);
    return out.str();
}
std::string sha256(const std::string &value) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    return hex(std::string(reinterpret_cast<const char *>(digest), sizeof(digest)));
}
std::string join(const std::vector<std::string> &items, char separator) {
    std::string result;
    for (const auto &item : items) { if (!result.empty()) result += separator; result += item; }
    return result;
}
void check_defaults(const fs::path &reference) {
    const auto source = read_file(reference / "Globals.py");
    const std::regex entry(R"re(^\s*"([A-Z0-9_]+)":\s*("[^"]*"|[0-9]+),)re");
    std::istringstream lines(source);
    std::map<std::string, std::string> defaults;
    for (std::string line; std::getline(lines, line);) {
        std::smatch match;
        if (!std::regex_search(line, match, entry)) continue;
        std::string expected = match[2];
        if (expected.front() == '"') expected = expected.substr(1, expected.size() - 2);
        defaults[match[1]] = expected;
    }
    expect(defaults.size() == std::size(ezlocalai::environment_defaults), "reference default count changed; audit every key");
    for (const auto &entry : defaults) {
        const auto *name = entry.first.c_str();
        unsetenv(name);
        expect(ezl_getenv_default(name) == entry.second, "reference default: " + entry.first);
        expect(std::string(ezl_getenv_fallback(name, "explicit default")) == "explicit default", "explicit fallback: " + entry.first);
        expect(ezl_getenv_fallback(name, "") == entry.second, "falsey fallback: " + entry.first);
        setenv(name, "", 1);
        expect(std::string(ezl_getenv_fallback(name, "explicit default")).empty(), "explicit empty env: " + entry.first);
        setenv(name, "environment override", 1);
        expect(std::string(ezl_getenv_fallback(name, "explicit default")) == "environment override", "env precedence: " + entry.first);
        unsetenv(name);
    }
    unsetenv("UNKNOWN_REFERENCE_SETTING");
    expect(std::string(ezl_getenv_default("UNKNOWN_REFERENCE_SETTING")).empty(), "unknown default is empty");
    expect(std::string(ezl_getenv_fallback("UNKNOWN_REFERENCE_SETTING", "fallback")) == "fallback", "unknown explicit fallback");
    std::cout << "Verified " << defaults.size() << " current reference defaults and precedence rules\n";
}
void check_gpu_tiers(const fs::path &reference) {
    const auto source = read_file(reference / "Router.py");
    const auto begin = source.find("GPU_TIERS:");
    const auto end = source.find("\n]", begin);
    expect(begin != std::string::npos && end != std::string::npos, "reference GPU tier table exists");
    const auto table = source.substr(begin, end - begin);
    const std::regex entry(R"re(\("([^"]+)",\s*([0-9]+)\))re");
    size_t count = 0;
    for (std::sregex_iterator it(table.begin(), table.end(), entry), finish; it != finish; ++it) {
        const auto name = (*it)[1].str();
        const auto tier = std::stoi((*it)[2]);
        expect(ezl_gpu_tier_for_name(name.c_str()) == tier, "reference GPU tier: " + name);
        ++count;
    }
    expect(count > 80, "full reference GPU tier table covered");
    std::cout << "Verified " << count << " reference GPU tiers\n";
}
} // namespace

int main(int argc, char **argv) {
    fs::path temp;
    const auto original_cwd = fs::current_path();
    try {
        expect(argc == 3, "reference checkout and configuration fixtures required");
        const auto reference = fs::absolute(argv[1]);
        const auto fixture_text = read_file(fs::absolute(argv[2]));
        setenv("EZLOCALAI_DOTENV_PATH", "", 1);
        check_defaults(reference);
        check_gpu_tiers(reference);
        std::string pattern = (fs::temp_directory_path() / "ezlocalai-config-XXXXXX").string();
        const auto created = mkdtemp(pattern.data());
        expect(created, "temporary directory creation");
        temp = created;
        fs::current_path(temp);
        size_t count = 0;
        std::istringstream lines(fixture_text);
        for (std::string line; std::getline(lines, line);) {
            if (line.empty()) continue;
            if (line.front() == '#') {
                const auto marker = line.find(" sha256=");
                if (marker != std::string::npos) {
                    const auto file = line.substr(2, marker - 2);
                    expect(sha256(read_file(reference / file)) == line.substr(marker + 8), "reference changed; refresh fixture oracle: " + file);
                }
                continue;
            }
            const auto columns = split(line, '\t');
            expect(columns.size() == 5, "malformed fixture");
            for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
            for (const auto &entry : split(columns[1], ';')) {
                if (entry.empty()) continue;
                const auto equals = entry.find('=');
                expect(equals != std::string::npos, "invalid environment fixture");
                setenv(entry.substr(0, equals).c_str(), unhex(entry.substr(equals + 1)).c_str(), 1);
            }
            fs::remove_all("models");
            for (const auto &name : split(columns[2], ';')) {
                if (name.empty()) continue;
                const auto path = fs::path("models") / name;
                fs::create_directories(path.parent_path());
                std::ofstream(path).put('x');
            }
            const auto caps = ezlocalai::local_capabilities();
            expect(join(caps, ',') == columns[3], columns[0] + " capabilities: got " + join(caps, ',') + " expected " + columns[3]);
            ezl_string_list abi_caps{};
            expect(ezl_detect_local_capabilities(&abi_caps) == 0, columns[0] + " capability ABI");
            std::vector<std::string> abi_values;
            for (size_t i = 0; i < abi_caps.count; ++i) abi_values.emplace_back(abi_caps.items[i]);
            ezl_string_list_free(&abi_caps);
            expect(abi_values == caps, columns[0] + " C ABI agrees with worker configuration");
            std::vector<std::string> models;
            for (const auto &entry : ezlocalai::capability_model_names(caps)) models.push_back(entry.first + "=" + hex(entry.second));
            expect(join(models, ';') == columns[4], columns[0] + " capability model names");
            ++count;
        }
        expect(count == 536, "all reference cases executed");
        fs::current_path(original_cwd);
        fs::remove_all(temp);
        std::cout << "Verified " << count << " reference capability and model-name cases\n";
        return 0;
    } catch (const std::exception &error) {
        fs::current_path(original_cwd);
        if (!temp.empty()) { std::error_code ignored; fs::remove_all(temp, ignored); }
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
