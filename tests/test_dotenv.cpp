#include "ezlocalai/ezlocalai_c.h"
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unistd.h>

extern char **environ;
using Json = nlohmann::json;
int main(int argc, char **argv) {
    std::filesystem::path directory;
    try {
        if (argc != 2) throw std::runtime_error("dotenv fixture required");
        std::ifstream fixture(argv[1]);
        const auto cases = Json::parse(fixture);
        std::string pattern = "/tmp/ezlocalai-dotenv-XXXXXX";
        const char *created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("temporary directory");
        directory = created;
        const auto path = directory / ".env";
        size_t count = 0;
        for (const auto &test : cases.at("cases")) {
            clearenv();
            for (const auto &[key, value] : test.at("environment").items())
                setenv(key.c_str(), value.get<std::string>().c_str(), 1);
            const auto hex = test.at("input_hex").get<std::string>();
            std::string bytes;
            for (size_t i = 0; i < hex.size(); i += 2) bytes += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
            { std::ofstream file(path, std::ios::binary); file.write(bytes.data(), bytes.size()); }
            const int rc = ezl_load_dotenv_file(path.c_str());
            int expected_error = 0;
            if (test.at("error") == "UnicodeDecodeError") expected_error = EILSEQ;
            else if (!test.at("error").is_null()) expected_error = EINVAL;
            Json observed = Json::object();
            for (char **entry = environ; entry && *entry; ++entry) {
                std::string item(*entry);
                const auto equals = item.find('=');
                observed[item.substr(0, equals)] = item.substr(equals + 1);
            }
            if (rc != expected_error || observed != test.at("expected"))
                throw std::runtime_error("dotenv mismatch: " + test.at("name").get<std::string>() + " rc=" + std::to_string(rc));
            ++count;
        }
        clearenv();
        if (ezl_load_dotenv_file((directory / "missing").c_str()) != 0) throw std::runtime_error("missing file is optional");
        setenv("EZLOCALAI_DOTENV_PATH", "", 1);
        if (ezl_load_dotenv() != 0) throw std::runtime_error("explicit disabled dotenv path");
        setenv("EZLOCALAI_DOTENV_PATH", path.c_str(), 1);
        if (ezl_load_dotenv() != 0) throw std::runtime_error("explicit dotenv path");
        std::filesystem::remove_all(directory);
        std::cout << "Verified " << count << " python-dotenv " << cases.at("version") << " file-loading cases and native path controls\n";
        return 0;
    } catch (const std::exception &error) {
        if (!directory.empty()) { std::error_code ignored; std::filesystem::remove_all(directory, ignored); }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
