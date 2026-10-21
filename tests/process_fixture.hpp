#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

struct ProcessFixture {
    std::string arguments_path;
    bool append_launch = false;
    int delay_ms = 0;
    int exit_status = 0;
    bool fail_first_launch = false;
    std::string output;
    std::string whisper_json;
};

inline void write_process_fixture(const std::filesystem::path &path, const ProcessFixture &fixture) {
    std::filesystem::copy_file(TEST_NATIVE_FIXTURE_BIN, path, std::filesystem::copy_options::overwrite_existing);
    std::ofstream out(path.string() + ".fixture");
    out << std::quoted(fixture.arguments_path) << '\n' << fixture.append_launch << '\n'
        << fixture.delay_ms << '\n' << fixture.exit_status << '\n' << fixture.fail_first_launch << '\n'
        << std::quoted(fixture.output) << '\n' << std::quoted(fixture.whisper_json) << '\n';
    if (!out) throw std::runtime_error("cannot write native process fixture");
}

inline void write_llama_fixture(const std::filesystem::path &path, const std::filesystem::path &arguments = {},
                               bool append = false, int status = 0, bool fail_first = false) {
    write_process_fixture(path, {arguments.string(), append, status ? 0 : 30000, status, fail_first, {}, {}});
}

inline void write_output_fixture(const std::filesystem::path &path, const std::string &output) {
    write_process_fixture(path, {{}, false, 0, 0, false, output, {}});
}

inline void write_whisper_fixture(const std::filesystem::path &path, const std::string &json,
                                 const std::filesystem::path &arguments = {}, int delay_ms = 0,
                                 const std::string &output = {}) {
    write_process_fixture(path, {arguments.string(), false, delay_ms, 0, false, output, json});
}
