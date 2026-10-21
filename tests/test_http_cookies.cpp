#include "ezlocalai/http_cookies.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::vector<std::string> split(const std::string &value, char delimiter) {
    std::vector<std::string> out; size_t pos = 0;
    for (;;) { const auto end = value.find(delimiter, pos); out.push_back(value.substr(pos, end == std::string::npos ? end : end - pos)); if (end == std::string::npos) return out; pos = end + 1; }
}
std::string unhex(const std::string &value) {
    std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out;
}
}
int main(int argc, char **argv) {
    try {
        expect(argc == 2, "cookie fixture path required"); std::ifstream input(argv[1]); expect(bool(input), "cookie fixture opens");
        size_t count = 0, operations = 0;
        for (std::string line; std::getline(input, line);) {
            if (line.empty() || line[0] == '#') continue;
            ++count; const auto c = split(line, '\t'); double now = 0; ezlocalai::HttpCookieJar jar([&] { return now; }); size_t step = 0;
            for (const auto &encoded : split(unhex(c[1]), '\n')) {
                ++operations; ++step; const auto op = split(encoded, '\t'); now = std::stod(op[1]); bool ok = true; std::string actual;
                std::vector<std::string> headers;
                try {
                    const auto url = ezlocalai::HttpUrl::parse(unhex(op[2]));
                    if (op[0] == "u") { for (const auto &header : split(unhex(op[3]), '\n')) headers.push_back(unhex(header)); jar.update(headers, url); }
                    else actual = jar.header(url);
                } catch (const std::exception &) { ok = false; }
                const auto label = "source cookies " + c[0] + " operation " + std::to_string(step) + " " + op[0] + " " + unhex(op[2]);
                expect(ok == (op[4] == "1"), label + " outcome");
                if (ok && op[0] == "q") expect(actual == unhex(op[5]), label + ": " + actual + " != " + unhex(op[5]));
            }
        }
        expect(count == 1159 && operations == 6180, "complete source cookie fixtures");
        std::cout << "Verified " << count << " source cookie scenarios and " << operations << " operations\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
