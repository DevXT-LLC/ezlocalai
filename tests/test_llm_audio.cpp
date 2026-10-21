#include "ezlocalai/llm_input.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream input(path); return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()}; }
std::string hex(const std::string &value) { std::ostringstream out; for (unsigned char c : value) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::string hash(const std::string &value) { unsigned char data[32]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), data); return hex({reinterpret_cast<const char *>(data), 32}); }
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::string quote(const std::string &value) { std::string out = "\""; for (unsigned char c : value) {
    if (c == '\\' || c == '"') { out += '\\'; out += c; } else if (c == '\n') out += "\\n"; else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t"; else if (c < 32) out += "\\u00" + hex({static_cast<char>(c)}); else out += c;
} return out + '"'; }
std::vector<std::string> columns(const std::string &value) { std::vector<std::string> out; for (size_t at = 0;;) { const auto end = value.find('\t', at); auto item = value.substr(at, end == std::string::npos ? end : end - at); out.push_back(out.empty() ? item : unhex(item)); if (end == std::string::npos) return out; at = end + 1; } }
std::string array(const std::vector<std::string> &values) { std::string out = "["; for (const auto &value : values) { if (out.size() > 1) out += ','; out += value; } return out + ']'; }
void policy(const fs::path &reference, const fs::path &fixture) {
    std::istringstream input(read(fixture)); size_t pipeline = 0, stt = 0, handoff = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') { const auto at = line.find(" sha256="); if (at != std::string::npos) expect(hash(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "audio source changed"); continue; }
        const auto c = columns(line);
        if (c[0] == "P") {
            expect(c.size() == 9, "pipeline fixture fields"); ++pipeline;
            std::vector<std::string> events;
            ezlocalai::LlmInputOperations ops; ops.local_uri = "http://local";
            ops.load = [&] { events.push_back("[\"load\"]"); if (c[4] == "load") throw ezlocalai::LlmResponseError{"RuntimeError", "load failed"}; return ezlocalai::LlmInputModel{"source", false}; };
            ops.fetch_audio = [&](const std::string &url) { events.push_back("[\"fetch\"," + quote(url) + ']'); if (c[4] == "fetch") throw ezlocalai::LlmResponseError{"RuntimeError", "fetch failed"}; return "remote audio"; };
            ops.transcribe = [&](const std::string &audio, const std::string &format) { events.push_back("[\"transcribe\"," + audio + ',' + format + ']'); if (c[4] == "stt") throw ezlocalai::LlmResponseError{"RuntimeError", "STT failed"}; return c[3]; };
            std::string category = "OK", message, result = "null";
            try { result = ezlocalai::llm_prepare_text_image_input(c[1], c[2] == "true", ops); }
            catch (const ezlocalai::LlmResponseError &error) { category = error.category; message = error.message; }
            const auto label = "audio pipeline " + std::to_string(pipeline) + ' ' + c[1];
            expect(quote(category) == c[5] && quote(message) == c[6], label + " error: " + category + ' ' + message + " expected " + c[5] + ' ' + c[6]);
            expect(result == c[7], label + " result: " + result + " expected " + c[7]);
            expect(array(events) == c[8], label + " order: " + array(events) + " expected " + c[8]);
        } else if (c[0] == "S") {
            expect(c.size() == 7, "STT fixture fields"); ++stt;
            std::string category = "OK", message, extension, bytes;
            try { const auto prepared = ezlocalai::stt_audio_input(c[1], c[2]); extension = prepared.extension; bytes = hex(prepared.bytes); }
            catch (const ezlocalai::LlmResponseError &error) { category = error.category; message = error.message; }
            expect(category == c[3] && message == c[4] && extension == c[5] && bytes == c[6], "STT input " + std::to_string(stt) + ' ' + c[1] + ' ' + c[2] + ": " + category + ' ' + message + ' ' + extension + ' ' + bytes + " expected " + c[3] + ' ' + c[4] + ' ' + c[5] + ' ' + c[6]);
        } else {
            expect(c[0] == "H" && c.size() == 8, "handoff fixture fields"); ++handoff;
            std::vector<std::string> models; std::istringstream names(c[6]); for (std::string name; std::getline(names, name);) models.push_back(name);
            expect(ezlocalai::llm_audio_handoff(c[1], c[2], c[3], c[4], c[5], models) == (c[7] == "true"), "handoff policy " + std::to_string(handoff));
        }
    }
    expect(pipeline == 1582 && stt == 1280 && handoff == 9720, "complete audio reference cases");
    std::cout << pipeline << " pipeline, " << stt << " STT format/base64 and " << handoff << " handoff policy cases passed\n";
}
}
void llm_audio_runtime(const std::vector<std::vector<std::string>> &guards);
int main(int argc, char **argv) { try {
    expect(argc == 4, "reference, pipeline and guard fixture arguments"); policy(argv[1], argv[2]);
    std::vector<std::vector<std::string>> guards; std::istringstream input(read(argv[3]));
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') { const auto at = line.find(" sha256="); if (at != std::string::npos) expect(hash(read(fs::path(argv[1]) / line.substr(2, at - 2))) == line.substr(at + 8), "voice guard source changed"); continue; }
        auto values = columns("G\t" + line); values.erase(values.begin());
        expect(values.size() == 9 && values[5] == R"([["stt",true],["stt",false]])" && values[6] == "false" && values[7] == "false", "source clears handoff/unavailable state and releases shared lock on acquisition error");
        guards.push_back(std::move(values));
    }
    expect(guards.size() == 10, "all source guard cases"); llm_audio_runtime(guards); return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
