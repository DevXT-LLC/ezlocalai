#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <thread>

namespace {
void be32(std::string &out, uint32_t value) { for (int shift = 24; shift >= 0; shift -= 8) out += static_cast<char>(value >> shift); }
void chunk(std::string &out, const std::string &type, const std::string &data) {
    be32(out, data.size()); out += type + data;
    uint32_t crc = 0xffffffff;
    for (unsigned char c : type + data) { crc ^= c; for (int i = 0; i < 8; ++i) crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320 : 0); }
    be32(out, crc ^ 0xffffffff);
}
std::string png(int width, int height, int frame) {
    std::string raw;
    for (int y = 0; y < height; ++y) {
        raw += '\0';
        for (int x = 0; x < width; ++x) { raw += static_cast<char>(x + frame); raw += static_cast<char>(y); raw += static_cast<char>(frame * 20); }
    }
    std::string zlib("\x78\x01", 2);
    for (size_t i = 0; i < raw.size();) {
        const auto size = static_cast<uint16_t>(std::min<size_t>(65535, raw.size() - i));
        zlib += static_cast<char>(i + size == raw.size() ? 1 : 0);
        zlib += static_cast<char>(size); zlib += static_cast<char>(size >> 8);
        zlib += static_cast<char>(~size); zlib += static_cast<char>((~size) >> 8);
        zlib.append(raw, i, size); i += size;
    }
    uint32_t a = 1, b = 0;
    for (unsigned char c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    be32(zlib, (b << 16) | a);
    std::string result("\x89PNG\r\n\x1a\n", 8), header;
    be32(header, width); be32(header, height); header.append("\x08\x02\0\0\0", 5);
    chunk(result, "IHDR", header); chunk(result, "IDAT", zlib); chunk(result, "IEND", "");
    return result;
}
}
int main(int argc, char **argv) {
    if (const char *gate = std::getenv("WAN_FIXTURE_GATE")) {
        std::ofstream(std::string(gate) + "-entered") << "entered";
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (!std::filesystem::exists(std::string(gate) + "-release")) {
            if (std::chrono::steady_clock::now() >= deadline) return 81;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    if (const char *path = std::getenv("WAN_FIXTURE_ARGUMENTS")) { std::ofstream out(path, std::ios::app); out << "launch\n"; for (int i = 1; i < argc; ++i) out << argv[i] << '\n'; }
    if (const char *delay = std::getenv("WAN_FIXTURE_DELAY")) std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(delay)));
    if (const char *exit = std::getenv("WAN_FIXTURE_EXIT")) return std::stoi(exit);
    std::map<std::string, std::string> arguments;
    for (int i = 1; i + 1 < argc; ++i) arguments[argv[i]] = argv[i + 1];
    if (const char *path = std::getenv("WAN_FIXTURE_INPUT_CAPTURE"); path && arguments.count("-i")) {
        std::ifstream input(arguments["-i"], std::ios::binary);
        std::ofstream output(path, std::ios::binary); output << input.rdbuf();
        if (!input || !output) return 80;
    }
    const int width = std::stoi(arguments["-W"]), height = std::stoi(arguments["-H"]);
    int frames = std::stoi(arguments["--video-frames"]);
    if (std::getenv("WAN_FIXTURE_MISSING_FRAME")) --frames;
    for (int i = 0; i < frames; ++i) {
        auto path = arguments["-o"];
        std::ostringstream number; number << std::setw(6) << std::setfill('0') << i;
        path.replace(path.find("%06d"), 4, number.str());
        std::ofstream out(path, std::ios::binary); out << png(width, height, i);
    }
    return 0;
}
