#include "ezlocalai/tunnel_websocket.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <sstream>

using Json = nlohmann::json;
std::string unhex(const std::string &text) {
    std::string result;
    for (size_t i = 0; i < text.size(); i += 2) result += static_cast<char>(std::stoul(text.substr(i, 2), nullptr, 16));
    return result;
}
std::string digest(const std::string &text) {
    unsigned char bytes[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(text.data()), text.size(), bytes);
    std::ostringstream out; for (const auto byte : bytes) out << std::hex << std::setw(2) << std::setfill('0') << int(byte); return out.str();
}
std::string frame(const Json &spec) {
    std::string payload;
    if (spec.contains("hex")) payload = unhex(spec.at("hex"));
    else { const auto byte = unhex(spec.value("byte", "61")); for (size_t i = 0; i < spec.value("count", size_t{}); ++i) payload += byte; }
    const auto length = spec.value("declared", static_cast<uint64_t>(payload.size()));
    const auto code = spec.value("length_code", length < 126 ? length : length < 65536 ? 126 : 127);
    const auto mask = unhex(spec.value("mask", ""));
    std::string wire;
    wire += static_cast<char>((spec.value("fin", true) ? 128 : 0) | spec.value("rsv", 0) | spec.value("opcode", 1));
    wire += static_cast<char>(code | (mask.empty() ? 0 : 128));
    const int width = code == 126 ? 2 : code == 127 ? 8 : 0;
    for (int i = width - 1; i >= 0; --i) wire += static_cast<char>(length >> (8 * i));
    if (!mask.empty()) for (size_t i = 0; i < payload.size(); ++i) payload[i] ^= mask[i % 4];
    return wire + mask + payload;
}
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("reference fixture required");
        std::ifstream file(argv[1]); const auto fixture = Json::parse(file);
        for (const auto &test : fixture.at("cases")) {
            std::string input;
            for (const auto &spec : test.at("frames")) input += frame(spec);
            ezlocalai::TunnelWebSocketReader reader(test.value("limit", size_t{64 * 1024 * 1024}));
            std::string wire;
            Json events = Json::array();
            const auto chunk = test.value("chunk", size_t{4096});
            for (size_t at = 0; at < input.size(); at += chunk) {
                wire.append(input, at, std::min(chunk, input.size() - at));
                ezlocalai::TunnelWebSocketMessage message;
                while (reader.pop(wire, message)) events.push_back({{"opcode", message.opcode}, {"size", message.payload.size()}, {"sha256", digest(message.payload)}});
                if (reader.error_code()) break;
            }
            if (events != test.at("events") || reader.error_code() != test.at("error_code"))
                throw std::runtime_error(test.at("name").get<std::string>() + ": code " + std::to_string(reader.error_code()) + " events " + events.dump());
            if (reader.error_code()) {
                wire = std::string("\x81\0", 2);
                ezlocalai::TunnelWebSocketMessage message;
                if (reader.pop(wire, message)) throw std::runtime_error("failed connection resumed parsing");
            }
        }
        std::cout << fixture.at("cases").size() << " compiled aiohttp frame/message comparisons passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
