#include "ezlocalai/tunnel_websocket.hpp"
#include <cstdint>
#include <limits>
#include <unicode/utf8.h>

namespace ezlocalai {
namespace {
bool valid_utf8(const std::string &text) {
    // Configured tunnel messages fit ICU's signed 32-bit string length.
    if (text.size() > static_cast<size_t>(INT32_MAX)) return false;
    for (int32_t at = 0; at < static_cast<int32_t>(text.size());) {
        UChar32 value;
        U8_NEXT(text.data(), at, static_cast<int32_t>(text.size()), value);
        if (value < 0) return false;
    }
    return true;
}
bool valid_close_code(unsigned code) {
    if (code >= 3000) return true;
    switch (code) {
        case 1000: case 1001: case 1002: case 1003: case 1006:
        case 1007: case 1008: case 1009: case 1010: case 1011:
        case 1012: case 1013: case 1014: return true;
        default: return false;
    }
}
}
bool TunnelWebSocketReader::fail(std::string &wire, int code) {
    error_code_ = code;
    std::string{}.swap(wire);
    std::string{}.swap(partial_);
    return false;
}
bool TunnelWebSocketReader::pop(std::string &wire, TunnelWebSocketMessage &message) {
    if (error_code_) return false;
    for (;;) {
        if (wire.size() < 2) return false;
        const auto first = static_cast<unsigned char>(wire[0]);
        const auto second = static_cast<unsigned char>(wire[1]);
        const bool final = (first & 128) != 0;
        unsigned opcode = first & 15;
        uint64_t length = second & 127;
        size_t at = 2;
        if ((first & 112) || (opcode > 2 && opcode < 8) || opcode > 10)
            return fail(wire, 1002);
        if (opcode >= 8 && (!final || length > 125)) return fail(wire, 1002);
        if (length == 126 || length == 127) {
            const size_t width = length == 126 ? 2 : 8;
            if (wire.size() < at + width) return false;
            length = 0;
            for (size_t i = 0; i < width; ++i) length = (length << 8) | static_cast<unsigned char>(wire[at++]);
        }
        if (length > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) return fail(wire, 1009);
        if (opcode <= 2 && limit_ && (partial_.size() >= limit_ || length >= limit_ - partial_.size()))
            return fail(wire, 1009);
        const bool masked = (second & 128) != 0;
        unsigned char mask[4]{};
        if (masked) {
            if (wire.size() < at + 4) return false;
            for (auto &byte : mask) byte = static_cast<unsigned char>(wire[at++]);
        }
        if (length > wire.size() - at) return false;
        std::string payload;
        if (wire.size() == at + length) {
            payload = std::move(wire);
            wire.clear();
            payload.erase(0, at);
        } else {
            payload.assign(wire.data() + at, static_cast<size_t>(length));
            wire.erase(0, at + static_cast<size_t>(length));
        }
        if (masked) for (size_t i = 0; i < payload.size(); ++i) payload[i] ^= mask[i % 4];
        if (opcode <= 2) {
            if (opcode == 0 && partial_opcode_ < 0) return fail(wire, 1002);
            if (!final) {
                if (opcode != 0) {
                    if (partial_opcode_ >= 0) return fail(wire, 1002);
                    partial_opcode_ = static_cast<int>(opcode);
                }
                partial_ += payload;
                continue;
            }
            const bool has_partial = !partial_.empty();
            if (opcode == 0) { opcode = static_cast<unsigned>(partial_opcode_); partial_opcode_ = -1; }
            else if (has_partial) return fail(wire, 1002);
            if (has_partial) {
                partial_ += payload;
                payload = std::move(partial_);
                partial_.clear();
            }
            if (opcode == 1 && !valid_utf8(payload)) return fail(wire, 1007);
        } else if (opcode == 8) {
            if (payload.size() == 1) return fail(wire, 1002);
            if (payload.size() >= 2) {
                const unsigned code = (static_cast<unsigned char>(payload[0]) << 8) | static_cast<unsigned char>(payload[1]);
                if (!valid_close_code(code)) return fail(wire, 1002);
                if (!valid_utf8(payload.substr(2))) return fail(wire, 1007);
            }
        }
        message.opcode = static_cast<unsigned char>(opcode);
        message.payload = std::move(payload);
        return true;
    }
}
}
