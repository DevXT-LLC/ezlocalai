#include "ezlocalai/json_validation.hpp"
#include <algorithm>
#include <stdexcept>

namespace ezlocalai {
namespace {
class JsonSyntax {
public:
    explicit JsonSyntax(const std::string &input) : text_(input) {}
    void parse() { space(); value(0); space(); if (at_ != text_.size()) fail("Extra data"); }
private:
    const std::string &text_; size_t at_ = 0;
    char peek() const { return at_ < text_.size() ? text_[at_] : '\0'; }
    bool digit() const { return peek() >= '0' && peek() <= '9'; }
    void space() { while (at_ < text_.size() && std::string(" \t\r\n").find(text_[at_]) != std::string::npos) ++at_; }
    [[noreturn]] void fail(const char *message) const { fail_at(message, at_); }
    [[noreturn]] void fail_at(const char *message, size_t offset) const {
        size_t position = 0;
        for (size_t i = 0; i < std::min(offset, text_.size()); ++i)
            if ((static_cast<unsigned char>(text_[i]) & 0xc0) != 0x80) ++position;
        throw JsonDecodeError{position, message};
    }
    void string() {
        const auto start = at_++;
        while (at_ < text_.size()) {
            const unsigned char c = text_[at_++];
            if (c == '"') return;
            if (c < 0x20) fail_at("Invalid control character at", at_ - 1);
            if (c != '\\') continue;
            if (at_ == text_.size()) break;
            const auto escape = at_++; const char escaped = text_[escape];
            if (escaped == 'u') {
                for (int i = 0; i < 4; ++i) {
                    const char hex = peek();
                    if (!((hex >= '0' && hex <= '9') || (hex >= 'a' && hex <= 'f') || (hex >= 'A' && hex <= 'F')))
                        fail_at("Invalid \\uXXXX escape", escape);
                    ++at_;
                }
            } else if (std::string("\"\\/bfnrt").find(escaped) == std::string::npos) fail_at("Invalid \\escape", escape - 1);
        }
        fail_at("Unterminated string starting at", start);
    }
    void number() {
        const auto start = at_;
        if (peek() == '-') ++at_;
        if (!digit()) fail_at("Expecting value", start);
        if (peek() == '0') ++at_; else while (digit()) ++at_;
        if (peek() == '.') {
            const auto dot = at_++;
            if (!digit()) { at_ = dot; return; }
            while (digit()) ++at_;
        }
        if (peek() == 'e' || peek() == 'E') {
            const auto exponent = at_++;
            if (peek() == '+' || peek() == '-') ++at_;
            if (!digit()) { at_ = exponent; return; }
            while (digit()) ++at_;
        }
    }
    void value(size_t depth) {
        if (depth > 1000) throw std::runtime_error("JSON nesting exceeds parser limit");
        if (peek() == '"') { string(); return; }
        if (peek() == '{') {
            ++at_; space(); if (peek() == '}') { ++at_; return; }
            while (true) {
                if (peek() != '"') fail("Expecting property name enclosed in double quotes");
                string(); space(); if (peek() != ':') fail("Expecting ':' delimiter");
                ++at_; space(); value(depth + 1); space();
                if (peek() == '}') { ++at_; return; }
                if (peek() != ',') fail("Expecting ',' delimiter");
                ++at_; space();
            }
        }
        if (peek() == '[') {
            ++at_; space(); if (peek() == ']') { ++at_; return; }
            while (true) {
                value(depth + 1); space(); if (peek() == ']') { ++at_; return; }
                if (peek() != ',') fail("Expecting ',' delimiter");
                ++at_; space();
            }
        }
        for (const std::string token : {"true", "false", "null", "NaN", "Infinity", "-Infinity"}) {
            if (text_.compare(at_, token.size(), token) == 0) { at_ += token.size(); return; }
        }
        if (peek() == '-' || digit()) { number(); return; }
        fail("Expecting value");
    }
};
}
std::optional<JsonDecodeError> json_decode_error(const std::string &input) {
    try { JsonSyntax(input).parse(); return std::nullopt; }
    catch (const JsonDecodeError &error) { return error; }
}
}
