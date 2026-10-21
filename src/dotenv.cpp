// File parsing and interpolation follow python-dotenv 1.2.2.
// See licenses/python-dotenv.txt and tests/fixtures/dotenv_cases.json.
#include "ezlocalai/ezlocalai_c.h"
#include "ezlocalai/configuration.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
std::mutex dotenv_mutex;
struct InvalidUtf8 {};
struct CloseFile { void operator()(FILE *file) const { std::fclose(file); } };
uint32_t codepoint(const std::string &text, size_t &at) {
    const auto first = static_cast<unsigned char>(text.at(at++));
    if (first < 0x80) return first;
    const unsigned count = first >= 0xc2 && first <= 0xdf ? 1 : first >= 0xe0 && first <= 0xef ? 2 : first >= 0xf0 && first <= 0xf4 ? 3 : 0;
    if (!count || at + count > text.size()) throw InvalidUtf8{};
    uint32_t value = first & (0x7f >> count);
    for (unsigned i = 0; i < count; ++i) {
        const auto next = static_cast<unsigned char>(text[at++]);
        if ((next & 0xc0) != 0x80) throw InvalidUtf8{};
        value = (value << 6) | (next & 0x3f);
    }
    if ((count == 1 && value < 0x80) || (count == 2 && value < 0x800)
        || (count == 3 && value < 0x10000) || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) throw InvalidUtf8{};
    return value;
}
bool whitespace(uint32_t value) {
    return (value >= 0x09 && value <= 0x0d) || (value >= 0x1c && value <= 0x20)
        || value == 0x85 || value == 0xa0 || value == 0x1680
        || (value >= 0x2000 && value <= 0x200a) || value == 0x2028 || value == 0x2029
        || value == 0x202f || value == 0x205f || value == 0x3000;
}
void skip_space(const std::string &text, size_t &at, bool multiline = false) {
    while (at < text.size()) {
        auto end = at;
        const auto value = codepoint(text, end);
        if (!whitespace(value) || (!multiline && value == '\n')) break;
        at = end;
    }
}
bool disabled() {
    const char *raw = std::getenv("PYTHON_DOTENV_DISABLED");
    if (!raw) return false;
    std::string value(raw);
    for (auto &ch : value) if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
    // U+017F casefolds to 's'; no other non-ASCII casefold affects these words.
    for (size_t at = 0; (at = value.find("\xc5\xbf", at)) != std::string::npos;) value.replace(at, 2, "s");
    return value == "1" || value == "true" || value == "t" || value == "yes" || value == "y";
}
std::optional<std::string> quoted(const std::string &text, size_t &at, bool key = false) {
    const auto quote = text[at];
    const auto begin = at + 1;
    size_t end = begin;
    for (; end < text.size(); ++end) {
        if (!key && text[end] == '\\' && end + 1 < text.size() && text[end + 1] == quote) { ++end; continue; }
        if (text[end] == quote) break;
    }
    if (end == text.size() || (key && end == begin)) return std::nullopt;
    at = end + 1;
    auto value = text.substr(begin, end - begin);
    if (key) return value;
    std::string decoded;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 1 < value.size()) {
            const auto next = value[i + 1];
            if (next == '\\' || next == '\'' || (quote == '"' && next == '"')) { decoded += next; ++i; continue; }
            if (quote == '"') {
                const std::string escapes = "abfnrtv";
                const auto pos = escapes.find(next);
                if (pos != std::string::npos) { decoded += "\a\b\f\n\r\t\v"[pos]; ++i; continue; }
            }
        }
        decoded += value[i];
    }
    return decoded;
}
using Values = std::unordered_map<std::string, std::optional<std::string>>;
std::string interpolate(const std::string &text, const Values &previous) {
    std::string out;
    size_t copied = 0, search = 0;
    while ((search = text.find("${", search)) != std::string::npos) {
        const auto name_end = text.find_first_of(":}", search + 2);
        if (name_end == std::string::npos) break;
        auto close = name_end;
        std::string fallback;
        if (text[name_end] == ':') {
            if (name_end + 1 >= text.size() || text[name_end + 1] != '-') { search += 2; continue; }
            close = text.find('}', name_end + 2);
            if (close == std::string::npos) { search += 2; continue; }
            fallback = text.substr(name_end + 2, close - name_end - 2);
        }
        const auto name = text.substr(search + 2, name_end - search - 2);
        out.append(text, copied, search - copied);
        const auto it = previous.find(name);
        const char *process = name.find('\0') == std::string::npos ? std::getenv(name.c_str()) : nullptr;
        if (process) out += process;
        else if (it != previous.end()) out += it->second.value_or("");
        else out += fallback;
        copied = search = close + 1;
    }
    out.append(text, copied, std::string::npos);
    return out;
}
int parse_and_apply(const std::string &text) {
    Values values;
    std::vector<std::string> order;
    size_t at = 0;
    while (at < text.size()) {
        skip_space(text, at, true);
        if (at == text.size()) break;
        if (text.compare(at, 6, "export") == 0) {
            auto end = at + 6;
            skip_space(text, end);
            if (end != at + 6) at = end;
        }
        bool valid = true;
        std::optional<std::string> key, value;
        if (at < text.size() && text[at] == '\'') { key = quoted(text, at, true); valid = key.has_value(); }
        else if (at < text.size() && text[at] != '#') {
            const auto begin = at;
            while (at < text.size() && text[at] != '=' && text[at] != '#') {
                auto end = at;
                if (whitespace(codepoint(text, end))) break;
                at = end;
            }
            if (at == begin) valid = false;
            else key = text.substr(begin, at - begin);
        }
        if (valid) {
            skip_space(text, at);
            if (at < text.size() && text[at] == '=') {
                ++at;
                skip_space(text, at);
                if (at < text.size() && (text[at] == '\'' || text[at] == '"')) {
                    value = quoted(text, at);
                    valid = value.has_value();
                } else {
                    const auto end = text.find('\n', at);
                    auto raw = text.substr(at, end == std::string::npos ? end : end - at);
                    at = end == std::string::npos ? text.size() : end;
                    for (size_t i = 0; i < raw.size();) {
                        auto next = i;
                        skip_space(raw, next, true);
                        if (next != i && next < raw.size() && raw[next] == '#') { raw.resize(i); break; }
                        if (next == raw.size()) break;
                        i = next;
                        codepoint(raw, i);
                    }
                    value = ezlocalai::strip_config_value(raw);
                }
            }
        }
        if (valid) {
            skip_space(text, at);
            if (at < text.size() && text[at] == '#') {
                const auto end = text.find('\n', at);
                at = end == std::string::npos ? text.size() : end;
            }
            valid = at == text.size() || text[at] == '\n';
        }
        if (!valid) {
            // Failed statements are ignored; resume after the current physical
            // line, including failures after a multiline quoted value.
            const auto end = text.find('\n', at);
            at = end == std::string::npos ? text.size() : end + 1;
            continue;
        }
        if (at < text.size()) ++at;
        if (key) {
            if (values.find(*key) == values.end()) order.push_back(*key);
            values[*key] = value ? std::optional<std::string>(interpolate(*value, values)) : std::nullopt;
        }
    }
    // Resolve the complete file before mutating the process environment. This
    // preserves original-process precedence and last-assignment semantics.
    for (const auto &key : order) {
        const auto &value = values.at(key);
        if (key.find('\0') == std::string::npos && std::getenv(key.c_str())) continue;
        if (!value) continue;
        if (key.find('\0') != std::string::npos || value->find('\0') != std::string::npos) return EINVAL;
        if (setenv(key.c_str(), value->c_str(), 0) != 0) return errno;
    }
    return 0;
}
}

int ezl_load_dotenv_file(const char *path) {
    try {
        std::lock_guard<std::mutex> lock(dotenv_mutex);
        if (disabled()) return 0;
        std::unique_ptr<FILE, CloseFile> file(std::fopen(path && *path ? path : ".env", "rb"));
        if (!file) return errno == ENOENT ? 0 : errno;
        std::string raw;
        char buffer[8192];
        while (const auto count = std::fread(buffer, 1, sizeof(buffer), file.get())) raw.append(buffer, count);
        if (std::ferror(file.get())) return errno ? errno : EIO;
        if (std::fclose(file.release()) != 0) return errno;
        // Python opens dotenv files as UTF-8 text with universal newlines.
        for (size_t at = 0; at < raw.size();) codepoint(raw, at);
        std::string text;
        text.reserve(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '\r') { if (i + 1 < raw.size() && raw[i + 1] == '\n') ++i; text += '\n'; }
            else text += raw[i];
        }
        return parse_and_apply(text);
    } catch (const InvalidUtf8 &) { return EILSEQ; }
    catch (const std::bad_alloc &) { return ENOMEM; }
    catch (const std::length_error &) { return ENOMEM; }
}

int ezl_load_dotenv(void) {
    const char *path = std::getenv("EZLOCALAI_DOTENV_PATH");
    if (path && !*path) return 0;
    return ezl_load_dotenv_file(path && *path ? path : ".env");
}
