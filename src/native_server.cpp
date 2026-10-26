#include "ezlocalai/worker_models.hpp"
#include "ezlocalai/runtime_version.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/wakeword_paths.hpp"
#include <nlohmann/json.hpp>
#include "ezlocalai/ctts_paths.hpp"
#include "ezlocalai/gtts.hpp"
#include "ezlocalai/tts_websocket.hpp"
#include "ezlocalai/tts_pipeline.hpp"
#include <poll.h>
#include "ezlocalai/llm_startup.hpp"
#include "ezlocalai/llm_load.hpp"
#include "ezlocalai/llm_context.hpp"
#include "ezlocalai/llm_request.hpp"
#include "ezlocalai/llm_input.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "ezlocalai/subtitles.hpp"
#include "ezlocalai/llm_stream.hpp"
#include "ezlocalai/llm_batch.hpp"
#include "ezlocalai/llm_cache.hpp"
#include "ezlocalai/llm_residency.hpp"
#include "ezlocalai/llm_memory.hpp"
#include "ezlocalai/llm_speculative.hpp"
#include "ezlocalai/engine_revision.hpp"
#include <unicode/uchar.h>
#include <unicode/utf8.h>
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/embedding_placement.hpp"
#include "ezlocalai/model_download.hpp"
#include "ezlocalai/hub_assets.hpp"
#include "ezlocalai/music_runtime.hpp"
#include "ezlocalai/music_video.hpp"
#include "ezlocalai/storyboard.hpp"
#include "ezlocalai/wan_video.hpp"
#include "ezlocalai/ltx_runtime.hpp"
#include "ezlocalai/video_runtime.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/json_validation.hpp"
#include "ezlocalai/http_deadline.hpp"
#include "ezlocalai/http_body.hpp"
#include "ezlocalai/http_url.hpp"
#include "ezlocalai/http_cookies.hpp"
#include "ezlocalai/http_socket.hpp"
#include "ezlocalai/process.hpp"
#include "generated_openapi.hpp"
#include "ezlocalai/worker_heartbeat.hpp"
#include "ezlocalai/tunnel_websocket.hpp"
#include "ezlocalai/tunnel_connection.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <ctime>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iconv.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>

#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509_vfy.h>

namespace ezlocalai {

std::vector<std::pair<std::string, std::string>> json_object_ordered_entries(const std::string &raw);
std::string python_float_string(double value);
std::string json_number_literal(double value);

namespace {

std::string now_id() {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string uuid_v4() {
    std::array<unsigned char, 16> bytes{};
    std::random_device rd;
    for (auto &byte : bytes) {
        byte = static_cast<unsigned char>(rd());
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3fU) | 0x80U);

    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setfill('0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            out << "-";
        }
        out << std::setw(2) << static_cast<int>(bytes[i]);
    }
    return out.str();
}

long unix_seconds() {
    return static_cast<long>(std::time(nullptr));
}

std::string trim(const std::string &value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool starts_with(const std::string &value, const std::string &prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool native_router_mode_enabled() {
    return to_lower(trim(ezl_getenv_default("ROUTER_MODE"))) == "true";
}

int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return 10 + ch - 'a';
    }
    if (ch >= 'A' && ch <= 'F') {
        return 10 + ch - 'A';
    }
    return -1;
}

std::string url_decode(const std::string &value, bool plus_as_space = false) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const char ch = value[i];
        if (ch == '%' && i + 2 < value.size()) {
            const int high = hex_value(value[i + 1]);
            const int low = hex_value(value[i + 2]);
            if (high >= 0 && low >= 0) {
                out.push_back(static_cast<char>((high << 4) | low));
                i += 2;
                continue;
            }
        }
        if (plus_as_space && ch == '+') {
            out.push_back(' ');
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

bool matches_template(const std::string &path, const std::string &route_template) {
    const auto open = route_template.find('{');
    if (open == std::string::npos) {
        return path == route_template;
    }
    const auto close = route_template.find('}', open);
    if (close == std::string::npos) {
        return false;
    }
    const auto prefix = route_template.substr(0, open);
    const auto suffix = route_template.substr(close + 1);
    if (!starts_with(path, prefix)) {
        return false;
    }
    if (!suffix.empty()) {
        if (path.size() < prefix.size() + suffix.size() || path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0) {
            return false;
        }
        const size_t end = path.size() - suffix.size();
        return end > prefix.size() && path.substr(prefix.size(), end - prefix.size()).find('/') == std::string::npos;
    }
    return path.size() > prefix.size() && path.substr(prefix.size()).find('/') == std::string::npos;
}

std::string path_param_for_template(const std::string &path, const std::string &route_template) {
    const auto open = route_template.find('{');
    const auto close = route_template.find('}', open);
    if (open == std::string::npos || close == std::string::npos || !matches_template(path, route_template)) {
        return "";
    }
    const auto prefix = route_template.substr(0, open);
    const auto suffix = route_template.substr(close + 1);
    const size_t end = suffix.empty() ? path.size() : path.size() - suffix.size();
    return url_decode(path.substr(prefix.size(), end - prefix.size()));
}

double monotonic_epoch_seconds() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

std::string make_error(int status, const std::string &message) {
    std::ostringstream out;
    out << "{\"error\":{\"message\":\"" << json_escape(message)
        << "\",\"type\":\"ezlocalai_native_error\",\"code\":" << status << "}}";
    return out.str();
}

std::string parse_json_string_at(const std::string &json, size_t quote_pos) {
    if (quote_pos == std::string::npos || quote_pos >= json.size() || json[quote_pos] != '"') {
        return "";
    }
    std::string out;
    bool escaped = false;
    for (size_t i = quote_pos + 1; i < json.size(); ++i) {
        char ch = json[i];
        if (escaped) {
            switch (ch) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(ch);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    auto unit = [&](size_t start) -> int {
                        if (start + 4 > json.size()) return -1;
                        int code = 0;
                        for (size_t j = start; j < start + 4; ++j) {
                            const auto digit = hex_value(json[j]); if (digit < 0) return -1;
                            code = code * 16 + digit;
                        }
                        return code;
                    };
                    int code = unit(i + 1);
                    if (code < 0) { out.push_back(ch); break; }
                    i += 4;
                    if (code >= 0xd800 && code <= 0xdbff && i + 6 < json.size() && json[i + 1] == '\\' && json[i + 2] == 'u') {
                        const auto low = unit(i + 3);
                        if (low >= 0xdc00 && low <= 0xdfff) { code = 0x10000 + (code - 0xd800) * 0x400 + low - 0xdc00; i += 6; }
                    }
                    // Preserve lone surrogate code units until serialization;
                    // json.loads accepts them, while UTF-8 responses reject them.
                    if (code < 0x80) out += static_cast<char>(code);
                    else if (code < 0x800) { out += static_cast<char>(0xc0 | (code >> 6)); out += static_cast<char>(0x80 | (code & 63)); }
                    else if (code < 0x10000) { out += static_cast<char>(0xe0 | (code >> 12)); out += static_cast<char>(0x80 | ((code >> 6) & 63)); out += static_cast<char>(0x80 | (code & 63)); }
                    else { out += static_cast<char>(0xf0 | (code >> 18)); out += static_cast<char>(0x80 | ((code >> 12) & 63)); out += static_cast<char>(0x80 | ((code >> 6) & 63)); out += static_cast<char>(0x80 | (code & 63)); }
                    break;
                }
                default:
                    out.push_back(ch);
                    break;
            }
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else if (ch == '"') {
            return out;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::string extract_last_json_string(const std::string &json, const std::string &key, const std::string &fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t search = 0;
    std::string last;
    bool found = false;
    while (true) {
        size_t key_pos = json.find(needle, search);
        if (key_pos == std::string::npos) {
            break;
        }
        size_t colon = json.find(':', key_pos + needle.size());
        if (colon == std::string::npos) {
            break;
        }
        size_t quote = json.find('"', colon + 1);
        if (quote == std::string::npos) {
            break;
        }
        last = parse_json_string_at(json, quote);
        found = true;
        search = quote + 1;
    }
    return found ? last : fallback;
}

size_t json_value_start(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return std::string::npos;
    }
    return json.find_first_not_of(" \t\r\n", colon + 1);
}

size_t json_value_end(const std::string &json, size_t start) {
    if (start == std::string::npos || start >= json.size()) {
        return std::string::npos;
    }
    if (json[start] == '"') {
        bool escaped = false;
        for (size_t i = start + 1; i < json.size(); ++i) {
            if (escaped) {
                escaped = false;
            } else if (json[i] == '\\') {
                escaped = true;
            } else if (json[i] == '"') {
                return i + 1;
            }
        }
        return std::string::npos;
    }
    if (json[start] == '{' || json[start] == '[') {
        const char open = json[start];
        const char close = open == '{' ? '}' : ']';
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        for (size_t i = start; i < json.size(); ++i) {
            const char ch = json[i];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (ch == '\\') {
                    escaped = true;
                } else if (ch == '"') {
                    in_string = false;
                }
                continue;
            }
            if (ch == '"') {
                in_string = true;
            } else if (ch == open) {
                depth++;
            } else if (ch == close) {
                depth--;
                if (depth == 0) {
                    return i + 1;
                }
            }
        }
        return std::string::npos;
    }
    size_t end = json.find_first_of(",}]\r\n", start);
    if (end == std::string::npos) {
        end = json.size();
    }
    while (end > start && std::isspace(static_cast<unsigned char>(json[end - 1]))) {
        end--;
    }
    return end;
}

std::string extract_json_value(const std::string &json, const std::string &key, const std::string &fallback_json) {
    const size_t start = json_value_start(json, key);
    const size_t end = json_value_end(json, start);
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        return fallback_json;
    }
    return json.substr(start, end - start);
}

bool is_json_number_token(const std::string &token) {
    if (token.empty()) {
        return false;
    }
    size_t pos = 0;
    if (token[pos] == '-') {
        pos++;
        if (pos == token.size()) {
            return false;
        }
    }
    if (token[pos] == '0') {
        pos++;
    } else if (token[pos] >= '1' && token[pos] <= '9') {
        while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos]))) {
            pos++;
        }
    } else {
        return false;
    }
    if (pos < token.size() && token[pos] == '.') {
        pos++;
        if (pos == token.size() || !std::isdigit(static_cast<unsigned char>(token[pos]))) {
            return false;
        }
        while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos]))) {
            pos++;
        }
    }
    if (pos < token.size() && (token[pos] == 'e' || token[pos] == 'E')) {
        pos++;
        if (pos < token.size() && (token[pos] == '+' || token[pos] == '-')) {
            pos++;
        }
        if (pos == token.size() || !std::isdigit(static_cast<unsigned char>(token[pos]))) {
            return false;
        }
        while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos]))) {
            pos++;
        }
    }
    return pos == token.size();
}

bool is_json_value_token(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (value.empty()) {
        return false;
    }
    if (value == "true" || value == "false" || value == "null") {
        return true;
    }
    if (value.front() == '"' || value.front() == '{' || value.front() == '[') {
        return json_value_end(value, 0) == value.size();
    }
    return is_json_number_token(value);
}

bool is_python_falsey_json_value(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (value.empty() || value == "null" || value == "false" || value == "[]" || value == "{}") {
        return true;
    }
    if (value.front() == '"') {
        return parse_json_string_at(value, 0).empty();
    }
    if (is_json_number_token(value)) {
        try {
            return std::stold(value) == 0.0L;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool parse_python_int_string(std::string value, int *out) {
    value = trim(value);
    if (value.empty()) {
        return false;
    }
    std::string normalized;
    normalized.reserve(value.size());
    size_t pos = 0;
    if (value[pos] == '+' || value[pos] == '-') {
        normalized.push_back(value[pos]);
        ++pos;
        if (pos == value.size()) {
            return false;
        }
    }
    bool last_was_digit = false;
    bool saw_digit = false;
    for (; pos < value.size(); ++pos) {
        const unsigned char ch = static_cast<unsigned char>(value[pos]);
        if (std::isdigit(ch)) {
            normalized.push_back(static_cast<char>(ch));
            saw_digit = true;
            last_was_digit = true;
            continue;
        }
        if (value[pos] == '_') {
            if (!last_was_digit || pos + 1 >= value.size()
                || !std::isdigit(static_cast<unsigned char>(value[pos + 1]))) {
                return false;
            }
            last_was_digit = false;
            continue;
        }
        return false;
    }
    if (!saw_digit) {
        return false;
    }
    try {
        const long long parsed = std::stoll(normalized);
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
            return false;
        }
        *out = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool json_value_would_fail_python_int(const std::string &raw_value, bool default_on_falsey) {
    const std::string value = trim(raw_value);
    if (value.empty()) {
        return false;
    }
    if (default_on_falsey && is_python_falsey_json_value(value)) {
        return false;
    }
    if (value == "true" || value == "false" || is_json_number_token(value)) {
        return false;
    }
    if (value.front() == '"') {
        const std::string parsed = trim(parse_json_string_at(value, 0));
        int ignored = 0;
        return !parse_python_int_string(parsed, &ignored);
    }
    return true;
}

bool parse_strict_python_float_string(const std::string &raw, double *out);

bool json_value_would_fail_python_float(const std::string &raw_value, bool default_on_falsey) {
    const std::string value = trim(raw_value);
    if (value.empty()) {
        return false;
    }
    if (default_on_falsey && is_python_falsey_json_value(value)) {
        return false;
    }
    if (value == "true" || value == "false" || is_json_number_token(value)) {
        return false;
    }
    if (value.front() == '"') {
        const std::string parsed = trim(parse_json_string_at(value, 0));
        double ignored = 0.0;
        return !parse_strict_python_float_string(parsed, &ignored);
    }
    return true;
}

bool is_valid_top_level_json_object(const std::string &json) {
    const std::string value = trim(json);
    if (value.size() < 2 || value.front() != '{' || value.back() != '}') {
        return false;
    }
    size_t pos = 1;
    while (pos < value.size()) {
        pos = value.find_first_not_of(" \t\r\n", pos);
        if (pos == std::string::npos) {
            return false;
        }
        if (value[pos] == '}') {
            return pos + 1 == value.size();
        }
        if (value[pos] != '"') {
            return false;
        }
        const size_t key_end = json_value_end(value, pos);
        if (key_end == std::string::npos || key_end <= pos) {
            return false;
        }
        pos = value.find_first_not_of(" \t\r\n", key_end);
        if (pos == std::string::npos || value[pos] != ':') {
            return false;
        }
        pos = value.find_first_not_of(" \t\r\n", pos + 1);
        const size_t value_end = json_value_end(value, pos);
        if (pos == std::string::npos || value_end == std::string::npos || value_end <= pos) {
            return false;
        }
        if (!is_json_value_token(value.substr(pos, value_end - pos))) {
            return false;
        }
        pos = value.find_first_not_of(" \t\r\n", value_end);
        if (pos == std::string::npos) {
            return false;
        }
        if (value[pos] == ',') {
            pos++;
            continue;
        }
        if (value[pos] == '}') {
            return pos + 1 == value.size();
        }
        return false;
    }
    return false;
}

HttpResponse invalid_json_object_response(const std::string &body) {
    const std::string value = trim(body);
    if (value.empty()) {
        return {
            422,
            "application/json",
            "{\"detail\":[{\"type\":\"missing\",\"loc\":[\"body\"],\"msg\":\"Field required\",\"input\":null}]}"
        };
    }
    if (value.front() == '{') {
        return {
            422,
            "application/json",
            "{\"detail\":[{\"type\":\"json_invalid\",\"loc\":[\"body\",1],\"msg\":\"JSON decode error\",\"input\":{},\"ctx\":{\"error\":\"Expecting property name enclosed in double quotes\"}}]}"
        };
    }
    std::string input_json = value;
    if (value.front() == '"') {
        input_json = "\"" + json_escape(parse_json_string_at(value, 0)) + "\"";
    }
    return {
        422,
        "application/json",
        "{\"detail\":[{\"type\":\"dict_type\",\"loc\":[\"body\"],\"msg\":\"Input should be a valid dictionary\",\"input\":" + input_json + "}]}"
    };
}

HttpResponse invalid_json_model_response(const std::string &body) {
    const std::string value = trim(body);
    if (value.empty()) {
        return {
            422,
            "application/json",
            "{\"detail\":[{\"type\":\"missing\",\"loc\":[\"body\"],\"msg\":\"Field required\",\"input\":null}]}"
        };
    }
    if (value.front() == '{') {
        return {
            422,
            "application/json",
            "{\"detail\":[{\"type\":\"json_invalid\",\"loc\":[\"body\",1],\"msg\":\"JSON decode error\",\"input\":{},\"ctx\":{\"error\":\"Expecting property name enclosed in double quotes\"}}]}"
        };
    }
    std::string input_json = value;
    if (value.front() == '"') {
        input_json = "\"" + json_escape(parse_json_string_at(value, 0)) + "\"";
    }
    return {
        422,
        "application/json",
        "{\"detail\":[{\"type\":\"model_attributes_type\",\"loc\":[\"body\"],\"msg\":\"Input should be a valid dictionary or object to extract fields from\",\"input\":" + input_json + "}]}"
    };
}

std::string pydantic_input_json(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (!value.empty() && value.front() == '"') {
        return "\"" + json_escape(parse_json_string_at(value, 0)) + "\"";
    }
    return value.empty() ? "null" : value;
}

bool parse_pydantic_integral_string(std::string value, int *out) {
    value = trim(value);
    if (value.empty()) {
        return false;
    }
    const auto dot = value.find('.');
    std::string integral = dot == std::string::npos ? value : value.substr(0, dot);
    if (dot != std::string::npos) {
        const std::string decimal = value.substr(dot + 1);
        if (decimal.empty()) {
            return false;
        }
        for (const char ch : decimal) {
            if (ch != '0') {
                return false;
            }
        }
    }
    return parse_python_int_string(integral, out);
}

struct PydanticIntParseResult {
    bool ok = false;
    int value = 0;
    std::string type;
    std::string message;
};

PydanticIntParseResult parse_pydantic_int_token(const std::string &raw_value) {
    PydanticIntParseResult result;
    const std::string value = trim(raw_value);
    if (value == "true" || value == "false") {
        result.ok = true;
        result.value = value == "true" ? 1 : 0;
        return result;
    }
    if (value.empty() || value == "null" || value.front() == '[' || value.front() == '{') {
        result.type = "int_type";
        result.message = "Input should be a valid integer";
        return result;
    }
    if (value.front() == '"') {
        int parsed = 0;
        if (parse_pydantic_integral_string(parse_json_string_at(value, 0), &parsed)) {
            result.ok = true;
            result.value = parsed;
            return result;
        }
        result.type = "int_parsing";
        result.message = "Input should be a valid integer, unable to parse string as an integer";
        return result;
    }
    if (is_json_number_token(value)) {
        try {
            const long double parsed = std::stold(value);
            if (std::isfinite(static_cast<double>(parsed)) && std::floor(parsed) == parsed) {
                result.ok = true;
                result.value = static_cast<int>(parsed);
                return result;
            }
        } catch (...) {
        }
        result.type = "int_from_float";
        result.message = "Input should be a valid integer, got a number with a fractional part";
        return result;
    }
    result.type = "int_type";
    result.message = "Input should be a valid integer";
    return result;
}

void append_pydantic_error(std::vector<std::string> *errors,
                           const std::string &type,
                           const std::vector<std::string> &loc_tail,
                           const std::string &message,
                           const std::string &input_json);
std::vector<std::string> json_array_elements(const std::string &raw);
std::map<std::string, std::string> top_level_json_object_values(const std::string &json);
bool parse_pydantic_bool_string(const std::string &value, bool *parsed);

void append_pydantic_error(std::vector<std::string> *errors,
                           const std::string &type,
                           const std::string &field,
                           const std::string &message,
                           const std::string &input_json) {
    append_pydantic_error(errors, type, std::vector<std::string>{field}, message, input_json);
}

void append_pydantic_error(std::vector<std::string> *errors,
                           const std::string &type,
                           const std::vector<std::string> &loc_tail,
                           const std::string &message,
                           const std::string &input_json) {
    std::ostringstream error;
    error << "{\"type\":\"" << json_escape(type) << "\","
          << "\"loc\":[\"body\"";
    for (const auto &part : loc_tail) {
        const bool numeric = !part.empty() && std::all_of(part.begin(), part.end(), [](unsigned char ch) {
            return std::isdigit(ch);
        });
        if (numeric) {
            error << "," << part;
        } else {
            error << ",\"" << json_escape(part) << "\"";
        }
    }
    error << "],"
          << "\"msg\":\"" << json_escape(message) << "\","
          << "\"input\":" << input_json << "}";
    errors->push_back(error.str());
}

std::string pydantic_error_response_body(const std::vector<std::string> &errors) {
    std::ostringstream out;
    out << "{\"detail\":[";
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << errors[i];
    }
    out << "]}";
    return out.str();
}

bool validate_pydantic_required_string_field(const std::map<std::string, std::string> &request_values,
                                             const std::string &body,
                                             const std::string &field,
                                             std::string *value_out,
                                             std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end()) {
        append_pydantic_error(
            validation_errors,
            "missing",
            field,
            "Field required",
            trim(body)
        );
        return false;
    }
    const std::string value = trim(it->second);
    if (value.empty() || value.front() != '"') {
        append_pydantic_error(
            validation_errors,
            "string_type",
            field,
            "Input should be a valid string",
            pydantic_input_json(it->second)
        );
        return false;
    }
    if (value_out) {
        *value_out = parse_json_string_at(value, 0);
    }
    return true;
}

void validate_pydantic_optional_string_field(const std::map<std::string, std::string> &request_values,
                                             const std::string &field,
                                             std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end()) {
        return;
    }
    const std::string value = trim(it->second);
    if (value == "null") {
        return;
    }
    if (value.empty() || value.front() != '"') {
        append_pydantic_error(
            validation_errors,
            "string_type",
            field,
            "Input should be a valid string",
            pydantic_input_json(it->second)
        );
    }
}

void validate_pydantic_default_string_field(const std::map<std::string, std::string> &request_values,
                                            const std::string &field,
                                            std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end()) {
        return;
    }
    const std::string value = trim(it->second);
    if (value.empty() || value.front() != '"') {
        append_pydantic_error(
            validation_errors,
            "string_type",
            field,
            "Input should be a valid string",
            pydantic_input_json(it->second)
        );
    }
}

void validate_pydantic_optional_int_field(const std::map<std::string, std::string> &request_values,
                                          const std::string &field,
                                          std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end() || trim(it->second) == "null") {
        return;
    }
    const auto parsed = parse_pydantic_int_token(it->second);
    if (!parsed.ok) {
        append_pydantic_error(
            validation_errors,
            parsed.type,
            field,
            parsed.message,
            pydantic_input_json(it->second)
        );
    }
}

void validate_pydantic_optional_float_field(const std::map<std::string, std::string> &request_values,
                                            const std::string &field,
                                            std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end() || trim(it->second) == "null") {
        return;
    }
    std::string value = trim(it->second);
    bool ok = value == "true" || value == "false" || is_json_number_token(value);
    if (!ok && !value.empty() && value.front() == '"') {
        const std::string parsed = trim(parse_json_string_at(value, 0));
        double ignored = 0.0;
        ok = parse_strict_python_float_string(parsed, &ignored);
    }
    if (!ok) {
        const bool string_input = !value.empty() && value.front() == '"';
        append_pydantic_error(
            validation_errors,
            string_input ? "float_parsing" : "float_type",
            field,
            string_input ? "Input should be a valid number, unable to parse string as a number" : "Input should be a valid number",
            pydantic_input_json(it->second)
        );
    }
}

void validate_pydantic_optional_bool_field(const std::map<std::string, std::string> &request_values,
                                           const std::string &field,
                                           std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end() || trim(it->second) == "null") {
        return;
    }
    std::string value = trim(it->second);
    bool ok = false;
    std::string error_type = "bool_parsing";
    std::string error_message = "Input should be a valid boolean, unable to interpret input";
    if (!value.empty() && value.front() == '"') {
        bool parsed = false;
        ok = parse_pydantic_bool_string(parse_json_string_at(value, 0), &parsed);
    } else if (value == "true" || value == "false") {
        ok = true;
    } else if (is_json_number_token(value)) {
        try {
            const long double parsed = std::stold(value);
            ok = parsed == 0.0L || parsed == 1.0L;
            if (!ok && (!std::isfinite(static_cast<double>(parsed)) || std::floor(parsed) != parsed)) {
                error_type = "bool_type";
                error_message = "Input should be a valid boolean";
            }
        } catch (...) {
            ok = false;
        }
    } else {
        error_type = "bool_type";
        error_message = "Input should be a valid boolean";
    }
    if (!ok) {
        append_pydantic_error(
            validation_errors,
            error_type,
            field,
            error_message,
            pydantic_input_json(it->second)
        );
    }
}

void validate_pydantic_optional_list_of_strings_field(const std::map<std::string, std::string> &request_values,
                                                      const std::string &field,
                                                      std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end() || trim(it->second) == "null") {
        return;
    }
    const std::string value = trim(it->second);
    if (value.empty() || value.front() != '[') {
        append_pydantic_error(validation_errors, "list_type", field, "Input should be a valid list", pydantic_input_json(it->second));
        return;
    }
    const auto elements = json_array_elements(value);
    for (size_t i = 0; i < elements.size(); ++i) {
        if (trim(elements[i]).empty() || trim(elements[i]).front() != '"') {
            append_pydantic_error(
                validation_errors,
                "string_type",
                std::vector<std::string>{field, std::to_string(i)},
                "Input should be a valid string",
                pydantic_input_json(elements[i])
            );
        }
    }
}

void validate_pydantic_optional_list_of_dicts_field(const std::map<std::string, std::string> &request_values,
                                                    const std::string &field,
                                                    std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end() || trim(it->second) == "null") {
        return;
    }
    const std::string value = trim(it->second);
    if (value.empty() || value.front() != '[') {
        append_pydantic_error(validation_errors, "list_type", field, "Input should be a valid list", pydantic_input_json(it->second));
        return;
    }
    const auto elements = json_array_elements(value);
    for (size_t i = 0; i < elements.size(); ++i) {
        if (!is_valid_top_level_json_object(elements[i])) {
            append_pydantic_error(
                validation_errors,
                "dict_type",
                std::vector<std::string>{field, std::to_string(i)},
                "Input should be a valid dictionary",
                pydantic_input_json(elements[i])
            );
        }
    }
}

void validate_pydantic_optional_dict_field(const std::map<std::string, std::string> &request_values,
                                           const std::string &field,
                                           std::vector<std::string> *validation_errors) {
    const auto it = request_values.find(field);
    if (it == request_values.end() || trim(it->second) == "null") {
        return;
    }
    if (!is_valid_top_level_json_object(it->second)) {
        append_pydantic_error(validation_errors, "dict_type", field, "Input should be a valid dictionary", pydantic_input_json(it->second));
    }
}

void validate_pydantic_embedding_input_field(const std::map<std::string, std::string> &request_values,
                                             const std::string &body,
                                             std::vector<std::string> *validation_errors) {
    const auto it = request_values.find("input");
    if (it == request_values.end()) {
        append_pydantic_error(validation_errors, "missing", "input", "Field required", trim(body));
        return;
    }
    const std::string value = trim(it->second);
    if (!value.empty() && value.front() == '"') {
        return;
    }
    if (value.empty() || value.front() != '[') {
        append_pydantic_error(validation_errors, "string_type", std::vector<std::string>{"input", "str"}, "Input should be a valid string", pydantic_input_json(it->second));
        append_pydantic_error(validation_errors, "list_type", std::vector<std::string>{"input", "list[str]"}, "Input should be a valid list", pydantic_input_json(it->second));
        return;
    }
    bool has_bad_element = false;
    const auto elements = json_array_elements(value);
    for (const auto &element : elements) {
        if (trim(element).empty() || trim(element).front() != '"') {
            has_bad_element = true;
            break;
        }
    }
    if (has_bad_element) {
        append_pydantic_error(validation_errors, "string_type", std::vector<std::string>{"input", "str"}, "Input should be a valid string", pydantic_input_json(it->second));
        for (size_t i = 0; i < elements.size(); ++i) {
            if (trim(elements[i]).empty() || trim(elements[i]).front() != '"') {
                append_pydantic_error(
                    validation_errors,
                    "string_type",
                    std::vector<std::string>{"input", "list[str]", std::to_string(i)},
                    "Input should be a valid string",
                    pydantic_input_json(elements[i])
                );
            }
        }
    }
}

void validate_pydantic_video_conditions_field(const std::map<std::string, std::string> &request_values,
                                              std::vector<std::string> *validation_errors) {
    const auto it = request_values.find("conditions");
    if (it == request_values.end() || trim(it->second) == "null") {
        return;
    }
    const std::string value = trim(it->second);
    if (value.empty() || value.front() != '[') {
        append_pydantic_error(validation_errors, "list_type", "conditions", "Input should be a valid list", pydantic_input_json(it->second));
        return;
    }
    const auto elements = json_array_elements(value);
    for (size_t i = 0; i < elements.size(); ++i) {
        const std::string element = trim(elements[i]);
        const std::vector<std::string> item_loc = {"conditions", std::to_string(i)};
        if (!is_valid_top_level_json_object(element)) {
            append_pydantic_error(
                validation_errors,
                "model_attributes_type",
                item_loc,
                "Input should be a valid dictionary or object to extract fields from",
                pydantic_input_json(element)
            );
            continue;
        }
        const auto condition_values = top_level_json_object_values(element);
        const auto image = condition_values.find("image");
        if (image == condition_values.end()) {
            append_pydantic_error(
                validation_errors,
                "missing",
                std::vector<std::string>{"conditions", std::to_string(i), "image"},
                "Field required",
                element
            );
        } else if (trim(image->second).empty() || trim(image->second).front() != '"') {
            append_pydantic_error(
                validation_errors,
                "string_type",
                std::vector<std::string>{"conditions", std::to_string(i), "image"},
                "Input should be a valid string",
                pydantic_input_json(image->second)
            );
        }
        const auto index = condition_values.find("index");
        if (index != condition_values.end() && trim(index->second) != "null") {
            const auto parsed = parse_pydantic_int_token(index->second);
            if (!parsed.ok) {
                append_pydantic_error(
                    validation_errors,
                    parsed.type,
                    std::vector<std::string>{"conditions", std::to_string(i), "index"},
                    parsed.message,
                    pydantic_input_json(index->second)
                );
            }
        }
        const auto strength = condition_values.find("strength");
        if (strength != condition_values.end() && trim(strength->second) != "null") {
            const std::string raw_strength = trim(strength->second);
            bool ok = raw_strength == "true" || raw_strength == "false" || is_json_number_token(raw_strength);
            if (!ok && !raw_strength.empty() && raw_strength.front() == '"') {
                const std::string parsed = trim(parse_json_string_at(raw_strength, 0));
                double ignored = 0.0;
                ok = parse_strict_python_float_string(parsed, &ignored);
            }
            if (!ok) {
                const bool string_input = !raw_strength.empty() && raw_strength.front() == '"';
                append_pydantic_error(
                    validation_errors,
                    string_input ? "float_parsing" : "float_type",
                    std::vector<std::string>{"conditions", std::to_string(i), "strength"},
                    string_input ? "Input should be a valid number, unable to parse string as a number" : "Input should be a valid number",
                    pydantic_input_json(strength->second)
                );
            }
        }
    }
}

HttpResponse wakeword_disabled_response() {
    return {404, "application/json", "{\"detail\":\"Wake word detection is disabled. Set VOICE_SERVER=true to enable.\"}"};
}

HttpResponse pydantic_query_int_error(const std::string &field, const std::string &input) {
    return {
        422,
        "application/json",
        "{\"detail\":[{\"type\":\"int_parsing\",\"loc\":[\"query\",\"" + json_escape(field) + "\"],\"msg\":\"Input should be a valid integer, unable to parse string as an integer\",\"input\":\"" + json_escape(input) + "\"}]}"
    };
}

std::string pydantic_missing_file_error_object() {
    return "{\"type\":\"missing\",\"loc\":[\"body\",\"file\"],\"msg\":\"Field required\",\"input\":null}";
}

std::string pydantic_upload_file_type_error_object(const std::string &input) {
    return "{\"type\":\"value_error\",\"loc\":[\"body\",\"file\"],\"msg\":\"Value error, Expected UploadFile, received: <class 'str'>\",\"input\":\""
        + json_escape(input)
        + "\",\"ctx\":{\"error\":{}}}";
}

HttpResponse pydantic_missing_file_error() {
    return {
        422,
        "application/json",
        pydantic_error_response_body({pydantic_missing_file_error_object()})
    };
}

HttpResponse pydantic_upload_file_type_error(const std::string &input) {
    return {
        422,
        "application/json",
        pydantic_error_response_body({pydantic_upload_file_type_error_object(input)})
    };
}

HttpResponse plain_internal_server_error() {
    return {500, "text/plain; charset=utf-8", "Internal Server Error"};
}

HttpResponse worker_transport_error_response(const std::string &detail) {
    HttpResponse response{503, "application/json", "{\"detail\":\"" + json_escape(detail) + "\"}"};
    response.transport_error = true;
    return response;
}

void validate_form_float_field(const std::unordered_map<std::string, std::string> &fields,
                               const std::string &field,
                               std::vector<std::string> *validation_errors) {
    const auto it = fields.find(field);
    if (it == fields.end()) {
        return;
    }
    const std::string value = trim(it->second);
    double ignored = 0.0;
    if (parse_strict_python_float_string(value, &ignored)) {
        return;
    }
    append_pydantic_error(
        validation_errors,
        "float_parsing",
        field,
        "Input should be a valid number, unable to parse string as a number",
        "\"" + json_escape(it->second) + "\""
    );
}

bool parse_pydantic_bool_string(const std::string &value, bool *parsed) {
    static const std::set<std::string> truthy = {"1", "true", "on", "yes", "y", "t"};
    static const std::set<std::string> falsey = {"0", "false", "off", "no", "n", "f"};
    const std::string normalized = to_lower(value);
    if (truthy.find(normalized) != truthy.end()) {
        *parsed = true;
        return true;
    }
    if (falsey.find(normalized) != falsey.end()) {
        *parsed = false;
        return true;
    }
    return false;
}

void validate_form_int_field(const std::unordered_map<std::string, std::string> &fields,
                             const std::string &field,
                             std::vector<std::string> *validation_errors) {
    const auto it = fields.find(field);
    if (it == fields.end()) {
        return;
    }
    int parsed = 0;
    if (parse_pydantic_integral_string(it->second, &parsed)) {
        return;
    }
    append_pydantic_error(
        validation_errors,
        "int_parsing",
        field,
        "Input should be a valid integer, unable to parse string as an integer",
        "\"" + json_escape(it->second) + "\""
    );
}

void validate_form_bool_field(const std::unordered_map<std::string, std::string> &fields,
                              const std::string &field,
                              std::vector<std::string> *validation_errors) {
    const auto it = fields.find(field);
    if (it == fields.end()) {
        return;
    }
    bool parsed = false;
    if (parse_pydantic_bool_string(it->second, &parsed)) {
        return;
    }
    append_pydantic_error(
        validation_errors,
        "bool_parsing",
        field,
        "Input should be a valid boolean, unable to interpret input",
        "\"" + json_escape(it->second) + "\""
    );
}

std::map<std::string, std::string> top_level_json_object_values(const std::string &json) {
    std::map<std::string, std::string> values;
    size_t pos = json.find_first_not_of(" \t\r\n");
    if (pos == std::string::npos || json[pos] != '{') {
        return values;
    }
    ++pos;
    while (pos < json.size()) {
        pos = json.find_first_not_of(" \t\r\n,", pos);
        if (pos == std::string::npos || pos >= json.size() || json[pos] == '}') {
            break;
        }
        if (json[pos] != '"') {
            break;
        }
        const std::string key = parse_json_string_at(json, pos);
        const size_t key_end = json_value_end(json, pos);
        if (key_end == std::string::npos) {
            break;
        }
        const size_t colon = json.find(':', key_end);
        if (colon == std::string::npos) {
            break;
        }
        const size_t value_start = json.find_first_not_of(" \t\r\n", colon + 1);
        const size_t value_end = json_value_end(json, value_start);
        if (value_start == std::string::npos || value_end == std::string::npos || value_end <= value_start) {
            break;
        }
        values[key] = json.substr(value_start, value_end - value_start);
        pos = value_end;
    }
    return values;
}

std::string json_object_from_values(const std::map<std::string, std::string> &values) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (const auto &entry : values) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << json_escape(entry.first) << "\":" << entry.second;
    }
    out << "}";
    return out.str();
}

void ordered_json_upsert(std::vector<std::pair<std::string, std::string>> *values,
                         const std::pair<std::string, std::string> &entry) {
    for (auto &existing : *values) {
        if (existing.first == entry.first) {
            existing.second = entry.second;
            return;
        }
    }
    values->push_back(entry);
}

std::string json_object_from_ordered_entries(const std::vector<std::pair<std::string, std::string>> &values) {
    std::ostringstream out;
    out << "{";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << "\"" << json_escape(values[i].first) << "\":" << values[i].second;
    }
    out << "}";
    return out.str();
}

std::string json_object_field(const std::map<std::string, std::string> &fields,
                              const std::string &key, const std::string &fallback) {
    const auto found = fields.find(key);
    return found == fields.end() ? fallback : found->second;
}

std::string json_object_field(const std::string &object, const std::string &key, const std::string &fallback) {
    return json_object_field(top_level_json_object_values(object), key, fallback);
}

std::string merge_json_objects(std::string base_json, const std::string &patch_json) {
    std::vector<std::pair<std::string, std::string>> values;
    for (const auto &entry : json_object_ordered_entries(base_json)) {
        ordered_json_upsert(&values, entry);
    }
    const auto patch = json_object_ordered_entries(patch_json);
    for (const auto &entry : patch) {
        ordered_json_upsert(&values, entry);
    }
    return json_object_from_ordered_entries(values);
}

std::string router_register_extra_json(const std::string &body) {
    static const std::set<std::string> known = {
        "worker_id", "label", "url", "port", "api_key", "capabilities", "models",
        "free_vram_gb", "total_vram_gb", "free_ram_gb", "total_ram_gb",
        "queue_depth", "queue_capacity", "in_flight", "cap_slots", "model_slots",
        "gpus", "best_tier", "model_context", "model_quant", "cap_models", "version", "extra",
    };
    std::vector<std::pair<std::string, std::string>> values;
    for (const auto &entry : json_object_ordered_entries(body)) {
        if (known.find(entry.first) == known.end()) {
            ordered_json_upsert(&values, entry);
        }
    }
    return merge_json_objects(json_object_from_ordered_entries(values), json_object_field(body, "extra", "{}"));
}

double extract_json_double_value(const std::string &json, const std::string &key, double fallback) {
    auto value = trim(extract_json_value(json, key, ""));
    if (value.empty()) {
        return fallback;
    }
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = parse_json_string_at(value, 0);
    }
    try {
        return std::stod(value);
    } catch (...) {
        return fallback;
    }
}

int extract_json_int_value(const std::string &json, const std::string &key, int fallback) {
    auto value = trim(extract_json_value(json, key, ""));
    if (value.empty()) {
        return fallback;
    }
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = parse_json_string_at(value, 0);
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

int coerce_python_int_value(std::string value, int fallback, bool default_on_falsey) {
    value = trim(value);
    if (value.empty() || (default_on_falsey && is_python_falsey_json_value(value))) {
        return fallback;
    }
    if (value == "true") {
        return 1;
    }
    if (value == "false") {
        return 0;
    }
    const bool was_string = value.size() >= 2 && value.front() == '"' && value.back() == '"';
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = trim(parse_json_string_at(value, 0));
        int parsed = fallback;
        return parse_python_int_string(value, &parsed) ? parsed : fallback;
    }
    try {
        size_t consumed = 0;
        const long double parsed = std::stold(value, &consumed);
        if (!was_string && consumed == value.size()) {
            return static_cast<int>(parsed);
        }
    } catch (...) {
    }
    return fallback;
}

int extract_python_int_value(const std::string &json, const std::string &key, int fallback, bool default_on_falsey) {
    return coerce_python_int_value(extract_json_value(json, key, ""), fallback, default_on_falsey);
}

double coerce_python_float_value(std::string value, double fallback, bool default_on_falsey) {
    value = trim(value);
    if (value.empty() || (default_on_falsey && is_python_falsey_json_value(value))) {
        return fallback;
    }
    if (value == "true") {
        return 1.0;
    }
    if (value == "false") {
        return 0.0;
    }
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = trim(parse_json_string_at(value, 0));
        double parsed = fallback;
        return parse_strict_python_float_string(value, &parsed) ? parsed : fallback;
    }
    try {
        size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed == value.size()) {
            return parsed;
        }
    } catch (...) {
    }
    return fallback;
}

double extract_python_float_value(const std::string &json, const std::string &key, double fallback, bool default_on_falsey) {
    return coerce_python_float_value(extract_json_value(json, key, ""), fallback, default_on_falsey);
}

std::vector<std::string> extract_json_string_array(const std::string &json, const std::string &key) {
    std::vector<std::string> values;
    const auto raw = extract_json_value(json, key, "[]");
    if (raw.size() < 2 || raw.front() != '[') {
        return values;
    }
    size_t i = 1;
    while (i < raw.size()) {
        i = raw.find('"', i);
        if (i == std::string::npos) {
            break;
        }
        values.push_back(parse_json_string_at(raw, i));
        i++;
        bool escaped = false;
        while (i < raw.size()) {
            if (escaped) {
                escaped = false;
            } else if (raw[i] == '\\') {
                escaped = true;
            } else if (raw[i] == '"') {
                i++;
                break;
            }
            i++;
        }
    }
    return values;
}

std::vector<std::string> extract_json_array_elements(const std::string &json, const std::string &key) {
    std::vector<std::string> values;
    const auto raw = extract_json_value(json, key, "[]");
    if (raw.size() < 2 || raw.front() != '[') {
        return values;
    }
    size_t pos = 1;
    while (pos < raw.size()) {
        pos = raw.find_first_not_of(" \t\r\n,", pos);
        if (pos == std::string::npos || pos >= raw.size() || raw[pos] == ']') {
            break;
        }
        const size_t end = json_value_end(raw, pos);
        if (end == std::string::npos || end <= pos) {
            break;
        }
        values.push_back(raw.substr(pos, end - pos));
        pos = end;
    }
    return values;
}

std::vector<std::string> json_array_elements(const std::string &raw) {
    std::vector<std::string> values;
    if (raw.size() < 2 || raw.front() != '[') {
        return values;
    }
    size_t pos = 1;
    while (pos < raw.size()) {
        pos = raw.find_first_not_of(" \t\r\n,", pos);
        if (pos == std::string::npos || pos >= raw.size() || raw[pos] == ']') {
            break;
        }
        const size_t end = json_value_end(raw, pos);
        if (end == std::string::npos || end <= pos) {
            break;
        }
        values.push_back(raw.substr(pos, end - pos));
        pos = end;
    }
    return values;
}

std::vector<std::string> python_list_strings_from_json_value(const std::string &raw_value, bool default_on_falsey, bool *ok) {
    if (ok) {
        *ok = true;
    }
    const std::string value = trim(raw_value);
    if (value.empty() || (default_on_falsey && is_python_falsey_json_value(value))) {
        return {};
    }
    if (value.front() == '"') {
        std::vector<std::string> chars;
        for (const char ch : parse_json_string_at(value, 0)) {
            chars.push_back(std::string(1, ch));
        }
        return chars;
    }
    if (value.front() == '[') {
        std::vector<std::string> out;
        for (const auto &element : json_array_elements(value)) {
            const std::string item = trim(element);
            if (!item.empty() && item.front() == '"') {
                out.push_back(parse_json_string_at(item, 0));
            }
        }
        return out;
    }
    if (value.front() == '{') {
        std::vector<std::string> out;
        for (const auto &entry : json_object_ordered_entries(value)) {
            out.push_back(entry.first);
        }
        return out;
    }
    if (ok) {
        *ok = false;
    }
    return {};
}

std::string python_public_list_json_from_value(const std::string &raw_value, bool default_on_falsey, bool *ok) {
    if (ok) {
        *ok = true;
    }
    const std::string value = trim(raw_value);
    if (value.empty() || (default_on_falsey && is_python_falsey_json_value(value))) {
        return "[]";
    }
    if (value == "null" || value == "false" || is_json_number_token(value)) {
        if (ok) {
            *ok = false;
        }
        return "[]";
    }
    if (value.front() == '[') {
        return value;
    }
    if (value.front() == '"') {
        std::ostringstream out;
        out << "[";
        bool first = true;
        for (const char ch : parse_json_string_at(value, 0)) {
            if (!first) {
                out << ",";
            }
            first = false;
            out << "\"" << json_escape(std::string(1, ch)) << "\"";
        }
        out << "]";
        return out.str();
    }
    if (value.front() == '{') {
        std::ostringstream out;
        out << "[";
        bool first = true;
        for (const auto &entry : json_object_ordered_entries(value)) {
            if (!first) {
                out << ",";
            }
            first = false;
            out << "\"" << json_escape(entry.first) << "\"";
        }
        out << "]";
        return out.str();
    }
    if (ok) {
        *ok = false;
    }
    return "[]";
}

bool router_register_gpus_would_fail_python(const std::string &raw_value, const std::string &best_tier_raw) {
    const std::string value = trim(raw_value);
    if (value.empty() || is_python_falsey_json_value(value)) {
        return false;
    }
    if (value.front() != '[') {
        return true;
    }
    const auto gpus = json_array_elements(value);
    for (const auto &gpu : gpus) {
        if (trim(gpu).empty() || trim(gpu).front() != '{') {
            return true;
        }
        const auto fields = top_level_json_object_values(gpu);
        const auto total = fields.find("total_vram_gb");
        if (total != fields.end()) {
            const std::string raw_total = trim(total->second);
            if (!(raw_total == "true" || raw_total == "false" || is_json_number_token(raw_total))) {
                return true;
            }
        }
    }
    if (!is_python_falsey_json_value(best_tier_raw)) {
        return false;
    }

    bool has_accel = false;
    bool has_cpu = false;
    bool cpu_has_tier = false;
    for (const auto &gpu : gpus) {
        const auto fields = top_level_json_object_values(gpu);
        const auto index = fields.find("index");
        if (index != fields.end()) {
            const std::string raw_index = trim(index->second);
            if (!(raw_index == "true" || raw_index == "false" || is_json_number_token(raw_index))) {
                return true;
            }
        }
        const int parsed_index = extract_json_int_value(gpu, "index", 0);
        if (parsed_index >= 0) {
            has_accel = true;
            const auto tier = fields.find("tier");
            if (tier != fields.end()) {
                const std::string raw_tier = trim(tier->second);
                if (!(raw_tier == "true" || raw_tier == "false" || is_json_number_token(raw_tier) || raw_tier.front() == '"')) {
                    return true;
                }
                if (raw_tier.front() == '"') {
                    int ignored = 0;
                    if (!parse_python_int_string(parse_json_string_at(raw_tier, 0), &ignored)) {
                        return true;
                    }
                }
            }
        } else if (extract_json_string(gpu, "backend", "") == "cpu") {
            has_cpu = true;
            const auto tier = fields.find("tier");
            cpu_has_tier = tier != fields.end();
            if (!cpu_has_tier) {
                continue;
            }
            const std::string raw_tier = trim(tier->second);
            if (!(raw_tier == "true" || raw_tier == "false" || is_json_number_token(raw_tier) || raw_tier.front() == '"')) {
                return true;
            }
            if (raw_tier.front() == '"') {
                int ignored = 0;
                if (!parse_python_int_string(parse_json_string_at(raw_tier, 0), &ignored)) {
                    return true;
                }
            }
        }
    }
    return !has_accel && has_cpu && !cpu_has_tier;
}

bool json_array_contains_non_object(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (value.empty() || value.front() != '[') {
        return false;
    }
    for (const auto &element : json_array_elements(value)) {
        const std::string item = trim(element);
        if (item.empty() || item.front() != '{') {
            return true;
        }
    }
    return false;
}

bool json_object_int_values_would_fail_python(const std::string &raw_value, bool fail_truthy_non_dict) {
    const std::string value = trim(raw_value);
    if (value.empty() || is_python_falsey_json_value(value)) {
        return false;
    }
    if (value.front() != '{') {
        return fail_truthy_non_dict;
    }
    for (const auto &entry : top_level_json_object_values(value)) {
        if (json_value_would_fail_python_int(entry.second, false)) {
            return true;
        }
    }
    return false;
}

bool json_slot_map_values_would_fail_python(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (value.empty() || is_python_falsey_json_value(value) || value.front() != '{') {
        return false;
    }
    for (const auto &slot_entry : top_level_json_object_values(value)) {
        const std::string slot = trim(slot_entry.second);
        if (slot.empty() || slot.front() != '{') {
            continue;
        }
        const auto slot_values = top_level_json_object_values(slot);
        for (const auto &field : {"capacity", "in_flight", "queued", "instances"}) {
            const auto it = slot_values.find(field);
            if (it != slot_values.end() && json_value_would_fail_python_int(it->second, true)) {
                return true;
            }
        }
    }
    return false;
}

bool json_truthy_non_dict(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    return !value.empty() && !is_python_falsey_json_value(value) && value.front() != '{';
}

int best_gpu_tier_from_json(const std::string &gpus_json) {
    int best_accel = -1;
    int cpu_tier = -1;
    for (const auto &gpu : json_array_elements(gpus_json)) {
        const int index = extract_json_int_value(gpu, "index", 0);
        const std::string backend = to_lower(extract_json_string(gpu, "backend", ""));
        if (index >= 0) {
            best_accel = std::max(best_accel, extract_json_int_value(gpu, "tier", 20));
        } else if (backend == "cpu" && cpu_tier < 0) {
            cpu_tier = extract_json_int_value(gpu, "tier", 2);
        }
    }
    if (best_accel >= 0) {
        return best_accel;
    }
    if (cpu_tier >= 0) {
        return cpu_tier;
    }
    return 2;
}

std::string string_array_json(const std::vector<std::string> &values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]";
    return out.str();
}

std::string json_array_from_values(const std::vector<std::string> &values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

bool python_json_truthy(const std::string &body) {
    const std::string raw = trim(body);
    if (raw.empty()) {
        return false;
    }
    const std::string lowered = to_lower(raw);
    if (lowered == "null" || lowered == "false") {
        return false;
    }
    if (raw.front() == '[') {
        return !json_array_elements(raw).empty();
    }
    if (raw.front() == '{') {
        const size_t value_pos = raw.find_first_not_of(" \t\r\n", 1);
        return value_pos != std::string::npos && raw[value_pos] != '}';
    }
    if (raw.front() == '"') {
        return !parse_json_string_at(raw, 0).empty();
    }
    try {
        size_t consumed = 0;
        const double number = std::stod(raw, &consumed);
        if (consumed == raw.size()) {
            return number != 0.0;
        }
    } catch (...) {
    }
    return true;
}

RouterSlotState normalize_slot(const std::string &raw_json, int fallback_capacity = 1) {
    RouterSlotState state;
    auto python_slot_int = [](const std::string &raw_value, int fallback) {
        return raw_value.empty() ? fallback : coerce_python_int_value(raw_value, 0, true);
    };
    state.capacity = std::max(0, python_slot_int(extract_json_value(raw_json, "capacity", ""), fallback_capacity));
    state.in_flight = std::max(0, python_slot_int(extract_json_value(raw_json, "in_flight", ""), 0));
    state.queued = std::max(0, python_slot_int(extract_json_value(raw_json, "queued", ""), 0));
    state.available = std::max(0, state.capacity - state.in_flight - state.queued);
    const auto instances = json_object_field(raw_json, "instances", "");
    if (!instances.empty()) state.instances = std::min(state.capacity, std::max(0, coerce_python_int_value(instances, 0, true)));
    return state;
}

RouterSlotState normalize_slot(const RouterSlotState &raw, int fallback_capacity = 1) {
    RouterSlotState state;
    (void)fallback_capacity;
    state.capacity = std::max(0, raw.capacity);
    state.in_flight = std::max(0, raw.in_flight);
    state.queued = std::max(0, raw.queued);
    state.available = std::max(0, state.capacity - state.in_flight - state.queued);
    if (raw.instances >= 0) state.instances = std::min(state.capacity, raw.instances);
    return state;
}

std::map<std::string, RouterSlotState> parse_slot_map_json(const std::string &raw_json) {
    std::map<std::string, RouterSlotState> slots;
    const std::string raw = trim(raw_json);
    if (raw.size() < 2 || raw.front() != '{') {
        return slots;
    }
    size_t pos = 1;
    while (pos < raw.size()) {
        pos = raw.find('"', pos);
        if (pos == std::string::npos) {
            break;
        }
        const std::string key = parse_json_string_at(raw, pos);
        pos++;
        bool escaped = false;
        while (pos < raw.size()) {
            if (escaped) {
                escaped = false;
            } else if (raw[pos] == '\\') {
                escaped = true;
            } else if (raw[pos] == '"') {
                pos++;
                break;
            }
            pos++;
        }
        const size_t colon = raw.find(':', pos);
        if (colon == std::string::npos) {
            break;
        }
        const size_t value_start = raw.find_first_not_of(" \t\r\n", colon + 1);
        const size_t value_end = json_value_end(raw, value_start);
        if (value_start != std::string::npos && value_end != std::string::npos
            && value_start < raw.size() && raw[value_start] == '{') {
            slots[key] = normalize_slot(raw.substr(value_start, value_end - value_start));
        }
        if (value_end == std::string::npos) {
            break;
        }
        pos = value_end;
    }
    return slots;
}

std::vector<std::string> parse_slot_map_order_json(const std::string &raw_json) {
    std::vector<std::string> first_seen;
    std::map<std::string, std::string> last_values;
    const std::string raw = trim(raw_json);
    if (raw.size() < 2 || raw.front() != '{') {
        return first_seen;
    }
    for (const auto &entry : json_object_ordered_entries(raw)) {
        if (std::find(first_seen.begin(), first_seen.end(), entry.first) == first_seen.end()) {
            first_seen.push_back(entry.first);
        }
        last_values[entry.first] = entry.second;
    }
    std::vector<std::string> ordered;
    for (const auto &key : first_seen) {
        const auto found = last_values.find(key);
        if (found == last_values.end()) {
            continue;
        }
        const std::string value = trim(found->second);
        if (!value.empty() && value.front() == '{') {
            ordered.push_back(key);
        }
    }
    return ordered;
}

std::string slot_json(const RouterSlotState &slot, bool clamp_instances = true) {
    RouterSlotState normalized = normalize_slot(slot, slot.capacity);
    if (!clamp_instances) normalized.instances = slot.instances;
    std::ostringstream out;
    out << "{"
        << "\"capacity\":" << normalized.capacity << ","
        << "\"in_flight\":" << normalized.in_flight << ","
        << "\"queued\":" << normalized.queued << ","
        << "\"available\":" << normalized.available;
    if (normalized.instances >= 0) out << ",\"instances\":" << normalized.instances;
    out << "}";
    return out.str();
}

std::string slot_map_json(const std::map<std::string, RouterSlotState> &slots,
                          const std::vector<std::string> &order = {}, bool clamp_instances = true) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    std::set<std::string> emitted;
    for (const auto &key : order) {
        const auto found = slots.find(key);
        if (found == slots.end() || emitted.find(key) != emitted.end()) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        emitted.insert(key);
        out << "\"" << json_escape(found->first) << "\":" << slot_json(found->second, clamp_instances);
    }
    for (const auto &entry : slots) {
        if (emitted.find(entry.first) != emitted.end()) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << json_escape(entry.first) << "\":" << slot_json(entry.second, clamp_instances);
    }
    out << "}";
    return out.str();
}

bool vector_contains(const std::vector<std::string> &values, const std::string &needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

bool is_model_strict_capability(const std::string &capability) {
    return capability == "text" || capability == "vision" || capability == "embedding";
}

bool is_tunnel_url(const std::string &url) {
    return starts_with(url, "tunnel://");
}

std::string tunnel_worker_id(const std::string &url) {
    return is_tunnel_url(url) ? url.substr(std::string("tunnel://").size()) : "";
}

struct NativeMemoryInfo {
    double total_ram_gb = 0.0;
    double free_ram_gb = 0.0;
};

struct NativeGpuInfo {
    int gpu_count = 0;
    int total_vram_gb = 0;
    double free_vram_gb = 0.0;
    std::vector<double> free_per_gpu_gb;
};

double meminfo_kb_value(const std::string &key) {
    std::ifstream in("/proc/meminfo");
    std::string name;
    double kb = 0.0;
    std::string unit;
    while (in >> name >> kb >> unit) {
        if (!name.empty() && name.back() == ':') {
            name.pop_back();
        }
        if (name == key) {
            return kb;
        }
    }
    return 0.0;
}

NativeMemoryInfo read_memory_info() {
    NativeMemoryInfo info;
    const double total_kb = meminfo_kb_value("MemTotal");
    const double available_kb = meminfo_kb_value("MemAvailable");
    info.total_ram_gb = total_kb > 0.0 ? total_kb / (1024.0 * 1024.0) : 128.0;
    info.free_ram_gb = available_kb > 0.0 ? available_kb / (1024.0 * 1024.0) : 0.0;
    return info;
}

bool parse_nvidia_smi_memory_line(const std::string &line, double *total_mib, double *free_mib) {
    const std::string value = trim(line);
    if (value.empty()) {
        return false;
    }
    const auto comma = value.find(',');
    if (comma == std::string::npos) {
        return false;
    }
    try {
        size_t total_pos = 0;
        size_t free_pos = 0;
        const double parsed_total = std::stod(trim(value.substr(0, comma)), &total_pos);
        const double parsed_free = std::stod(trim(value.substr(comma + 1)), &free_pos);
        if (parsed_total <= 0.0 || parsed_free < 0.0) {
            return false;
        }
        *total_mib = parsed_total;
        *free_mib = parsed_free;
        return true;
    } catch (...) {
        return false;
    }
}

std::string nvidia_smi_output(const std::string &query) {
    try {
        return run_process({"nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"}, true, true).output;
    } catch (const std::system_error &) { return {}; }
}

NativeGpuInfo read_gpu_info() {
    NativeGpuInfo info;
    std::istringstream lines(nvidia_smi_output("memory.total,memory.free"));
    for (std::string line; std::getline(lines, line);) {
        double total_mib = 0.0;
        double free_mib = 0.0;
        if (!parse_nvidia_smi_memory_line(line, &total_mib, &free_mib)) {
            continue;
        }
        ++info.gpu_count;
        info.total_vram_gb += static_cast<int>(std::floor(total_mib / 1024.0));
        const double free_gb = free_mib / 1024.0;
        info.free_vram_gb += free_gb;
        info.free_per_gpu_gb.push_back(free_gb);
    }
    return info;
}

std::string json_number_array(const std::vector<double> &values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << json_number_literal(values[i]);
    }
    out << "]";
    return out.str();
}

double parse_double_env(const std::string &name, double fallback) {
    try {
        return std::stod(ezl_getenv_default(name.c_str()));
    } catch (...) {
        return fallback;
    }
}

double parse_double_env_explicit(const std::string &name, double fallback) {
    const char *raw = getenv(name.c_str());
    if (!raw || !*raw) {
        return fallback;
    }
    try {
        return std::stod(raw);
    } catch (...) {
        return fallback;
    }
}

bool parse_python_float_digits(const std::string &value, size_t *pos, std::string *normalized, bool *had_digits) {
    bool last_was_digit = false;
    while (*pos < value.size()) {
        const unsigned char ch = static_cast<unsigned char>(value[*pos]);
        if (std::isdigit(ch)) {
            normalized->push_back(static_cast<char>(ch));
            *had_digits = true;
            last_was_digit = true;
            ++(*pos);
            continue;
        }
        if (value[*pos] == '_') {
            if (!last_was_digit || *pos + 1 >= value.size()
                || !std::isdigit(static_cast<unsigned char>(value[*pos + 1]))) {
                return false;
            }
            last_was_digit = false;
            ++(*pos);
            continue;
        }
        break;
    }
    return true;
}

bool parse_strict_python_float_string(const std::string &raw, double *out) {
    const std::string value = trim(raw);
    if (value.empty()) {
        return false;
    }
    std::string normalized;
    normalized.reserve(value.size());
    size_t pos = 0;
    if (value[pos] == '+' || value[pos] == '-') {
        normalized.push_back(value[pos]);
        ++pos;
        if (pos >= value.size()) {
            return false;
        }
    }
    const std::string rest_lower = to_lower(value.substr(pos));
    if (rest_lower == "nan" || rest_lower == "inf" || rest_lower == "infinity") {
        normalized += rest_lower;
        char *end = nullptr;
        *out = std::strtod(normalized.c_str(), &end);
        return end && *end == '\0';
    }

    bool digits_before = false;
    if (!parse_python_float_digits(value, &pos, &normalized, &digits_before)) {
        return false;
    }
    bool digits_after = false;
    if (pos < value.size() && value[pos] == '.') {
        normalized.push_back('.');
        ++pos;
        if (!parse_python_float_digits(value, &pos, &normalized, &digits_after)) {
            return false;
        }
    }
    if (!digits_before && !digits_after) {
        return false;
    }
    if (pos < value.size() && (value[pos] == 'e' || value[pos] == 'E')) {
        normalized.push_back('e');
        ++pos;
        if (pos < value.size() && (value[pos] == '+' || value[pos] == '-')) {
            normalized.push_back(value[pos]);
            ++pos;
        }
        bool exponent_digits = false;
        if (!parse_python_float_digits(value, &pos, &normalized, &exponent_digits) || !exponent_digits) {
            return false;
        }
    }
    if (pos != value.size()) {
        return false;
    }
    char *end = nullptr;
    *out = std::strtod(normalized.c_str(), &end);
    return end && *end == '\0';
}

bool parse_strict_int_string(const std::string &raw, int *out) {
    return parse_python_int_string(raw, out);
}

// TTL bounds and positive worker counts only need exact values through 300.
// Keep parsing after saturation to validate the entire integer and its length.
bool parse_python_int_saturated(const std::string &raw, int *out, size_t *digit_count) {
    for (const unsigned char ch : raw) if (ch >= 0x1c && ch <= 0x1f) return false;
    const auto value = strip_config_value(raw);
    if (value.empty()) return false;
    // Unicode 15.0 decimal blocks, matching the reference Python runtime.
    static constexpr uint32_t zeroes[] = {
        0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6,
        0xc66, 0xce6, 0xd66, 0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090,
        0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0,
        0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50,
        0xabf0, 0xff10, 0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136,
        0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x11730,
        0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60,
        0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6,
        0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0
    };
    size_t pos = 0;
    const bool negative = value.front() == '-';
    if (value.front() == '-' || value.front() == '+') ++pos;
    bool previous_digit = false;
    int number = 0;
    *digit_count = 0;
    while (pos < value.size()) {
        const auto first = static_cast<unsigned char>(value[pos++]);
        if (first == '_') {
            if (!previous_digit) return false;
            previous_digit = false;
            continue;
        }
        uint32_t cp = first;
        if (first >= 0x80) {
            const unsigned count = first >= 0xc2 && first <= 0xdf ? 1 : first >= 0xe0 && first <= 0xef ? 2 : first >= 0xf0 && first <= 0xf4 ? 3 : 0;
            if (!count || pos + count > value.size()) return false;
            cp = first & (0x7f >> count);
            for (unsigned i = 0; i < count; ++i) {
                const auto byte = static_cast<unsigned char>(value[pos++]);
                if ((byte & 0xc0) != 0x80) return false;
                cp = (cp << 6) | (byte & 0x3f);
            }
            if ((count == 1 && cp < 0x80) || (count == 2 && cp < 0x800) || (count == 3 && cp < 0x10000)) return false;
        }
        int digit = -1;
        for (const auto zero : zeroes) if (cp >= zero && cp < zero + 10) { digit = static_cast<int>(cp - zero); break; }
        if (digit < 0) return false;
        number = std::min(301, number * 10 + digit);
        ++*digit_count;
        previous_digit = true;
    }
    if (!previous_digit) return false;
    *out = negative ? -number : number;
    return true;
}

bool positive_python_integer(const std::string &raw) {
    const auto value = trim(raw);
    if (value == "true") return true;
    if (value.empty() || value == "false" || value == "null") return false;
    const bool quoted = value.front() == '"';
    if (quoted || (is_json_number_token(value) && value.find_first_of(".eE") == std::string::npos)) {
        int integer = 0; size_t digits = 0;
        return parse_python_int_saturated(quoted ? parse_json_string_at(value, 0) : value, &integer, &digits)
            && digits <= 4300 && integer > 0;
    }
    if (!is_json_number_token(value)) return false;
    try { const double number = std::stod(value); return std::isfinite(number) && number >= 1.0; }
    catch (...) { return false; }
}

int parse_first_csv_int_env(const std::string &name, int fallback) {
    std::string raw = ezl_getenv_default(name.c_str());
    const auto comma = raw.find(',');
    if (comma != std::string::npos) {
        raw = raw.substr(0, comma);
    }
    int parsed = 0;
    return parse_strict_int_string(raw, &parsed) ? parsed : fallback;
}

std::vector<std::string> configured_model_names(std::string raw) {
    std::vector<std::string> models;
    if (to_lower(trim(raw)) == "none") {
        return models;
    }
    size_t start = 0;
    std::map<std::string, int> counts;
    while (start <= raw.size()) {
        const auto comma = raw.find(',', start);
        std::string model = trim(raw.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        const auto at = model.rfind('@');
        if (at != std::string::npos) {
            model = model.substr(0, at);
        }
        if (!model.empty()) {
            const int count = ++counts[model];
            models.push_back(count == 1 ? model : model + "#" + std::to_string(count));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return models;
}

std::string source_model_name(std::string model_id) {
    const auto replica = model_id.rfind('#');
    if (replica != std::string::npos) {
        bool numeric_suffix = replica + 1 < model_id.size();
        for (size_t i = replica + 1; i < model_id.size(); ++i) {
            numeric_suffix = numeric_suffix && std::isdigit(static_cast<unsigned char>(model_id[i]));
        }
        if (numeric_suffix) {
            model_id = model_id.substr(0, replica);
        }
    }
    return model_id;
}

std::vector<std::string> configured_public_model_names(const std::string &raw) {
    std::vector<std::string> public_models;
    std::set<std::string> seen;
    for (const auto &model_id : configured_model_names(raw)) {
        const std::string source = source_model_name(model_id);
        if (!source.empty() && seen.insert(source).second) {
            public_models.push_back(source);
        }
    }
    return public_models;
}

bool model_has_local_vision_projector(const std::string &model_name) {
    const auto slash = model_name.find_last_of('/');
    if (slash == std::string::npos) {
        return false;
    }
    std::string short_name = model_name.substr(slash + 1);
    const auto gguf = short_name.find("-GGUF");
    if (gguf != std::string::npos) {
        short_name = short_name.substr(0, gguf);
    }
    if (short_name.empty()) {
        return false;
    }
    const std::string local_dir = "models/" + short_name;
    DIR *dir = opendir(local_dir.c_str());
    if (!dir) {
        return false;
    }
    bool found = false;
    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        const std::string name = entry->d_name;
        if (to_lower(name).find("mmproj") != std::string::npos
            && name.size() >= 5
            && name.substr(name.size() - 5) == ".gguf") {
            found = true;
            break;
        }
    }
    closedir(dir);
    return found;
}

std::vector<int> configured_parallel_values(size_t model_count) {
    std::vector<int> values;
    ezl_load_dotenv();
    const char *n_parallel_env = std::getenv("N_PARALLEL");
    const char *max_concurrent_env = std::getenv("MAX_CONCURRENT_REQUESTS");
    std::string raw = n_parallel_env && n_parallel_env[0] != '\0'
        ? n_parallel_env
        : (max_concurrent_env && max_concurrent_env[0] != '\0'
            ? max_concurrent_env
            : ezl_getenv_default("N_PARALLEL"));
    size_t start = 0;
    while (start <= raw.size()) {
        const auto comma = raw.find(',', start);
        std::string value = trim(raw.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        try {
            values.push_back(std::max(1, std::stoi(value)));
        } catch (...) {
            values.push_back(1);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (values.empty()) {
        values.push_back(1);
    }
    while (values.size() < model_count) {
        values.push_back(values.back());
    }
    return values;
}

bool configured_forward_url(const std::string &env_name) {
    const std::string value = to_lower(trim(ezl_getenv_default(env_name.c_str())));
    return !value.empty() && value != "true" && value != "none";
}

bool configured_enabled(const std::string &env_name) {
    std::string value = ezl_getenv_default(env_name.c_str());
    if (value.empty() && (env_name == "TTS_ENABLED" || env_name == "STT_ENABLED" || env_name == "EMBEDDING_ENABLED")) value = "true";
    return to_lower(strip_config_value(value)) == "true";
}

int speech_parallel_slots(const char *name) {
    try {
        const auto value = llm_parameter_integer(ezl_getenv_fallback(name, "1"));
        if (value.empty() || value.front() == '-') return 1;
        try { return static_cast<int>(std::clamp<int64_t>(std::stoll(value), 1, INT_MAX)); }
        catch (const std::out_of_range &) { return INT_MAX; }
    } catch (const std::exception &) { return 1; }
}

bool has_local_capability(const std::string &capability) {
    const auto caps = local_capabilities();
    return std::find(caps.begin(), caps.end(), capability) != caps.end();
}

bool env_false(const std::string &env_name) {
    const std::string value = to_lower(trim(ezl_getenv_default(env_name.c_str())));
    return value == "0" || value == "false" || value == "no";
}

bool env_true(const std::string &env_name) {
    const std::string value = to_lower(trim(ezl_getenv_default(env_name.c_str())));
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string short_random_hex() {
    std::string id = uuid_v4();
    id.erase(std::remove(id.begin(), id.end(), '-'), id.end());
    return id.substr(0, std::min<size_t>(8, id.size()));
}

std::string local_hostname() {
    char buffer[256] = {0};
    if (gethostname(buffer, sizeof(buffer) - 1) == 0 && buffer[0] != '\0') {
        return buffer;
    }
    return "ezlocalai";
}

std::vector<std::string> native_worker_capabilities() {
    return local_capabilities();
}

int gpu_tier_for_name(const std::string &name) {
    return ezl_gpu_tier_for_name(name.c_str());
}

std::string cpu_name() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        const std::string lowered = to_lower(line);
        if (starts_with(lowered, "model name") || starts_with(lowered, "hardware")) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                const std::string value = trim(line.substr(colon + 1));
                if (!value.empty()) {
                    return value;
                }
            }
        }
    }
    return local_hostname();
}

double python_round2(double value) {
    // Python round(x, 2) rounds the exact binary value half-to-even; glibc's
    // correctly rounded printf conversion makes the same choice.
    char text[64];
    std::snprintf(text, sizeof(text), "%.2f", value);
    return std::strtod(text, nullptr);
}

std::string native_worker_gpu_json(int index, const std::string &name, double total_gb) {
    std::ostringstream gpu;
    gpu << "{\"index\":" << index
        << ",\"name\":\"" << json_escape(name) << "\""
        << ",\"total_vram_gb\":" << json_number_literal(python_round2(total_gb))
        << ",\"tier\":" << gpu_tier_for_name(name)
        << ",\"backend\":\"" << gpu_backend_name() << "\"}";
    return gpu.str();
}

std::string native_worker_gpus_json() {
    std::vector<std::string> values;
    // Match torch's process-visible ordinals, including an intentionally empty
    // CUDA_VISIBLE_DEVICES. Physical nvidia-smi devices are not usable fallback
    // capacity. A failed properties probe retains a generic entry like Python.
    for (int ordinal = 0, count = visible_cuda_device_count(); ordinal < count; ++ordinal) {
        std::string name = gpu_backend_name() + ":" + std::to_string(ordinal);
        double total_gb = 0;
        try {
            const auto device = cuda_device_properties(ordinal);
            name = device.name;
            total_gb = static_cast<double>(device.total_memory) / (1024.0 * 1024.0 * 1024.0);
        } catch (const std::exception &) {}
        values.push_back(native_worker_gpu_json(ordinal, name, total_gb));
    }

    std::ostringstream cpu;
    cpu << "{\"index\":-1,\"name\":\"" << json_escape(cpu_name())
        << "\",\"total_vram_gb\":0.0,\"tier\":2,\"backend\":\"cpu\"}";
    values.push_back(cpu.str());
    return json_array_from_values(values);
}

std::string native_worker_model_context_json(const LlmCatalog &catalog, const InferenceBackend &backend) {
    std::map<std::string, int64_t> configured;
    for (const auto &id : catalog.available) {
        const auto &config = catalog.configs.at(id);
        const auto context = config.context / std::max<int64_t>(1, config.parallel);
        if (context > 0) configured[catalog.source(id)] = std::max(configured[catalog.source(id)], context);
    }
    // Loaded instances can have a smaller context after a memory-fit retry.
    // The source uses live values first and configured values only as fallback.
    const auto loaded = backend.loaded_model_contexts();
    for (const auto &entry : loaded) if (entry.second > 0) configured[entry.first] = entry.second;
    const auto delegated = to_lower(trim(ezl_getenv_default("EMBEDDING_SERVER")));
    const auto model = trim(ezl_getenv_default("EMBEDDING_MODEL"));
    if (configured_enabled("EMBEDDING_ENABLED") && !model.empty()
        && !starts_with(delegated, "http://") && !starts_with(delegated, "https://")) {
        int context = 0;
        if (parse_strict_int_string(ezl_getenv_default("EMBEDDING_CONTEXT_LENGTH"), &context)) {
            configured.emplace(model, context);
            const auto alias = trim(ezl_getenv_default("EMBEDDING_MODEL_ALIAS"));
            if (!alias.empty()) configured.emplace(alias, context);
        }
    }
    std::map<std::string, std::string> values;
    for (const auto &entry : configured) values[entry.first] = std::to_string(entry.second);
    return json_object_from_values(values);
}

std::string native_worker_model_quant_json(const InferenceBackend &backend) {
    std::map<std::string, std::string> values;
    for (const auto &entry : worker_model_quants(backend.loaded_model_quants()))
        values[entry.first] = "\"" + json_escape(entry.second) + "\"";
    return json_object_from_values(values);
}

std::string native_worker_cap_models_json(const std::vector<std::string> &caps) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (const auto &entry : capability_model_names(caps)) {
        if (!first) out << ",";
        first = false;
        out << "\"" << json_escape(entry.first) << "\":\"" << json_escape(entry.second) << "\"";
    }
    out << "}";
    return out.str();
}

bool router_proxy_timeout_seconds(const char *fallback_seconds, int *out) {
    if (!out) {
        return false;
    }
    const char *raw = getenv("REQUEST_TIMEOUT");
    const std::string value = raw ? raw : fallback_seconds;
    double parsed = 0.0;
    if (!parse_strict_python_float_string(value, &parsed)) {
        return false;
    }
    if (parsed <= 0.0) {
        *out = 0;
        return true;
    }
    const double capped = std::min<double>(std::ceil(parsed), std::numeric_limits<int>::max());
    *out = static_cast<int>(std::max<double>(1.0, capped));
    return true;
}

std::string request_base_url_from_host(const std::string &host_header) {
    std::string host = trim(host_header);
    while (!host.empty() && host.back() == '/') {
        host.pop_back();
    }
    if (host.empty()) {
        return "";
    }
    const std::string lowered = to_lower(host);
    if (starts_with(lowered, "http://") || starts_with(lowered, "https://")) {
        return host;
    }
    return "http://" + host;
}

std::string outputs_dir() {
    std::string configured = trim(ezl_getenv_default("ROUTER_OUTPUTS_DIR"));
    return configured.empty() ? "outputs" : configured;
}

bool safe_output_relative_path(const std::string &worker_path, std::string *relative_out) {
    const std::string prefix = "/outputs/";
    if (!starts_with(worker_path, prefix)) {
        return false;
    }
    const std::string decoded = url_decode(worker_path.substr(prefix.size()));
    if (decoded.empty() || decoded.find('\0') != std::string::npos) {
        return false;
    }
    std::filesystem::path raw(decoded);
    if (raw.is_absolute()) {
        return false;
    }
    const auto normalized = raw.lexically_normal();
    if (normalized.empty() || normalized == ".") {
        return false;
    }
    for (const auto &part : normalized) {
        if (part == "..") {
            return false;
        }
    }
    if (relative_out) {
        *relative_out = normalized.generic_string();
    }
    return true;
}

std::string content_type_for_path(const std::string &path) {
    const std::string lower = to_lower(path);
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".txt") {
        return "text/plain; charset=utf-8";
    }
    if (lower.size() >= 5 && lower.substr(lower.size() - 5) == ".html") {
        return "text/html; charset=utf-8";
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".css") {
        return "text/css; charset=utf-8";
    }
    if (lower.size() >= 3 && lower.substr(lower.size() - 3) == ".js") {
        return "text/javascript; charset=utf-8";
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".png") {
        return "image/png";
    }
    if ((lower.size() >= 4 && lower.substr(lower.size() - 4) == ".jpg")
        || (lower.size() >= 5 && lower.substr(lower.size() - 5) == ".jpeg")) {
        return "image/jpeg";
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".svg") {
        return "image/svg+xml";
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".wav") {
        return "audio/x-wav";
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".mp3") {
        return "audio/mpeg";
    }
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".mp4") {
        return "video/mp4";
    }
    if (lower.size() >= 5 && lower.substr(lower.size() - 5) == ".webm") {
        return "video/webm";
    }
    if (lower.size() >= 5 && lower.substr(lower.size() - 5) == ".json") {
        return "application/json";
    }
    return "application/octet-stream";
}

uint32_t md5_left_rotate(uint32_t value, uint32_t count) {
    return (value << count) | (value >> (32U - count));
}

std::string md5_hex(const std::string &input) {
    static const uint32_t shifts[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    };
    static const uint32_t constants[64] = {
        0xd76aa478U, 0xe8c7b756U, 0x242070dbU, 0xc1bdceeeU,
        0xf57c0fafU, 0x4787c62aU, 0xa8304613U, 0xfd469501U,
        0x698098d8U, 0x8b44f7afU, 0xffff5bb1U, 0x895cd7beU,
        0x6b901122U, 0xfd987193U, 0xa679438eU, 0x49b40821U,
        0xf61e2562U, 0xc040b340U, 0x265e5a51U, 0xe9b6c7aaU,
        0xd62f105dU, 0x02441453U, 0xd8a1e681U, 0xe7d3fbc8U,
        0x21e1cde6U, 0xc33707d6U, 0xf4d50d87U, 0x455a14edU,
        0xa9e3e905U, 0xfcefa3f8U, 0x676f02d9U, 0x8d2a4c8aU,
        0xfffa3942U, 0x8771f681U, 0x6d9d6122U, 0xfde5380cU,
        0xa4beea44U, 0x4bdecfa9U, 0xf6bb4b60U, 0xbebfbc70U,
        0x289b7ec6U, 0xeaa127faU, 0xd4ef3085U, 0x04881d05U,
        0xd9d4d039U, 0xe6db99e5U, 0x1fa27cf8U, 0xc4ac5665U,
        0xf4292244U, 0x432aff97U, 0xab9423a7U, 0xfc93a039U,
        0x655b59c3U, 0x8f0ccc92U, 0xffeff47dU, 0x85845dd1U,
        0x6fa87e4fU, 0xfe2ce6e0U, 0xa3014314U, 0x4e0811a1U,
        0xf7537e82U, 0xbd3af235U, 0x2ad7d2bbU, 0xeb86d391U,
    };

    std::vector<unsigned char> msg(input.begin(), input.end());
    const uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8ULL;
    msg.push_back(0x80U);
    while ((msg.size() % 64) != 56) {
        msg.push_back(0x00U);
    }
    for (int i = 0; i < 8; ++i) {
        msg.push_back(static_cast<unsigned char>((bit_len >> (8 * i)) & 0xffU));
    }

    uint32_t a0 = 0x67452301U;
    uint32_t b0 = 0xefcdab89U;
    uint32_t c0 = 0x98badcfeU;
    uint32_t d0 = 0x10325476U;

    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t m[16]{};
        for (int i = 0; i < 16; ++i) {
            const size_t j = offset + static_cast<size_t>(i) * 4;
            m[i] = static_cast<uint32_t>(msg[j])
                | (static_cast<uint32_t>(msg[j + 1]) << 8)
                | (static_cast<uint32_t>(msg[j + 2]) << 16)
                | (static_cast<uint32_t>(msg[j + 3]) << 24);
        }

        uint32_t a = a0;
        uint32_t b = b0;
        uint32_t c = c0;
        uint32_t d = d0;
        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t f = 0;
            uint32_t g = 0;
            if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3 * i + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7 * i) % 16;
            }
            const uint32_t next = d;
            d = c;
            c = b;
            b = b + md5_left_rotate(a + f + constants[i] + m[g], shifts[i]);
            a = next;
        }
        a0 += a;
        b0 += b;
        c0 += c;
        d0 += d;
    }

    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setfill('0');
    const uint32_t words[4] = {a0, b0, c0, d0};
    for (uint32_t word : words) {
        for (int i = 0; i < 4; ++i) {
            out << std::setw(2) << static_cast<int>((word >> (8 * i)) & 0xffU);
        }
    }
    return out.str();
}

double stat_mtime_float(const struct stat &st) {
#if defined(__APPLE__)
    return static_cast<double>(st.st_mtimespec.tv_sec)
        + static_cast<double>(st.st_mtimespec.tv_nsec) / 1000000000.0;
#else
    return static_cast<double>(st.st_mtim.tv_sec)
        + static_cast<double>(st.st_mtim.tv_nsec) / 1000000000.0;
#endif
}

time_t stat_mtime_seconds(const struct stat &st) {
#if defined(__APPLE__)
    return st.st_mtimespec.tv_sec;
#else
    return st.st_mtim.tv_sec;
#endif
}

std::string http_date(time_t value) {
    std::tm tm{};
    gmtime_r(&value, &tm);
    char buffer[64]{};
    std::strftime(buffer, sizeof(buffer), "%a, %d %b %Y %H:%M:%S GMT", &tm);
    return buffer;
}

bool parse_http_date(const std::string &value, time_t *out) {
    // HTTP recipients accept IMF-fixdate and the two obsolete HTTP date
    // formats; the Python StaticFiles validator accepts all three.
    for (const auto *format : {"%a, %d %b %Y %H:%M:%S GMT",
                               "%A, %d-%b-%y %H:%M:%S GMT",
                               "%a %b %d %H:%M:%S %Y"}) {
        std::tm tm{};
        char *end = strptime(value.c_str(), format, &tm);
        if (end && *end == '\0') {
            *out = timegm(&tm);
            return true;
        }
    }
    return false;
}

std::string static_file_etag(const struct stat &st) {
    const std::string base = python_float_string(stat_mtime_float(st))
        + "-"
        + std::to_string(static_cast<long long>(st.st_size));
    return "\"" + md5_hex(base) + "\"";
}

std::string strip_starlette_if_none_match_tag(std::string tag) {
    tag = trim(tag);
    if (tag.rfind("W/", 0) == 0) tag.erase(0, 2);
    return tag;
}

std::string request_header_value(
    const std::unordered_map<std::string, std::string> &headers,
    const std::string &name
) {
    const auto it = headers.find(to_lower(name));
    return it == headers.end() ? "" : it->second;
}

bool static_file_not_modified(
    const std::unordered_map<std::string, std::string> &request_headers,
    const std::string &etag,
    const std::string &last_modified
) {
    const std::string if_none_match = request_header_value(request_headers, "if-none-match");
    if (!if_none_match.empty()) {
        if (trim(if_none_match) == "*") return true;
        size_t start = 0;
        while (start <= if_none_match.size()) {
            const size_t comma = if_none_match.find(',', start);
            const std::string tag = if_none_match.substr(
                start,
                comma == std::string::npos ? std::string::npos : comma - start
            );
            if (strip_starlette_if_none_match_tag(tag) == etag) {
                return true;
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        return false;
    }

    const std::string if_modified_since = request_header_value(request_headers, "if-modified-since");
    if (!if_modified_since.empty()) {
        time_t request_time = 0;
        time_t response_time = 0;
        if (parse_http_date(if_modified_since, &request_time)
            && parse_http_date(last_modified, &response_time)
            && request_time >= response_time) {
            return true;
        }
    }
    return false;
}

struct StaticRangeParse {
    bool has_range = false;
    bool ok = false;
    bool satisfiable = true;
    size_t start = 0;
    size_t end = 0;
    std::vector<std::pair<size_t, size_t>> ranges;
    std::string error = "Range header: range must be requested";
};

StaticRangeParse parse_static_range(const std::string &header, size_t file_size) {
    StaticRangeParse parsed{};
    parsed.has_range = true;
    parsed.error = "Malformed range header.";
    const auto equals = header.find('=');
    if (equals == std::string::npos)
        return parsed;
    if (to_lower(trim(header.substr(0, equals))) != "bytes") {
        parsed.error = "Only support bytes range";
        return parsed;
    }
    const auto range = header.substr(equals + 1);
    // The reference ignores excessive ranges rather than allocating an
    // arbitrarily large multipart response.
    if (std::count(range.begin(), range.end(), ',') >= 100) {
        parsed.ok = true;
        return parsed;
    }
    parsed.error = "Range header: range must be requested";
    const auto clamp = [&](const std::string &number) {
        if (configuration_integer_less(number, "0"))
            return size_t(0);
        if (!configuration_integer_less(number, std::to_string(file_size)))
            return file_size;
        return static_cast<size_t>(std::stoull(number));
    };
    size_t position = 0;
    while (position <= range.size()) {
        const auto comma = range.find(',', position);
        const auto part =
            trim(range.substr(position, comma == std::string::npos ? std::string::npos : comma - position));
        const auto dash = part.find('-');
        if (!part.empty() && part != "-" && dash != std::string::npos) {
            const auto first = trim(part.substr(0, dash)), last = trim(part.substr(dash + 1));
            const auto start_number =
                first.empty() ? std::optional<std::string>{"0"} : parse_configuration_integer(first);
            const auto end_number =
                last.empty() ? std::optional<std::string>{"0"} : parse_configuration_integer(last);
            if (start_number && end_number && (!first.empty() || !last.empty())) {
                size_t start = 0, end = file_size;
                if (first.empty()) {
                    start = configuration_integer_less(*end_number, "0") ? file_size
                                                                         : file_size - clamp(*end_number);
                } else {
                    start = clamp(*start_number);
                    if (!last.empty() && configuration_integer_less(*end_number, std::to_string(file_size)))
                        end = configuration_integer_less(*end_number, "0") ? 0 : clamp(*end_number) + 1;
                }
                parsed.ranges.emplace_back(start, end);
            }
        }
        if (comma == std::string::npos)
            break;
        position = comma + 1;
    }
    if (parsed.ranges.empty())
        return parsed;
    for (const auto &range : parsed.ranges)
        if (range.first >= file_size) {
            parsed.satisfiable = false;
            return parsed;
        }
    for (const auto &range : parsed.ranges)
        if (range.first >= range.second) {
            parsed.error = "Range header: start must be less than end";
            return parsed;
        }
    std::sort(parsed.ranges.begin(), parsed.ranges.end());
    std::vector<std::pair<size_t, size_t>> merged;
    for (const auto &range : parsed.ranges) {
        if (merged.empty() || range.first > merged.back().second)
            merged.push_back(range);
        else
            merged.back().second = std::max(merged.back().second, range.second);
    }
    parsed.ranges = std::move(merged);
    parsed.ok = true;
    parsed.start = parsed.ranges.front().first;
    parsed.end = parsed.ranges.front().second;
    return parsed;
}

std::string random_multipart_boundary() {
    static const char alphabet[] = "0123456789abcdef";
    std::random_device rd;
    std::string boundary;
    boundary.reserve(26);
    for (int i = 0; i < 26; ++i) {
        boundary.push_back(alphabet[rd() % 16]);
    }
    return boundary;
}

std::string url_encode_component(const std::string &value);
HttpResponse serve_file_response(const std::string &path,
                                 const std::unordered_map<std::string, std::string> &headers,
                                 const std::string &media_type, const std::string &filename, bool conditional,
                                 bool deferred) {
    // Keep the opened inode alive through transmission, and read only requested
    // slices. Large output files never need a whole-file response buffer.
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    FILE *opened = descriptor < 0 ? nullptr : fdopen(descriptor, "rb");
    if (!opened && descriptor >= 0)
        close(descriptor);
    const auto file = std::shared_ptr<FILE>(opened, [](FILE *f) {
        if (f)
            std::fclose(f);
    });
    struct stat st {};
    if (!file || fstat(fileno(file.get()), &st) != 0 || !S_ISREG(st.st_mode))
        return conditional ? HttpResponse{404, "application/json", "{\"detail\":\"Not Found\"}"}
                           : HttpResponse{500, "text/plain; charset=utf-8", "Internal Server Error"};
    const auto size = static_cast<size_t>(st.st_size);
    const auto modified = http_date(stat_mtime_seconds(st)), etag = static_file_etag(st);
    if (conditional && static_file_not_modified(headers, etag, modified)) {
        HttpResponse response{304, "", "", {{"ETag", etag}}};
        response.header_order = HttpResponse::HeaderOrder::StaticFile;
        return response;
    }
    HttpResponse response{200, media_type, "", {{"Accept-Ranges", "bytes"}}};
    response.header_order = HttpResponse::HeaderOrder::StaticFile;
    if (!filename.empty()) {
        const auto quoted = url_encode_component(filename);
        response.headers.push_back({"Content-Disposition", quoted == filename
                                                               ? "attachment; filename=\"" + filename + "\""
                                                               : "attachment; filename*=utf-8''" + quoted});
    }
    response.headers.push_back({"Last-Modified", modified});
    response.headers.push_back({"ETag", etag});
    std::vector<std::pair<size_t, size_t>> ranges{{0, size}};
    const auto range = headers.find("range"), if_range = headers.find("if-range");
    if (range != headers.end() &&
        (if_range == headers.end() || if_range->second == modified || if_range->second == etag)) {
        const auto parsed = parse_static_range(range->second, size);
        if (!parsed.satisfiable) {
            HttpResponse error{
                416, "text/plain; charset=utf-8", "", {{"Content-Range", "bytes */" + std::to_string(size)}}};
            error.header_order = HttpResponse::HeaderOrder::StaticUnsatisfiable;
            return error;
        }
        if (!parsed.ok) {
            HttpResponse error{400, "text/plain; charset=utf-8", parsed.error};
            error.header_order = HttpResponse::HeaderOrder::StaticRangeError;
            return error;
        }
        if (!parsed.ranges.empty()) {
            ranges = parsed.ranges;
            response.status = 206;
        }
    }
    std::vector<std::string> prefixes(ranges.size());
    std::string ending;
    size_t length = 0;
    if (ranges.size() > 1) {
        const auto boundary = random_multipart_boundary();
        response.content_type = "multipart/byteranges; boundary=" + boundary;
        for (size_t i = 0; i < ranges.size(); ++i) {
            const auto [start, end] = ranges[i];
            prefixes[i] = "--" + boundary + "\r\nContent-Type: " + media_type + "\r\nContent-Range: bytes " +
                          std::to_string(start) + "-" + std::to_string(end - 1) + "/" + std::to_string(size) +
                          "\r\n\r\n";
            length += prefixes[i].size() + end - start + 2;
        }
        ending = "--" + boundary + "--";
        length += ending.size();
    } else {
        const auto [start, end] = ranges.front();
        length = end - start;
        if (response.status == 206)
            response.headers.push_back({"Content-Range", "bytes " + std::to_string(start) + "-" +
                                                             std::to_string(end - 1) + "/" +
                                                             std::to_string(size)});
    }
    response.content_length = length;
    const auto transfer = [file, ranges, prefixes,
                           ending](const std::function<bool(const char *, size_t)> &emit) {
        std::array<char, 65536> buffer{};
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (!emit(prefixes[i].data(), prefixes[i].size()))
                return false;
            auto offset = ranges[i].first;
            while (offset < ranges[i].second) {
                const auto n = pread(fileno(file.get()), buffer.data(),
                                     std::min(buffer.size(), ranges[i].second - offset), offset);
                if (n < 0 && errno == EINTR)
                    continue;
                if (n <= 0 || !emit(buffer.data(), static_cast<size_t>(n)))
                    return false;
                offset += static_cast<size_t>(n);
            }
            if (ranges.size() > 1 && !emit("\r\n", 2))
                return false;
        }
        return emit(ending.data(), ending.size());
    };
    if (deferred) {
        response.stream_writer = [transfer](int fd) {
            transfer([fd](const char *data, size_t length) {
                while (length) {
                    const auto n = send(fd, data, length, MSG_NOSIGNAL);
                    if (n < 0 && errno == EINTR)
                        continue;
                    if (n <= 0)
                        return false;
                    data += n;
                    length -= static_cast<size_t>(n);
                }
                return true;
            });
        };
    } else {
        if (!transfer([&](const char *data, size_t length) {
                response.body.append(data, length);
                return true;
            }))
            throw std::runtime_error("File changed or became unreadable during download");
    }
    return response;
}

std::vector<std::string> extract_asset_urls(const std::string &text) {
    std::set<std::string> urls;
    size_t search = 0;
    while (search < text.size()) {
        size_t http = text.find("http://", search);
        size_t https = text.find("https://", search);
        size_t pos = std::min(
            http == std::string::npos ? text.size() : http,
            https == std::string::npos ? text.size() : https
        );
        if (pos >= text.size()) {
            break;
        }
        size_t end = pos;
        while (end < text.size()) {
            const char ch = text[end];
            if (std::isspace(static_cast<unsigned char>(ch)) || ch == '"' || ch == '\'' || ch == '<' || ch == '>') {
                break;
            }
            end++;
        }
        std::string url = text.substr(pos, end - pos);
        if (url.find("/outputs/") != std::string::npos) {
            urls.insert(url);
        }
        search = end + 1;
    }
    return std::vector<std::string>(urls.begin(), urls.end());
}

bool is_valid_utf8(const std::string &value) {
    size_t i = 0;
    while (i < value.size()) {
        const auto byte = static_cast<unsigned char>(value[i]);
        size_t remaining = 0;
        if (byte <= 0x7f) {
            i++;
            continue;
        }
        if ((byte & 0xe0) == 0xc0) {
            if (byte < 0xc2) {
                return false;
            }
            remaining = 1;
        } else if ((byte & 0xf0) == 0xe0) {
            remaining = 2;
        } else if ((byte & 0xf8) == 0xf0) {
            if (byte > 0xf4) {
                return false;
            }
            remaining = 3;
        } else {
            return false;
        }
        if (i + remaining >= value.size()) {
            return false;
        }
        for (size_t j = 1; j <= remaining; ++j) {
            if ((static_cast<unsigned char>(value[i + j]) & 0xc0) != 0x80) {
                return false;
            }
        }
        i += remaining + 1;
    }
    return true;
}

void replace_all(std::string *text, const std::string &from, const std::string &to) {
    if (!text || from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = text->find(from, pos)) != std::string::npos) {
        text->replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string strip_trailing_slashes(std::string value);

struct FallbackDecision {
    bool configured = false;
    std::string url;
    bool should_use = false;
    std::string reason = "No fallback server configured";
    double free_vram_gb = 0.0;
    double free_ram_gb = 0.0;
    double memory_threshold_gb = 1.0;
};

std::string one_decimal(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << value;
    return out.str();
}

bool fallback_float(const std::string &raw, double *out) {
    // float() accepts Unicode decimal digits and Unicode whitespace, but not
    // the four ASCII separators that str.strip() also treats as whitespace.
    for (const unsigned char c : raw) if (c >= 0x1c && c <= 0x1f) return false;
    return parse_strict_python_float_string(normalize_decimal_digits(strip_config_value(raw)), out);
}

double fallback_memory_threshold() {
    double threshold;
    const std::string raw = ezl_getenv_fallback("FALLBACK_MEMORY_THRESHOLD", "8.0");
    if (!fallback_float(raw, &threshold)) throw std::invalid_argument("Invalid FALLBACK_MEMORY_THRESHOLD");
    return threshold;
}

FallbackDecision native_fallback_decision(const ResourceManager *resources, const std::string &url) {
    FallbackDecision decision;
    decision.url = url;
    decision.configured = !decision.url.empty();
    if (!decision.configured) return decision;
    decision.free_vram_gb = resources->get_total_free_vram();
    decision.free_ram_gb = resources->get_free_ram();
    decision.memory_threshold_gb = fallback_memory_threshold();
    const double free_memory = decision.free_vram_gb + decision.free_ram_gb;
    if (free_memory < decision.memory_threshold_gb) {
        decision.should_use = true;
        decision.reason = "Low combined memory (" + one_decimal(decision.free_vram_gb)
            + "GB VRAM + " + one_decimal(decision.free_ram_gb)
            + "GB RAM = " + one_decimal(free_memory)
            + "GB < " + python_float_string(decision.memory_threshold_gb) + "GB threshold)";
    } else {
        decision.reason = "Local resources sufficient";
    }
    return decision;
}

bool fallback_free_vram_value(const std::string &resources_body, double *out) {
    if (!out) {
        return false;
    }
    if (json_decode_error(resources_body) || resources_body.empty() || resources_body.front() != '{') return false;
    const std::string raw = trim(json_object_field(resources_body, "free_vram_gb", ""));
    if (raw.empty()) {
        *out = 0.0;
        return true;
    }
    if (raw == "true") {
        *out = 1.0;
        return true;
    }
    if (raw == "false") {
        *out = 0.0;
        return true;
    }
    double parsed = 0.0;
    if ((is_json_number_token(raw) || raw == "NaN" || raw == "Infinity" || raw == "-Infinity") && parse_strict_python_float_string(raw, &parsed)) {
        // Python's integer-to-float conversion for the displayed .1f value
        // raises on an enormous integer; exponent-form floats may be infinite.
        if (std::isinf(parsed) && raw.find_first_of(".eEI") == std::string::npos) return false;
        *out = parsed;
        return true;
    }
    return false;
}

bool is_loopback_url(std::string url) {
    url = to_lower(url);
    return url.find("://localhost") != std::string::npos
        || url.find("://127.0.0.1") != std::string::npos
        || url.find("://0.0.0.0") != std::string::npos
        || url.find("://::1") != std::string::npos;
}

std::string strip_trailing_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}


std::string read_file_or_empty(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "";
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string read_file_limited_or_empty(const std::string &path, size_t max_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "";
    }
    std::string out;
    std::array<char, 4096> buffer{};
    while (in && out.size() < max_bytes) {
        const size_t remaining = max_bytes - out.size();
        const auto chunk = static_cast<std::streamsize>(std::min(buffer.size(), remaining));
        in.read(buffer.data(), chunk);
        const std::streamsize got = in.gcount();
        if (got <= 0) {
            break;
        }
        out.append(buffer.data(), static_cast<size_t>(got));
    }
    return out;
}

std::string temporary_llama_startup_log_path() {
    char path[] = "/tmp/ezlocalai-llama-startup-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        return "";
    }
    close(fd);
    return path;
}

std::filesystem::path temporary_named_path(const std::string &prefix, const std::string &suffix = "") {
    return std::filesystem::temp_directory_path()
        / (prefix + "-" + now_id() + "-" + short_random_hex() + suffix);
}

bool executable_file(const std::string &path) {
    return !path.empty() && access(path.c_str(), X_OK) == 0;
}

std::string find_executable_path(const std::vector<std::string> &candidates) {
    for (const auto &candidate : candidates) {
        const std::string trimmed = trim(candidate);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed.find('/') != std::string::npos) {
            if (executable_file(trimmed)) {
                return trimmed;
            }
            continue;
        }
        const char *path_env = getenv("PATH");
        std::string path = path_env ? path_env : "";
        size_t start = 0;
        while (start <= path.size()) {
            const size_t colon = path.find(':', start);
            const std::string dir = colon == std::string::npos
                ? path.substr(start)
                : path.substr(start, colon - start);
            const std::filesystem::path full = (dir.empty() ? std::filesystem::path(".") : std::filesystem::path(dir)) / trimmed;
            if (executable_file(full.string())) {
                return full.string();
            }
            if (colon == std::string::npos) {
                break;
            }
            start = colon + 1;
        }
    }
    return "";
}

std::vector<std::string> split_command_words(const std::string &raw) {
    std::vector<std::string> words;
    std::string current;
    bool single = false;
    bool dbl = false;
    bool escaped = false;
    for (char ch : raw) {
        if (escaped) {
            current.push_back(ch);
            escaped = false;
            continue;
        }
        if (ch == '\\' && !single) {
            escaped = true;
            continue;
        }
        if (ch == '\'' && !dbl) {
            single = !single;
            continue;
        }
        if (ch == '"' && !single) {
            dbl = !dbl;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch)) && !single && !dbl) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

struct ProcessCaptureResult {
    int exit_code = 127;
    bool timed_out = false;
    std::string output;
};

ProcessCaptureResult run_process_capture(const std::vector<std::string> &args, int timeout_seconds) {
    ProcessCaptureResult result;
    if (args.empty()) {
        return result;
    }
    int pipe_fds[2] = {-1, -1};
    if (pipe(pipe_fds) != 0) {
        result.output = std::strerror(errno);
        return result;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        result.output = std::strerror(errno);
        return result;
    }
    if (pid == 0) {
        close(pipe_fds[0]);
        dup2(pipe_fds[1], STDOUT_FILENO);
        dup2(pipe_fds[1], STDERR_FILENO);
        if (pipe_fds[1] > STDERR_FILENO) {
            close(pipe_fds[1]);
        }
        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (const auto &arg : args) {
            argv.push_back(const_cast<char *>(arg.c_str()));
        }
        argv.push_back(nullptr);
        execvp(args.front().c_str(), argv.data());
        _exit(127);
    }

    close(pipe_fds[1]);
    fcntl(pipe_fds[0], F_SETFL, fcntl(pipe_fds[0], F_GETFL, 0) | O_NONBLOCK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(std::max(1, timeout_seconds));
    bool child_done = false;
    int status = 0;
    std::array<char, 4096> buffer{};
    while (true) {
        bool pipe_eof = false;
        for (;;) {
            const ssize_t n = read(pipe_fds[0], buffer.data(), buffer.size());
            if (n > 0) {
                result.output.append(buffer.data(), static_cast<size_t>(n));
                if (result.output.size() > 4 * 1024 * 1024) {
                    result.output.resize(4 * 1024 * 1024);
                }
                continue;
            }
            if (n == 0) {
                pipe_eof = true;
            }
            break;
        }
        const pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            child_done = true;
        }
        if (pipe_eof && !child_done) {
            waitpid(pid, &status, 0);
            child_done = true;
        }
        if (child_done) {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            result.timed_out = true;
            child_done = true;
            break;
        }
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(pipe_fds[0], &read_fds);
        timeval wait{};
        wait.tv_sec = 0;
        wait.tv_usec = 100000;
        select(pipe_fds[0] + 1, &read_fds, nullptr, nullptr, &wait);
    }
    for (;;) {
        const ssize_t n = read(pipe_fds[0], buffer.data(), buffer.size());
        if (n <= 0) {
            break;
        }
        result.output.append(buffer.data(), static_cast<size_t>(n));
    }
    close(pipe_fds[0]);
    if (result.timed_out) {
        result.exit_code = 124;
    } else if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }
    return result;
}

std::string resolve_whisper_model_path() {
    const auto direct_paths = {
        trim(ezl_getenv_default("WHISPER_MODEL_PATH")),
        trim(ezl_getenv_default("WHISPER_CPP_MODEL_PATH"))
    };
    for (const auto &path : direct_paths) {
        if (!path.empty() && std::filesystem::exists(path)) {
            return path;
        }
    }
    const std::string model = trim(ezl_getenv_default("WHISPER_MODEL"));
    if (!model.empty() && std::filesystem::exists(model)) {
        return model;
    }
    std::vector<std::filesystem::path> roots;
    const std::string configured_dir = trim(ezl_getenv_default("WHISPER_CPP_MODEL_DIR"));
    if (!configured_dir.empty()) {
        roots.emplace_back(configured_dir);
    }
    roots.emplace_back("whispercpp/models");
    roots.emplace_back("/app/whispercpp/models");
    const std::string models_path = trim(ezl_getenv_default("MODELS_PATH"));
    if (!models_path.empty()) {
        roots.emplace_back(models_path);
    }
    roots.emplace_back("models");

    std::vector<std::string> names;
    if (!model.empty()) {
        names.push_back(model);
        names.push_back("ggml-" + model + ".bin");
        names.push_back(model + ".bin");
        names.push_back(model + ".gguf");
    }
    for (const auto &root : roots) {
        for (const auto &name : names) {
            const auto candidate = root / name;
            if (std::filesystem::exists(candidate)) {
                return candidate.string();
            }
        }
    }
    return "";
}

std::string native_stt_resource_model_name() {
    std::string name = trim(ezl_getenv_default("WHISPER_MODEL"));
    if (name.empty()) {
        name = trim(ezl_getenv_default("WHISPER_MODEL_PATH"));
    }
    if (name.empty()) {
        name = trim(ezl_getenv_default("WHISPER_CPP_MODEL_PATH"));
    }
    if (name.empty()) {
        name = resolve_whisper_model_path();
    }
    if (name.empty()) {
        return "whisper.cpp";
    }
    std::filesystem::path path(name);
    if (name.find('/') != std::string::npos && !path.filename().empty()) {
        return path.filename().string();
    }
    return name;
}

std::string native_tts_resource_model_name() {
    std::string provider = trim(ezl_getenv_default("TTS_PROVIDER"));
    if (provider.empty()) {
        provider = "chatterbox";
    }
    if (to_lower(provider) == "chatterbox") {
        return "Chatterbox TTS";
    }
    return provider;
}

std::string native_image_resource_model_name() {
    const std::string model = trim(ezl_getenv_default("IMG_MODEL"));
    return model.empty() ? "image" : model;
}

std::string native_video_resource_model_name() {
    const std::string model = trim(ezl_getenv_default("VIDEO_MODEL"));
    return model.empty() ? "video" : model;
}

std::string aux_loaded_models_json(
    const std::map<std::string, int> &active_counts,
    const std::map<std::string, double> &last_used,
    bool video_loaded = false, const std::string &video_device = "video",
    bool legacy_tts = true, bool legacy_stt = true
) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto write_model = [&](const std::string &type, const std::string &name, const std::string &device, bool retained = false) {
        const int active = std::max(0, active_counts.count(type) ? active_counts.at(type) : 0);
        if (active <= 0 && !(retained && last_used.count(type))) {
            return;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        const auto last_it = last_used.find(type);
        const double used = last_it == last_used.end() ? monotonic_epoch_seconds() : last_it->second;
        out << "\"" << json_escape(type) << "\":{"
            << "\"name\":\"" << json_escape(name) << "\","
            << "\"device\":\"" << json_escape(device) << "\","
            << "\"vram_gb\":" << json_number_literal(0.0) << ","
            << "\"in_use\":" << (active > 0 ? "true" : "false") << ","
            << "\"active_requests\":" << active << ","
            << "\"last_used\":" << json_number_literal(used)
            << "}";
    };
    if (legacy_tts) write_model("tts", native_tts_resource_model_name(), configured_forward_url("VOICE_SERVER") ? "voice-server" : "tts");
    if (legacy_stt) write_model("stt", native_stt_resource_model_name(), "whisper.cpp");
    write_model("video", native_video_resource_model_name(), video_device, video_loaded);
    auto music_model = strip_config_value(ezl_getenv_default("MUSIC_MODEL"));
    if (music_model.empty()) music_model = "Serveurperso/ACE-Step-1.5-GGUF";
    write_model("music", music_model, "external", true);
    out << "}";
    return out.str();
}

std::string merge_loaded_model_json_objects(const std::string &left_raw, const std::string &right_raw) {
    const std::string left = trim(left_raw);
    const std::string right = trim(right_raw);
    if (left.empty() || left == "{}") {
        return right.empty() ? "{}" : right;
    }
    if (right.empty() || right == "{}") {
        return left;
    }
    if (left.front() != '{' || left.back() != '}' || right.front() != '{' || right.back() != '}') {
        return left;
    }
    // Later authoritative records replace earlier approximations. Concatenating
    // JSON objects produced duplicate keys when both described the same model.
    auto merged = nlohmann::ordered_json::parse(left);
    const auto updates = nlohmann::ordered_json::parse(right);
    for (const auto &entry : updates.items()) merged[entry.key()] = entry.value();
    return merged.dump();
}

template<class Snapshot> nlohmann::json speech_pool_json(const Snapshot &pool) {
    nlohmann::json devices = nlohmann::json::array();
    for (const auto &model : pool.resident) devices.push_back(model && model->device ? model->device() : "unknown");
    return {{"configured", pool.capacity}, {"resident", pool.resident.size()}, {"devices", devices}};
}

std::string resource_models_json(const std::vector<ModelResource> &models) {
    std::ostringstream out; out << '{'; bool first = true;
    for (const auto &model : models) {
        if (!first) out << ','; first = false;
        out << '"' << model_type_name(model.type) << "\":{\"name\":\"" << json_escape(model.name)
            << "\",\"device\":\"" << json_escape(model.device) << "\",\"vram_gb\":" << json_number_literal(model.vram_gb)
            << ",\"in_use\":" << (model.in_use ? "true" : "false") << ",\"active_requests\":" << model.active_requests
            << ",\"last_used\":" << json_number_literal(model.last_used) << '}';
    }
    out << '}'; return out.str();
}
std::string resource_cuda_json(const std::vector<CudaAllocatorStats> &stats) {
    std::ostringstream out; out << '['; bool first = true;
    for (const auto &gpu : stats) {
        if (!first) out << ','; first = false;
        out << "{\"index\":" << gpu.index;
        if (!gpu.error.empty()) out << ",\"error\":\"" << json_escape(gpu.error) << '"';
        else out << ",\"allocated_gb\":" << json_number_literal(gpu.allocated_gb)
            << ",\"reserved_gb\":" << json_number_literal(gpu.reserved_gb)
            << ",\"max_allocated_gb\":" << json_number_literal(gpu.max_allocated_gb)
            << ",\"free_gb\":" << json_number_literal(gpu.free_gb) << ",\"total_gb\":" << json_number_literal(gpu.total_gb);
        out << '}';
    }
    out << ']'; return out.str();
}

std::vector<std::string> json_string_values_for_key(const std::string &json, const std::string &key) {
    std::vector<std::string> values;
    const std::string marker = "\"" + key + "\":";
    size_t search = 0;
    while ((search = json.find(marker, search)) != std::string::npos) {
        size_t value = search + marker.size();
        while (value < json.size() && std::isspace(static_cast<unsigned char>(json[value]))) {
            ++value;
        }
        if (value < json.size() && json[value] == '"') {
            values.push_back(parse_json_string_at(json, value));
        }
        search = value + 1;
    }
    return values;
}

std::string whisper_text_from_output(const std::string &raw) {
    const std::string text = extract_json_string(raw, "text", "");
    if (!text.empty()) {
        return trim(text);
    }
    std::vector<std::string> parts = json_string_values_for_key(raw, "text");
    std::ostringstream joined;
    for (const auto &part : parts) {
        const std::string value = trim(part);
        if (value.empty()) {
            continue;
        }
        if (joined.tellp() > 0) {
            joined << " ";
        }
        joined << value;
    }
    const std::string combined = trim(joined.str());
    if (!combined.empty()) {
        return combined;
    }
    return trim(raw);
}

struct WhisperSegment {
    int id = 0;
    double start = 0.0;
    double end = 0.0;
    std::string text;
    std::string speaker;
};

std::vector<WhisperSegment> whisper_segments_from_output(const std::string &raw) {
    std::string segment_array = trim(extract_json_value(raw, "segments", ""));
    if (segment_array.empty() || segment_array.front() != '[') {
        segment_array = trim(extract_json_value(raw, "transcription", ""));
    }
    std::vector<WhisperSegment> segments;
    int fallback_id = 0;
    for (const auto &element : json_array_elements(segment_array)) {
        WhisperSegment segment;
        segment.id = extract_json_int_value(element, "id", fallback_id);
        segment.start = extract_json_double_value(element, "start", -1.0);
        segment.end = extract_json_double_value(element, "end", -1.0);
        if (segment.start < 0.0 || segment.end < 0.0) {
            const std::string offsets = extract_json_value(element, "offsets", "{}");
            segment.start = static_cast<double>(extract_json_int_value(offsets, "from", 0)) / 1000.0;
            segment.end = static_cast<double>(extract_json_int_value(offsets, "to", 0)) / 1000.0;
        }
        segment.text = trim(extract_json_string(element, "text", ""));
        segment.speaker = extract_json_string(element, "speaker", "");
        if (!segment.text.empty()) {
            segments.push_back(segment);
            ++fallback_id;
        }
    }
    if (segments.empty()) {
        const std::string text = whisper_text_from_output(raw);
        if (!text.empty()) {
            segments.push_back({0, 0.0, 0.0, text, ""});
        }
    }
    return segments;
}

std::string whisper_segments_json(const std::vector<WhisperSegment> &segments) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << "{"
            << "\"id\":" << segments[i].id << ","
            << "\"start\":" << json_number_literal(segments[i].start) << ","
            << "\"end\":" << json_number_literal(segments[i].end) << ","
            << "\"text\":\"" << json_escape(segments[i].text) << "\"";
        if (!segments[i].speaker.empty()) {
            out << ",\"speaker\":\"" << json_escape(segments[i].speaker) << "\"";
        }
        out << "}";
    }
    out << "]";
    return out.str();
}

std::string subtitle_timestamp(double seconds, bool srt) {
    if (seconds < 0.0) {
        seconds = 0.0;
    }
    const int hours = static_cast<int>(seconds / 3600.0);
    const int minutes = static_cast<int>(std::fmod(seconds, 3600.0) / 60.0);
    const int secs = static_cast<int>(std::fmod(seconds, 60.0));
    const int millis = std::max(0, std::min(999, static_cast<int>((seconds - std::floor(seconds)) * 1000.0)));
    std::ostringstream out;
    out << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << minutes << ":"
        << std::setw(2) << secs
        << (srt ? "," : ".")
        << std::setw(3) << millis;
    return out.str();
}

std::string whisper_segments_srt(const std::vector<WhisperSegment> &segments) {
    std::ostringstream out;
    for (size_t i = 0; i < segments.size(); ++i) {
        out << (i + 1) << "\n"
            << subtitle_timestamp(segments[i].start, true) << " --> " << subtitle_timestamp(segments[i].end, true) << "\n";
        if (!segments[i].speaker.empty()) {
            out << "[" << segments[i].speaker << "]: ";
        }
        out << segments[i].text << "\n";
        if (i + 1 < segments.size()) {
            out << "\n";
        }
    }
    return out.str();
}

std::string whisper_segments_vtt(const std::vector<WhisperSegment> &segments) {
    std::ostringstream out;
    out << "WEBVTT\n\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        out << subtitle_timestamp(segments[i].start, false) << " --> " << subtitle_timestamp(segments[i].end, false) << "\n";
        if (!segments[i].speaker.empty()) {
            out << "<v " << segments[i].speaker << ">" << segments[i].text << "</v>\n";
        } else {
            out << segments[i].text << "\n";
        }
        if (i + 1 < segments.size()) {
            out << "\n";
        }
    }
    return out.str();
}

int count_words(const std::string &text) {
    std::istringstream in(text);
    int count = 0;
    std::string word;
    while (in >> word) {
        ++count;
    }
    return count;
}

std::string numbered_subtitle_block(const std::vector<WhisperSegment> &segments, size_t start, size_t end) {
    std::ostringstream out;
    for (size_t i = start; i < end; ++i) {
        if (i != start) {
            out << "\n";
        }
        out << "[" << (i - start + 1) << "] " << segments[i].text;
    }
    return out.str();
}

std::map<int, std::string> parse_numbered_translation_lines(const std::string &text) {
    std::map<int, std::string> translations;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line.front() != '[') {
            continue;
        }
        const auto close = line.find(']');
        if (close == std::string::npos || close <= 1) {
            continue;
        }
        int index = 0;
        if (!parse_strict_int_string(line.substr(1, close - 1), &index) || index <= 0) {
            continue;
        }
        translations[index] = trim(line.substr(close + 1));
    }
    return translations;
}

std::string first_chat_response_content(const std::string &response_body) {
    const auto choices = extract_json_array_elements(response_body, "choices");
    if (choices.empty()) {
        return "";
    }
    const std::string message = trim(extract_json_value(choices.front(), "message", "{}"));
    return extract_json_string(message, "content", "");
}

std::string native_session_speaker_id(const std::string &session_id) {
    if (session_id.empty()) {
        return "SPEAKER_00";
    }
    uint64_t hash = 1469598103934665603ULL;
    for (const char ch : session_id) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << "SPK_" << std::hex << std::nouppercase << std::setw(12) << std::setfill('0')
        << (hash & 0xffffffffffffULL);
    return out.str();
}

void apply_single_speaker_diarization(std::vector<WhisperSegment> *segments, const std::string &session_id) {
    if (!segments) {
        return;
    }
    const std::string speaker = native_session_speaker_id(session_id);
    for (auto &segment : *segments) {
        if (segment.speaker.empty()) {
            segment.speaker = speaker;
        }
    }
}

std::string whisper_diarized_text(const std::vector<WhisperSegment> &segments) {
    std::ostringstream out;
    std::string current_speaker;
    for (const auto &segment : segments) {
        const std::string speaker = segment.speaker.empty() ? "SPEAKER_00" : segment.speaker;
        if (speaker != current_speaker) {
            current_speaker = speaker;
            if (out.tellp() > 0) {
                out << "\n";
            }
            out << "[" << speaker << "]: ";
        }
        out << segment.text << " ";
    }
    return trim(out.str());
}

struct PcmAudio {
    int sample_rate = 0;
    int channels = 0;
    std::vector<double> mono;
};

uint16_t read_le16(const std::string &data, size_t offset) {
    if (offset + 2 > data.size()) {
        return 0;
    }
    return static_cast<uint16_t>(
        static_cast<unsigned char>(data[offset])
        | (static_cast<unsigned char>(data[offset + 1]) << 8)
    );
}

uint32_t read_le32(const std::string &data, size_t offset) {
    if (offset + 4 > data.size()) {
        return 0;
    }
    return static_cast<uint32_t>(
        static_cast<unsigned char>(data[offset])
        | (static_cast<unsigned char>(data[offset + 1]) << 8)
        | (static_cast<unsigned char>(data[offset + 2]) << 16)
        | (static_cast<unsigned char>(data[offset + 3]) << 24)
    );
}

bool parse_pcm16_wav(const std::string &data, PcmAudio *audio) {
    if (!audio || data.size() < 44 || data.compare(0, 4, "RIFF") != 0 || data.compare(8, 4, "WAVE") != 0) {
        return false;
    }
    bool have_fmt = false;
    bool have_data = false;
    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    size_t data_offset = 0;
    size_t data_size = 0;

    size_t offset = 12;
    while (offset + 8 <= data.size()) {
        const std::string chunk = data.substr(offset, 4);
        const uint32_t chunk_size = read_le32(data, offset + 4);
        const size_t payload = offset + 8;
        if (payload > data.size() || chunk_size > data.size() - payload) {
            return false;
        }
        if (chunk == "fmt " && chunk_size >= 16) {
            format = read_le16(data, payload);
            channels = read_le16(data, payload + 2);
            sample_rate = read_le32(data, payload + 4);
            bits_per_sample = read_le16(data, payload + 14);
            have_fmt = true;
        } else if (chunk == "data") {
            data_offset = payload;
            data_size = chunk_size;
            have_data = true;
        }
        offset = payload + chunk_size + (chunk_size % 2);
    }

    if (!have_fmt || !have_data || format != 1 || channels == 0 || sample_rate == 0 || bits_per_sample != 16) {
        return false;
    }
    const size_t frame_bytes = static_cast<size_t>(channels) * 2;
    if (frame_bytes == 0 || data_size < frame_bytes) {
        return false;
    }
    const size_t frames = data_size / frame_bytes;
    audio->sample_rate = static_cast<int>(sample_rate);
    audio->channels = static_cast<int>(channels);
    audio->mono.clear();
    audio->mono.reserve(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (uint16_t ch = 0; ch < channels; ++ch) {
            const size_t sample_offset = data_offset + frame * frame_bytes + ch * 2;
            int16_t sample = static_cast<int16_t>(read_le16(data, sample_offset));
            sum += static_cast<double>(sample) / 32768.0;
        }
        audio->mono.push_back(sum / static_cast<double>(channels));
    }
    return !audio->mono.empty();
}

std::vector<double> segment_audio_features(const PcmAudio &audio, double start_seconds, double end_seconds) {
    if (audio.sample_rate <= 0 || audio.mono.empty() || end_seconds <= start_seconds) {
        return {};
    }
    size_t start = static_cast<size_t>(std::max(0.0, std::floor(start_seconds * audio.sample_rate)));
    size_t end = static_cast<size_t>(std::max(0.0, std::ceil(end_seconds * audio.sample_rate)));
    start = std::min(start, audio.mono.size());
    end = std::min(end, audio.mono.size());
    if (end <= start || end - start < static_cast<size_t>(std::max(1, audio.sample_rate / 10))) {
        return {};
    }

    double sum = 0.0;
    double abs_sum = 0.0;
    double square_sum = 0.0;
    double peak = 0.0;
    int zero_crossings = 0;
    double previous = audio.mono[start];
    for (size_t i = start; i < end; ++i) {
        const double sample = audio.mono[i];
        sum += sample;
        abs_sum += std::fabs(sample);
        square_sum += sample * sample;
        peak = std::max(peak, std::fabs(sample));
        if ((sample < 0.0 && previous >= 0.0) || (sample >= 0.0 && previous < 0.0)) {
            zero_crossings++;
        }
        previous = sample;
    }

    const double n = static_cast<double>(end - start);
    std::vector<double> features = {
        sum / n,
        abs_sum / n,
        std::sqrt(square_sum / n),
        peak,
        static_cast<double>(zero_crossings) / n,
    };

    constexpr size_t bins = 8;
    for (size_t bin = 0; bin < bins; ++bin) {
        const size_t bin_start = start + ((end - start) * bin) / bins;
        const size_t bin_end = start + ((end - start) * (bin + 1)) / bins;
        double bin_square = 0.0;
        for (size_t i = bin_start; i < bin_end; ++i) {
            bin_square += audio.mono[i] * audio.mono[i];
        }
        const double bin_n = static_cast<double>(std::max<size_t>(1, bin_end - bin_start));
        features.push_back(std::sqrt(bin_square / bin_n));
    }
    return features;
}

double feature_distance(const std::vector<double> &left, const std::vector<double> &right) {
    const size_t n = std::min(left.size(), right.size());
    if (n == 0) {
        return std::numeric_limits<double>::max();
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double delta = left[i] - right[i];
        sum += delta * delta;
    }
    return std::sqrt(sum);
}

std::vector<double> centroid_for_indices(const std::vector<std::vector<double>> &features, const std::vector<int> &indices) {
    if (indices.empty()) {
        return {};
    }
    std::vector<double> centroid(features[indices.front()].size(), 0.0);
    for (const int index : indices) {
        for (size_t dim = 0; dim < centroid.size(); ++dim) {
            centroid[dim] += features[index][dim];
        }
    }
    for (auto &value : centroid) {
        value /= static_cast<double>(indices.size());
    }
    return centroid;
}

std::vector<int> cluster_audio_features(std::vector<std::vector<double>> features, int requested_speakers) {
    if (features.empty()) {
        return {};
    }
    if (features.size() == 1) {
        return {0};
    }

    const size_t dims = features.front().size();
    for (size_t dim = 0; dim < dims; ++dim) {
        double mean = 0.0;
        for (const auto &feature : features) {
            mean += feature[dim];
        }
        mean /= static_cast<double>(features.size());
        double variance = 0.0;
        for (const auto &feature : features) {
            const double delta = feature[dim] - mean;
            variance += delta * delta;
        }
        const double stddev = std::sqrt(variance / static_cast<double>(features.size()));
        for (auto &feature : features) {
            feature[dim] = stddev > 1e-9 ? (feature[dim] - mean) / stddev : 0.0;
        }
    }

    int k = requested_speakers > 0
        ? std::min<int>(requested_speakers, static_cast<int>(features.size()))
        : 1;
    if (requested_speakers <= 0) {
        double max_distance = 0.0;
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = i + 1; j < features.size(); ++j) {
                max_distance = std::max(max_distance, feature_distance(features[i], features[j]));
            }
        }
        if (max_distance > 2.0) {
            k = std::min<int>(2, static_cast<int>(features.size()));
        }
    }
    if (k <= 1) {
        return std::vector<int>(features.size(), 0);
    }

    std::vector<std::vector<double>> centroids;
    centroids.reserve(static_cast<size_t>(k));
    centroids.push_back(features.front());
    while (static_cast<int>(centroids.size()) < k) {
        size_t farthest = 0;
        double best = -1.0;
        for (size_t i = 0; i < features.size(); ++i) {
            double nearest = std::numeric_limits<double>::max();
            for (const auto &centroid : centroids) {
                nearest = std::min(nearest, feature_distance(features[i], centroid));
            }
            if (nearest > best) {
                best = nearest;
                farthest = i;
            }
        }
        centroids.push_back(features[farthest]);
    }

    std::vector<int> labels(features.size(), 0);
    for (int iter = 0; iter < 12; ++iter) {
        bool changed = false;
        for (size_t i = 0; i < features.size(); ++i) {
            int best_label = 0;
            double best_distance = std::numeric_limits<double>::max();
            for (int label = 0; label < k; ++label) {
                const double distance = feature_distance(features[i], centroids[static_cast<size_t>(label)]);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_label = label;
                }
            }
            if (labels[i] != best_label) {
                labels[i] = best_label;
                changed = true;
            }
        }

        std::vector<std::vector<int>> grouped(static_cast<size_t>(k));
        for (size_t i = 0; i < labels.size(); ++i) {
            grouped[static_cast<size_t>(labels[i])].push_back(static_cast<int>(i));
        }
        for (int label = 0; label < k; ++label) {
            if (!grouped[static_cast<size_t>(label)].empty()) {
                centroids[static_cast<size_t>(label)] = centroid_for_indices(features, grouped[static_cast<size_t>(label)]);
            }
        }
        if (!changed) {
            break;
        }
    }
    return labels;
}

struct NativeVoicePrint {
    std::string id;
    std::vector<double> centroid;
    int count = 0;
};

std::mutex g_voice_print_mutex;
std::map<std::string, std::vector<NativeVoicePrint>> g_voice_prints;

double cosine_similarity(const std::vector<double> &left, const std::vector<double> &right) {
    const size_t n = std::min(left.size(), right.size());
    if (n == 0) {
        return -1.0;
    }
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += left[i] * right[i];
        left_norm += left[i] * left[i];
        right_norm += right[i] * right[i];
    }
    if (left_norm <= 0.0 || right_norm <= 0.0) {
        return -1.0;
    }
    return dot / (std::sqrt(left_norm) * std::sqrt(right_norm));
}

std::string new_native_speaker_uuid() {
    std::string id = uuid_v4();
    id.erase(std::remove(id.begin(), id.end(), '-'), id.end());
    return id.substr(0, std::min<size_t>(12, id.size()));
}

std::string native_session_speaker_for_centroid(const std::string &session_id, const std::vector<double> &centroid) {
    std::lock_guard<std::mutex> lock(g_voice_print_mutex);
    auto &prints = g_voice_prints[session_id];
    int best_index = -1;
    double best_similarity = -1.0;
    for (size_t i = 0; i < prints.size(); ++i) {
        const double similarity = cosine_similarity(centroid, prints[i].centroid);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_index = static_cast<int>(i);
        }
    }
    if (best_index >= 0 && best_similarity >= 0.7) {
        auto &matched = prints[static_cast<size_t>(best_index)];
        const int new_count = matched.count + 1;
        if (matched.centroid.size() == centroid.size()) {
            for (size_t i = 0; i < centroid.size(); ++i) {
                matched.centroid[i] = ((matched.centroid[i] * matched.count) + centroid[i]) / new_count;
            }
        }
        matched.count = new_count;
        return "SPK_" + matched.id;
    }
    NativeVoicePrint print;
    print.id = new_native_speaker_uuid();
    print.centroid = centroid;
    print.count = 1;
    prints.push_back(print);
    return "SPK_" + print.id;
}

bool apply_wav_feature_diarization(const std::string &audio_data,
                                   std::vector<WhisperSegment> *segments,
                                   int requested_speakers,
                                   const std::string &session_id) {
    if (!segments || segments->empty()) {
        return false;
    }
    PcmAudio audio;
    if (!parse_pcm16_wav(audio_data, &audio)) {
        return false;
    }

    std::vector<size_t> segment_indices;
    std::vector<std::vector<double>> features;
    for (size_t i = 0; i < segments->size(); ++i) {
        auto feature = segment_audio_features(audio, (*segments)[i].start, (*segments)[i].end);
        if (!feature.empty()) {
            segment_indices.push_back(i);
            features.push_back(std::move(feature));
        }
    }
    if (features.empty()) {
        return false;
    }

    const auto labels = cluster_audio_features(features, requested_speakers);
    std::map<int, std::string> label_names;
    std::map<int, std::vector<int>> grouped;
    for (size_t i = 0; i < labels.size(); ++i) {
        grouped[labels[i]].push_back(static_cast<int>(i));
    }
    for (const auto &entry : grouped) {
        if (!session_id.empty()) {
            const auto centroid = centroid_for_indices(features, entry.second);
            label_names[entry.first] = native_session_speaker_for_centroid(session_id, centroid);
        } else {
            std::ostringstream label;
            label << "SPEAKER_" << std::setw(2) << std::setfill('0') << label_names.size();
            label_names[entry.first] = label.str();
        }
    }
    for (size_t i = 0; i < labels.size(); ++i) {
        (*segments)[segment_indices[i]].speaker = label_names[labels[i]];
    }
    for (auto &segment : *segments) {
        if (segment.speaker.empty()) {
            segment.speaker = label_names.empty() ? "SPEAKER_00" : label_names.begin()->second;
        }
    }
    return true;
}

std::filesystem::path wakeword_models_dir() {
    return std::filesystem::current_path() / "wakeword_models";
}

std::string wakeword_key_for_word(const std::string &word) { return wakeword_name(word); }

bool wakeword_model_dir_for_word(const std::string &word, std::filesystem::path *out) {
    const auto directory = wakeword_model_directory(wakeword_models_dir(), word);
    if (!directory) return false;
    if (out) *out = *directory;
    return true;
}

std::string iso_from_system_time(std::chrono::system_clock::time_point time);

std::string iso_from_file_time(std::filesystem::file_time_type file_time) {
    const auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        file_time - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now()
    );
    return iso_from_system_time(system_time);
}

std::string iso_from_system_time(std::chrono::system_clock::time_point time) {
    auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(time);
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(time - seconds).count();
    if (micros < 0) {
        seconds -= std::chrono::seconds(1);
        micros += 1000000;
    }
    const std::time_t raw = std::chrono::system_clock::to_time_t(seconds);
    std::tm local{};
    localtime_r(&raw, &local);
    std::ostringstream out;
    out << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    if (micros != 0) {
        out << "." << std::setw(6) << std::setfill('0') << micros;
    }
    return out.str();
}

std::string wakeword_models_json() {
    using Json = nlohmann::ordered_json;
    const auto base = wakeword_models_dir();
    std::filesystem::create_directories(base);
    auto models = Json::array();
    for (const auto &entry : std::filesystem::directory_iterator(base)) {
        std::optional<std::filesystem::path> directory;
        try { directory = wakeword_model_directory(base, entry.path().filename().string()); }
        catch (const std::invalid_argument &) { continue; }
        if (!directory || !std::filesystem::is_directory(*directory)) continue;
        const auto config_path = wakeword_contained_path(*directory, "config.json");
        auto config = Json::object();
        if (std::filesystem::exists(config_path)) {
            std::ifstream input(config_path);
            config = Json::parse(input);
        }
        if (!config.is_object()) throw std::runtime_error("Wake word config must be an object");
        auto files = Json::object();
        for (const auto *ext : {"pt", "onnx", "espdl"}) {
            const auto path = wakeword_contained_path(*directory, std::string("model.") + ext);
            if (std::filesystem::exists(path))
                files[ext] = {{"path", path.string()}, {"size_bytes", std::filesystem::file_size(path)}};
        }
        models.push_back({{"word", config.contains("word") ? config["word"] : Json(directory->filename().string())},
            {"directory", directory->string()}, {"config", config}, {"files", files},
            {"created_at", iso_from_file_time(std::filesystem::last_write_time(*directory))}});
    }
    return Json{{"models", models}, {"total", models.size()}}.dump();
}

std::filesystem::path wakeword_jobs_dir() {
    return std::filesystem::current_path() / "wakeword_jobs";
}

std::string json_value_or_null(const std::string &json, const std::string &key) {
    const std::string value = extract_json_value(json, key, "");
    return value.empty() ? "null" : value;
}

std::string wakeword_job_response_json(const std::string &job_json) {
    const std::string metrics = extract_json_value(job_json, "metrics", "{}");
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{"
        << "\"job_id\":\"" << json_escape(extract_json_string(job_json, "job_id", "")) << "\","
        << "\"word\":\"" << json_escape(extract_json_string(job_json, "word", "")) << "\","
        << "\"status\":\"" << json_escape(extract_json_string(job_json, "status", "")) << "\","
        << "\"progress\":" << extract_json_double_value(job_json, "progress", 0.0) << ","
        << "\"current_stage\":\"" << json_escape(extract_json_string(job_json, "current_stage", "")) << "\","
        << "\"error_message\":" << json_value_or_null(job_json, "error_message") << ","
        << "\"estimated_completion\":" << json_value_or_null(job_json, "estimated_completion") << ","
        << "\"model_path\":" << json_value_or_null(job_json, "model_path") << ","
        << "\"metrics\":" << (metrics.empty() ? "{}" : metrics) << ","
        << "\"created_at\":\"" << json_escape(extract_json_string(job_json, "created_at", "")) << "\","
        << "\"updated_at\":\"" << json_escape(extract_json_string(job_json, "updated_at", "")) << "\""
        << "}";
    return out.str();
}

struct WakewordJobFile {
    std::filesystem::path path;
    std::string json;
    std::string job_id;
    std::string status;
    std::string created_at;
};

bool wakeword_valid_status(const std::string &status) {
    static const std::set<std::string> statuses = {
        "pending",
        "generating_samples",
        "augmenting",
        "training",
        "exporting",
        "completed",
        "failed",
        "cancelled",
    };
    return statuses.find(status) != statuses.end();
}

bool json_raw_is_string_literal(const std::string &raw) {
    const std::string value = trim(raw);
    return !value.empty() && value.front() == '"' && json_value_end(value, 0) == value.size();
}

bool wakeword_job_json_loads_like_python(const std::string &raw) {
    if (!is_valid_top_level_json_object(raw)) {
        return false;
    }
    const auto values = top_level_json_object_values(raw);
    for (const auto &required : {"job_id", "word", "status", "created_at", "updated_at"}) {
        if (values.find(required) == values.end()) {
            return false;
        }
    }
    if (!json_raw_is_string_literal(values.at("word"))
        || !json_raw_is_string_literal(values.at("status"))
        || !json_raw_is_string_literal(values.at("created_at"))
        || !json_raw_is_string_literal(values.at("updated_at"))) {
        return false;
    }
    if (!wakeword_valid_status(parse_json_string_at(trim(values.at("status")), 0))) {
        return false;
    }
    if (parse_json_string_at(trim(values.at("created_at")), 0).empty()
        || parse_json_string_at(trim(values.at("updated_at")), 0).empty()) {
        return false;
    }
    const auto estimated = values.find("estimated_completion");
    if (estimated != values.end()
        && !is_python_falsey_json_value(estimated->second)
        && !json_raw_is_string_literal(estimated->second)) {
        return false;
    }
    return true;
}

std::vector<WakewordJobFile> wakeword_load_jobs() {
    std::error_code ec;
    const auto base = wakeword_jobs_dir();
    std::filesystem::create_directories(base, ec);
    std::vector<WakewordJobFile> jobs;
    for (const auto &entry : std::filesystem::directory_iterator(base, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".json") {
            continue;
        }
        const std::string raw = read_file_or_empty(entry.path().string());
        if (!wakeword_job_json_loads_like_python(raw)) {
            continue;
        }
        const std::string job_id = extract_json_string(raw, "job_id", "");
        if (job_id.empty()) {
            continue;
        }
        jobs.push_back({
            entry.path(),
            raw,
            job_id,
            extract_json_string(raw, "status", ""),
            extract_json_string(raw, "created_at", ""),
        });
    }
    return jobs;
}

std::string wakeword_jobs_json(const std::string &status_filter, int limit) {
    auto jobs = wakeword_load_jobs();
    std::sort(jobs.begin(), jobs.end(), [](const WakewordJobFile &lhs, const WakewordJobFile &rhs) {
        return lhs.created_at > rhs.created_at;
    });
    std::vector<WakewordJobFile> filtered;
    for (const auto &job : jobs) {
        if (status_filter.empty() || job.status == status_filter) {
            filtered.push_back(job);
        }
    }
    size_t effective_limit = 0;
    if (limit < 0) {
        const int sliced = static_cast<int>(filtered.size()) + limit;
        effective_limit = static_cast<size_t>(std::max(0, sliced));
    } else {
        effective_limit = static_cast<size_t>(limit);
    }
    std::ostringstream out;
    out << "[";
    size_t written = 0;
    for (const auto &job : filtered) {
        if (written >= effective_limit) {
            break;
        }
        if (written) {
            out << ",";
        }
        out << wakeword_job_response_json(job.json);
        written++;
    }
    out << "]";
    return out.str();
}

bool wakeword_job_by_id(const std::string &job_id, WakewordJobFile *out) {
    for (const auto &job : wakeword_load_jobs()) {
        if (job.job_id == job_id) {
            if (out) {
                *out = job;
            }
            return true;
        }
    }
    return false;
}

bool wakeword_active_job_for_word(const std::string &word, WakewordJobFile *out) {
    const std::string requested = llm_unicode_lower(word);
    for (const auto &job : wakeword_load_jobs()) {
        const std::string job_word = llm_unicode_lower(extract_json_string(job.json, "word", ""));
        if (job_word == requested
            && job.status != "completed"
            && job.status != "failed"
            && job.status != "cancelled") {
            if (out) {
                *out = job;
            }
            return true;
        }
    }
    return false;
}

std::string replace_json_string_field(std::string json, const std::string &key, const std::string &value) {
    const std::string marker = "\"" + key + "\"";
    const auto key_pos = json.find(marker);
    if (key_pos == std::string::npos) {
        return json;
    }
    const auto colon = json.find(':', key_pos + marker.size());
    if (colon == std::string::npos) {
        return json;
    }
    const auto value_start = json.find_first_not_of(" \t\r\n", colon + 1);
    const auto value_end = json_value_end(json, value_start);
    if (value_start == std::string::npos || value_end == std::string::npos) {
        return json;
    }
    json.replace(value_start, value_end - value_start, "\"" + json_escape(value) + "\"");
    return json;
}

std::string replace_last_json_string_field(std::string json, const std::string &key, const std::string &value) {
    const std::string marker = "\"" + key + "\"";
    const auto key_pos = json.rfind(marker);
    if (key_pos == std::string::npos) {
        return json;
    }
    const auto colon = json.find(':', key_pos + marker.size());
    if (colon == std::string::npos) {
        return json;
    }
    const auto value_start = json.find_first_not_of(" \t\r\n", colon + 1);
    const auto value_end = json_value_end(json, value_start);
    if (value_start == std::string::npos || value_end == std::string::npos) {
        return json;
    }
    json.replace(value_start, value_end - value_start, "\"" + json_escape(value) + "\"");
    return json;
}

std::string query_param_value(const std::string &query, const std::string &name, const std::string &fallback) {
    size_t start = 0;
    const std::string prefix = name + "=";
    while (start <= query.size()) {
        const auto end = query.find('&', start);
        const auto part = query.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (starts_with(part, prefix)) {
            return url_decode(part.substr(prefix.size()), true);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return fallback;
}

bool query_param_present(const std::string &query, const std::string &name) {
    size_t start = 0;
    const std::string prefix = name + "=";
    while (start <= query.size()) {
        const auto end = query.find('&', start);
        const auto part = query.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (starts_with(part, prefix)) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}


void write_file(const std::string &path, const std::string &content) {
    const auto slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        std::string dir = path.substr(0, slash);
        if (!dir.empty()) {
            std::string partial;
            for (size_t i = 0; i < dir.size(); ++i) {
                partial.push_back(dir[i]);
                if (dir[i] == '/' || i + 1 == dir.size()) {
                    if (!partial.empty()) {
                        mkdir(partial.c_str(), 0755);
                    }
                }
            }
        }
    }
    std::ofstream out(path, std::ios::binary);
    out << content;
}

std::string usage_file_path() {
    const std::string configured = ezl_getenv_default("USAGE_FILE");
    return configured.empty() ? "/data/usage.json" : configured;
}

std::string usage_history_file_path() {
    const std::string path = usage_file_path();
    const auto slash = path.find_last_of('/');
    const auto dot = path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        return path.substr(0, dot) + ".history" + path.substr(dot);
    }
    return path + ".history.json";
}

int usage_history_max() {
    try {
        return std::max(1, std::stoi(ezl_getenv_default("USAGE_HISTORY_MAX")));
    } catch (...) {
        return 500;
    }
}

std::string normalize_usage_model_name(const std::string &model) {
    std::string value = trim(model);
    if (value.empty()) {
        return "unknown";
    }
    const std::string lowered = to_lower(value);
    if (lowered.size() >= 5 && lowered.substr(lowered.size() - 5) == "-gguf") {
        value.resize(value.size() - 5);
    } else if (lowered.size() >= 5 && lowered.substr(lowered.size() - 5) == ".gguf") {
        value.resize(value.size() - 5);
    }
    return value.empty() ? "unknown" : value;
}

std::string extract_multipart_boundary(const std::string &content_type) {
    const std::string marker = "boundary=";
    auto pos = content_type.find(marker);
    if (pos == std::string::npos) {
        return "";
    }
    std::string boundary = content_type.substr(pos + marker.size());
    const auto semi = boundary.find(';');
    if (semi != std::string::npos) {
        boundary = boundary.substr(0, semi);
    }
    boundary = trim(boundary);
    if (boundary.size() >= 2 && boundary.front() == '"' && boundary.back() == '"') {
        boundary = boundary.substr(1, boundary.size() - 2);
    }
    return boundary;
}

std::string extract_disposition_name(const std::string &headers) {
    const std::string marker = "name=\"";
    auto pos = headers.find(marker);
    if (pos == std::string::npos) {
        return "";
    }
    pos += marker.size();
    const auto end = headers.find('"', pos);
    if (end == std::string::npos) {
        return "";
    }
    return headers.substr(pos, end - pos);
}

std::string extract_disposition_filename(const std::string &headers) {
    const std::string marker = "filename=\"";
    auto pos = headers.find(marker);
    if (pos == std::string::npos) {
        return "";
    }
    pos += marker.size();
    const auto end = headers.find('"', pos);
    if (end == std::string::npos) {
        return "";
    }
    return headers.substr(pos, end - pos);
}

bool disposition_has_filename(const std::string &headers) {
    return headers.find("filename=\"") != std::string::npos;
}

std::string multipart_header_value(const std::string &headers, const std::string &name) {
    std::istringstream lines(headers);
    std::string line;
    const std::string normalized_name = to_lower(name);
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        if (to_lower(trim(line.substr(0, colon))) == normalized_name) {
            return trim(line.substr(colon + 1));
        }
    }
    return "";
}

struct MultipartPart {
    std::string name;
    std::string filename;
    bool has_filename;
    std::string content_type;
    std::string content;
    std::string raw_headers;
};

std::vector<MultipartPart> parse_multipart_parts(const std::string &body, const std::string &boundary) {
    std::vector<MultipartPart> parts;
    if (boundary.empty()) {
        return parts;
    }
    const std::string delimiter = "--" + boundary;
    size_t pos = 0;
    while (true) {
        pos = body.find(delimiter, pos);
        if (pos == std::string::npos) {
            break;
        }
        pos += delimiter.size();
        if (body.compare(pos, 2, "--") == 0) {
            break;
        }
        if (body.compare(pos, 2, "\r\n") == 0) {
            pos += 2;
        }
        const auto header_end = body.find("\r\n\r\n", pos);
        if (header_end == std::string::npos) {
            break;
        }
        const auto headers = body.substr(pos, header_end - pos);
        const auto name = extract_disposition_name(headers);
        const auto filename = extract_disposition_filename(headers);
        const bool has_filename = disposition_has_filename(headers);
        size_t content_start = header_end + 4;
        size_t next = body.find(delimiter, content_start);
        if (next == std::string::npos) {
            break;
        }
        size_t content_end = next;
        if (content_end >= 2 && body.compare(content_end - 2, 2, "\r\n") == 0) {
            content_end -= 2;
        }
        if (!name.empty()) {
            parts.push_back({name, filename, has_filename, multipart_header_value(headers, "content-type"), body.substr(content_start, content_end - content_start), headers});
        }
        pos = next;
    }
    return parts;
}

void append_multipart_file_upload_validation_errors(
    const std::string &body,
    const std::string &content_type,
    std::vector<std::string> *validation_errors
) {
    const auto parts = parse_multipart_parts(body, extract_multipart_boundary(content_type));
    bool saw_file_field = false;
    std::string first_file_field_content;
    for (const auto &part : parts) {
        if (part.name != "file") {
            continue;
        }
        if (part.has_filename) {
            return;
        }
        if (!saw_file_field) {
            first_file_field_content = part.content;
            saw_file_field = true;
        }
    }
    if (!validation_errors) {
        return;
    }
    validation_errors->push_back(
        saw_file_field
            ? pydantic_upload_file_type_error_object(first_file_field_content)
            : pydantic_missing_file_error_object()
    );
}

bool multipart_file_upload_validation_error(
    const std::string &body,
    const std::string &content_type,
    HttpResponse *out
) {
    std::vector<std::string> validation_errors;
    append_multipart_file_upload_validation_errors(body, content_type, &validation_errors);
    if (validation_errors.empty()) {
        return false;
    }
    if (out) {
        *out = {422, "application/json", pydantic_error_response_body(validation_errors)};
    }
    return true;
}

std::unordered_map<std::string, std::string> parse_multipart_form(const std::string &body, const std::string &boundary) {
    std::unordered_map<std::string, std::string> fields;
    if (boundary.empty()) {
        return fields;
    }
    const std::string delimiter = "--" + boundary;
    size_t pos = 0;
    while (true) {
        pos = body.find(delimiter, pos);
        if (pos == std::string::npos) {
            break;
        }
        pos += delimiter.size();
        if (body.compare(pos, 2, "--") == 0) {
            break;
        }
        if (body.compare(pos, 2, "\r\n") == 0) {
            pos += 2;
        }
        const auto header_end = body.find("\r\n\r\n", pos);
        if (header_end == std::string::npos) {
            break;
        }
        const auto headers = body.substr(pos, header_end - pos);
        const auto name = extract_disposition_name(headers);
        size_t content_start = header_end + 4;
        size_t next = body.find(delimiter, content_start);
        if (next == std::string::npos) {
            break;
        }
        size_t content_end = next;
        if (content_end >= 2 && body.compare(content_end - 2, 2, "\r\n") == 0) {
            content_end -= 2;
        }
        if (!name.empty()) {
            fields[name] = body.substr(content_start, content_end - content_start);
        }
        pos = next;
    }
    return fields;
}

std::string sanitize_voice_name(std::string voice) {
    if (voice.empty()) {
        voice = "default";
    }
    std::string cleaned;
    for (char ch : voice) {
        switch (ch) {
            case '/':
            case '\\':
            case ':':
            case '*':
            case '?':
            case '"':
            case '<':
            case '>':
            case '|':
                break;
            default:
                cleaned.push_back(ch);
                break;
        }
    }
    while (true) {
        const auto pos = cleaned.find("..");
        if (pos == std::string::npos) {
            break;
        }
        cleaned.erase(pos, 2);
    }
    std::string allowlisted;
    for (char ch : cleaned) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_') {
            allowlisted.push_back(ch);
        }
    }
    if (allowlisted.empty()) {
        allowlisted = "default";
    }
    if (allowlisted.size() > 100) {
        allowlisted.resize(100);
    }
    return allowlisted;
}

struct ParsedHttpUrl {
    bool tls = false;
    std::string host;
    int port = 80;
    std::string base_path;
};

std::unordered_map<std::string, std::string> parse_headers(std::istream &stream);

bool parse_http_url(const std::string &url, ParsedHttpUrl *out) {
    if (!out) {
        return false;
    }
    ParsedHttpUrl parsed;
    std::string rest;
    if (starts_with(url, "http://")) {
        parsed.tls = false;
        parsed.port = 80;
        rest = url.substr(7);
    } else if (starts_with(url, "https://")) {
        parsed.tls = true;
        parsed.port = 443;
        rest = url.substr(8);
    } else {
        return false;
    }
    const auto slash = rest.find('/');
    std::string host_port = slash == std::string::npos ? rest : rest.substr(0, slash);
    parsed.base_path = slash == std::string::npos ? "" : rest.substr(slash);
    const auto colon = host_port.rfind(':');
    if (colon != std::string::npos) {
        parsed.host = host_port.substr(0, colon);
        try {
            parsed.port = std::stoi(host_port.substr(colon + 1));
        } catch (...) {
            return false;
        }
    } else {
        parsed.host = host_port;
    }
    if (parsed.host.empty() || parsed.port <= 0 || parsed.port > 65535) {
        return false;
    }
    *out = parsed;
    return true;
}

bool host_is_ip_address(const std::string &host) {
    in_addr ipv4{};
    if (inet_pton(AF_INET, host.c_str(), &ipv4) == 1) {
        return true;
    }
    in6_addr ipv6{};
    return inet_pton(AF_INET6, host.c_str(), &ipv6) == 1;
}

std::string aiohttp_host_port(const ParsedHttpUrl &parsed) {
    return parsed.host + ":" + std::to_string(parsed.port);
}

std::string aiohttp_request_url(const std::string &url, const std::string &path) {
    return strip_trailing_slashes(url) + path;
}

std::string sockaddr_python_tuple(const sockaddr *addr) {
    if (!addr) {
        return "";
    }
    char ip[INET6_ADDRSTRLEN] = {0};
    if (addr->sa_family == AF_INET) {
        const auto *in = reinterpret_cast<const sockaddr_in *>(addr);
        if (inet_ntop(AF_INET, &in->sin_addr, ip, sizeof(ip))) {
            return "('" + std::string(ip) + "', " + std::to_string(ntohs(in->sin_port)) + ")";
        }
    } else if (addr->sa_family == AF_INET6) {
        const auto *in6 = reinterpret_cast<const sockaddr_in6 *>(addr);
        if (inet_ntop(AF_INET6, &in6->sin6_addr, ip, sizeof(ip))) {
            return "('" + std::string(ip) + "', " + std::to_string(ntohs(in6->sin6_port)) + ", 0, 0)";
        }
    }
    return "";
}

std::string aiohttp_dns_error_message(const ParsedHttpUrl &parsed, int gai_code) {
    std::string reason = gai_code == 0 ? "Name or service not known" : gai_strerror(gai_code);
    if (reason.empty()) {
        reason = "Name or service not known";
    }
    return "ClientConnectorDNSError: Cannot connect to host "
        + aiohttp_host_port(parsed)
        + " ssl:default ["
        + reason
        + "]";
}

std::string aiohttp_connect_error_message(const ParsedHttpUrl &parsed, const std::string &endpoint, int error_code) {
    (void)error_code;
    const std::string tuple = endpoint.empty()
        ? "('" + parsed.host + "', " + std::to_string(parsed.port) + ")"
        : endpoint;
    return "ClientConnectorError: Cannot connect to host "
        + aiohttp_host_port(parsed)
        + " ssl:default [Connect call failed "
        + tuple
        + "]";
}

std::string aiohttp_connection_timeout_message(const std::string &url, const std::string &path) {
    return "ConnectionTimeoutError: Connection timeout to host " + aiohttp_request_url(url, path);
}

bool connect_with_timeout(int fd, const sockaddr *addr, socklen_t addrlen, int timeout_seconds, bool *timed_out) {
    if (timed_out) {
        *timed_out = false;
    }
    const int original_flags = fcntl(fd, F_GETFL, 0);
    if (original_flags < 0 || fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) < 0) {
        return connect(fd, addr, addrlen) == 0;
    }

    int rc = connect(fd, addr, addrlen);
    if (rc == 0) {
        fcntl(fd, F_SETFL, original_flags);
        return true;
    }
    if (errno != EINPROGRESS) {
        fcntl(fd, F_SETFL, original_flags);
        return false;
    }

    fd_set write_fds;
    FD_ZERO(&write_fds);
    FD_SET(fd, &write_fds);
    timeval timeout{};
    timeout.tv_sec = timeout_seconds;
    timeout.tv_usec = 0;
    do {
        rc = select(fd + 1, nullptr, &write_fds, nullptr, &timeout);
    } while (rc < 0 && errno == EINTR);

    if (rc <= 0) {
        if (timed_out && rc == 0) {
            *timed_out = true;
        }
        fcntl(fd, F_SETFL, original_flags);
        return false;
    }
    if (!FD_ISSET(fd, &write_fds)) {
        fcntl(fd, F_SETFL, original_flags);
        return false;
    }

    int socket_error = 0;
    socklen_t socket_error_len = sizeof(socket_error);
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &socket_error_len) != 0 || socket_error != 0) {
        if (socket_error != 0) {
            errno = socket_error;
        }
        fcntl(fd, F_SETFL, original_flags);
        return false;
    }
    fcntl(fd, F_SETFL, original_flags);
    return true;
}

bool parse_http_chunk_size(const std::string &line, size_t *size_out) {
    if (!size_out) {
        return false;
    }
    std::string value = trim(line);
    const auto semi = value.find(';');
    if (semi != std::string::npos) {
        value = trim(value.substr(0, semi));
    }
    if (value.empty()) {
        return false;
    }
    size_t parsed = 0;
    for (char ch : value) {
        int digit = -1;
        if (ch >= '0' && ch <= '9') {
            digit = ch - '0';
        } else if (ch >= 'a' && ch <= 'f') {
            digit = 10 + (ch - 'a');
        } else if (ch >= 'A' && ch <= 'F') {
            digit = 10 + (ch - 'A');
        } else {
            return false;
        }
        if (parsed > (std::numeric_limits<size_t>::max() - static_cast<size_t>(digit)) / 16U) {
            return false;
        }
        parsed = parsed * 16U + static_cast<size_t>(digit);
    }
    *size_out = parsed;
    return true;
}

enum class ChunkedDecodeResult {
    Complete,
    Incomplete,
    Malformed,
};

ChunkedDecodeResult decode_http_chunked_body_state(const std::string &encoded, std::string *decoded_out) {
    if (!decoded_out) {
        return ChunkedDecodeResult::Malformed;
    }
    std::string decoded;
    size_t pos = 0;
    while (true) {
        const auto line_end = encoded.find("\r\n", pos);
        if (line_end == std::string::npos) {
            return ChunkedDecodeResult::Incomplete;
        }
        size_t chunk_size = 0;
        if (!parse_http_chunk_size(encoded.substr(pos, line_end - pos), &chunk_size)) {
            return ChunkedDecodeResult::Malformed;
        }
        pos = line_end + 2;
        if (chunk_size == 0) {
            // The zero chunk begins the trailer section. The response is not
            // complete until its final empty line arrives, even with no trailers.
            for (;;) {
                const auto end = encoded.find("\r\n",pos);
                if (end == std::string::npos) return ChunkedDecodeResult::Incomplete;
                if (end == pos) { *decoded_out = decoded; return ChunkedDecodeResult::Complete; }
                pos = end+2;
            }
        }
        if (pos > encoded.size() || chunk_size > encoded.size() - pos) {
            return ChunkedDecodeResult::Incomplete;
        }
        decoded.append(encoded, pos, chunk_size);
        pos += chunk_size;
        if (pos + 2 > encoded.size()) {
            return ChunkedDecodeResult::Incomplete;
        }
        if (encoded.compare(pos, 2, "\r\n") != 0) {
            return ChunkedDecodeResult::Malformed;
        }
        pos += 2;
    }
}

bool decode_http_chunked_body(const std::string &encoded, std::string *decoded_out) {
    return decode_http_chunked_body_state(encoded, decoded_out) == ChunkedDecodeResult::Complete;
}

using HttpHeaderList = std::vector<std::pair<std::string,std::string>>;
HttpHeaderList response_header_list(const std::string &raw) {
    HttpHeaderList out; std::istringstream input(raw);
    for (std::string line;std::getline(input,line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon=line.find(':'); if (colon==std::string::npos) continue;
        const auto value=line.find_first_not_of(" \t",colon+1);
        out.emplace_back(to_lower(line.substr(0,colon)),value==std::string::npos?"":line.substr(value));
    }
    return out;
}
std::optional<std::string> response_header(const HttpHeaderList &headers,const std::string &name) {
    for (const auto &entry:headers) if(entry.first==name) return entry.second;
    return {};
}
std::optional<std::string> redirect_location(const HttpHeaderList &headers) {
    const auto location=response_header(headers,"location");
    return location && !location->empty()?location:response_header(headers,"uri");
}
bool redirect_status(int status) { return status==301 || status==302 || status==303 || status==307 || status==308; }
struct FallbackHttpOptions {
    const HttpUrl &url;
    HttpDeadline &deadline;
    const std::string &authorization;
    const std::string &cookies;
    HttpCookieJar &cookie_jar;
    bool read_error_body;
    bool read_success_body;
    unsigned redirects;
    std::string user_agent;
    TtsEmit emit;
    TtsInterrupt interrupt;
    bool default_connection = false;
    bool *retryable_disconnect = nullptr;
    std::function<void(int)> received_status;
};
// A duplicate descriptor prevents a late cancellation callback from shutting
// down an unrelated socket after the request's original fd has been reused.
struct HttpStreamSocket {
    int fd;
    std::atomic<bool> cancelled{false};
    explicit HttpStreamSocket(int original) : fd(fcntl(original,F_DUPFD_CLOEXEC,3)) {}
    ~HttpStreamSocket() { if (fd >= 0) close(fd); }
    void cancel() { cancelled = true; if (fd >= 0) shutdown(fd,SHUT_RDWR); }
};
std::string base64_encode(const unsigned char *data,size_t len);

HttpResponse http_worker_request(
    const std::string &method,
    const std::string &url,
    const std::string &path,
    const std::string &api_key,
    const std::string &content_type,
    const std::string &body,
    int timeout_seconds = 0,
    bool half_close_after_send = true,
    const FallbackHttpOptions *fallback = nullptr,
    const HttpDeadline *supplied_deadline = nullptr
) {
    auto *deadline = fallback ? &fallback->deadline : supplied_deadline;
    const bool read_error_body = !fallback || fallback->read_error_body;
    const bool decompress_response = fallback != nullptr;
    ParsedHttpUrl parsed;
    if (fallback) {
        parsed.tls=fallback->url.scheme=="https";parsed.host=fallback->url.host;parsed.port=fallback->url.port();parsed.base_path=fallback->url.request_target();
    } else if (!parse_http_url(url, &parsed)) {
        return worker_transport_error_response("Worker URL is not a supported http:// or https:// URL");
    }
    addrinfo hints{};
    hints.ai_family = fallback ? AF_UNSPEC : AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo *results = nullptr;
    const std::string port = std::to_string(parsed.port);
    HttpAddressResult resolved;
    int gai_result;
    if (deadline) {
        resolved = http_resolve(parsed.host,port,hints,*deadline);
        if (resolved.timed_out) return worker_transport_error_response("Worker request timed out");
        results = resolved.addresses.get(); gai_result = resolved.error;
    } else gai_result = getaddrinfo(parsed.host.c_str(), port.c_str(), &hints, &results);
    if (gai_result != 0 || !results) {
        return worker_transport_error_response(aiohttp_dns_error_message(parsed, gai_result));
    }

    int fd = -1;
    bool connect_timed_out = false;
    std::shared_ptr<HttpStreamSocket> interrupt_socket;
    int last_connect_errno = 0;
    std::string last_connect_endpoint;
    for (addrinfo *item = results; item; item = item->ai_next) {
        fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) {
            continue;
        }
        bool item_timed_out = false;
        if (fallback && fallback->interrupt) {
            interrupt_socket = std::make_shared<HttpStreamSocket>(fd);
            if (interrupt_socket->fd < 0) { close(fd); return worker_transport_error_response("Failed to install stream cancellation"); }
            fallback->interrupt([weak=std::weak_ptr<HttpStreamSocket>(interrupt_socket)] { if (auto socket=weak.lock()) socket->cancel(); });
        }
        const bool cancelled = interrupt_socket && interrupt_socket->cancelled;
        const bool connected = cancelled ? false : deadline ? http_connect(fd,item->ai_addr,item->ai_addrlen,*deadline) : timeout_seconds > 0
            ? connect_with_timeout(fd, item->ai_addr, item->ai_addrlen, timeout_seconds, &item_timed_out)
            : connect(fd, item->ai_addr, item->ai_addrlen) == 0;
        if (connected && !(interrupt_socket && interrupt_socket->cancelled)) {
            break;
        }
        connect_timed_out = connect_timed_out || item_timed_out;
        if (!item_timed_out) {
            last_connect_errno = errno;
            last_connect_endpoint = sockaddr_python_tuple(item->ai_addr);
        }
        close(fd);
        fd = -1;
        if (interrupt_socket && interrupt_socket->cancelled) break;
        if (deadline && deadline->expired()) break;
    }
    if (!deadline) freeaddrinfo(results);
    if (fd < 0) {
        if (deadline && deadline->expired()) return worker_transport_error_response("Worker request timed out");
        if (connect_timed_out) {
            return worker_transport_error_response(aiohttp_connection_timeout_message(url, path));
        }
        return worker_transport_error_response(aiohttp_connect_error_message(parsed, last_connect_endpoint, last_connect_errno));
    }
    if (!deadline && timeout_seconds > 0) {
        timeval timeout{};
        timeout.tv_sec = timeout_seconds;
        timeout.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }

    SSL_CTX *ssl_ctx = nullptr;
    SSL *ssl = nullptr;
    auto cleanup_tls = [&]() {
        if (ssl) {
            SSL_free(ssl);
            ssl = nullptr;
        }
        if (ssl_ctx) {
            SSL_CTX_free(ssl_ctx);
            ssl_ctx = nullptr;
        }
    };
    if (parsed.tls) {
        OPENSSL_init_ssl(0, nullptr);
        ssl_ctx = SSL_CTX_new(TLS_client_method());
        if (!ssl_ctx) {
            close(fd);
            return worker_transport_error_response("Failed to create TLS context");
        }
        SSL_CTX_set_verify(ssl_ctx, SSL_VERIFY_PEER, nullptr);
        SSL_CTX_set_default_verify_paths(ssl_ctx);
        const char *cert_file = getenv("SSL_CERT_FILE");
        const char *cert_dir = getenv("SSL_CERT_DIR");
        if ((cert_file && *cert_file) || (cert_dir && *cert_dir)) {
            SSL_CTX_load_verify_locations(
                ssl_ctx,
                cert_file && *cert_file ? cert_file : nullptr,
                cert_dir && *cert_dir ? cert_dir : nullptr
            );
        }
        ssl = SSL_new(ssl_ctx);
        if (!ssl) {
            cleanup_tls();
            close(fd);
            return worker_transport_error_response("Failed to create TLS connection");
        }
        auto tls_host=parsed.host;
        if(fallback)while(!tls_host.empty()&&tls_host.back()=='.')tls_host.pop_back();
        if (!host_is_ip_address(tls_host)) {
            SSL_set_tlsext_host_name(ssl, tls_host.c_str());
            X509_VERIFY_PARAM_set1_host(SSL_get0_param(ssl), tls_host.c_str(), 0);
        } else {
            X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(ssl), tls_host.c_str());
        }
        SSL_set_fd(ssl, fd);
        if (deadline ? !http_tls_connect(ssl,fd,*deadline) : SSL_connect(ssl) != 1) {
            cleanup_tls();
            close(fd);
            return worker_transport_error_response("Failed to establish TLS connection to worker");
        }
        const long verify_result = SSL_get_verify_result(ssl);
        if (verify_result != X509_V_OK) {
            cleanup_tls();
            close(fd);
            return worker_transport_error_response(std::string("Worker TLS certificate verification failed: ") + X509_verify_cert_error_string(verify_result));
        }
    }

    const std::string full_path = fallback ? parsed.base_path : parsed.base_path + path;
    const int default_port = parsed.tls ? 443 : 80;
    const std::string host_header = fallback ? fallback->url.host_header() : parsed.port == default_port ? parsed.host : parsed.host + ":" + std::to_string(parsed.port);
    std::ostringstream request;
    request << method << " " << full_path << " HTTP/1.1\r\n"
        << "Host: " << host_header << "\r\n";
    if (!fallback || !fallback->default_connection) request << "Connection: close\r\n";
    if (decompress_response) request << "Accept: */*\r\nAccept-Encoding: gzip, deflate, br\r\n";
    if (fallback && !fallback->user_agent.empty()) request << "User-Agent: " << fallback->user_agent << "\r\n";
    if (!body.empty() || method != "GET") {
        request << "Content-Length: " << body.size() << "\r\n";
    }
    if (!content_type.empty()) {
        request << "Content-Type: " << content_type << "\r\n";
    }
    if (fallback) {
        if (!fallback->authorization.empty()) request << "Authorization: " << fallback->authorization << "\r\n";
        if (!fallback->cookies.empty()) request << "Cookie: " << fallback->cookies << "\r\n";
    } else if (!api_key.empty() && api_key != "none") {
        request << "Authorization: Bearer " << api_key << "\r\n";
    }
    request << "\r\n" << body;
    const std::string raw = request.str();
    size_t sent = 0;
    while (sent < raw.size()) {
        ssize_t n = 0;
        if (deadline) {
            n = http_write(fd,ssl,raw.data()+sent,raw.size()-sent,*deadline);
        } else if (ssl) {
            n = SSL_write(ssl, raw.data() + sent, static_cast<int>(std::min<size_t>(raw.size() - sent, 16384)));
        } else {
            n = send(fd, raw.data() + sent, raw.size() - sent, 0);
        }
        if (n <= 0) {
            const int send_error=errno;
            const int ssl_error = ssl ? SSL_get_error(ssl, static_cast<int>(n)) : 0;
            cleanup_tls();
            close(fd);
            if ((deadline && deadline->expired()) || errno == EAGAIN || errno == EWOULDBLOCK
                || ssl_error == SSL_ERROR_WANT_READ
                || ssl_error == SSL_ERROR_WANT_WRITE) {
                return worker_transport_error_response("Worker request timed out");
            }
            if(fallback&&fallback->retryable_disconnect&&!(interrupt_socket&&interrupt_socket->cancelled)
                &&(send_error==EPIPE||send_error==ECONNRESET))*fallback->retryable_disconnect=true;
            return worker_transport_error_response("Failed to send request to worker");
        }
        sent += static_cast<size_t>(n);
    }
    if (!ssl && half_close_after_send) {
        shutdown(fd, SHUT_WR);
    }

    std::string response_raw;
    char buffer[4096];
    size_t response_header_end = std::string::npos;
    size_t expected_response_bytes = 0;
    bool have_expected_response_bytes = false;
    bool response_chunked = false;
    bool skip_response_body = false;
    bool stream_response_body = false;
    std::unique_ptr<HttpBodyDecoder> body_decoder;
    size_t body_offset = 0;
    HttpHeaderList response_headers;
    auto parse_content_length = [](const std::string &value, size_t *out) {
        if (!out) {
            return false;
        }
        const std::string trimmed = trim(value);
        if (trimmed.empty()) {
            return false;
        }
        size_t parsed = 0;
        for (char ch : trimmed) {
            if (ch < '0' || ch > '9') {
                return false;
            }
            const size_t digit = static_cast<size_t>(ch - '0');
            if (parsed > (std::numeric_limits<size_t>::max() - digit) / 10U) {
                return false;
            }
            parsed = parsed * 10U + digit;
        }
        *out = parsed;
        return true;
    };
    for (;;) {
        ssize_t n = deadline ? http_read(fd,ssl,buffer,sizeof(buffer),*deadline) : ssl ? SSL_read(ssl, buffer, sizeof(buffer)) : recv(fd, buffer, sizeof(buffer), 0);
        if (deadline && n < 0) {
            const bool timed_out = deadline->expired();
            if(fallback&&fallback->retryable_disconnect&&!timed_out&&response_header_end==std::string::npos
                &&!(interrupt_socket&&interrupt_socket->cancelled)&&(errno==ECONNRESET||errno==EPIPE))*fallback->retryable_disconnect=true;
            cleanup_tls(); close(fd);
            return worker_transport_error_response(timed_out ? "Worker request timed out" : "Failed to read worker response");
        }
        if (!deadline && n <= 0 && ssl) {
            const int ssl_error = SSL_get_error(ssl, static_cast<int>(n));
            if (ssl_error == SSL_ERROR_ZERO_RETURN) {
                break;
            }
            if (ssl_error == SSL_ERROR_WANT_READ || ssl_error == SSL_ERROR_WANT_WRITE) {
                cleanup_tls();
                close(fd);
                return worker_transport_error_response("Worker request timed out");
            }
            break;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            cleanup_tls();
            close(fd);
            return worker_transport_error_response("Worker request timed out");
        }
        if (n <= 0) {
            break;
        }
        response_raw.append(buffer, static_cast<size_t>(n));
        if (response_header_end == std::string::npos) {
            response_header_end = response_raw.find("\r\n\r\n");
            if (response_header_end != std::string::npos) {
                const auto status_end_for_headers = response_raw.find("\r\n");
                if (status_end_for_headers != std::string::npos && status_end_for_headers < response_header_end) {
                    response_headers=response_header_list(response_raw.substr(status_end_for_headers+2,response_header_end-status_end_for_headers-2));
                    std::istringstream partial_headers_stream(response_raw.substr(
                        status_end_for_headers + 2,
                        response_header_end - status_end_for_headers - 2
                    ));
                    const auto partial_headers = parse_headers(partial_headers_stream);
                    const auto transfer_encoding_it = partial_headers.find("transfer-encoding");
                    response_chunked = transfer_encoding_it != partial_headers.end()
                        && to_lower(transfer_encoding_it->second).find("chunked") != std::string::npos;
                    if (!response_chunked) {
                        const auto length_it = partial_headers.find("content-length");
                        size_t content_length = 0;
                        if (length_it != partial_headers.end() && parse_content_length(length_it->second, &content_length)) {
                            expected_response_bytes = response_header_end + 4 + content_length;
                            have_expected_response_bytes = true;
                        }
                    }
                    if (decompress_response) {
                        // llhttp strips leading OWS but retains trailing OWS in
                        // this header; aiohttp matches the entire raw value.
                        std::string encoding;
                        std::istringstream raw_headers(response_raw.substr(status_end_for_headers+2,response_header_end-status_end_for_headers-2));
                        for (std::string line; std::getline(raw_headers,line);) {
                            if (!line.empty() && line.back() == '\r') line.pop_back();
                            const auto colon = line.find(':');
                            if (colon == std::string::npos || to_lower(line.substr(0,colon)) != "content-encoding") continue;
                            const auto start = line.find_first_not_of(" \t",colon+1);
                            encoding = start == std::string::npos ? "" : line.substr(start);
                        }
                        const auto length = have_expected_response_bytes ? std::optional<size_t>(expected_response_bytes-response_header_end-4) : std::nullopt;
                        try { body_decoder = std::make_unique<HttpBodyDecoder>(encoding,response_chunked,length); }
                        catch (const std::exception &) { cleanup_tls(); close(fd); return worker_transport_error_response("Invalid worker content encoding"); }
                        body_offset = response_header_end+4;
                    }
                    if (fallback || !read_error_body) {
                        if (fallback) {
                            std::vector<std::string> cookies;
                            for (const auto &header : response_headers) if (header.first == "set-cookie") cookies.push_back(header.second);
                            try { fallback->cookie_jar.update(cookies,fallback->url); }
                            catch (const std::exception &) { cleanup_tls(); close(fd); return worker_transport_error_response("Invalid worker cookies"); }
                        }
                        std::istringstream status_line(response_raw.substr(0,status_end_for_headers));
                        std::string protocol; int status = 0; status_line >> protocol >> status;
                        if (fallback && fallback->received_status) fallback->received_status(status);
                        stream_response_body = fallback && fallback->emit && status == 200;
                        const bool follow_redirect=fallback && redirect_status(status) && (redirect_location(response_headers).has_value() || fallback->redirects>=9);
                        if (follow_redirect || (!read_error_body && status != 200) || (fallback && !fallback->read_success_body && status == 200)) {
                            skip_response_body = true;
                            // aiohttp parses bytes already received with the
                            // headers before handing the response to the caller.
                            // A feed error belongs to the unread payload; an EOF
                            // decoder error aborts response parsing itself.
                            if (body_decoder) {
                                bool payload_error = false;
                                try { body_decoder->feed(std::string_view(response_raw).substr(body_offset)); }
                                catch (const HttpContentDecodingError &) { payload_error = true; }
                                catch (const std::exception &) { cleanup_tls(); close(fd); return worker_transport_error_response("Invalid worker response payload"); }
                                if (!payload_error && body_decoder->complete()) {
                                    try { body_decoder->finish(); }
                                    catch (const std::exception &) { cleanup_tls(); close(fd); return worker_transport_error_response("Invalid worker response payload"); }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        if (body_decoder) {
            try { body_decoder->feed(std::string_view(response_raw).substr(body_offset)); }
            catch (const std::exception &) { cleanup_tls(); close(fd); return worker_transport_error_response("Invalid worker response payload"); }
            body_offset = response_raw.size();
            if (stream_response_body) {
                auto bytes = body_decoder->take();
                bool delivered = true;
                try { if (!bytes.empty()) delivered = fallback->emit(bytes); }
                catch (...) { cleanup_tls(); close(fd); throw; }
                if (!delivered) { cleanup_tls(); close(fd); return worker_transport_error_response("Stream consumer disconnected"); }
                response_raw.resize(response_header_end+4);
                body_offset = response_raw.size();
            }
            if (body_decoder->complete()) break;
            continue;
        }
        if (have_expected_response_bytes && response_raw.size() >= expected_response_bytes) {
            response_raw.resize(expected_response_bytes);
            break;
        }
        if (response_chunked && response_header_end != std::string::npos) {
            std::string decoded_probe;
            const auto chunked_state = decode_http_chunked_body_state(
                response_raw.substr(response_header_end + 4),
                &decoded_probe
            );
            if (chunked_state == ChunkedDecodeResult::Complete) {
                break;
            }
            if (chunked_state == ChunkedDecodeResult::Malformed) {
                cleanup_tls();
                close(fd);
                return worker_transport_error_response("Malformed worker response");
            }
        }
    }
    cleanup_tls();
    close(fd);

    if (!skip_response_body && !stream_response_body && deadline && have_expected_response_bytes && response_raw.size() < expected_response_bytes) {
        return worker_transport_error_response("Incomplete worker response");
    }

    const auto header_end = response_raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        if(fallback&&fallback->retryable_disconnect&&!(interrupt_socket&&interrupt_socket->cancelled)
            &&(response_raw.empty()||std::string("HTTP/").compare(0,std::min<size_t>(5,response_raw.size()),response_raw.substr(0,5))==0))
            *fallback->retryable_disconnect=true;
        return worker_transport_error_response("Malformed worker response");
    }
    std::istringstream status_line(response_raw.substr(0, response_raw.find("\r\n")));
    std::string http_version;
    int status = 502;
    status_line >> http_version >> status;
    const auto status_end = response_raw.find("\r\n");
    std::istringstream headers_stream(response_raw.substr(status_end + 2, header_end - status_end - 2));
    auto headers = parse_headers(headers_stream);
    std::string response_content_type = "application/octet-stream";
    auto content_type_it = headers.find("content-type");
    if (content_type_it != headers.end()) {
        response_content_type = content_type_it->second;
    }
    std::string response_body = skip_response_body ? "" : response_raw.substr(header_end + 4);
    if (body_decoder && !skip_response_body) {
        try { response_body = body_decoder->finish(); }
        catch (const std::exception &) { return worker_transport_error_response("Invalid worker response payload"); }
        if (stream_response_body) {
            if (!response_body.empty() && !fallback->emit(response_body)) return worker_transport_error_response("Stream consumer disconnected");
            response_body.clear();
        }
    }
    auto transfer_encoding_it = headers.find("transfer-encoding");
    if (!skip_response_body && !body_decoder && transfer_encoding_it != headers.end()
        && to_lower(transfer_encoding_it->second).find("chunked") != std::string::npos) {
        std::string decoded_body;
        if (!decode_http_chunked_body(response_body, &decoded_body)) {
            return worker_transport_error_response("Malformed worker response");
        }
        response_body = decoded_body;
    }
    HttpResponse response{status, response_content_type, response_body};
    if(fallback)response.headers=std::move(response_headers);
    return response;
}

HttpResponse fallback_http_request(const std::string &method, const std::string &url, const std::string &path, const std::string &key, const std::string &type, const std::string &body, double total, bool read_error_body = true, bool read_success_body = true, bool sdk = false, const std::string &image_user_agent = "", const TtsEmit &emit = {}, const TtsInterrupt &interrupt = {}, const std::string &client_user_agent = "", const std::function<void(int)> &received_status = {}, const std::function<bool()> &cancelled = {}) {
    try {
        auto target=HttpUrl::parse(url+path);target.require_http();HttpDeadline deadline(total,true,cancelled);
        std::string header_auth=key.empty()||(!sdk && key=="none")?"":"Bearer "+key;
        std::optional<std::string> basic_auth;
        HttpCookieJar cookie_jar;
        std::string current_method=method,current_type=type,current_body=body;
        unsigned redirects=0;
        bool retry_connection=!sdk&&image_user_agent.empty()&&(method=="GET"||method=="HEAD"||method=="OPTIONS"||method=="TRACE"||method=="PUT"||method=="DELETE");
        for(;;) {
            if(const auto credentials=target.credentials()) {
                const auto bytes=credentials->first+":"+credentials->second;
                basic_auth="Basic "+base64_encode(reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size());
            }
            target.remove_credentials();
            if(basic_auth && !header_auth.empty()) return worker_transport_error_response("Conflicting fallback authorization");
            const auto authorization=basic_auth.value_or(header_auth);
            if(authorization.find_first_of("\r\n")!=std::string::npos)return worker_transport_error_response("Invalid fallback authorization");
            const auto cookies=cookie_jar.header(target);
            bool disconnected=false;
            const FallbackHttpOptions options{target,deadline,authorization,cookies,cookie_jar,read_error_body,read_success_body,redirects,client_user_agent.empty()?image_user_agent:client_user_agent,emit,interrupt,!client_user_agent.empty(),&disconnected,received_status};
            auto response=http_worker_request(current_method,target.str(),"","",current_type,current_body,0,false,&options);
            if(response.transport_error&&disconnected&&retry_connection&&!deadline.expired()){retry_connection=false;continue;}
            if(response.transport_error)return response;
            if(!redirect_status(response.status)){if (!sdk && image_user_agent.empty()) response.headers.clear();return response;}
            if(++redirects >= (!image_user_agent.empty() ? 31u : sdk ? 21u : 10u))return worker_transport_error_response("Too many fallback redirects");
            const auto location=redirect_location(response.headers);
            if(!location){response.headers.clear();return response;}
            if((response.status==303 && current_method!="HEAD") || ((response.status==301 || response.status==302) && current_method=="POST")) {
                current_method="GET";current_body.clear();
                // Multipart Content-Type was generated by the payload, while
                // JSON's Content-Type was an explicit client header.
                if(starts_with(to_lower(current_type),"multipart/form-data"))current_type.clear();
            }
            auto next=target.resolve(*location);next.require_http();
            if(!target.same_origin(next)){header_auth.clear();basic_auth.reset();}
            target=std::move(next);
        }
    } catch (const std::exception &) { return worker_transport_error_response("Invalid fallback HTTP request"); }
}

HttpResponse fallback_json_request(const std::string &method, const std::string &url, const std::string &path, const std::string &key, const std::string &body, double total, bool read_error_body = true, bool read_success_body = true) {
    return fallback_http_request(method,url,path,key,"application/json",body,total,read_error_body,read_success_body);
}

bool send_all_fd(int fd, const std::string &data) {
    return http_socket::send_all(fd, data);
}

bool send_http_chunk(int fd, const std::string &chunk) {
    return http_socket::send_chunk(fd, chunk);
}

void append_stream_capture(const std::shared_ptr<std::string> &capture, const std::string &chunk) {
    if (!capture || chunk.empty()) {
        return;
    }
    constexpr size_t kMaxCapturedStreamBytes = 1024 * 1024;
    capture->append(chunk);
    if (capture->size() > kMaxCapturedStreamBytes) {
        capture->erase(0, capture->size() - (kMaxCapturedStreamBytes / 2));
    }
}

struct WorkerStreamState {
    int fd = -1;
    std::mutex fd_mutex;
    bool interrupted = false;
    SSL_CTX *ssl_ctx = nullptr;
    SSL *ssl = nullptr;
    bool upstream_chunked = false;
    std::optional<size_t> content_length;
    std::string initial_body;
    std::shared_ptr<std::string> stream_capture;

    ~WorkerStreamState() {
        close_all();
    }

    void close_all() {
        if (ssl) {
            SSL_free(ssl);
            ssl = nullptr;
        }
        if (ssl_ctx) {
            SSL_CTX_free(ssl_ctx);
            ssl_ctx = nullptr;
        }
        close_fd();
    }
    void close_fd() {
        std::lock_guard<std::mutex> lock(fd_mutex);
        if (fd >= 0) { close(fd); fd = -1; }
    }
    bool install_fd(int value) {
        std::lock_guard<std::mutex> lock(fd_mutex);
        if (interrupted) { if (value >= 0) close(value); return false; }
        fd = value; return true;
    }
    void interrupt_read() {
        std::lock_guard<std::mutex> lock(fd_mutex);
        interrupted = true;
        if (fd >= 0) shutdown(fd, SHUT_RDWR);
    }

    ssize_t read_some(char *buffer, size_t size, bool *timed_out) {
        if (timed_out) {
            *timed_out = false;
        }
        if (ssl) {
            const ssize_t n = SSL_read(ssl, buffer, static_cast<int>(size));
            if (n <= 0) {
                const int ssl_error = SSL_get_error(ssl, static_cast<int>(n));
                if (ssl_error == SSL_ERROR_WANT_READ || ssl_error == SSL_ERROR_WANT_WRITE) {
                    if (timed_out) {
                        *timed_out = true;
                    }
                    return -1;
                }
                return 0;
            }
            return n;
        }
        const ssize_t n = recv(fd, buffer, size, 0);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (timed_out) {
                *timed_out = true;
            }
            return -1;
        }
        return n;
    }
};

class StreamingChunkedDecoder {
public:
    bool feed(const std::string &data, std::vector<std::string> *chunks, bool *done) {
        if (done) {
            *done = false;
        }
        buffer_ += data;
        while (true) {
            if (reading_size_) {
                const auto line_end = buffer_.find("\r\n");
                if (line_end == std::string::npos) {
                    return true;
                }
                std::string line = trim(buffer_.substr(0, line_end));
                const auto extension = line.find(';');
                if (extension != std::string::npos) {
                    line = trim(line.substr(0, extension));
                }
                if (line.empty()) {
                    return false;
                }
                errno = 0;
                char *end = nullptr;
                const unsigned long long parsed = std::strtoull(line.c_str(), &end, 16);
                if (errno != 0 || end == line.c_str() || trim(end ? std::string(end) : "").size() != 0) {
                    return false;
                }
                next_size_ = static_cast<size_t>(parsed);
                buffer_.erase(0, line_end + 2);
                if (next_size_ == 0) {
                    if (done) {
                        *done = true;
                    }
                    return true;
                }
                reading_size_ = false;
            }
            if (buffer_.size() < next_size_ + 2) {
                return true;
            }
            if (buffer_.substr(next_size_, 2) != "\r\n") {
                return false;
            }
            if (chunks) {
                chunks->push_back(buffer_.substr(0, next_size_));
            }
            buffer_.erase(0, next_size_ + 2);
            next_size_ = 0;
            reading_size_ = true;
        }
    }

private:
    std::string buffer_;
    bool reading_size_ = true;
    size_t next_size_ = 0;
};

bool stream_worker_body_to_client(int client_fd, const std::shared_ptr<WorkerStreamState> &state) {
    bool ok = true;
    if (state->upstream_chunked) {
        StreamingChunkedDecoder decoder;
        bool done = false;
        std::vector<std::string> decoded_chunks;
        if (!state->initial_body.empty()) {
            ok = decoder.feed(state->initial_body, &decoded_chunks, &done);
            for (const auto &chunk : decoded_chunks) {
                append_stream_capture(state->stream_capture, chunk);
                ok = ok && send_http_chunk(client_fd, chunk);
            }
        }
        char buffer[4096];
        while (ok && !done) {
            bool timed_out = false;
            const ssize_t n = state->read_some(buffer, sizeof(buffer), &timed_out);
            if (n <= 0) {
                break;
            }
            decoded_chunks.clear();
            ok = decoder.feed(std::string(buffer, static_cast<size_t>(n)), &decoded_chunks, &done);
            for (const auto &chunk : decoded_chunks) {
                append_stream_capture(state->stream_capture, chunk);
                ok = ok && send_http_chunk(client_fd, chunk);
            }
        }
    } else {
        append_stream_capture(state->stream_capture, state->initial_body);
        ok = send_http_chunk(client_fd, state->initial_body);
        char buffer[4096];
        while (ok) {
            bool timed_out = false;
            const ssize_t n = state->read_some(buffer, sizeof(buffer), &timed_out);
            if (n <= 0) {
                break;
            }
            const std::string chunk(buffer, static_cast<size_t>(n));
            append_stream_capture(state->stream_capture, chunk);
            ok = send_http_chunk(client_fd, chunk);
        }
    }
    send_all_fd(client_fd, "0\r\n\r\n");
    state->close_all();
    return ok;
}

HttpResponse http_worker_stream_request(
    const std::string &method,
    const std::string &url,
    const std::string &path,
    const std::string &api_key,
    const std::string &content_type,
    const std::string &body,
    int timeout_seconds = 0,
    std::shared_ptr<WorkerStreamState> *opened_stream = nullptr,
    const LlmInferenceStream::Interrupt &register_interrupt = {},
    const std::map<std::string, std::string> &forward_headers = {}
) {
    ParsedHttpUrl parsed;
    if (!parse_http_url(url, &parsed)) {
        return worker_transport_error_response("Worker URL is not a supported http:// or https:// URL");
    }
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo *results = nullptr;
    const std::string port = std::to_string(parsed.port);
    const int gai_result = getaddrinfo(parsed.host.c_str(), port.c_str(), &hints, &results);
    if (gai_result != 0 || !results) {
        return worker_transport_error_response(aiohttp_dns_error_message(parsed, gai_result));
    }

    auto state = std::make_shared<WorkerStreamState>();
    if (register_interrupt) register_interrupt([state] { state->interrupt_read(); });
    bool connect_timed_out = false;
    int last_connect_errno = 0;
    std::string last_connect_endpoint;
    for (addrinfo *item = results; item; item = item->ai_next) {
        if (!state->install_fd(socket(item->ai_family, item->ai_socktype, item->ai_protocol))) break;
        if (state->fd < 0) {
            continue;
        }
        bool item_timed_out = false;
        const bool connected = timeout_seconds > 0
            ? connect_with_timeout(state->fd, item->ai_addr, item->ai_addrlen, timeout_seconds, &item_timed_out)
            : connect(state->fd, item->ai_addr, item->ai_addrlen) == 0;
        if (connected) {
            break;
        }
        connect_timed_out = connect_timed_out || item_timed_out;
        if (!item_timed_out) {
            last_connect_errno = errno;
            last_connect_endpoint = sockaddr_python_tuple(item->ai_addr);
        }
        state->close_fd();
    }
    freeaddrinfo(results);
    if (state->fd < 0) {
        if (connect_timed_out) {
            return worker_transport_error_response(aiohttp_connection_timeout_message(url, path));
        }
        return worker_transport_error_response(aiohttp_connect_error_message(parsed, last_connect_endpoint, last_connect_errno));
    }
    http_socket::low_latency(state->fd);
    if (timeout_seconds > 0) {
        timeval timeout{};
        timeout.tv_sec = timeout_seconds;
        timeout.tv_usec = 0;
        setsockopt(state->fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(state->fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }

    if (parsed.tls) {
        OPENSSL_init_ssl(0, nullptr);
        state->ssl_ctx = SSL_CTX_new(TLS_client_method());
        if (!state->ssl_ctx) {
            state->close_all();
            return worker_transport_error_response("Failed to create TLS context");
        }
        SSL_CTX_set_verify(state->ssl_ctx, SSL_VERIFY_PEER, nullptr);
        SSL_CTX_set_default_verify_paths(state->ssl_ctx);
        const char *cert_file = getenv("SSL_CERT_FILE");
        const char *cert_dir = getenv("SSL_CERT_DIR");
        if ((cert_file && *cert_file) || (cert_dir && *cert_dir)) {
            SSL_CTX_load_verify_locations(
                state->ssl_ctx,
                cert_file && *cert_file ? cert_file : nullptr,
                cert_dir && *cert_dir ? cert_dir : nullptr
            );
        }
        state->ssl = SSL_new(state->ssl_ctx);
        if (!state->ssl) {
            state->close_all();
            return worker_transport_error_response("Failed to create TLS connection");
        }
        if (!host_is_ip_address(parsed.host)) {
            SSL_set_tlsext_host_name(state->ssl, parsed.host.c_str());
            X509_VERIFY_PARAM_set1_host(SSL_get0_param(state->ssl), parsed.host.c_str(), 0);
        } else {
            X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(state->ssl), parsed.host.c_str());
        }
        SSL_set_fd(state->ssl, state->fd);
        if (SSL_connect(state->ssl) != 1) {
            state->close_all();
            return worker_transport_error_response("Failed to establish TLS connection to worker");
        }
        const long verify_result = SSL_get_verify_result(state->ssl);
        if (verify_result != X509_V_OK) {
            state->close_all();
            return worker_transport_error_response(std::string("Worker TLS certificate verification failed: ") + X509_verify_cert_error_string(verify_result));
        }
    }

    const std::string full_path = parsed.base_path + path;
    const int default_port = parsed.tls ? 443 : 80;
    const std::string host_header = parsed.port == default_port ? parsed.host : parsed.host + ":" + std::to_string(parsed.port);
    std::ostringstream request;
    request << method << " " << full_path << " HTTP/1.1\r\n"
        << "Host: " << host_header << "\r\n";
    request << "Connection: close\r\n";
    if (!body.empty() || method != "GET") {
        request << "Content-Length: " << body.size() << "\r\n";
    }
    if (!content_type.empty()) {
        request << "Content-Type: " << content_type << "\r\n";
    }
    if (!api_key.empty() && api_key != "none") {
        request << "Authorization: Bearer " << api_key << "\r\n";
    }
    for (const auto &header : forward_headers) {
        const auto name = to_lower(header.first);
        if (name == "host" || name == "connection" || name == "content-length"
            || name == "transfer-encoding" || name == "content-type"
            || (!api_key.empty() && name == "authorization")) continue;
        if (header.first.find_first_of("\r\n:") != std::string::npos
            || header.second.find_first_of("\r\n") != std::string::npos) continue;
        request << header.first << ": " << header.second << "\r\n";
    }
    request << "\r\n" << body;

    const std::string raw = request.str();
    size_t sent = 0;
    while (sent < raw.size()) {
        ssize_t n = state->ssl
            ? SSL_write(state->ssl, raw.data() + sent, static_cast<int>(std::min<size_t>(raw.size() - sent, 16384)))
            : send(state->fd, raw.data() + sent, raw.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            const int ssl_error = state->ssl ? SSL_get_error(state->ssl, static_cast<int>(n)) : 0;
            state->close_all();
            if (errno == EAGAIN || errno == EWOULDBLOCK
                || ssl_error == SSL_ERROR_WANT_READ
                || ssl_error == SSL_ERROR_WANT_WRITE) {
                return worker_transport_error_response("Worker request timed out");
            }
            return worker_transport_error_response("Failed to send request to worker");
        }
        sent += static_cast<size_t>(n);
    }
    if (!state->ssl && !opened_stream) {
        shutdown(state->fd, SHUT_WR);
    }

    std::string response_raw;
    char buffer[4096];
    size_t header_end = std::string::npos;
    while ((header_end = response_raw.find("\r\n\r\n")) == std::string::npos) {
        bool timed_out = false;
        const ssize_t n = state->read_some(buffer, sizeof(buffer), &timed_out);
        if (n < 0 && timed_out) {
            state->close_all();
            return worker_transport_error_response("Worker request timed out");
        }
        if (n <= 0) {
            state->close_all();
            return worker_transport_error_response("Malformed worker response");
        }
        response_raw.append(buffer, static_cast<size_t>(n));
    }

    std::istringstream status_line(response_raw.substr(0, response_raw.find("\r\n")));
    std::string http_version;
    int status = 502;
    status_line >> http_version >> status;
    const auto status_end = response_raw.find("\r\n");
    std::istringstream headers_stream(response_raw.substr(status_end + 2, header_end - status_end - 2));
    auto headers = parse_headers(headers_stream);
    std::string response_content_type = "application/json";
    auto content_type_it = headers.find("content-type");
    if (content_type_it != headers.end()) {
        response_content_type = content_type_it->second;
    }
    auto transfer_encoding_it = headers.find("transfer-encoding");
    state->upstream_chunked = transfer_encoding_it != headers.end()
        && to_lower(transfer_encoding_it->second).find("chunked") != std::string::npos;
    if (!state->upstream_chunked) {
        const auto length = headers.find("content-length");
        if (length != headers.end()) {
            try { state->content_length = std::stoull(length->second); } catch (const std::exception &) {}
        }
    }
    state->initial_body = response_raw.substr(header_end + 4);

    HttpResponse response{status, response_content_type, ""};
    if (opened_stream) {
        for (const auto &header : headers) response.headers.push_back(header);
        *opened_stream = state;
        return response;
    }
    response.chunked = true;
    response.stream_capture = std::make_shared<std::string>();
    state->stream_capture = response.stream_capture;
    response.stream_writer = [state](int client_fd) {
        stream_worker_body_to_client(client_fd, state);
    };
    return response;
}

HttpResponse http_json_request(
    const std::string &method,
    const std::string &url,
    const std::string &path,
    const std::string &api_key,
    const std::string &body,
    int timeout_seconds = 0,
    bool half_close_after_send = true,
    const HttpDeadline *deadline = nullptr
) {
    return http_worker_request(method, url, path, api_key, "application/json", body, timeout_seconds, half_close_after_send, nullptr, deadline);
}

std::string queue_snapshot_json(ezl_queue *queue) {
    ezl_queue_snapshot snapshot{};
    ezl_queue_get_snapshot(queue, &snapshot);
    std::ostringstream out;
    out << "{"
        << "\"max_queue_size\":" << snapshot.max_queue_size << ","
        << "\"max_concurrent_requests\":" << snapshot.max_concurrent_requests << ","
        << "\"queued\":" << snapshot.queued << ","
        << "\"processing\":" << snapshot.processing << ","
        << "\"active\":" << snapshot.active << ","
        << "\"total_queued\":" << snapshot.total_queued << ","
        << "\"total_processed\":" << snapshot.total_processed << ","
        << "\"total_failed\":" << snapshot.total_failed
        << "}";
    return out.str();
}

std::string http_reason(int status) {
    switch (status) {
        case 200:
            return "OK";
        case 202:
            return "Accepted";
        case 206:
            return "Partial Content";
        case 304:
            return "Not Modified";
        case 400:
            return "Bad Request";
        case 401:
            return "Unauthorized";
        case 403:
            return "Forbidden";
        case 404:
            return "Not Found";
        case 405:
            return "Method Not Allowed";
        case 408:
            return "Request Timeout";
        case 410:
            return "Gone";
        case 413:
            return "Payload Too Large";
        case 416:
            return "Range Not Satisfiable";
        case 422:
            return "Unprocessable Entity";
        case 500:
            return "Internal Server Error";
        case 502:
            return "Bad Gateway";
        case 501:
            return "Not Implemented";
        case 503:
            return "Service Unavailable";
        default:
            return "OK";
    }
}

bool response_has_header(const HttpResponse &response, const std::string &name) {
    const std::string wanted = to_lower(name);
    for (const auto &header : response.headers) {
        if (to_lower(header.first) == wanted) {
            return true;
        }
    }
    return false;
}

HttpResponse cors_preflight_response(
    const std::string &origin,
    const std::string &request_headers
) {
    HttpResponse response{200, "text/plain; charset=utf-8", "OK"};
    response.headers.push_back({"Vary", "Origin"});
    response.headers.push_back({"Access-Control-Allow-Methods", "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT"});
    response.headers.push_back({"Access-Control-Max-Age", "600"});
    response.headers.push_back({"Access-Control-Allow-Credentials", "true"});
    response.headers.push_back({"Access-Control-Allow-Origin", origin});
    if (!request_headers.empty()) {
        response.headers.push_back({"Access-Control-Allow-Headers", request_headers});
    }
    response.add_default_cors_headers = false;
    return response;
}

std::string serialize_response(
    const HttpResponse &response,
    bool head_request = false,
    const std::string &request_origin = "",
    bool request_has_cookie = false
) {
    std::ostringstream out;
    out << "HTTP/1.1 " << response.status << " " << http_reason(response.status) << "\r\n";
    auto write_cors_headers = [&]() {
        if (response.add_default_cors_headers && !request_origin.empty()) {
            if (!response_has_header(response, "Access-Control-Allow-Origin")) {
                out << "Access-Control-Allow-Origin: " << (request_has_cookie ? request_origin : "*") << "\r\n";
            }
            if (!response_has_header(response, "Access-Control-Allow-Credentials")) {
                out << "Access-Control-Allow-Credentials: true\r\n";
            }
            if (request_has_cookie && !response_has_header(response, "Vary")) {
                out << "Vary: " << (native_router_mode_enabled() ? "Origin" : "Origin, Origin") << "\r\n";
            }
        }
    };
    if (response.header_order == HttpResponse::HeaderOrder::StaticFile && response.status == 304) {
        for (const auto &header : response.headers) {
            out << header.first << ": " << header.second << "\r\n";
        }
        write_cors_headers();
        out << "Connection: close\r\n";
        out << "\r\n";
        return out.str();
    }
    if (response.header_order == HttpResponse::HeaderOrder::StaticFile) {
        if (!response.content_type.empty()) {
            out << "Content-Type: " << response.content_type << "\r\n";
        }
        for (const auto &header : response.headers) {
            if (to_lower(header.first) == "accept-ranges" || to_lower(header.first) == "content-disposition") {
                out << header.first << ": " << header.second << "\r\n";
            }
        }
        out << "Content-Length: " << response.content_length.value_or(response.body.size()) << "\r\n";
        for (const auto &header : response.headers) {
            if (to_lower(header.first) != "accept-ranges" && to_lower(header.first) != "content-disposition") {
                out << header.first << ": " << header.second << "\r\n";
            }
        }
        write_cors_headers();
        out << "Connection: close\r\n";
        out << "\r\n";
        if (!head_request && !response.stream_writer) {
            out << response.body;
        }
        return out.str();
    }
    if (response.header_order == HttpResponse::HeaderOrder::StaticRangeError) {
        out << "Content-Length: " << response.body.size() << "\r\n";
        if (!response.content_type.empty()) {
            out << "Content-Type: " << response.content_type << "\r\n";
        }
        write_cors_headers();
        out << "Connection: close\r\n";
        out << "\r\n";
        if (!head_request) {
            out << response.body;
        }
        return out.str();
    }
    if (response.header_order == HttpResponse::HeaderOrder::StaticUnsatisfiable) {
        for (const auto &header : response.headers) {
            if (to_lower(header.first) == "content-range") {
                out << header.first << ": " << header.second << "\r\n";
            }
        }
        out << "Content-Length: " << response.body.size() << "\r\n";
        if (!response.content_type.empty()) {
            out << "Content-Type: " << response.content_type << "\r\n";
        }
        for (const auto &header : response.headers) {
            if (to_lower(header.first) != "content-range") {
                out << header.first << ": " << header.second << "\r\n";
            }
        }
        write_cors_headers();
        out << "Connection: close\r\n";
        out << "\r\n";
        if (!head_request) {
            out << response.body;
        }
        return out.str();
    }
    for (const auto &header : response.headers) {
        out << header.first << ": " << header.second << "\r\n";
    }
    if (response.status != 304 && response.chunked) {
        out << "Transfer-Encoding: chunked\r\n";
    } else if (response.status != 304) {
        out << "Content-Length: " << response.body.size() << "\r\n";
    }
    if (response.status != 304 && !response.content_type.empty()) {
        out << "Content-Type: " << response.content_type << "\r\n";
    }
    write_cors_headers();
    out << "Connection: close\r\n";
    out << "\r\n";
    if (!head_request && !response.stream_writer) {
        if (response.chunked && response.status != 304) {
            if (!response.body.empty()) {
                out << std::hex << response.body.size() << "\r\n" << response.body << "\r\n";
            }
            out << "0\r\n\r\n";
        } else {
            out << response.body;
        }
    }
    return out.str();
}

uint32_t sha1_left_rotate(uint32_t value, unsigned int count) {
    return (value << count) | (value >> (32U - count));
}

std::array<unsigned char, 20> sha1_digest(const std::string &input) {
    std::vector<unsigned char> msg(input.begin(), input.end());
    const uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8ULL;
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) {
        msg.push_back(0x00);
    }
    for (int i = 7; i >= 0; --i) {
        msg.push_back(static_cast<unsigned char>((bit_len >> (i * 8)) & 0xff));
    }

    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xefcdab89;
    uint32_t h2 = 0x98badcfe;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xc3d2e1f0;

    for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        uint32_t w[80]{};
        for (int i = 0; i < 16; ++i) {
            const size_t j = chunk + static_cast<size_t>(i) * 4;
            w[i] = (static_cast<uint32_t>(msg[j]) << 24)
                | (static_cast<uint32_t>(msg[j + 1]) << 16)
                | (static_cast<uint32_t>(msg[j + 2]) << 8)
                | static_cast<uint32_t>(msg[j + 3]);
        }
        for (int i = 16; i < 80; ++i) {
            w[i] = sha1_left_rotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }

        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;
        for (int i = 0; i < 80; ++i) {
            uint32_t f = 0;
            uint32_t k = 0;
            if (i < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5a827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ed9eba1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8f1bbcdc;
            } else {
                f = b ^ c ^ d;
                k = 0xca62c1d6;
            }
            const uint32_t temp = sha1_left_rotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = sha1_left_rotate(b, 30);
            b = a;
            a = temp;
        }
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    std::array<unsigned char, 20> digest{};
    const uint32_t hs[] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i) {
        digest[static_cast<size_t>(i) * 4] = static_cast<unsigned char>((hs[i] >> 24) & 0xff);
        digest[static_cast<size_t>(i) * 4 + 1] = static_cast<unsigned char>((hs[i] >> 16) & 0xff);
        digest[static_cast<size_t>(i) * 4 + 2] = static_cast<unsigned char>((hs[i] >> 8) & 0xff);
        digest[static_cast<size_t>(i) * 4 + 3] = static_cast<unsigned char>(hs[i] & 0xff);
    }
    return digest;
}

std::string base64_encode(const unsigned char *data, size_t len) {
    static const char *table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < len; i += 3) {
        const uint32_t octet_a = data[i];
        const uint32_t octet_b = (i + 1 < len) ? data[i + 1] : 0;
        const uint32_t octet_c = (i + 2 < len) ? data[i + 2] : 0;
        const uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;
        out.push_back(table[(triple >> 18) & 0x3f]);
        out.push_back(table[(triple >> 12) & 0x3f]);
        out.push_back(i + 1 < len ? table[(triple >> 6) & 0x3f] : '=');
        out.push_back(i + 2 < len ? table[triple & 0x3f] : '=');
    }
    return out;
}

std::string websocket_accept_value(const std::string &key) {
    const auto digest = sha1_digest(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    return base64_encode(digest.data(), digest.size());
}

std::string join_strings(const std::vector<std::string> &values, const std::string &separator) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out << separator;
        }
        out << values[i];
    }
    return out.str();
}

std::vector<std::string> allowed_methods_for_route_path(const std::string &route_path, bool router_mode_enabled) {
    if (route_path == "/" || route_path == "/openapi.json" || route_path == "/docs/oauth2-redirect" || route_path == "/redoc") {
        return {"HEAD", "GET"};
    }
    if (starts_with(route_path, "/outputs/")) {
        return {"HEAD", "GET"};
    }

    if (route_path == "/health" || route_path == "/v1/models") {
        return {"GET"};
    }
    if (route_path == "/v1/chat/completions"
        || route_path == "/v1/completions"
        || route_path == "/v1/embeddings"
        || route_path == "/v1/audio/speech"
        || route_path == "/v1/audio/speech/stream"
        || route_path == "/v1/audio/music"
        || route_path == "/v1/audio/music/generations"
        || route_path == "/v1/videos/music"
        || route_path == "/v1/videos/music/generations"
        || route_path == "/v1/audio/transcriptions"
        || route_path == "/v1/images/generations"
        || route_path == "/v1/images/edits"
        || route_path == "/v1/videos/generations") {
        return {"POST"};
    }

    if (router_mode_enabled) {
        if (route_path == "/v1/router/health"
            || route_path == "/v1/router/workers"
            || route_path == "/v1/router/dashboard"
            || route_path == "/v1/router/tunnels"
            || route_path == "/v1/router/usage"
            || route_path == "/v1/router/history"
            || route_path == "/v1/router/errors"
            || route_path == "/dashboard"
            || route_path == "/v1/audio/voices") {
            return {"GET"};
        }
        if (route_path == "/v1/router/register"
            || route_path == "/v1/router/heartbeat"
            || route_path == "/v1/router/deregister"
            || route_path == "/v1/router/drain"
            || route_path == "/v1/router/resume") {
            return {"POST"};
        }
        return {};
    }

    if (route_path == "/v1/resources"
        || route_path == "/v1/fallback/status"
        || route_path == "/v1/queue"
        || route_path == "/v1/queue/status"
        || matches_template(route_path, "/v1/queue/request/{request_id}")
        || route_path == "/v1/wakeword/models"
        || matches_template(route_path, "/v1/wakeword/models/{word}/config")
        || route_path == "/v1/wakeword/jobs") {
        return {"GET"};
    }
    if (route_path == "/v1/audio/translations"
        || route_path == "/v1/audio/subtitles"
        || route_path == "/v1/wakeword/train"
        || matches_template(route_path, "/v1/wakeword/predict/{word}")
        || matches_template(route_path, "/v1/wakeword/predict/{word}/file")) {
        return {"POST"};
    }
    if (route_path == "/v1/audio/voices") {
        return {"GET", "POST"};
    }
    if (matches_template(route_path, "/v1/wakeword/models/{word}")) {
        return {"DELETE", "GET"};
    }
    if (matches_template(route_path, "/v1/wakeword/jobs/{job_id}")) {
        return {"DELETE", "GET"};
    }
    return {};
}

bool websocket_route_available(const std::string &route_path, bool router_mode_enabled) {
    if (route_path == "/v1/router/tunnel") {
        return router_mode_enabled;
    }
    if (route_path == "/v1/audio/speech/ws") {
        return !router_mode_enabled;
    }
    return false;
}

HttpResponse websocket_plain_response(int status, const std::string &body) {
    HttpResponse response{status, "text/plain", body};
    response.add_default_cors_headers = false;
    return response;
}

HttpResponse websocket_bad_request_response(const std::string &message) {
    return websocket_plain_response(
        400,
        "Failed to open a WebSocket connection: " + message + "\n"
    );
}

HttpResponse invalid_http_request_response() {
    HttpResponse response{400, "text/plain; charset=utf-8", "Invalid HTTP request received."};
    response.add_default_cors_headers = false;
    return response;
}

bool is_websocket_upgrade(const std::unordered_map<std::string, std::string> &headers) {
    auto upgrade = headers.find("upgrade");
    if (upgrade == headers.end() || to_lower(upgrade->second) != "websocket") {
        return false;
    }
    auto connection = headers.find("connection");
    return connection != headers.end() && to_lower(connection->second).find("upgrade") != std::string::npos;
}

std::string websocket_close_frame(int code, const std::string &reason = "") {
    std::string payload;
    payload.push_back(static_cast<char>((code >> 8) & 0xff));
    payload.push_back(static_cast<char>(code & 0xff));
    payload += reason.substr(0, 123);

    std::string frame;
    frame.push_back(static_cast<char>(0x88));
    frame.push_back(static_cast<char>(payload.size()));
    frame += payload;
    return frame;
}

std::string websocket_pong_frame() {
    std::string frame;
    frame.push_back(static_cast<char>(0x8a));
    frame.push_back(static_cast<char>(0x00));
    return frame;
}

std::string websocket_data_frame(unsigned char opcode, const std::string &payload) {
    std::string frame;
    frame.push_back(static_cast<char>(0x80U | (opcode & 0x0fU)));
    const uint64_t len = payload.size();
    if (len < 126) {
        frame.push_back(static_cast<char>(len));
    } else if (len <= 0xffff) {
        frame.push_back(static_cast<char>(126));
        frame.push_back(static_cast<char>((len >> 8) & 0xff));
        frame.push_back(static_cast<char>(len & 0xff));
    } else {
        frame.push_back(static_cast<char>(127));
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<char>((len >> shift) & 0xff));
        }
    }
    frame += payload;
    return frame;
}

std::string websocket_text_frame(const std::string &payload) {
    return websocket_data_frame(0x1U, payload);
}

std::string websocket_binary_frame(const std::string &payload) {
    return websocket_data_frame(0x2U, payload);
}

int base64_decode_value(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') {
        return ch - 'A';
    }
    if (ch >= 'a' && ch <= 'z') {
        return 26 + ch - 'a';
    }
    if (ch >= '0' && ch <= '9') {
        return 52 + ch - '0';
    }
    if (ch == '+') {
        return 62;
    }
    if (ch == '/') {
        return 63;
    }
    return -1;
}

std::string base64_decode_string(const std::string &input) {
    std::string out;
    int value = 0;
    int bits = -8;
    for (unsigned char ch : input) {
        if (std::isspace(ch)) {
            continue;
        }
        if (ch == '=') {
            break;
        }
        const int decoded = base64_decode_value(ch);
        if (decoded < 0) {
            break;
        }
        value = (value << 6) | decoded;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<char>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

bool valid_websocket_key(const std::string &key) {
    const std::string decoded = base64_decode_string(key);
    if (decoded.size() != 16) {
        return false;
    }
    return base64_encode(reinterpret_cast<const unsigned char *>(decoded.data()), decoded.size()) == key;
}

struct WebSocketFrame {
    unsigned char opcode = 0;
    std::string payload;
};

bool pop_websocket_frame(std::string *buffer, WebSocketFrame *frame) {
    if (buffer->size() < 2) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>((*buffer)[0]);
    const unsigned char second = static_cast<unsigned char>((*buffer)[1]);
    uint64_t len = second & 0x7fU;
    size_t pos = 2;
    if (len == 126) {
        if (buffer->size() < pos + 2) {
            return false;
        }
        len = (static_cast<uint64_t>(static_cast<unsigned char>((*buffer)[pos])) << 8)
            | static_cast<unsigned char>((*buffer)[pos + 1]);
        pos += 2;
    } else if (len == 127) {
        if (buffer->size() < pos + 8) {
            return false;
        }
        len = 0;
        for (int i = 0; i < 8; ++i) {
            len = (len << 8) | static_cast<unsigned char>((*buffer)[pos + static_cast<size_t>(i)]);
        }
        pos += 8;
    }
    if (len > 16 * 1024 * 1024) {
        buffer->clear();
        return false;
    }
    const bool masked = (second & 0x80U) != 0;
    unsigned char mask[4] = {0, 0, 0, 0};
    if (masked) {
        if (buffer->size() < pos + 4) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            mask[i] = static_cast<unsigned char>((*buffer)[pos + static_cast<size_t>(i)]);
        }
        pos += 4;
    }
    if (buffer->size() < pos + static_cast<size_t>(len)) {
        return false;
    }
    frame->opcode = first & 0x0fU;
    frame->payload.assign(buffer->data() + pos, buffer->data() + pos + static_cast<size_t>(len));
    if (masked) {
        for (size_t i = 0; i < frame->payload.size(); ++i) {
            frame->payload[i] = static_cast<char>(static_cast<unsigned char>(frame->payload[i]) ^ mask[i % 4]);
        }
    }
    buffer->erase(0, pos + static_cast<size_t>(len));
    return true;
}

std::string url_encode_component(const std::string &value) {
    std::ostringstream out;
    out << std::hex << std::uppercase << std::setfill('0');
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            out << static_cast<char>(ch);
        } else {
            out << "%" << std::setw(2) << static_cast<int>(ch);
        }
    }
    return out.str();
}


std::unordered_map<std::string, std::string> parse_headers(std::istream &stream) {
    std::unordered_map<std::string, std::string> headers;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            break;
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        headers[to_lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }
    return headers;
}

std::string read_socket_request(int fd) {
    std::string buffer;
    char chunk[4096];
    while (buffer.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            return buffer;
        }
        buffer.append(chunk, static_cast<size_t>(n));
        if (buffer.size() > 1024 * 1024) {
            return buffer;
        }
    }

    std::istringstream head(buffer.substr(0, buffer.find("\r\n\r\n") + 4));
    std::string request_line;
    std::getline(head, request_line);
    auto headers = parse_headers(head);
    size_t content_length = 0;
    auto it = headers.find("content-length");
    if (it != headers.end()) {
        content_length = static_cast<size_t>(std::stoul(it->second));
    }
    const size_t body_start = buffer.find("\r\n\r\n") + 4;
    auto transfer_encoding_it = headers.find("transfer-encoding");
    if (transfer_encoding_it != headers.end()
        && to_lower(transfer_encoding_it->second).find("chunked") != std::string::npos) {
        while (true) {
            std::string decoded_body;
            const auto decoded = decode_http_chunked_body_state(buffer.substr(body_start), &decoded_body);
            if (decoded == ChunkedDecodeResult::Complete) {
                return buffer.substr(0, body_start) + decoded_body;
            }
            if (decoded == ChunkedDecodeResult::Malformed) {
                return buffer;
            }
            ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
            if (n <= 0) {
                return buffer;
            }
            buffer.append(chunk, static_cast<size_t>(n));
            if (buffer.size() > 16 * 1024 * 1024) {
                return buffer;
            }
        }
    }
    while (buffer.size() < body_start + content_length) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            break;
        }
        buffer.append(chunk, static_cast<size_t>(n));
        if (buffer.size() > 16 * 1024 * 1024) {
            break;
        }
    }
    return buffer;
}

} // namespace

std::string json_escape(const std::string &value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[static_cast<unsigned char>(ch) >> 4];
                    out += hex[static_cast<unsigned char>(ch) & 15];
                } else {
                    out.push_back(ch);
                }
                break;
        }
    }
    return out;
}

std::string extract_json_string(const std::string &json, const std::string &key, const std::string &fallback) {
    const size_t start = json_value_start(json, key);
    const size_t end = json_value_end(json, start);
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        return fallback;
    }
    const std::string raw = trim(json.substr(start, end - start));
    if (raw.empty() || raw == "null" || raw.front() == '{' || raw.front() == '[') {
        return fallback;
    }
    if (raw.front() != '"') {
        return raw;
    }
    auto parsed = parse_json_string_at(json, start);
    return parsed.empty() ? fallback : parsed;
}

std::string python_string_repr(const std::string &value) {
    const char quote = value.find('\'') != std::string::npos && value.find('"') == std::string::npos ? '"' : '\'';
    std::string out(1, quote);
    for (size_t i = 0; i < value.size();) {
        const size_t begin = i; const auto first = static_cast<unsigned char>(value[i++]);
        uint32_t cp = first;
        if (first >= 0xc0) {
            const unsigned remaining = first < 0xe0 ? 1 : first < 0xf0 ? 2 : 3;
            cp &= 0x7f >> remaining;
            for (unsigned j = 0; j < remaining && i < value.size(); ++j) cp = (cp << 6) | (static_cast<unsigned char>(value[i++]) & 0x3f);
        }
        if (cp == '\\' || cp == static_cast<unsigned char>(quote)) { out += '\\'; out += static_cast<char>(cp); }
        else if (cp == '\n') out += "\\n";
        else if (cp == '\r') out += "\\r";
        else if (cp == '\t') out += "\\t";
        else {
            const auto category = u_charType(cp);
            const bool printable = cp == 0x20 || (category != U_UNASSIGNED && category != U_CONTROL_CHAR && category != U_FORMAT_CHAR
                && category != U_PRIVATE_USE_CHAR && category != U_SURROGATE && category != U_SPACE_SEPARATOR
                && category != U_LINE_SEPARATOR && category != U_PARAGRAPH_SEPARATOR);
            if (printable) out.append(value, begin, i - begin);
            else { char escape[12]; std::snprintf(escape, sizeof(escape), cp <= 0xff ? "\\x%02x" : cp <= 0xffff ? "\\u%04x" : "\\U%08x", cp); out += escape; }
        }
    }
    out += quote; return out;
}
namespace {
std::string startup_repr(const std::string &raw) {
    const auto value = trim(raw);
    if (value == "true") return "True";
    if (value == "false") return "False";
    if (value == "null") return "None";
    if (!value.empty() && value.front() == '"') return python_string_repr(parse_json_string_at(value, 0));
    if (!value.empty() && value.front() == '[') {
        std::string out = "["; bool first = true;
        for (const auto &item : json_array_elements(value)) { if (!first) out += ", "; first = false; out += startup_repr(item); }
        return out + ']';
    }
    if (!value.empty() && value.front() == '{') {
        std::vector<std::pair<std::string, std::string>> entries;
        for (const auto &entry : json_object_ordered_entries(value)) ordered_json_upsert(&entries, entry);
        std::string out = "{"; bool first = true;
        for (const auto &entry : entries) { if (!first) out += ", "; first = false; out += python_string_repr(entry.first) + ": " + startup_repr(entry.second); }
        return out + '}';
    }
    if (value.find_first_of(".eE") == std::string::npos && is_json_number_token(value)) return value == "-0" ? "0" : value;
    double number = 0;
    if (parse_strict_python_float_string(value, &number)) return python_float_string(number);
    return value;
}
}
std::optional<std::string> llm_startup_failure(const std::string &model, const std::string &result, const std::optional<std::string> &exception) {
    const auto prefix = "Failed to initialize LLM server for " + model + ": ";
    if (exception) return prefix + *exception;
    const auto value = trim(result);
    if (!value.empty() && value.front() == '{') {
        for (const auto &entry : json_object_ordered_entries(value)) if (entry.first == "error") return prefix + "LLM server test failed: " + startup_repr(value);
    }
    return std::nullopt;
}

std::string python_str_from_json_value(const std::string &raw_value);
std::string python_float_string(double value);

std::string extract_router_worker_id(const std::string &json) {
    const auto values = top_level_json_object_values(json);
    const auto found = values.find("worker_id");
    const std::string raw_value = found == values.end() ? "" : trim(found->second);
    if (is_python_falsey_json_value(raw_value)) {
        return "";
    }
    return python_str_from_json_value(raw_value);
}

bool extract_json_bool(const std::string &json, const std::string &key, bool fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    size_t begin = json.find_first_not_of(" \t\r\n", colon + 1);
    if (begin == std::string::npos) {
        return fallback;
    }
    if (json.compare(begin, 4, "true") == 0) {
        return true;
    }
    if (json.compare(begin, 5, "false") == 0) {
        return false;
    }
    const std::string raw = trim(extract_json_value(json, key, ""));
    if (!raw.empty() && raw.front() == '"') {
        bool parsed = false;
        if (parse_pydantic_bool_string(parse_json_string_at(raw, 0), &parsed)) {
            return parsed;
        }
    } else if (is_json_number_token(raw)) {
        try {
            const long double parsed = std::stold(raw);
            if (parsed == 0.0L || parsed == 1.0L) {
                return parsed == 1.0L;
            }
        } catch (...) {
        }
    }
    return fallback;
}

std::string python_repr_string(const std::string &value) {
    const bool has_single_quote = value.find('\'') != std::string::npos;
    const bool has_double_quote = value.find('"') != std::string::npos;
    const char quote = (has_single_quote && !has_double_quote) ? '"' : '\'';
    std::ostringstream out;
    out << quote;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '\'':
                if (quote == '\'') {
                    out << "\\'";
                } else {
                    out << ch;
                }
                break;
            case '"':
                if (quote == '"') {
                    out << "\\\"";
                } else {
                    out << ch;
                }
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    out << quote;
    return out.str();
}

std::vector<std::pair<std::string, std::string>> json_object_ordered_entries(const std::string &raw) {
    std::vector<std::pair<std::string, std::string>> entries;
    const std::string json = trim(raw);
    size_t pos = json.find_first_not_of(" \t\r\n");
    if (pos == std::string::npos || json[pos] != '{') {
        return entries;
    }
    ++pos;
    while (pos < json.size()) {
        pos = json.find_first_not_of(" \t\r\n,", pos);
        if (pos == std::string::npos || pos >= json.size() || json[pos] == '}') {
            break;
        }
        if (json[pos] != '"') {
            break;
        }
        const std::string key = parse_json_string_at(json, pos);
        const size_t key_end = json_value_end(json, pos);
        if (key_end == std::string::npos) {
            break;
        }
        const size_t colon = json.find(':', key_end);
        if (colon == std::string::npos) {
            break;
        }
        const size_t value_start = json.find_first_not_of(" \t\r\n", colon + 1);
        const size_t value_end = json_value_end(json, value_start);
        if (value_start == std::string::npos || value_end == std::string::npos || value_end <= value_start) {
            break;
        }
        entries.emplace_back(key, json.substr(value_start, value_end - value_start));
        pos = value_end;
    }
    return entries;
}

std::string python_list_json_from_value(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (is_python_falsey_json_value(value)) {
        return "[]";
    }
    if (value.front() == '[') {
        return value;
    }
    if (value.front() == '"') {
        std::vector<std::string> chars;
        for (const char ch : parse_json_string_at(value, 0)) {
            chars.push_back(std::string(1, ch));
        }
        return string_array_json(chars);
    }
    if (value.front() == '{') {
        std::vector<std::string> keys;
        for (const auto &entry : json_object_ordered_entries(value)) {
            keys.push_back(entry.first);
        }
        return string_array_json(keys);
    }
    return "[]";
}

std::vector<std::string> python_iterable_json_values(const std::string &raw_value, bool *ok) {
    if (ok) {
        *ok = true;
    }
    const std::string value = trim(raw_value);
    if (is_python_falsey_json_value(value)) {
        return {};
    }
    if (value.front() == '[') {
        return json_array_elements(value);
    }
    if (value.front() == '"') {
        std::vector<std::string> chars;
        for (const char ch : parse_json_string_at(value, 0)) {
            chars.push_back("\"" + json_escape(std::string(1, ch)) + "\"");
        }
        return chars;
    }
    if (value.front() == '{') {
        std::vector<std::string> keys;
        for (const auto &entry : json_object_ordered_entries(value)) {
            keys.push_back("\"" + json_escape(entry.first) + "\"");
        }
        return keys;
    }
    if (ok) {
        *ok = false;
    }
    return {};
}

std::string python_repr_json_value(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (value.empty()) {
        return "";
    }
    if (value == "true") {
        return "True";
    }
    if (value == "false") {
        return "False";
    }
    if (value == "null") {
        return "None";
    }
    if (value.front() == '"') {
        return python_repr_string(parse_json_string_at(value, 0));
    }
    if (value.front() == '[') {
        std::ostringstream out;
        out << "[";
        const auto elements = json_array_elements(value);
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << python_repr_json_value(elements[i]);
        }
        out << "]";
        return out.str();
    }
    if (value.front() == '{') {
        std::ostringstream out;
        out << "{";
        const auto entries = json_object_ordered_entries(value);
        for (size_t i = 0; i < entries.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << python_repr_string(entries[i].first) << ": " << python_repr_json_value(entries[i].second);
        }
        out << "}";
        return out.str();
    }
    if (is_json_number_token(value)) {
        const bool float_token = value.find_first_of(".eE") != std::string::npos;
        try {
            if (float_token) {
                size_t consumed = 0;
                const double parsed = std::stod(value, &consumed);
                if (consumed == value.size()) {
                    return python_float_string(parsed);
                }
            } else {
                size_t consumed = 0;
                const long long parsed = std::stoll(value, &consumed);
                if (consumed == value.size()) {
                    return std::to_string(parsed);
                }
            }
        } catch (...) {
        }
    }
    return value;
}

std::string python_str_from_json_value(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (value.empty()) {
        return "";
    }
    if (value.front() == '"') {
        return parse_json_string_at(value, 0);
    }
    return python_repr_json_value(value);
}

std::string python_truthy_string_field_or_default(const std::string &json,
                                                  const std::string &key,
                                                  const std::string &fallback) {
    const std::string raw = trim(extract_json_value(json, key, ""));
    if (is_python_falsey_json_value(raw)) {
        return fallback;
    }
    return python_str_from_json_value(raw);
}

std::string python_truthy_string_value_or_default(const std::string &raw_value,
                                                  const std::string &fallback) {
    const std::string raw = trim(raw_value);
    if (is_python_falsey_json_value(raw)) {
        return fallback;
    }
    return python_str_from_json_value(raw);
}

std::string router_payload_model_or_empty(const std::string &json) {
    const std::string raw = trim(extract_json_value(json, "model", ""));
    if (is_python_falsey_json_value(raw)) {
        return "";
    }
    return python_str_from_json_value(raw);
}

std::string router_payload_model_or_unknown(const std::string &json) {
    const std::string model = router_payload_model_or_empty(json);
    return model.empty() ? "unknown" : model;
}

bool router_chat_payload_needs_vision(const std::string &json) {
    for (const auto &message : extract_json_array_elements(json, "messages")) {
        const std::string content = trim(extract_json_value(message, "content", ""));
        if (content.empty() || content.front() != '[') {
            continue;
        }
        for (const auto &part : json_array_elements(content)) {
            const std::string part_type = extract_json_string(part, "type", "");
            if (part_type == "image_url" || part_type == "input_image") {
                return true;
            }
        }
    }
    return false;
}

std::string python_string_map_json_from_value(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (is_python_falsey_json_value(value) || value.front() != '{') {
        return "{}";
    }
    const auto entries = json_object_ordered_entries(value);
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (const auto &entry : entries) {
        if (is_python_falsey_json_value(entry.second)) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << json_escape(entry.first) << "\":"
            << "\"" << json_escape(python_str_from_json_value(entry.second)) << "\"";
    }
    out << "}";
    return out.str();
}

int python_int_from_json_value(const std::string &raw_value) {
    std::string value = trim(raw_value);
    if (value == "true") {
        return 1;
    }
    if (value == "false") {
        return 0;
    }
    if (!value.empty() && value.front() == '"') {
        value = trim(parse_json_string_at(value, 0));
        int parsed = 0;
        if (!parse_python_int_string(value, &parsed)) {
            throw std::invalid_argument("invalid literal for int()");
        }
        return parsed;
    }
    return static_cast<int>(std::stold(value));
}

std::string python_int_map_json_from_value(const std::string &raw_value) {
    const std::string value = trim(raw_value);
    if (is_python_falsey_json_value(value) || value.front() != '{') {
        return "{}";
    }
    const auto entries = json_object_ordered_entries(value);
    std::ostringstream out;
    out << "{";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "\"" << json_escape(entries[i].first) << "\":"
            << python_int_from_json_value(entries[i].second);
    }
    out << "}";
    return out.str();
}

const char kVoiceForwardBoundary[] = "ezlocalai-native-voice-boundary";

std::string voice_server_api_key(const std::string &fallback) {
    std::string api_key = ezl_getenv_default("VOICE_SERVER_API_KEY");
    if (api_key.empty()) {
        api_key = fallback;
    }
    return api_key;
}



std::string tts_optional_string_json(const std::string &body, const std::string &key, const std::string &fallback) {
    const std::string raw = trim(extract_json_value(body, key, ""));
    if (raw.empty()) {
        return "\"" + json_escape(fallback) + "\"";
    }
    if (raw == "null") {
        return std::string("null");
    }
    const std::string value = raw.front() == '"' ? parse_json_string_at(raw, 0) : raw;
    return "\"" + json_escape(value) + "\"";
}

std::string voice_tts_forward_payload(const std::string &body) {
    const auto input = extract_json_string(body, "input", "");
    std::ostringstream out;
    out << "{"
        << "\"input\":\"" << json_escape(input) << "\","
        << "\"voice\":" << tts_optional_string_json(body, "voice", "default") << ","
        << "\"language\":" << tts_optional_string_json(body, "language", "en")
        << "}";
    return out.str();
}

std::string tts_text_payload_from_original(const std::string &original_body, const std::string &text) {
    std::ostringstream out;
    out << "{"
        << "\"input\":\"" << json_escape(text) << "\","
        << "\"voice\":" << tts_optional_string_json(original_body, "voice", "default") << ","
        << "\"language\":" << tts_optional_string_json(original_body, "language", "en")
        << "}";
    return out.str();
}

struct ParsedDataUri {
    bool ok = false;
    std::string media_type;
    std::string data;
};

ParsedDataUri parse_data_uri(const std::string &uri) {
    ParsedDataUri parsed;
    if (!starts_with(uri, "data:")) {
        return parsed;
    }
    const size_t comma = uri.find(',');
    if (comma == std::string::npos) {
        return parsed;
    }
    const std::string metadata = uri.substr(5, comma - 5);
    const std::string metadata_lower = to_lower(metadata);
    const size_t semicolon = metadata_lower.find(';');
    parsed.media_type = semicolon == std::string::npos
        ? metadata_lower
        : metadata_lower.substr(0, semicolon);
    const std::string payload = uri.substr(comma + 1);
    parsed.data = metadata_lower.find(";base64") != std::string::npos
        ? base64_decode_string(payload)
        : url_decode(payload);
    parsed.ok = !parsed.media_type.empty() && !parsed.data.empty();
    return parsed;
}

const MultipartPart *find_multipart_part(const std::vector<MultipartPart> &parts, const std::string &name) {
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        if (it->name == name) {
            return &*it;
        }
    }
    return nullptr;
}

std::string multipart_value_or(const std::vector<MultipartPart> &parts, const std::string &name, const std::string &fallback) {
    const auto *part = find_multipart_part(parts, name);
    return part ? part->content : fallback;
}

std::string python_float_string(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value < 0 ? "-inf" : "inf";
    }

    char buffer[128];
    const double abs_value = std::fabs(value);
    const bool fixed_python_range = abs_value == 0.0 || (abs_value >= 1e-4 && abs_value < 1e16);
    auto result = fixed_python_range
        ? std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed)
        : std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec == std::errc()) {
        std::string text(buffer, result.ptr);
        if (text.find_first_of(".eE") == std::string::npos) {
            text += ".0";
        }
        return text;
    }

    std::ostringstream out;
    out << std::setprecision(15) << value;
    std::string text = out.str();
    if (text.find_first_of(".eE") == std::string::npos) {
        text += ".0";
    }
    return text;
}

std::string multipart_python_float_value_or(const std::vector<MultipartPart> &parts,
                                            const std::string &name,
                                            const std::string &fallback) {
    const auto *part = find_multipart_part(parts, name);
    if (!part) {
        return fallback;
    }
    try {
        const std::string value = trim(part->content);
        double parsed = 0.0;
        if (parse_strict_python_float_string(value, &parsed)) {
            return python_float_string(parsed);
        }
    } catch (...) {
    }
    return part->content;
}

std::string multipart_python_bool_value(const MultipartPart &part) {
    bool parsed = false;
    if (parse_pydantic_bool_string(part.content, &parsed)) {
        return parsed ? "True" : "False";
    }
    return part.content;
}

void append_whisper_temperature_arg(const std::vector<MultipartPart> &parts, std::vector<std::string> *args) {
    if (!args) {
        return;
    }
    args->push_back("-tp");
    args->push_back(multipart_python_float_value_or(parts, "temperature", "0.0"));
}

void append_whisper_beam_size_arg(const std::vector<MultipartPart> &parts,
                                  std::vector<std::string> *args,
                                  int fallback) {
    if (!args) {
        return;
    }
    int beam_size = fallback;
    if (const auto *part = find_multipart_part(parts, "beam_size")) {
        parse_pydantic_integral_string(part->content, &beam_size);
    }
    args->push_back("-bs");
    args->push_back(std::to_string(beam_size));
}

void append_whisper_subtitle_decode_args(const std::vector<MultipartPart> &parts, std::vector<std::string> *args) {
    if (!args) {
        return;
    }
    bool fast_mode = false;
    if (const auto *part = find_multipart_part(parts, "fast_mode")) {
        parse_pydantic_bool_string(part->content, &fast_mode);
    }
    if (fast_mode) {
        args->push_back("-bs");
        args->push_back("1");
        args->push_back("-mc");
        args->push_back("0");
        return;
    }
    append_whisper_beam_size_arg(parts, args, 5);
}

std::string build_voice_transcription_payload(const std::string &body, const std::string &content_type, bool fallback_semantics) {
    const auto boundary = extract_multipart_boundary(content_type);
    const auto parts = parse_multipart_parts(body, boundary);
    const auto *file = find_multipart_part(parts, "file");
    if (!file) {
        return "";
    }

    const auto add_field = [](std::ostringstream &out, const std::string &name, const std::string &value) {
        out << "--" << kVoiceForwardBoundary << "\r\n"
            << "Content-Disposition: form-data; name=\"" << name << "\"\r\n\r\n"
            << value << "\r\n";
    };

    std::ostringstream out;
    out << "--" << kVoiceForwardBoundary << "\r\n"
        << "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
        << "Content-Type: " << (file->content_type.empty() ? "application/octet-stream" : file->content_type) << "\r\n\r\n"
        << file->content << "\r\n";

    const auto *model_part = find_multipart_part(parts, "model");
    const auto *language_part = find_multipart_part(parts, "language");
    const auto *prompt_part = find_multipart_part(parts, "prompt");
    const auto model = model_part ? model_part->content : ezl_getenv_default("WHISPER_MODEL");
    const auto language = language_part ? language_part->content : "";
    const auto prompt = prompt_part ? prompt_part->content : "";
    const auto response_format = multipart_value_or(parts, "response_format", "json");
    const auto temperature = multipart_python_float_value_or(parts, "temperature", "0.0");
    if (fallback_semantics || !model.empty()) {
        add_field(out, "model", model);
    }
    if ((fallback_semantics && language_part) || !language.empty()) {
        add_field(out, "language", language);
    }
    if ((fallback_semantics && prompt_part) || !prompt.empty()) {
        add_field(out, "prompt", prompt);
    }
    add_field(out, "response_format", response_format);
    add_field(out, "temperature", temperature);
    out << "--" << kVoiceForwardBoundary << "--\r\n";
    return out.str();
}

std::string build_router_transcription_payload(const std::string &body, const std::string &content_type) {
    const auto boundary = extract_multipart_boundary(content_type);
    const auto parts = parse_multipart_parts(body, boundary);
    const auto *file = find_multipart_part(parts, "file");
    if (!file) {
        return "";
    }

    const auto add_field = [](std::ostringstream &out, const std::string &name, const std::string &value) {
        out << "--" << kVoiceForwardBoundary << "\r\n"
            << "Content-Disposition: form-data; name=\"" << name << "\"\r\n\r\n"
            << value << "\r\n";
    };
    const auto add_present_field = [&parts, &add_field](std::ostringstream &out, const std::string &name) {
        const auto *part = find_multipart_part(parts, name);
        if (part) {
            add_field(out, name, part->content);
        }
    };

    std::ostringstream out;
    const std::string filename = file->filename.empty() ? "audio.wav" : file->filename;
    out << "--" << kVoiceForwardBoundary << "\r\n"
        << "Content-Disposition: form-data; name=\"file\"; filename=\"" << filename << "\"\r\n"
        << "Content-Type: " << (file->content_type.empty() ? "audio/wav" : file->content_type) << "\r\n\r\n"
        << file->content << "\r\n";

    add_present_field(out, "model");
    add_present_field(out, "language");
    add_present_field(out, "prompt");
    add_field(out, "response_format", multipart_value_or(parts, "response_format", "json"));
    add_field(out, "temperature", multipart_python_float_value_or(parts, "temperature", "0.0"));
    if (const auto *timestamps = find_multipart_part(parts, "timestamps")) {
        add_field(out, "timestamps", multipart_python_bool_value(*timestamps));
    }
    out << "--" << kVoiceForwardBoundary << "--\r\n";
    return out.str();
}

std::string build_router_image_edit_payload(const std::string &body, const std::string &content_type) {
    const auto boundary = extract_multipart_boundary(content_type);
    const auto parts = parse_multipart_parts(body, boundary);
    if (parts.empty()) {
        return "";
    }

    const auto add_field = [](std::ostringstream &out, const std::string &name, const std::string &value) {
        out << "--" << kVoiceForwardBoundary << "\r\n"
            << "Content-Disposition: form-data; name=\"" << name << "\"\r\n\r\n"
            << value << "\r\n";
    };
    const auto add_file = [](std::ostringstream &out, const MultipartPart &part) {
        const std::string filename = part.filename.empty() ? "file" : part.filename;
        const std::string content_type = part.content_type.empty() ? "application/octet-stream" : part.content_type;
        out << "--" << kVoiceForwardBoundary << "\r\n"
            << "Content-Disposition: form-data; name=\"" << part.name << "\"; filename=\"" << filename << "\"\r\n"
            << "Content-Type: " << content_type << "\r\n\r\n"
            << part.content << "\r\n";
    };

    std::ostringstream out;
    std::vector<MultipartPart> files;
    std::vector<MultipartPart> fields;
    const auto upsert = [](std::vector<MultipartPart> *selected, const MultipartPart &part) {
        for (auto &existing : *selected) {
            if (existing.name == part.name) {
                existing = part;
                return;
            }
        }
        selected->push_back(part);
    };
    for (const auto &part : parts) {
        if (part.has_filename) {
            upsert(&files, part);
        } else {
            upsert(&fields, part);
        }
    }
    for (const auto &part : files) {
        add_file(out, part);
    }
    for (const auto &part : fields) {
        add_field(out, part.name, part.content);
    }
    if (files.empty() && fields.empty()) {
        return "";
    }
    out << "--" << kVoiceForwardBoundary << "--\r\n";
    return out.str();
}

std::string image_edit_multipart_json_body(const std::string &body, const std::string &content_type) {
    const auto boundary = extract_multipart_boundary(content_type);
    if (boundary.empty()) {
        return "";
    }
    const auto parts = parse_multipart_parts(body, boundary);
    if (parts.empty()) {
        return "";
    }

    std::map<std::string, std::string> fields;
    std::string image_value;
    for (const auto &part : parts) {
        if (part.name == "image") {
            if (part.has_filename) {
                const std::string media_type = part.content_type.empty() ? "application/octet-stream" : part.content_type;
                const std::string encoded = base64_encode(
                    reinterpret_cast<const unsigned char *>(part.content.data()),
                    part.content.size()
                );
                image_value = "data:" + media_type + ";base64," + encoded;
            } else {
                image_value = part.content;
            }
            continue;
        }
        if (!part.has_filename) {
            fields[part.name] = part.content;
        } else if (image_value.empty()) {
            const std::string media_type = part.content_type.empty() ? "application/octet-stream" : part.content_type;
            const std::string encoded = base64_encode(
                reinterpret_cast<const unsigned char *>(part.content.data()),
                part.content.size()
            );
            image_value = "data:" + media_type + ";base64," + encoded;
        }
    }

    std::ostringstream out;
    out << "{";
    bool first = true;
    auto add_string_field = [&](const std::string &name, const std::string &value) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << name << "\":\"" << json_escape(value) << "\"";
    };
    if (!image_value.empty()) {
        add_string_field("image", image_value);
    }
    for (const auto &name : {"prompt", "model", "n", "size", "response_format"}) {
        auto it = fields.find(name);
        if (it != fields.end()) {
            add_string_field(name, it->second);
        }
    }
    out << "}";
    return out.str();
}

std::string transcription_text_response_body(const std::string &response_body) {
    const std::string text_value = extract_json_value(response_body, "text", "");
    if (text_value.empty()) {
        return response_body;
    }
    const std::string trimmed = trim(text_value);
    if (!trimmed.empty() && trimmed.front() == '"') {
        return parse_json_string_at(trimmed, 0);
    }
    if (trimmed == "null") {
        return "";
    }
    return trimmed;
}

std::string json_optional_string_or_default(const std::string &body, const std::string &key, const std::string &fallback);

std::string embedding_forward_payload(const std::string &body, bool fallback = false) {
    std::vector<std::pair<std::string, std::string>> values;
    values.emplace_back("input", json_object_field(body, "input", "\"\""));
    values.emplace_back("model", json_object_field(body, "model", "\"" + json_escape(ezl_getenv_default("EMBEDDING_MODEL")) + "\""));
    if (!fallback) values.emplace_back("user", json_object_field(body, "user", "null"));
    const auto dimensions = json_object_field(body, "dimensions", "null");
    const auto parsed = parse_pydantic_int_token(dimensions);
    values.emplace_back("dimensions", parsed.ok ? std::to_string(parsed.value) : "null");
    values.emplace_back("encoding_format", json_object_field(body, "encoding_format", "null"));
    if (!fallback) {
        values.erase(std::remove_if(values.begin(), values.end(), [](const auto &entry) {
            return trim(entry.second) == "null";
        }), values.end());
    }
    return json_object_from_ordered_entries(values);
}

std::string json_with_default_int_field(const std::string &body, const std::string &key, int value) {
    if (!extract_json_value(body, key, "").empty()) {
        return body;
    }
    const auto end = body.find_last_of('}');
    if (end == std::string::npos) {
        return body;
    }
    size_t before_end = end;
    while (before_end > 0 && std::isspace(static_cast<unsigned char>(body[before_end - 1]))) {
        before_end--;
    }
    const bool has_existing_fields = before_end > 0 && body[before_end - 1] != '{';
    std::ostringstream out;
    out << body.substr(0, before_end)
        << (has_existing_fields ? "," : "")
        << "\"" << key << "\":" << value
        << body.substr(before_end);
    return out.str();
}

std::string json_optional_string_or_default(const std::string &body, const std::string &key, const std::string &fallback) {
    const std::string raw = trim(extract_json_value(body, key, ""));
    if (raw.empty()) {
        return "\"" + json_escape(fallback) + "\"";
    }
    if (raw == "null") {
        return "null";
    }
    if (!raw.empty() && raw.front() == '"') {
        return "\"" + json_escape(parse_json_string_at(raw, 0)) + "\"";
    }
    return "\"" + json_escape(raw) + "\"";
}

std::string json_optional_int_or_default(const std::string &body, const std::string &key, int fallback) {
    const std::string raw = trim(extract_json_value(body, key, ""));
    if (raw.empty()) {
        return std::to_string(fallback);
    }
    if (raw == "null") {
        return "null";
    }
    const auto parsed = parse_pydantic_int_token(raw);
    return parsed.ok ? std::to_string(parsed.value) : std::to_string(fallback);
}

std::string json_number_literal(double value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    if (std::isinf(value)) {
        return value < 0 ? "-Infinity" : "Infinity";
    }
    return python_float_string(value);
}

std::string json_optional_float_or_default(const std::string &body, const std::string &key, double fallback) {
    std::string raw = trim(extract_json_value(body, key, ""));
    if (raw.empty()) {
        return json_number_literal(fallback);
    }
    if (raw == "null") {
        return "null";
    }
    if (raw == "true") {
        return json_number_literal(1.0);
    }
    if (raw == "false") {
        return json_number_literal(0.0);
    }
    if (!raw.empty() && raw.front() == '"') {
        raw = trim(parse_json_string_at(raw, 0));
        double parsed = fallback;
        return parse_strict_python_float_string(raw, &parsed) ? json_number_literal(parsed) : json_number_literal(fallback);
    }
    try {
        size_t consumed = 0;
        const double parsed = std::stod(raw, &consumed);
        if (consumed == raw.size()) {
            return json_number_literal(parsed);
        }
    } catch (...) {
    }
    return json_number_literal(fallback);
}

void validate_image_edit_fields(const std::string &body, std::vector<std::string> *errors) {
    const auto values = top_level_json_object_values(body);
    auto error = [&](const std::string &type, const std::vector<std::string> &loc, const std::string &message,
                     const std::string &input, const std::string &context = "") {
        append_pydantic_error(errors, type, loc, message, input);
        if (!context.empty()) { errors->back().pop_back(); errors->back() += ",\"ctx\":" + context + "}"; }
    };
    validate_pydantic_optional_string_field(values, "image", errors);
    validate_pydantic_optional_list_of_strings_field(values, "images", errors);
    const auto refs_raw = json_object_field(body, "images", "null");
    const auto refs = json_array_elements(refs_raw);
    if (refs.size() > 10) error("too_long", {"images"}, "List should have at most 10 items after validation, not " + std::to_string(refs.size()), refs_raw,
        "{\"field_type\":\"List\",\"max_length\":10,\"actual_length\":" + std::to_string(refs.size()) + "}");
    validate_pydantic_required_string_field(values, body, "prompt", nullptr, errors);
    validate_pydantic_optional_string_field(values, "model", errors);
    const auto n = json_object_field(body, "n", "1");
    if (n == "null") error("int_type", {"n"}, "Input should be a valid integer", n);
    else {
        validate_pydantic_optional_int_field(values, "n", errors);
        const auto parsed = parse_pydantic_int_token(n);
        if (parsed.ok && parsed.value < 1) error("greater_than_equal", {"n"}, "Input should be greater than or equal to 1", n, "{\"ge\":1}");
        if (parsed.ok && parsed.value > 10) error("less_than_equal", {"n"}, "Input should be less than or equal to 10", n, "{\"le\":10}");
    }
    const auto size = json_object_field(body, "size", "\"1024x1024\"");
    if (size == "null") error("string_type", {"size"}, "Input should be a valid string", size);
    else validate_pydantic_optional_string_field(values, "size", errors);
    const auto format = json_object_field(body, "response_format", "\"url\"");
    if (format != "\"url\"" && format != "\"b64_json\"") error("literal_error", {"response_format"}, "Input should be 'url' or 'b64_json'", format, "{\"expected\":\"'url' or 'b64_json'\"}");
    const auto strength = json_object_field(body, "strength", "0.75");
    if (strength == "null") error("float_type", {"strength"}, "Input should be a valid number", strength);
    else {
        const auto before = errors->size();
        validate_pydantic_optional_float_field(values, "strength", errors);
        if (before == errors->size()) {
            const double number = coerce_python_float_value(strength, 0.75, true);
            if (!(number >= 0)) error("greater_than_equal", {"strength"}, "Input should be greater than or equal to 0", strength, "{\"ge\":0.0}");
            else if (!(number <= 1)) error("less_than_equal", {"strength"}, "Input should be less than or equal to 1", strength, "{\"le\":1.0}");
        }
    }
    if (!errors->empty()) return;
    std::string message;
    const auto primary = json_object_field(body, "image", "null");
    if (is_python_falsey_json_value(primary) && refs.empty()) message = "image or images is required";
    else if (primary == "\"\"" || std::any_of(refs.begin(), refs.end(), [](const auto &s) { return is_python_falsey_json_value(s); })) message = "Reference images must not be empty";
    else try { qwen_image_dimensions(parse_json_string_at(size, 0)); } catch (...) { message = "size must contain two positive multiples of 32"; }
    if (!message.empty()) error("value_error", {}, "Value error, " + message, body, "{\"error\":{}}");
}

std::string image_forward_payload(const std::string &body, bool edit) {
    std::ostringstream out;
    out << "{"
        << "\"prompt\":\"" << json_escape(extract_json_string(body, "prompt", "")) << "\","
        << "\"response_format\":" << json_optional_string_or_default(body, "response_format", "url") << ","
        << "\"size\":" << json_optional_string_or_default(body, "size", "1024x1024") << ","
        << "\"n\":" << json_optional_int_or_default(body, "n", 1);
    const auto image = extract_json_string(body, "image", "");
    if (edit && !image.empty()) {
        out << ",\"image\":\"" << json_escape(image) << "\"";
    }
    if (edit) {
        const auto images = json_object_field(body, "images", "null");
        if (!is_python_falsey_json_value(images)) out << ",\"images\":" << images;
        out << ",\"strength\":" << json_optional_float_or_default(body, "strength", 0.75);
    }
    out << "}";
    return out.str();
}

std::string fallback_image_payload(const std::string &body) {
    std::ostringstream out;
    out << "{"
        << "\"prompt\":\"" << json_escape(extract_json_string(body, "prompt", "")) << "\","
        << "\"response_format\":" << json_optional_string_or_default(body, "response_format", "url") << ","
        << "\"size\":" << json_optional_string_or_default(body, "size", "1024x1024")
        << ",\"n\":1}";
    return out.str();
}

struct MusicField {
    const char *name;
    char type;
    const char *default_json;
};

bool music_integer_value(std::string raw, std::string *normalized) {
    raw = trim(raw);
    if (raw == "true" || raw == "false") { *normalized = raw == "true" ? "1" : "0"; return true; }
    if (raw.empty()) return false;
    const bool quoted = raw.front() == '"';
    if (quoted) raw = trim(parse_json_string_at(raw, 0));
    else if (raw.find_first_of(".eE") != std::string::npos && is_json_number_token(raw)) {
        try {
            const double value = std::stod(raw);
            if (!std::isfinite(value) || std::floor(value) != value) return false;
            std::ostringstream out; out << std::fixed << std::setprecision(0) << value;
            raw = out.str();
        } catch (...) { return false; }
    }
    if (raw.empty()) return false;
    const bool negative = raw.front() == '-';
    size_t position = raw.front() == '+' || negative ? 1 : 0;
    std::string digits;
    bool previous_digit = false;
    for (; position < raw.size(); ++position) {
        const char ch = raw[position];
        if (ch >= '0' && ch <= '9') { digits += ch; previous_digit = true; }
        else if (quoted && ch == '_' && previous_digit) previous_digit = false;
        else if (quoted && ch == '.' && previous_digit && position + 1 < raw.size()) {
            for (++position; position < raw.size(); ++position) if (raw[position] != '0') return false;
            break;
        } else return false;
    }
    if (!previous_digit || digits.empty() || (quoted && digits.size() > 4300)) return false;
    const auto first = digits.find_first_not_of('0');
    *normalized = first == std::string::npos ? "0" : (negative ? "-" : "") + digits.substr(first);
    return true;
}

std::vector<MusicField> music_schema(bool video) {
    std::vector<MusicField> fields = {{"prompt", 's', nullptr}, {"lyrics", 's', nullptr}, {"duration", 'f', nullptr}, {"keyscale", 's', nullptr}};
    const std::vector<MusicField> audio = {
        {"model", 's', "\"Serveurperso/ACE-Step-1.5-GGUF\""}, {"instrumental", 'b', "false"}, {"n", 'i', "1"},
        {"response_format", 's', "\"url\""}, {"output_format", 's', "\"wav16\""}, {"seed", 'i', "null"}
    };
    const std::vector<MusicField> visual = {
        {"model", 's', "null"}, {"music_model", 's', "\"Serveurperso/ACE-Step-1.5-GGUF\""},
        {"video_model", 's', "\"QuantStack/Wan2.2-T2V-A14B-GGUF\""}, {"video_prompt", 's', "null"},
        {"scene_prompts", 'a', "null"}, {"scene_duration", 'f', "null"}, {"size", 's', "\"768x512\""},
        {"num_inference_steps", 'i', "18"}, {"guidance_scale", 'f', "3.5"}, {"video_guidance_scale", 'f', "null"},
        {"frame_rate", 'i', "24"}, {"response_format", 's', "\"url\""}, {"instrumental", 'b', "false"},
        {"seed", 'i', "null"}, {"music_seed", 'i', "null"}, {"music_output_format", 's', "\"wav16\""},
        {"audio_url", 's', "null"}, {"include_scene_audio", 'b', "false"}, {"storyboard", 'b', "true"}, {"scene_images", 'a', "null"}
    };
    const auto &specific = video ? visual : audio;
    fields.insert(fields.end(), specific.begin(), specific.end());
    for (const MusicField field : std::vector<MusicField>{{"bpm", 'i', "128"}, {"timesignature", 's', "\"4/4\""},
        {"vocal_language", 's', "\"en\""}, {"inference_steps", 'i', "16"}}) fields.push_back(field);
    if (video) fields.push_back({"music_inference_steps", 'i', "null"});
    fields.push_back({video ? "music_guidance_scale" : "guidance_scale", 'f', "1.0"});
    for (const MusicField field : std::vector<MusicField>{{"shift", 'f', "3.0"}, {"solver", 's', "\"euler\""},
        {"lm_model", 's', "null"}, {"synth_model", 's', "null"}, {"audio_codes", 's', "null"}}) fields.push_back(field);
    if (video) { fields.push_back({"image", 's', "null"}); fields.push_back({"conditions", 'c', "null"}); }
    else fields.push_back({"extra_parameters", 'd', "null"});
    return fields;
}

std::string music_field_value(const MusicField &field, const std::string &raw) {
    if (field.type == 's') return "\"" + json_escape(parse_json_string_at(raw, 0)) + "\"";
    if (field.type == 'i') { std::string value; return music_integer_value(raw, &value) ? value : std::to_string(parse_pydantic_int_token(raw).value); }
    if (field.type == 'f') return json_optional_float_or_default("{\"value\":" + raw + "}", "value", 0.0);
    if (field.type == 'b') return extract_json_bool("{\"value\":" + raw + "}", "value", false) ? "true" : "false";
    if (field.type == 'c') {
        std::vector<std::string> conditions;
        for (const auto &item : json_array_elements(raw)) {
            std::vector<std::pair<std::string, std::string>> normalized;
            for (const MusicField nested : std::vector<MusicField>{{"image", 's', "\"\""}, {"index", 'i', "0"}, {"strength", 'f', "1.0"}}) {
                const auto value = trim(json_object_field(item, nested.name, nested.default_json));
                if (value != "null") normalized.emplace_back(nested.name, music_field_value(nested, value));
            }
            conditions.push_back(json_object_from_ordered_entries(normalized));
        }
        return "[" + join_strings(conditions, ",") + "]";
    }
    return raw;
}

bool normalized_music_payload(const std::string &body, bool video, std::string *payload, HttpResponse *failure) {
    if (!is_valid_top_level_json_object(body)) { *failure = invalid_json_model_response(trim(body) == "null" ? "" : body); return false; }
    const auto values = top_level_json_object_values(body);
    std::vector<std::string> errors;
    const auto fields = music_schema(video);
    for (const auto &field : fields) {
        const auto it = values.find(field.name);
        if (!field.default_json && it == values.end()) { append_pydantic_error(&errors, "missing", field.name, "Field required", trim(body)); continue; }
        if (!field.default_json && it != values.end() && trim(it->second) == "null") {
            append_pydantic_error(&errors, field.type == 'f' ? "float_type" : "string_type", field.name,
                field.type == 'f' ? "Input should be a valid number" : "Input should be a valid string", "null"); continue;
        }
        switch (field.type) {
            case 's': validate_pydantic_optional_string_field(values, field.name, &errors); break;
            case 'i': {
                std::string normalized;
                if (it == values.end() || !music_integer_value(it->second, &normalized)) validate_pydantic_optional_int_field(values, field.name, &errors);
                break;
            }
            case 'f': validate_pydantic_optional_float_field(values, field.name, &errors); break;
            case 'b': validate_pydantic_optional_bool_field(values, field.name, &errors); break;
            case 'a': validate_pydantic_optional_list_of_strings_field(values, field.name, &errors); break;
            case 'd': validate_pydantic_optional_dict_field(values, field.name, &errors); break;
            case 'c': validate_pydantic_video_conditions_field(values, &errors); break;
        }
    }
    if (!errors.empty()) { *failure = {422, "application/json", pydantic_error_response_body(errors)}; return false; }
    std::vector<std::pair<std::string, std::string>> normalized;
    for (const auto &field : fields) {
        const auto raw = trim(json_object_field(body, field.name, field.default_json ? field.default_json : "null"));
        if (raw != "null") normalized.emplace_back(field.name, music_field_value(field, raw));
    }
    *payload = json_object_from_ordered_entries(normalized);
    return true;
}

std::string video_forward_payload(const std::string &body, bool with_image = true, bool with_conditions = false) {
    std::vector<std::pair<std::string, std::string>> fields;
    for (const MusicField field : std::vector<MusicField>{{"prompt", 's', "\"\""}, {"response_format", 's', "\"url\""},
        {"size", 's', "\"768x512\""}, {"num_frames", 'i', "121"}, {"num_inference_steps", 'i', "18"},
        {"guidance_scale", 'f', "3.5"}, {"frame_rate", 'i', "24"}}) {
        const auto raw = trim(json_object_field(body, field.name, field.default_json));
        fields.emplace_back(field.name, raw == "null" ? raw : music_field_value(field, raw));
    }
    const auto image = json_object_field(body, "image", "null");
    if (with_image && !is_python_falsey_json_value(image)) fields.emplace_back("image", image);
    const auto raw_conditions = json_object_field(body, "conditions", "null");
    if (with_conditions && !is_python_falsey_json_value(raw_conditions)) {
        std::vector<std::string> conditions;
        for (const auto &condition : json_array_elements(raw_conditions)) {
            std::vector<std::pair<std::string, std::string>> values;
            for (const MusicField field : std::vector<MusicField>{{"image", 's', "\"\""}, {"index", 'i', "0"}, {"strength", 'f', "1.0"}}) {
                const auto raw = trim(json_object_field(condition, field.name, field.default_json));
                values.emplace_back(field.name, raw == "null" ? raw : music_field_value(field, raw));
            }
            conditions.push_back(json_object_from_ordered_entries(values));
        }
        fields.emplace_back("conditions", json_array_from_values(conditions));
    }
    return json_object_from_ordered_entries(fields);
}

bool json_response_value(const HttpResponse &response, std::string *value) {
    const auto mime = to_lower(trim(response.content_type.substr(0, response.content_type.find(';'))));
    if (mime != "application/json" && !(starts_with(mime, "application/") && mime.size() >= 5 && mime.compare(mime.size() - 5, 5, "+json") == 0)) return false;
    *value = trim(response.body);
    if (value->empty()) *value = "null"; // aiohttp treats an empty JSON body as None.
    return is_json_value_token(*value);
}

bool fallback_decode(const std::string &body, std::string charset, std::string *decoded) {
    // Python codecs.lookup accepts these aliases; glibc iconv does not.
    if(charset=="latin-1"||charset=="latin_1")charset="ISO-8859-1";
    bool strip_bom = charset == "utf-8-sig" || charset == "utf_8_sig";
    if (strip_bom) charset = "utf-8";
    auto converter = iconv_open("UTF-8", charset.empty() ? "UTF-8" : charset.c_str());
    // aiohttp ignores unknown charset names and falls back to UTF-8 for JSON.
    if (converter == reinterpret_cast<iconv_t>(-1)) converter = iconv_open("UTF-8", "UTF-8");
    if (converter == reinterpret_cast<iconv_t>(-1)) return false;
    struct Close { iconv_t value; ~Close() { iconv_close(value); } } close{converter};
    std::string converted(body.size() * 4 + 16, '\0');
    char *input = const_cast<char *>(body.data()), *output = converted.data();
    size_t remaining = body.size(), available = converted.size();
    if (iconv(converter, &input, &remaining, &output, &available) == static_cast<size_t>(-1)) return false;
    converted.resize(converted.size() - available);
    if (strip_bom && starts_with(converted, "\xef\xbb\xbf")) converted.erase(0, 3);
    *decoded = std::move(converted); return true;
}

bool fallback_json_response(const HttpResponse &response, std::string *value) {
    // aiohttp enforces the response's JSON media type before decoding. The
    // syntax parser also accepts json.loads' NaN/Infinity constants.
    const auto type = to_lower(response.content_type);
    static const std::regex json_type(R"(^application/(?:[\w.+-]+?\+)?json)"), charset_pattern(R"re((?:^|;)\s*charset\s*=\s*(?:"([^"]*)"|([^;]*)))re");
    if (!std::regex_search(type, json_type)) return false;
    std::smatch charset_match; std::string charset = "utf-8";
    if (std::regex_search(type, charset_match, charset_pattern)) charset = trim(charset_match[1].matched ? charset_match[1].str() : charset_match[2].str());
    const auto body = trim(response.body);
    if (body.empty()) { *value = "null"; return true; }
    if (!fallback_decode(body, charset, value)) return false;
    if (json_decode_error(*value)) return false;
    // json.loads applies Python's default decimal integer conversion limit to
    // the entire document, including fields the caller subsequently ignores.
    for (size_t i = 0; i < value->size();) {
        const auto ch = (*value)[i];
        if (ch == '"') { i = json_value_end(*value, i); continue; }
        if (ch == '-' || (ch >= '0' && ch <= '9')) {
            const auto start = i++;
            while (i < value->size() && std::string("0123456789.eE+-").find((*value)[i]) != std::string::npos) ++i;
            const auto number = value->substr(start, i - start);
            if (number.find_first_of(".eE") == std::string::npos && number.size() - (number.front() == '-') > 4300) return false;
        } else ++i;
    }
    *value = trim(*value); return true;
}

std::string fallback_model_json(const std::string &raw, size_t depth = 0) {
    if (depth > 1000) throw std::runtime_error("Model JSON nesting exceeds parser limit");
    const auto value = trim(raw);
    if (value.empty()) throw std::invalid_argument("Missing model JSON value");
    if (value.front() == '"') return '"' + json_escape(parse_json_string_at(value, 0)) + '"';
    if (value.front() == '[') {
        std::vector<std::string> items;
        for (const auto &item : json_array_elements(value)) items.push_back(fallback_model_json(item, depth + 1));
        return json_array_from_values(items);
    }
    if (value.front() == '{') {
        std::vector<std::pair<std::string, std::string>> items;
        for (const auto &item : json_object_ordered_entries(value)) ordered_json_upsert(&items, item);
        for (auto &item : items) item.second = fallback_model_json(item.second, depth + 1);
        return json_object_from_ordered_entries(items);
    }
    if (value == "null" || value == "true" || value == "false") return value;
    if (is_json_number_token(value) && value.find_first_of(".eE") == std::string::npos) {
        if (value.size() - (value.front() == '-') > 4300) throw std::invalid_argument("Model integer exceeds Python conversion limit");
        return value == "-0" ? "0" : value;
    }
    double number = 0;
    if (!parse_strict_python_float_string(value, &number) || !std::isfinite(number)) throw std::invalid_argument("Non-finite model JSON value");
    return json_number_literal(number);
}

bool service_json_response(const HttpResponse &response, HttpResponse *out) {
    std::string value;
    if (response.status != 200 || !fallback_json_response(response, &value) || value == "null") return false;
    try {
        value = fallback_model_json(value);
        std::string decoded;
        if (!fallback_decode(value, "utf-8", &decoded)) throw std::invalid_argument("Invalid service JSON UTF-8");
        *out = {200, "application/json", value};
    } catch (const std::exception &) { *out = {500, "text/plain; charset=utf-8", "Internal Server Error"}; }
    return true;
}

std::string echo_subtitle_translation_response(const std::string &request_body) {
    bool subtitle_translation = false;
    std::string user_content;
    for (const auto &message : extract_json_array_elements(request_body, "messages")) {
        const std::string role = extract_json_string(message, "role", "");
        const std::string content = extract_json_string(message, "content", "");
        if (role == "system" && content.find("professional subtitle translator") != std::string::npos) {
            subtitle_translation = true;
        } else if (role == "user") {
            user_content = content;
        }
    }
    if (!subtitle_translation || user_content.empty()) {
        return "";
    }

    std::ostringstream translated;
    std::istringstream in(user_content);
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto close = line.find(']');
        if (line.front() != '[' || close == std::string::npos) {
            continue;
        }
        if (!first) {
            translated << "\n";
        }
        first = false;
        translated << line.substr(0, close + 1) << " translated " << trim(line.substr(close + 1));
    }
    return translated.str();
}

std::optional<LlmResponseError> llm_response_error(const std::string &json, bool chat_stream) {
    const auto falsey = [](const std::string &raw) {
        const auto value = trim(raw);
        if (!value.empty() && value.front() == '{') return json_object_ordered_entries(value).empty();
        if (!value.empty() && value.front() == '[') return json_array_elements(value).empty();
        if (is_json_number_token(value) && value.find_first_of(".eE") != std::string::npos) {
            double number = 0;
            if (parse_strict_python_float_string(value, &number)) return number == 0;
        }
        return is_python_falsey_json_value(value);
    };
    std::function<std::string(const std::string &)> repr = [&](const std::string &raw) -> std::string {
        const auto value = trim(raw);
        if (!value.empty() && value.front() == '[') {
            std::string out = "[";
            for (const auto &item : json_array_elements(value)) { if (out.size() > 1) out += ", "; out += repr(item); }
            return out + ']';
        }
        if (!value.empty() && value.front() == '{') {
            std::vector<std::pair<std::string, std::string>> fields;
            for (const auto &item : json_object_ordered_entries(value)) ordered_json_upsert(&fields, item);
            std::string out = "{";
            for (const auto &item : fields) { if (out.size() > 1) out += ", "; out += python_repr_string(item.first) + ": " + repr(item.second); }
            return out + '}';
        }
        if (value == "NaN" || value == "Infinity" || value == "-Infinity"
            || (is_json_number_token(value) && value.find_first_of(".eE") != std::string::npos)) {
            double number = 0;
            if (parse_strict_python_float_string(value, &number)) return python_float_string(number);
        }
        return python_repr_json_value(value);
    };
    const auto str = [&](const std::string &raw) {
        const auto value = trim(raw);
        return !value.empty() && value.front() == '"' ? parse_json_string_at(value, 0) : repr(value);
    };
    const auto value = trim(json);
    if (value.empty() || value.front() != '{') return std::nullopt;
    const auto nested = json_object_field(value, "error", "");
    const auto code = json_object_field(value, "code", "");
    std::string info;
    if (!nested.empty()) {
        if (chat_stream && nested.front() != '{') info = "{\"message\":" + nested + '}';
        else info = nested;
    } else {
        double number = 0;
        const bool code_error = chat_stream ? !code.empty()
            : is_json_number_token(code) && parse_strict_python_float_string(code, &number) && number == 400;
        if (code_error || str(json_object_field(value, "type", "null")) == "exceed_context_size_error") info = value;
    }
    if (info.empty() || (!chat_stream && falsey(info))) return std::nullopt;
    if (info.front() != '{') {
        const auto type = info.front() == '[' ? "list" : info.front() == '"' ? "str"
            : info == "true" || info == "false" ? "bool" : info == "null" ? "NoneType"
            : info == "NaN" || info == "Infinity" || info == "-Infinity" || info.find_first_of(".eE") != std::string::npos ? "float" : "int";
        return LlmResponseError{"AttributeError", "'" + std::string(type) + "' object has no attribute 'get'"};
    }
    const auto raw_message = json_object_field(info, "message", "");
    std::string message = raw_message.empty() ? (chat_stream ? str(info) : "Unknown error")
        : str(raw_message);
    const auto prompt = json_object_field(info, "n_prompt_tokens", "null");
    const auto context = json_object_field(info, "n_ctx", "null");
    if (!falsey(prompt)) message += " [n_prompt_tokens=" + str(prompt)
        + ", n_ctx=" + (falsey(context) ? "unknown" : str(context)) + ']';
    return LlmResponseError{"Exception", std::move(message)};
}

namespace {
std::string llm_output_type(const std::string &value) {
    if (value == "NaN" || value == "Infinity" || value == "-Infinity") return "float";
    if (value == "null") return "NoneType";
    if (value == "true" || value == "false") return "bool";
    if (!value.empty() && value.front() == '"') return "str";
    if (!value.empty() && value.front() == '[') return "list";
    if (!value.empty() && value.front() == '{') return "dict";
    return value.find_first_of(".eE") == std::string::npos ? "int" : "float";
}
bool llm_output_falsey(const std::string &value) {
    if (!value.empty() && value.front() == '{') return json_object_ordered_entries(value).empty();
    if (!value.empty() && value.front() == '[') return json_array_elements(value).empty();
    double number = 1;
    if (is_json_number_token(value) && parse_strict_python_float_string(value, &number)) return number == 0;
    return is_python_falsey_json_value(value);
}
std::string llm_output_get(const std::string &value, const std::string &key, const std::string &fallback) {
    if (value.empty() || value.front() != '{')
        throw LlmResponseError{"AttributeError", "'" + llm_output_type(value) + "' object has no attribute 'get'"};
    return json_object_field(value, key, fallback);
}
std::string llm_output_clean(const std::string &raw) {
    if (llm_output_falsey(raw)) return raw;
    static const std::vector<std::string> stops = {"<|im_end|", "<|im_end|>", "</|im_end|>", "</s>", "<s>",
        "User:", "### \n###", "[/INST]", "<|eot_id|>", "<|end_of_text|>", "assistant\n\n"};
    const auto type = llm_output_type(raw);
    if (type != "str") {
        if (type != "list" && type != "dict")
            throw LlmResponseError{"TypeError", "argument of type '" + type + "' is not iterable"};
        for (const auto &token : stops) {
            bool contains = false;
            if (type == "dict") contains = !json_object_field(raw, token, "").empty();
            else for (const auto &item : json_array_elements(raw))
                if (!item.empty() && item.front() == '"' && parse_json_string_at(item, 0) == token) contains = true;
            if (contains) throw LlmResponseError{"AttributeError", "'" + type + "' object has no attribute 'split'"};
        }
        throw LlmResponseError{"AttributeError", "'" + type + "' object has no attribute 'strip'"};
    }
    auto text = parse_json_string_at(raw, 0);
    for (const auto &token : stops) {
        const auto at = text.find(token);
        if (at != std::string::npos) text.resize(at);
    }
    text = strip_config_value(text);
    // The reference's subsequent whitespace slices are unreachable after
    // strip(). Its placeholder substitution deliberately does not strip again.
    if (text.find("[Insert ") != std::string::npos) {
        for (size_t at = 0; (at = text.find("[Insert", at)) != std::string::npos;) {
            const auto end = text.find(']', at + 7), newline = text.find('\n', at + 7);
            if (end != std::string::npos && (newline == std::string::npos || end < newline)) text.erase(at, end + 1 - at);
            else at += 7;
        }
    }
    return '"' + json_escape(text) + '"';
}
}

InferenceResult llm_buffered_response(const std::string &json, const std::string &model, bool chat_mode, bool completion_stream) {
    InferenceResult result(json); result.from_engine = true;
    result.failure = llm_response_error(json);
    if (result.failure) return result;
    const auto value = trim(json);
    if (value.empty() || value.front() != '{') return result;
    try {
        std::vector<std::pair<std::string, std::string>> fields;
        for (const auto &field : json_object_ordered_entries(value)) ordered_json_upsert(&fields, field);
        const auto choices = json_object_field(value, "choices", "null");
        if (!llm_output_falsey(choices) && (chat_mode || !completion_stream)) {
            std::string choice;
            std::vector<std::string> items;
            if (choices.front() == '[') { items = json_array_elements(choices); choice = items.front(); }
            else if (choices.front() == '{') throw LlmResponseError{"KeyError", "0"};
            else if (choices.front() == '"') throw LlmResponseError{"AttributeError", "'str' object has no attribute 'get'"};
            else throw LlmResponseError{"TypeError", "'" + llm_output_type(choices) + "' object is not subscriptable"};
            std::string text;
            if (chat_mode) text = llm_output_get(llm_output_get(choice, "message", "{}"), "content", "\"\"");
            else {
                text = llm_output_get(choice, "text", "\"\"");
                if (llm_output_falsey(text)) text = llm_output_get(llm_output_get(choice, "message", "{}"), "content", "\"\"");
            }
            text = llm_output_clean(text);
            std::vector<std::pair<std::string, std::string>> choice_fields;
            for (const auto &field : json_object_ordered_entries(choice)) ordered_json_upsert(&choice_fields, field);
            if (chat_mode) {
                const auto message = json_object_field(choice, "message", "");
                if (message.empty()) throw LlmResponseError{"KeyError", "'message'"};
                std::vector<std::pair<std::string, std::string>> message_fields;
                for (const auto &field : json_object_ordered_entries(message)) ordered_json_upsert(&message_fields, field);
                ordered_json_upsert(&message_fields, {"content", text});
                ordered_json_upsert(&choice_fields, {"message", json_object_from_ordered_entries(message_fields)});
            } else ordered_json_upsert(&choice_fields, {"text", text});
            items.front() = json_object_from_ordered_entries(choice_fields);
            ordered_json_upsert(&fields, {"choices", '[' + join_strings(items, ",") + ']'});
        }
        ordered_json_upsert(&fields, {"model", '"' + json_escape(model) + '"'});
        result.body = json_object_from_ordered_entries(fields);
    } catch (const LlmResponseError &failure) { result.failure = failure; }
    return result;
}

bool llm_qwen38_model(const std::string &model) {
    auto name = to_lower(strip_config_value(model));
    const auto slash = name.rfind('/'); if (slash != std::string::npos) name.erase(0, slash + 1);
    for (;;) {
        bool changed = false;
        for (const std::string suffix : {"-gguf", ".gguf", "-tee", "-mtp"}) {
            if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                name.resize(name.size() - suffix.size()); changed = true; break;
            }
        }
        if (!changed) return name == "qwen3.8-27b";
    }
}

bool llm_minicpm5_model(const std::string &model) {
    auto name = strip_config_value(model);
    const auto slash = name.rfind('/'); if (slash != std::string::npos) name.erase(0, slash + 1);
    const auto hash = name.find('#'); if (hash != std::string::npos) name.resize(hash);
    std::string normalized;
    for (int32_t at = 0; at < static_cast<int32_t>(name.size());) {
        UChar32 code; U8_NEXT(name.data(), at, static_cast<int32_t>(name.size()), code);
        if (code == 0x130 || code == 0x131) code = 'i';
        else if (code == 0x17f) code = 's';
        else code = u_tolower(code);
        if (code < 128 && code >= 0) normalized += static_cast<char>(code);
        else {
            char encoded[4]; int32_t length = 0; UBool error = false; U8_APPEND(encoded, length, 4, code, error);
            if (error) return false; normalized.append(encoded, length);
        }
    }
    const std::string prefix = "minicpm5-2b";
    if (normalized.compare(0, prefix.size(), prefix)) return false;
    auto tail = normalized.substr(prefix.size());
    if (starts_with(tail, "-gguf")) tail.erase(0, 5);
    if (tail.size() >= 5 && tail.compare(tail.size() - 5, 5, ".gguf") == 0) tail.resize(tail.size() - 5);
    if (tail.empty()) return true;
    if (tail.front() != '-' && tail.front() != ':') return false;
    tail.erase(0, 1);
    if (tail == "f16" || tail == "bf16") return true;
    if (tail.empty() || tail.front() != 'q') return false;
    int32_t at = 1; UChar32 code;
    if (at >= static_cast<int32_t>(tail.size())) return false;
    U8_NEXT(tail.data(), at, static_cast<int32_t>(tail.size()), code);
    if (u_charType(code) != U_DECIMAL_DIGIT_NUMBER) return false;
    while (at < static_cast<int32_t>(tail.size())) {
        U8_NEXT(tail.data(), at, static_cast<int32_t>(tail.size()), code);
        const auto category = u_charType(code);
        if (code != '_' && !(category >= U_UPPERCASE_LETTER && category <= U_OTHER_LETTER)
            && category != U_DECIMAL_DIGIT_NUMBER && category != U_LETTER_NUMBER && category != U_OTHER_NUMBER) return false;
    }
    return true;
}

std::string llm_model_request_settings(const std::string &body, const std::string &model) {
    using Fields = std::vector<std::pair<std::string, std::string>>;
    Fields data; for (const auto &item : json_object_ordered_entries(body)) ordered_json_upsert(&data, item);
    const auto erase = [](Fields &fields, const std::string &key) {
        fields.erase(std::remove_if(fields.begin(), fields.end(), [&](const auto &item) { return item.first == key; }), fields.end());
    };
    const bool qwen = llm_qwen38_model(model);
    static const std::map<std::string, std::string> profiles = {
        {"unsloth/Qwen3-VL-4B-Instruct-GGUF", R"({"top_p":0.8,"top_k":20,"temperature":0.7,"repetition_penalty":1.0,"presence_penalty":1.5})"},
        {"unsloth/Qwen3-4B-Instruct-2507-GGUF", R"({"top_p":1.0,"top_k":40,"temperature":1.0,"repetition_penalty":1.0,"presence_penalty":2.0})"},
        {"unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", R"({"top_p":0.8,"top_k":20,"temperature":0.7,"repetition_penalty":1.05})"},
    };
    static const std::set<std::string> small_instruct = {
        "unsloth/Qwen3.5-35B-A3B-GGUF", "unsloth/Qwen3.5-27B-GGUF", "unsloth/Qwen3.6-27B-GGUF",
        "unsloth/Qwen3.6-27B-MTP-GGUF", "unsloth/Qwen3.5-2B-MTP-GGUF", "unsloth/Qwen3.5-0.8B-MTP-GGUF"};
    static const std::set<std::string> regular_instruct = {
        "unsloth/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-4B-GGUF", "unsloth/Qwen3.5-2B-GGUF", "unsloth/Qwen3.5-0.8B-GGUF",
        "unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "unsloth/Qwen3.5-9B-MTP-GGUF", "unsloth/Qwen3.5-4B-MTP-GGUF"};
    std::string profile;
    if (qwen) profile = R"({"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"xhigh"}})";
    else if (llm_minicpm5_model(model)) profile = R"({"temperature":1.0,"top_p":0.95,"min_p":0.0,"chat_template_kwargs":{"enable_thinking":false}})";
    else if (const auto it = profiles.find(model); it != profiles.end()) profile = it->second;
    else if (small_instruct.count(model)) profile = R"({"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0,"chat_template_kwargs":{"enable_thinking":false}})";
    else if (regular_instruct.count(model)) profile = R"({"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0,"chat_template_kwargs":{"enable_thinking":false}})";
    else return json_object_from_ordered_entries(data);
    // Reuse the same last-key-wins parser without rescanning a long prompt
    // for every sampling/reasoning field.
    const auto request_fields = top_level_json_object_values(body);
    auto defaults = json_object_ordered_entries(profile);
    auto options = json_object_ordered_entries(json_object_field(profile, "chat_template_kwargs", "{}"));
    const auto user_options = json_object_field(request_fields, "chat_template_kwargs", "null");
    if (!user_options.empty() && user_options.front() == '{')
        for (const auto &item : json_object_ordered_entries(user_options)) ordered_json_upsert(&options, item);
    const auto effort = json_object_field(request_fields, "reasoning_effort", "null");
    erase(data, "reasoning_effort");
    if (effort != "null" && (qwen || !json_object_field(json_object_from_ordered_entries(options), "reasoning_effort", "").empty()))
        ordered_json_upsert(&options, {"reasoning_effort", effort});
    if (qwen) {
        const auto reasoning = json_object_field(request_fields, "reasoning", "null");
        if (!reasoning.empty() && reasoning.front() == '{') {
            const auto effort_value = json_object_field(reasoning, "effort", "null");
            auto enabled = json_object_field(reasoning, "enabled", "null");
            const bool no_effort = !llm_output_falsey(effort_value) && to_lower(python_str_from_json_value(effort_value)) == "none";
            if (no_effort) enabled = "false";
            else if (effort_value != "null" && enabled == "null") enabled = "true";
            if (enabled != "null") {
                bool on;
                if (!enabled.empty() && enabled.front() == '"') {
                    const auto word = to_lower(strip_config_value(parse_json_string_at(enabled, 0)));
                    on = word == "1" || word == "true" || word == "yes" || word == "on";
                } else on = !llm_output_falsey(enabled);
                ordered_json_upsert(&options, {"enable_thinking", on ? "true" : "false"});
            }
            if (effort_value != "null" && !no_effort) ordered_json_upsert(&options, {"reasoning_effort", effort_value});
        }
        if (json_object_field(json_object_from_ordered_entries(options), "enable_thinking", "null") == "false") {
            for (const auto &item : json_object_ordered_entries(R"({"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0})")) ordered_json_upsert(&defaults, item);
            erase(options, "reasoning_effort");
        }
    }
    for (const auto &item : defaults) ordered_json_upsert(&data, {item.first, item.first == "chat_template_kwargs" ? json_object_from_ordered_entries(options) : item.second});
    return json_object_from_ordered_entries(data);
}

std::string llm_engine_request(const std::string &body, bool chat_mode, const std::string &output_tokens, const std::string &system_message) {
    const auto request_fields = top_level_json_object_values(body);
    const std::string input_key = chat_mode ? "messages" : "prompt";
    auto input = json_object_field(request_fields, input_key, "");
    if (input.empty()) throw LlmParameterError("TypeError", "LLM." + std::string(chat_mode ? "chat" : "completion") + "() missing 1 required positional argument: '" + input_key + "'");
    std::vector<std::pair<std::string, std::string>> fields = {
        {input_key, input}, {"max_tokens", json_object_field(request_fields, "max_tokens", output_tokens)},
        {"temperature", json_object_field(request_fields, "temperature", "1.31")}, {"top_p", json_object_field(request_fields, "top_p", "0.95")},
        {"stream", json_object_field(request_fields, "stream", "false")}};
    if (chat_mode) {
        fields.push_back({"cache_prompt", "true"});
        // WorkConductor/WorkerRouter use this conversation hint to preserve
        // separate prompt states. It must reach the owned engine as metadata.
        const auto cache_key = json_object_field(request_fields, "prompt_cache_key", "");
        if (cache_key.size() > 2 && cache_key.front() == '"')
            fields.push_back({"prompt_cache_key", cache_key});
    }
    for (const auto *key : {"top_k", "min_p", "presence_penalty", "frequency_penalty"}) {
        const auto value = json_object_field(request_fields, key, ""); if (!value.empty()) fields.push_back({key, value});
    }
    const auto penalty = json_object_field(request_fields, "repeat_penalty", json_object_field(request_fields, "repetition_penalty", ""));
    if (!penalty.empty()) fields.push_back({"repeat_penalty", penalty});
    if (chat_mode) {
        const auto options = json_object_field(request_fields, "chat_template_kwargs", "");
        if (!options.empty()) fields.push_back({"chat_template_kwargs", options});
        bool has_system = false;
        if (input.front() != '[') {
            if (input.front() == '"' && parse_json_string_at(input, 0).empty()) { /* empty iterable */ }
            else if (input.front() == '{' && json_object_ordered_entries(input).empty()) { /* empty iterable */ }
            else if (input.front() == '"' || input.front() == '{') throw LlmParameterError("AttributeError", "'str' object has no attribute 'get'");
            else throw LlmParameterError("TypeError", "'" + llm_output_type(input) + "' object is not iterable");
        } else for (const auto &message : json_array_elements(input)) {
            if (message.empty() || message.front() != '{') throw LlmParameterError("AttributeError", "'" + llm_output_type(message) + "' object has no attribute 'get'");
            const auto role = json_object_field(message, "role", "null");
            if (!role.empty() && role.front() == '"' && parse_json_string_at(role, 0) == "system") { has_system = true; break; }
        }
        if (!has_system && !system_message.empty()) {
            if (input.front() != '[') throw LlmParameterError("TypeError", "can only concatenate list (not \"" + llm_output_type(input) + "\") to list");
            auto messages = json_array_elements(input);
            messages.insert(messages.begin(), "{\"role\":\"system\",\"content\":\"" + json_escape(system_message) + "\"}");
            fields.front().second = '[' + join_strings(messages, ",") + ']';
        }
    }
    return json_object_from_ordered_entries(fields);
}

void llm_completion_callback_guard(const std::string &payload) {
    if (!llm_output_falsey(json_object_field(payload, "stream", "null")))
        throw LlmParameterError("ValueError", "Server.handle_completions requires a callback for streaming or a non dict prompt.");
}

namespace {
const std::vector<std::string> stream_content_keys = {"content", "text", "token", "response", "output", "output_text", "generated_text"};
const std::vector<std::string> stream_reasoning_keys = {"reasoning_content", "reasoning", "thinking"};

bool stream_json_integer_limit(const std::string &value) {
    for (size_t at = 0; at < value.size();) {
        if (value[at] == '"') { at = json_value_end(value, at); continue; }
        if (value[at] == '-' || (value[at] >= '0' && value[at] <= '9')) {
            const auto start = at++;
            while (at < value.size() && std::string("0123456789.eE+-").find(value[at]) != std::string::npos) ++at;
            const auto number = value.substr(start, at - start);
            if (number.find_first_of(".eE") == std::string::npos && number.size() - (number.front() == '-') > 4300) return true;
        } else ++at;
    }
    return false;
}

std::string stream_text_value(const std::string &raw, size_t depth = 0) {
    if (depth > 900) return "";
    const auto value = trim(raw);
    if (value.empty()) return "";
    if (value.front() == '"') return parse_json_string_at(value, 0);
    if (value.front() == '[') {
        std::string result;
        for (const auto &item : json_array_elements(value)) result += stream_text_value(item, depth + 1);
        return result;
    }
    if (value.front() == '{') {
        for (const auto *keys : {&stream_content_keys, &stream_reasoning_keys})
            for (const auto &key : *keys) {
                const auto text = stream_text_value(json_object_field(value, key, "null"), depth + 1);
                if (!text.empty()) return text;
            }
    }
    return "";
}

std::string stream_ascii_string(const std::string &value) {
    std::string escaped = "\"";
    const auto hex16 = [&](uint32_t code) {
        constexpr char digits[] = "0123456789abcdef";
        escaped += "\\u";
        for (int shift = 12; shift >= 0; shift -= 4) escaped += digits[(code >> shift) & 15];
    };
    for (size_t pos = 0; pos < value.size();) {
        const auto first = static_cast<unsigned char>(value[pos++]);
        if (first < 0x7f) { escaped += json_escape(std::string(1, static_cast<char>(first))); continue; }
        uint32_t code = first;
        const unsigned tails = first < 0x80 ? 0 : first < 0xe0 ? 1 : first < 0xf0 ? 2 : 3;
        if (tails) {
            code &= 0x7f >> tails;
            for (unsigned i = 0; i < tails && pos < value.size(); ++i) code = (code << 6) | (static_cast<unsigned char>(value[pos++]) & 63);
        }
        if (code <= 0xffff) hex16(code);
        else { code -= 0x10000; hex16(0xd800 + (code >> 10)); hex16(0xdc00 + (code & 1023)); }
    }
    return escaped + '"';
}

std::string stream_json_dumps(const std::string &raw) {
    const auto value = trim(raw);
    if (value.empty()) return "null";
    if (value.front() == '"') return stream_ascii_string(parse_json_string_at(value, 0));
    if (value.front() == '[') {
        std::vector<std::string> items;
        for (const auto &item : json_array_elements(value)) items.push_back(stream_json_dumps(item));
        return '[' + join_strings(items, ", ") + ']';
    }
    if (value.front() == '{') {
        std::vector<std::pair<std::string, std::string>> fields;
        for (const auto &field : json_object_ordered_entries(value)) ordered_json_upsert(&fields, field);
        std::vector<std::string> items;
        for (const auto &field : fields) items.push_back(stream_ascii_string(field.first) + ": " + stream_json_dumps(field.second));
        return '{' + join_strings(items, ", ") + '}';
    }
    if (value == "null" || value == "true" || value == "false" || value == "NaN" || value == "Infinity" || value == "-Infinity") return value;
    if (value.find_first_of(".eE") == std::string::npos) return value == "-0" ? "0" : value;
    double number = 0;
    if (parse_strict_python_float_string(value, &number)) {
        if (!std::isfinite(number)) return std::isnan(number) ? "NaN" : number < 0 ? "-Infinity" : "Infinity";
        return python_float_string(number);
    }
    return value;
}

std::string stream_python_str(const std::string &value) {
    return llm_response_error("{\"error\":{\"message\":" + value + "}}", true)->message;
}
}

namespace {
std::string input_set(const std::string &object, const std::string &key, const std::string &value) {
    std::vector<std::pair<std::string, std::string>> fields;
    for (const auto &entry : json_object_ordered_entries(object)) ordered_json_upsert(&fields, entry);
    ordered_json_upsert(&fields, {key, value});
    return json_object_from_ordered_entries(fields);
}
bool input_type_is(const std::string &object, const std::string &type) {
    const auto value = json_object_field(object, "type", "null");
    return llm_output_type(value) == "str" && parse_json_string_at(value, 0) == type;
}
std::string input_json_value(const std::string &value, size_t depth = 0) {
    if (depth > 1000) throw LlmResponseError{"RecursionError", "maximum recursion depth exceeded"};
    const auto type = llm_output_type(value);
    if (type == "str") return '"' + json_escape(parse_json_string_at(value, 0)) + '"';
    if (type == "list") {
        std::vector<std::string> items;
        for (const auto &item : json_array_elements(value)) items.push_back(input_json_value(item, depth + 1));
        return json_array_from_values(items);
    }
    if (type == "dict") {
        std::vector<std::pair<std::string, std::string>> fields;
        for (const auto &entry : json_object_ordered_entries(value)) ordered_json_upsert(&fields, entry);
        for (auto &entry : fields) entry.second = input_json_value(entry.second, depth + 1);
        return json_object_from_ordered_entries(fields);
    }
    return stream_json_dumps(value);
}
std::string input_at(const std::string &object, const std::string &key) {
    const auto type = llm_output_type(object);
    if (type == "dict") {
        const auto value = json_object_field(object, key, "");
        if (value.empty()) throw LlmResponseError{"KeyError", python_repr_string(key)};
        return value;
    }
    if (type == "list") throw LlmResponseError{"TypeError", "list indices must be integers or slices, not str"};
    if (type == "str") throw LlmResponseError{"TypeError", "string indices must be integers, not 'str'"};
    throw LlmResponseError{"TypeError", "'" + type + "' object is not subscriptable"};
}
std::string input_last(const std::string &messages) {
    const auto type = llm_output_type(messages);
    if (type == "list") {
        const auto items = json_array_elements(messages);
        if (items.empty()) throw LlmResponseError{"IndexError", "list index out of range"};
        return items.back();
    }
    if (type == "dict") throw LlmResponseError{"KeyError", "-1"};
    if (type == "str") throw LlmResponseError{"IndexError", "string index out of range"};
    throw LlmResponseError{"TypeError", "'" + type + "' object is not subscriptable"};
}
std::string input_image(const std::string &mime, const std::string &bytes) {
    return "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:" + json_escape(mime)
        + ";base64," + base64_encode(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size()) + "\"}}";
}
std::string input_decode_image(const std::string &encoded) {
    // binascii's default decoder ignores non-alphabet bytes, but still rejects
    // incomplete padding and non-ASCII input before opening the image.
    for (unsigned char c : encoded) if (c >= 128) throw std::runtime_error("Non-ASCII base64 image");
    unsigned position = 0, left = 0, padding = 0;
    std::string out;
    for (unsigned char c : encoded) {
        if (c == '=') { if (position >= 2 && position + ++padding >= 4) return out; continue; }
        const auto value = base64_decode_value(c);
        if (value < 0) continue;
        padding = 0;
        switch (position) {
        case 0: left = value; position = 1; break;
        case 1: out += static_cast<char>((left << 2) | (value >> 4)); left = value & 15; position = 2; break;
        case 2: out += static_cast<char>((left << 4) | (value >> 2)); left = value & 3; position = 3; break;
        case 3: out += static_cast<char>((left << 6) | value); left = 0; position = 0; break;
        }
    }
    if (position) throw std::runtime_error("Incorrect base64 padding");
    return out;
}
}

bool llm_text_image_input_supported(const std::string &) { return true; }

std::string llm_video_transcript(const std::string &transcript) {
    const auto stripped = [](const std::string &value) {
        if (llm_output_type(value) != "str") throw LlmResponseError{"AttributeError", "'" + llm_output_type(value) + "' object has no attribute 'strip'"};
        return strip_config_value(parse_json_string_at(value, 0));
    };
    const auto formatted = [](const std::string &value) {
        const auto type = llm_output_type(value);
        if (type == "str") throw LlmResponseError{"ValueError", "Unknown format code 'f' for object of type 'str'"};
        if (type != "int" && type != "float" && type != "bool") throw LlmResponseError{"TypeError", "unsupported format string passed to " + type + ".__format__"};
        double number = 0;
        if (value == "true") number = 1;
        else if (value != "false" && !parse_strict_python_float_string(value, &number)) throw LlmResponseError{"ValueError", "Invalid timestamp"};
        if (type == "int" && !std::isfinite(number) && value != "NaN" && value != "Infinity" && value != "-Infinity") throw LlmResponseError{"OverflowError", "int too large to convert to float"};
        return one_decimal(number);
    };
    const auto segments = llm_output_get(transcript, "segments", "[]");
    if (llm_output_falsey(segments)) return stripped(llm_output_get(transcript, "text", "\"\""));
    const auto type = llm_output_type(segments);
    if (type == "dict" || type == "str") throw LlmResponseError{"AttributeError", "'str' object has no attribute 'get'"};
    if (type != "list") throw LlmResponseError{"TypeError", "'" + type + "' object is not iterable"};
    std::string output;
    for (const auto &segment : json_array_elements(segments)) {
        const auto start = llm_output_get(segment, "start", "0"), end = llm_output_get(segment, "end", "0");
        const auto text = stripped(llm_output_get(segment, "text", "\"\""));
        const auto speaker = llm_output_get(segment, "speaker", "null");
        if (text.empty()) continue;
        if (!output.empty()) output += '\n';
        output += '[' + formatted(start) + "s - " + formatted(end) + "s] ";
        if (!llm_output_falsey(speaker)) output += stream_python_str(speaker) + ": ";
        output += text;
    }
    return output;
}

std::string stt_audio_extension(const std::string &raw_format) {
    auto format = llm_output_falsey(raw_format) ? std::string("\"wav\"") : raw_format;
    auto type = llm_output_type(format);
    bool slash = false;
    if (type == "str") slash = parse_json_string_at(format, 0).find('/') != std::string::npos;
    else if (type == "dict") slash = !json_object_field(format, "/", "").empty();
    else if (type == "list") {
        for (const auto &item : json_array_elements(format)) if (llm_output_type(item) == "str" && parse_json_string_at(item, 0) == "/") slash = true;
    } else throw LlmResponseError{"TypeError", "argument of type '" + type + "' is not iterable"};
    if (slash) {
        if (type != "str") throw LlmResponseError{"AttributeError", "'" + type + "' object has no attribute 'split'"};
        const auto value = parse_json_string_at(format, 0);
        const auto first = value.find('/');
        format = '"' + json_escape(value.substr(first + 1, value.find('/', first + 1) - first - 1)) + '"';
    }
    if (type == "list" || type == "dict") throw LlmResponseError{"TypeError", "unhashable type: '" + type + "'"};
    SttAudioInput result;
    result.extension = parse_json_string_at(format, 0);
    if (result.extension == "mpeg") result.extension = "mp3";
    else if (result.extension == "x-wav") result.extension = "wav";
    else if (result.extension == "x-m4a") result.extension = "m4a";
    return result.extension;
}
std::string stt_base64_decode(const std::string &audio) {
    if (llm_output_type(audio) != "str")
        throw LlmResponseError{"TypeError", "argument should be a bytes-like object or ASCII string, not '" + llm_output_type(audio) + "'"};
    const auto encoded = parse_json_string_at(audio, 0);
    for (unsigned char c : encoded) if (c >= 128)
        throw LlmResponseError{"ValueError", "string argument should contain only ASCII characters"};
    return stt_base64_decode_bytes(encoded);
}
std::string stt_base64_decode_bytes(const std::string &encoded) {
    SttAudioInput result;
    unsigned position = 0, left = 0, padding = 0; size_t digits = 0;
    bool complete = false;
    for (unsigned char c : encoded) {
        if (c == '=') { if (position >= 2 && position + ++padding >= 4) { complete = true; break; } continue; }
        const auto value = base64_decode_value(c); if (value < 0) continue;
        padding = 0; ++digits;
        switch (position) {
        case 0: left = value; position = 1; break;
        case 1: result.bytes += static_cast<char>((left << 2) | (value >> 4)); left = value & 15; position = 2; break;
        case 2: result.bytes += static_cast<char>((left << 4) | (value >> 2)); left = value & 3; position = 3; break;
        case 3: result.bytes += static_cast<char>((left << 6) | value); left = 0; position = 0; break;
        }
    }
    if (!complete && position == 1) throw LlmResponseError{"Error", "Invalid base64-encoded string: number of data characters ("
        + std::to_string(digits) + ") cannot be 1 more than a multiple of 4"};
    if (!complete && position) throw LlmResponseError{"Error", "Incorrect padding"};
    return result.bytes;
}
SttAudioInput stt_audio_input(const std::string &audio, const std::string &raw_format) {
    SttAudioInput result; result.extension = stt_audio_extension(raw_format); result.bytes = stt_base64_decode(audio);
    if (result.extension.find('\0') != std::string::npos) throw LlmResponseError{"ValueError", "embedded null byte"};
    return result;
}

bool llm_audio_handoff(const std::string &generic_mode, const std::string &service_mode,
    const std::string &keep_loaded, const std::string &voice_server,
    const std::string &text_server, const std::vector<std::string> &models) {
    const auto flag = [](const std::string &value) { const auto word = to_lower(strip_config_value(value)); return word == "1" || word == "true" || word == "yes" || word == "on"; };
    const auto configured = [](std::string value) {
        if (to_lower(value) == "true") return false;
        while (!value.empty() && value.back() == '/') value.pop_back();
        return !value.empty();
    };
    const auto mode = to_lower(strip_config_value(service_mode.empty() ? generic_mode.empty() ? "auto" : generic_mode : service_mode));
    if (mode == "0" || mode == "false" || mode == "no" || mode == "off") return false;
    if (mode == "1" || mode == "true" || mode == "yes" || mode == "on") return !configured(voice_server);
    if (flag(keep_loaded) || configured(voice_server) || configured(text_server)) return false;
    if (to_lower(voice_server) == "true" && to_lower(text_server) != "true") return false;
    if (!models.empty() && std::all_of(models.begin(), models.end(), llm_minicpm5_model)) return false;
    return !models.empty();
}

std::string llm_apply_audio_format(const std::string &body, bool chat, const LlmInputOperations &operations) {
    const auto format = json_object_field(body, "audio_format", "");
    if (format.empty()) return body;
    const auto audio = chat ? input_at(input_last(input_at(body, "messages")), "content") : input_at(body, "prompt");
    const auto text = operations.transcribe(audio, format);
    if (!chat) return input_set(body, "prompt", text);
    auto messages = json_array_elements(input_at(body, "messages"));
    messages.back() = input_set(messages.back(), "content", text);
    return input_set(body, "messages", json_array_from_values(messages));
}

std::vector<std::string> llm_prepare_images(const std::vector<std::string> &images,
    const LlmInputOperations &operations, bool description) {
    std::vector<std::string> processed;
    const std::string user_agent = description ? "Mozilla/5.0"
        : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";
    for (const auto &img : images) {
        auto url = json_object_field(img, "image_url", "");
        if (url.empty()) continue;
        if (llm_output_type(url) == "dict") url = json_object_field(url, "url", "\"\"");
        // Both branches call startswith, including falsey non-string URLs.
        if (llm_output_type(url) != "str")
            throw LlmResponseError{"AttributeError", "'" + llm_output_type(url) + "' object has no attribute 'startswith'"};
        const auto text = parse_json_string_at(url, 0);
        if (!text.empty() && !starts_with(text, "data:")) {
            try {
                auto image = operations.fetch_image(text, user_agent);
                const auto &type = image.content_type;
                if (type == "image/webp" || type == "image/gif" || type == "image/bmp" || type == "image/tiff" || type == "image/avif") {
                    try { image.bytes = operations.png(image.bytes); image.content_type = "image/png"; }
                    catch (...) { if (description) throw; }
                }
                if (!starts_with(image.content_type, "image/")) image.content_type = "image/jpeg";
                processed.push_back(input_image(image.content_type, image.bytes));
            } catch (...) { /* Fetch failures omit this image. */ }
        } else if (starts_with(text, "data:image/webp") || starts_with(text, "data:image/gif")) {
            try {
                const auto comma = text.find(',');
                if (comma == std::string::npos) throw std::runtime_error("Missing data URL payload");
                processed.push_back(input_image("image/png", operations.png(input_decode_image(text.substr(comma + 1)))));
            } catch (...) { processed.push_back(img); }
        } else processed.push_back(img);
    }
    return processed;
}

std::string llm_prepare_text_image_input(const std::string &body, bool chat, const LlmInputOperations &operations) {
    // Keep the outer request parsed while preparing its messages. Rebuilding
    // and reparsing the entire history for each scalar update is expensive on
    // a returning long conversation. Upsert retains Python dict ordering and
    // the last value of duplicate decoded keys, as input_set does.
    using Fields = std::vector<std::pair<std::string, std::string>>;
    const auto parse_fields = [](const std::string &object) {
        Fields fields;
        for (const auto &entry : json_object_ordered_entries(object)) ordered_json_upsert(&fields, entry);
        return fields;
    };
    auto data = parse_fields(body);
    const auto field = [&](const std::string &key, const std::string &fallback) {
        const auto found = std::find_if(data.begin(), data.end(), [&](const auto &entry) { return entry.first == key; });
        return found == data.end() ? fallback : found->second;
    };
    const auto required = [&](const std::string &key) {
        auto value = field(key, "");
        if (value.empty()) throw LlmResponseError{"KeyError", python_repr_string(key)};
        return value;
    };
    const auto assign = [&](const std::string &key, const std::string &value) { ordered_json_upsert(&data, {key, value}); };
    assign("local_uri", '"' + json_escape(operations.local_uri) + '"');
    auto messages = field("messages", "");
    std::vector<std::string> images, items;
    if (!messages.empty()) {
        const auto type = llm_output_type(messages);
        if (type == "list") items = json_array_elements(messages);
        else if (type == "dict" || type == "str") {
            if (!llm_output_falsey(messages)) throw LlmResponseError{"AttributeError", "'str' object has no attribute 'get'"};
        } else throw LlmResponseError{"TypeError", "'" + type + "' object is not iterable"};
        for (auto &message : items) {
            const auto content = llm_output_get(message, "content", "null");
            if (llm_output_type(content) != "list") continue;
            std::string text;
            for (const auto &part : json_array_elements(content)) {
                if (llm_output_type(part) == "dict") {
                    if (input_type_is(part, "text")) {
                        const auto value = json_object_field(part, "text", "\"\"");
                        if (llm_output_type(value) != "str")
                            throw LlmResponseError{"TypeError", "can only concatenate str (not \"" + llm_output_type(value) + "\") to str"};
                        text += parse_json_string_at(value, 0);
                    } else if (!json_object_field(part, "image_url", "").empty()) images.push_back(input_json_value(part));
                    else if (!json_object_field(part, "video_url", "").empty() || input_type_is(part, "video")) {
                        auto url = json_object_field(part, "video_url", "");
                        if (url.empty()) url = json_object_field(part, "video", "null");
                        else if (llm_output_type(url) == "dict") url = json_object_field(url, "url", "\"\"");
                        if (!llm_output_falsey(url)) {
                            double fps = 0;
                            if (!fallback_float(operations.video_fps, &fps))
                                throw LlmResponseError{"ValueError", "could not convert string to float: " + python_repr_string(operations.video_fps)};
                            int64_t maximum;
                            try { maximum = std::stoll(llm_parameter_integer(operations.video_max_frames)); }
                            catch (const LlmParameterError &error) { throw LlmResponseError{error.category, error.what()}; }
                            auto video = operations.extract_video(url, fps, maximum);
                            if (!video.frames.empty()) {
                                for (const auto &frame : video.frames) images.push_back("{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + json_escape(frame) + "\"}}");
                                std::string times;
                                for (size_t i = 0; i < video.timestamps.size(); ++i) { if (i) times += ", "; times += "Frame " + std::to_string(i + 1) + ": " + one_decimal(video.timestamps[i]) + 's'; }
                                auto header = "[Video: " + one_decimal(video.duration) + "s duration, " + std::to_string(video.frames.size())
                                    + " frames extracted]\n[Frame timestamps: " + times + ']';
                                if (!video.audio_path.empty()) {
                                    try {
                                        const auto bytes = operations.read_audio(video.audio_path);
                                        const auto encoded = base64_encode(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
                                        const auto transcript = operations.transcribe_segments('"' + json_escape(encoded) + '"', "\"wav\"");
                                        if (!llm_output_falsey(transcript)) {
                                            const auto formatted = llm_video_transcript(transcript);
                                            if (!formatted.empty()) header += "\n[Video Audio Transcript:\n" + formatted + "\n]";
                                        }
                                    } catch (...) { /* Video transcript failures do not discard its frames. */ }
                                    operations.remove_audio(video.audio_path);
                                }
                                text = header + '\n' + text;
                            } else if (!video.audio_path.empty()) operations.remove_audio(video.audio_path);
                        }
                    } else if (!json_object_field(part, "audio_url", "").empty()) {
                        auto url = json_object_field(part, "audio_url", "null");
                        const auto type = llm_output_type(url);
                        bool contains = false;
                        if (type == "dict") contains = !json_object_field(url, "url", "").empty();
                        else if (type == "str") contains = parse_json_string_at(url, 0).find("url") != std::string::npos;
                        else if (type == "list") { for (const auto &item : json_array_elements(url)) if (llm_output_type(item) == "str" && parse_json_string_at(item, 0) == "url") contains = true; }
                        else throw LlmResponseError{"TypeError", "argument of type '" + type + "' is not iterable"};
                        if (contains) url = input_at(url, "url");
                        if (llm_output_type(url) != "str") throw LlmResponseError{"AttributeError", "'" + llm_output_type(url) + "' object has no attribute 'startswith'"};
                        auto audio = parse_json_string_at(url, 0);
                        std::string format = "wav";
                        if (starts_with(audio, "data:")) {
                            const auto comma = audio.find(',');
                            if (comma == std::string::npos) throw LlmResponseError{"IndexError", "list index out of range"};
                            audio = audio.substr(comma + 1, audio.find(',', comma + 1) - comma - 1);
                            format = audio.substr(0, audio.find(';'));
                        } else {
                            const auto bytes = operations.fetch_audio(audio);
                            audio = base64_encode(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
                        }
                        const auto transcript = operations.transcribe('"' + json_escape(audio) + '"', '"' + json_escape(format) + '"');
                        text = "Transcribed Audio: " + stream_python_str(transcript) + "\n\n" + text;
                    }
                } else if (llm_output_type(part) == "str") text += parse_json_string_at(part, 0);
            }
            message = input_set(message, "content", '"' + json_escape(text) + '"');
        }
        if (type == "list") messages = json_array_from_values(items);
        // The legacy branch is unreachable after JSON lists were flattened,
        // but its condition still indexes the last content when images is empty.
        if (images.empty()) (void)input_at(input_last(messages), "content");
        assign("messages", messages);
    }
    const auto model = operations.load();
    assign("model", '"' + json_escape(model.source) + '"');
    const auto stop = field("stop", "null");
    if (!llm_output_falsey(stop)) {
        auto stops = json_array_elements(model.stops);
        if (llm_output_type(stop) == "list") { const auto requested = json_array_elements(stop); stops.insert(stops.end(), requested.begin(), requested.end()); }
        else stops.push_back(stop);
        assign("stop", json_array_from_values(stops));
    }
    // A present falsey audio format still invokes transcription in the source.
    // Keep that helper's error and callback ordering; the ordinary text path
    // needs no serialization boundary here.
    if (!field("audio_format", "").empty()) {
        data = parse_fields(llm_apply_audio_format(json_object_from_ordered_entries(data), chat, operations));
        if (!messages.empty()) items = json_array_elements(field("messages", "[]"));
    }
    std::string user_message;
    if (!chat) user_message = required("prompt");
    else if (!items.empty()) user_message = input_at(items.back(), "content");
    else user_message = input_at(input_last(required("messages")), "content");
    if (model.vision && !images.empty()) {
        auto processed = llm_prepare_images(images, operations);
        if (chat && !processed.empty()) {
            processed.insert(processed.begin(), "{\"type\":\"text\",\"text\":" + user_message + '}');
            items.back() = input_set(items.back(), "content", json_array_from_values(processed));
            assign("messages", json_array_from_values(items));
        }
    } else if (!images.empty() && operations.describe) {
        const auto user_text = stream_python_str(user_message);
        const auto description = operations.describe(images, user_text);
        if (!llm_output_falsey(description)) {
            const auto enhanced = '"' + json_escape("[Image Description: " + stream_python_str(description)
                + "]\n\nUser's question: " + user_text) + '"';
            if (chat) { items.back() = input_set(items.back(), "content", enhanced); assign("messages", json_array_from_values(items)); }
            else assign("prompt", enhanced);
        }
    }
    return json_object_from_ordered_entries(data);
}

std::string llm_stream_delta(const std::string &json) {
    const auto value = trim(json);
    if (value.empty() || value.front() != '{') return "{}";
    const auto raw_delta = json_object_field(value, "delta", "{}");
    for (const auto *keys : {&stream_content_keys, &stream_reasoning_keys}) {
        for (const auto *mapping : {&value, &raw_delta}) {
            if (mapping->empty() || mapping->front() != '{') continue;
            for (const auto &key : *keys) {
                const auto text = stream_text_value(json_object_field(*mapping, key, "null"));
                if (!text.empty()) return "{\"" + std::string(keys == &stream_content_keys ? "content" : "reasoning_content") + "\":\"" + json_escape(text) + "\"}";
            }
        }
    }
    return "{}";
}

bool llm_stream_has_assistant_text(const std::string &json) {
    const auto value = trim(json);
    if (value.empty()) return false;
    if (value.front() == '{') {
        const auto choices = json_object_field(value, "choices", "null");
        if (!choices.empty() && choices.front() == '[') {
            for (const auto &choice : json_array_elements(choices)) {
                if (choice.empty() || choice.front() != '{') continue;
                if (llm_stream_has_assistant_text(json_object_field(choice, "delta", "{}"))
                    || llm_stream_has_assistant_text(json_object_field(choice, "message", "{}"))
                    || llm_stream_delta(choice) != "{}") return true;
            }
            return false;
        }
        return llm_stream_delta(value) != "{}";
    }
    if (value.front() == '"') {
        const auto parsed = parse_json_string_at(value, 0);
        if (strip_config_value(parsed).empty()) return false;
        if (json_decode_error(parsed) || stream_json_integer_limit(parsed)) return true;
        return llm_stream_has_assistant_text(parsed);
    }
    return false;
}

std::string llm_stream_finish_reason(const std::string &json) {
    const auto value = trim(json);
    if (value.empty()) return "";
    if (value.front() == '{') {
        const auto choices = json_object_field(value, "choices", "null");
        if (!choices.empty() && choices.front() == '[') {
            for (const auto &choice : json_array_elements(choices)) {
                if (choice.empty() || choice.front() != '{') continue;
                const auto reason = json_object_field(choice, "finish_reason", "null");
                if (!llm_output_falsey(reason)) return stream_python_str(reason);
            }
            return "";
        }
        const auto reason = json_object_field(value, "finish_reason", "null");
        return llm_output_falsey(reason) ? "" : stream_python_str(reason);
    }
    if (value.front() == '"') {
        const auto parsed = parse_json_string_at(value, 0);
        if (!strip_config_value(parsed).empty() && !json_decode_error(parsed) && !stream_json_integer_limit(parsed)) return llm_stream_finish_reason(parsed);
    }
    return "";
}

LlmChatStreamNormalizer::LlmChatStreamNormalizer(std::string model, std::string backend, std::string version,
    int64_t id_time, int64_t created)
    : model_(std::move(model)), backend_(std::move(backend)), version_(std::move(version)),
      id_("chatcmpl-" + std::to_string(id_time)), created_(created) {}

std::string LlmChatStreamNormalizer::wrap(const std::string &delta, const std::string &reason) const {
    return stream_json_dumps("{\"id\":\"" + id_ + "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created_)
        + ",\"model\":\"" + json_escape(model_) + "\",\"choices\":[{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":" + reason + "}]}");
}

void LlmChatStreamNormalizer::chunk(const std::string &json, std::vector<std::string> *output) {
    ++raw_chunks_;
    auto value = trim(json);
    const bool string_chunk = !value.empty() && value.front() == '"';
    if (string_chunk) {
        const auto parsed = parse_json_string_at(value, 0);
        if (json_decode_error(parsed)) {
            ++assistant_chunks_;
            output->push_back(wrap("{\"content\":" + value + '}', "null"));
            return;
        }
        // Python's integer conversion limit raises ValueError, caught by the
        // outer processing handler, rather than the raw-text JSONDecodeError.
        if (stream_json_integer_limit(parsed)) return;
        value = trim(parsed);
    }
    if (!value.empty() && value.front() == '{') {
        for (const auto &source : {"usage", "timings"}) {
            const auto values = json_object_field(value, source, "{}");
            if (values.empty() || values.front() != '{') continue;
            const std::vector<std::string> keys = std::string(source) == "usage"
                ? std::vector<std::string>{"prompt_tokens", "completion_tokens"}
                : std::vector<std::string>{"prompt_n", "predicted_n", "prompt_ms", "predicted_ms"};
            for (const auto &key : keys) {
                const auto number = json_object_field(values, key, "null");
                if (number == "true" || number == "false" || number == "NaN" || number == "Infinity" || number == "-Infinity" || is_json_number_token(number)) diagnostics_[key] = number;
            }
        }
    }
    bool choices_present = !value.empty() && value.front() == '{' && !json_object_field(value, "choices", "").empty();
    // Python's `"choices" in parsed` also accepts strings and lists. These
    // unusual JSON-string callbacks are yielded unchanged by the reference.
    if (string_chunk && !value.empty() && value.front() == '"') choices_present = parse_json_string_at(value, 0).find("choices") != std::string::npos;
    if (string_chunk && !value.empty() && value.front() == '[')
        for (const auto &item : json_array_elements(value))
            if (item == "\"choices\"") choices_present = true;
    if (choices_present) {
        const bool has_text = llm_stream_has_assistant_text(value);
        if (has_text) ++assistant_chunks_;
        if (!has_text && !llm_stream_finish_reason(value).empty()) pending_final_ = stream_json_dumps(value);
        else output->push_back(stream_json_dumps(value));
        return;
    }
    const auto delta = llm_stream_delta(value);
    if (delta != "{}") {
        ++assistant_chunks_;
        output->push_back(wrap(delta, json_object_field(value, "finish_reason", "null")));
    }
}

std::vector<std::string> LlmChatStreamNormalizer::callback(const std::string &json) {
    std::vector<std::string> output;
    if (finished_ || failure_) return output;
    const auto value = trim(json);
    failure_ = llm_response_error(value, true);
    if (failure_) return output;
    if (!value.empty() && value.front() == '[') {
        for (const auto &item : json_array_elements(value)) chunk(item, &output);
    } else chunk(value, &output);
    return output;
}

std::vector<std::string> llm_stream_callback_items(const std::string &json) {
    const auto value = trim(json);
    return !value.empty() && value.front() == '[' ? json_array_elements(value) : std::vector<std::string>{value};
}

std::vector<std::string> LlmChatStreamNormalizer::queued_chunk(const std::string &json) {
    std::vector<std::string> output;
    if (!finished_) chunk(json, &output);
    return output;
}

std::vector<std::string> LlmChatStreamNormalizer::finish() {
    if (finished_) return {};
    finished_ = true;
    if (failure_) return {};
    if (!assistant_chunks_) {
        std::vector<std::pair<std::string, std::string>> diagnostics(diagnostics_.begin(), diagnostics_.end());
        const auto reason = llm_stream_finish_reason(pending_final_);
        const auto message = "LLM stream completed without any assistant text. raw_chunks=" + std::to_string(raw_chunks_)
            + "; model=" + model_ + "; finish_reason=" + (reason.empty() ? "missing" : reason)
            + "; backend=" + backend_ + "; xllamacpp=" + version_ + "; timings=" + stream_json_dumps(json_object_from_ordered_entries(diagnostics));
        return {stream_json_dumps("{\"error\":{\"message\":\"" + json_escape(message) + "\",\"type\":\"empty_stream\"}}")};
    }
    if (!pending_final_.empty()) return {pending_final_};
    return {wrap("{}", "\"stop\"")};
}

std::string LlmChatStreamNormalizer::keepalive() const { return wrap("{}", "null"); }

std::string llm_stream_error_event(const std::string &message) {
    // app.py slices Python characters before json.dumps' ASCII escaping.
    const auto &value = message.empty() ? std::string("Streaming failed") : message;
    std::string escaped;
    const auto hex16 = [&](uint32_t code) {
        constexpr char digits[] = "0123456789abcdef";
        escaped += "\\u";
        for (int shift = 12; shift >= 0; shift -= 4) escaped += digits[(code >> shift) & 15];
    };
    size_t pos = 0;
    for (size_t characters = 0; pos < value.size() && characters < 1000; ++characters) {
        const auto first = static_cast<unsigned char>(value[pos++]);
        if (first < 0x7f) { escaped += json_escape(std::string(1, static_cast<char>(first))); continue; }
        uint32_t code = first;
        const unsigned tails = first < 0x80 ? 0 : first < 0xe0 ? 1 : first < 0xf0 ? 2 : 3;
        if (tails) {
            code &= 0x7f >> tails;
            for (unsigned i = 0; i < tails && pos < value.size(); ++i) code = (code << 6) | (static_cast<unsigned char>(value[pos++]) & 63);
        }
        if (code <= 0xffff) hex16(code);
        else { code -= 0x10000; hex16(0xd800 + (code >> 10)); hex16(0xdc00 + (code & 1023)); }
    }
    return "data: {\"error\": {\"message\": \"" + escaped + "\", \"type\": \"streaming_error\"}}\n\n";
}

namespace {
void produce_local_chat_stream(const std::string &url, const std::string &path, const std::string &api_key,
    const std::string &body, const LlmInferenceStream::Emit &emit, const LlmInferenceStream::Interrupt &interrupt) {
    std::shared_ptr<WorkerStreamState> state;
    const auto response = http_worker_stream_request("POST", url, path, api_key, "application/json", body, 0, &state, interrupt);
    if (response.transport_error || !state) throw std::runtime_error(response.body.empty() ? "llama.cpp request failed" : response.body);
    std::string buffered, event;
    bool mode_known = starts_with(to_lower(response.content_type), "text/event-stream"), sse = mode_known;
    bool stopped = false;
    const auto callback = [&](const std::string &value) {
        // The pinned dict wrapper aborts silently before invoking Python when
        // json.loads fails, including its decimal integer conversion limit.
        if (json_decode_error(value) || stream_json_integer_limit(value)) return true;
        return emit(value);
    };
    const auto consume_event = [&] {
        if (event == "[DONE]") stopped = true;
        else if (!event.empty()) stopped = callback(event);
        event.clear();
    };
    const auto feed = [&](const std::string &bytes) {
        if (stopped) return;
        buffered += bytes;
        if (!mode_known) {
            const auto value = trim(buffered);
            if (value.size() < 5) return;
            sse = starts_with(value, "data:"); mode_known = true;
        }
        if (!sse) return;
        size_t end;
        while (!stopped && (end = buffered.find('\n')) != std::string::npos) {
            auto line = buffered.substr(0, end); buffered.erase(0, end + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) consume_event();
            else if (starts_with(line, "data:")) {
                if (!event.empty()) event += '\n';
                event += line.substr(line.size() > 5 && line[5] == ' ' ? 6 : 5);
            }
        }
    };
    StreamingChunkedDecoder decoder;
    bool framed_done = false;
    size_t received = 0;
    const auto framed = [&](const std::string &bytes) {
        if (state->upstream_chunked) {
            std::vector<std::string> chunks;
            if (!decoder.feed(bytes, &chunks, &framed_done)) throw std::runtime_error("Malformed worker response");
            for (const auto &chunk : chunks) feed(chunk);
        } else {
            const auto count = state->content_length ? std::min(bytes.size(), *state->content_length - received) : bytes.size();
            received += count; feed(bytes.substr(0, count));
            framed_done = state->content_length && received >= *state->content_length;
        }
    };
    framed(state->initial_body);
    char buffer[4096];
    while (!stopped && !framed_done) {
        bool timed_out = false;
        const auto count = state->read_some(buffer, sizeof(buffer), &timed_out);
        if (count <= 0) break;
        framed(std::string(buffer, static_cast<size_t>(count)));
    }
    if (!stopped) {
        if (sse) {
            if (!buffered.empty()) feed("\n");
            if (!stopped) consume_event();
        } else if (!buffered.empty()) callback(buffered);
    }
    state->close_all();
}
}

EchoInferenceBackend::EchoInferenceBackend(std::string default_model)
    : default_model_(std::move(default_model)) {}

std::string EchoInferenceBackend::chat(const std::string &request_body) {
    const auto model = extract_json_string(request_body, "model", default_model_);
    const auto last_content = extract_last_json_string(request_body, "content", "Hello from ezlocalai-native.");
    const std::string subtitle_translation = echo_subtitle_translation_response(request_body);

    std::ostringstream out;
    out << "{"
        << "\"id\":\"chatcmpl-native-" << now_id() << "\","
        << "\"object\":\"chat.completion\","
        << "\"created\":" << unix_seconds() << ","
        << "\"model\":\"" << json_escape(model) << "\","
        << "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
        << json_escape(subtitle_translation.empty() ? "Native ezlocalai echo backend received: " + last_content : subtitle_translation)
        << "\"},\"finish_reason\":\"stop\"}],"
        << "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        << "}";
    return out.str();
}

std::string EchoInferenceBackend::completion(const std::string &request_body) {
    const auto model = extract_json_string(request_body, "model", default_model_);
    const auto prompt = extract_json_string(request_body, "prompt", "");
    std::ostringstream out;
    out << "{"
        << "\"id\":\"cmpl-native-" << now_id() << "\","
        << "\"object\":\"text_completion\","
        << "\"created\":" << unix_seconds() << ","
        << "\"model\":\"" << json_escape(model) << "\","
        << "\"choices\":[{\"text\":\""
        << json_escape("Native ezlocalai echo backend received: " + prompt)
        << "\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"stop\"}],"
        << "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        << "}";
    return out.str();
}

std::string EchoInferenceBackend::embedding(const std::string &request_body) {
    std::vector<std::string> inputs;
    const auto raw_input = trim(extract_json_value(request_body, "input", "\"\""));
    if (!raw_input.empty() && raw_input.front() == '[') {
        for (const auto &element : json_array_elements(raw_input)) {
            const auto trimmed = trim(element);
            if (!trimmed.empty() && trimmed.front() == '"') {
                inputs.push_back(parse_json_string_at(trimmed, 0));
            }
        }
    } else {
        inputs.push_back(extract_json_string(request_body, "input", ""));
    }
    if (inputs.empty()) {
        inputs.push_back("");
    }

    auto embedding_values = [](const std::string &input) {
        unsigned long hash = 1469598103UL;
        for (char ch : input) {
            hash ^= static_cast<unsigned char>(ch);
            hash *= 16777619UL;
        }
        std::ostringstream values;
        for (int i = 0; i < 8; ++i) {
            if (i) {
                values << ",";
            }
            const double value = static_cast<double>((hash >> (i * 4)) & 0xff) / 255.0;
            values << value;
        }
        return values.str();
    };

    std::ostringstream out;
    out << "{"
        << "\"object\":\"list\","
        << "\"model\":\"bge-m3\","
        << "\"data\":[";
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << "{\"object\":\"embedding\",\"index\":" << i
            << ",\"embedding\":[" << embedding_values(inputs[i]) << "]}";
    }
    out << "],"
        << "\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}"
        << "}";
    return out.str();
}

std::vector<std::string> split_csv_values(std::string raw) {
    std::vector<std::string> values;
    size_t start = 0;
    while (start <= raw.size()) {
        const auto comma = raw.find(',', start);
        std::string value = trim(raw.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!value.empty()) {
            values.push_back(value);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return values;
}

std::string csv_env_value_for_index(const char *key, size_t index, const std::string &fallback = "") {
    const auto values = split_csv_values(ezl_getenv_default(key));
    if (values.empty()) {
        return fallback;
    }
    return index < values.size() ? values[index] : values.back();
}

std::string configured_parallel_value_for_index(size_t index) {
    ezl_load_dotenv();
    const char *n_parallel_env = std::getenv("N_PARALLEL");
    const char *max_concurrent_env = std::getenv("MAX_CONCURRENT_REQUESTS");
    const std::string raw = n_parallel_env && n_parallel_env[0] != '\0'
        ? n_parallel_env
        : (max_concurrent_env && max_concurrent_env[0] != '\0'
            ? max_concurrent_env
            : ezl_getenv_default("N_PARALLEL"));
    const auto values = split_csv_values(raw);
    if (values.empty()) {
        return "";
    }
    return index < values.size() ? values[index] : values.back();
}

std::vector<std::string> split_shell_words_simple(const std::string &raw) {
    std::vector<std::string> out;
    std::string current;
    char quote = '\0';
    bool escaped = false;
    for (const char ch : raw) {
        if (escaped) {
            current.push_back(ch);
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (quote) {
            if (ch == quote) {
                quote = '\0';
            } else {
                current.push_back(ch);
            }
            continue;
        }
        if (ch == '\'' || ch == '"') {
            quote = ch;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

bool llm_fallback_disabled(const std::string &body) {
    const auto value = to_lower(strip_config_value(python_str_from_json_value(json_object_field(body, "disable_fallback", "false"))));
    return value == "true" || value == "1" || value == "yes" || value == "on";
}

bool env_truthy(const char *name) {
    const char *raw = getenv(name);
    if (!raw) {
        return false;
    }
    const std::string lowered = to_lower(trim(raw));
    return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

constexpr int kLlamaGpuLayersUnset = -1000000;

int parse_llama_gpu_layers_env() {
    int parsed = 0;
    if (!parse_strict_int_string(ezl_getenv_default("GPU_LAYERS"), &parsed)) {
        return kLlamaGpuLayersUnset;
    }
    return parsed == -2 ? -1 : parsed;
}

int llama_auto_gpu_layers() {
    const auto gpu = read_gpu_info();
    if (gpu.gpu_count <= 0) return 0;
    // If VRAM is unknown (unified memory, nvidia-smi returns N/A), trust the
    // CUDA device and offload all layers. Only fall back to CPU when we can
    // positively determine VRAM is too small (< 1 GB).
    if (gpu.free_vram_gb <= 0.0) return -1;
    return gpu.free_vram_gb > 1.0 ? -1 : 0;
}

bool read_exact(std::ifstream &in, char *data, std::streamsize size) {
    in.read(data, size);
    return in.good();
}

template <typename T>
bool read_little(std::ifstream &in, T *value) {
    return read_exact(in, reinterpret_cast<char *>(value), static_cast<std::streamsize>(sizeof(T)));
}

bool skip_bytes(std::ifstream &in, uint64_t count) {
    if (count > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }
    in.seekg(static_cast<std::streamoff>(count), std::ios::cur);
    return in.good();
}

bool read_gguf_string(std::ifstream &in, std::string *value) {
    uint64_t length = 0;
    if (!read_little(in, &length) || length > (64ULL * 1024ULL * 1024ULL)) {
        return false;
    }
    value->assign(static_cast<size_t>(length), '\0');
    return length == 0 || read_exact(in, value->data(), static_cast<std::streamsize>(length));
}

uint64_t gguf_scalar_size(uint32_t type) {
    switch (type) {
    case 0:
    case 1:
    case 7:
        return 1;
    case 2:
    case 3:
        return 2;
    case 4:
    case 5:
    case 6:
        return 4;
    case 10:
    case 11:
    case 12:
        return 8;
    default:
        return 0;
    }
}

bool skip_gguf_value(std::ifstream &in, uint32_t type) {
    const uint64_t scalar = gguf_scalar_size(type);
    if (scalar > 0) {
        return skip_bytes(in, scalar);
    }
    if (type == 8) {
        std::string ignored;
        return read_gguf_string(in, &ignored);
    }
    if (type == 9) {
        uint32_t element_type = 0;
        uint64_t count = 0;
        if (!read_little(in, &element_type) || !read_little(in, &count)) {
            return false;
        }
        const uint64_t element_size = gguf_scalar_size(element_type);
        if (element_size > 0) {
            if (count > std::numeric_limits<uint64_t>::max() / element_size) {
                return false;
            }
            return skip_bytes(in, count * element_size);
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (!skip_gguf_value(in, element_type)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool read_gguf_integer_value(std::ifstream &in, uint32_t type, int64_t *value) {
    switch (type) {
    case 0: {
        uint8_t v = 0;
        if (!read_little(in, &v)) return false;
        *value = v;
        return true;
    }
    case 1: {
        int8_t v = 0;
        if (!read_little(in, &v)) return false;
        *value = v;
        return true;
    }
    case 2: {
        uint16_t v = 0;
        if (!read_little(in, &v)) return false;
        *value = v;
        return true;
    }
    case 3: {
        int16_t v = 0;
        if (!read_little(in, &v)) return false;
        *value = v;
        return true;
    }
    case 4: {
        uint32_t v = 0;
        if (!read_little(in, &v)) return false;
        *value = v;
        return true;
    }
    case 5: {
        int32_t v = 0;
        if (!read_little(in, &v)) return false;
        *value = v;
        return true;
    }
    case 10: {
        uint64_t v = 0;
        if (!read_little(in, &v) || v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return false;
        *value = static_cast<int64_t>(v);
        return true;
    }
    case 11: {
        int64_t v = 0;
        if (!read_little(in, &v)) return false;
        *value = v;
        return true;
    }
    default:
        return false;
    }
}

int gguf_total_layers_for_path(const std::string &model_path) {
    if (model_path.empty()) {
        return 0;
    }
    std::ifstream in(model_path, std::ios::binary);
    if (!in) {
        return 0;
    }
    char magic[4] = {};
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!read_exact(in, magic, 4)
        || std::string(magic, magic + 4) != "GGUF"
        || !read_little(in, &version)
        || !read_little(in, &tensor_count)
        || !read_little(in, &kv_count)) {
        return 0;
    }
    (void)version;
    (void)tensor_count;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_string(in, &key) || !read_little(in, &type)) {
            return 0;
        }
        if (key.size() >= 12 && key.compare(key.size() - 12, 12, ".block_count") == 0) {
            int64_t block_count = 0;
            if (read_gguf_integer_value(in, type, &block_count) && block_count >= 0 && block_count < 1000000) {
                return static_cast<int>(block_count) + 1;
            }
            return 0;
        }
        if (!skip_gguf_value(in, type)) {
            return 0;
        }
    }
    return 0;
}

int llama_total_layers_for_path(const std::string &model_path) {
    int configured = 0;
    if (parse_strict_int_string(ezl_getenv_default("LLAMA_CPP_TOTAL_LAYERS"), &configured) && configured > 0) {
        return configured;
    }
    const int parsed = gguf_total_layers_for_path(model_path);
    if (parsed > 0) {
        return parsed;
    }
    return 41;
}

int llama_next_reduced_gpu_layers(int current_layers, int total_layers) {
    const int layers = total_layers > 0 ? total_layers : 41;
    int effective_current = current_layers;
    if (effective_current == kLlamaGpuLayersUnset || effective_current == -1) {
        effective_current = layers;
    }
    effective_current = std::min(std::max(effective_current, 0), layers);
    if (effective_current == 0) {
        return kLlamaGpuLayersUnset;
    }
    for (double fraction : {0.75, 0.5, 0.25, 0.0}) {
        const int candidate = std::max(0, static_cast<int>(layers * fraction));
        if (candidate < effective_current) {
            return candidate;
        }
    }
    return 0;
}

std::string single_gpu_tensor_split_arg(int main_gpu, int gpu_count) {
    if (main_gpu < 0 || gpu_count <= 1 || main_gpu >= gpu_count) {
        return "";
    }
    std::ostringstream out;
    for (int i = 0; i < gpu_count; ++i) {
        if (i) {
            out << ",";
        }
        out << (i == main_gpu ? "1" : "0");
    }
    return out.str();
}

std::string llama_model_file_arg(const std::filesystem::path &path) {
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec);
    return ec ? path.string() : absolute.string();
}

std::string first_existing_llama_model_file(const std::vector<std::filesystem::path> &roots, const std::string &model_id) {
    std::string base = model_id;
    const auto slash = base.find_last_of('/');
    if (slash != std::string::npos) {
        base = base.substr(slash + 1);
    }
    std::string lowered_base = to_lower(base);
    const auto gguf = lowered_base.find("-gguf");
    if (gguf != std::string::npos) {
        lowered_base = lowered_base.substr(0, gguf);
    }

    std::filesystem::path fallback;
    for (const auto &root : roots) {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) {
            continue;
        }
        for (std::filesystem::recursive_directory_iterator it(root, std::filesystem::directory_options::skip_permission_denied, ec), end;
             !ec && it != end;
             it.increment(ec)) {
            if (!it->is_regular_file(ec)) {
                continue;
            }
            const auto path = it->path();
            const std::string filename = to_lower(path.filename().string());
            if (path.extension() != ".gguf" || filename.find("mmproj") != std::string::npos) {
                continue;
            }
            if (fallback.empty()) {
                fallback = path;
            }
            if (!lowered_base.empty() && filename.find(lowered_base) != std::string::npos) {
                return llama_model_file_arg(path);
            }
        }
    }
    return fallback.empty() ? "" : llama_model_file_arg(fallback);
}

std::vector<std::filesystem::path> llama_model_roots() {
    std::vector<std::filesystem::path> roots;
    const std::string models_path = trim(ezl_getenv_default("MODELS_PATH"));
    if (!models_path.empty()) {
        roots.emplace_back(models_path);
    }
    roots.emplace_back("models");
    roots.emplace_back("/data/models");
    return roots;
}

std::string resolve_llama_model_path(const std::string &model_id, size_t index) {
    const auto configured_paths = split_csv_values(ezl_getenv_default("LLAMA_CPP_MODEL_PATHS"));
    if (index < configured_paths.size()) {
        return configured_paths[index];
    }
    const std::string single_path = trim(ezl_getenv_default("LLAMA_CPP_MODEL_PATH"));
    if (!single_path.empty()) {
        return single_path;
    }
    const std::string source_model = model_id;
    std::error_code ec;
    if (!source_model.empty() && std::filesystem::exists(source_model, ec)) {
        return source_model;
    }

    // Repository-backed text models are selected by LLM.download_model's
    // per-model quantization policy at load time, not by a recursive fallback
    // which can silently choose another model or a different quantization.
    return {};
}

std::string llama_hf_model_arg(const std::string &model_id, size_t index, bool embedding_mode) {
    std::string model = source_model_name(model_id);
    if (embedding_mode) return embedding_model_config().hf_argument;
    if (model.empty() || model.find(':') != std::string::npos) {
        return model;
    }
    std::string quant_type = trim(csv_env_value_for_index("QUANT_TYPE", index));
    if (quant_type.empty()) {
        return model;
    }
    return model + ":" + quant_type;
}

int parse_positive_int_env(const char *name, int fallback) {
    int parsed = 0;
    return parse_strict_int_string(ezl_getenv_default(name), &parsed) && parsed > 0 ? parsed : fallback;
}

int embedding_int_setting(const char *name, int fallback) {
    int parsed = 0;
    const auto value = strip_config_value(ezl_getenv_fallback(name, std::to_string(fallback).c_str()));
    return parse_strict_int_string(value, &parsed) ? parsed : fallback;
}

bool llama_env_truthy_default(const char *name, bool fallback) {
    const std::string raw = to_lower(trim(ezl_getenv_default(name)));
    if (raw.empty()) {
        return fallback;
    }
    if (raw == "1" || raw == "true" || raw == "yes" || raw == "on") {
        return true;
    }
    if (raw == "0" || raw == "false" || raw == "no" || raw == "off") {
        return false;
    }
    return fallback;
}

int llama_configured_context_size(size_t index, bool embedding_mode) {
    const std::string ctx = trim(csv_env_value_for_index(
        embedding_mode ? "LLAMA_CPP_EMBEDDING_CONTEXT" : "LLAMA_CPP_CONTEXT",
        index
    ));
    int parsed = 0;
    if (parse_strict_int_string(ctx, &parsed) && parsed > 0) {
        return parsed;
    }
    if (embedding_mode) {
        return embedding_int_setting("EMBEDDING_CONTEXT_LENGTH", 10000);
    }
    return 0;
}

int embedding_startup_context_size(size_t index, int requested_context) {
    if (requested_context > 0) return requested_context;
    const int configured = llama_configured_context_size(index, true);
    if (configured > 0) return configured;
    const int fallback = parse_first_csv_int_env("LLM_MAX_TOKENS", 0);
    return fallback > 0 ? fallback : 0;
}

LlmCatalog captured_llm_catalog(const std::string &default_model) {
    LlmCatalogSettings settings;
    settings.models = default_model;
    settings.main_gpus = ezl_getenv_fallback("MAIN_GPU", "0");
    settings.contexts = ezl_getenv_fallback("LLM_MAX_TOKENS", "65536");
    settings.parallels = ezl_getenv_fallback("N_PARALLEL", "1");
    settings.quants = ezl_getenv_fallback("QUANT_TYPE", "Q4_K_XL");
    settings.speculative_type = ezl_getenv_fallback("LLM_SPECULATIVE_TYPE", "auto");
    settings.gpu_count = visible_cuda_device_count();
    auto catalog = parse_llm_catalog(settings);
    // Preserve the existing explicit llama-server context extension. Added
    // instances inherit the originating CSV entry, not their appended index.
    for (auto &entry : catalog.configs) {
        int context = llama_configured_context_size(entry.second.original_index, false);
        if (context <= 0) parse_strict_int_string(csv_env_value_for_index("LLAMA_CPP_DEFAULT_CONTEXT", entry.second.original_index), &context);
        if (context > 0) entry.second.context = context;
    }
    return catalog;
}

bool llama_memory_error_text(const std::string &text) {
    const std::string lowered = to_lower(text);
    for (const auto *pattern : {
             "out of memory",
             "cuda error",
             "cuda out of memory",
             "failed to allocate",
             "alloc_buffer",
             "ggml_backend",
             "insufficient memory",
             "vram",
             "cublas",
             "cudamalloc",
         }) {
        if (lowered.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string resolve_llama_server_bin() {
    const std::string configured = trim(ezl_getenv_default("LLAMA_CPP_SERVER_BIN"));
    if (!configured.empty()) {
        return configured;
    }
    for (const std::string &candidate : {
             "build/llama.cpp/bin/llama-server",
             "build/llama.cpp/tools/server/llama-server",
             "build/llama.cpp/llama-server",
             "llama.cpp/bin/llama-server",
         }) {
        if (access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
    }
    return "llama-server";
}

LlamaCppInferenceBackend::LlamaCppInferenceBackend(std::string default_model, std::shared_ptr<ResourceManager> resources)
    : default_model_(std::move(default_model)),
      server_bin_(resolve_llama_server_bin()),
      api_key_(trim(ezl_getenv_default("LLAMA_CPP_API_KEY"))),
      resources_(resources ? std::move(resources) : std::make_shared<ResourceManager>()),
      lifecycle_([this] { return resources_->current_time(); }) {
    const std::string explicit_url = trim(ezl_getenv_default("LLAMA_CPP_SERVER_URL"));
    const std::string explicit_embedding_url = trim(ezl_getenv_default("LLAMA_CPP_EMBEDDING_SERVER_URL"));
    llm_catalog_ = captured_llm_catalog(default_model_);
    optimal_context_ = llm_optimal_context(ezl_getenv_fallback("LLM_MAX_TOKENS", "40000"));
    llm_replicas_ = llm_catalog_.replicas();
    const auto &models = llm_catalog_.available;
    const auto &model_ids = models;
    llm_server_count_ = model_ids.size();
    const int configured_gpu_layers = parse_llama_gpu_layers_env();
    if (!explicit_url.empty() && !model_ids.empty()) {
        Server server;
        server.model_id = model_ids.front();
        server.public_model = llm_catalog_.source(server.model_id);
        server.url = strip_trailing_slashes(explicit_url);
        server.output_tokens = llm_parameter_integer(ezl_getenv_fallback("LLM_MAX_OUTPUT_TOKENS", "8192"));
        server.started = true;
        server.config_index = 0;
        server.configured_gpu_layers = configured_gpu_layers;
        servers_.push_back(server);
    } else {
        const int base_port = parse_positive_int_env("LLAMA_CPP_SERVER_PORT", 8092);
        for (size_t i = 0; i < model_ids.size(); ++i) {
            Server server;
            server.model_id = model_ids[i];
            server.public_model = llm_catalog_.source(model_ids[i]);
            server.model_path = resolve_llama_model_path(llm_catalog_.source(model_ids[i]), llm_catalog_.configs.at(model_ids[i]).original_index);
            server.llm_acquire = server.model_path.empty();
            server.port = base_port + static_cast<int>(i);
            server.url = "http://127.0.0.1:" + std::to_string(server.port);
            server.config_index = llm_catalog_.configs.at(model_ids[i]).original_index;
            server.configured_gpu_layers = configured_gpu_layers;
            server.total_layers = llama_total_layers_for_path(server.model_path);
            servers_.push_back(server);
        }
    }

    for (auto &server : servers_) server.lifecycle_context = llm_catalog_.configs.at(server.model_id).context;
    swap_llm_residency_ = select_llm_swap_residency(llm_catalog_, resources_->gpu_capacity_gb(),
        ezl_getenv_fallback("LLM_MODEL_RESIDENCY", "auto"), ezl_getenv_fallback("LLM_MODEL_RESIDENCY_MARGIN_GB", "1.5"),
        [this](const std::string &model) {
            const auto &config = llm_catalog_.configs.at(model);
            auto server = std::find_if(servers_.begin(), servers_.end(), [&](const auto &candidate) { return candidate.model_id == model; });
            if (server == servers_.end()) throw std::runtime_error("No model file for external engine residency estimate");
            return estimate_configured_llm_locked(*server, config.context);
        });
    embedding_server_begin_ = servers_.size();
    const auto embedding = embedding_model_config();
    {
        Server embedding_server;
        embedding_server.model_id = embedding.model;
        embedding_server.public_model = embedding.alias;
        embedding_server.model_path = embedding.path;
        embedding_server.embedding_mode = true;
        embedding_server.embedding_acquire = embedding.acquire;
        embedding_server.config_index = model_ids.size();
        const auto embedding_layers = to_lower(strip_config_value(ezl_getenv_default("EMBEDDING_GPU_LAYERS")));
        embedding_server.configured_gpu_layers = kLlamaGpuLayersUnset;
        int parsed_layers = 0;
        if (embedding_layers != "auto" && embedding_layers != "-2" && parse_strict_int_string(embedding_layers, &parsed_layers)) embedding_server.configured_gpu_layers = parsed_layers;
        if (!explicit_embedding_url.empty()) {
            embedding_server.url = strip_trailing_slashes(explicit_embedding_url);
            embedding_server.started = true;
        } else {
            const int embedding_port = parse_positive_int_env(
                "LLAMA_CPP_EMBEDDING_SERVER_PORT",
                parse_positive_int_env("LLAMA_CPP_SERVER_PORT", 8092) + static_cast<int>(model_ids.size())
            );
            embedding_server.port = embedding_port;
            embedding_server.url = "http://127.0.0.1:" + std::to_string(embedding_server.port);
            embedding_server.total_layers = llama_total_layers_for_path(embedding_server.model_path);
        }
        embedding_template_ = embedding_server;
    }
    embedding_pool_ = std::make_unique<EmbeddingPool>(embedding_parallel_slots(),
        [this](size_t slot) { return load_embedding_instance(slot); },
        [this](EmbeddingPool::Handle handle) { destroy_embedding_instance(handle); }, embedding_keep_loaded,
        [this](const EmbeddingPool::Snapshot &pool) { register_embedding_resources(pool); });
    publish_observation_locked();
}

LlamaCppInferenceBackend::~LlamaCppInferenceBackend() {
    for (const auto &handle : streams_) handle.second();
    stop_owned_servers();
}

bool LlamaCppInferenceBackend::should_use_from_env(const ezl_env_config &config) {
    const std::string backend = to_lower(trim(ezl_getenv_default("EZLOCALAI_NATIVE_BACKEND")));
    if (backend == "echo") {
        return false;
    }
    if (backend == "llama" || backend == "llamacpp" || backend == "llama.cpp") {
        return true;
    }
    const std::string configured_model = trim(config.default_model);
    if (configured_model.empty() || to_lower(configured_model) == "none") {
        return to_lower(ezl_getenv_default("EMBEDDING_ENABLED")) != "false";
    }
    if (!trim(ezl_getenv_default("LLAMA_CPP_SERVER_URL")).empty()
        || !trim(ezl_getenv_default("LLAMA_CPP_EMBEDDING_SERVER_URL")).empty()
        || !trim(ezl_getenv_default("LLAMA_CPP_MODEL_PATH")).empty()
        || !trim(ezl_getenv_default("LLAMA_CPP_MODEL_PATHS")).empty()
        || !trim(ezl_getenv_default("LLAMA_CPP_EMBEDDING_MODEL_PATH")).empty()
        || !trim(ezl_getenv_default("LLAMA_CPP_EMBEDDING_MODEL")).empty()) {
        return true;
    }
    return true;
}

LlamaCppInferenceBackend::Server *LlamaCppInferenceBackend::select_server_locked(const std::string &request_body, bool embedding_request, const std::string &replica) {
    const auto resolved = replica.empty() ? llm_catalog_.resolve(extract_json_string(request_body, "model", "")) : std::optional<std::string>(replica);
    if (resolved) for (auto &server : servers_) if (server.embedding_mode == embedding_request && server.model_id == *resolved) return &server;
    for (auto &server : servers_) if (server.embedding_mode == embedding_request) return &server;
    return nullptr;
}

double LlamaCppInferenceBackend::estimate_configured_llm_locked(Server &server, int64_t context) {
    const auto &configured = llm_catalog_.configs.at(server.model_id);
    if (!context) context = configured.context;
    const auto cached = llm_vram_estimates_.find(server.model_id);
    if (cached != llm_vram_estimates_.end() && context <= configured.context) return std::max(0.0, cached->second);
    if (server.llm_acquire) {
        auto directory = strip_config_value(ezl_getenv_default("MODELS_PATH"));
        if (directory.empty()) directory = "./models";
        const auto files = download_gguf_model(llm_catalog_.source(server.model_id), directory, configured.quant);
        server.model_path = files.model; server.projector_path = files.projector;
        server.total_layers = llama_total_layers_for_path(files.model);
    }
    const auto profile = gpu_inference_profile(native_llm_batch_probes(), configured.main_gpu);
    const auto cache = configured_llm_kv_cache_type(profile.family, native_llm_cache_environment());
    const double estimate = std::max(0.0, estimate_llm_memory_gb(server.model_path, context,
        server.projector_path.empty() ? std::vector<std::string>{} : std::vector<std::string>{server.projector_path}, cache));
    if (context <= configured.context) llm_vram_estimates_[server.model_id] = estimate;
    return estimate;
}

void LlamaCppInferenceBackend::prepare_request_llm_locked(Server *server, int64_t context, std::unique_lock<std::mutex> &lock) {
    if (!server || server->embedding_mode) return;
    unload_other_llms_for_swap_locked(server);
    if (server->started) return;
    publish_observation_locked();
    const auto headroom = llm_load_headroom(llm_catalog_.configs.at(server->model_id).main_gpu, {
        [&] { return estimate_configured_llm_locked(*server, context); },
        [this] { return resources_->get_per_gpu_free_vram(); },
        [this] { return resources_->get_total_free_vram(); },
        [] { return std::string(ezl_getenv_fallback("LLM_MODEL_RESIDENCY_MARGIN_GB", "1.5")); }
    });
    server->llm_request_estimate_gb = headroom.estimate_gb;
    if (!(headroom.reclaim_gb > 0)) return;
    lock.unlock();
    try {
        if (llm_reclaimer_) llm_reclaimer_(headroom.reclaim_gb);
        else reclaim_llm_auxiliary_memory(*resources_, headroom.reclaim_gb, [this](ModelType type) {
            if (type == ModelType::Embedding) embedding_pool_->unload_idle();
        });
    } catch (...) { lock.lock(); throw; }
    lock.lock();
}

bool LlamaCppInferenceBackend::ensure_server_locked(Server *server, std::string *error_out, std::optional<int64_t> requested_context) {
    if (server && !server->embedding_mode) unload_other_llms_for_swap_locked(server);
    publish_observation_locked();
    if (!server || server->embedding_mode || server->llm_loaded || (server->started && !server->owns_process)) return load_server_locked(server, error_out, requested_context);
    const auto start = std::chrono::steady_clock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    bool loaded;
    try { loaded = load_server_locked(server, error_out, requested_context); }
    catch (...) { lifecycle_.record("load", "llm", server->model_id, elapsed(), false); throw; }
    lifecycle_.record("load", "llm", server->model_id, elapsed(), loaded);
    if (loaded) note_llm_loaded_locked(server);
    else server->llm_request_estimate_gb.reset();
    return loaded;
}

void LlamaCppInferenceBackend::unload_other_llms_for_swap_locked(Server *target) {
    if (!swap_llm_residency_) return;
    // The HTTP scheduler serializes all text models. Also protect direct
    // backend callers and streaming lifetimes that bypass that scheduler.
    for (const auto &server : servers_) {
        if (&server != target && !server.embedding_mode && server.active_requests > 0)
            throw std::runtime_error("Cannot switch LLMs while another configured model is in use: " + server.public_model);
    }
    std::vector<Server *> evicted;
    for (auto &server : servers_) if (&server != target && !server.embedding_mode && server.llm_loaded) evicted.push_back(&server);
    if (evicted.empty()) return;
    resources_->unregister_model(ModelType::Llm);
    resources_->unregister_model(ModelType::VisionLlm);
    const auto start = std::chrono::steady_clock::now();
    for (auto *server : evicted) {
        stop_owned_server_locked(server);
        server->llm_loaded = false;
        publish_llm_replica_locked(*server);
        if (active_llm_ == server->model_id) active_llm_.clear();
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (const auto *server : evicted) lifecycle_.record("unload", "llm", server->model_id, elapsed);
}

ModelType LlamaCppInferenceBackend::llm_resource_type(const Server &server) const {
    return !server.projector_path.empty()
        || model_has_local_vision_projector(llm_catalog_.source(server.model_id))
        || !trim(ezl_getenv_default("LLAMA_CPP_MMPROJ_PATH")).empty()
        ? ModelType::VisionLlm : ModelType::Llm;
}

void LlamaCppInferenceBackend::note_llm_loaded_locked(Server *server) {
    server->llm_loaded = true;
    publish_llm_replica_locked(*server);
    server->last_used = resources_->current_time();
    if (server->owns_process) {
        if (server->llm_vram_load_delta_gb) {
            auto &estimate = llm_vram_estimates_[server->model_id];
            estimate = std::max(estimate, *server->llm_vram_load_delta_gb);
        }
        const auto type = llm_resource_type(*server);
        const auto estimate = llm_vram_estimates_.find(server->model_id);
        // Pipes registers managed LLMs as cuda even after a CPU load retry.
        // Keep its estimate fallback when telemetry was unavailable.
        double registered = estimate == llm_vram_estimates_.end() ? model_vram_estimate(type) : estimate->second;
        if (server->llm_request_estimate_gb) registered = std::max(*server->llm_request_estimate_gb,
            server->llm_vram_load_delta_gb.value_or(0.0));
        resources_->register_model(type, server->model_id, "cuda", registered);
    }
    server->llm_request_estimate_gb.reset();
    for (size_t i = 0; i < servers_.size(); ++i) if (&servers_[i] == server) {
        if (std::find(llm_load_order_.begin(), llm_load_order_.end(), i) == llm_load_order_.end()) llm_load_order_.push_back(i);
        break;
    }
}

bool LlamaCppInferenceBackend::load_server_locked(Server *server, std::string *error_out, std::optional<int64_t> requested_context) {
    if (!server || server->embedding_mode || server->started) return load_server_once_locked(server, error_out, requested_context);
    const auto &configured = llm_catalog_.configs.at(server->model_id);
    const int64_t context = requested_context.value_or(configured_load_context(*server));
    LlmLoadConfiguration load;
    load.model = server->model_id;
    load.context = context;
    if (server->llm_acquire) {
        // Pipes first resolves files for placement estimation. A failure here
        // is tolerated; construction repeats acquisition and reports its own
        // error. Each later retry also repeats the reference file selection.
        try {
            auto directory = strip_config_value(ezl_getenv_default("MODELS_PATH"));
            if (directory.empty()) directory = "./models";
            const auto files = download_gguf_model(llm_catalog_.source(server->model_id), directory, configured.quant);
            server->model_path = files.model;
            server->projector_path = files.projector;
            server->total_layers = llama_total_layers_for_path(files.model);
        } catch (const std::exception &) {
            server->model_path.clear(); server->projector_path.clear(); server->total_layers = 41;
        }
    }
    load.total_layers = server->total_layers;
    if (server->retry_gpu_layers != kLlamaGpuLayersUnset) load.gpu_layers = server->retry_gpu_layers;
    else if (server->known_load_context == context) load.gpu_layers = server->known_gpu_layers;
    LlmLoadOperations operations;
    operations.configured_ubatch = [&] {
        const auto automatic = calculate_llm_auto_batch(configured.main_gpu, context, llm_catalog_.source(server->model_id));
        return configured_llm_ubatch(ezl_getenv_default("LLM_UBATCH_SIZE"), automatic.ubatch, context);
    };
    operations.cleanup = [&] { stop_owned_server_locked(server); };
    operations.load = [&](std::optional<int64_t> layers, std::optional<int64_t> ubatch) {
        std::string error;
        if (!load_server_once_locked(server, &error, context, layers, ubatch)) throw std::runtime_error(error);
    };
    operations.probe = [&](int64_t layers, int64_t ubatch) {
        int timeout;
        if (!parse_strict_int_string(ezl_getenv_fallback("LLM_LOAD_PROBE_TIMEOUT", "300"), &timeout)) timeout = 300;
        const HttpDeadline deadline(std::max(30, timeout), false);
        std::string error;
        bool fits;
        try { fits = load_server_once_locked(server, &error, context, layers, ubatch, &deadline); }
        catch (...) { stop_owned_server_locked(server); throw; }
        // Successful fit probes are discarded too; the winner is loaded afresh.
        stop_owned_server_locked(server);
        if (error == "Failed to fork llama.cpp server") throw std::runtime_error(error);
        return fits;
    };
    try {
        load_llm_resilient(load, operations);
        server->lifecycle_context = context;
        server->known_load_context = context;
        server->known_gpu_layers = server->current_gpu_layers;
        server->retry_gpu_layers = kLlamaGpuLayersUnset;
        server->cpu_only_retry = false;
        return true;
    } catch (const std::exception &error) {
        stop_owned_server_locked(server);
        if (error_out) *error_out = error.what();
        return false;
    }
}

bool LlamaCppInferenceBackend::load_server_once_locked(Server *server, std::string *error_out, std::optional<int64_t> requested_context,
    std::optional<int64_t> gpu_layers, std::optional<int64_t> ubatch, const HttpDeadline *probe_deadline) {
    if (!server) {
        if (error_out) {
            *error_out = "No llama.cpp server is configured";
        }
        return false;
    }
    if (server->started) {
        return true;
    }

    const auto effective_context = server->embedding_mode ? std::string() : llm_effective_context(
        std::to_string(requested_context.value_or(configured_load_context(*server))), ezl_getenv_default("LLM_MAX_TOKENS"));
    if (!server->embedding_mode) server->output_tokens = llm_parameter_integer(ezl_getenv_fallback("LLM_MAX_OUTPUT_TOKENS", "8192"));

    if (!server->embedding_mode && server->llm_acquire) {
        auto directory = strip_config_value(ezl_getenv_default("MODELS_PATH"));
        if (directory.empty()) directory = "./models";
        const auto files = download_gguf_model(llm_catalog_.source(server->model_id), directory,
            llm_catalog_.configs.at(server->model_id).quant);
        server->model_path = files.model;
        server->projector_path = files.projector;
    }

    if (server->embedding_mode && server->embedding_acquire) {
        try {
            server->model_path = download_gguf_model(server->model_id, "./models",
                std::string(ezl_getenv_fallback("EMBEDDING_QUANT_TYPE", "Q8_0")), true).model;
            server->model_path = std::filesystem::absolute(server->model_path).lexically_normal().string();
        } catch (const std::exception &error) {
            if (error_out) *error_out = error.what();
            return false;
        }
    }

    if (server->model_path.empty()) {
        const std::string hf_model = llm_catalog_.source(server->model_id);
        const std::string lowered_model = to_lower(hf_model);
        const bool missing_local_gguf =
            lowered_model.size() >= 5
            && lowered_model.compare(lowered_model.size() - 5, 5, ".gguf") == 0;
        if (hf_model.empty() || missing_local_gguf) {
            if (error_out) {
                *error_out = "No GGUF model path found for '" + server->public_model
                    + "'. Set LLAMA_CPP_MODEL_PATH, LLAMA_CPP_MODEL_PATHS, or MODELS_PATH.";
            }
            return false;
        }
    }

    const int context_size = server->embedding_mode
        ? embedding_startup_context_size(server->config_index, static_cast<int>(requested_context.value_or(0)))
        : llm_context_parameter(effective_context);
    const std::string context = !server->embedding_mode || context_size > 0 ? std::to_string(context_size) : "";

    const int embedding_batch = std::min(embedding_int_setting("EMBEDDING_BATCH_SIZE", 512), context_size);
    const int embedding_ubatch = std::min(embedding_int_setting("EMBEDDING_UBATCH_SIZE", 512), embedding_batch);
    const auto embedding_cache = to_lower(strip_config_value(ezl_getenv_fallback("EMBEDDING_KV_CACHE_TYPE", "f16")));
    std::string embedding_main_gpu = ezl_getenv_fallback("EMBEDDING_MAIN_GPU", "");
    if (embedding_main_gpu.empty()) embedding_main_gpu = ezl_getenv_fallback("MAIN_GPU", "0");
    int embedding_gpu_index = 0;
    if (!parse_strict_int_string(embedding_main_gpu, &embedding_gpu_index)) embedding_gpu_index = 0;
    if (server->embedding_main_gpu) embedding_gpu_index = *server->embedding_main_gpu;
    std::optional<EmbeddingMemory> embedding_memory;
    std::vector<int> embedding_attempts;
    bool embedding_cuda = false;
    if (server->embedding_mode) {
        try {
            embedding_cuda = visible_cuda_device_count() > 0;
            std::optional<int> requested;
            if (server->configured_gpu_layers != kLlamaGpuLayersUnset) requested = server->configured_gpu_layers;
            int selected = embedding_cuda ? requested.value_or(-1) : 0;
            if (!server->model_path.empty()) {
                embedding_memory = estimate_embedding_memory(server->model_path, context_size, 1, embedding_ubatch, embedding_cache);
                double safety = .75;
                if (!parse_strict_python_float_string(ezl_getenv_fallback("EMBEDDING_VRAM_SAFETY_MARGIN", "0.75"), &safety)) safety = .75;
                selected = resolve_embedding_gpu_layers(*embedding_memory, requested, false, embedding_cuda,
                    requested || !embedding_cuda ? 0 : cuda_free_memory_gb(embedding_gpu_index), safety);
            }
            embedding_attempts = embedding_fallback_layers(selected, embedding_memory ? embedding_memory->layers : 28);
        } catch (const std::exception &error) {
            if (error_out) *error_out = error.what();
            return false;
        }
    }

    const int max_startup_attempts = server->embedding_mode ? static_cast<int>(embedding_attempts.size()) : 1;
    for (int startup_attempt = 0; startup_attempt < max_startup_attempts; ++startup_attempt) {
        const auto speculative_environment = native_llm_cache_environment();
        std::string speculative_type = "none";
        std::vector<std::string> args;
        args.push_back(server_bin_);
        if (!server->model_path.empty()) {
            args.push_back("-m");
            args.push_back(server->model_path);
        } else {
            args.push_back("-hf");
            auto model = server->embedding_mode ? embedding_model_config().hf_argument : llm_catalog_.source(server->model_id);
            if (!server->embedding_mode) {
                const auto &quant = llm_catalog_.configs.at(server->model_id).quant;
                if (quant && model.find(':') == std::string::npos) model += ":" + *quant;
            }
            args.push_back(model);
        }
        args.push_back("--host");
        args.push_back("127.0.0.1");
        args.push_back("--port");
        args.push_back(std::to_string(server->port));
        if (!server->public_model.empty()) {
            args.push_back("--alias");
            args.push_back(server->public_model);
        }
        if (server->embedding_mode) {
            args.push_back("--embedding");
            args.push_back("-np");
            args.push_back("1");
            auto pooling = trim(ezl_getenv_default("LLAMA_CPP_EMBEDDING_POOLING"));
            args.push_back("--pooling");
            args.push_back(pooling.empty() ? "last" : pooling);
            const std::string ubatch = trim(ezl_getenv_default("LLAMA_CPP_EMBEDDING_UBATCH"));
            args.push_back("-b");
            args.push_back(std::to_string(embedding_ubatch));
            args.push_back("-ub");
            args.push_back(ubatch.empty() ? std::to_string(embedding_ubatch) : ubatch);
            args.insert(args.end(), {"--cache-ram", "0", "--no-cache-prompt", "--fit", "off", "--embd-normalize", std::to_string(embedding_int_setting("EMBEDDING_NORMALIZE", 2))});
            const auto &cache = embedding_cache;
            if (cache == "f16" || cache == "f32" || cache == "q8_0" || cache == "q4_0") {
                args.push_back("-ctk");
                args.push_back(cache);
                args.push_back("-ctv");
                args.push_back(cache);
            }
        }
        if (!context.empty()) {
            args.push_back("-c");
            args.push_back(context);
        }
        const std::string configured_main_gpu = server->embedding_mode
            ? std::to_string(embedding_gpu_index)
            : std::to_string(llm_catalog_.configs.at(server->model_id).main_gpu);
        const std::string main_gpu = configured_main_gpu.empty() ? "0" : configured_main_gpu;
        int parsed_main_gpu = 0;
        const bool valid_main_gpu = parse_strict_int_string(main_gpu, &parsed_main_gpu) && (server->embedding_mode || parsed_main_gpu >= 0);
        std::string tensor_split = trim(ezl_getenv_default("TENSOR_SPLIT"));
        std::vector<double> source_tensor_split;
        if (!server->embedding_mode && tensor_split.empty()) {
            const auto &configured_split = llm_catalog_.configs.at(server->model_id).tensor_split;
            if (configured_split) source_tensor_split.assign(configured_split->begin(), configured_split->end());
        }
        const bool pass_main_gpu = !configured_main_gpu.empty() || !tensor_split.empty();
        if (pass_main_gpu && valid_main_gpu) {
            args.push_back("--main-gpu");
            args.push_back(std::to_string(parsed_main_gpu));
        }
        if (!server->embedding_mode && !tensor_split.empty()) {
            args.push_back("--tensor-split");
            args.push_back(tensor_split);
        }
        if (!server->embedding_mode) {
            const auto automatic = calculate_llm_auto_batch(llm_catalog_.configs.at(server->model_id).main_gpu,
                context_size, llm_catalog_.source(server->model_id));
            const auto batch = resolve_llm_batch_parameters(automatic, ezl_getenv_default("LLM_BATCH_SIZE"),
                ezl_getenv_default("LLM_UBATCH_SIZE"), std::nullopt, ubatch);
            args.insert(args.end(), {"-b", batch.batch, "-ub", batch.ubatch});
        }
        PromptCache requested_cache{"0", ""}, effective_cache{"0", ""};
        int64_t prompt_cache_swap_mib = 0;
        if (!server->embedding_mode) {
            const auto budget = native_llm_prompt_cache_budget(std::string(ezl_getenv_default("LLM_PROMPT_CACHE_MIB")),
                context_size, native_llm_cache_environment(), native_prompt_cache_ram);
            requested_cache = budget.requested;
            effective_cache = budget.effective;
            prompt_cache_swap_mib = budget.swap_allowance_mib;
            validate_llm_reasoning_budget(ezl_getenv_fallback("REASONING_BUDGET", "-1"));
        }
        auto mmproj = trim(ezl_getenv_default("LLAMA_CPP_MMPROJ_PATH"));
        if (mmproj.empty() && !server->embedding_mode) mmproj = server->projector_path;
        if (!server->embedding_mode && !mmproj.empty()) {
            // Report the projector actually passed to this owned engine,
            // including explicit paths outside the model download directory.
            server->projector_path = mmproj;
            args.push_back("--mmproj");
            args.push_back(mmproj);
        }
        const auto extra_args = split_shell_words_simple(ezl_getenv_default("LLAMA_CPP_EXTRA_ARGS"));
        auto extra_has_any = [&](std::initializer_list<const char *> names) {
            for (const auto &extra : extra_args) {
                for (const auto *name : names) {
                    if (extra == name) {
                        return true;
                    }
                }
            }
            return false;
        };
        if (!server->embedding_mode
            && llama_env_truthy_default("LLAMA_CPP_FLASH_ATTENTION", true)
            && !extra_has_any({"-fa", "--flash-attn", "--flash-attn-auto"})) {
            args.push_back("-fa");
            args.push_back("on");
        }
        std::string kv_cache_type;
        if (!server->embedding_mode) {
            const auto profile = gpu_inference_profile(native_llm_batch_probes(), llm_catalog_.configs.at(server->model_id).main_gpu);
            const auto environment = native_llm_cache_environment();
            kv_cache_type = configured_llm_kv_cache_type(profile.family, environment);
            if (!extra_has_any({"--cache-ram", "-cram"})) args.insert(args.end(), {"--cache-ram", effective_cache.mib});
            if (!extra_has_any({"-ctk", "--cache-type-k"})) {
                args.push_back("-ctk");
                args.push_back(kv_cache_type);
            }
            if (!extra_has_any({"-ctv", "--cache-type-v"})) {
                args.push_back("-ctv");
                args.push_back(kv_cache_type);
            }
            speculative_type = llm_speculative_backend(llm_catalog_.source(server->model_id), speculative_environment("LLM_SPECULATIVE_TYPE", "auto"));
            const auto parallel = llm_parallel_parameter(llm_catalog_.configs.at(server->model_id).parallel, context_size, speculative_type != "none");
            args.insert(args.end(), {"-np", std::to_string(parallel)});
            if (!extra_has_any({"--cont-batching", "-cb", "--no-cont-batching", "-nocb"})) args.push_back("--cont-batching");
            if (!extra_has_any({"--warmup", "--no-warmup"})) args.push_back("--warmup");
            if (!extra_has_any({"--ctx-checkpoints", "-ctxcp", "--swa-checkpoints"}))
                args.insert(args.end(), {"--ctx-checkpoints", std::to_string(llm_default_context_checkpoints)});
            if (!extra_has_any({"--checkpoint-min-step", "-cms"}))
                args.insert(args.end(), {"--checkpoint-min-step", std::to_string(llm_default_checkpoint_min_step)});
        }
        LlmDraftSettings draft_settings;
        std::string draft_path;
        if (speculative_type == "mtp") {
            const auto length = llm_mtp_draft_length(llm_catalog_.configs.at(server->model_id).main_gpu,
                llm_catalog_.source(server->model_id), speculative_environment);
            draft_settings = {length.first, llm_mtp_draft_probability(llm_catalog_.source(server->model_id), speculative_environment), length.second};
        } else if (speculative_type == "dflash2") {
            draft_settings = llm_dflash_settings(llm_catalog_.configs.at(server->model_id).main_gpu, speculative_environment);
            draft_path = download_dflash_model(speculative_environment);
            if (!extra_has_any({"--fit", "-fit"})) args.insert(args.end(), {"--fit", "off"});
            if (!extra_has_any({"--spec-draft-model", "-md", "--model-draft"})) args.insert(args.end(), {"--spec-draft-model", draft_path});
            if (visible_cuda_device_count() > 0 && !extra_has_any({"--spec-draft-device", "-devd", "--device-draft"}))
                args.insert(args.end(), {"--spec-draft-device", llama_gpu_device_name(main_gpu)});
        }
        if (!server->embedding_mode && !extra_has_any({"--spec-type"}))
            args.insert(args.end(), {"--spec-type", speculative_type == "mtp" ? "draft-mtp" : speculative_type == "dflash2" ? "draft-dflash" : "none"});
        if (speculative_type != "none") {
            if (!extra_has_any({"--spec-draft-n-max"})) args.insert(args.end(), {"--spec-draft-n-max", std::to_string(draft_settings.n_max)});
            // The engine stores this as float. Preserve NaN and signed zero,
            // which the reference MTP helper deliberately passes through.
            if (!extra_has_any({"--spec-draft-p-min", "--draft-p-min"})) {
                std::ostringstream probability;
                const float native_probability = static_cast<float>(draft_settings.p_min);
                // std::stof rejects inexact decimal underflow, unlike the
                // reference's direct float assignment. Exact hexadecimal
                // notation also represents subnormal float values faithfully.
                if (std::isfinite(native_probability) && native_probability != 0 && std::fpclassify(native_probability) == FP_SUBNORMAL)
                    probability << std::hexfloat << native_probability;
                else probability << std::setprecision(std::numeric_limits<double>::max_digits10) << (native_probability == 0 ? static_cast<double>(native_probability) : draft_settings.p_min);
                args.insert(args.end(), {"--spec-draft-p-min", probability.str()});
            }
        }
        if (!server->embedding_mode) {
            const auto placement = llm_tensor_parameters(source_tensor_split, visible_cuda_device_count(), speculative_type != "dflash2");
            if (placement.split_none && !extra_has_any({"--split-mode", "-sm"})) args.insert(args.end(), {"--split-mode", "none"});
            if (!placement.fit && speculative_type != "dflash2" && !extra_has_any({"--fit", "-fit"})) args.insert(args.end(), {"--fit", "off"});
            const auto image_tokens = llm_image_min_tokens(llm_catalog_.source(server->model_id));
            if (!mmproj.empty() && image_tokens > 0 && !extra_has_any({"--image-min-tokens"}))
                args.insert(args.end(), {"--image-min-tokens", std::to_string(image_tokens)});
        }
        for (const auto &extra : extra_args) args.push_back(extra);
        if (server->embedding_mode) {
            for (const auto &extra : split_shell_words_simple(ezl_getenv_default("LLAMA_CPP_EMBEDDING_EXTRA_ARGS"))) {
                args.push_back(extra);
            }
        }
        int selected_gpu_layers = server->retry_gpu_layers != kLlamaGpuLayersUnset
            ? server->retry_gpu_layers
            : server->configured_gpu_layers;
        if (server->cpu_only_retry && !server->embedding_mode && !gpu_layers) {
            selected_gpu_layers = 0;
        }
        if (gpu_layers && !server->embedding_mode) selected_gpu_layers = static_cast<int>(*gpu_layers);
        if (selected_gpu_layers == kLlamaGpuLayersUnset && !server->embedding_mode) {
            selected_gpu_layers = llama_auto_gpu_layers();
        }
        if (selected_gpu_layers != kLlamaGpuLayersUnset && !server->embedding_mode) {
            args.push_back("-ngl");
            args.push_back(std::to_string(selected_gpu_layers));
        }
        if (speculative_type == "dflash2" && !extra_has_any({"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}))
            args.insert(args.end(), {"--spec-draft-ngl", std::to_string(selected_gpu_layers)});
        if (server->embedding_mode) {
            selected_gpu_layers = embedding_attempts.at(startup_attempt);
            args.push_back("-ngl");
            args.push_back(std::to_string(selected_gpu_layers));
            args.push_back("-fa");
            args.push_back(selected_gpu_layers == 0 ? "off" : "on");
            if (selected_gpu_layers == 0) args.insert(args.end(), {"--no-kv-offload", "--no-op-offload"});
        }

        std::optional<double> gpu_free_before_load;
        const int measured_gpu = valid_main_gpu ? parsed_main_gpu : 0;
        if (!server->embedding_mode) {
            server->llm_vram_load_delta_gb.reset();
            try {
                const auto free = resources_->get_per_gpu_free_vram();
                if (measured_gpu >= 0 && static_cast<size_t>(measured_gpu) < free.size()) gpu_free_before_load = free[measured_gpu];
            } catch (...) { /* Telemetry must not prevent model loading. */ }
        }
        const std::string startup_log_path = temporary_llama_startup_log_path();
        const pid_t pid = fork();
        if (pid < 0) {
            if (!startup_log_path.empty()) {
                unlink(startup_log_path.c_str());
            }
            if (error_out) {
                *error_out = "Failed to fork llama.cpp server";
            }
            return false;
        }
        if (pid == 0) {
            if (!startup_log_path.empty()) {
                const int log_fd = open(startup_log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
                if (log_fd >= 0) {
                    dup2(log_fd, STDOUT_FILENO);
                    dup2(log_fd, STDERR_FILENO);
                    if (log_fd > STDERR_FILENO) {
                        close(log_fd);
                    }
                }
            }
            std::vector<char *> argv;
            argv.reserve(args.size() + 1);
            for (auto &arg : args) {
                argv.push_back(arg.data());
            }
            argv.push_back(nullptr);
            execvp(server_bin_.c_str(), argv.data());
            _exit(127);
        }

        server->pid = static_cast<int>(pid);
        server->owns_process = true;
        const int timeout_seconds = parse_positive_int_env("LLAMA_CPP_START_TIMEOUT", 120);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
        bool retry_startup = false;
        while (probe_deadline ? !probe_deadline->expired() : std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            const pid_t done = waitpid(pid, &status, WNOHANG);
            if (done == pid) {
                server->pid = -1;
                server->owns_process = false;
                const std::string startup_output = startup_log_path.empty()
                    ? ""
                    : read_file_limited_or_empty(startup_log_path, 1024 * 1024);
                if (!startup_log_path.empty()) {
                    unlink(startup_log_path.c_str());
                }
                if (server->embedding_mode && selected_gpu_layers != 0 && embedding_memory_error(startup_output)
                    && startup_attempt + 1 < max_startup_attempts) {
                    retry_startup = true;
                    break;
                }
                if (error_out) {
                    *error_out = "llama.cpp server exited before becoming ready";
                    if (!startup_output.empty()) *error_out += ": " + startup_output;
                }
                return false;
            }
            const auto ready = http_json_request("GET", server->url, "/v1/models", api_key_, "", 0, true, probe_deadline);
            if (ready.status >= 200 && ready.status < 300) {
                if (!server->embedding_mode) {
                    std::optional<std::string> failure;
                    const auto source = llm_catalog_.source(server->model_id);
                    try {
                        const auto probe = http_json_request("POST", server->url, "/v1/completions", api_key_, llm_startup_request, 0, false, probe_deadline);
                        if (probe.transport_error) failure = llm_startup_failure(source, "", probe.body.empty() ? "llama.cpp startup probe transport failed" : probe.body);
                        else if (json_decode_error(probe.body)) failure = llm_startup_failure(source, "", "Invalid JSON response from llama.cpp startup probe");
                        else failure = llm_startup_failure(source, probe.body);
                    } catch (const std::exception &error) { failure = llm_startup_failure(source, "", error.what()); }
                    if (failure) {
                        stop_owned_server_locked(server);
                        if (!startup_log_path.empty()) unlink(startup_log_path.c_str());
                        if (error_out) *error_out = *failure;
                        return false;
                    }
                }
                if (!startup_log_path.empty()) {
                    unlink(startup_log_path.c_str());
                }
                server->started = true;
                int parsed_context = 0;
                server->context_size = parse_strict_int_string(context, &parsed_context) && parsed_context > 0 ? parsed_context : 0;
                server->current_gpu_layers = selected_gpu_layers;
                if (!server->embedding_mode) {
                    if (gpu_free_before_load) try {
                        const auto free = resources_->get_per_gpu_free_vram();
                        if (static_cast<size_t>(measured_gpu) < free.size())
                            server->llm_vram_load_delta_gb = std::max(0.0, *gpu_free_before_load - free[measured_gpu]);
                    } catch (...) { /* Match the reference's optional allocation report. */ }
                    std::vector<std::pair<std::string, std::string>> runtime;
                    runtime.emplace_back("gpu_layers", std::to_string(selected_gpu_layers));
                    runtime.emplace_back("main_gpu", std::to_string(valid_main_gpu ? parsed_main_gpu : 0));
                    if (!tensor_split.empty()) {
                        std::vector<std::string> ratios;
                        for (const auto &part : split_csv_values(tensor_split)) { double ratio; if (parse_strict_python_float_string(part, &ratio)) ratios.push_back(json_number_literal(ratio)); }
                        if (!ratios.empty()) runtime.emplace_back("tensor_split", json_array_from_values(ratios));
                    } else if (!source_tensor_split.empty()) {
                        std::vector<std::string> ratios;
                        for (double ratio : source_tensor_split) ratios.push_back(json_number_literal(ratio));
                        runtime.emplace_back("tensor_split", json_array_from_values(ratios));
                    }
                    runtime.emplace_back("prompt_cache_mib", effective_cache.mib);
                    runtime.emplace_back("prompt_cache_requested_mib", requested_cache.mib);
                    runtime.emplace_back("prompt_cache_swap_allowance_mib", std::to_string(prompt_cache_swap_mib));
                    runtime.emplace_back("checkpoint_min_step", std::to_string(llm_default_checkpoint_min_step));
                    runtime.emplace_back("n_ctx_checkpoints", std::to_string(llm_default_context_checkpoints));
                    server->llm_parallel = 1;
                    for (size_t i = 0; i + 1 < args.size(); ++i) {
                        const std::string field = args[i] == "-b" || args[i] == "--batch-size" ? "n_batch"
                            : args[i] == "-ub" || args[i] == "--ubatch-size" ? "n_ubatch"
                            : args[i] == "--cache-ram" || args[i] == "-cram" ? "prompt_cache_mib"
                            : args[i] == "--checkpoint-min-step" || args[i] == "-cms" ? "checkpoint_min_step"
                            : args[i] == "--ctx-checkpoints" || args[i] == "-ctxcp" || args[i] == "--swa-checkpoints" ? "n_ctx_checkpoints"
                            : args[i] == "-np" || args[i] == "--parallel" ? "n_parallel" : "";
                        int number;
                        if (!field.empty() && parse_strict_int_string(args[i+1], &number)) {
                            if (field == "n_parallel") server->llm_parallel = number;
                            else ordered_json_upsert(&runtime, {field, std::to_string(number)});
                        }
                    }
                    runtime.emplace_back("kv_cache_type", '"' + json_escape(kv_cache_type) + '"');
                    runtime.emplace_back("speculative_type", '"' + speculative_type + '"');
                    server->runtime_json = json_object_from_ordered_entries(runtime);
                }
                if (server->embedding_mode) {
                    server->embedding_device = selected_gpu_layers != 0 && embedding_cuda ? "cuda" : "cpu";
                    server->embedding_vram_gb = embedding_memory ? embedding_gpu_memory(*embedding_memory, selected_gpu_layers) : 0;
                }
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (!startup_log_path.empty()) {
            unlink(startup_log_path.c_str());
        }
        if (retry_startup) {
            continue;
        }
        stop_owned_server_locked(server);
        if (error_out) {
            *error_out = "Timed out waiting for llama.cpp server at " + server->url;
        }
        return false;
    }
    if (error_out) {
        *error_out = "llama.cpp server exited before becoming ready";
    }
    return false;
}

int64_t LlamaCppInferenceBackend::configured_load_context(const Server &server) const {
    // Explicit native overrides remain live for restoration. Source defaults
    // come from the captured per-model catalog, with no adaptive 256k floor.
    const auto override = llama_configured_context_size(server.config_index, false);
    if (override > 0) return override;
    int legacy_override = 0;
    if (parse_strict_int_string(csv_env_value_for_index("LLAMA_CPP_DEFAULT_CONTEXT", server.config_index), &legacy_override) && legacy_override > 0) return legacy_override;
    return llm_catalog_.configs.at(server.model_id).context;
}

InferenceResult LlamaCppInferenceBackend::forward(const std::string &path, const std::string &request_body, const std::string &replica,
    bool model_profile, std::optional<int64_t> context) {
    const bool embedding_request = path == "/v1/embeddings";
    const bool lazy_chat = path == "/v1/chat/completions"
        && !llm_output_falsey(json_object_field(request_body, "stream", "false"));
    std::string engine_body = request_body;
    const bool inference_retry = !embedding_request
        && (path == "/v1/chat/completions" || path == "/v1/completions");
    // Three inner attempts plus Pipes' outer memory-reduction attempt.
    constexpr int max_retries = 4;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        Server *server = nullptr;
        std::string url;
        std::string source_model;
        std::string speculative_type;
        bool single_slot = true;
        {
            std::unique_lock<std::mutex> transition(llm_transition_mutex_, std::defer_lock);
            if (!embedding_request) transition.lock();
            std::unique_lock<std::mutex> lock(mutex_);
            server = select_server_locked(request_body, embedding_request, replica);
            if (server && !embedding_request) {
                source_model = llm_catalog_.source(replica.empty() ? server->model_id : replica);
            }
            std::string error;
            std::optional<int64_t> request_context;
            if (server && !embedding_request) {
                request_context = context ? context : std::optional<int64_t>(std::stoll(llm_configured_context_limit(
                    std::to_string(llm_catalog_.configs.at(server->model_id).context), optimal_context_)));
                if (server->started && server->owns_process && server->context_size < *request_context && server->active_requests == 0)
                    stop_owned_server_locked(server);
            }
            // Pipes loads the configured ceiling before inference. Its
            // streaming estimate is clamped to that same ceiling, and cannot
            // enlarge the allocation. Tokenizer errors retain that ceiling.
            if (!embedding_request && server) prepare_request_llm_locked(server, request_context.value_or(configured_load_context(*server)), lock);
            if (!ensure_server_locked(server, &error, request_context)) {
                return make_error(503, error);
            }
            if (!embedding_request) {
                try {
                    engine_body = llm_engine_request(model_profile ? llm_model_request_settings(request_body, source_model) : request_body,
                        path == "/v1/chat/completions", server->output_tokens);
                    if (lazy_chat) {
                        // LLM.chat preserves the raw value; _chat_stream's
                        // inference thread then forces true in its own copy.
                        auto fields = json_object_ordered_entries(engine_body);
                        ordered_json_upsert(&fields, {"stream", "true"});
                        engine_body = json_object_from_ordered_entries(fields);
                        speculative_type = extract_json_string(server->runtime_json, "speculative_type",
                            llm_catalog_.configs.at(server->model_id).speculative_type);
                        single_slot = swap_llm_residency_ || server->llm_parallel.value_or(
                            llm_parallel_parameter(llm_catalog_.configs.at(server->model_id).parallel,
                                server->context_size, speculative_type != "none")) == 1;
                    }
                    if (path == "/v1/completions") llm_completion_callback_guard(engine_body);
                } catch (const LlmParameterError &failure) {
                    return InferenceResult(make_error(500, failure.message), {failure.category, failure.message});
                }
            }
            server->active_requests++;
            if (!embedding_request && server->owns_process) resources_->mark_model_in_use(llm_resource_type(*server), true);
            server->last_used = resources_->current_time();
            if (!embedding_request) active_llm_ = server->model_id;
            url = server->url;
        }

        if (lazy_chat) {
            auto stream = std::make_shared<LlmInferenceStream>(source_model, speculative_type, xllamacpp_reference_version,
                single_slot, [this, server, url, path, engine_body](const auto &emit, const auto &interrupt) {
                    bool delivered = false, failed = false;
                    produce_local_chat_stream(url, path, api_key_, engine_body, [&](const std::string &value) {
                        failed |= llm_response_error(value, true).has_value();
                        const bool stop = emit(value); delivered |= !stop; return stop;
                    }, interrupt);
                    if (delivered && !failed) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (!server->llm_loaded) note_llm_loaded_locked(server);
                    }
                }, [this, server] {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (server->active_requests > 0) --server->active_requests;
                    if (server->owns_process) resources_->mark_model_in_use(llm_resource_type(*server), false);
                    server->last_used = resources_->current_time();
                });
            { std::lock_guard<std::mutex> lock(mutex_);
              streams_.erase(std::remove_if(streams_.begin(), streams_.end(), [](const auto &handle) { return handle.first.expired(); }), streams_.end());
              streams_.push_back(stream->owner_handle()); }
            InferenceResult result(""); result.from_engine = true; result.stream = std::move(stream); return result;
        }
        const auto response = http_json_request("POST", url, path, api_key_, engine_body, 0, false);
        InferenceResult outcome(response.body);
        outcome.from_engine = true;
        if (inference_retry) {
            if (!json_decode_error(response.body)) {
                outcome = llm_buffered_response(response.body, source_model, path == "/v1/chat/completions",
                    !llm_output_falsey(json_object_field(request_body, "stream", "false")));
            } else outcome.failure = llm_response_error(response.body, lazy_chat);
            outcome.lazy_chat_stream = lazy_chat && outcome.failure.has_value();
        }
        bool retry_request = false;
        {
            std::unique_lock<std::mutex> transition(llm_transition_mutex_, std::defer_lock);
            // Ordinary inference completion must release its active lease
            // even while another model is reclaiming memory. Only a possible
            // reload needs the transition gate before taking mutex_.
            if (inference_retry && outcome.failure && !outcome.lazy_chat_stream
                && attempt + 1 < max_retries && llama_memory_error_text(outcome.failure->message)) transition.lock();
            std::lock_guard<std::mutex> lock(mutex_);
            if (server->active_requests > 0) {
                server->active_requests--;
            }
            if (!embedding_request && server->owns_process) resources_->mark_model_in_use(llm_resource_type(*server), false);
            if (((response.status >= 200 && response.status < 300) || (lazy_chat && !response.transport_error)) && !outcome.failure) {
                const std::string response_model = extract_json_string(outcome.body, "model", "");
                if (!embedding_request && !response_model.empty()) {
                    server->public_model = response_model;
                }
                if (path == "/v1/embeddings") {
                    server->embedding_loaded = true;
                } else if (!server->llm_loaded) {
                    // An external URL is not a verified local engine load.
                    // Observe its residency only after a successful response,
                    // without inventing a local lifecycle load event.
                    note_llm_loaded_locked(server);
                }
            } else if (inference_retry
                && outcome.failure && !outcome.lazy_chat_stream
                && attempt + 1 < max_retries
                && server->owns_process
                && server->active_requests == 0
                && llm_inference_reload_allowed(llm_catalog_.available.size(),
                    std::max<int64_t>(1, server->llm_parallel.value_or(
                        llm_parallel_parameter(llm_catalog_.configs.at(server->model_id).parallel,
                            server->context_size, llm_catalog_.configs.at(server->model_id).speculative_type != "none"))))
                && llama_memory_error_text(outcome.failure->message)) {
                const int next_gpu_layers = llama_next_reduced_gpu_layers(
                    server->current_gpu_layers,
                    server->total_layers
                );
                if (next_gpu_layers != kLlamaGpuLayersUnset) {
                    std::string error;
                    const int retry_context = server->context_size > 0 ? server->context_size : 0;
                    server->retry_gpu_layers = next_gpu_layers;
                    server->cpu_only_retry = next_gpu_layers == 0;
                    stop_owned_server_locked(server);
                    if (!ensure_server_locked(server, &error, retry_context)) {
                        return InferenceResult(make_error(503, error), {"RuntimeError", error});
                    }
                    retry_request = true;
                }
            }
            server->last_used = monotonic_epoch_seconds();
        }
        if (retry_request) {
            continue;
        }
        // The pinned binding uses the same dict callback for both successful
        // and error handler responses. Stream semantics come from the value,
        // including an empty-stream result when no assistant text is present.
        if ((response.status < 200 || response.status >= 300) && (!lazy_chat || response.transport_error)) {
            const std::string body = trim(response.body);
            if (is_valid_top_level_json_object(body)
                && (!trim(extract_json_value(body, "error", "")).empty()
                    || extract_json_int_value(body, "code", 0) != 0)) {
                return outcome;
            }
            const auto message = response.body.empty() ? "llama.cpp request failed" : response.body;
            outcome.body = make_error(response.status, message);
            if (!outcome.failure) outcome.failure = LlmResponseError{"RuntimeError", message};
            outcome.lazy_chat_stream = lazy_chat;
        }
        return outcome;
    }
    return InferenceResult(make_error(500, "llama.cpp context retry failed"), {"RuntimeError", "llama.cpp context retry failed"});
}

std::shared_ptr<const LlamaCppInferenceBackend::Observation> LlamaCppInferenceBackend::publish_observation_locked() const {
    auto snapshot = std::make_shared<const Observation>(Observation{
        servers_, llm_catalog_, llm_load_order_, active_llm_, llm_vram_estimates_, swap_llm_residency_});
    std::atomic_store(&observation_, snapshot);
    return snapshot;
}

std::shared_ptr<const LlamaCppInferenceBackend::Observation> LlamaCppInferenceBackend::observe() const {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (lock.owns_lock()) return publish_observation_locked();
    return std::atomic_load(&observation_);
}

std::string LlamaCppInferenceBackend::loaded_models_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaded_models_from_observation(publish_observation_locked());
}

std::string LlamaCppInferenceBackend::observed_loaded_models_json() const {
    return loaded_models_from_observation(observe());
}

std::string LlamaCppInferenceBackend::loaded_models_from_observation(const std::shared_ptr<const Observation> &state_snapshot) const {
    const Server *llm = nullptr;
    const Server *vision_llm = nullptr;
    for (const auto &server : state_snapshot->servers) {
        if (server.llm_loaded && !server.owns_process) {
            const bool is_vision_llm =
                !server.embedding_mode
                && (model_has_local_vision_projector(state_snapshot->catalog.source(server.model_id))
                    || !trim(ezl_getenv_default("LLAMA_CPP_MMPROJ_PATH")).empty());
            const Server *&slot = is_vision_llm ? vision_llm : llm;
            if (!slot || server.last_used >= slot->last_used) {
                slot = &server;
            }
        }
    }
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto write_model = [&](const char *type, const Server &server) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << type << "\":{"
            << "\"name\":\"" << json_escape(server.public_model.empty() ? state_snapshot->catalog.source(server.model_id) : server.public_model) << "\","
            << "\"device\":\"" << (server.embedding_mode ? server.embedding_device : "llama.cpp") << "\","
            << "\"vram_gb\":" << json_number_literal(server.embedding_mode ? server.embedding_vram_gb : 0.0) << ","
            << "\"in_use\":" << (server.active_requests > 0 ? "true" : "false") << ","
            << "\"active_requests\":" << server.active_requests << ","
            << "\"last_used\":" << json_number_literal(server.last_used)
            << "}";
    };
    if (llm) {
        write_model("llm", *llm);
    }
    if (vision_llm) {
        write_model("vision_llm", *vision_llm);
    }
    out << "}";
    auto resources = resources_->loaded_models();
    resources.erase(std::remove_if(resources.begin(), resources.end(), [](const auto &m) {
        return m.type != ModelType::Embedding && m.type != ModelType::Llm && m.type != ModelType::VisionLlm;
    }), resources.end());
    return merge_loaded_model_json_objects(out.str(), resource_models_json(resources));
}

std::string LlamaCppInferenceBackend::model_lifecycle_json() const {
    const auto state_snapshot = observe();
    const auto snapshot = lifecycle_.snapshot();
    const auto number = [](double value) { if (!std::isfinite(value)) throw std::invalid_argument("Non-finite lifecycle JSON value"); return json_number_literal(value); };
    const auto quoted = [](const std::string &value) { return '"' + json_escape(value) + '"'; };
    std::vector<std::string> configured, loaded, runtime, summaries, events;
    for (const auto &model : state_snapshot->catalog.available) configured.push_back(quoted(state_snapshot->catalog.source(model)));
    bool active_loaded = false;
    for (const auto index : state_snapshot->load_order) {
        const auto &server = state_snapshot->servers.at(index);
        if (!server.llm_loaded) continue;
        active_loaded |= server.model_id == state_snapshot->active;
        loaded.push_back(quoted(state_snapshot->catalog.source(server.model_id)));
        std::vector<std::pair<std::string, std::string>> fields{{"model", loaded.back()}, {"context", std::to_string(server.lifecycle_context)}};
        for (const auto &field : json_object_ordered_entries(fallback_model_json(server.runtime_json))) ordered_json_upsert(&fields, field);
        runtime.push_back(json_object_from_ordered_entries(fields));
    }
    for (const auto &summary : snapshot.summary) {
        summaries.push_back("{\"operation\":" + quoted(summary.operation) + ",\"model_type\":" + quoted(summary.model_type)
            + ",\"model\":" + quoted(summary.model) + ",\"count\":" + std::to_string(summary.count)
            + ",\"failures\":" + std::to_string(summary.failures) + ",\"min_seconds\":" + number(summary.min_seconds)
            + ",\"max_seconds\":" + number(summary.max_seconds) + ",\"last_seconds\":" + number(summary.last_seconds)
            + ",\"average_seconds\":" + number(summary.average_seconds) + '}');
    }
    for (const auto &event : snapshot.recent_events) {
        events.push_back("{\"timestamp\":" + number(event.timestamp) + ",\"operation\":" + quoted(event.operation)
            + ",\"model_type\":" + quoted(event.model_type) + ",\"model\":" + quoted(event.model)
            + ",\"duration_seconds\":" + number(event.duration_seconds) + ",\"success\":" + (event.success ? "true" : "false") + '}');
    }
    std::map<std::string, std::string> estimates;
    for (const auto &entry : state_snapshot->estimates) estimates[entry.first] = number(entry.second);
    return "{\"llm_model_residency\":" + quoted(state_snapshot->swap ? "swap" : "resident") + ",\"configured_llms\":" + json_array_from_values(configured)
        + ",\"loaded_llms\":" + json_array_from_values(loaded) + ",\"active_llm\":" + (active_loaded ? quoted(state_snapshot->catalog.source(state_snapshot->active)) : "null")
        + ",\"loaded_llm_runtime\":" + json_array_from_values(runtime) + ",\"vram_estimates_gb\":" + json_object_from_values(estimates) + ",\"summary\":" + json_array_from_values(summaries)
        + ",\"recent_events\":" + json_array_from_values(events) + '}';
}

int LlamaCppInferenceBackend::llm_context_size() const {
    const auto state_snapshot = observe();
    for (const auto &server : state_snapshot->servers) if (server.llm_loaded && server.model_id == state_snapshot->active) return server.context_size;
    return 0;
}

std::optional<int64_t> LlamaCppInferenceBackend::llm_parallel_count() const {
    const auto state_snapshot = observe();
    for (const auto &server : state_snapshot->servers) if (server.llm_loaded && server.model_id == state_snapshot->active) return server.llm_parallel;
    return 1;
}

std::map<std::string, int64_t> LlamaCppInferenceBackend::loaded_model_contexts() const {
    const auto state_snapshot = observe();
    std::map<std::string, int64_t> result;
    for (const auto index : state_snapshot->load_order) {
        const auto &server = state_snapshot->servers.at(index);
        if (!server.llm_loaded || server.context_size <= 0) continue;
        const auto parallel = std::max<int64_t>(1, server.llm_parallel.value_or(state_snapshot->catalog.configs.at(server.model_id).parallel));
        result[state_snapshot->catalog.source(server.model_id)] = server.context_size / parallel;
    }
    return result;
}

std::map<std::string, std::string> LlamaCppInferenceBackend::loaded_model_quants() const {
    const auto snapshot = observe();
    std::map<std::string, std::string> result;
    for (const auto index : snapshot->load_order) {
        const auto &server = snapshot->servers.at(index);
        if (!server.llm_loaded) continue;
        const auto quant = worker_model_file_quant(server.model_path);
        if (!quant.empty()) result[server.model_id] = quant;
    }
    return result;
}

bool LlamaCppInferenceBackend::loaded_vision_model() const {
    const auto snapshot = observe();
    for (const auto &server : snapshot->servers)
        if (server.llm_loaded && !server.projector_path.empty()) return true;
    return false;
}

std::optional<std::vector<LlmReplica>> LlamaCppInferenceBackend::llm_replica_state() const {
    std::vector<LlmReplica> replicas;
    { std::lock_guard<std::mutex> lock(replica_state_mutex_); replicas = llm_replicas_; }
    for (auto &replica : replicas) {
        replica.speculative = llm_speculative_backend(replica.source, ezl_getenv_fallback("LLM_SPECULATIVE_TYPE", "auto")) != "none";
    }
    return replicas;
}

void LlamaCppInferenceBackend::publish_llm_replica_locked(const Server &server) {
    std::lock_guard<std::mutex> lock(replica_state_mutex_);
    for (auto &replica : llm_replicas_) if (replica.id == server.model_id) {
        replica.actual_parallel = server.llm_loaded ? server.llm_parallel : std::nullopt;
        replica.resident = server.llm_loaded;
        break;
    }
}

bool LlamaCppInferenceBackend::warmup(std::string *error_out) {
    std::lock_guard<std::mutex> transition(llm_transition_mutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    std::string primary;
    for (const auto &server : servers_) {
        if (server.embedding_mode) continue;
        if (primary.empty()) primary = server.model_id;
        if (!model_has_local_vision_projector(llm_catalog_.source(server.model_id)) && trim(ezl_getenv_default("LLAMA_CPP_MMPROJ_PATH")).empty()) {
            primary = server.model_id;
            break;
        }
    }
    if (error_out) error_out->clear();
    const auto failed = preload_llm_instances(llm_catalog_, swap_llm_residency_, [&](const std::string &id) {
        const auto found = std::find_if(servers_.begin(), servers_.end(), [&](const auto &candidate) { return !candidate.embedding_mode && candidate.model_id == id; });
        // An explicitly configured external endpoint represents the whole
        // catalog through one connection and owns no additional replicas.
        if (found == servers_.end()) return;
        auto &server = *found;
        std::string error;
        if (!ensure_server_locked(&server, &error)) throw std::runtime_error(error);
        if (server.model_id == primary) active_llm_ = server.model_id;
    }, [&](const std::string &id, const std::string &error) {
        std::cerr << "[LLM] Failed to pre-load model " << id << ": " << error << '\n';
        if (error_out) { if (!error_out->empty()) *error_out += '\n'; *error_out += id + ": " + error; }
    });
    { std::lock_guard<std::mutex> state(replica_state_mutex_);
      for (auto &replica : llm_replicas_) if (failed.count(replica.id)) replica.failed = true; }
    return true;
}

std::function<void(bool)> LlamaCppInferenceBackend::unload_llms_for_media() {
    return make_media_handoff(false);
}

std::function<void(bool)> LlamaCppInferenceBackend::unload_llms_for_video() {
    return make_media_handoff(true);
}

bool LlamaCppInferenceBackend::unload_aux_model(const std::string &model) {
    if (model != "embedding") return false;
    return embedding_pool_->unload_idle();
}

void LlamaCppInferenceBackend::warm_aux_model(const std::string &model) {
    if (model == "embedding") embedding_pool_->warm();
}

EmbeddingPool::Handle LlamaCppInferenceBackend::load_embedding_instance(size_t) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t index = embedding_server_begin_;
    while (index < servers_.size() && servers_[index].embedding_loaded) ++index;
    if (index == servers_.size()) {
        auto server = embedding_template_;
        if (!server.started) {
            const auto port = static_cast<size_t>(server.port) + index - embedding_server_begin_;
            if (port > 65535) throw std::runtime_error("Embedding pool exhausted local server ports");
            server.port = static_cast<int>(port);
            server.url = "http://127.0.0.1:" + std::to_string(server.port);
        }
        servers_.push_back(std::move(server));
    }
    auto &server = servers_[index];
    const auto admission = resources_->can_load_model(ModelType::Embedding);
    server.embedding_main_gpu.reset();
    if (starts_with(admission.device, "cuda:")) {
        int ordinal = 0;
        if (parse_strict_int_string(admission.device.substr(5), &ordinal)) server.embedding_main_gpu = ordinal;
    }
    // CPU recommendations are only coarse estimates. Embedding's own model
    // estimator still decides the GPU layer count, as in Pipes.py.
    std::string error;
    if (!ensure_server_locked(&server, &error)) throw std::runtime_error(error);
    try {
        const int context = embedding_startup_context_size(server.config_index, 0);
        const int ubatch = std::min({embedding_int_setting("EMBEDDING_BATCH_SIZE", 512), embedding_int_setting("EMBEDDING_UBATCH_SIZE", 512), context});
        const auto inputs = embedding_warmup_inputs(server.embedding_device == "cuda", context, ubatch);
        for (const auto &input : inputs) {
            const std::string configured = ezl_getenv_fallback("EMBEDDING_DIMENSIONS", "");
            int dimensions = 0;
            if (!configured.empty()) {
                if (!parse_strict_int_string(configured, &dimensions)) throw std::runtime_error("dimensions must be an integer");
                if (dimensions < 32 || dimensions > 1024) throw std::runtime_error("Qwen3-Embedding-0.6B dimensions must be 32..1024");
            }
            const auto body = "{\"input\":" + input + ",\"model\":\"" + json_escape(server.public_model) + "\"}";
            const auto response = http_json_request("POST", server.url, "/v1/embeddings", api_key_, body, 0, false);
            const auto failure = trim(json_object_field(response.body, "error", ""));
            if (response.status < 200 || response.status >= 300 || !failure.empty()) {
                throw std::runtime_error("Embedding warmup failed: " + (failure.empty() ? response.body : failure));
            }
            if (!is_valid_top_level_json_object(response.body)) throw std::runtime_error("Invalid embedding warmup response");
        }
    } catch (...) {
        // Warmup runs after successful construction in the reference: failure
        // discards the instance, without retrying a smaller GPU layer count.
        stop_owned_server_locked(&server);
        throw;
    }
    server.embedding_loaded = true; server.last_used = monotonic_epoch_seconds();
    publish_observation_locked();
    return index;
}

void LlamaCppInferenceBackend::register_embedding_resources(const EmbeddingPool::Snapshot &pool) {
    auto loaded = pool.resident; loaded.insert(loaded.end(), pool.transient.begin(), pool.transient.end());
    if (loaded.empty()) { resources_->unregister_model(ModelType::Embedding); return; }
    // The pool calls this with its own mutex held. Waiting for process startup
    // here would also block every pool/status observer.
    const auto state_snapshot = observe();
    double vram = 0; bool cuda = false;
    for (const auto handle : loaded) {
        const auto &server = state_snapshot->servers.at(handle); vram += server.embedding_vram_gb;
        cuda |= server.embedding_device.find("cuda") != std::string::npos;
    }
    const auto &first = state_snapshot->servers.at(loaded.front());
    resources_->register_model(ModelType::Embedding, first.model_id, cuda ? "cuda" : first.embedding_device, vram);
}

void LlamaCppInferenceBackend::destroy_embedding_instance(EmbeddingPool::Handle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto &server = servers_.at(handle);
    stop_owned_server_locked(&server);
    // External processes belong to the operator; release only local metadata.
    server.embedding_loaded = false;
}

bool LlamaCppInferenceBackend::embedding_slot_state(RouterSlotState *state) const {
    const auto pool = embedding_pool_->snapshot();
    const auto state_snapshot = observe();
    state->capacity = static_cast<int>(pool.capacity);
    state->in_flight = 0;
    for (const auto &server : state_snapshot->servers) if (server.embedding_mode) state->in_flight += server.active_requests;
    state->queued = 0; // The reference slot snapshot reports only active inference.
    state->available = std::max(0, state->capacity - state->in_flight);
    return true;
}

std::string LlamaCppInferenceBackend::embedding_pool_json() const {
    const auto pool = embedding_pool_->snapshot();
    const auto state_snapshot = observe();
    nlohmann::json devices = nlohmann::json::array();
    for (const auto handle : pool.resident) devices.push_back(state_snapshot->servers.at(handle).embedding_device);
    return nlohmann::json{{"configured", pool.capacity}, {"resident", pool.resident.size()}, {"devices", devices}}.dump();
}

std::function<void(bool)> LlamaCppInferenceBackend::make_media_handoff(bool video) {
    std::lock_guard<std::mutex> transition(llm_transition_mutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<size_t, Server>> saved;
    for (const auto i : llm_load_order_) {
        const auto &server = servers_[i];
        if (server.embedding_mode || !server.started) continue;
        if (server.active_requests > 0) throw std::runtime_error("Cannot unload an LLM with active inference");
        if (server.owns_process || server.llm_loaded) saved.emplace_back(i, server);
    }
    const auto prior_active = active_llm_;
    const auto unload_start = std::chrono::steady_clock::now();
    for (const auto &entry : saved) {
        auto &server = servers_[entry.first];
        stop_owned_server_locked(&server);
        // An explicitly configured external llama-server is never our process
        // to terminate. Detach its loaded-model metadata during the handoff.
        server.llm_loaded = false;
        publish_llm_replica_locked(server);
    }
    active_llm_.clear();
    const auto unloaded_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - unload_start).count();
    if (saved.empty()) lifecycle_.record("unload", "llm", "LLM", unloaded_seconds);
    else for (const auto &entry : saved) lifecycle_.record("unload", "llm", entry.second.model_id, unloaded_seconds);
    return [this, saved = std::move(saved), prior_active, video, attempt = 0](bool reload) mutable {
        const bool retry = video && attempt++ > 0;
        if (!reload) return;
        std::lock_guard<std::mutex> transition(llm_transition_mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &entry : saved) {
            auto &server = servers_[entry.first];
            const auto &prior = entry.second;
            // The reference reuses known-good placement when the configured
            // context still matches the previous runtime context.
            const bool reuse_runtime = !retry || configured_load_context(server) == prior.context_size;
            server.retry_gpu_layers = reuse_runtime ? prior.current_gpu_layers : kLlamaGpuLayersUnset;
            if (retry) server.cpu_only_retry = false;
            std::string error;
            if (!ensure_server_locked(&server, &error, retry ? std::nullopt : std::optional<int64_t>(prior.context_size))) {
                // The reference logs failed restores and resumes admission so
                // the next text request can retry loading the affected model.
                std::cerr << (video ? "[VIDEO]" : "[MUSIC]") << " Failed to reload model " << prior.model_id << ": " << error << '\n';
                continue;
            }
            if (!server.owns_process && prior.llm_loaded) note_llm_loaded_locked(&server);
            if (active_llm_.empty() || prior_active == server.model_id) active_llm_ = server.model_id;
        }
    };
}

InferenceResult LlamaCppInferenceBackend::text_result(const std::string &body, const std::string &replica, bool chat_mode) {
    return collect_llm_stream(lazy_text_result(body, replica, chat_mode));
}

InferenceResult LlamaCppInferenceBackend::lazy_text_result(const std::string &body, const std::string &replica, bool chat_mode) {
    return forward(chat_mode ? "/v1/chat/completions" : "/v1/completions", body, replica);
}

InferenceResult InferenceBackend::pipeline_text_result(const std::string &body, const std::string &replica, bool chat_mode,
    const LlmInputOperations &input) {
    return lazy_text_result(llm_apply_audio_format(body, chat_mode, input), replica, chat_mode);
}

InferenceResult LlamaCppInferenceBackend::pipeline_text_result(const std::string &body, const std::string &replica, bool chat_mode,
    const LlmInputOperations &input) {
    LlmInputOperations operations = input;
    operations.local_uri = ezl_getenv_default("EZLOCALAI_URL");
    Server *request_server = nullptr;
    struct Release { std::function<void()> callback; ~Release() { if (callback) callback(); } } release;
    std::string load_error;
    operations.load = [&] {
        std::lock_guard<std::mutex> transition(llm_transition_mutex_);
        std::unique_lock<std::mutex> lock(mutex_);
        request_server = select_server_locked(body, false, replica);
        std::optional<int64_t> context;
        if (request_server) {
            context = std::stoll(llm_configured_context_limit(
                std::to_string(llm_catalog_.configs.at(request_server->model_id).context), optimal_context_));
            if (request_server->started && request_server->owns_process && request_server->context_size < *context
                && request_server->active_requests == 0) stop_owned_server_locked(request_server);
        }
        if (request_server) prepare_request_llm_locked(request_server, context.value_or(configured_load_context(*request_server)), lock);
        if (!ensure_server_locked(request_server, &load_error, context)) throw std::runtime_error(load_error);
        ++request_server->active_requests;
        release.callback = [&] {
            std::lock_guard<std::mutex> lock(mutex_);
            --request_server->active_requests;
            request_server->last_used = resources_->current_time();
        };
        return LlmInputModel{llm_catalog_.source(request_server->model_id),
            !request_server->projector_path.empty() || !trim(ezl_getenv_default("LLAMA_CPP_MMPROJ_PATH")).empty()};
    };
    operations.fetch_image = [](const std::string &url, const std::string &agent) {
        const auto response = fallback_http_request("GET", url, "", "", "", "", 30, true, true, false, agent);
        if (response.transport_error || response.status >= 400) throw std::runtime_error("Image download failed");
        return LlmInputImage{response.body, response_header(response.headers, "content-type").value_or("image/jpeg")};
    };
    operations.png = [](const std::string &bytes) { return encode_rgb_png(decode_rgb_image(bytes)); };
    operations.fetch_audio = [](const std::string &url) {
        const auto response = fallback_http_request("GET", url, "", "", "", "", 0, true, true, false, "python-requests/2.32.5");
        if (response.transport_error) throw LlmResponseError{"ConnectionError", "Audio download failed"};
        // requests.get(...).content is used even for an HTTP error status.
        return response.body;
    };
    operations.video_fps = ezl_getenv_fallback("VIDEO_FPS", "1.0");
    operations.video_max_frames = ezl_getenv_fallback("VIDEO_MAX_FRAMES", "16");
    operations.extract_video = [](const std::string &raw, double fps, int64_t maximum) {
        if (llm_output_type(raw) != "str") throw LlmResponseError{"AttributeError", "'" + llm_output_type(raw) + "' object has no attribute 'startswith'"};
        return extract_llm_video(parse_json_string_at(raw, 0), fps, maximum, [](const std::string &url) {
            const auto response = fallback_http_request("GET", url, "", "", "", "", 60, true, true, false, "Mozilla/5.0");
            if (response.transport_error || response.status >= 400) throw std::runtime_error("Video download failed");
            return response.body;
        });
    };
    operations.read_audio = [](const std::string &path) {
        std::ifstream file(path, std::ios::binary); if (!file) throw std::runtime_error("Cannot read video audio");
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    };
    operations.remove_audio = [](const std::string &path) { if (std::filesystem::exists(path)) std::filesystem::remove(path); };
    operations.describe = [&](const std::vector<std::string> &images, const std::string &user_text) -> std::string {
        try {
            std::vector<std::pair<std::string, std::string>> models;
            { std::lock_guard<std::mutex> lock(mutex_);
              for (const auto &id : llm_catalog_.available) models.emplace_back(id, llm_catalog_.source(id)); }
            std::string selected;
            for (const auto &model : models) {
                bool vision = false;
                bool cached = false;
                { std::lock_guard<std::mutex> lock(mutex_);
                  const auto found = vision_model_cache_.find(model.first);
                  if (found != vision_model_cache_.end()) { cached = true; vision = found->second; }
                  else for (const auto &server : servers_) if (server.model_id == model.first) vision = !server.projector_path.empty(); }
                if (cached && !vision) continue;
                if (!vision && model.second.find('/') != std::string::npos) {
                    auto short_name = model.second.substr(model.second.rfind('/') + 1);
                    const auto suffix = short_name.find("-GGUF"); if (suffix != std::string::npos) short_name.resize(suffix);
                    const auto folder = std::filesystem::path("models") / short_name;
                    const auto projector = [](const std::string &name) {
                        return to_lower(name).find("mmproj") != std::string::npos
                            && name.size() >= 5 && name.substr(name.size() - 5) == ".gguf";
                    };
                    try {
                        if (std::filesystem::is_directory(folder))
                            for (const auto &entry : std::filesystem::directory_iterator(folder))
                                if (projector(entry.path().filename().string())) { vision = true; break; }
                        if (!vision) for (const auto &file : list_hub_files(model.second)) if (projector(file)) {
                            download_hub_file(model.second, file, folder.string()); vision = true; break;
                        }
                    } catch (...) { vision = false; }
                }
                { std::lock_guard<std::mutex> lock(mutex_); vision_model_cache_[model.first] = vision; }
                if (vision) { selected = model.first; break; }
            }
            if (selected.empty()) return "null";
            Server *vision_server;
            { std::lock_guard<std::mutex> transition(llm_transition_mutex_);
              std::unique_lock<std::mutex> lock(mutex_);
              vision_server = select_server_locked("{}", false, selected);
              std::string error;
              prepare_request_llm_locked(vision_server, 16384, lock);
              if (!ensure_server_locked(vision_server, &error, 16384) || vision_server->projector_path.empty()) return "null";
              ++vision_server->active_requests; }
            Release vision_release{[&] { std::lock_guard<std::mutex> lock(mutex_); --vision_server->active_requests; }};
            auto processed = llm_prepare_images(images, operations, true);
            if (processed.empty()) return "null";
            processed.insert(processed.begin(), "{\"type\":\"text\",\"text\":\"" + json_escape(
                "Describe the contents of the image(s) in detail. The user's question about the image(s) is: " + user_text) + "\"}");
            auto response = forward("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":"
                + json_array_from_values(processed) + "}],\"max_tokens\":2048,\"temperature\":0.3}", selected, false, 16384);
            if (response.failure) return "null";
            const auto choices = json_object_field(response.body, "choices", "");
            if (choices.empty()) return '"' + json_escape(stream_python_str(response.body)) + '"';
            const auto entries = json_array_elements(choices);
            if (entries.empty()) return "null";
            return input_at(input_at(entries.front(), "message"), "content");
        } catch (...) { return "null"; }
    };
    std::string source_model;
    { std::lock_guard<std::mutex> lock(mutex_);
      if (auto *server = select_server_locked(body, false, replica)) source_model = llm_catalog_.source(server->model_id); }
    try {
        const auto settings = llm_model_request_settings(body, source_model);
        std::string prepared;
        prepared = llm_prepare_text_image_input(settings, chat_mode, operations);
        release.callback(); release.callback = {};
        auto result = forward(chat_mode ? "/v1/chat/completions" : "/v1/completions", prepared, replica, false);
        if (result.failure) result.prepared_request = prepared;
        return result;
    } catch (const std::runtime_error &) {
        if (!load_error.empty()) return make_error(503, load_error);
        throw;
    }
}

std::string LlamaCppInferenceBackend::chat_for_replica(const std::string &body, const std::string &replica) { return text_result(body, replica, true).body; }
std::string LlamaCppInferenceBackend::completion_for_replica(const std::string &body, const std::string &replica) { return forward("/v1/completions", body, replica).body; }

std::string LlamaCppInferenceBackend::chat(const std::string &request_body) {
    return text_result(request_body, "", true).body;
}

std::string LlamaCppInferenceBackend::completion(const std::string &request_body) {
    return forward("/v1/completions", request_body).body;
}

std::string LlamaCppInferenceBackend::embedding(const std::string &request_body) {
    const auto handle = embedding_pool_->acquire();
    Server *server;
    std::string url;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        server = &servers_.at(handle); ++server->active_requests;
        server->last_used = monotonic_epoch_seconds(); url = server->url;
        resources_->mark_model_in_use(ModelType::Embedding, true);
    }
    auto release = [&] {
        { std::lock_guard<std::mutex> lock(mutex_); --server->active_requests; resources_->mark_model_in_use(ModelType::Embedding, false); }
        embedding_pool_->release(handle);
    };
    HttpResponse response;
    try { response = http_json_request("POST", url, "/v1/embeddings", api_key_, request_body, 0, false); }
    catch (...) { release(); throw; }
    release();
    if (response.status < 200 || response.status >= 300) {
        const auto body = trim(response.body);
        if (is_valid_top_level_json_object(body) && (!trim(extract_json_value(body, "error", "")).empty() || extract_json_int_value(body, "code", 0) != 0)) return body;
        return make_error(response.status, response.body.empty() ? "llama.cpp request failed" : response.body);
    }
    return response.body;
}

void LlamaCppInferenceBackend::stop_owned_server_locked(Server *server) {
    publish_observation_locked();
    if (!server || !server->owns_process || server->pid <= 0) {
        return;
    }
    if (!server->embedding_mode && server->llm_loaded) {
        const auto type = llm_resource_type(*server);
        for (const auto &resource : resources_->loaded_models()) {
            if (resource.type == type && resource.name == server->model_id) resources_->unregister_model(type);
        }
    }
    kill(static_cast<pid_t>(server->pid), SIGTERM);
    int status = 0;
    for (int i = 0; i < 20; ++i) {
        const pid_t done = waitpid(static_cast<pid_t>(server->pid), &status, WNOHANG);
        if (done == server->pid || done < 0) {
            server->pid = -1;
            server->owns_process = false;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (server->pid > 0) {
        kill(static_cast<pid_t>(server->pid), SIGKILL);
        waitpid(static_cast<pid_t>(server->pid), &status, 0);
        server->pid = -1;
        server->owns_process = false;
    }
    server->started = false;
    server->llm_loaded = false;
    if (!server->embedding_mode) publish_llm_replica_locked(*server);
    server->embedding_loaded = false;
}

void LlamaCppInferenceBackend::stop_owned_servers() {
    std::lock_guard<std::mutex> lock(mutex_);
    resources_->unregister_model(ModelType::Embedding);
    for (auto &server : servers_) {
        stop_owned_server_locked(&server);
    }
}

std::unique_ptr<NativeServer> create_native_server(ezl_env_config config,
    std::unique_ptr<InferenceBackend> backend, std::shared_ptr<ResourceManager> resources,
    SttPoolFactory stt_factory, TtsPoolFactory tts_factory, std::optional<bool> precache_done) {
    const auto voice = capture_service_client_settings("VOICE_SERVER");
    const auto text = capture_service_client_settings("TEXT_SERVER");
    const auto image = capture_service_client_settings("IMAGE_SERVER");
    const bool text_mode = text.dedicated || (!image.dedicated && !voice.dedicated);
    // Python preloads by default. Retain the native opt-out for lazy startup.
    const char *configured = std::getenv("EZLOCALAI_PRELOAD_MODEL_ON_START");
    const auto preload = configured ? llm_unicode_lower(configured) : "true";
    const bool requested = preload == "1" || preload == "true" || preload == "yes" || preload == "on";
    if (requested && !(voice.dedicated && !text_mode) && text.url.empty()
        && !captured_llm_catalog(config.default_model).available.empty()) {
        std::string error;
        if (!backend->warmup(&error)) throw std::runtime_error("Failed to preload ezlocalai native backend" + (error.empty() ? std::string() : ": " + error));
    }
    auto *model_backend = backend.get();
    // NativeServer initializes Qwen TTS and then STT. Embeddings must follow
    // both so their placement observes the same resident pools as Pipes.
    auto server = std::make_unique<NativeServer>(config, std::move(backend), std::move(resources),
        std::move(stt_factory), std::move(tts_factory), TtsRequestOperations{}, precache_done);
    if (embedding_should_preload()) model_backend->warm_aux_model("embedding");
    server->preload_media();
    return server;
}

void NativeServer::preload_media() {
    const auto enabled = [](const char *key) {
        return llm_unicode_lower(strip_config_value(ezl_getenv_fallback(key, "false"))) == "true";
    };
    if ((!enabled("VIDEO_ENABLED") && !enabled("IMAGE_ENABLED")) || !image_client().url.empty()) return;
    const auto loaded = backend_->loaded_models_json();
    const bool resident = json_object_field(loaded, "llm", "null") != "null"
        || json_object_field(loaded, "vision_llm", "null") != "null";
    const auto secondary = [] { return secondary_cuda_device(visible_cuda_devices()); };
    const auto forwarded = [this] { return !image_client().url.empty(); };
    auto video_model = std::string(ezl_getenv_default("VIDEO_MODEL"));
    if (video_model.empty()) video_model = "QuantStack/Wan2.2-T2V-A14B-GGUF";
    video_model = strip_config_value(video_model);
    if (enabled("VIDEO_ENABLED") && !video_model.empty() && llm_unicode_lower(video_model) != "none"
        && !video_should_unload_llm(resident, secondary, forwarded)) {
        try {
            load_video_backend(video_model, "cuda");
        } catch (const std::exception &error) {
            video_.reset(); ltx_video_.reset();
            std::cerr << "[VIDEO] Failed to preload: " << error.what() << "; will retry on request\n";
        }
    }
    const std::string image_model = ezl_getenv_default("IMG_MODEL");
    if (enabled("IMAGE_ENABLED") && !image_model.empty() && llm_unicode_lower(image_model) != "none"
        && !image_should_unload_llm(resident, bool(image_), secondary,
            [this] { return resources_->get_total_free_vram(); }, forwarded)) {
        try {
            image_ = std::make_unique<QwenImage>("cuda");
            // Preserve Python's resource estimate, even when its constructor
            // returns an unavailable wrapper. This is not allocated GPU RAM.
            resources_->register_model(ModelType::Image, image_model, "cuda", 4.0);
        } catch (const std::exception &error) {
            image_.reset();
            std::cerr << "[IMG] Failed to preload: " << error.what() << "; will retry on request\n";
        }
    }
}

NativeServer::NativeServer(ezl_env_config config, std::unique_ptr<InferenceBackend> backend, std::shared_ptr<ResourceManager> resources,SttPoolFactory stt_factory,TtsPoolFactory tts_factory,TtsRequestOperations tts_request,std::optional<bool> precache_done)
    : config_(config), backend_(std::move(backend)), resources_(resources ? std::move(resources) : backend_->resource_manager()) {
    // Python's class and Form defaults capture configuration at app import,
    // before its lazily generated OpenAPI schema is first requested.
    auto schema = nlohmann::ordered_json::parse(generated_openapi::kApiOpenApiJson);
    auto &schemas = schema.at("components").at("schemas");
    for (const auto *name : {"ChatCompletions", "Completions"})
        schemas.at(name).at("properties").at("model")["default"] = ezl_getenv_default("DEFAULT_MODEL");
    schemas.at("EmbeddingModel").at("properties").at("model")["default"] = ezl_getenv_default("EMBEDDING_MODEL");
    for (const auto *name : {"Body_speech_to_text_v1_audio_transcriptions_post",
                             "Body_audio_translations_v1_audio_translations_post",
                             "Body_generate_subtitles_v1_audio_subtitles_post"})
        schemas.at(name).at("properties").at("model")["default"] = ezl_getenv_default("WHISPER_MODEL");
    worker_openapi_json_ = schema.dump();
    if (!resources_) resources_ = std::make_shared<ResourceManager>();
    // Pipes initialization always reaches the shared voice singleton, including
    // when both modalities are disabled. Subsequent environment edits do not
    // replace its URL, authentication or dedicated-server mode.
    voice_client();
    llm_catalog_ = captured_llm_catalog(config_.default_model);
    llm_replicas_ = llm_catalog_.replicas();
    try {
        const auto slots = llm_parameter_integer(ezl_getenv_fallback("STT_N_PARALLEL", "1"));
        if (!slots.empty() && slots.front() != '-') {
            try { llm_audio_capacity_ = std::max<int64_t>(1, std::stoll(slots)); }
            catch (const std::out_of_range &) { llm_audio_capacity_ = INT64_MAX; }
        }
    } catch (const std::exception &) { llm_audio_capacity_ = 1; }
    embedding_pool_configured_ = embedding_parallel_slots();
    try { const auto value=llm_parameter_integer(ezl_getenv_fallback("TTS_N_PARALLEL","1"));try {tts_pool_configured_=std::max<int64_t>(1,std::stoll(value));}catch(const std::out_of_range &){tts_pool_configured_=!value.empty()&&value[0]=='-'?1:INT64_MAX;} } catch(...) {}
    if (tts_factory) {
        const auto capacity = tts_pool_configured_;
        auto pool=tts_factory(resources_);pool.should_preload=[this]{return native_tts_preload();};
        VoiceSlotOperations voice;
        voice.should_handoff=[this](const auto &){return native_tts_handoff();};
        voice.set_unavailable=[this](bool value){std::lock_guard<std::mutex> lock(resource_mutex_);llm_temporarily_unavailable_=value;llm_idle_cv_.notify_all();};
        voice.wait_for_idle=[this](const auto &,const VoiceCancellation &cancel){
            auto generic=std::string(ezl_getenv_fallback("VOICE_WAIT_FOR_LLM_IDLE_TIMEOUT","60"));if(generic.empty())generic="60";
            auto configured=std::string(ezl_getenv_fallback("TTS_WAIT_FOR_LLM_IDLE_TIMEOUT",generic.c_str()));if(configured.empty())configured=generic;
            double timeout=0;if(!fallback_float(configured,&timeout))throw TtsError("ValueError","could not convert string to float: "+python_repr_string(configured));
            const auto start=std::chrono::steady_clock::now();std::unique_lock<std::mutex> lock(resource_mutex_);
            while(llm_slots_.total>0){if(cancel.requested())throw VoiceCancelled("");if(timeout<=0||std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>=std::max(0.,timeout))throw TtsError("RuntimeError","LLM inference is active; TTS cannot unload the LLM yet");llm_idle_cv_.wait_for(lock,std::chrono::milliseconds(100));}
        };
        voice.unload=[this](const auto &,bool){tts_restore_llm_=backend_->unload_llms_for_media();return "{\"handoff\":true}";};
        voice.restore=[this](const auto &,const auto &,const auto &){
            auto generic=std::string(ezl_getenv_fallback("VOICE_RELOAD_LLM_AFTER_GENERATION","false"));if(generic.empty())generic="false";
            auto value=std::string(ezl_getenv_fallback("TTS_RELOAD_LLM_AFTER_GENERATION",generic.c_str()));if(value.empty())value=generic;value=to_lower(trim(value));
            auto restore=std::move(tts_restore_llm_);if(restore)restore(value!="0"&&value!="false"&&value!="no"&&value!="off");
        };
        if(!tts_request.voice_server_mode)tts_request.voice_server_mode=[this]{return voice_client().dedicated;};
        tts_service_=std::make_unique<TtsService>(capacity,std::move(pool),std::move(voice),voice_handoff_,resources_,tts_gtts_operations(std::move(tts_request)));
        if(llm_unicode_lower(ezl_getenv_default("TTS_ENABLED"))=="true"&&voice_client().url.empty()) {
            if(native_tts_preload())tts_service_->warm();
            else if(!native_tts_handoff()&&!(precache_done?*precache_done:std::filesystem::exists("/tmp/ezlocalai_precache.done")))tts_service_->warm_cache();
        }
    }
    if (stt_factory) {
        std::filesystem::create_directories("outputs");
        auto pool = stt_factory(resources_); pool.should_preload = [this] { return native_stt_preload(); };
        VoiceSlotOperations voice;
        voice.should_handoff = [this](const auto &) { return native_stt_handoff(); };
        voice.set_unavailable = [this](bool value) { std::lock_guard<std::mutex> lock(resource_mutex_); llm_temporarily_unavailable_ = value; llm_idle_cv_.notify_all(); };
        voice.wait_for_idle = [this](const auto &,const VoiceCancellation &cancel) {
            auto generic = std::string(ezl_getenv_fallback("VOICE_WAIT_FOR_LLM_IDLE_TIMEOUT","60")); if (generic.empty()) generic = "60";
            auto configured = std::string(ezl_getenv_fallback("STT_WAIT_FOR_LLM_IDLE_TIMEOUT",generic.c_str())); if (configured.empty()) configured = generic;
            double timeout = 0; if (!fallback_float(configured,&timeout)) throw LlmResponseError{"ValueError","could not convert string to float: " + python_repr_string(configured)};
            const auto start = std::chrono::steady_clock::now(); std::unique_lock<std::mutex> lock(resource_mutex_);
            while (llm_slots_.total > 0) {
                if (cancel.requested()) throw VoiceCancelled("");
                if (timeout <= 0 || std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= std::max(0.,timeout)) throw LlmResponseError{"RuntimeError","LLM inference is active; STT cannot unload the LLM yet"};
                llm_idle_cv_.wait_for(lock,std::chrono::milliseconds(100));
            }
        };
        voice.unload = [this](const auto &,bool) { stt_restore_llm_ = backend_->unload_llms_for_media(); return "{\"handoff\":true}"; };
        voice.restore = [this](const auto &,const auto &,const auto &) {
            auto generic = std::string(ezl_getenv_fallback("VOICE_RELOAD_LLM_AFTER_GENERATION","false")); if (generic.empty()) generic = "false";
            auto value = std::string(ezl_getenv_fallback("STT_RELOAD_LLM_AFTER_GENERATION",generic.c_str())); if (value.empty()) value = generic; value = to_lower(trim(value));
            const bool reload = value != "0" && value != "false" && value != "no" && value != "off";
            auto restore = std::move(stt_restore_llm_); if (restore) restore(reload);
        };
        stt_service_ = std::make_unique<SttService>(ezl_getenv_default("WHISPER_MODEL"),llm_audio_capacity_,std::move(pool),std::move(voice),voice_handoff_,resources_,[this] { return voice_client().dedicated; });
        if (native_stt_preload() && llm_unicode_lower(ezl_getenv_default("STT_ENABLED")) == "true" && voice_client().url.empty()) stt_service_->warm();
    }
    backend_->set_llm_reclaimer([this](double required) { reclaim_auxiliary_memory_for_llm(required); });
    size_t text_capacity = config_.max_concurrent_requests;
    if (const auto replicas = backend_->llm_replica_state()) {
        llm_replicas_ = *replicas;
        text_capacity = 0;
        for (const auto &replica : *replicas) if (!replica.failed) text_capacity += llm_replica_capacity(replica);
        text_capacity = std::max<size_t>(1, text_capacity);
    }
    queue_ = ezl_queue_create(backend_->swap_llm_residency() ? 1 : text_capacity, config_.max_queue_size);
    if (!queue_) {
        throw std::bad_alloc();
    }
    int rc = ezl_queue_set_state_observer(queue_, [](void *signal) {
        static_cast<WorkerHeartbeatSignal *>(signal)->notify();
    }, &worker_heartbeat_signal_);
    if (!rc) rc = ezl_queue_start(queue_, &NativeServer::queue_processor, this);
    if (rc != 0) {
        ezl_queue_destroy(queue_);
        queue_ = nullptr;
        throw std::runtime_error("failed to start native request queue");
    }
}

WakewordJobs &NativeServer::wakeword_jobs_manager() {
    std::lock_guard<std::mutex> lock(wakeword_jobs_mutex_);
    if (!wakeword_jobs_) {
        WakewordJobOperations operations;
        operations.trainer = wakeword_trainer_;
        operations.load = [] { std::vector<std::string> out; for (const auto &job : wakeword_load_jobs()) out.push_back(job.json); return out; };
        const auto voices_dir = std::filesystem::current_path() / "voices";
        operations.samples = [this, voices_dir](const VoiceCancellation &cancel) {
            if (!tts_service_) throw std::runtime_error("Native Qwen TTS is unavailable for wake-word training");
            std::vector<std::string> voices; std::error_code error;
            for (const auto &entry : std::filesystem::directory_iterator(voices_dir, error)) {
                if (entry.is_regular_file(error) && to_lower(entry.path().extension().string()) == ".wav") voices.push_back(entry.path().stem().string());
            }
            std::sort(voices.begin(), voices.end()); if (voices.empty()) voices.push_back("default");
            return wakeword_qwen_tts_operations(*tts_service_, std::move(voices), cancel);
        };
        wakeword_jobs_ = std::make_unique<WakewordJobs>(std::filesystem::current_path(), std::move(operations));
    }
    return *wakeword_jobs_;
}

NativeServer::~NativeServer() {
    if (stt_service_) stt_service_->cancel();
    if (tts_service_) tts_service_->cancel();
    llm_audio_stopping_ = true; llm_audio_cv_.notify_all();
    { std::lock_guard<std::mutex> lock(resource_mutex_); llm_slots_stopping_ = true; llm_idle_cv_.notify_all(); }
    stop();
    wakeword_jobs_.reset();
    if (queue_) {
        ezl_queue_destroy(queue_);
        queue_ = nullptr;
    }
    for (const auto &handle : local_streams_) handle.second();
    backend_->set_llm_reclaimer({});
    stt_service_.reset();
    tts_service_.reset();
}

bool NativeServer::start(const std::string &host, int port, std::string *error_out) {
    if (running_) {
        return true;
    }
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        if (error_out) {
            *error_out = std::strerror(errno);
        }
        return false;
    }
    int yes = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (host.empty() || host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        if (error_out) {
            *error_out = "invalid IPv4 host: " + host;
        }
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (bind(server_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        if (error_out) {
            *error_out = std::strerror(errno);
        }
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }
    if (listen(server_fd_, 64) != 0) {
        if (error_out) {
            *error_out = std::strerror(errno);
        }
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_ = true;
    accept_thread_ = std::thread(&NativeServer::accept_loop, this);
    usage_flush_thread_ = std::thread(&NativeServer::usage_flush_loop, this);
    start_worker_heartbeat(host, port);
    return true;
}

void NativeServer::stop() {
    if (!running_) {
        stop_worker_heartbeat();
        flush_usage_if_dirty();
        return;
    }
    running_ = false;
    stop_worker_heartbeat();
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    {
        std::unique_lock<std::mutex> lock(client_mutex_);
        for (const int fd : client_fds_) shutdown(fd, SHUT_RDWR);
        client_cv_.wait(lock, [this] { return client_fds_.empty(); });
    }
    if (usage_flush_thread_.joinable()) {
        usage_flush_thread_.join();
    }
    flush_usage_if_dirty();
}

bool NativeServer::running() const {
    return running_;
}

void NativeServer::start_worker_heartbeat(const std::string &host, int port) {
    (void)host;
    if (worker_heartbeat_running_) {
        return;
    }
    if (native_router_mode_enabled()) {
        return;
    }
    const std::string router_url = strip_trailing_slashes(trim(ezl_getenv_default("ROUTER_URL")));
    if (router_url.empty()) {
        return;
    }
    if (!fallback_float(ezl_getenv_default("WORKER_HEARTBEAT_INTERVAL"), &worker_heartbeat_interval_)) {
        std::cerr << "[Heartbeat] Invalid heartbeat interval; registration is disabled\n";
        return;
    }
    worker_heartbeat_interval_ = std::max(2.0, worker_heartbeat_interval_);
    worker_heartbeat_router_url_ = router_url;
    worker_heartbeat_label_ = trim(ezl_getenv_default("WORKER_LABEL"));
    if (worker_heartbeat_label_.empty()) {
        worker_heartbeat_label_ = local_hostname();
    }
    worker_heartbeat_id_ = worker_heartbeat_label_ + "-" + short_random_hex();
    worker_heartbeat_port_ = port > 0 ? port : 8091;
    worker_heartbeat_api_key_ = trim(ezl_getenv_default("ROUTER_API_KEY"));
    if (worker_heartbeat_api_key_.empty()) {
        worker_heartbeat_api_key_ = trim(config_.api_key);
    }
    if (env_true("WORKER_TUNNEL")) {
        worker_heartbeat_url_ = "tunnel://" + worker_heartbeat_id_;
    } else {
    worker_heartbeat_url_ = strip_trailing_slashes(trim(ezl_getenv_default("EZLOCALAI_URL")));
    }
    worker_heartbeat_running_ = true;
    worker_heartbeat_signal_.start();
    worker_heartbeat_cancelled_ = false;
    worker_heartbeat_done_ = false;
    worker_heartbeat_thread_ = std::thread(&NativeServer::worker_heartbeat_loop, this);
    start_worker_tunnel(port);
}

void NativeServer::stop_worker_heartbeat() {
    stop_worker_tunnel();
    if (!worker_heartbeat_running_ && !worker_heartbeat_thread_.joinable()) {
        return;
    }
    worker_heartbeat_running_ = false;
    worker_heartbeat_signal_.stop();
    worker_heartbeat_cv_.notify_all();
    if (worker_heartbeat_thread_.joinable()) {
        {
            std::unique_lock<std::mutex> lock(worker_heartbeat_mutex_);
            if (!worker_heartbeat_cv_.wait_for(lock, std::chrono::seconds(5),
                    [this] { return worker_heartbeat_done_; }))
                worker_heartbeat_cancelled_ = true;
        }
        worker_heartbeat_thread_.join();
    }
}

bool NativeServer::worker_router_post(const std::string &path, const std::string &body) const {
    if (worker_heartbeat_router_url_.empty()) {
        return false;
    }
    int timeout_seconds = 10;
    int parsed_timeout = 0;
    if (parse_strict_int_string(ezl_getenv_default("WORKER_HEARTBEAT_TIMEOUT"), &parsed_timeout) && parsed_timeout > 0) {
        timeout_seconds = parsed_timeout;
    }
    int status = 0;
    // Python gives the running task five seconds, then cancels it. The
    // subsequent deregistration has its own ordinary HTTP timeout.
    const bool deregister = path == "/v1/router/deregister";
    const auto cancelled = [this,deregister] { return !deregister && worker_heartbeat_cancelled_.load(); };
    fallback_http_request("POST", worker_heartbeat_router_url_, path,
        worker_heartbeat_api_key_, "application/json", body, timeout_seconds,
        true, true, false, "", {}, {}, "Python/3.12 aiohttp/3.14.5", [&](int value) { status = value; }, cancelled);
    if (cancelled()) throw WorkerHeartbeatCancelled();
    // The reference accepts a 2xx response even when its optional JSON body
    // cannot be decoded, is truncated, or times out after headers arrive.
    return status >= 200 && status < 300;
}

std::string NativeServer::worker_state_json() const {
    const auto memory = read_memory_info();
    ezl_queue_snapshot snapshot{};
    ezl_queue_get_snapshot(queue_, &snapshot);

    const auto model_ids = llm_catalog_.available;
    std::vector<std::string> public_models;
    std::set<std::string> seen_models;
    for (const auto &model_id : model_ids) {
        const std::string public_name = llm_catalog_.source(model_id);
        if (!public_name.empty() && seen_models.insert(public_name).second) {
            public_models.push_back(public_name);
        }
    }

    const std::string resources_body = resources(false).body;
    const std::string slots = trim(extract_json_value(resources_body, "slots", "{}"));
    const std::string cap_slots = trim(extract_json_value(slots, "cap_slots", "{}"));
    const std::string model_slots = trim(extract_json_value(slots, "model_slots", "{}"));
    const int slot_in_flight = extract_json_int_value(slots, "slot_total_in_flight", static_cast<int>(snapshot.processing));
    const int llm_queue_capacity = extract_json_int_value(slots, "llm_queue_capacity", static_cast<int>(snapshot.max_concurrent_requests));

    auto capabilities = native_worker_capabilities();
    if (backend_->loaded_vision_model()) worker_vision_seen_ = true;
    if (worker_vision_seen_ && std::find(capabilities.begin(), capabilities.end(), "vision") == capabilities.end())
        capabilities.push_back("vision");
    const std::string gpus_json = native_worker_gpus_json();

    std::map<std::string, std::string> values;
    // Use the same visible-device accounting as admission and /v1/resources.
    // nvidia-smi lists physical devices even when CUDA_VISIBLE_DEVICES hides
    // them, and the legacy summary also rounded each card down to whole GiB.
    values["free_vram_gb"] = extract_json_value(resources_body, "free_vram_gb", "0.0");
    values["total_vram_gb"] = extract_json_value(resources_body, "total_vram_gb", "0.0");
    values["free_ram_gb"] = json_number_literal(memory.free_ram_gb);
    values["total_ram_gb"] = json_number_literal(memory.total_ram_gb);
    values["queue_depth"] = std::to_string(static_cast<int>(snapshot.queued + snapshot.processing));
    values["queue_capacity"] = std::to_string(std::max(1, llm_queue_capacity));
    values["in_flight"] = std::to_string(std::max(0, slot_in_flight));
    values["cap_slots"] = cap_slots.empty() ? "{}" : cap_slots;
    values["model_slots"] = model_slots.empty() ? "{}" : model_slots;
    values["models"] = string_array_json(public_models);
    values["capabilities"] = string_array_json(capabilities);
    values["gpus"] = gpus_json;
    values["best_tier"] = std::to_string(best_gpu_tier_from_json(gpus_json));
    values["model_context"] = native_worker_model_context_json(llm_catalog_, *backend_);
    values["model_quant"] = native_worker_model_quant_json(*backend_);
    values["cap_models"] = native_worker_cap_models_json(capabilities);
    values["extra"] = "{}";
    try {
        const auto lifecycle = json_object_field(resources_body, "model_lifecycle", "{}");
        const auto loaded_models = json_object_field(resources_body, "loaded_models", "{}");
        const bool llm_resident = json_object_field(loaded_models, "llm", "null") != "null"
            || json_object_field(loaded_models, "vision_llm", "null") != "null"
            || !json_array_elements(json_object_field(lifecycle, "loaded_llms", "[]")).empty();
        const bool image_resident = json_object_field(loaded_models, "img", "null") != "null";
        const auto secondary = [] { return secondary_cuda_device(visible_cuda_devices()); };
        std::vector<std::string> llm_dependencies;
        if (image_should_unload_llm(llm_resident, image_resident, secondary, [this] { return resources_->get_total_free_vram(); }, [this] { return !image_client().url.empty(); })) llm_dependencies.push_back("image");
        const bool video_dependency = video_should_unload_llm(llm_resident, secondary, [this] { return !image_client().url.empty(); });
        if (video_dependency) llm_dependencies.push_back("video");
        if (music_should_unload_llm()) llm_dependencies.push_back("music");
        for (const auto &capability : {"tts", "stt"}) {
            if (std::find(capabilities.begin(), capabilities.end(), capability) != capabilities.end()
                && (std::string(capability) == "tts" ? native_tts_handoff() : native_stt_handoff()))
                llm_dependencies.push_back(capability);
        }
        if (video_dependency && std::find(llm_dependencies.begin(), llm_dependencies.end(), "music") != llm_dependencies.end()) llm_dependencies.push_back("music_video");
        // The lifecycle snapshot already resolves internal replica IDs to their
        // public source names. Preserve duplicate resident replicas and their order.
        values["extra"] = json_object_from_ordered_entries({
            {"llm_model_residency", json_object_field(lifecycle, "llm_model_residency", "\"resident\"")},
            {"loaded_llms", json_object_field(lifecycle, "loaded_llms", "[]")},
            {"active_llm", json_object_field(lifecycle, "active_llm", "null")},
            {"llm_unload_dependent_capabilities", string_array_json(llm_dependencies)}
        });
    } catch (const std::exception &) {
        // Python leaves extra empty when a media-policy probe fails. The
        // worker must still register and send its remaining state.
    }
    values["version"] = "\"" + json_escape(runtime_version()) + "\"";
    return json_object_from_values(values);
}

std::string NativeServer::worker_registration_body() const {
    auto values = top_level_json_object_values(worker_state_json());
    values["worker_id"] = "\"" + json_escape(worker_heartbeat_id_) + "\"";
    values["label"] = "\"" + json_escape(worker_heartbeat_label_) + "\"";
    values["url"] = "\"" + json_escape(worker_heartbeat_url_) + "\"";
    values["port"] = std::to_string(worker_heartbeat_port_);
    values["api_key"] = "\"" + json_escape(config_.api_key) + "\"";
    return json_object_from_values(values);
}

std::string NativeServer::worker_heartbeat_body() const {
    auto values = top_level_json_object_values(worker_state_json());
    values["worker_id"] = "\"" + json_escape(worker_heartbeat_id_) + "\"";
    return json_object_from_values(values);
}

void NativeServer::worker_heartbeat_loop() {
    WorkerHeartbeatOperations operations;
    operations.stopped = [this] { return !worker_heartbeat_running_.load(); };
    operations.wait = [this](double seconds) { worker_heartbeat_signal_.wait(seconds, false); };
    operations.wait_for_update = [this](double seconds) { worker_heartbeat_signal_.wait(seconds, true); };
    operations.post = [this](const std::string &action) {
        if (action != "deregister") worker_heartbeat_signal_.begin_update();
        std::string body;
        if (action == "register") {
            // Python's _gather_state guards each probe, so registration never
            // stops for one failed probe. Retry with backoff instead of
            // leaving the worker permanently unregistered.
            try { body = worker_registration_body(); }
            catch (const std::exception &error) {
                std::cerr << "[Heartbeat] Registration state collection failed: " << error.what() << "\n";
                return false;
            }
        } else {
            body = action == "heartbeat" ? worker_heartbeat_body()
                : "{\"worker_id\":\"" + json_escape(worker_heartbeat_id_) + "\"}";
        }
        return worker_router_post("/v1/router/" + action, body);
    };
    try { run_worker_heartbeat(worker_heartbeat_interval_, operations); }
    catch (...) { std::cerr << "[Heartbeat] Registration loop stopped after a state collection failure\n"; }
    {
        std::lock_guard<std::mutex> lock(worker_heartbeat_mutex_);
        worker_heartbeat_done_ = true;
    }
    worker_heartbeat_cv_.notify_all();
}

void NativeServer::notify_worker_state_changed() const {
    if (worker_heartbeat_running_.load()) worker_heartbeat_signal_.notify();
}

void NativeServer::start_worker_tunnel(int port) {
    if (worker_tunnel_running_) {
        return;
    }
    if (!env_true("WORKER_TUNNEL") || worker_heartbeat_router_url_.empty() || worker_heartbeat_id_.empty()) {
        return;
    }
    std::string base = strip_trailing_slashes(worker_heartbeat_router_url_);
    if (starts_with(base, "https://")) {
        worker_tunnel_ws_url_ = "wss://" + base.substr(std::string("https://").size()) + "/v1/router/tunnel";
    } else if (starts_with(base, "http://")) {
        worker_tunnel_ws_url_ = "ws://" + base.substr(std::string("http://").size()) + "/v1/router/tunnel";
    } else {
        worker_tunnel_ws_url_ = base + "/v1/router/tunnel";
    }
    worker_tunnel_local_url_ = strip_trailing_slashes(trim(ezl_getenv_default("EZLOCALAI_URL")));
    if (worker_tunnel_local_url_.empty()) {
        worker_tunnel_local_url_ = "http://127.0.0.1:" + std::to_string(port > 0 ? port : worker_heartbeat_port_);
    }
    worker_tunnel_running_ = true;
    worker_tunnel_thread_ = std::thread(&NativeServer::worker_tunnel_loop, this);
}

void NativeServer::stop_worker_tunnel() {
    if (!worker_tunnel_running_ && !worker_tunnel_thread_.joinable()) {
        return;
    }
    worker_tunnel_running_ = false;
    worker_heartbeat_cv_.notify_all();
    if (worker_tunnel_thread_.joinable()) {
        worker_tunnel_thread_.join();
    }
}

void NativeServer::worker_tunnel_loop() {
    struct RequestTask {
        std::atomic<bool> done{false};
        std::atomic<bool> cancelled{false};
        std::mutex mutex;
        std::function<void()> interrupt;
        std::thread thread;
        void cancel() {
            std::lock_guard<std::mutex> lock(mutex);
            cancelled = true;
            if (interrupt) interrupt();
        }
    };
    TunnelReconnectBackoff backoff;
    auto reconnect_wait = [&](double duration, bool failed) {
        const auto backoff_seconds = backoff.disconnected(duration, failed);
        std::unique_lock<std::mutex> lock(worker_heartbeat_mutex_);
        worker_heartbeat_cv_.wait_for(lock, std::chrono::milliseconds(static_cast<int>(backoff_seconds * 1000)),
            [&] { return !worker_tunnel_running_.load(); });
    };
    while (worker_tunnel_running_) {
        const auto connected_at = std::chrono::system_clock::now();
        TunnelWebSocketConnection connection([this] { return !worker_tunnel_running_.load(); });
        std::string frame_buffer, error;
        if (!connection.connect_to(worker_tunnel_ws_url_, worker_heartbeat_id_, worker_heartbeat_api_key_, &frame_buffer, &error)) {
            connection.close_all();
            reconnect_wait(std::chrono::duration<double>(std::chrono::system_clock::now() - connected_at).count(), true);
            continue;
        }
        // Only this thread accesses the WebSocket (including its SSL object).
        // Inference tasks publish into a bounded queue so slow routers exert
        // backpressure without blocking reception of pings and cancellations.
        std::mutex outgoing_mutex;
        std::condition_variable outgoing_cv;
        std::deque<std::pair<std::shared_ptr<RequestTask>, std::string>> outgoing;
        bool closed = false;
        std::map<std::string, std::shared_ptr<RequestTask>> tasks;
        auto last_message = std::chrono::steady_clock::now();
        bool connected = true, close_sent = false, connection_failed = false;
        int close_code = 1000;
        TunnelWebSocketReader reader;
        std::deque<std::pair<unsigned char, std::string>> controls;
        bool writing_control = false;
        char buffer[4096];
        while (worker_tunnel_running_ && connected) {
            for (auto it = tasks.begin(); it != tasks.end();) {
                if (it->second->done && it->second.use_count() == 1) {
                    it->second->thread.join();
                    it = tasks.erase(it);
                } else ++it;
            }
            // Keep receiving cancellations and keepalives while the peer's
            // receive window is full. A partial frame owns its stable buffer
            // until the final byte (including across SSL WANT_WRITE retries).
            const auto write_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
            do {
                if (!connection.writing()) {
                    if (!controls.empty()) {
                        connection.queue_frame(controls.front().first, controls.front().second);
                        controls.pop_front(); writing_control = true;
                    } else {
                        std::lock_guard<std::mutex> lock(outgoing_mutex);
                        while (!outgoing.empty() && outgoing.front().first->cancelled) outgoing.pop_front();
                        if (outgoing.empty()) { outgoing_cv.notify_all(); break; }
                        connection.queue_frame(0x1U, outgoing.front().second);
                        outgoing.pop_front(); writing_control = false;
                    }
                    outgoing_cv.notify_all();
                }
                const auto status = connection.flush_frame();
                if (status < 0) { connected = false; connection_failed = writing_control; break; }
                if (!status) break;
            } while (std::chrono::steady_clock::now() < write_until);
            if (!connected) break;
            TunnelWebSocketMessage frame;
            while (controls.size() < 64 && reader.pop(frame_buffer, frame)) {
                if (frame.opcode == 0x8U) { connected = false; break; }
                if (frame.opcode == 0x9U) {
                    controls.emplace_back(0xAU, std::move(frame.payload));
                    continue;
                }
                // aiohttp autoping consumes control frames inside receive(). Only
                // complete data messages restart TunnelClient's outer idle wait.
                if (frame.opcode == 0x1U || frame.opcode == 0x2U) last_message = std::chrono::steady_clock::now();
                if (frame.opcode != 0x1U) continue;
                const auto type = extract_json_string(frame.payload, "t", "");
                const auto id = extract_json_string(frame.payload, "id", "");
                if (type == "ping") {
                    controls.emplace_back(0x1U, "{\"t\":\"pong\"}");
                } else if (type == "cancel") {
                    const auto task = tasks.find(id);
                    if (task != tasks.end()) task->second->cancel();
                    outgoing_cv.notify_all();
                } else if (type == "req" && !id.empty() && !tasks.count(id)) {
                    auto task = std::make_shared<RequestTask>();
                    tasks.emplace(id, task);
                    task->thread = std::thread([&, task, message = frame.payload] {
                        const auto send_text = [&](const std::string &payload) {
                            std::unique_lock<std::mutex> lock(outgoing_mutex);
                            outgoing_cv.wait(lock, [&] { return closed || task->cancelled || outgoing.size() < 64; });
                            if (closed || task->cancelled) return false;
                            outgoing.emplace_back(task, payload);
                            return true;
                        };
                        const auto register_interrupt = [&](std::function<void()> interrupt) {
                            std::lock_guard<std::mutex> lock(task->mutex);
                            task->interrupt = std::move(interrupt);
                            if (task->cancelled && task->interrupt) task->interrupt();
                        };
                        try {
                            worker_tunnel_handle_message(message, send_text, register_interrupt);
                        } catch (const std::exception &e) {
                            send_text("{\"t\":\"resp_err\",\"id\":\"" + json_escape(extract_json_string(message, "id", ""))
                                + "\",\"error\":\"" + json_escape(e.what()) + "\"}");
                        }
                        // The interrupt closure owns the local HTTP stream.
                        // Release it on this task's thread after I/O ends, even
                        // if queued responses still keep RequestTask alive.
                        // shutdown() alone can leave a large peer write blocked
                        // until close() releases an unread receive buffer.
                        {
                            std::lock_guard<std::mutex> lock(task->mutex);
                            task->interrupt = {};
                        }
                        task->done = true;
                    });
                }
            }
            if (reader.error_code()) { close_code = reader.error_code(); connected = false; }
            if (!connected) { connection.send_close(close_code); close_sent = true; break; }
            if (std::chrono::steady_clock::now() - last_message >= std::chrono::seconds(60)) { connection_failed = true; break; }
            // Stop admitting more controls under backpressure, while retaining
            // the idle deadline and checking shutdown on every write slice.
            if (controls.size() >= 64) continue;
            if (!connection.readable(connection.writing() || !controls.empty() ? 0 : 20)) continue;
            const ssize_t n = connection.read_some(buffer, sizeof(buffer));
            if (n < 0) continue;
            if (n == 0) break;
            frame_buffer.append(buffer, static_cast<size_t>(n));
        }
        {
            std::lock_guard<std::mutex> lock(outgoing_mutex);
            closed = true;
        }
        outgoing_cv.notify_all();
        for (const auto &entry : tasks) entry.second->cancel();
        for (const auto &entry : tasks) entry.second->thread.join();
        if (!close_sent) connection.send_close(close_code);
        connection.close_all();
        if (worker_tunnel_running_) reconnect_wait(std::chrono::duration<double>(std::chrono::system_clock::now() - connected_at).count(), connection_failed);
    }
}

void NativeServer::worker_tunnel_handle_message(
    const std::string &message,
    const std::function<bool(const std::string &)> &send_text,
    const LlmInferenceStream::Interrupt &register_interrupt
) {
    const std::string type = extract_json_string(message, "t", "");
    if (type == "ping") {
        send_text("{\"t\":\"pong\"}");
        return;
    }
    if (type != "req") {
        return;
    }
    const std::string request_id = extract_json_string(message, "id", "");
    if (request_id.empty()) {
        return;
    }
    std::string method = extract_json_string(message, "method", "GET");
    std::transform(method.begin(), method.end(), method.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    std::string path = extract_json_string(message, "path", "/");
    if (path.empty()) {
        path = "/";
    }
    std::string content_type;
    std::map<std::string, std::string> headers;
    const std::string headers_raw = trim(extract_json_value(message, "headers", "{}"));
    for (const auto &entry : json_object_ordered_entries(headers_raw)) {
        const auto raw_value = trim(entry.second);
        if (!raw_value.empty() && raw_value.front() == '"') {
            headers[entry.first] = parse_json_string_at(raw_value, 0);
            if (to_lower(entry.first) == "content-type") content_type = headers[entry.first];
        }
    }
    const std::string body = base64_decode_string(extract_json_string(message, "body_b64", ""));
    // A fixed ten-minute local timeout can abandon media generation while
    // WorkerRouter is still waiting for it. Retain the ordinary default, but
    // allow the existing request and generation budgets to extend it.
    int timeout_seconds = 600;
    const auto extend_timeout = [&](const char *name, double fallback) {
        double seconds = fallback;
        const char *raw = getenv(name);
        if (raw && (!parse_strict_python_float_string(raw, &seconds)
                    || !std::isfinite(seconds) || seconds <= 0))
            seconds = fallback;
        timeout_seconds = std::max(timeout_seconds, static_cast<int>(
            std::min(std::ceil(seconds), static_cast<double>(std::numeric_limits<int>::max()))));
    };
    extend_timeout("REQUEST_TIMEOUT", 600);
    const auto route = path.substr(0, path.find('?'));
    const bool music_video = route == "/v1/videos/music" || route == "/v1/videos/music/generations";
    if (route == "/v1/videos/generations" || music_video)
        extend_timeout("VIDEO_GENERATION_TIMEOUT", 3600);
    if (route == "/v1/audio/music" || route == "/v1/audio/music/generations" || music_video)
        extend_timeout("ACE_STEP_TIMEOUT", 1800);
    std::shared_ptr<WorkerStreamState> state;
    const auto response = http_worker_stream_request(
        method, worker_tunnel_local_url_, path, "", content_type, body, timeout_seconds, &state, register_interrupt, headers);

    if (response.transport_error) {
        send_text(
            "{\"t\":\"resp_err\",\"id\":\"" + json_escape(request_id)
            + "\",\"error\":\"" + json_escape(extract_json_string(response.body, "detail", "Worker request failed")) + "\"}"
        );
        return;
    }
    std::map<std::string, std::string> response_headers;
    for (const auto &header : response.headers) response_headers[header.first] = "\"" + json_escape(header.second) + "\"";
    if (!send_text("{\"t\":\"resp_start\",\"id\":\"" + json_escape(request_id)
        + "\",\"status\":" + std::to_string(response.status) + ",\"headers\":"
        + json_object_from_values(response_headers) + ",\"stream\":true}")) return;

    auto send_chunk = [&](const std::string &chunk) {
        if (chunk.empty()) {
            return true;
        }
        const std::string encoded = base64_encode(
            reinterpret_cast<const unsigned char *>(chunk.data()),
            chunk.size()
        );
        return send_text(
            "{\"t\":\"resp_chunk\",\"id\":\"" + json_escape(request_id)
            + "\",\"data_b64\":\"" + encoded + "\"}"
        );
    };

    StreamingChunkedDecoder decoder;
    bool done = method == "HEAD" || response.status == 204 || response.status == 304;
    size_t received = 0;
    const auto consume = [&](const std::string &bytes) {
        if (state->upstream_chunked) {
            std::vector<std::string> chunks;
            if (!decoder.feed(bytes, &chunks, &done)) throw std::runtime_error("Malformed local response chunks");
            for (const auto &chunk : chunks) if (!send_chunk(chunk)) return false;
            return true;
        }
        const size_t count = state->content_length ? std::min(bytes.size(), *state->content_length - received) : bytes.size();
        received += count;
        if (state->content_length && received == *state->content_length) done = true;
        return send_chunk(bytes.substr(0, count));
    };
    if (!done && !consume(state->initial_body)) return;
    char buffer[4096];
    while (!done) {
        bool timed_out = false;
        const auto n = state->read_some(buffer, sizeof(buffer), &timed_out);
        if (n <= 0) {
            if (timed_out) throw std::runtime_error("Local response read timed out");
            if (state->upstream_chunked || (state->content_length && received != *state->content_length))
                throw std::runtime_error("Incomplete local response body");
            break;
        }
        if (!consume(std::string(buffer, static_cast<size_t>(n)))) return;
    }

    send_text("{\"t\":\"resp_end\",\"id\":\"" + json_escape(request_id) + "\"}");
}

bool NativeServer::authorized(const std::string &authorization) const {
    const std::string api_key = config_.api_key;
    if (api_key.empty() || api_key == "none") {
        return true;
    }
    if (authorization.empty()) {
        return false;
    }
    if (to_lower(trim(ezl_getenv_default("ROUTER_MODE"))) != "true") {
        std::string token = authorization;
        if (to_lower(authorization).find("bearer ") != std::string::npos) {
            const auto space = authorization.find(' ');
            token = space == std::string::npos ? "" : authorization.substr(space + 1);
        }
        return token == api_key;
    }
    std::istringstream parts(authorization);
    std::string first;
    std::string second;
    parts >> first >> second;
    const std::string token = second.empty() ? first : second;
    return token == api_key;
}

bool NativeServer::authorized_for_worker(const std::string &authorization) const {
    std::string expected = trim(ezl_getenv_default("ROUTER_REGISTER_KEY"));
    if (expected.empty()) {
        expected = trim(config_.api_key);
    }
    if (expected.empty() || expected == "none") {
        return true;
    }
    if (authorization.empty()) {
        return false;
    }
    std::istringstream parts(authorization);
    std::string first;
    std::string second;
    parts >> first >> second;
    const std::string token = second.empty() ? first : second;
    return token == expected;
}

bool NativeServer::authorized_for_worker_websocket(const std::string &authorization) const {
    std::string expected = trim(ezl_getenv_default("ROUTER_REGISTER_KEY"));
    if (expected.empty()) {
        expected = trim(config_.api_key);
    }
    if (expected.empty() || expected == "none") {
        return true;
    }
    const std::string lowered = to_lower(authorization);
    if (!starts_with(lowered, "bearer ")) {
        return false;
    }
    return trim(authorization.substr(std::string("bearer ").size())) == expected;
}

NativeServer::ScopedCapabilityUse::ScopedCapabilityUse(const NativeServer *server_in, std::string capability_in)
    : server(server_in), capability(std::move(capability_in)) {
    if (server) {
        server->resource_increment_capability_in_flight(capability, 1);
        if (capability == "video") server->resources_->mark_model_in_use(ModelType::Video, true);
    }
}

NativeServer::ScopedCapabilityUse::~ScopedCapabilityUse() {
    if (server) {
        if (capability == "video") server->resources_->mark_model_in_use(ModelType::Video, false);
        server->resource_increment_capability_in_flight(capability, -1);
    }
}

HttpResponse NativeServer::method_not_allowed_or_not_found(const std::string &path) const {
    const auto query_pos = path.find('?');
    const std::string route_path = query_pos == std::string::npos ? path : path.substr(0, query_pos);
    const auto allow = allowed_methods_for_route_path(starts_with(route_path, "/v1/wakeword/") ? url_decode(route_path) : route_path, native_router_mode_enabled());
    if (allow.empty()) {
        return {404, "application/json", "{\"detail\":\"Not Found\"}"};
    }
    HttpResponse response{405, "application/json", "{\"detail\":\"Method Not Allowed\"}"};
    if (!starts_with(route_path, "/outputs/")) {
        response.headers.push_back({"Allow", join_strings(allow, ", ")});
    }
    return response;
}

HttpResponse NativeServer::predict_wakeword(const std::string &word, const std::string &directory, const std::string &audio) {
    std::pair<bool, double> prediction;
    try {
        std::shared_ptr<WakewordModel> model;
        {
            std::lock_guard<std::mutex> lock(wakeword_models_mutex_);
            const auto key = llm_unicode_lower(word);
            const auto found = wakeword_loaded_models_.find(key);
            if (found != wakeword_loaded_models_.end()) model = found->second;
            else {
                if (!wakeword_factory_) throw std::runtime_error("Wake-word Torch adapter unavailable");
                model = wakeword_factory_(directory);
                if (!model) throw std::runtime_error("Cannot load wake-word model");
                wakeword_loaded_models_.emplace(key, model);
            }
        }
        prediction = model->predict(audio);
    } catch (...) { return {500, "application/json", "{\"detail\":\"Prediction failed\"}"}; }
    if (!std::isfinite(prediction.second)) return plain_internal_server_error();
    nlohmann::ordered_json response{{"detected", prediction.first}, {"confidence", prediction.second}, {"word", word}};
    return {200, "application/json", response.dump()};
}

HttpResponse NativeServer::handle_request(
    const std::string &method,
    const std::string &path,
    const std::string &authorization,
    const std::string &body,
    const std::string &content_type,
    const std::string &client_host,
    const std::string &host_header,
    const std::unordered_map<std::string, std::string> &request_headers,
    bool allow_deferred_streaming
) {
    const auto query_pos = path.find('?');
    const std::string route_path = query_pos == std::string::npos ? path : path.substr(0, query_pos);
    if (method == "OPTIONS") {
        return method_not_allowed_or_not_found(path);
    }
    const auto allowed_methods = allowed_methods_for_route_path(starts_with(route_path, "/v1/wakeword/") ? url_decode(route_path) : route_path, native_router_mode_enabled());
    if (allowed_methods.empty()) {
        return method_not_allowed_or_not_found(path);
    }
    if (std::find(allowed_methods.begin(), allowed_methods.end(), method) == allowed_methods.end()) {
        return method_not_allowed_or_not_found(path);
    }
    const bool docs_method = method == "GET" || method == "HEAD";
    if (docs_method && route_path == "/") {
        return docs_html();
    }
    if (docs_method && route_path == "/openapi.json") {
        return openapi_json();
    }
    if (docs_method && route_path == "/docs/oauth2-redirect") {
        return swagger_oauth2_redirect_html();
    }
    if (docs_method && route_path == "/redoc") {
        return redoc_html();
    }
    if (route_path == "/health") {
        if (method != "GET") {
            return method_not_allowed_or_not_found(path);
        }
        if (!ezl_queue_is_running(queue_)) {
            return {503, "application/json", "{\"detail\":\"Queue processor not running\"}"};
        }
        return {200, "application/json", "{\"status\":\"healthy\"}"};
    }
    if ((method == "GET" || method == "HEAD") && starts_with(path, "/outputs/")) {
        return serve_output_file(query_pos == std::string::npos ? path : path.substr(0, query_pos), request_headers, allow_deferred_streaming);
    }
    if (method == "GET" && route_path == "/v1/router/health") {
        if (!ensure_router_drains_loaded()) return plain_internal_server_error();
        return router_health();
    }
    if (method == "GET" && route_path == "/dashboard") {
        if (!ensure_router_drains_loaded()) return plain_internal_server_error();
        return router_dashboard_html();
    }
    if ((method == "POST" && route_path == "/v1/router/register")
        || (method == "POST" && route_path == "/v1/router/heartbeat")
        || (method == "POST" && route_path == "/v1/router/deregister")
        || (method == "POST" && route_path == "/v1/router/drain")
        || (method == "POST" && route_path == "/v1/router/resume")) {
        if (!authorized_for_worker(authorization)) {
            if (authorization.empty()) {
                return {401, "application/json", "{\"detail\":\"Missing Authorization header\"}"};
            }
            return {401, "application/json", "{\"detail\":\"Invalid API key\"}"};
        }
        if (!is_valid_top_level_json_object(body)) return invalid_json_object_response(trim(body) == "null" ? "" : body);
        if (route_path == "/v1/router/drain" || route_path == "/v1/router/resume") {
            return router_drain(body, route_path == "/v1/router/resume");
        }
        if (!ensure_router_drains_loaded()) return plain_internal_server_error();
        if (route_path == "/v1/router/register") {
            return router_register(body, client_host);
        }
        if (route_path == "/v1/router/heartbeat") {
            return router_heartbeat(body);
        }
        return router_deregister(body);
    }
    if (!authorized(authorization)) {
        if (to_lower(trim(ezl_getenv_default("ROUTER_MODE"))) == "true") {
            if (authorization.empty()) {
                return {401, "application/json", "{\"detail\":\"Missing Authorization header\"}"};
            }
            return {401, "application/json", "{\"detail\":\"Invalid API key\"}"};
        }
        return {401, "application/json", "{\"detail\":\"Invalid API Key\"}"};
    }
    if (native_router_mode_enabled() && !ensure_router_drains_loaded()) return plain_internal_server_error();
    return route_authorized(method, path, body, content_type, request_base_url_from_host(host_header), request_headers, allow_deferred_streaming);
}

HttpResponse NativeServer::route_authorized(
    const std::string &method,
    const std::string &path,
    const std::string &body,
    const std::string &content_type,
    const std::string &request_base_url,
    const std::unordered_map<std::string, std::string> &request_headers,
    bool allow_deferred_streaming
) {
    const auto query_pos = path.find('?');
    const std::string route_path = query_pos == std::string::npos ? path : path.substr(0, query_pos);
    const std::string query = query_pos == std::string::npos ? "" : path.substr(query_pos + 1);
    const bool router_mode_enabled = native_router_mode_enabled();

    if (method == "GET" && route_path == "/v1/models") {
        return models();
    }
    if (method == "GET" && route_path == "/v1/resources") {
        try { return resources(); }
        catch (const std::exception &) { return {500, "text/plain; charset=utf-8", "Internal Server Error"}; }
    }
    if (method == "GET" && route_path == "/v1/router/workers") {
        return router_workers();
    }
    if (method == "GET" && route_path == "/v1/router/dashboard") {
        return router_dashboard();
    }
    if (method == "GET" && route_path == "/v1/router/tunnels") {
        return router_tunnels();
    }
    if (method == "GET" && route_path == "/v1/router/usage") {
        return router_usage();
    }
    if (method == "GET" && route_path == "/v1/router/history") {
        int limit = 200;
        if (query_param_present(query, "limit")) {
            const std::string raw_limit = query_param_value(query, "limit", "");
            if (!parse_pydantic_integral_string(raw_limit, &limit)) {
                return pydantic_query_int_error("limit", raw_limit);
            }
        }
        return router_history(limit);
    }
    if (method == "GET" && route_path == "/v1/router/errors") {
        return router_errors();
    }
    if (method == "GET" && route_path == "/v1/audio/voices") {
        if (router_mode_enabled) {
            const double now = monotonic_epoch_seconds();
            const double ttl = router_ttl_seconds();
            std::vector<RouterWorker> tts_workers;
            {
                std::lock_guard<std::mutex> lock(router_mutex_);
                for (const auto &worker : router_workers_in_registration_order_locked()) {
                    if (now - worker.last_heartbeat <= ttl && vector_contains(worker.capabilities, "tts")) {
                        tts_workers.push_back(worker);
                    }
                }
            }
            std::set<std::string> seen_voice_keys;
            bool seen_none_voice_key = false;
            std::vector<std::string> voices;
            for (const auto &worker : tts_workers) {
	                const std::string api_key = worker.api_key.empty() ? config_.api_key : worker.api_key;
	                auto response = is_tunnel_url(worker.url)
	                    ? router_tunnel_request(worker, "GET", "/v1/audio/voices", api_key, "", "", false)
	                    : http_json_request("GET", worker.url, "/v1/audio/voices", api_key, "");
	                if (response.transport_error) {
	                    std::string message = extract_json_string(response.body, "detail", "");
	                    if (message.empty()) {
	                        message = response.body.empty() ? "Worker request failed" : response.body;
	                    }
	                    if (is_tunnel_url(worker.url)) {
	                        const std::string tunnel_id = tunnel_worker_id(worker.url);
	                        const std::string lower_message = to_lower(message);
	                        if (lower_message.find("not connected") != std::string::npos) {
	                            const std::string record_message = tunnel_id.empty()
	                                ? message
	                                : "Tunnel " + tunnel_id + " not connected";
	                            router_record_error(
	                                worker.worker_id,
	                                "tunnel_disconnect",
	                                "/v1/audio/voices",
	                                record_message,
	                                response.status
	                            );
	                        }
	                    } else {
	                        {
	                            std::lock_guard<std::mutex> lock(router_mutex_);
	                            auto found = router_workers_.find(worker.worker_id);
	                            if (found != router_workers_.end()) {
	                                found->second.connection_failures += 1;
	                            }
	                        }
		                        router_record_error(
		                            worker.worker_id,
		                            "connection",
		                            "/v1/audio/voices",
		                            message,
		                            0
		                        );
	                    }
	                    continue;
	                }
	                if (response.status < 200 || response.status >= 300) {
	                    continue;
	                }
                bool iterable_ok = true;
                auto elements = python_iterable_json_values(extract_json_value(response.body, "voices", ""), &iterable_ok);
                if (!iterable_ok) {
                    continue;
                }
                if (elements.empty()) {
                    elements = python_iterable_json_values(extract_json_value(response.body, "data", ""), &iterable_ok);
                    if (!iterable_ok) {
                        continue;
                    }
                }
                for (const auto &element : elements) {
                    std::string key = element;
                    bool key_is_none = false;
                    const std::string trimmed = trim(element);
                    if (!trimmed.empty() && trimmed.front() == '"') {
                        key = parse_json_string_at(trimmed, 0);
                    } else if (!trimmed.empty() && trimmed.front() == '[') {
                        break;
                    } else if (!trimmed.empty() && trimmed.front() == '{') {
                        const std::string raw_id = trim(extract_json_value(trimmed, "id", ""));
                        if (raw_id.empty() || raw_id == "null") {
                            key_is_none = true;
                        } else if (raw_id.front() == '"') {
                            key = parse_json_string_at(raw_id, 0);
                        } else if (raw_id.front() == '[' || raw_id.front() == '{') {
                            break;
                        } else {
                            key = python_repr_json_value(raw_id);
                        }
                    }
                    if (key_is_none) {
                        if (!seen_none_voice_key) {
                            seen_none_voice_key = true;
                            voices.push_back(element);
                        }
                    } else if (seen_voice_keys.insert(key).second) {
                        voices.push_back(element);
                    }
                }
            }
            std::ostringstream out;
            out << "{\"voices\":[";
            for (size_t i = 0; i < voices.size(); ++i) {
                if (i) {
                    out << ",";
                }
                out << voices[i];
            }
            out << "]}";
            return {200, "application/json", out.str()};
        }
        return audio_voices();
    }
    if (method == "POST" && route_path == "/v1/audio/voices") {
        return upload_voice(body, content_type);
    }
    if (method == "GET" && route_path == "/v1/fallback/status") {
        try { return fallback_status(); }
        catch (const std::exception &) { return {500, "text/plain; charset=utf-8", "Internal Server Error"}; }
    }
    if (method == "GET" && (route_path == "/v1/queue" || route_path == "/v1/queue/status")) {
        return queue_status();
    }
    if (method == "GET" && matches_template(route_path, "/v1/queue/request/{request_id}")) {
        const std::string prefix = "/v1/queue/request/";
        const std::string request_id = route_path.substr(prefix.size());
        ezl_request_info info{};
        int rc = ezl_queue_get_request_info(queue_, request_id.c_str(), &info);
        if (rc == ENOENT) {
            return {404, "application/json", "{\"detail\":\"Request " + json_escape(request_id) + " not found\"}"};
        }
        if (rc != 0) {
            return {500, "application/json", make_error(500, "failed to get request status")};
        }
        std::ostringstream out;
        out << "{"
            << "\"id\":\"" << json_escape(info.id) << "\","
            << "\"status\":\"" << ezl_request_status_name(info.status) << "\","
            << "\"completion_type\":\"" << json_escape(info.completion_type) << "\","
            << "\"timestamp\":" << std::setprecision(std::numeric_limits<double>::max_digits10) << info.timestamp << ",";
        if (info.error[0]) {
            out << "\"error\":\"" << json_escape(info.error) << "\"";
        } else {
            out << "\"error\":null";
        }
        out << "}";
        return {200, "application/json", out.str()};
    }
    if (method == "POST" && route_path == "/v1/chat/completions") {
        if (router_mode_enabled) {
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_object_response(body);
            }
            const bool needs_vision = router_chat_payload_needs_vision(body);
            return router_proxy_request(
                needs_vision ? "vision" : "text",
                router_payload_model_or_unknown(body),
                route_path,
                body,
                "application/json",
                "",
                0,
                allow_deferred_streaming
            );
        }
        if (!is_valid_top_level_json_object(body)) {
            return invalid_json_model_response(body);
        }
        {
            const auto request_values = top_level_json_object_values(body);
            std::vector<std::string> validation_errors;
            validate_pydantic_default_string_field(request_values, "model", &validation_errors);
            validate_pydantic_optional_list_of_dicts_field(request_values, "messages", &validation_errors);
            validate_pydantic_optional_float_field(request_values, "temperature", &validation_errors);
            validate_pydantic_optional_float_field(request_values, "top_p", &validation_errors);
            validate_pydantic_optional_list_of_dicts_field(request_values, "functions", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "function_call", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "n", &validation_errors);
            validate_pydantic_optional_bool_field(request_values, "stream", &validation_errors);
            validate_pydantic_optional_list_of_strings_field(request_values, "stop", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "max_tokens", &validation_errors);
            validate_pydantic_optional_float_field(request_values, "presence_penalty", &validation_errors);
            validate_pydantic_optional_float_field(request_values, "frequency_penalty", &validation_errors);
            validate_pydantic_optional_dict_field(request_values, "logit_bias", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "user", &validation_errors);
            validate_pydantic_optional_dict_field(request_values, "chat_template_kwargs", &validation_errors);
            if (!validation_errors.empty()) {
                return {422, "application/json", pydantic_error_response_body(validation_errors)};
            }
        }
        std::string request_body = body;
        HttpResponse forwarded;
        if (forward_text_server_request(route_path, request_body, &forwarded)) {
            return forwarded;
        }
        if (trim(ezl_getenv_default("DEFAULT_MODEL")).empty() && trim(ezl_getenv_default("VISION_MODEL")).empty()) {
            return {404, "application/json", "{\"detail\":\"Chat completions are disabled.\"}"};
        }
        return enqueue_inference("chat", request_body, allow_deferred_streaming);
    }
    if (method == "POST" && route_path == "/v1/completions") {
        if (router_mode_enabled) {
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_object_response(body);
            }
            return router_proxy_request(
                "text",
                router_payload_model_or_unknown(body),
                route_path,
                body,
                "application/json",
                "",
                0,
                allow_deferred_streaming
            );
        }
        if (!is_valid_top_level_json_object(body)) {
            return invalid_json_model_response(body);
        }
        {
            const auto request_values = top_level_json_object_values(body);
            std::vector<std::string> validation_errors;
            validate_pydantic_default_string_field(request_values, "model", &validation_errors);
            validate_pydantic_default_string_field(request_values, "prompt", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "max_tokens", &validation_errors);
            validate_pydantic_optional_float_field(request_values, "temperature", &validation_errors);
            validate_pydantic_optional_float_field(request_values, "top_p", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "n", &validation_errors);
            validate_pydantic_optional_bool_field(request_values, "stream", &validation_errors);
            validate_pydantic_optional_dict_field(request_values, "logit_bias", &validation_errors);
            validate_pydantic_optional_list_of_strings_field(request_values, "stop", &validation_errors);
            validate_pydantic_optional_bool_field(request_values, "echo", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "user", &validation_errors);
            validate_pydantic_optional_bool_field(request_values, "format_prompt", &validation_errors);
            if (!validation_errors.empty()) {
                return {422, "application/json", pydantic_error_response_body(validation_errors)};
            }
        }
        std::string request_body = body;
        HttpResponse forwarded;
        if (forward_text_server_request(route_path, request_body, &forwarded)) {
            return forwarded;
        }
        if (trim(ezl_getenv_default("DEFAULT_MODEL")).empty() && trim(ezl_getenv_default("VISION_MODEL")).empty()) {
            return {404, "application/json", "{\"detail\":\"Completions are disabled.\"}"};
        }
        return enqueue_inference("completion", request_body);
    }
    if (method == "POST" && route_path == "/v1/embeddings") {
        if (router_mode_enabled) {
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_object_response(body);
            }
            const std::string model = router_payload_model_or_empty(body);
            const std::string capability = has_alive_router_worker("embedding", model) ? "embedding" : "text";
            return router_proxy_request(capability, model, route_path, body, "application/json");
        }
        if (!is_valid_top_level_json_object(body)) {
            return invalid_json_model_response(body);
        }
        {
            const auto request_values = top_level_json_object_values(body);
            std::vector<std::string> validation_errors;
            validate_pydantic_embedding_input_field(request_values, body, &validation_errors);
            validate_pydantic_optional_string_field(request_values, "model", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "user", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "dimensions", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "encoding_format", &validation_errors);
            if (!validation_errors.empty()) {
                return {422, "application/json", pydantic_error_response_body(validation_errors)};
            }
        }
        HttpResponse forwarded;
        if (to_lower(ezl_getenv_default("EMBEDDING_ENABLED")) == "false") {
            if (!has_local_capability("embedding") && forward_embedding_server_request(body, &forwarded, true)) return forwarded;
            return {404, "application/json", "{\"detail\":\"Embeddings are disabled.\"}"};
        }
        if (forward_embedding_server_request(body, &forwarded)) {
            return forwarded;
        }
        if (forward_fallback_embeddings(body, &forwarded)) {
            return forwarded;
        }
        return local_embeddings(body);
    }
    if (method == "POST"
        && (route_path == "/v1/audio/transcriptions"
            || route_path == "/v1/audio/translations"
            || route_path == "/v1/audio/subtitles")) {
        const auto boundary = extract_multipart_boundary(content_type);
        auto fields = parse_multipart_form(body, boundary);
        // FastAPI uses a Form parameter's default for an empty scalar field.
        for (const auto *name : {"temperature","enable_diarization","num_speakers","beam_size","fast_mode"}) {
            const auto at = fields.find(name); if (at != fields.end() && at->second.empty()) fields.erase(at);
        }
        std::vector<std::string> validation_errors;
        append_multipart_file_upload_validation_errors(body, content_type, &validation_errors);
        validate_form_float_field(fields, "temperature", &validation_errors);
        if (route_path == "/v1/audio/transcriptions") {
            if (router_mode_enabled) {
                validate_form_bool_field(fields, "timestamps", &validation_errors);
            } else {
                validate_form_bool_field(fields, "enable_diarization", &validation_errors);
                validate_form_int_field(fields, "num_speakers", &validation_errors);
            }
        } else if (route_path == "/v1/audio/subtitles") {
            validate_form_int_field(fields, "beam_size", &validation_errors);
            validate_form_bool_field(fields, "fast_mode", &validation_errors);
        }
        if (!validation_errors.empty()) {
            return {422, "application/json", pydantic_error_response_body(validation_errors)};
        }
    }
    if (router_mode_enabled) {
        if (route_path == "/v1/audio/speech" || route_path == "/v1/audio/speech/stream") {
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_object_response(body);
            }
            return router_proxy_request(
                "tts",
                router_payload_model_or_empty(body),
                route_path,
                body,
                "application/json",
                route_path == "/v1/audio/speech" ? request_base_url : "",
                0,
                allow_deferred_streaming
            );
        }
        if (route_path == "/v1/audio/transcriptions") {
            std::string model = extract_json_string(body, "model", "");
            const auto boundary = extract_multipart_boundary(content_type);
            if (!boundary.empty()) {
                auto fields = parse_multipart_form(body, boundary);
                auto model_it = fields.find("model");
                if (model_it != fields.end()) {
                    model = model_it->second;
                }
            }
            const std::string payload = build_router_transcription_payload(body, content_type);
            return router_proxy_request(
                "stt",
                model,
                route_path,
                payload.empty() ? body : payload,
                payload.empty() ? (content_type.empty() ? "multipart/form-data" : content_type) : std::string("multipart/form-data; boundary=") + kVoiceForwardBoundary
            );
        }
        if (route_path == "/v1/images/generations") {
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_object_response(body);
            }
            return router_proxy_request("image", router_payload_model_or_empty(body), route_path, body, "application/json", request_base_url);
        }
        if (route_path == "/v1/images/edits") {
            std::string model = extract_json_string(body, "model", "");
            const auto boundary = extract_multipart_boundary(content_type);
            if (!boundary.empty()) {
                auto fields = parse_multipart_form(body, boundary);
                auto model_it = fields.find("model");
                if (model_it != fields.end()) {
                    model = model_it->second;
                }
            }
            const std::string payload = build_router_image_edit_payload(body, content_type);
            return router_proxy_request(
                "image",
                model,
                route_path,
                payload.empty() ? body : payload,
                payload.empty() ? (content_type.empty() ? "multipart/form-data" : content_type) : std::string("multipart/form-data; boundary=") + kVoiceForwardBoundary,
                request_base_url
            );
        }
        if (route_path == "/v1/videos/generations") {
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_object_response(body);
            }
            int timeout_seconds = 0;
            if (!router_proxy_timeout_seconds("1800", &timeout_seconds)) {
                return plain_internal_server_error();
            }
            return router_proxy_request("video", router_payload_model_or_empty(body), route_path, body, "application/json", request_base_url, timeout_seconds);
        }
        if (route_path == "/v1/audio/music" || route_path == "/v1/audio/music/generations"
            || route_path == "/v1/videos/music" || route_path == "/v1/videos/music/generations") {
            if (!is_valid_top_level_json_object(body)) return invalid_json_object_response(trim(body) == "null" ? "" : body);
            const bool video = starts_with(route_path, "/v1/videos/");
            double request_timeout = 0, ace_timeout = 0;
            if (!parse_strict_python_float_string(ezl_getenv_fallback("REQUEST_TIMEOUT", "1800"), &request_timeout)
                || !parse_strict_python_float_string(ezl_getenv_fallback("ACE_STEP_TIMEOUT", "1800"), &ace_timeout)) return plain_internal_server_error();
            const double timeout = std::max(request_timeout, ace_timeout);
            const int seconds = timeout <= 0 ? 0 : timeout >= std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : static_cast<int>(std::ceil(timeout));
            auto model = router_payload_model_or_empty(body);
            if (video) model = python_truthy_string_value_or_default(json_object_field(body, "video_model", ""), model);
            return router_proxy_request(video ? "music_video" : "music", model, video ? "/v1/videos/music" : "/v1/audio/music", body, "application/json", request_base_url, seconds);
        }
    }
    if (method == "POST" && (route_path == "/v1/audio/music" || route_path == "/v1/audio/music/generations"
        || route_path == "/v1/videos/music" || route_path == "/v1/videos/music/generations")) {
        return music_generation(body, starts_with(route_path, "/v1/videos/"));
    }
    if (method == "POST" && (route_path == "/v1/audio/speech" || route_path == "/v1/audio/speech/stream")) {
        if (body.empty() || trim(body)=="null") return invalid_json_model_response("");
        try {
            if(const auto error=json_decode_error(body))return {422,"application/json","{\"detail\":[{\"type\":\"json_invalid\",\"loc\":[\"body\","+std::to_string(error->position)+"],\"msg\":\"JSON decode error\",\"input\":{},\"ctx\":{\"error\":\""+json_escape(error->message)+"\"}}]}"};
        }catch(const std::exception &){return {400,"application/json","{\"detail\":\"There was an error parsing the body\"}"};}
        if (!is_valid_top_level_json_object(body)) {
            return invalid_json_model_response(body);
        }
        const auto request_values = top_level_json_object_values(body);
        std::vector<std::string> validation_errors;
        validate_pydantic_required_string_field(request_values, body, "input", nullptr, &validation_errors);
        validate_pydantic_optional_string_field(request_values, "model", &validation_errors);
        validate_pydantic_optional_string_field(request_values, "voice", &validation_errors);
        validate_pydantic_optional_string_field(request_values, "language", &validation_errors);
        validate_pydantic_optional_string_field(request_values, "user", &validation_errors);
        if (!validation_errors.empty()) {
            return {422, "application/json", pydantic_error_response_body(validation_errors)};
        }
    }
    std::string normalized_image_edit_body;
    if (!router_mode_enabled
        && method == "POST"
        && route_path == "/v1/images/edits"
        && !extract_multipart_boundary(content_type).empty()) {
        normalized_image_edit_body = image_edit_multipart_json_body(body, content_type);
        if (normalized_image_edit_body.empty()) {
            normalized_image_edit_body = "{}";
        }
    }
    if (method == "POST" && (route_path == "/v1/images/generations" || route_path == "/v1/images/edits" || route_path == "/v1/videos/generations")) {
        const std::string &request_body = normalized_image_edit_body.empty() ? body : normalized_image_edit_body;
        if (route_path == "/v1/videos/generations") {
            if (body.empty()) return invalid_json_model_response("");
            try {
                if (const auto error = json_decode_error(body)) return {422, "application/json", "{\"detail\":[{\"type\":\"json_invalid\",\"loc\":[\"body\","
                    + std::to_string(error->position) + "],\"msg\":\"JSON decode error\",\"input\":{},\"ctx\":{\"error\":\"" + json_escape(error->message) + "\"}}]}"};
            } catch (const std::exception &) { return {400, "application/json", "{\"detail\":\"There was an error parsing the body\"}"}; }
            if (trim(body) == "null") return invalid_json_model_response("");
        }
        if (!is_valid_top_level_json_object(request_body)) {
            return invalid_json_model_response(request_body);
        }
        const auto request_values = top_level_json_object_values(request_body);
        std::vector<std::string> validation_errors;
        if (route_path == "/v1/images/generations") {
            validate_pydantic_required_string_field(request_values, request_body, "prompt", nullptr, &validation_errors);
            validate_pydantic_optional_string_field(request_values, "model", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "n", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "size", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "quality", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "response_format", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "style", &validation_errors);
        } else if (route_path == "/v1/images/edits") {
            validate_image_edit_fields(request_body, &validation_errors);
        } else {
            validate_pydantic_optional_string_field(request_values, "prompt", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "model", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "n", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "size", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "num_frames", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "num_inference_steps", &validation_errors);
            validate_pydantic_optional_float_field(request_values, "guidance_scale", &validation_errors);
            validate_pydantic_optional_int_field(request_values, "frame_rate", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "response_format", &validation_errors);
            validate_pydantic_optional_string_field(request_values, "image", &validation_errors);
            validate_pydantic_video_conditions_field(request_values, &validation_errors);
        }
        if (!validation_errors.empty()) {
            return {422, "application/json", pydantic_error_response_body(validation_errors)};
        }
    }
    if ((route_path == "/v1/audio/transcriptions"
            || route_path == "/v1/audio/translations"
            || route_path == "/v1/audio/subtitles")
        && to_lower(trim(ezl_getenv_default("STT_ENABLED"))) == "false") {
        return {404, "application/json", "{\"detail\":\"Speech to text is disabled.\"}"};
    }
    if (method == "POST" && route_path == "/v1/audio/transcriptions") {
        HttpResponse forwarded;
        if (forward_voice_server_transcription(body, content_type, &forwarded)) {
            return forwarded;
        }
        if (forward_fallback_transcription(body, content_type, &forwarded)) {
            return forwarded;
        }
        if (local_native_transcription(body, content_type, &forwarded,false) || local_whisper_transcription(body, content_type, &forwarded)) {
            return forwarded;
        }
    }
    if (method == "POST" && route_path == "/v1/audio/translations") {
        HttpResponse translated;
        if (local_native_transcription(body, content_type, &translated,true) || local_whisper_transcription(body, content_type, &translated, true)) {
            return translated;
        }
    }
    if (method == "POST" && route_path == "/v1/audio/subtitles") {
        HttpResponse subtitles;
        if (local_native_subtitles(body, content_type, &subtitles) || local_whisper_subtitles(body, content_type, &subtitles)) {
            return subtitles;
        }
    }
    if (method == "POST" && (route_path == "/v1/audio/speech" || route_path == "/v1/audio/speech/stream")) {
        return routed_speech(body,route_path=="/v1/audio/speech/stream",allow_deferred_streaming);
    }
    if (method == "POST" && route_path == "/v1/videos/generations") return video_generation(body);
    if (method == "POST"
        && (route_path == "/v1/images/generations"
            || route_path == "/v1/images/edits"
            || route_path == "/v1/videos/generations")) {
        const std::string &request_body = normalized_image_edit_body.empty() ? body : normalized_image_edit_body;
        HttpResponse forwarded;
        if (forward_image_server_request(route_path, request_body, &forwarded)) {
            return forwarded;
        }
        if (!has_local_capability("image") && (route_path == "/v1/images/generations" || route_path == "/v1/images/edits")
            && forward_router_media_request(request_body, &forwarded, route_path)) return forwarded;
        if (forward_fallback_image_video_request(route_path, request_body, &forwarded)) {
            return forwarded;
        }
    }
    if (method == "POST" && (route_path == "/v1/images/generations" || route_path == "/v1/images/edits") && has_local_capability("image")) {
        return local_image_generation(normalized_image_edit_body.empty() ? body : normalized_image_edit_body, route_path == "/v1/images/edits");
    }
    if (route_path == "/v1/images/edits" && !has_local_capability("image")) {
        return {503, "application/json", "{\"detail\":\"No image editing service available\"}"};
    }
    if (route_path == "/v1/images/generations" && !has_local_capability("image")) {
        std::ostringstream out;
        out << "{\"created\":" << unix_seconds()
            << ",\"data\":[{\"url\":\"https://demofree.sirv.com/nope-not-here.jpg\"}]}";
        return {200, "application/json", out.str()};
    }
    if (route_path == "/v1/videos/generations") {
        if (!has_local_capability("video")) {
            std::ostringstream out;
            out << "{\"created\":" << unix_seconds()
                << ",\"data\":[{\"error\":\"Video generation not available. Set VIDEO_ENABLED=true and VIDEO_MODEL to enable.\"}]}";
            return {200, "application/json", out.str()};
        }
        return local_video_generation(body);
    }
    if (starts_with(route_path, "/v1/audio/") || starts_with(route_path, "/v1/images/") || starts_with(route_path, "/v1/videos/")) {
        return {501, "application/json", make_error(501, "this native backend does not implement the requested modality yet")};
    }
    if (starts_with(route_path, "/v1/wakeword/")) try {
        // FastAPI resolves name dependencies before body field validation and
        // endpoint enablement, but after parsing malformed JSON/form bodies.
        const auto decoded_path = url_decode(route_path);
        std::string named_word;
        bool named_route = false;
        for (const auto *pattern : {"/v1/wakeword/models/{word}", "/v1/wakeword/models/{word}/config",
                                   "/v1/wakeword/predict/{word}", "/v1/wakeword/predict/{word}/file"}) {
            if (matches_template(decoded_path, pattern)) {
                named_word = path_param_for_template(route_path, pattern);
                named_route = true;
                break;
            }
        }
        if (named_route) {
            if (method == "POST" && matches_template(decoded_path, "/v1/wakeword/predict/{word}")
                && !body.empty() && json_decode_error(body)) return invalid_json_model_response(body);
            if (method == "POST" && matches_template(decoded_path, "/v1/wakeword/predict/{word}/file")) {
                HttpResponse error;
                if (multipart_file_upload_validation_error(body, content_type, &error) && error.status == 400)
                    return error;
            }
            try { wakeword_name(named_word); }
            catch (const std::invalid_argument &) { return {400, "application/json", "{\"detail\":\"Invalid wake word name\"}"}; }
        }
        const bool wakeword_enabled = voice_client().dedicated
            || llm_unicode_lower(ezl_getenv_default("STT_ENABLED")) == "true";
        const bool wakeword_json_body_route =
            method == "POST"
            && (route_path == "/v1/wakeword/train"
                || matches_template(route_path, "/v1/wakeword/predict/{word}"));
        const bool wakeword_file_body_route =
            method == "POST" && matches_template(route_path, "/v1/wakeword/predict/{word}/file");
        const bool wakeword_typed_query_route = method == "GET" && route_path == "/v1/wakeword/jobs";
        if (!wakeword_enabled && !wakeword_json_body_route && !wakeword_file_body_route && !wakeword_typed_query_route) {
            return wakeword_disabled_response();
        }
        if (method == "GET" && route_path == "/v1/wakeword/models") {
            return {200, "application/json", wakeword_models_json()};
        }
        if (method == "GET" && matches_template(route_path, "/v1/wakeword/models/{word}/config")) {
            const auto word = path_param_for_template(route_path, "/v1/wakeword/models/{word}/config");
            std::filesystem::path model_dir;
            if (wakeword_model_dir_for_word(word, &model_dir)) {
                const auto config_path = wakeword_contained_path(model_dir, "config.json");
                std::error_code ec;
                if (!std::filesystem::exists(config_path, ec)) {
                    return {404, "application/json", "{\"detail\":\"Model config not found\"}"};
                }
                const auto config = read_file_or_empty(config_path.string());
                return {200, "application/json", nlohmann::ordered_json::parse(config).dump()};
            }
            return {404, "application/json", "{\"detail\":\"No model found for word '" + json_escape(word) + "'\"}"};
        }
        if (method == "GET" && matches_template(route_path, "/v1/wakeword/models/{word}")) {
            const auto word = path_param_for_template(route_path, "/v1/wakeword/models/{word}");
            std::filesystem::path model_dir;
            if (wakeword_model_dir_for_word(word, &model_dir)) {
                const std::string format = query_param_value(query, "format", "pytorch");
                std::string ext = "pt";
                if (format == "onnx") {
                    ext = "onnx";
                } else if (format == "espdl") {
                    ext = "espdl";
                }
                const auto model_file = wakeword_contained_path(model_dir, "model." + ext);
                std::error_code ec;
                if (!std::filesystem::exists(model_file, ec)) {
                    return {404, "application/json", "{\"detail\":\"Model format '" + json_escape(format) + "' not available for word '" + json_escape(word) + "'\"}"};
                }
                const std::string filename = wakeword_key_for_word(word) + "_model." + ext;
                return serve_file_response(model_file.string(), request_headers, "application/octet-stream", filename, false, allow_deferred_streaming);
            }
            if (const auto active_job = wakeword_jobs_manager().active(word)) {
                const auto active_id = extract_json_string(*active_job, "job_id", "");
                std::ostringstream out;
                out << "{"
                    << "\"status\":\"training_in_progress\","
                    << "\"job_id\":\"" << json_escape(active_id) << "\","
                    << "\"progress\":" << std::setprecision(std::numeric_limits<double>::max_digits10) << extract_json_double_value(*active_job, "progress", 0.0) << ","
                    << "\"current_stage\":\"" << json_escape(extract_json_string(*active_job, "current_stage", "")) << "\","
                    << "\"message\":\"Model for '" << json_escape(word) << "' is being trained. Check back soon.\","
                    << "\"check_status_url\":\"/v1/wakeword/jobs/" << json_escape(active_id) << "\""
                    << "}";
                return {200, "application/json", out.str()};
            }
            return {404, "application/json", "{\"detail\":\"No model found for word '" + json_escape(word) + "'. Use POST /v1/wakeword/train to create one.\"}"};
        }
        if (method == "DELETE" && matches_template(route_path, "/v1/wakeword/models/{word}")) {
            const auto word = path_param_for_template(route_path, "/v1/wakeword/models/{word}");
            std::filesystem::path model_dir;
            if (wakeword_model_dir_for_word(word, &model_dir)) {
                std::error_code ec;
                std::filesystem::remove_all(model_dir, ec);
                if (!ec) {
                    std::lock_guard<std::mutex> lock(wakeword_models_mutex_);
                    wakeword_loaded_models_.erase(llm_unicode_lower(word));
                    return {200, "application/json", "{\"message\":\"Model for '" + json_escape(word) + "' deleted successfully\"}"};
                }
            }
            return {404, "application/json", "{\"detail\":\"No model found for word '" + json_escape(word) + "'\"}"};
        }
        if (method == "POST" && route_path == "/v1/wakeword/train") {
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_model_response(body);
            }
            const auto request_values = top_level_json_object_values(body);
            std::vector<std::string> validation_errors;
            std::string word;
            validate_pydantic_required_string_field(request_values, body, "word", &word, &validation_errors);
            word = strip_config_value(word);
            for (const auto &field : {"sample_count", "epochs", "batch_size"}) {
                const auto it = request_values.find(field);
                if (it == request_values.end()) {
                    continue;
                }
                const auto parsed = parse_pydantic_int_token(it->second);
                if (!parsed.ok) {
                    append_pydantic_error(
                        &validation_errors,
                        parsed.type,
                        field,
                        parsed.message,
                        pydantic_input_json(it->second)
                    );
                }
            }
            if (!validation_errors.empty()) {
                return {422, "application/json", pydantic_error_response_body(validation_errors)};
            }
            if (!wakeword_enabled) {
                return wakeword_disabled_response();
            }
            try { wakeword_name(word); }
            catch (const std::invalid_argument &) { return {400, "application/json", "{\"detail\":\"Invalid wake word name\"}"}; }
            std::filesystem::path model_dir;
            if (wakeword_model_dir_for_word(word, &model_dir)) {
                return {
                    200,
                    "application/json",
                    "{\"job_id\":\"\",\"word\":\"" + json_escape(word) + "\",\"status\":\"completed\","
                        "\"message\":\"Model for '" + json_escape(word) + "' already exists. Use GET /v1/wakeword/models/" + json_escape(word) + " to download.\","
                        "\"estimated_minutes\":null,"
                        "\"check_status_url\":\"/v1/wakeword/models/" + json_escape(word) + "\"}"
                };
            }
            const auto parameter = [&](const char *name, int fallback) { const auto it = request_values.find(name); return it == request_values.end() ? fallback : parse_pydantic_int_token(it->second).value; };
            std::pair<std::string, bool> created_job;
            try { created_job = wakeword_jobs_manager().create(word, parameter("sample_count", 500), parameter("epochs", 50), parameter("batch_size", 32)); }
            catch (const std::invalid_argument &) { return {409, "application/json", "{\"detail\":\"Training conflict or invalid request\"}"}; }
            const auto job_id = extract_json_string(created_job.first, "job_id", "");
            if (!created_job.second) {
                return {
                    200,
                    "application/json",
                    "{\"job_id\":\"" + json_escape(job_id) + "\","
                        "\"word\":\"" + json_escape(word) + "\","
                        "\"status\":\"" + json_escape(extract_json_string(created_job.first, "status", "")) + "\","
                        "\"message\":\"Training already in progress for '" + json_escape(word) + "'\","
                        "\"estimated_minutes\":15,"
                        "\"check_status_url\":\"/v1/wakeword/jobs/" + json_escape(job_id) + "\"}"
                };
            }
            return {
                200,
                "application/json",
                "{\"job_id\":\"" + json_escape(job_id) + "\","
                    "\"word\":\"" + json_escape(word) + "\","
                    "\"status\":\"pending\","
                    "\"message\":\"Training started for '" + json_escape(word) + "'. This typically takes 10-20 minutes.\","
                    "\"estimated_minutes\":15,"
                    "\"check_status_url\":\"/v1/wakeword/jobs/" + json_escape(job_id) + "\"}"
            };
        }
        if (method == "GET" && route_path == "/v1/wakeword/jobs") {
            int limit = 50;
            if (query_param_present(query, "limit")) {
                const std::string raw_limit = query_param_value(query, "limit", "");
                if (!parse_pydantic_integral_string(raw_limit, &limit)) {
                    return pydantic_query_int_error("limit", raw_limit);
                }
            }
            if (!wakeword_enabled) {
                return wakeword_disabled_response();
            }
            std::string status_filter;
            if (query_param_present(query, "status")) {
                const auto status = query_param_value(query, "status", "");
                const std::set<std::string> valid_statuses = {
                    "pending",
                    "generating_samples",
                    "augmenting",
                    "training",
                    "exporting",
                    "completed",
                    "failed",
                    "cancelled",
                };
                if (!status.empty() && valid_statuses.find(status) == valid_statuses.end()) {
                    return {400, "application/json", "{\"detail\":\"Invalid status. Must be one of: ['pending', 'generating_samples', 'augmenting', 'training', 'exporting', 'completed', 'failed', 'cancelled']\"}"};
                }
                status_filter = status;
            }
            std::string response = "[";
            for (const auto &job : wakeword_jobs_manager().list(status_filter, limit)) { if (response.size() > 1) response += ','; response += wakeword_job_response_json(job); }
            return {200, "application/json", response + "]"};
        }
        if (method == "GET" && matches_template(route_path, "/v1/wakeword/jobs/{job_id}")) {
            const auto job_id = path_param_for_template(route_path, "/v1/wakeword/jobs/{job_id}");
            if (const auto job = wakeword_jobs_manager().get(job_id)) {
                return {200, "application/json", wakeword_job_response_json(*job)};
            }
            return {404, "application/json", "{\"detail\":\"Job not found\"}"};
        }
        if (method == "DELETE" && matches_template(route_path, "/v1/wakeword/jobs/{job_id}")) {
            const auto job_id = path_param_for_template(route_path, "/v1/wakeword/jobs/{job_id}");
            if (wakeword_jobs_manager().cancel(job_id)) {
                return {200, "application/json", "{\"message\":\"Job cancelled successfully\"}"};
            }
            return {400, "application/json", "{\"detail\":\"Could not cancel job (may already be completed or cancelled)\"}"};
        }
        if (method == "POST" && matches_template(route_path, "/v1/wakeword/predict/{word}/file")) {
            const auto word = path_param_for_template(route_path, "/v1/wakeword/predict/{word}/file");
            HttpResponse file_error;
            if (multipart_file_upload_validation_error(body, content_type, &file_error)) {
                return file_error;
            }
            if (!wakeword_enabled) {
                return wakeword_disabled_response();
            }
            std::filesystem::path model_dir;
            if (wakeword_model_dir_for_word(word, &model_dir)) {
                const auto parts = parse_multipart_parts(body, extract_multipart_boundary(content_type));
                const auto *file = find_multipart_part(parts, "file");
                return predict_wakeword(word, model_dir.string(), file ? file->content : "");
            }
            return {404, "application/json", "{\"detail\":\"No model found for word '" + json_escape(word) + "'. Train one first with POST /v1/wakeword/train\"}"};
        }
        if (method == "POST" && matches_template(route_path, "/v1/wakeword/predict/{word}")) {
            const auto word = path_param_for_template(route_path, "/v1/wakeword/predict/{word}");
            if (!is_valid_top_level_json_object(body)) {
                return invalid_json_model_response(body);
            }
            const auto request_values = top_level_json_object_values(body);
            std::vector<std::string> validation_errors;
            std::string audio_base64;
            validate_pydantic_required_string_field(request_values, body, "audio_base64", &audio_base64, &validation_errors);
            if (!validation_errors.empty()) {
                return {422, "application/json", pydantic_error_response_body(validation_errors)};
            }
            if (!wakeword_enabled) {
                return wakeword_disabled_response();
            }
            std::filesystem::path model_dir;
            if (wakeword_model_dir_for_word(word, &model_dir)) {
                try {
                    const auto audio = stt_base64_decode(request_values.at("audio_base64"));
                    return predict_wakeword(word, model_dir.string(), audio);
                } catch (...) { return {500, "application/json", "{\"detail\":\"Prediction failed\"}"}; }
            }
            return {404, "application/json", "{\"detail\":\"No model found for word '" + json_escape(word) + "'. Train one first with POST /v1/wakeword/train\"}"};
        }
        return {501, "application/json", make_error(501, "native wake word endpoints are not implemented yet")};
    } catch (const std::exception &) { return plain_internal_server_error(); }
    if (method == "GET" && route_path == "/dashboard") {
        return router_dashboard_html();
    }
    if (starts_with(route_path, "/v1/router/")) {
        return {501, "application/json", make_error(501, "native router mode is not implemented yet")};
    }
    return {404, "application/json", make_error(404, "route not found")};
}

HttpResponse NativeServer::docs_html() const {
    const std::string title = native_router_mode_enabled() ? "ezlocalai Router" : "ezlocalai Server";
    std::ostringstream out;
    out << R"DOC(
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <title>)DOC" << title << R"DOC( - Swagger UI</title>
    </head>
    <body>
    <div id="swagger-ui">
    </div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <!-- `SwaggerUIBundle` is now available on the page -->
    <script>
    const ui = SwaggerUIBundle({
        url: '/openapi.json',
    "dom_id": "#swagger-ui",
"layout": "BaseLayout",
"deepLinking": true,
"showExtensions": true,
"showCommonExtensions": true,
oauth2RedirectUrl: window.location.origin + '/docs/oauth2-redirect',
    presets: [
        SwaggerUIBundle.presets.apis,
        SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
    })
    </script>
    </body>
    </html>
    )DOC";
    return {200, "text/html; charset=utf-8", out.str()};
}

HttpResponse NativeServer::redoc_html() const {
    const std::string title = native_router_mode_enabled() ? "ezlocalai Router" : "ezlocalai Server";
    std::ostringstream out;
    out << R"DOC(
    <!DOCTYPE html>
    <html>
    <head>
    <title>)DOC" << title << R"DOC( - ReDoc</title>
    <!-- needed for adaptive design -->
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
)DOC" << "    \n" << R"DOC(    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
)DOC" << "    \n" << R"DOC(    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <!--
    ReDoc doesn't change outer page styles
    -->
    <style>
      body {
        margin: 0;
        padding: 0;
      }
    </style>
    </head>
    <body>
    <noscript>
        ReDoc requires Javascript to function. Please enable it to browse the documentation.
    </noscript>
    <redoc spec-url="/openapi.json"></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"> </script>
    </body>
    </html>
    )DOC";
    return {200, "text/html; charset=utf-8", out.str()};
}

HttpResponse NativeServer::swagger_oauth2_redirect_html() const {
    const std::string body = R"OAUTH(
    <!doctype html>
    <html lang="en-US">
    <head>
        <title>Swagger UI: OAuth2 Redirect</title>
    </head>
    <body>
    <script>
        'use strict';
        function run () {
            var oauth2 = window.opener.swaggerUIRedirectOauth2;
            var sentState = oauth2.state;
            var redirectUrl = oauth2.redirectUrl;
            var isValid, qp, arr;

            if (/code|token|error/.test(window.location.hash)) {
                qp = window.location.hash.substring(1).replace('?', '&');
            } else {
                qp = location.search.substring(1);
            }

            arr = qp.split("&");
            arr.forEach(function (v,i,_arr) { _arr[i] = '"' + v.replace('=', '":"') + '"';});
            qp = qp ? JSON.parse('{' + arr.join() + '}',
                    function (key, value) {
                        return key === "" ? value : decodeURIComponent(value);
                    }
            ) : {};

            isValid = qp.state === sentState;

            if ((
              oauth2.auth.schema.get("flow") === "accessCode" ||
              oauth2.auth.schema.get("flow") === "authorizationCode" ||
              oauth2.auth.schema.get("flow") === "authorization_code"
            ) && !oauth2.auth.code) {
                if (!isValid) {
                    oauth2.errCb({
                        authId: oauth2.auth.name,
                        source: "auth",
                        level: "warning",
                        message: "Authorization may be unsafe, passed state was changed in server. The passed state wasn't returned from auth server."
                    });
                }

                if (qp.code) {
                    delete oauth2.state;
                    oauth2.auth.code = qp.code;
                    oauth2.callback({auth: oauth2.auth, redirectUrl: redirectUrl});
                } else {
                    let oauthErrorMsg;
                    if (qp.error) {
                        oauthErrorMsg = "["+qp.error+"]: " +
                            (qp.error_description ? qp.error_description+ ". " : "no accessCode received from the server. ") +
                            (qp.error_uri ? "More info: "+qp.error_uri : "");
                    }

                    oauth2.errCb({
                        authId: oauth2.auth.name,
                        source: "auth",
                        level: "error",
                        message: oauthErrorMsg || "[Authorization failed]: no accessCode received from the server."
                    });
                }
            } else {
                oauth2.callback({auth: oauth2.auth, token: qp, isValid: isValid, redirectUrl: redirectUrl});
            }
            window.close();
        }

        if (document.readyState !== 'loading') {
            run();
        } else {
            document.addEventListener('DOMContentLoaded', function () {
                run();
            });
        }
    </script>
    </body>
    </html>
        )OAUTH";
    return {200, "text/html; charset=utf-8", body};
}

HttpResponse NativeServer::openapi_json() const {
    return {200, "application/json", native_router_mode_enabled()
        ? std::string(generated_openapi::kRouterOpenApiJson) : worker_openapi_json_};
}

HttpResponse NativeServer::enqueue_inference(const std::string &kind, const std::string &body, bool allow_deferred_streaming) {
    sync_text_queue_capacity();
    char request_id[64];
    const bool llm_request = kind == "chat" || kind == "completion";
    const std::string request_body = llm_request ? json_with_default_int_field(body, "max_tokens", 8192) : body;
    // Queue processors can return a complete forwarding response without
    // encoding status/type metadata into model-generated text. The scoped
    // registration also discards late results after a timeout or cancellation.
    const auto response_key = uuid_v4();
    auto forwarded = std::make_shared<QueuedForwardResponse>();
    { std::lock_guard<std::mutex> lock(queued_forward_mutex_); queued_forward_responses_[response_key] = forwarded; }
    auto registration = std::shared_ptr<int>(new int(0), [this,response_key](int *value) {
        delete value; std::lock_guard<std::mutex> lock(queued_forward_mutex_); queued_forward_responses_.erase(response_key);
    });
    const std::string payload = kind + "\n" + response_key + "\n" + request_body;
    int rc = ezl_queue_enqueue_typed(queue_, payload.c_str(), kind.c_str(), request_id, sizeof(request_id));
    if (rc == ENOSPC) {
        return {503, "application/json", make_error(503, "request queue is full")};
    }
    if (rc != 0) {
        return {500, "application/json", make_error(500, "failed to enqueue request")};
    }

    const int result_bytes = std::max(65536, std::min(parse_positive_int_env("NATIVE_QUEUE_RESULT_BYTES", 4 * 1024 * 1024), 128 * 1024 * 1024));
    std::vector<char> result(static_cast<size_t>(result_bytes));
    char error[512];
    long wait_timeout_ms = config_.request_timeout_ms;
    bool tried_queue_fallback = false;
    // The reference obtains its singleton even when this request disables
    // fallback; only the subsequent routing decision is conditional.
    const auto *fallback = llm_request ? &fallback_client() : nullptr;
    const bool llm_fallback_candidate =
        llm_request
        && !llm_fallback_disabled(body)
        && (!fallback->url.empty() || !router_fallback().url.empty());
    if (llm_fallback_candidate) {
        const double queue_wait_seconds = parse_double_env("QUEUE_WAIT_TIMEOUT", 30.0);
        if (queue_wait_seconds > 0.0) {
            wait_timeout_ms = std::max(1L, static_cast<long>(queue_wait_seconds * 1000.0));
        }
    }

    rc = ezl_queue_wait(queue_, request_id, wait_timeout_ms, result.data(), result.size(), error, sizeof(error));
    if (rc == ETIMEDOUT && llm_fallback_candidate && wait_timeout_ms != config_.request_timeout_ms) {
        ezl_request_status status = EZL_REQUEST_QUEUED;
        if (ezl_queue_get_request_status(queue_, request_id, &status) == 0 && status == EZL_REQUEST_QUEUED) {
            const std::string path = kind == "chat" ? "/v1/chat/completions" : "/v1/completions";
            HttpResponse fallback_response;
            // Without FALLBACK_SERVER the router serves the queued request.
            // This layer returns complete responses, so a stream is buffered.
            if (fallback->url.empty() ? forward_llm_to_router(path, request_body, &fallback_response, nullptr)
                    : forward_fallback_llm(path, request_body, &fallback_response)) {
                ezl_queue_cancel(queue_, request_id);
                return fallback_response;
            }
            tried_queue_fallback = true;
        }

        const long remaining = std::max(1000L, config_.request_timeout_ms - wait_timeout_ms);
        rc = ezl_queue_wait(queue_, request_id, remaining, result.data(), result.size(), error, sizeof(error));
    }
    if (rc == ETIMEDOUT) {
        ezl_queue_cancel(queue_, request_id);
        if (tried_queue_fallback) {
            return {408, "application/json", make_error(408, "request timed out after fallback attempt")};
        }
        return {408, "application/json", make_error(408, "request timed out")};
    }
    if (rc != 0) {
        return {500, "application/json", make_error(500, error[0] ? error : "request failed")};
    }
    bool source_response = false, fallback_dictionary = false;
    std::string response_body;
    std::shared_ptr<LlmInferenceStream> local_stream;
    { std::lock_guard<std::mutex> lock(forwarded->mutex); if (forwarded->response) return std::move(*forwarded->response);
      source_response = forwarded->source_response; fallback_dictionary = forwarded->fallback_dictionary;
      local_stream = std::move(forwarded->stream);
      response_body = forwarded->local_body ? std::move(*forwarded->local_body) : std::string(result.data()); }
    if (local_stream) {
        source_response = true;
        if (allow_deferred_streaming && extract_json_bool(request_body, "stream", false)) {
            HttpResponse response{200, "text/event-stream; charset=utf-8", ""};
            response.chunked = true; response.stream_capture = std::make_shared<std::string>();
            response.stream_writer = [stream = local_stream, capture = response.stream_capture](int fd) {
                struct Close { std::shared_ptr<LlmInferenceStream> stream; ~Close() { stream->close(); } } close{stream};
                const auto disconnected = [fd] {
                    char value;
                    const auto count = recv(fd, &value, 1, MSG_PEEK | MSG_DONTWAIT);
                    return count == 0 || (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
                };
                const auto send_event = [&](const std::string &event) {
                    append_stream_capture(capture, event); return send_http_chunk(fd, event);
                };
                try {
                    std::string chunk;
                    while (stream->next(&chunk, disconnected)) if (!send_event("data: " + chunk + "\n\n")) return;
                    if (disconnected()) return;
                    const auto error = stream->failure();
                    if (!send_event(error ? llm_stream_error_event(error->message) : "data: [DONE]\n\n")) return;
                } catch (const std::exception &error) { send_event(llm_stream_error_event(error.what())); }
                send_all_fd(fd, "0\r\n\r\n");
            };
            return response;
        }
        InferenceResult deferred(""); deferred.from_engine = true; deferred.stream = std::move(local_stream);
        const auto collected = collect_llm_stream(std::move(deferred));
        if (collected.failure && !extract_json_bool(request_body, "stream", false))
            return {500, "text/plain; charset=utf-8", "Internal Server Error"};
        response_body = collected.body;
    }
    const std::string backend_error = trim(extract_json_value(response_body, "error", ""));
    if (!source_response && !backend_error.empty() && backend_error != "null") {
        if (extract_json_string(backend_error, "type", "") == "temporarily_unavailable") {
            // app.py returns this dictionary directly. Its chat stream iterates
            // the dictionary keys; completion streaming requires choices.
            if (extract_json_bool(request_body, "stream", false)) {
                if (kind == "completion") return {500, "text/plain; charset=utf-8", "Internal Server Error"};
                HttpResponse response{200, "text/event-stream; charset=utf-8", "data: \"error\"\n\ndata: [DONE]\n\n"};
                response.chunked = true;
                return response;
            }
            return {200, "application/json", response_body};
        }
        int status = extract_json_int_value(backend_error, "code", 0);
        if (status < 400 || status > 599) {
            status = 500;
        }
        return {status, "application/json", response_body};
    }
    const int top_level_error_code = extract_json_int_value(response_body, "code", 0);
    if (!source_response && top_level_error_code >= 400 && top_level_error_code <= 599
        && !extract_json_string(response_body, "message", "").empty()) {
        return {top_level_error_code, "application/json", response_body};
    }
    if (llm_request && extract_json_bool(request_body, "stream", false)) {
        HttpResponse stream_response{200, "text/event-stream; charset=utf-8", ""};
        if (starts_with(trim(response_body), "data:")) {
            stream_response.body = response_body;
            stream_response.chunked = true;
            return stream_response;
        }
        if (kind == "chat") {
            if (fallback_dictionary) {
                for (const auto &field : json_object_ordered_entries(response_body)) stream_response.body += "data: \"" + json_escape(field.first) + "\"\n\n";
                stream_response.body += "data: [DONE]\n\n";
            } else stream_response.body = "data: " + response_body + "\n\ndata: [DONE]\n\n";
            stream_response.chunked = true;
            return stream_response;
        }
        if (fallback_dictionary) {
            const auto choices = json_array_elements(json_object_field(response_body, "choices", "[]"));
            if (choices.empty()) return {500, "text/plain; charset=utf-8", "Internal Server Error"};
            const auto text = json_object_field(choices.front(), "text", "null");
            if (!text.empty() && text.front() == '"') stream_response.body = parse_json_string_at(text, 0);
            else if (!text.empty() && text.front() == '[') {
                for (const auto &chunk : json_array_elements(text)) {
                    if (chunk.empty() || chunk.front() != '"') return {500, "text/plain; charset=utf-8", "Internal Server Error"};
                    stream_response.body += parse_json_string_at(chunk, 0);
                }
            } else if (!text.empty() && text.front() == '{') {
                for (const auto &field : json_object_ordered_entries(text)) stream_response.body += field.first;
            } else return {500, "text/plain; charset=utf-8", "Internal Server Error"};
        } else stream_response.body = extract_json_string(response_body, "text", "");
        stream_response.chunked = true;
        return stream_response;
    }
    std::string final_body = response_body;
    if (source_response && !starts_with(trim(response_body), "data:")) {
        try { final_body = fallback_model_json(response_body); }
        catch (const std::exception &) { return {500, "text/plain; charset=utf-8", "Internal Server Error"}; }
    }
    if (llm_request && !is_python_falsey_json_value(json_object_field(request_body, "stream", "false"))
        && starts_with(trim(response_body), "data:")) {
        // A truthy raw string such as "false" selects a local generator, but
        // Pydantic validates it as false. FastAPI consumes that generator into
        // a JSON list. The managed engine represents its yielded values as SSE.
        try {
            std::vector<std::string> chunks;
            std::string event, line;
            const auto append = [&] {
                if (event.empty() || event == "[DONE]") return;
                if (json_decode_error(event)) throw std::invalid_argument("Invalid stream JSON");
                chunks.push_back(fallback_model_json(event));
            };
            std::istringstream stream(response_body);
            while (std::getline(stream, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) { append(); if (event == "[DONE]") break; event.clear(); }
                else if (starts_with(line, "data:")) {
                    if (!event.empty()) event += '\n';
                    event += line.substr(line.size() > 5 && line[5] == ' ' ? 6 : 5);
                }
            }
            if (!event.empty() && event != "[DONE]") append();
            final_body = json_array_from_values(chunks);
            std::string decoded;
            if (!fallback_decode(final_body, "utf-8", &decoded)) throw std::invalid_argument("Invalid stream UTF-8");
        } catch (const std::exception &) { return {500, "text/plain; charset=utf-8", "Internal Server Error"}; }
    }
    if (llm_request) {
        append_voice_audio_to_llm_response(kind, request_body, &final_body);
    }
    return {200, "application/json", final_body};
}

HttpResponse NativeServer::queue_status() const {
    ezl_queue_snapshot snapshot{};
    ezl_queue_get_snapshot(queue_, &snapshot);
    std::ostringstream out;
    out << "{"
        << "\"queue_size\":" << snapshot.queued << ","
        << "\"processing_count\":" << snapshot.processing << ","
        << "\"max_concurrent\":" << snapshot.max_concurrent_requests << ","
        << "\"max_queue_size\":" << snapshot.max_queue_size << ","
        << "\"total_processed\":" << snapshot.total_processed << ","
        << "\"total_failed\":" << snapshot.total_failed << ","
        << "\"total_queued\":" << snapshot.total_queued << ","
        << "\"active_requests\":" << snapshot.active << ","
        << "\"is_running\":" << (running_ ? "true" : "false")
        << "}";
    return {200, "application/json", out.str()};
}

std::string NativeServer::resolve_request_slot_model(const std::string &body) const {
    return llm_catalog_.slot_model(extract_json_string(body, "model", "")).value_or("");
}

bool NativeServer::should_use_llm_fallback() const {
    if (resources_->is_model_loaded(ModelType::Llm) || resources_->is_model_loaded(ModelType::VisionLlm)) return false;
    // Managed text engines currently publish residency through their backend
    // snapshot; embedding engines already register with the shared tracker.
    const auto loaded = backend_->loaded_models_json();
    if (!json_object_field(loaded, "llm", "").empty() || !json_object_field(loaded, "vision_llm", "").empty()) return false;
    const auto decision = native_fallback_decision(resources_.get(), effective_llm_fallback_url());
    if (decision.should_use) return true;
    if (!decision.configured) return false;
    const auto admission = resources_->can_load_model(ModelType::Llm);
    return !admission.can_load && admission.device == "fallback";
}

std::string NativeServer::begin_llm_inference(const std::string &model, const std::string &body) {
    const auto requested = extract_json_string(body, "model", "");
    const auto preferred = requested.find('#') == std::string::npos ? std::nullopt : llm_catalog_.resolve(requested);
    auto replicas = backend_->llm_replica_state().value_or(llm_replicas_);
    if (preferred) for (const auto &replica : replicas) if (replica.id == *preferred && replica.failed)
        throw std::runtime_error("Requested replica " + *preferred + " failed to load");
    std::unique_lock<std::mutex> lock(resource_mutex_);
    for (;;) {
        if (llm_slots_stopping_) throw std::runtime_error("LLM slot scheduler is stopping");
        const auto acquired = try_acquire_llm_slot(replicas, llm_slots_, model.empty() ? std::nullopt : std::optional<std::string>(model), preferred, llm_temporarily_unavailable_, backend_->swap_llm_residency());
        if (acquired.acquired) {
            lock.unlock();
            notify_worker_state_changed();
            return acquired.replica.value_or("");
        }
        llm_idle_cv_.wait_for(lock, std::chrono::milliseconds(50));
        lock.unlock();
        replicas = backend_->llm_replica_state().value_or(llm_replicas_);
        lock.lock();
    }
}

void NativeServer::end_llm_inference(const std::string &model, const std::string &replica) {
    {
        std::lock_guard<std::mutex> lock(resource_mutex_);
        release_llm_slot(llm_slots_, model.empty() ? std::nullopt : std::optional<std::string>(model), replica.empty() ? std::nullopt : std::optional<std::string>(replica));
        llm_idle_cv_.notify_all();
    }
    notify_worker_state_changed();
}

void NativeServer::resource_increment_capability_in_flight(const std::string &capability, int delta) const {
    if (capability.empty() || delta == 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(resource_mutex_);
        int &count = resource_capability_in_flight_[capability];
        count = std::max(0, count + delta);
        resource_capability_last_used_[capability] = monotonic_epoch_seconds();
        if (count == 0) resource_capability_in_flight_.erase(capability);
    }
    notify_worker_state_changed();
}

void NativeServer::sync_text_queue_capacity() const {
    const auto replicas = backend_->llm_replica_state();
    if (!replicas) return;
    int64_t capacity = 0;
    bool unavailable;
    { std::lock_guard<std::mutex> lock(resource_mutex_); unavailable = llm_temporarily_unavailable_; }
    if (!unavailable) for (const auto &replica : *replicas) if (!replica.failed) capacity += llm_replica_capacity(replica);
    if (backend_->swap_llm_residency()) capacity = std::min<int64_t>(1, capacity);
    const int error = ezl_queue_set_concurrency(queue_, static_cast<size_t>(std::max<int64_t>(1, capacity)));
    if (error) throw std::runtime_error("Could not resize text queue: " + std::string(std::strerror(error)));
}

HttpResponse NativeServer::resources(bool sync_queue) const {
    const auto gpu = resources_->get_status();
    auto fallback = native_fallback_decision(resources_.get(), fallback_client().url);
    // app.get_resources reports another live reading even when fallback is
    // disabled; should_use_ezlocalai_fallback itself returns early in that case.
    fallback.free_vram_gb = resources_->get_total_free_vram();
    fallback.free_ram_gb = resources_->get_free_ram();
    fallback.memory_threshold_gb = fallback_memory_threshold();
    if (!std::isfinite(fallback.memory_threshold_gb)) throw std::invalid_argument("Non-finite resource JSON value");
    ezl_queue_snapshot snapshot{};
    ezl_queue_get_snapshot(queue_, &snapshot);

    std::map<std::string, int64_t> active_model_counts;
    std::map<std::string, int> active_capability_counts;
    std::map<std::string, double> capability_last_used;
    bool llm_unavailable;
    int llm_in_flight;
    bool video_loaded;
    std::string video_device;
    {
        std::lock_guard<std::mutex> lock(resource_mutex_);
        active_model_counts = llm_slots_.models;
        active_capability_counts = resource_capability_in_flight_;
        capability_last_used = resource_capability_last_used_;
        llm_unavailable = llm_temporarily_unavailable_;
        llm_in_flight = static_cast<int>(std::max<int64_t>(llm_slots_.total, snapshot.processing));
        video_loaded = resource_video_loaded_;
        video_device = resource_video_device_;
    }

    const auto models = llm_catalog_.available;

    const bool swap_residency = backend_->swap_llm_residency();
    const auto backend_replicas = backend_->llm_replica_state();
    const auto replicas = backend_replicas.value_or(llm_replicas_);

    std::map<std::string, int> model_capacities;
    std::map<std::string, int> model_instances;
    for (const auto &replica : replicas) if (replica.resident) ++model_instances[replica.source];
    std::set<std::string> vision_models;
    int vision_capacity = 0;
    for (size_t i = 0; i < models.size(); ++i) {
        if (replicas.at(i).failed && !llm_unavailable) continue;
        std::string public_name = llm_catalog_.source(models[i]);
        const auto capacity = swap_residency ? 1 : static_cast<int>(llm_replica_capacity(replicas.at(i)));
        if (swap_residency) model_capacities[public_name] = std::max(model_capacities[public_name], capacity);
        else model_capacities[public_name] += capacity;
        if (model_has_local_vision_projector(public_name)) {
            vision_models.insert(public_name);
            vision_capacity = swap_residency ? std::max(vision_capacity, capacity) : vision_capacity + capacity;
        }
    }

    std::map<std::string, RouterSlotState> model_slots;
    int llm_capacity = 0;
    for (const auto &entry : model_capacities) {
        RouterSlotState state;
        state.capacity = llm_unavailable ? 0 : std::max(1, entry.second);
        state.in_flight = llm_unavailable ? 0 : std::max<int64_t>(0, active_model_counts[entry.first]);
        if (swap_residency && !llm_unavailable && llm_in_flight > 0) state.in_flight = std::max(1, state.in_flight);
        state.queued = 0;
        state.available = std::max(0, state.capacity - state.in_flight - state.queued);
        if (!swap_residency && !llm_unavailable && model_instances[entry.first] > 1)
            state.instances = model_instances[entry.first];
        model_slots[entry.first] = state;
        llm_capacity += state.capacity;
    }
    if (swap_residency) llm_capacity = std::min(1, llm_capacity);

    std::map<std::string, RouterSlotState> cap_slots;
    if (!model_capacities.empty()) {
        RouterSlotState text;
        text.capacity = llm_capacity;
        text.in_flight = llm_unavailable ? 0 : llm_in_flight;
        text.queued = llm_unavailable ? 0 : static_cast<int>(snapshot.queued);
        text.available = std::max(0, text.capacity - text.in_flight - text.queued);
        if (model_slots.size() == 1) text.instances = model_slots.begin()->second.instances;
        cap_slots["text"] = text;
    }
    if (vision_capacity > 0) {
        RouterSlotState vision;
        vision.capacity = llm_unavailable ? 0 : vision_capacity;
        for (const auto &model : vision_models) {
            if (!llm_unavailable) vision.in_flight += std::max<int64_t>(0, active_model_counts[model]);
        }
        vision.queued = 0;
        vision.available = std::max(0, vision.capacity - vision.in_flight - vision.queued);
        if (vision_models.size() == 1) vision.instances = model_slots.at(*vision_models.begin()).instances;
        cap_slots["vision"] = vision;
    }
    const int tts_active = tts_service_ ? resources_->get_model_active_count(ModelType::Tts) : std::max(0, active_capability_counts["tts"]);
    const int stt_active = stt_service_ ? resources_->get_model_active_count(ModelType::Stt) : std::max(0, active_capability_counts["stt"]);
    const bool voice_handoff_active = (tts_service_ && tts_service_->handoff_active())
        || (stt_service_ && stt_service_->handoff_active());
    if (configured_enabled("TTS_ENABLED") && voice_client().url.empty()) {
        const bool shared = native_tts_handoff();
        RouterSlotState tts;
        tts.capacity = shared ? 1 : speech_parallel_slots("TTS_N_PARALLEL");
        tts.in_flight = tts_active;
        if (shared && (llm_in_flight > 0 || stt_active > 0 || voice_handoff_active)) tts.in_flight = std::max(1, tts.in_flight);
        tts.available = std::max(0, tts.capacity - tts.in_flight);
        cap_slots["tts"] = tts;
    }
    if (configured_enabled("STT_ENABLED") && voice_client().url.empty()) {
        const bool shared = native_stt_handoff();
        RouterSlotState stt;
        stt.capacity = shared ? 1 : speech_parallel_slots("STT_N_PARALLEL");
        stt.in_flight = stt_active;
        if (shared && (llm_in_flight > 0 || tts_active > 0 || voice_handoff_active)) stt.in_flight = std::max(1, stt.in_flight);
        stt.available = std::max(0, stt.capacity - stt.in_flight);
        cap_slots["stt"] = stt;
    }
    const auto backend_loaded_models = backend_->observed_loaded_models_json();
    const bool resident_llm = json_object_field(backend_loaded_models, "llm", "null") != "null"
        || json_object_field(backend_loaded_models, "vision_llm", "null") != "null";
    const auto secondary = [] { return secondary_cuda_device(visible_cuda_devices()); };
    const bool image_loaded = std::any_of(gpu.loaded_models.begin(), gpu.loaded_models.end(),
        [](const auto &model) { return model.type == ModelType::Image; });
    const std::string img_model = to_lower(trim(ezl_getenv_default("IMG_MODEL")));
    if (configured_enabled("IMAGE_ENABLED") && !img_model.empty() && img_model != "none" && image_client().url.empty()) {
        RouterSlotState image;
        image.capacity = 1;
        image.in_flight = resources_->get_model_active_count(ModelType::Image);
        if (image_should_unload_llm(resident_llm, image_loaded, secondary,
                [this] { return resources_->get_total_free_vram(); }, [this] { return !image_client().url.empty(); }) && llm_in_flight > 0)
            image.in_flight = std::max(1, image.in_flight);
        image.available = std::max(0, image.capacity - image.in_flight);
        cap_slots["image"] = image;
    }
    const std::string video_model = to_lower(trim(ezl_getenv_default("VIDEO_MODEL")));
    if (configured_enabled("VIDEO_ENABLED") && !video_model.empty() && video_model != "none" && image_client().url.empty()) {
        RouterSlotState video;
        video.capacity = 1;
        video.in_flight = resources_->get_model_active_count(ModelType::Video);
        if (video_should_unload_llm(resident_llm, secondary, [this] { return !image_client().url.empty(); }) && llm_in_flight > 0)
            video.in_flight = std::max(1, video.in_flight);
        video.available = std::max(0, video.capacity - video.in_flight);
        cap_slots["video"] = video;
    }

    if (has_local_capability("music")) {
        RouterSlotState music;
        music.capacity = 1;
        music.in_flight = std::max(0, active_capability_counts["music"]);
        if (music_should_unload_llm() && llm_in_flight > 0) music.in_flight = std::max(1, music.in_flight);
        music.available = std::max(0, 1 - music.in_flight);
        cap_slots["music"] = music;
        if (cap_slots.count("video")) {
            RouterSlotState composite;
            composite.capacity = 1;
            composite.in_flight = std::max(music.in_flight, cap_slots.at("video").in_flight);
            composite.available = std::max(0, 1 - composite.in_flight);
            cap_slots["music_video"] = composite;
        }
    }
    if (configured_enabled("EMBEDDING_ENABLED") && embedding_client().url.empty()) {
        RouterSlotState embedding;
        embedding.capacity = embedding_parallel_slots();
        embedding.in_flight = std::max(0, active_capability_counts["embedding"]);
        embedding.available = std::max(0, embedding.capacity - embedding.in_flight);
        backend_->embedding_slot_state(&embedding);
        cap_slots["embedding"] = embedding;
    }

    if (video_session_active_.load()) {
        const bool exclusive = video_requires_exclusive_worker(resident_llm, secondary, [this] { return !image_client().url.empty(); });
        for (auto &[capability, state] : cap_slots) {
            if (capability == "video" || capability == "music_video") state = {1, 1, 0, 0};
            else if (exclusive) { state.capacity = 0; state.available = 0; }
        }
        if (exclusive) for (auto &[model, state] : model_slots) { state.capacity = 0; state.available = 0; }
    }

    int64_t slot_total_capacity = 0;
    int64_t slot_total_in_flight = 0;
    int64_t slot_total_queued = 0;
    for (const auto &entry : cap_slots) {
        if ((entry.first == "vision" && cap_slots.find("text") != cap_slots.end()) || entry.first == "music_video") {
            continue;
        }
        const auto state = normalize_slot(entry.second, entry.second.capacity);
        slot_total_capacity += state.capacity;
        slot_total_in_flight += state.in_flight;
        slot_total_queued += state.queued;
    }

    std::string loaded_models_json = merge_loaded_model_json_objects(
        backend_loaded_models,
        aux_loaded_models_json(active_capability_counts, capability_last_used, video_loaded, video_device, !tts_service_, !stt_service_)
    );
    loaded_models_json = merge_loaded_model_json_objects(loaded_models_json, resource_models_json(gpu.loaded_models));
    const auto empty_pool = [](int64_t capacity) { return nlohmann::json{{"configured", capacity}, {"resident", 0}, {"devices", nlohmann::json::array()}}; };
    const auto embedding_pool = backend_->embedding_pool_json();
    const nlohmann::json pools{
        {"tts", tts_service_ ? speech_pool_json(tts_service_->snapshot()) : empty_pool(tts_pool_configured_)},
        {"stt", stt_service_ ? speech_pool_json(stt_service_->snapshot()) : empty_pool(llm_audio_capacity_)},
        {"embedding", embedding_pool.empty() ? empty_pool(embedding_pool_configured_) : nlohmann::json::parse(embedding_pool)}
    };
    int n_parallel = 1;
    parse_strict_int_string(configured_parallel_value_for_index(0), &n_parallel);
    n_parallel = std::max(1, n_parallel);
    const auto actual_parallel = backend_->llm_parallel_count();
    if (actual_parallel) n_parallel = static_cast<int>(*actual_parallel);
    int n_ctx = backend_->llm_context_size();
    if (!actual_parallel && !trim(extract_json_value(loaded_models_json, "llm", "")).empty()) {
        if (n_ctx <= 0) {
            n_ctx = parse_first_csv_int_env("LLAMA_CPP_CONTEXT", 0);
            if (n_ctx <= 0) {
                n_ctx = parse_first_csv_int_env("LLM_MAX_TOKENS", 0);
            }
        }
    }
    auto lifecycle = backend_->model_lifecycle_json();
    if (lifecycle.empty()) {
        std::vector<std::string> configured;
        for (const auto &model : models) configured.push_back('"' + json_escape(llm_catalog_.source(model)) + '"');
        lifecycle = std::string("{\"llm_model_residency\":\"") + (swap_residency ? "swap" : "resident") + "\",\"configured_llms\":" + json_array_from_values(configured)
            + ",\"loaded_llms\":[],\"active_llm\":null,\"loaded_llm_runtime\":[],\"vram_estimates_gb\":{},\"summary\":[],\"recent_events\":[]}";
    }
    double queue_wait_timeout = parse_double_env_explicit("QUEUE_WAIT_TIMEOUT", 30.0);

    std::ostringstream out;
    out << "{"
        << "\"gpu_count\":" << gpu.gpu_count << ","
        << "\"total_vram_gb\":" << gpu.total_vram_gb << ","
        << "\"free_vram_gb\":" << json_number_literal(gpu.free_vram_gb) << ","
        << "\"free_per_gpu_gb\":" << json_number_array(gpu.free_per_gpu_gb) << ","
        << "\"torch_cuda\":" << resource_cuda_json(gpu.torch_cuda) << ","
        << "\"system_ram_gb\":" << json_number_literal(gpu.system_ram_gb) << ","
        << "\"loaded_models\":" << loaded_models_json << ","
        << "\"fallback\":{"
        << "\"configured\":" << (fallback.configured ? "true" : "false") << ",";
    if (fallback.configured) {
        out << "\"url\":\"" << json_escape(fallback.url) << "\",";
    } else {
        out << "\"url\":null,";
    }
    out << "\"should_use_fallback\":" << (fallback.should_use ? "true" : "false") << ","
        << "\"reason\":\"" << json_escape(fallback.reason) << "\","
        << "\"free_vram_gb\":" << json_number_literal(fallback.free_vram_gb) << ","
        << "\"free_ram_gb\":" << json_number_literal(fallback.free_ram_gb) << ","
        << "\"free_combined_gb\":" << json_number_literal(fallback.free_vram_gb + fallback.free_ram_gb) << ","
        << "\"memory_threshold_gb\":" << json_number_literal(fallback.memory_threshold_gb)
        << "},"
        << "\"parallel\":{"
        << "\"n_parallel\":" << n_parallel << ","
        << "\"n_ctx\":" << n_ctx << ","
        << "\"per_slot_context\":" << (n_parallel > 0 ? n_ctx / n_parallel : n_ctx) << ","
        << "\"text_queue_capacity\":" << snapshot.max_concurrent_requests << ",";
    if (queue_wait_timeout > 0.0) {
        out << "\"queue_wait_timeout\":" << json_number_literal(queue_wait_timeout);
    } else {
        out << "\"queue_wait_timeout\":null";
    }
    out << "},"
        << "\"model_lifecycle\":" << lifecycle << ","
        << "\"model_pools\":" << pools.dump() << ","
        << "\"slots\":{"
        << "\"cap_slots\":" << slot_map_json(cap_slots, {}, false) << ","
        << "\"model_slots\":" << slot_map_json(model_slots, {}, false) << ","
        << "\"llm_queue_capacity\":" << (models.empty() ? 1 : std::max(1, llm_capacity)) << ","
        << "\"slot_total_capacity\":" << slot_total_capacity << ","
        << "\"slot_total_in_flight\":" << slot_total_in_flight << ","
        << "\"slot_total_queued\":" << slot_total_queued << ","
        << "\"slot_total_available\":" << std::max<int64_t>(0, slot_total_capacity - slot_total_in_flight - slot_total_queued)
        << "}"
        << "}";
    // Python takes the parallel/queue snapshot before synchronizing admission.
    // Worker heartbeat collection reads slots without changing the app queue.
    if (sync_queue && backend_replicas) {
        const int error = ezl_queue_set_concurrency(queue_, static_cast<size_t>(std::max(1, llm_capacity)));
        if (error) throw std::runtime_error("Could not resize text queue: " + std::string(std::strerror(error)));
    }
    return {200, "application/json", out.str()};
}

HttpResponse NativeServer::models() const {
    if (to_lower(trim(ezl_getenv_default("ROUTER_MODE"))) == "true") {
        const double now = monotonic_epoch_seconds();
        const double ttl = router_ttl_seconds();
        std::vector<RouterWorker> workers;
        std::map<std::string, std::vector<std::string>> model_workers;
        std::vector<std::string> model_order;
        {
            std::lock_guard<std::mutex> lock(router_mutex_);
            for (const auto &worker : router_workers_in_registration_order_locked()) {
                if (now - worker.last_heartbeat > ttl) {
                    continue;
                }
                workers.push_back(worker);
            }
        }
        for (const auto &worker : workers) {
            for (const auto &model : worker.models) {
                if (model_workers.find(model) == model_workers.end()) {
                    model_order.push_back(model);
                }
                model_workers[model].push_back(worker.label);
            }
        }
        std::ostringstream routed;
        routed << "{\"object\":\"list\",\"data\":[";
        for (size_t i = 0; i < model_order.size(); ++i) {
            if (i) {
                routed << ",";
            }
            const auto &model = model_order[i];
            routed << "{"
                << "\"id\":\"" << json_escape(model) << "\","
                << "\"object\":\"model\","
                << "\"owned_by\":\"ezlocalai\","
                << "\"workers\":" << string_array_json(model_workers[model])
                << "}";
        }
        routed << "]}";
        return {200, "application/json", routed.str()};
    }

    const auto public_models = llm_catalog_.public_models();
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < public_models.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << "{"
            << "\"id\":\"" << json_escape(public_models[i]) << "\","
            << "\"object\":\"model\","
            << "\"owned_by\":\"ezlocalai\""
            << "}";
    }
    out << "]}";
    return {200, "application/json", out.str()};
}

HttpResponse NativeServer::serve_output_file(
    const std::string &path,
    const std::unordered_map<std::string, std::string> &request_headers,
    bool deferred
) const {
    std::string relative;
    if (!safe_output_relative_path(path, &relative))
        return {404, "application/json", "{\"detail\":\"Not Found\"}"};
    const auto fullpath = outputs_dir() + "/" + relative;
    return serve_file_response(fullpath, request_headers, content_type_for_path(fullpath), "", true, deferred);
}

HttpResponse NativeServer::audio_voices() const {
    if (llm_unicode_lower(ezl_getenv_default("TTS_ENABLED")) == "false") {
        return {404, "application/json", "{\"detail\":\"Text to speech is disabled.\"}"};
    }

    try {
        const auto client = tts_remote_client(TtsProvider::Voice);
        if (!voice_client().dedicated && client->configured && client->available()) {
            const auto voices = client->voices();
            if (voices && python_json_truthy(*voices)) {
                const auto result=fallback_model_json(*voices);std::string decoded;
                if(!fallback_decode(result,"utf-8",&decoded))return plain_internal_server_error();
                return {200, "application/json", result};
            }
        }
        std::vector<std::string> voices;
        try {
            const auto directory = std::filesystem::current_path() / "voices";
            std::filesystem::create_directories(directory);
            for (const auto &entry : std::filesystem::directory_iterator(directory)) {
                const auto name = entry.path().filename().string();
                if (name.size() >= 4 && name.compare(name.size()-4, 4, ".wav") == 0)
                    voices.push_back(name == ".wav" ? name : name.substr(0, name.size()-4));
            }
            std::sort(voices.begin(), voices.end());
        } catch (const std::filesystem::filesystem_error &) { voices.clear(); }
        for (const auto &voice : voices) {
            std::string decoded;
            if (!fallback_decode(voice, "utf-8", &decoded)) return plain_internal_server_error();
        }
        return {200, "application/json", "{\"voices\":" + string_array_json(voices) + "}"};
    } catch (...) { return plain_internal_server_error(); }
}

bool NativeServer::voice_server_available_cached(const std::string &voice_server, const std::string &api_key) const {
    const double now = resources_->current_time();
    {
        std::lock_guard<std::mutex> lock(voice_mutex_);
        if (voice_cache_valid_
            && voice_cache_url_ == voice_server
            && now - voice_cache_last_check_ < 30.0) {
            return voice_cache_available_;
        }
    }
    const auto available = fallback_http_request("GET",voice_server,"/v1/audio/voices",api_key,"application/json","",5,false,false,false,"",{},{},"Python/3.12 aiohttp/3.13.2");
    const bool ok = !available.transport_error && available.status == 200;
    {
        std::lock_guard<std::mutex> lock(voice_mutex_);
        voice_cache_valid_ = true;
        voice_cache_url_ = voice_server;
        voice_cache_last_check_ = now;
        voice_cache_available_ = ok;
    }
    return ok;
}

bool NativeServer::forward_voice_server_tts(
    const std::string &path,
    const std::string &body,
    HttpResponse *out,
    bool allow_deferred_streaming
) const {
    if (!out) {
        return false;
    }
    if (starts_with(extract_json_string(body, "input", ""), "data:")) {
        return false;
    }
    const std::string voice_server = voice_client().url;
    if (voice_server.empty()) {
        return false;
    }
    const std::string voice_api_key = voice_client().api_key;
    if (!voice_server_available_cached(voice_server, voice_api_key)) {
        return false;
    }

    if (path == "/v1/audio/speech/stream") {
        std::vector<std::pair<std::string, std::string>> stream_headers = {
            {"X-Audio-Format", "pcm"},
            {"X-Sample-Rate", "24000"},
            {"X-Bits-Per-Sample", "16"},
            {"X-Channels", "1"},
        };
        if (allow_deferred_streaming) {
            auto response = http_worker_stream_request(
                "POST",
                voice_server,
                path,
                voice_api_key,
                "application/json",
                voice_tts_forward_payload(body),
                300
            );
            if (!response.transport_error && response.status == 200) {
                response.status = 200;
                response.content_type = "application/octet-stream";
                response.headers = stream_headers;
                response.chunked = true;
                *out = std::move(response);
                return true;
            }
            *out = {200, "application/octet-stream", "", stream_headers};
            out->chunked = true;
            return true;
        }

        auto response = http_json_request("POST", voice_server, path, voice_api_key, voice_tts_forward_payload(body));
        if (response.status == 200) {
            *out = {200, "application/octet-stream", response.body, stream_headers};
        } else {
            *out = {200, "application/octet-stream", "", stream_headers};
        }
        out->chunked = true;
        return true;
    }
    auto response = http_json_request("POST", voice_server, path, voice_api_key, voice_tts_forward_payload(body));
    if (response.status == 200 && !response.body.empty()) {
        *out = {200, "audio/wav", response.body};
        return true;
    }
    return false;
}

bool NativeServer::forward_voice_server_transcription(
    const std::string &body,
    const std::string &content_type,
    HttpResponse *out
) const {
    if (!out) {
        return false;
    }
    const std::string voice_server = voice_client().url;
    if (voice_server.empty()) {
        return false;
    }
    const std::string voice_api_key = voice_client().api_key;
    if (!voice_server_available_cached(voice_server, voice_api_key)) {
        return false;
    }

    const std::string payload = build_voice_transcription_payload(body, content_type, false);
    if (payload.empty()) {
        return false;
    }
    const auto fields = parse_multipart_parts(body, extract_multipart_boundary(content_type));
    const auto response_format = multipart_value_or(fields, "response_format", "json");
    auto response = http_worker_request(
        "POST",
        voice_server,
        "/v1/audio/transcriptions",
        voice_api_key,
        std::string("multipart/form-data; boundary=") + kVoiceForwardBoundary,
        payload
    );
    if (response.status != 200 || trim(response.body) == "{}") {
        return false;
    }
    if (response_format == "text") {
        *out = {200, "text/plain; charset=utf-8", transcription_text_response_body(response.body)};
    } else {
        *out = response;
    }
    return true;
}

bool NativeServer::forward_fallback_tts(const std::string &body, HttpResponse *out) const {
    if (!out) {
        return false;
    }
    if (starts_with(extract_json_string(body, "input", ""), "data:")) {
        return false;
    }
    const auto fallback_decision = native_fallback_decision(resources_.get(), fallback_client().url);
    if (!fallback_decision.should_use || fallback_decision.url.empty()) {
        return false;
    }
    const std::string fallback_api_key = fallback_client().api_key;
    if (!fallback_server_available_cached(fallback_decision.url, fallback_api_key)) {
        return false;
    }

    auto response = fallback_json_request("POST", fallback_decision.url, "/v1/audio/speech", fallback_api_key, voice_tts_forward_payload(body), 120);
    if (response.status != 200) {
        return false;
    }
    *out = {200, "audio/wav", response.body};
    return true;
}

bool NativeServer::data_uri_audio_speech(const std::string &body, HttpResponse *out) const {
    if (!out) {
        return false;
    }
    const std::string input = extract_json_string(body, "input", "");
    const auto data_uri = parse_data_uri(input);
    if (!data_uri.ok) {
        return false;
    }

    std::string text;
    if (starts_with(data_uri.media_type, "audio/")) {
        const std::string boundary = "ezlocalai-native-data-uri-audio";
        std::ostringstream multipart;
        multipart
            << "--" << boundary << "\r\n"
            << "Content-Disposition: form-data; name=\"file\"; filename=\"input."
            << json_escape(data_uri.media_type.substr(6)) << "\"\r\n"
            << "Content-Type: " << data_uri.media_type << "\r\n\r\n"
            << data_uri.data << "\r\n"
            << "--" << boundary << "\r\n"
            << "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
            << "json\r\n"
            << "--" << boundary << "--\r\n";
        HttpResponse transcribed;
        if (!local_whisper_transcription(
                multipart.str(),
                "multipart/form-data; boundary=" + boundary,
                &transcribed
            )) {
            return false;
        }
        if (transcribed.status != 200) {
            *out = transcribed;
            return true;
        }
        text = whisper_text_from_output(transcribed.body);
    } else if (data_uri.media_type.find("pdf") != std::string::npos) {
        const std::string pdftotext_bin = find_executable_path({
            trim(ezl_getenv_default("PDFTOTEXT_BIN")),
            "/usr/bin/pdftotext",
            "pdftotext"
        });
        if (pdftotext_bin.empty()) {
            return false;
        }
        const auto pdf_path = temporary_named_path("ezlocalai-tts-pdf", ".pdf");
        write_file(pdf_path.string(), data_uri.data);
        const int timeout_seconds = parse_positive_int_env("PDFTOTEXT_TIMEOUT", 60);
        const auto extracted = run_process_capture({pdftotext_bin, pdf_path.string(), "-"}, timeout_seconds);
        std::error_code ec;
        std::filesystem::remove(pdf_path, ec);
        if (extracted.timed_out) {
            *out = {500, "application/json", make_error(500, "pdftotext timed out")};
            return true;
        }
        if (extracted.exit_code != 0 && extracted.output.empty()) {
            *out = {500, "application/json", make_error(500, "pdftotext failed")};
            return true;
        }
        text = trim(extracted.output);
        if (text.empty()) {
            *out = {200, "application/json", "null"};
            return true;
        }
    } else {
        return false;
    }

    const std::string tts_body = tts_text_payload_from_original(body, text);
    HttpResponse audio;
    if (!forward_voice_server_tts("/v1/audio/speech", tts_body, &audio)
        && !forward_fallback_tts(tts_body, &audio)) {
        return false;
    }
    if (audio.status != 200 || audio.body.empty()) {
        *out = audio;
        return true;
    }

    const std::string local_uri = strip_trailing_slashes(trim(config_.url));
    if (!local_uri.empty()) {
        const std::string filename = now_id() + "-" + short_random_hex() + ".wav";
        write_file(outputs_dir() + "/" + filename, audio.body);
        *out = {200, "application/json", "\"" + json_escape(local_uri + "/outputs/" + filename) + "\""};
        return true;
    }
    const std::string encoded = base64_encode(
        reinterpret_cast<const unsigned char *>(audio.body.data()),
        audio.body.size()
    );
    *out = {200, "application/json", "\"" + encoded + "\""};
    return true;
}

std::string NativeServer::transcribe_llm_audio(const std::string &audio, const std::string &format, bool return_segments) {
    if (stt_service_) {
        std::vector<std::pair<std::string,std::string>> request{{"base64_audio",audio},{"audio_format",format}};
        if (return_segments) request.emplace_back("return_segments","true");
        return stt_service_->transcribe(json_object_from_ordered_entries(request),false,0,
            return_segments ? SttServiceUse::LlmVideo : SttServiceUse::LlmAudio);
    }
    struct Cleanup { std::function<void()> callback; ~Cleanup() { if (callback) callback(); } };
    std::vector<std::string> models;
    for (const auto &id : llm_catalog_.available) models.push_back(llm_catalog_.source(id));
    const auto generic = std::string(ezl_getenv_fallback("VOICE_UNLOAD_LLM_DURING_GENERATION", "auto"));
    const bool handoff = llm_audio_handoff(generic, ezl_getenv_fallback("STT_UNLOAD_LLM_DURING_GENERATION", generic.c_str()),
        ezl_getenv_fallback("MEDIA_KEEP_MODELS_LOADED", "false"), voice_client().raw, ezl_getenv_default("TEXT_SERVER"), models);
    std::unique_lock<std::mutex> shared(llm_audio_handoff_mutex_, std::defer_lock);
    Cleanup handoff_cleanup;
    if (handoff) {
        shared.lock();
        { std::lock_guard<std::mutex> lock(resource_mutex_); llm_temporarily_unavailable_ = true; }
        handoff_cleanup.callback = [this] {
            std::lock_guard<std::mutex> lock(resource_mutex_); llm_temporarily_unavailable_ = false; llm_idle_cv_.notify_all();
        };
        std::string generic_timeout = ezl_getenv_fallback("VOICE_WAIT_FOR_LLM_IDLE_TIMEOUT", "60");
        if (generic_timeout.empty()) generic_timeout = "60";
        std::string configured = ezl_getenv_fallback("STT_WAIT_FOR_LLM_IDLE_TIMEOUT", generic_timeout.c_str());
        if (configured.empty()) configured = generic_timeout;
        double timeout = 0;
        if (!fallback_float(configured, &timeout))
            throw LlmResponseError{"ValueError", "could not convert string to float: " + python_repr_string(configured)};
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(resource_mutex_);
        // These calls are inside get_response's already-counted LLM lease.
        // The source waits on that same count and can time out before loading STT.
        while (llm_slots_.total > 0) {
            if (llm_slots_stopping_) throw LlmResponseError{"RuntimeError", "LLM slot scheduler is stopping"};
            if (timeout <= 0 || std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= std::max(0.0, timeout))
                throw LlmResponseError{"RuntimeError", "LLM inference is active; STT cannot unload the LLM yet"};
            llm_idle_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
    }
    {
        std::unique_lock<std::mutex> lock(llm_audio_mutex_);
        llm_audio_cv_.wait(lock, [this] { return llm_audio_stopping_ || llm_audio_active_ < llm_audio_capacity_; });
        if (llm_audio_stopping_) throw LlmResponseError{"RuntimeError", "STT admission is stopping"};
        ++llm_audio_active_;
    }
    Cleanup lease{[this] { std::lock_guard<std::mutex> lock(llm_audio_mutex_); --llm_audio_active_; llm_audio_cv_.notify_one(); }};
    ScopedCapabilityUse stt_use(this, "stt");
    const auto executable = find_executable_path({trim(ezl_getenv_default("WHISPER_CPP_BIN")), trim(ezl_getenv_default("WHISPER_CPP_CLI")),
        "/usr/local/bin/whisper-cli", "/app/whispercpp/whisper-cli", "/app/whispercpp/main", "whisper-cli"});
    const auto model = resolve_whisper_model_path();
    if (executable.empty() || model.empty()) throw LlmResponseError{"RuntimeError", "Local STT engine is unavailable"};
    const auto input = stt_audio_input(audio, format);
    const auto directory = std::filesystem::path(outputs_dir());
    std::filesystem::create_directories(directory);
    auto audio_id = uuid_v4();
    audio_id.erase(std::remove(audio_id.begin(), audio_id.end(), '-'), audio_id.end());
    const auto audio_path = directory / (audio_id + "." + input.extension);
    const auto output = temporary_named_path("ezlocalai-llm-stt-output");
    Cleanup files{[&] { std::error_code ignored; std::filesystem::remove(audio_path, ignored); std::filesystem::remove(output.string() + ".json", ignored); }};
    { std::ofstream file(audio_path, std::ios::binary); file.write(input.bytes.data(), input.bytes.size()); file.close(); if (!file) throw std::runtime_error("Failed to write audio"); }
    std::vector<std::string> arguments{executable, "-m", model, "-f", audio_path.string(), "-oj", "-of", output.string(), "-bs", "1"};
    for (const auto &arg : split_command_words(ezl_getenv_default("WHISPER_CPP_EXTRA_ARGS"))) arguments.push_back(arg);
    const auto response = run_process_capture(arguments, parse_positive_int_env("WHISPER_CPP_TIMEOUT", 600));
    if (response.timed_out) throw std::runtime_error("whisper.cpp transcription timed out");
    if (response.exit_code) throw std::runtime_error("whisper.cpp transcription failed");
    auto result = read_file_or_empty(output.string() + ".json"); if (result.empty()) result = response.output;
    if (json_decode_error(result)) throw std::runtime_error("Invalid STT engine response");
    std::string text;
    const auto segments = json_object_field(result, "transcription", json_object_field(result, "segments", ""));
    if (!segments.empty() && llm_output_type(segments) == "list") {
        for (const auto &segment : json_array_elements(segments)) {
            const auto value = llm_output_get(segment, "text", "\"\"");
            if (llm_output_type(value) != "str") throw LlmResponseError{"TypeError", "STT segment text is not a string"};
            text += parse_json_string_at(value, 0);
        }
    } else {
        const auto value = json_object_field(result, "text", "\"\"");
        if (llm_output_type(value) != "str") throw LlmResponseError{"TypeError", "STT text is not a string"};
        text = parse_json_string_at(value, 0);
    }
    if (return_segments) {
        const auto native_segments = json_object_field(result, "segments", "");
        return "{\"text\":\"" + json_escape(strip_config_value(text)) + "\",\"segments\":"
            + (native_segments.empty() ? whisper_segments_json(whisper_segments_from_output(result)) : native_segments)
            + ",\"language\":" + json_object_field(result, "language", "null") + '}';
    }
    return '"' + json_escape(text) + '"';
}

bool NativeServer::append_voice_audio_to_llm_response(
    const std::string &kind,
    const std::string &request_body,
    std::string *response_body
) const {
    if (!response_body) {
        return false;
    }
    const auto request_values = top_level_json_object_values(request_body);
    if (request_values.find("voice") == request_values.end()) {
        return false;
    }

    const auto choices = extract_json_array_elements(*response_body, "choices");
    if (choices.empty()) {
        return false;
    }

    std::string text;
    if (kind == "chat") {
        const std::string message = trim(extract_json_value(choices.front(), "message", "{}"));
        text = extract_json_string(message, "content", "");
    } else if (kind == "completion") {
        text = extract_json_string(choices.front(), "text", "");
    }
    if (text.empty()) {
        return false;
    }

    const std::string tts_body = tts_text_payload_from_original(request_body, text);
    HttpResponse audio;
    {
        if (!forward_voice_server_tts("/v1/audio/speech", tts_body, &audio)
            && !forward_fallback_tts(tts_body, &audio)) {
            return false;
        }
    }
    if (audio.status != 200 || audio.body.empty()) {
        return false;
    }

    std::string audio_response;
    const std::string local_uri = strip_trailing_slashes(trim(config_.url));
    if (!local_uri.empty()) {
        const std::string filename = now_id() + "-" + short_random_hex() + ".wav";
        write_file(outputs_dir() + "/" + filename, audio.body);
        audio_response = local_uri + "/outputs/" + filename;
    } else {
        audio_response = base64_encode(
            reinterpret_cast<const unsigned char *>(audio.body.data()),
            audio.body.size()
        );
    }

    const std::string appended = text + "\n" + audio_response;
    *response_body = kind == "chat"
        ? replace_last_json_string_field(*response_body, "content", appended)
        : replace_json_string_field(*response_body, "text", appended);
    return true;
}

bool NativeServer::forward_fallback_transcription(
    const std::string &body,
    const std::string &content_type,
    HttpResponse *out
) const {
    if (!out) {
        return false;
    }
    const auto fallback_decision = native_fallback_decision(resources_.get(), fallback_client().url);
    if (!fallback_decision.should_use || fallback_decision.url.empty()) {
        return false;
    }
    const std::string fallback_api_key = fallback_client().api_key;
    if (!fallback_server_available_cached(fallback_decision.url, fallback_api_key)) {
        return false;
    }

    const std::string payload = build_voice_transcription_payload(body, content_type, true);
    if (payload.empty()) {
        return false;
    }
    const auto fields = parse_multipart_parts(body, extract_multipart_boundary(content_type));
    const auto response_format = multipart_value_or(fields, "response_format", "json");
    auto response = fallback_http_request(
        "POST",
        fallback_decision.url,
        "/v1/audio/transcriptions",
        fallback_api_key,
        std::string("multipart/form-data; boundary=") + kVoiceForwardBoundary,
        payload,
        120
    );
    if (response.status != 200 || trim(response.body) == "{}") {
        return false;
    }
    if (response_format == "text") {
        *out = {200, "text/plain; charset=utf-8", transcription_text_response_body(response.body)};
    } else {
        *out = response;
    }
    return true;
}

const NativeServer::VoiceClientSettings &NativeServer::voice_client() const {
    std::call_once(voice_client_once_,[this]{
        auto &client=voice_client_settings_;client.raw=ezl_getenv_default("VOICE_SERVER");
        client.dedicated=llm_unicode_lower(client.raw)=="true";
        client.url=client.dedicated?"":client.raw;
        while(!client.url.empty()&&client.url.back()=='/')client.url.pop_back();
        client.api_key=ezl_getenv_default("VOICE_SERVER_API_KEY");
        if(client.api_key.empty())client.api_key=ezl_getenv_default("EZLOCALAI_API_KEY");
    });
    return voice_client_settings_;
}
const ServiceClientSettings &NativeServer::image_client() const {
    std::call_once(image_client_once_, [this] { image_client_settings_ = capture_service_client_settings("IMAGE_SERVER"); });
    return image_client_settings_;
}
const ServiceClientSettings &NativeServer::text_client() const {
    std::call_once(text_client_once_, [this] { text_client_settings_ = capture_service_client_settings("TEXT_SERVER"); });
    return text_client_settings_;
}
const ServiceClientSettings &NativeServer::embedding_client() const {
    std::call_once(embedding_client_once_, [this] { embedding_client_settings_ = capture_service_client_settings("EMBEDDING_SERVER"); });
    return embedding_client_settings_;
}
std::shared_ptr<TtsRemoteClient> NativeServer::tts_remote_client(TtsProvider provider) const {
    auto &once = provider==TtsProvider::Voice ? tts_voice_once_ : provider==TtsProvider::Router ? tts_router_once_ : tts_general_once_;
    auto &stored = provider==TtsProvider::Voice ? tts_voice_client_ : provider==TtsProvider::Router ? tts_router_client_ : tts_general_client_;
    std::call_once(once,[this,provider,&stored] {
        std::string url,key;
        if (provider==TtsProvider::Voice) {
            url=voice_client().url;key=voice_client().api_key;
        } else if (provider==TtsProvider::Router) {
            url=strip_config_value(ezl_getenv_default("ROUTER_URL"));
            if(llm_unicode_lower(strip_config_value(ezl_getenv_default("ROUTER_MODE")))=="true")url.clear();
            for(const auto *name:{"ROUTER_CLIENT_API_KEY","EZLOCALAI_API_KEY","ROUTER_API_KEY"}) {
                key=strip_config_value(ezl_getenv_default(name));
                if(!key.empty())break;
            }
        } else { url=fallback_client().url; key=fallback_client().api_key; }
        while(!url.empty()&&url.back()=='/')url.pop_back();
        auto client=std::make_shared<TtsRemoteClient>();client->configured=!url.empty();
        struct Health {std::mutex mutex;std::optional<bool> available;double checked=0;};
        auto health=std::make_shared<Health>();
        client->available=[this,provider,url,key,health] {
            if(provider==TtsProvider::Voice)return voice_server_available_cached(url,key);
            if(provider==TtsProvider::General)return fallback_server_available_cached(url,key);
            const double now=resources_->current_time();
            {std::lock_guard<std::mutex> lock(health->mutex);if(health->available&&now-health->checked<30)return *health->available;}
            bool available=false;
            try {
                const auto response=fallback_http_request("GET",url,provider==TtsProvider::Voice?"/v1/audio/voices":"/v1/router/health",key,"application/json","",5,false,provider!=TtsProvider::Voice,false,"",{},{},"Python/3.12 aiohttp/3.13.2");
                if(response.status==200&&!response.transport_error) {
                    if(provider==TtsProvider::Voice)available=true;
                    else {
                        std::string data;
                        if(fallback_json_response(response,&data)&&!data.empty()&&data.front()=='{') {
                            const auto value=json_object_field(data,"alive_workers","0");
                            if(!is_python_falsey_json_value(value)) {
                                if(!value.empty()&&value.front()=='"')available=configuration_integer_less("0",llm_parameter_integer(parse_json_string_at(value,0)));
                                else if(value=="true")available=true;
                                else if(is_json_number_token(value)) {
                                    if(value.find_first_of(".eE")==std::string::npos)available=configuration_integer_less("0",llm_parameter_integer(value));
                                    else {double count=0;available=parse_strict_python_float_string(value,&count)&&std::isfinite(count)&&count>=1;}
                                }
                            }
                        }
                    }
                }
            } catch(...) {available=false;}
            {std::lock_guard<std::mutex> lock(health->mutex);health->available=available;health->checked=now;}
            return available;
        };
        client->generate=[provider,url,key](const std::string &request)->std::optional<std::string> {
            const auto response=fallback_http_request("POST",url,"/v1/audio/speech",key,"application/json",tts_remote_payload(request),120,true,true,false,"",{},{},"Python/3.12 aiohttp/3.13.2");
            if(!response.transport_error&&response.status==200)return response.body;
            if(provider==TtsProvider::Voice)return {};
            throw TtsError("RuntimeError","Fallback TTS request failed");
        };
        if(provider==TtsProvider::Voice)client->voices=[url,key]()->std::optional<std::string> {
            try {
                const auto response=fallback_http_request("GET",url,"/v1/audio/voices",key,"application/json","",10,false,true,false,"",{},{},"Python/3.12 aiohttp/3.13.2");
                std::string value;
                if(!response.transport_error&&response.status==200&&fallback_json_response(response,&value))return value;
            } catch (...) {}
            return {};
        };
        client->stream=[provider,url,key](const std::string &request,const TtsEmit &emit,const TtsInterrupt &interrupt) {
            const auto response=fallback_http_request("POST",url,"/v1/audio/speech/stream",key,"application/json",tts_remote_payload(request),300,true,true,false,"",emit,interrupt,"Python/3.12 aiohttp/3.13.2");
            if((response.transport_error||response.status!=200)&&provider!=TtsProvider::Voice)throw TtsError("RuntimeError","Fallback TTS stream failed");
        };
        stored=std::move(client);
    });
    return stored;
}

HttpResponse NativeServer::routed_speech(const std::string &body,bool streaming,bool deferred) {
    try {
        TtsRoutingOperations operations;
        operations.disabled=[]{return llm_unicode_lower(ezl_getenv_default("TTS_ENABLED"))=="false";};
        operations.voice_server_mode=[this]{return voice_client().dedicated;};
        operations.local_capability=[]{return has_local_capability("tts");};
        operations.general_policy=[this]{return native_fallback_decision(resources_.get(),fallback_client().url).should_use;};
        operations.client=[this](auto provider){return tts_remote_client(provider);};
        operations.uuid=[]{auto value=uuid_v4();value.erase(std::remove(value.begin(),value.end(),'-'),value.end());return value;};
        auto route=route_tts(body,streaming,operations);
        if(route.kind==TtsRouteKind::Disabled)return {404,"application/json","{\"detail\":\"Text to speech is disabled.\"}"};
        if(route.kind==TtsRouteKind::Buffered)return {200,"audio/wav",std::move(route.bytes)};
        if(route.kind==TtsRouteKind::Audio) {
            TtsAudioPipelineOperations pipeline;
            pipeline.transcribe=[this](const auto &request,const auto &audio){if(!stt_service_)throw TtsError("RuntimeError","Native STT service is unavailable");return stt_service_->transcribe_bytes(request,audio);};
            pipeline.generate=[this](const auto &request){if(!tts_service_)throw TtsError("RuntimeError","Native TTS service is unavailable");return tts_service_->generate_pipeline(request);};
            return {200,"application/json",'"'+json_escape(tts_audio_to_audio(route.request,'"'+json_escape(config_.url)+'"',pipeline))+'"'};
        }
        if(route.kind==TtsRouteKind::Pdf) {
            HttpResponse result;
            if(data_uri_audio_speech(body,&result))return result;
            return plain_internal_server_error();
        }
        if(route.kind==TtsRouteKind::Local) {
            HttpResponse result;
            if(local_native_speech(body,streaming,deferred,&result))return result;
            return {501,"application/json",make_error(501,"this native backend does not implement the requested modality yet")};
        }
        HttpResponse result{200,"application/octet-stream","",{{"X-Audio-Format","pcm"},{"X-Sample-Rate","24000"},{"X-Bits-Per-Sample","16"},{"X-Channels","1"}}};result.chunked=true;
        if(!deferred) {
            try{route.produce([&](const auto &bytes){result.body+=bytes;return true;},{});}catch(...){result.transport_error=true;}
            return result;
        }
        struct Control {
            std::mutex mutex;std::function<void()> interrupt;std::atomic<bool> cancelled{false};
            void cancel(){std::function<void()> action;{std::lock_guard<std::mutex> lock(mutex);cancelled=true;action=interrupt;}if(action)action();}
            void install(std::function<void()> action){bool stopped;{std::lock_guard<std::mutex> lock(mutex);interrupt=action;stopped=cancelled;}if(stopped&&action)action();}
        };
        auto control=std::make_shared<Control>();
        {std::lock_guard<std::mutex> lock(queued_forward_mutex_);
            local_streams_.erase(std::remove_if(local_streams_.begin(),local_streams_.end(),[](const auto &entry){return entry.first.expired();}),local_streams_.end());
            local_streams_.push_back({control,[weak=std::weak_ptr<Control>(control)]{if(auto active=weak.lock())active->cancel();}});
        }
        result.stream_capture=std::make_shared<std::string>();
        result.stream_writer=[produce=std::move(route.produce),control,capture=result.stream_capture](int fd) {
            if(control->cancelled)return;
            std::mutex mutex;std::condition_variable wake;bool finished=false;
            std::thread watcher([&]{std::unique_lock<std::mutex> lock(mutex);while(!wake.wait_for(lock,std::chrono::milliseconds(25),[&]{return finished;})){lock.unlock();pollfd ready{fd,POLLIN|POLLRDHUP,0};poll(&ready,1,0);char byte;const auto count=recv(fd,&byte,1,MSG_PEEK|MSG_DONTWAIT);if((ready.revents&(POLLHUP|POLLRDHUP|POLLERR|POLLNVAL))||count==0||(count<0&&errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR)){control->cancel();return;}lock.lock();}});
            struct Join {std::thread &thread;std::mutex &mutex;std::condition_variable &wake;bool &finished;~Join(){{std::lock_guard<std::mutex> lock(mutex);finished=true;}wake.notify_all();thread.join();}} join{watcher,mutex,wake,finished};
            bool complete=false;
            try {
                produce([&](const auto &bytes){if(control->cancelled)return false;append_stream_capture(capture,bytes);if(send_http_chunk(fd,bytes))return true;control->cancel();return false;},[control](auto interrupt){control->install(std::move(interrupt));});
                complete=!control->cancelled;
            } catch(...) {control->cancel();}
            if(complete)send_all_fd(fd,"0\r\n\r\n");
        };
        return result;
    } catch(...) {return plain_internal_server_error();}
}

bool NativeServer::native_tts_handoff() const {
    std::vector<std::string> models;for(const auto &id:llm_catalog_.available)models.push_back(llm_catalog_.source(id));
    const auto generic=std::string(ezl_getenv_fallback("VOICE_UNLOAD_LLM_DURING_GENERATION","auto"));
    return llm_audio_handoff(generic,ezl_getenv_fallback("TTS_UNLOAD_LLM_DURING_GENERATION",generic.c_str()),ezl_getenv_fallback("MEDIA_KEEP_MODELS_LOADED","false"),voice_client().raw,ezl_getenv_default("TEXT_SERVER"),models);
}
bool NativeServer::native_tts_preload() const {
    bool fast=!llm_catalog_.available.empty();for(const auto &id:llm_catalog_.available)if(!llm_minicpm5_model(llm_catalog_.source(id)))fast=false;
    return (to_lower(ezl_getenv_default("LAZY_LOAD_VOICE"))=="false"||voice_client().dedicated||fast||env_truthy("MEDIA_KEEP_MODELS_LOADED"))&&!native_tts_handoff();
}
bool NativeServer::local_native_speech(const std::string &body,bool streaming,bool deferred,HttpResponse *out) {
    if(!tts_service_||!out)return false;
    const auto request=json_object_from_ordered_entries({{"text",json_object_field(body,"input","\"\"")},{"voice",json_object_field(body,"voice","\"default\"")},{"language",json_object_field(body,"language","\"auto\"")}});
    try {
        if(!streaming){*out={200,"audio/wav",tts_service_->generate(request)};return true;}
        // Acquiring the model must finish before committing the response.
        auto stream=std::shared_ptr<TtsServiceStream>(tts_service_->stream(request));
        *out={200,"application/octet-stream","",{{"X-Audio-Format","pcm"},{"X-Sample-Rate","24000"},{"X-Bits-Per-Sample","16"},{"X-Channels","1"}}};out->chunked=true;
        if(!deferred){try{while(auto chunk=stream->next())out->body+=*chunk;stream->close();}catch(...){out->transport_error=true;try{stream->close();}catch(...){}}return true;}
        {
            std::lock_guard<std::mutex> lock(queued_forward_mutex_);
            local_streams_.erase(std::remove_if(local_streams_.begin(),local_streams_.end(),[](const auto &entry){return entry.first.expired();}),local_streams_.end());
            local_streams_.push_back({stream,[weak=std::weak_ptr<TtsServiceStream>(stream)]{if(const auto current=weak.lock()){current->request_cancel();try{current->close();}catch(...){}}}});
        }
        out->stream_capture=std::make_shared<std::string>();
        out->stream_writer=[stream,capture=out->stream_capture](int fd){
            std::mutex mutex;std::condition_variable wake;bool finished=false;
            std::thread watcher([&]{std::unique_lock<std::mutex> lock(mutex);while(!wake.wait_for(lock,std::chrono::milliseconds(25),[&]{return finished;})){lock.unlock();pollfd ready{fd,POLLIN|POLLRDHUP,0};poll(&ready,1,0);char byte;const auto count=recv(fd,&byte,1,MSG_PEEK|MSG_DONTWAIT);if((ready.revents&(POLLHUP|POLLRDHUP|POLLERR|POLLNVAL))||count==0||(count<0&&errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR)){stream->request_cancel();return;}lock.lock();}});
            struct Join {std::thread &thread;std::mutex &mutex;std::condition_variable &wake;bool &finished;~Join(){{std::lock_guard<std::mutex> lock(mutex);finished=true;}wake.notify_all();thread.join();}} join{watcher,mutex,wake,finished};
            bool complete=false;
            try{while(auto chunk=stream->next()){append_stream_capture(capture,*chunk);if(!send_http_chunk(fd,*chunk)){stream->request_cancel();stream->close();return;}}stream->close();complete=true;}catch(...){stream->request_cancel();try{stream->close();}catch(...){}}
            // A post-header exception truncates the transfer, as StreamingResponse does.
            if(complete)send_all_fd(fd,"0\r\n\r\n");
        };
    } catch(...) {*out=plain_internal_server_error();}
    return true;
}
bool NativeServer::native_stt_handoff() const {
    std::vector<std::string> models; for (const auto &id : llm_catalog_.available) models.push_back(llm_catalog_.source(id));
    const auto generic = std::string(ezl_getenv_fallback("VOICE_UNLOAD_LLM_DURING_GENERATION","auto"));
    return llm_audio_handoff(generic,ezl_getenv_fallback("STT_UNLOAD_LLM_DURING_GENERATION",generic.c_str()),ezl_getenv_fallback("MEDIA_KEEP_MODELS_LOADED","false"),voice_client().raw,ezl_getenv_default("TEXT_SERVER"),models);
}
bool NativeServer::native_stt_preload() const {
    bool fast = !llm_catalog_.available.empty();
    for (const auto &id : llm_catalog_.available) if (!llm_minicpm5_model(llm_catalog_.source(id))) fast = false;
    return (to_lower(ezl_getenv_default("LAZY_LOAD_VOICE")) == "false" || voice_client().dedicated || fast || env_truthy("MEDIA_KEEP_MODELS_LOADED")) && !native_stt_handoff();
}
std::string NativeServer::direct_llm_chat(const std::string &body) {
    // app.py's subtitle helper calls Pipes.get_response directly: no HTTP
    // defaults, queue admission, queue timeout or API response conversion.
    const auto key = uuid_v4(); auto target = std::make_shared<QueuedForwardResponse>(); target->internal = true;
    { std::lock_guard<std::mutex> lock(queued_forward_mutex_); queued_forward_responses_[key] = target; }
    auto registration = std::shared_ptr<int>(new int(0),[this,key](int *value) { delete value; std::lock_guard<std::mutex> lock(queued_forward_mutex_); queued_forward_responses_.erase(key); });
    char unused[1]{}; queue_processor(("chat\n" + key + '\n' + body).c_str(),unused,sizeof(unused),this);
    std::lock_guard<std::mutex> lock(target->mutex);
    if (target->failure) std::rethrow_exception(target->failure);
    if (target->response) { if (target->response->status >= 400) throw std::runtime_error("Subtitle translation failed"); return target->response->body; }
    if (target->stream) { target->stream->close(); return "null"; }
    return target->local_body.value_or("null");
}
bool NativeServer::local_native_subtitles(const std::string &body,const std::string &content_type,HttpResponse *out) {
    if (!stt_service_ || !out) return false;
    try {
        const auto parts = parse_multipart_parts(body,extract_multipart_boundary(content_type)); const auto *file = find_multipart_part(parts,"file"); if (!file) return false;
        const auto field = [&](const std::string &name,const std::string &fallback) { std::string value = fallback; for (const auto &part : parts) if (part.name == name) value = part.content.empty() ? fallback : part.content; return value; };
        const auto nullable = [&](const std::string &name) { const auto value = field(name,""); return value.empty() ? std::string("null") : '"' + json_escape(value) + '"'; };
        double temperature = 0; fallback_float(field("temperature","0"),&temperature); bool fast = false; parse_pydantic_bool_string(field("fast_mode","false"),&fast);
        auto beam = trim(field("beam_size","5")); const auto decimal = beam.find('.'); if (decimal != std::string::npos) beam.resize(decimal);
        const auto request = json_object_from_ordered_entries({{"base64_audio",'"' + base64_encode(reinterpret_cast<const unsigned char *>(file->content.data()),file->content.size()) + '"'},
            {"audio_format",file->content_type.empty() ? "null" : '"' + json_escape(file->content_type) + '"'}, {"language",nullable("source_language")},{"prompt",nullable("prompt")},{"temperature",json_number_literal(temperature)},
            {"return_segments","true"},{"beam_size",fast ? "1" : llm_parameter_integer(beam)},{"condition_on_previous_text",fast ? "false" : "true"}});
        const auto transcription = stt_service_->transcribe(request,false);
        *out = {200,"application/json",generate_native_subtitles(transcription,field("source_language",""),field("target_languages","en"),field("format","vtt"),ezl_getenv_default("DEFAULT_MODEL"),[this](const auto &request) { return direct_llm_chat(request); })};
    } catch (...) { *out = {500,"text/plain; charset=utf-8","Internal Server Error"}; }
    return true;
}
bool NativeServer::local_native_transcription(const std::string &body,const std::string &content_type,HttpResponse *out,bool translate) {
    if (!stt_service_ || !out) return false;
    try {
        const auto parts = parse_multipart_parts(body,extract_multipart_boundary(content_type)); const auto *file = find_multipart_part(parts,"file"); if (!file) return false;
        auto field = [&](const std::string &name,const std::string &fallback) { std::string value = fallback; for (const auto &part : parts) if (part.name == name) value = part.content.empty() ? fallback : part.content; return value; };
        auto nullable = [&](const std::string &name) { const auto value = field(name,""); return value.empty() ? std::string("null") : '"' + json_escape(value) + '"'; };
        double temperature = 0; fallback_float(field("temperature","0"),&temperature);
        std::vector<std::pair<std::string,std::string>> request{{"base64_audio",'"' + base64_encode(reinterpret_cast<const unsigned char *>(file->content.data()),file->content.size()) + '"'},
            {"audio_format",file->content_type.empty() ? "null" : '"' + json_escape(file->content_type) + '"'}, {"language",nullable("language")},{"prompt",nullable("prompt")},{"temperature",json_number_literal(temperature)}};
        const auto format = field("response_format","json");
        if (translate) request.emplace_back("translate","true");
        else {
            bool diarize = false; parse_pydantic_bool_string(field("enable_diarization","false"),&diarize);
            std::vector<std::string> granularities,bracketed; for (const auto &part : parts) { if (part.name == "timestamp_granularities") granularities.push_back('"' + json_escape(part.content) + '"'); if (part.name == "timestamp_granularities[]") bracketed.push_back('"' + json_escape(part.content) + '"'); }
            if (!bracketed.empty()) granularities = bracketed; if (granularities.empty()) granularities.push_back("\"segment\"");
            request.emplace_back("return_segments",(format == "verbose_json" || format == "srt" || format == "vtt" || diarize) ? "true" : "false");
            request.emplace_back("timestamp_granularities",json_array_from_values(granularities)); request.emplace_back("enable_diarization",diarize ? "true" : "false");
            auto speakers = trim(field("num_speakers","")); const auto decimal = speakers.find('.'); if (decimal != std::string::npos) speakers.resize(decimal);
            request.emplace_back("num_speakers",speakers.empty() ? "null" : llm_parameter_integer(speakers)); request.emplace_back("session_id",nullable("session_id"));
        }
        const auto result = stt_service_->transcribe(json_object_from_ordered_entries(request),!translate);
        const auto text = llm_output_type(result) == "dict" ? llm_output_get(result,"text","null") : result;
        if (translate) *out = {200,"application/json","{\"text\":" + result + '}'};
        else if (format == "text") { if (llm_output_type(text) != "str") throw std::runtime_error("Invalid STT text response"); *out = {200,"text/plain; charset=utf-8",parse_json_string_at(text,0)}; }
        else if (format == "verbose_json") *out = {200,"application/json",result};
        else if (format == "srt" || format == "vtt") *out = {200,format == "vtt" ? "text/vtt; charset=utf-8" : "text/plain; charset=utf-8",stt_segments_subtitles(llm_output_get(result,"segments","null"),format == "vtt")};
        else *out = {200,"application/json","{\"text\":" + text + '}'};
    } catch (const SttRequestError &error) { *out = {error.status,"application/json","{\"detail\":\"" + json_escape(error.what()) + "\"}"}; }
    catch (...) { *out = {500,"text/plain; charset=utf-8","Internal Server Error"}; }
    return true;
}

bool NativeServer::local_whisper_transcription(
    const std::string &body,
    const std::string &content_type,
    HttpResponse *out,
    bool translate
) const {
    if (!out) {
        return false;
    }
    const std::string whisper_bin = find_executable_path({
        trim(ezl_getenv_default("WHISPER_CPP_BIN")),
        trim(ezl_getenv_default("WHISPER_CPP_CLI")),
        "/usr/local/bin/whisper-cli",
        "/app/whispercpp/whisper-cli",
        "/app/whispercpp/main",
        "whisper-cli"
    });
    const std::string model_path = resolve_whisper_model_path();
    if (whisper_bin.empty() || model_path.empty()) {
        return false;
    }

    const auto parts = parse_multipart_parts(body, extract_multipart_boundary(content_type));
    const auto *file = find_multipart_part(parts, "file");
    if (!file) {
        return false;
    }
    const std::string response_format = multipart_value_or(parts, "response_format", "json");
    const std::string language = multipart_value_or(parts, "language", "");
    const std::string prompt = multipart_value_or(parts, "prompt", "");
    bool enable_diarization = false;
    parse_pydantic_bool_string(multipart_value_or(parts, "enable_diarization", "false"), &enable_diarization);
    int num_speakers = 0;
    parse_pydantic_integral_string(multipart_value_or(parts, "num_speakers", "0"), &num_speakers);
    const std::string session_id = multipart_value_or(parts, "session_id", "");

    const auto audio_path = temporary_named_path("ezlocalai-stt-audio");
    const auto output_prefix = temporary_named_path("ezlocalai-stt-output");
    write_file(audio_path.string(), file->content);

    std::vector<std::string> args = {
        whisper_bin,
        "-m",
        model_path,
        "-f",
        audio_path.string(),
        "-oj",
        "-of",
        output_prefix.string()
    };
    if (!language.empty()) {
        args.push_back("-l");
        args.push_back(language);
    }
    if (!prompt.empty()) {
        args.push_back("--prompt");
        args.push_back(prompt);
    }
    if (translate) {
        args.push_back("-tr");
    }
    append_whisper_temperature_arg(parts, &args);
    args.push_back("-bs");
    args.push_back("1");
    for (const auto &arg : split_command_words(ezl_getenv_default("WHISPER_CPP_EXTRA_ARGS"))) {
        args.push_back(arg);
    }

    ScopedCapabilityUse stt_use(this, "stt");
    const int timeout_seconds = parse_positive_int_env("WHISPER_CPP_TIMEOUT", 600);
    const auto result = run_process_capture(args, timeout_seconds);
    const std::string output_json_path = output_prefix.string() + ".json";
    std::string raw_output = read_file_or_empty(output_json_path);
    if (raw_output.empty()) {
        raw_output = result.output;
    }
    std::error_code ec;
    std::filesystem::remove(audio_path, ec);
    std::filesystem::remove(output_json_path, ec);

    if (result.timed_out) {
        *out = {500, "application/json", make_error(500, "whisper.cpp transcription timed out")};
        return true;
    }
    if (result.exit_code != 0 && raw_output.empty()) {
        *out = {500, "application/json", make_error(500, "whisper.cpp transcription failed")};
        return true;
    }
    auto segments = whisper_segments_from_output(raw_output);
    if (enable_diarization) {
        if (!apply_wav_feature_diarization(file->content, &segments, num_speakers, session_id)) {
            apply_single_speaker_diarization(&segments, session_id);
        }
    }
    const std::string text = enable_diarization
        ? whisper_diarized_text(segments)
        : whisper_text_from_output(raw_output);
    if (translate) {
        *out = {200, "application/json", "{\"text\":\"" + json_escape(text) + "\"}"};
        return true;
    }
    if (response_format == "text") {
        *out = {200, "text/plain; charset=utf-8", text};
        return true;
    }
    if (response_format == "verbose_json") {
        *out = {200, "application/json", "{\"text\":\"" + json_escape(text) + "\",\"segments\":" + whisper_segments_json(segments) + "}"};
        return true;
    }
    if (response_format == "srt") {
        *out = {200, "text/plain; charset=utf-8", whisper_segments_srt(segments)};
        return true;
    }
    if (response_format == "vtt") {
        *out = {200, "text/vtt; charset=utf-8", whisper_segments_vtt(segments)};
        return true;
    }
    *out = {200, "application/json", "{\"text\":\"" + json_escape(text) + "\"}"};
    return true;
}

bool NativeServer::local_whisper_subtitles(
    const std::string &body,
    const std::string &content_type,
    HttpResponse *out
) const {
    if (!out) {
        return false;
    }
    const std::string whisper_bin = find_executable_path({
        trim(ezl_getenv_default("WHISPER_CPP_BIN")),
        trim(ezl_getenv_default("WHISPER_CPP_CLI")),
        "/usr/local/bin/whisper-cli",
        "/app/whispercpp/whisper-cli",
        "/app/whispercpp/main",
        "whisper-cli"
    });
    const std::string model_path = resolve_whisper_model_path();
    if (whisper_bin.empty() || model_path.empty()) {
        return false;
    }

    const auto parts = parse_multipart_parts(body, extract_multipart_boundary(content_type));
    const auto *file = find_multipart_part(parts, "file");
    if (!file) {
        return false;
    }
    const std::string source_language = multipart_value_or(parts, "source_language", "");
    const std::string target_languages = multipart_value_or(parts, "target_languages", "en");
    const std::string format = multipart_value_or(parts, "format", "vtt");
    const std::string prompt = multipart_value_or(parts, "prompt", "");

    const auto audio_path = temporary_named_path("ezlocalai-subtitles-audio");
    const auto output_prefix = temporary_named_path("ezlocalai-subtitles-output");
    write_file(audio_path.string(), file->content);

    std::vector<std::string> args = {
        whisper_bin,
        "-m",
        model_path,
        "-f",
        audio_path.string(),
        "-oj",
        "-of",
        output_prefix.string()
    };
    if (!source_language.empty()) {
        args.push_back("-l");
        args.push_back(source_language);
    }
    if (!prompt.empty()) {
        args.push_back("--prompt");
        args.push_back(prompt);
    }
    append_whisper_temperature_arg(parts, &args);
    append_whisper_subtitle_decode_args(parts, &args);
    for (const auto &arg : split_command_words(ezl_getenv_default("WHISPER_CPP_EXTRA_ARGS"))) {
        args.push_back(arg);
    }

    ScopedCapabilityUse stt_use(this, "stt");
    const int timeout_seconds = parse_positive_int_env("WHISPER_CPP_TIMEOUT", 600);
    const auto result = run_process_capture(args, timeout_seconds);
    const std::string output_json_path = output_prefix.string() + ".json";
    std::string raw_output = read_file_or_empty(output_json_path);
    if (raw_output.empty()) {
        raw_output = result.output;
    }
    std::error_code ec;
    std::filesystem::remove(audio_path, ec);
    std::filesystem::remove(output_json_path, ec);

    if (result.timed_out) {
        *out = {500, "application/json", make_error(500, "whisper.cpp subtitle generation timed out")};
        return true;
    }
    if (result.exit_code != 0 && raw_output.empty()) {
        *out = {500, "application/json", make_error(500, "whisper.cpp subtitle generation failed")};
        return true;
    }

    std::string detected_language = extract_json_string(raw_output, "language", "");
    if (detected_language.empty()) {
        detected_language = source_language.empty() ? "en" : source_language;
    }
    std::vector<std::string> languages;
    size_t start = 0;
    while (start <= target_languages.size()) {
        const size_t comma = target_languages.find(',', start);
        languages.push_back(trim(target_languages.substr(start, comma == std::string::npos ? std::string::npos : comma - start)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (languages.empty()) {
        languages.push_back("en");
    }

    const auto segments = whisper_segments_from_output(raw_output);
    auto translate_segments = [&](const std::string &target_language) {
        if (target_language == detected_language) {
            return segments;
        }

        std::vector<WhisperSegment> translated;
        translated.reserve(segments.size());
        const auto models = llm_catalog_.available;
        const std::string model = models.empty() ? trim(config_.default_model) : llm_catalog_.source(models.front());
        if (model.empty()) {
            return segments;
        }

        size_t batch_start = 0;
        while (batch_start < segments.size()) {
            size_t batch_end = batch_start;
            int batch_words = 0;
            while (batch_end < segments.size()) {
                const int segment_words = std::max(1, count_words(segments[batch_end].text));
                if (batch_end > batch_start && batch_words + segment_words > 4000) {
                    break;
                }
                batch_words += segment_words;
                ++batch_end;
            }

            const std::string text_block = numbered_subtitle_block(segments, batch_start, batch_end);
            std::ostringstream body_json;
            body_json << "{"
                << "\"model\":\"" << json_escape(model) << "\","
                << "\"messages\":["
                << "{\"role\":\"system\",\"content\":\""
                << json_escape(
                    "You are a professional subtitle translator. Translate the numbered subtitle lines from "
                    + detected_language
                    + " to "
                    + target_language
                    + ".\nKeep translations concise and natural for subtitles.\n"
                        "Maintain the exact same numbering format [N] for each line.\n"
                        "Return ONLY the translated lines with their numbers, nothing else."
                )
                << "\"},"
                << "{\"role\":\"user\",\"content\":\"" << json_escape(text_block) << "\"}"
                << "]}";

            std::map<int, std::string> translations;
            try {
                translations = parse_numbered_translation_lines(
                    first_chat_response_content(backend_->chat(body_json.str()))
                );
            } catch (...) {
                translations.clear();
            }

            for (size_t i = batch_start; i < batch_end; ++i) {
                WhisperSegment segment = segments[i];
                const int batch_index = static_cast<int>(i - batch_start + 1);
                const auto found = translations.find(batch_index);
                if (found != translations.end() && !found->second.empty()) {
                    segment.text = found->second;
                }
                translated.push_back(segment);
            }
            batch_start = batch_end;
        }
        return translated.empty() ? segments : translated;
    };

    std::ostringstream response;
    response << "{\"source_language\":\"" << json_escape(detected_language) << "\",\"subtitles\":{";
    bool wrote = false;
    for (const auto &language : languages) {
        if (format != "json" && format != "srt" && format != "vtt") {
            continue;
        }
        if (wrote) {
            response << ",";
        }
        wrote = true;
        response << "\"" << json_escape(language) << "\":";
        const auto language_segments = translate_segments(language);
        if (format == "json") {
            response << whisper_segments_json(language_segments);
        } else if (format == "srt") {
            response << "\"" << json_escape(whisper_segments_srt(language_segments)) << "\"";
        } else {
            response << "\"" << json_escape(whisper_segments_vtt(language_segments)) << "\"";
        }
    }
    response << "}}";
    *out = {200, "application/json", response.str()};
    return true;
}

namespace {
// Python's context_retry.classify_inference_capacity_error, applied where the
// source returned an HTTP 200 "Local resources exhausted" completion. A 200
// reads as success to WorkerRouter, so it neither failed over nor told the
// client to compact. Context overflow keeps the source's 413 shape, capacity
// and engine failures become 503 (router-retryable), and llama.cpp client
// errors keep their 4xx status.
HttpResponse llm_failure_response(const std::string &message, const std::string &engine_body) {
    const auto number = [&](const std::regex &primary, const std::regex &secondary) -> int64_t {
        std::smatch match;
        if (std::regex_search(message, match, primary) || std::regex_search(message, match, secondary)) {
            try { return std::stoll(match[1].str()); } catch (const std::exception &) {}
        }
        return 0;
    };
    static const std::regex prompt_tokens(R"(n_prompt_tokens=(\d+))");
    static const std::regex prompt_request(R"(request\s*\((\d+)\s+tokens?\))", std::regex::icase);
    static const std::regex context_tokens(R"(n_ctx=(\d+))");
    static const std::regex context_available(R"(available context size\s*\((\d+)\s+tokens?\))", std::regex::icase);
    const int64_t prompt = number(prompt_tokens, prompt_request);
    const int64_t context = number(context_tokens, context_available);
    const auto quoted = '"' + json_escape(message) + '"';
    if (prompt > 0 && context > 0) {
        return {413, "application/json", "{\"detail\":{\"type\":\"context_capacity_error\",\"message\":" + quoted
            + ",\"n_prompt_tokens\":" + std::to_string(prompt) + ",\"n_ctx\":" + std::to_string(context)
            + ",\"compact_and_retry\":true}}"};
    }
    const auto lowered = to_lower(message);
    for (const auto *marker : {"out of memory", "cuda out of memory", "failed to allocate", "alloc_buffer", "cudamalloc", "insufficient memory"}) {
        if (lowered.find(marker) != std::string::npos) {
            return {503, "application/json", "{\"detail\":{\"type\":\"inference_capacity_error\",\"message\":" + quoted
                + (context > 0 ? ",\"n_ctx\":" + std::to_string(context) : std::string()) + ",\"compact_and_retry\":true}}"};
        }
    }
    const auto error = json_object_field(engine_body, "error", "");
    const auto source = !error.empty() && error.front() == '{' ? error : engine_body;
    const int code = extract_json_int_value(source, "code", 0);
    if (code >= 400 && code < 500 && code != 408 && code != 429) {
        auto type = extract_json_string(source, "type", "");
        if (type.empty() || type == "ezlocalai_native_error") type = "invalid_request_error";
        return {code, "application/json", "{\"detail\":{\"type\":\"" + json_escape(type) + "\",\"message\":" + quoted + "}}"};
    }
    return {503, "application/json", "{\"detail\":{\"type\":\"inference_error\",\"message\":" + quoted + "}}"};
}
}

std::string NativeServer::post_inference_fallback(const std::string &body, bool chat_mode) const {
    const auto quote = [](const std::string &value) { return '"' + json_escape(value) + '"'; };
    const auto wrap = [&](const std::string &content, const std::string &model) {
        return "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":" + content
            + "},\"finish_reason\":\"stop\",\"index\":0}],\"model\":" + quote(model) + '}';
    };
    const auto convert = [&](const std::string &response) {
        if (chat_mode) return response;
        const auto choices = json_object_field(response, "choices", "");
        if (choices.empty()) return response;
        if (choices.front() != '[' || json_array_elements(choices).empty()) throw std::runtime_error("Invalid fallback choices");
        const auto first = json_array_elements(choices).front();
        if (first.empty() || first.front() != '{') throw std::runtime_error("Invalid fallback choice");
        const auto message = json_object_field(first, "message", "{}");
        if (message.empty() || message.front() != '{') throw std::runtime_error("Invalid fallback message");
        return "{\"choices\":[{\"text\":" + json_object_field(message, "content", "\"\"")
            + "}],\"model\":" + json_object_field(response, "model", "\"fallback\"") + '}';
    };
    const auto &client = fallback_client();
    if (client.url.empty()) return convert(wrap(quote("Unable to process request. Local resources exhausted and no fallback server configured."), "fallback"));
    const auto messages = chat_mode ? json_object_field(body, "messages", "[]")
        : "[{\"role\":\"user\",\"content\":" + json_object_field(body, "prompt", "\"\"") + "}]";
    // Pipes passes only messages after generation errors. Request model,
    // generation parameters and stream are intentionally absent here.
    const std::string override_model = ezl_getenv_default("FALLBACK_MODEL");
    const auto payload = [&](const std::string &model) { return "{\"messages\":" + messages + ",\"model\":" + quote(model) + '}'; };
    if (fallback_availability(client.url, client.api_key).first) {
        std::optional<std::string> forwarded;
        try {
            double timeout = 0;
            if (!fallback_float(ezl_getenv_fallback("REQUEST_TIMEOUT", "300"), &timeout)) throw std::runtime_error("Invalid timeout");
            const auto response = fallback_json_request("POST", client.url, "/v1/chat/completions", client.api_key, payload(ezl_getenv_default("DEFAULT_MODEL")), timeout);
            std::string value;
            if (response.status != 200 || !fallback_json_response(response, &value)) throw std::runtime_error("Fallback forwarding failed");
            if (!value.empty() && value.front() == '{' && !json_object_field(value, "choices", "").empty()) forwarded = value;
            else forwarded = wrap(quote(python_str_from_json_value(value)), "fallback");
        } catch (const std::exception &) { /* Source tries its SDK after a failed ezlocalai client call. */ }
        if (forwarded) return convert(*forwarded);
    }
    const auto model = override_model.empty() ? std::string(ezl_getenv_default("DEFAULT_MODEL")) : override_model;
    std::string response;
    try {
        HttpResponse remote;
        for (int attempt = 0; attempt < 3; ++attempt) {
            remote = fallback_http_request("POST", client.url, "/chat/completions", client.api_key,
                "application/json", payload(model), 600, true, true, true);
            const auto explicit_retry = response_header(remote.headers, "x-should-retry");
            const bool retry = remote.transport_error || (explicit_retry && (*explicit_retry == "true" || *explicit_retry == "false") ? *explicit_retry == "true"
                : remote.status == 408 || remote.status == 409 || remote.status == 429 || remote.status >= 500);
            if (remote.status >= 200 && remote.status < 300 && !remote.transport_error) break;
            if (!retry || attempt == 2) break;
            double delay = 0;
            const auto milliseconds = response_header(remote.headers, "retry-after-ms");
            const auto seconds = response_header(remote.headers, "retry-after");
            if (milliseconds && parse_strict_python_float_string(*milliseconds, &delay)) delay /= 1000;
            else if (!seconds || !parse_strict_python_float_string(*seconds, &delay)) delay = 0;
            if (!(delay > 0 && delay <= 60)) {
                thread_local std::mt19937 generator(std::random_device{}());
                delay = (0.5 * (1 << attempt)) * (1 - 0.25 * std::uniform_real_distribution<double>(0, 1)(generator));
            }
            std::this_thread::sleep_for(std::chrono::duration<double>(delay));
        }
        if (remote.transport_error) throw std::runtime_error(remote.body.find("timed out") != std::string::npos ? "Request timed out." : "Connection error.");
        if (remote.status < 200 || remote.status >= 300) {
            const auto text = trim(remote.body);
            const auto error = json_decode_error(text);
            throw std::runtime_error(error ? (text.empty() ? "Error code: " + std::to_string(remote.status) : text)
                : "Error code: " + std::to_string(remote.status) + " - " + python_str_from_json_value(text));
        }
        if (auto error = json_decode_error(remote.body)) {
            size_t line = 1, column = 1, characters = 0;
            for (const auto byte : remote.body) {
                if (characters >= error->position) break;
                if ((static_cast<unsigned char>(byte) & 0xc0) == 0x80) continue;
                ++characters;
                if (byte == '\n') { ++line; column = 1; } else ++column;
            }
            throw std::runtime_error(error->message + ": line " + std::to_string(line) + " column " + std::to_string(column)
                + " (char " + std::to_string(error->position) + ')');
        }
        const auto value = trim(remote.body);
        const auto type = [](const std::string &raw) {
            if (raw == "null") return std::string("NoneType");
            if (raw == "true" || raw == "false") return std::string("bool");
            if (!raw.empty() && raw.front() == '"') return std::string("str");
            if (!raw.empty() && raw.front() == '[') return std::string("list");
            if (!raw.empty() && raw.front() == '{') return std::string("dict");
            return std::string(raw.find_first_of(".eE") == std::string::npos ? "int" : "float");
        };
        const auto attribute = [&](const std::string &raw, const std::string &name) {
            if (raw.empty() || raw.front() != '{') throw std::runtime_error("'" + type(raw) + "' object has no attribute '" + name + "'");
            return json_object_field(raw, name, "null");
        };
        const auto choices = attribute(value, "choices");
        std::string first;
        if (choices.front() == '[') {
            const auto values = json_array_elements(choices);
            if (values.empty()) throw std::runtime_error("list index out of range");
            first = values.front();
        } else if (choices.front() == '{') throw std::runtime_error("0");
        else if (choices.front() == '"') {
            if (parse_json_string_at(choices, 0).empty()) throw std::runtime_error("string index out of range");
            first = "\"x\"";
        } else throw std::runtime_error("'" + type(choices) + "' object is not subscriptable");
        const auto message = attribute(first, "message");
        const auto content = attribute(message, "content");
        response = wrap(content, model);
    } catch (const std::exception &error) {
        response = wrap(quote("Unable to process request. Fallback server error: " + std::string(error.what())), "fallback");
    }
    return convert(response);
}

const NativeServer::FallbackClientSettings &NativeServer::fallback_client() const {
    // Each NativeServer is an application instance. Like get_fallback_client,
    // capture settings once on first use, including an unconfigured client.
    std::call_once(fallback_client_once_, [this] {
        fallback_client_settings_.url = strip_trailing_slashes(ezl_getenv_default("FALLBACK_SERVER"));
        fallback_client_settings_.api_key = ezl_getenv_default("FALLBACK_API_KEY");
        if (fallback_client_settings_.api_key.empty()) fallback_client_settings_.api_key = ezl_getenv_default("EZLOCALAI_API_KEY");
    });
    return fallback_client_settings_;
}

const NativeServer::FallbackClientSettings &NativeServer::router_fallback() const {
    // With no FALLBACK_SERVER, a worker registered with a router sends text it
    // cannot serve locally to that router. Keys follow get_router_dispatch_client.
    std::call_once(router_fallback_once_, [this] {
        if (!fallback_client().url.empty() || native_router_mode_enabled()) return;
        router_fallback_settings_.url = strip_trailing_slashes(trim(ezl_getenv_default("ROUTER_URL")));
        for (const auto *name : {"ROUTER_CLIENT_API_KEY", "EZLOCALAI_API_KEY", "ROUTER_API_KEY"}) {
            router_fallback_settings_.api_key = trim(ezl_getenv_default(name));
            if (!router_fallback_settings_.api_key.empty()) break;
        }
    });
    return router_fallback_settings_;
}

const std::string &NativeServer::effective_llm_fallback_url() const {
    return fallback_client().url.empty() ? router_fallback().url : fallback_client().url;
}

bool NativeServer::router_fallback_available() const {
    const auto &router = router_fallback();
    if (router.url.empty()) return false;
    const double now = resources_->current_time();
    {
        std::lock_guard<std::mutex> lock(router_fallback_mutex_);
        if (router_fallback_checked_ >= 0 && now - router_fallback_checked_ < 30.0) return router_fallback_alive_;
    }
    bool alive = false;
    try {
        const auto response = fallback_http_request("GET", router.url, "/v1/router/health", router.api_key, "application/json", "", 5);
        alive = !response.transport_error && response.status == 200 && extract_json_int_value(response.body, "alive_workers", 0) > 0;
    } catch (const std::exception &) {}
    std::lock_guard<std::mutex> lock(router_fallback_mutex_);
    router_fallback_checked_ = now; router_fallback_alive_ = alive;
    return alive;
}

bool NativeServer::forward_llm_to_router(const std::string &path, const std::string &body, HttpResponse *out,
    std::shared_ptr<LlmInferenceStream> *stream) const {
    const auto &router = router_fallback();
    if (!out || router.url.empty() || !router_fallback_available()) return false;
    // The receiving worker must not forward again: one router hop at most.
    auto fields = json_object_ordered_entries(body);
    ordered_json_upsert(&fields, {"disable_fallback", "true"});
    const auto payload = json_object_from_ordered_entries(fields);
    double timeout = 0;
    if (!fallback_float(ezl_getenv_fallback("REQUEST_TIMEOUT", "300"), &timeout)) timeout = 300;
    const bool streaming = !is_python_falsey_json_value(json_object_field(body, "stream", "false"));
    std::cerr << "[Fallback] Forwarding " << path << " to router " << router.url << "\n";
    if (streaming && stream && path == "/v1/chat/completions") {
        const auto url = router.url, key = router.api_key;
        auto relay = std::make_shared<LlmInferenceStream>(extract_json_string(body, "model", ""), "router", "", false,
            [url, key, path, payload, timeout](const LlmInferenceStream::Emit &emit, const LlmInferenceStream::Interrupt &interrupt) {
                int status = 0; std::string pending, error_body; bool stopped = false;
                const auto deliver = [&](const std::string &bytes) {
                    if (status != 200) { error_body += bytes; return true; }
                    pending += bytes;
                    for (size_t end; (end = pending.find("\n\n")) != std::string::npos;) {
                        const auto event = pending.substr(0, end); pending.erase(0, end + 2);
                        std::istringstream lines(event);
                        for (std::string line; std::getline(lines, line);) {
                            if (!line.empty() && line.back() == '\r') line.pop_back();
                            if (line.rfind("data:", 0) != 0) continue;
                            const auto data = trim(line.substr(5));
                            if (data == "[DONE]" || data.empty() || data.front() != '{') continue;
                            if (emit(data)) { stopped = true; return false; }
                        }
                    }
                    return true;
                };
                const auto response = fallback_http_request("POST", url, path, key, "application/json", payload, timeout,
                    true, true, false, "", deliver, interrupt, "", [&](int value) { status = value; });
                if (stopped) return;
                if (response.transport_error) throw std::runtime_error("Router fallback failed: " + response.body);
                if (status != 200) throw std::runtime_error("Router fallback returned HTTP " + std::to_string(status)
                    + (error_body.empty() ? std::string() : ": " + error_body));
            }, [] {});
        relay->set_passthrough(true);
        *stream = std::move(relay);
        return true;
    }
    const auto response = fallback_http_request("POST", router.url, path, router.api_key, "application/json", payload, timeout);
    if (response.transport_error) return false;
    const auto text = trim(response.body);
    const auto type = starts_with(text, "data:") ? "text/event-stream; charset=utf-8"
        : !text.empty() && (text.front() == '{' || text.front() == '[') ? "application/json" : "text/plain; charset=utf-8";
    *out = {response.status, type, response.body};
    if (starts_with(text, "data:")) out->chunked = true;
    return true;
}

std::pair<bool, std::string> NativeServer::fallback_availability(const std::string &fallback_server, const std::string &api_key) const {
    if (fallback_server.empty()) return {false, "No fallback server configured"};
    const double now = resources_->current_time();
    {
        std::lock_guard<std::mutex> lock(fallback_mutex_);
        if (fallback_cache_valid_
            && now - fallback_cache_last_check_ < 30.0) {
            return {fallback_cache_available_, "cached"};
        }
    }

    bool available = false;
    std::string reason;
    try {
        const auto resources_response = fallback_json_request("GET", fallback_server, "/v1/resources", api_key, "", 5, false);
        if (resources_response.status != 200) {
            if (resources_response.transport_error) {
                reason = "Fallback server unreachable";
            } else {
                reason = "Fallback server returned status " + std::to_string(resources_response.status);
            }
        } else {
            double free_vram = 0.0;
            double min_vram = 2.0;
            const std::string raw_min_vram = ezl_getenv_fallback("FALLBACK_VRAM_THRESHOLD", "2.0");
            std::string value;
            if (!fallback_json_response(resources_response, &value) || !fallback_free_vram_value(value, &free_vram)
                || !fallback_float(raw_min_vram, &min_vram)) {
                reason = "Fallback server unreachable";
            } else if (free_vram < min_vram) {
                reason = "Fallback server low on VRAM (" + one_decimal(free_vram) + "GB < " + python_float_string(min_vram) + "GB)";
            } else {
                available = true;
                reason = "Fallback server available (" + one_decimal(free_vram) + "GB VRAM free)";
            }
        }
    } catch (const std::exception &) { available = false; reason = "Fallback server unreachable"; }

    {
        std::lock_guard<std::mutex> lock(fallback_mutex_);
        fallback_cache_valid_ = true;
        fallback_cache_last_check_ = now;
        fallback_cache_available_ = available;
        fallback_cache_reason_ = reason;
    }
    return {available, reason};
}

bool NativeServer::fallback_server_available_cached(const std::string &url, const std::string &key) const {
    return fallback_availability(url, key).first;
}

HttpResponse NativeServer::fallback_status() const {
    const auto &client = fallback_client();
    const auto &url = client.url;
    if (url.empty()) return {200, "application/json", "{\"configured\":false,\"available\":false,\"reason\":\"No fallback server configured\",\"should_use\":false}"};
    const auto &key = client.api_key;
    const auto availability = fallback_availability(url, key);
    // The remote availability result is cached even if local probing or
    // threshold parsing fails. Models are fetched only after local policy.
    const auto decision = native_fallback_decision(resources_.get(), fallback_client().url);
    std::vector<std::string> models;
    if (availability.first) {
        try {
            const auto response = fallback_json_request("GET", url, "/v1/models", key, "", 10, false);
            std::string value;
            if (response.status == 200 && fallback_json_response(response, &value) && !value.empty() && value.front() == '{') {
                const auto data = json_object_field(value, "data", "[]");
                if (!data.empty() && data.front() == '[') {
                    for (const auto &item : json_array_elements(data)) {
                        if (item.empty() || item.front() != '{') { models.clear(); break; }
                        models.push_back(json_object_field(item, "id", json_object_field(item, "name", "\"unknown\"")));
                    }
                }
            }
        } catch (const std::exception &) { models.clear(); }
    }
    // Serialization happens outside get_models' exception handler in FastAPI.
    // Non-finite ID values fail the response, while malformed model lists
    // simply produce an empty list.
    for (auto &model : models) {
        model = fallback_model_json(model); std::string decoded;
        if (!fallback_decode(model, "utf-8", &decoded)) throw std::invalid_argument("Invalid model UTF-8");
    }
    std::ostringstream out;
    out << "{\"configured\":true,\"url\":\"" << json_escape(url)
        << "\",\"available\":" << (availability.first ? "true" : "false")
        << ",\"availability_reason\":\"" << json_escape(availability.second)
        << "\",\"should_use\":" << (decision.should_use ? "true" : "false")
        << ",\"should_use_reason\":\"" << json_escape(decision.reason)
        << "\",\"remote_models\":" << json_array_from_values(models) << '}';
    return {200, "application/json", out.str()};
}

bool NativeServer::forward_text_server_request(const std::string &path, const std::string &body, HttpResponse *out) const {
    if (!out) {
        return false;
    }
    const std::string text_server = text_client().url;
    if (text_server.empty()) {
        return false;
    }
    const std::string api_key = text_client().api_key;
    if (!text_server_available_cached(text_server, api_key)) {
        return false;
    }

    const auto response = fallback_json_request("POST", text_server, path, api_key, body, 300);
    return service_json_response(response, out);
}

bool NativeServer::text_server_available_cached(const std::string &text_server, const std::string &api_key) const {
    const double now = monotonic_epoch_seconds();
    {
        std::lock_guard<std::mutex> lock(text_mutex_);
        if (text_cache_valid_
            && text_cache_url_ == text_server
            && now - text_cache_last_check_ < 30.0) {
            return text_cache_available_;
        }
    }
    const auto available = fallback_json_request("GET", text_server, "/v1/models", api_key, "", 5, false, false);
    const bool ok = available.status == 200;
    {
        std::lock_guard<std::mutex> lock(text_mutex_);
        text_cache_valid_ = true;
        text_cache_url_ = text_server;
        text_cache_last_check_ = now;
        text_cache_available_ = ok;
    }
    return ok;
}

bool NativeServer::forward_fallback_llm(const std::string &path, const std::string &body, HttpResponse *out) const {
    if (!out) {
        return false;
    }
    const std::string fallback_server = fallback_client().url;
    if (fallback_server.empty()) {
        return false;
    }
    const std::string fallback_api_key = fallback_client().api_key;
    if (!fallback_server_available_cached(fallback_server, fallback_api_key)) {
        return false;
    }

    double timeout = 0;
    if (!fallback_float(ezl_getenv_fallback("REQUEST_TIMEOUT", "300"), &timeout)) return false;
    const bool streaming = !is_python_falsey_json_value(json_object_field(body, "stream", "false"));
    auto response = fallback_json_request(
        "POST",
        fallback_server,
        path,
        fallback_api_key,
        body,
        timeout,
        true,
        !streaming
    );
    if (response.status != 200) {
        return false;
    }
    if (streaming) {
        // The pinned source returns an async generator after leaving the
        // response/session contexts. app.py iterates it synchronously for chat
        // and subscripts it for completion. Preserve these observable failures.
        if (path == "/v1/chat/completions" && extract_json_bool(body,"stream",false)) {
            *out = {200,"text/event-stream; charset=utf-8","data: {\"error\": {\"message\": \"'async_generator' object is not iterable\", \"type\": \"streaming_error\"}}\n\n"};
            out->chunked = true;
        } else *out = {500,"text/plain; charset=utf-8","Internal Server Error"};
        return true;
    }
    std::string value;
    if (!fallback_json_response(response,&value)) return false;
    try {
        value = fallback_model_json(value); std::string decoded;
        if (!fallback_decode(value,"utf-8",&decoded)) throw std::invalid_argument("Invalid response UTF-8");
        *out = {200,"application/json",value};
    } catch (const std::exception &) { *out = {500,"text/plain; charset=utf-8","Internal Server Error"}; }
    return true;
}

bool NativeServer::forward_embedding_server_request(const std::string &body, HttpResponse *out, bool through_router) const {
    std::string url = through_router ? ezl_getenv_default("ROUTER_URL") : embedding_client().url;
    if (through_router) url = strip_config_value(url);
    url = strip_trailing_slashes(url);
    if (url.empty()) return false;
    std::string key = through_router ? ezl_getenv_default("ROUTER_CLIENT_API_KEY") : embedding_client().api_key;
    if (through_router) key = strip_config_value(key);
    if (through_router && key.empty()) key = strip_config_value(config_.api_key);
    if (through_router && key.empty()) key = strip_config_value(ezl_getenv_default("ROUTER_API_KEY"));
    const std::string cache_key = std::string(through_router ? "router:" : "embedding:") + url;
    const double now = monotonic_epoch_seconds();
    bool available = false, cached = false;
    {
        std::lock_guard<std::mutex> lock(embedding_mutex_);
        const auto it = embedding_availability_.find(cache_key);
        if (it != embedding_availability_.end() && now - it->second.first < 30.0) {
            cached = true;
            available = it->second.second;
        }
    }
    if (!cached) {
        const auto probe = through_router ? http_json_request("GET", url, "/v1/router/health", key, "", 5)
            : fallback_json_request("GET", url, "/v1/models", key, "", 5, false, false);
        available = probe.status == 200 && (!through_router || extract_json_int_value(probe.body, "alive_workers", 0) > 0);
        std::lock_guard<std::mutex> lock(embedding_mutex_);
        embedding_availability_[cache_key] = {now, available};
    }
    if (!available) return false;
    if (!through_router) return service_json_response(fallback_json_request("POST", url, "/v1/embeddings", key, embedding_forward_payload(body), 120), out);
    auto response = http_json_request("POST", url, "/v1/embeddings", key, embedding_forward_payload(body), 60);
    if (response.status != 200 || !is_json_value_token(trim(response.body)) || trim(response.body) == "null") return false;
    *out = std::move(response);
    return true;
}

HttpResponse NativeServer::local_embeddings(const std::string &body) {
    auto bad_request = [](const std::string &detail) -> HttpResponse {
        return {400, "application/json", "{\"detail\":\"" + json_escape(detail) + "\"}"};
    };
    int dimensions = 0;
    bool has_dimensions = true;
    const auto raw_dimensions = trim(json_object_field(body, "dimensions", "null"));
    if (raw_dimensions != "null") {
        const auto parsed = parse_pydantic_int_token(raw_dimensions);
        if (!parsed.ok) return bad_request("dimensions must be an integer");
        dimensions = parsed.value;
    } else {
        const std::string configured = ezl_getenv_default("EMBEDDING_DIMENSIONS");
        if (!configured.empty() && !parse_strict_int_string(configured, &dimensions)) return bad_request("dimensions must be an integer");
        has_dimensions = !configured.empty();
    }
    if (has_dimensions && (dimensions < 32 || dimensions > 1024)) return bad_request("Qwen3-Embedding-0.6B dimensions must be 32..1024");

    const auto raw_model = trim(json_object_field(body, "model", ""));
    auto model = raw_model.empty() ? std::string(ezl_getenv_default("EMBEDDING_MODEL")) : raw_model == "null" ? std::string() : parse_json_string_at(raw_model, 0);
    if (model.empty()) {
        model = ezl_getenv_default("EMBEDDING_MODEL_ALIAS");
        if (model.empty()) model = ezl_getenv_default("EMBEDDING_MODEL");
    }
    const auto payload = json_object_from_ordered_entries({
        {"input", json_object_field(body, "input", "\"\"")},
        {"model", "\"" + json_escape(model) + "\""},
    });
    ScopedCapabilityUse embedding_use(this, "embedding");
    try {
        const auto result = backend_->embedding(payload);
        const auto error = trim(json_object_field(result, "error", ""));
        if (!error.empty()) {
            if (!error.empty() && error.front() == '{') {
                auto message = extract_json_string(error, "message", "Embedding request failed");
                if (message.empty()) message = "Embedding request failed";
                if (extract_json_string(error, "type", "") == "exceed_context_size_error") {
                    const int context = parse_positive_int_env("EMBEDDING_CONTEXT_LENGTH", 10000);
                    return bad_request("Embedding input exceeds the context limit: " + message + ". Configured EMBEDDING_CONTEXT_LENGTH=" + std::to_string(context) + ". Split the input into smaller chunks; input was not truncated.");
                }
                const int code = extract_json_int_value(error, "code", 0);
                if (code == 400 || code == 422) return bad_request(message);
            }
            return {500, "text/plain; charset=utf-8", "Internal Server Error"};
        }
        if (!is_valid_top_level_json_object(result)) return {500, "text/plain; charset=utf-8", "Internal Server Error"};
        auto fields = json_object_ordered_entries(result);
        if (dimensions > 0) {
            auto entries = json_array_elements(json_object_field(result, "data", "[]"));
            for (auto &entry : entries) {
                auto values = json_array_elements(json_object_field(entry, "embedding", "[]"));
                if (values.size() <= static_cast<size_t>(dimensions)) continue;
                values.resize(static_cast<size_t>(dimensions));
                std::vector<double> numbers;
                double sum = 0;
                for (const auto &value : values) {
                    double number = 0;
                    if (!parse_strict_python_float_string(value, &number)) throw std::runtime_error("Invalid embedding vector");
                    numbers.push_back(number);
                    sum += number * number;
                }
                const double norm = std::sqrt(sum);
                if (norm > 0) for (size_t i = 0; i < values.size(); ++i) values[i] = json_number_literal(numbers[i] / norm);
                auto item = json_object_ordered_entries(entry);
                ordered_json_upsert(&item, {"embedding", "[" + join_strings(values, ",") + "]"});
                entry = json_object_from_ordered_entries(item);
            }
            ordered_json_upsert(&fields, {"data", "[" + join_strings(entries, ",") + "]"});
        }
        ordered_json_upsert(&fields, {"model", "\"" + json_escape(model) + "\""});
        return {200, "application/json", json_object_from_ordered_entries(fields)};
    } catch (const std::exception &) {
        return {500, "text/plain; charset=utf-8", "Internal Server Error"};
    }
}

bool NativeServer::forward_fallback_embeddings(const std::string &body, HttpResponse *out) const {
    if (!out) {
        return false;
    }
    const auto fallback_decision = native_fallback_decision(resources_.get(), fallback_client().url);
    if (!fallback_decision.should_use || fallback_decision.url.empty()) {
        return false;
    }
    const std::string fallback_api_key = fallback_client().api_key;
    if (!fallback_server_available_cached(fallback_decision.url, fallback_api_key)) {
        return false;
    }

    auto response = fallback_json_request(
        "POST",
        fallback_decision.url,
        "/v1/embeddings",
        fallback_api_key,
        embedding_forward_payload(body, true),
        60
    );
    if (response.status != 200) {
        return false;
    }
    *out = response;
    return true;
}

HttpResponse NativeServer::music_generation(const std::string &body, bool video) {
    HttpResponse failure;
    std::string payload;
    if (!normalized_music_payload(body, video, &payload, &failure)) return failure;
    auto bad_request = [](const std::string &detail) -> HttpResponse { return {400, "application/json", "{\"detail\":\"" + json_escape(detail) + "\"}"}; };
    for (const auto *field : {"prompt", "lyrics", "keyscale"}) {
        if (strip_config_value(parse_json_string_at(json_object_field(payload, field, "\"\""), 0)).empty()) return bad_request(std::string(field) + " is required");
    }
    if (coerce_python_float_value(json_object_field(payload, "duration", "0"), 0, true) <= 0) return bad_request("duration must be greater than 0");
    const auto scene_duration = json_object_field(payload, "scene_duration", "");
    if (video && !scene_duration.empty() && coerce_python_float_value(scene_duration, 0, true) <= 0) return bad_request("scene_duration must be greater than 0");
    if (!has_local_capability(video ? "music_video" : "music")) {
        HttpResponse forwarded;
        if (forward_music_request(payload, video, true, &forwarded) || forward_music_request(payload, video, false, &forwarded)) return forwarded;
        const std::string model = video
            ? "{\"music\":" + json_object_field(payload, "music_model", "null") + ",\"video\":" + json_object_field(payload, "video_model", "null") + "}"
            : json_object_field(payload, "model", "null");
        const std::string detail = video
            ? "Music video generation not available. Set MUSIC_ENABLED=true and VIDEO_ENABLED=true on the same worker."
            : "Music generation not available. Set MUSIC_ENABLED=true to use the internal ACE-Step server, or set ACE_STEP_SERVER_URL to an external acestep.cpp server.";
        return {200, "application/json", "{\"created\":" + std::to_string(unix_seconds()) + ",\"model\":" + model + ",\"data\":[{\"error\":\"" + detail + "\"}]}"};
    }
    if (video) return local_music_video_generation(payload);
    try {
        return generate_ace_music(payload);
    } catch (const std::invalid_argument &error) {
        return bad_request(error.what());
    } catch (const std::exception &error) {
        return {502, "application/json", "{\"detail\":\"Music generation failed: " + json_escape(error.what()) + "\"}"};
    }
}

bool NativeServer::forward_music_request(const std::string &payload, bool video, bool through_router, HttpResponse *out) const {
    const auto url = through_router ? strip_trailing_slashes(strip_config_value(ezl_getenv_default("ROUTER_URL"))) : fallback_client().url;
    if (url.empty()) return false;
    std::string key = through_router ? strip_config_value(ezl_getenv_default("ROUTER_CLIENT_API_KEY")) : fallback_client().api_key;
    if (through_router && key.empty()) key = strip_config_value(config_.api_key);
    if (through_router && key.empty()) key = strip_config_value(ezl_getenv_default("ROUTER_API_KEY"));
    if (through_router) {
        const auto cache_key = "router:" + url;
        const double now = monotonic_epoch_seconds();
        bool available = false, cached = false;
        {
            std::lock_guard<std::mutex> lock(embedding_mutex_);
            const auto it = embedding_availability_.find(cache_key);
            if (it != embedding_availability_.end() && now - it->second.first < 30.0) { available = it->second.second; cached = true; }
        }
        if (!cached) {
            const auto probe = http_json_request("GET", url, "/v1/router/health", key, "", 5);
            available = probe.status == 200 && coerce_python_int_value(json_object_field(probe.body, "alive_workers", "0"), 0, true) > 0;
            std::lock_guard<std::mutex> lock(embedding_mutex_);
            embedding_availability_[cache_key] = {now, available};
        }
        if (!available) return false;
    } else if (!fallback_server_available_cached(url, key)) return false;
    double request_timeout = 0, ace_timeout = 0;
    if (!fallback_float(ezl_getenv_fallback("REQUEST_TIMEOUT", "1800"), &request_timeout)
        || !fallback_float(ezl_getenv_fallback("ACE_STEP_TIMEOUT", "1800"), &ace_timeout)) return false;
    const double timeout = std::max(request_timeout, ace_timeout);
    auto response = fallback_json_request("POST", url, video ? "/v1/videos/music" : "/v1/audio/music", key, payload, timeout);
    std::string value;
    if (response.status != 200 || !fallback_json_response(response,&value)) return false;
    response.content_type = "application/json";
    response.body = std::move(value);
    *out = std::move(response);
    return true;
}

HttpResponse NativeServer::video_generation(const std::string &body) {
    HttpResponse response;
    if (forward_image_server_request("/v1/videos/generations", body, &response)) return response;
    const bool local = has_local_capability("video");
    if (!local && forward_router_media_request(body, &response, "/v1/videos/generations")) return response;
    if (forward_fallback_image_video_request("/v1/videos/generations", body, &response)) return response;
    if (!local) return {200, "application/json", "{\"created\":" + std::to_string(unix_seconds())
        + ",\"data\":[{\"error\":\"Video generation not available. Set VIDEO_ENABLED=true and VIDEO_MODEL to enable.\"}]}"};
    // Pipes skips local backend construction when an image server is configured.
    if (!image_client().url.empty())
        return {500, "application/json", "{\"detail\":\"Video generation produced no output\"}"};
    return local_video_generation(body);
}

bool NativeServer::forward_router_media_request(const std::string &body, HttpResponse *out, const std::string &path) const {
    auto url = strip_trailing_slashes(strip_config_value(ezl_getenv_default("ROUTER_URL")));
    if (url.empty()) return false;
    auto key = strip_config_value(ezl_getenv_default("ROUTER_CLIENT_API_KEY"));
    if (key.empty()) key = strip_config_value(config_.api_key);
    if (key.empty()) key = strip_config_value(ezl_getenv_default("ROUTER_API_KEY"));
    const auto cache_key = "router:" + url;
    const auto now = monotonic_epoch_seconds();
    bool available = false, cached = false;
    {
        std::lock_guard<std::mutex> lock(embedding_mutex_);
        const auto it = embedding_availability_.find(cache_key);
        if (it != embedding_availability_.end() && now - it->second.first < 30.0) { available = it->second.second; cached = true; }
    }
    if (!cached) {
        const auto probe = http_json_request("GET", url, "/v1/router/health", key, "", 5);
        std::string value;
        available = probe.status == 200 && json_response_value(probe, &value) && is_valid_top_level_json_object(value)
            && positive_python_integer(json_object_field(value, "alive_workers", "0"));
        std::lock_guard<std::mutex> lock(embedding_mutex_);
        embedding_availability_[cache_key] = {now, available};
    }
    if (!available) return false;
    const bool video = path == "/v1/videos/generations";
    const auto response = http_json_request("POST", url, path, key, video ? video_forward_payload(body, true, true) : image_forward_payload(body, path == "/v1/images/edits"), video ? 600 : 660);
    std::string value;
    if (response.status != 200 || !json_response_value(response, &value) || value == "null") return false;
    *out = {200, "application/json", value};
    return true;
}

void NativeServer::reclaim_auxiliary_memory_for_llm(double required_gb) {
    reclaim_llm_auxiliary_memory(*resources_, required_gb, [this](ModelType type) {
        // A selected idle model may have acquired a lease since the LRU
        // snapshot. Pool destruction rechecks leases; media destruction also
        // requires its generation lock so a handoff cannot deadlock a loader.
        if (resources_->is_model_in_use(type)) return;
        if (type == ModelType::Tts) { if (tts_service_) tts_service_->unload_idle(); }
        else if (type == ModelType::Stt) { if (stt_service_) stt_service_->unload_idle(); }
        else if (type == ModelType::Embedding) backend_->unload_aux_model("embedding");
        else if (type == ModelType::Image) {
            std::unique_lock<std::mutex> lock(image_generation_mutex_, std::try_to_lock);
            if (lock && !resources_->is_model_in_use(type) && image_) {
                image_.reset(); resources_->unregister_model(type);
            }
        } else if (type == ModelType::Video) {
            std::unique_lock<std::mutex> lock(video_mutex_, std::try_to_lock);
            if (lock && !resources_->is_model_in_use(type) && (video_ || ltx_video_)) {
                video_.reset(); ltx_video_.reset(); resources_->unregister_model(type);
                std::lock_guard<std::mutex> state(resource_mutex_);
                resource_video_loaded_ = false; resource_video_device_.clear();
            }
        }
    });
}

void NativeServer::unload_aux_models_for_video(bool unload_llm, VideoAuxiliaryState &state) {
    if (!video_should_unload_auxiliary(unload_llm)) return;
    const auto loaded = backend_->loaded_models_json();
    constexpr std::array<ModelType, 4> types{ModelType::Tts, ModelType::Stt, ModelType::Embedding, ModelType::Image};
    std::array<AuxiliaryResidency, 4> models{};
    {
        std::lock_guard<std::mutex> lock(resource_mutex_);
        for (size_t i = 0; i < models.size(); ++i) {
            const auto *name = video_auxiliary_models[i];
            const auto entry = json_object_field(loaded, name, "null");
            models[i].loaded = entry != "null" || resources_->is_model_loaded(types[i]);
            const std::string capability = i == 3 ? "image" : name;
            const auto active = resource_capability_in_flight_.find(capability);
            models[i].active_requests = std::max(extract_json_int_value(entry, "active_requests", 0),
                active == resource_capability_in_flight_.end() ? 0 : active->second);
            if (resources_->is_model_in_use(types[i])) models[i].active_requests = std::max(1, models[i].active_requests);
        }
    }
    state.native_owner[0] = bool(tts_service_);
    state.native_owner[1] = bool(stt_service_);
    if (tts_service_) {
        const auto pool = tts_service_->snapshot();
        models[0].loaded = pool.primary || !pool.resident.empty() || !pool.transient.empty();
        if (!pool.leases.empty()) models[0].active_requests = std::max(1, models[0].active_requests);
    }
    if (stt_service_) {
        const auto pool = stt_service_->snapshot();
        models[1].loaded = pool.primary || !pool.resident.empty() || !pool.transient.empty();
        if (!pool.leases.empty()) models[1].active_requests = std::max(1, models[1].active_requests);
    }
    const auto selected = video_auxiliary_unload(models, unload_llm);
    for (size_t i = 0; i < selected.size(); ++i) if (selected[i]) {
        // Pool destruction rechecks leases under its own lock. Image requests
        // hold image_generation_mutex_ while waiting for LLM handoff, so video
        // must never wait for that lock while holding llm_handoff_mutex_.
        if (i == 0 && tts_service_) state.unloaded[i] = tts_service_->unload_idle();
        else if (i == 1 && stt_service_) state.unloaded[i] = stt_service_->unload_idle();
        else if (i == 3) {
            std::unique_lock<std::mutex> image_lock(image_generation_mutex_, std::try_to_lock);
            if (!image_lock || resources_->is_model_in_use(ModelType::Image)) continue;
            if (image_) {
                state.native_owner[i] = true;
                image_.reset(); resources_->unregister_model(ModelType::Image);
                state.unloaded[i] = true;
            } else state.unloaded[i] = backend_->unload_aux_model(video_auxiliary_models[i]);
        } else state.unloaded[i] = backend_->unload_aux_model(video_auxiliary_models[i]);
    }
    resources_->release_unused_cuda_memory();
}

void NativeServer::restore_aux_models_after_video(const VideoAuxiliaryState &state) {
    auto models = llm_catalog_.available;
    for (auto &model : models) model = llm_catalog_.source(model);
    const auto loaded = backend_->loaded_models_json();
    const auto restore = video_auxiliary_restore(state.unloaded, models,
        resources_->is_model_loaded(ModelType::Image) || json_object_field(loaded, "img", "null") != "null");
    for (size_t i = 0; i < restore.size(); ++i) if (restore[i]) {
        try {
            if (i == 0 && state.native_owner[i]) tts_service_->warm();
            else if (i == 1 && state.native_owner[i]) stt_service_->warm();
            else if (i == 3 && state.native_owner[i]) {
                std::unique_lock<std::mutex> image_lock(image_generation_mutex_, std::try_to_lock);
                // A concurrent image request owns its lazy load. Waiting here
                // could invert its image/LLM handoff lock order.
                if (image_lock && !image_) {
                    image_ = std::make_unique<QwenImage>(visible_cuda_device_count() ? "cuda" : "cpu");
                    resources_->register_model(ModelType::Image, ezl_getenv_default("IMG_MODEL"), "cuda", 4.0);
                }
            } else backend_->warm_aux_model(video_auxiliary_models[i]);
        }
        catch (const std::exception &error) {
            // Reference warm-load methods log failed instances and leave them
            // eligible for lazy loading, without failing a completed video.
            std::cerr << "[VIDEO] Failed to restore " << video_auxiliary_models[i] << ": " << error.what() << '\n';
        }
    }
}

HttpResponse NativeServer::local_image_generation(const std::string &body, bool edit) {
    auto failure_response = [](int status, const std::string &message) -> HttpResponse {
        return {status, "application/json", "{\"detail\":\"" + json_escape(message) + "\"}"};
    };
    try {
        if (json_object_field(body, "n", "1") == "null") return plain_internal_server_error();
        const int count = std::max(1, parse_pydantic_int_token(json_object_field(body, "n", "1")).value);
        const bool as_url = json_object_field(body, "response_format", "\"url\"") == "\"url\"";
        QwenImageSettings settings;
        settings.size = extract_json_string(body, "size", "1024x1024");
        if (json_object_field(body, "size", "") == "null") settings.size.clear();
        if (edit) {
            const auto primary = json_object_field(body, "image", "null");
            if (primary != "null") settings.image = parse_json_string_at(primary, 0);
            for (const auto &ref : json_array_elements(json_object_field(body, "images", "[]"))) settings.images.push_back(parse_json_string_at(ref, 0));
            settings.strength = coerce_python_float_value(json_object_field(body, "strength", "0.75"), 0.75, true);
        }
        std::vector<std::string> data;
        for (int i = 0; i < count; ++i) {
            std::lock_guard<std::mutex> lock(image_generation_mutex_);
            const auto secondary = secondary_cuda_device(visible_cuda_devices());
            const auto loaded = backend_->loaded_models_json();
            const bool resident = json_object_field(loaded, "llm", "null") != "null" || json_object_field(loaded, "vision_llm", "null") != "null";
            const bool unload = image_should_unload_llm(resident, bool(image_), secondary, resources_->get_total_free_vram(), [this] { return !image_client().url.empty(); });
            std::unique_lock<std::mutex> handoff(llm_handoff_mutex_, std::defer_lock);
            if (unload) handoff.lock();
            if (unload) wait_for_llm_idle_for_media("image", "IMAGE_WAIT_FOR_LLM_IDLE_TIMEOUT", true);
            std::function<void(bool)> restore;
            std::array<bool, 4> auxiliary{};
            std::exception_ptr failure;
            std::string output;
            auto sources = llm_catalog_.available; for (auto &source : sources) source = llm_catalog_.source(source);
            try {
                if (unload && resident) restore = backend_->unload_llms_for_media();
                if (image_should_unload_auxiliary(unload, sources)) {
                    if (tts_service_ && resources_->is_model_loaded(ModelType::Tts) && !resources_->is_model_in_use(ModelType::Tts)) { auxiliary[0] = tts_service_->unload_idle(); }
                    if (stt_service_ && resources_->is_model_loaded(ModelType::Stt) && !resources_->is_model_in_use(ModelType::Stt)) { auxiliary[1] = stt_service_->unload_idle(); }
                    if (resources_->is_model_loaded(ModelType::Embedding) && !resources_->is_model_in_use(ModelType::Embedding)) auxiliary[2] = backend_->unload_aux_model("embedding");
                    std::unique_lock<std::mutex> video_lock(video_mutex_, std::try_to_lock);
                    if (video_lock && (video_ || ltx_video_)) { video_.reset(); ltx_video_.reset(); resources_->unregister_model(ModelType::Video); auxiliary[3] = true; std::lock_guard<std::mutex> resource_lock(resource_mutex_); resource_video_loaded_ = false; }
                }
                // Pipes._get_img skips local loading while its client is configured.
                if (image_client().url.empty()) {
                    if (!image_) {
                        const auto admission = resources_->can_load_model(ModelType::Image, 6.0);
                        const auto device = video_device(secondary, false, admission.can_load, admission.device);
                        image_ = std::make_unique<QwenImage>(device);
                        resources_->register_model(ModelType::Image, ezl_getenv_default("IMG_MODEL"), device, device == "cuda" ? 4.0 : 0.0);
                    }
                    ScopedCapabilityUse use(this, "image");
                    // _get_img and generate_image each acquire one resource mark;
                    // the generation finally and _destroy_img each release one.
                    resources_->mark_model_in_use(ModelType::Image, true);
                    resources_->mark_model_in_use(ModelType::Image, true);
                    try { output = image_->generate(extract_json_string(body, "prompt", ""), settings, as_url ? ezl_getenv_default("EZLOCALAI_URL") : ""); }
                    catch (...) { resources_->mark_model_in_use(ModelType::Image, false); resources_->mark_model_in_use(ModelType::Image, false); throw; }
                    resources_->mark_model_in_use(ModelType::Image, false);
                    resources_->mark_model_in_use(ModelType::Image, false);
                }
            } catch (...) { failure = std::current_exception(); }
            if (restore) { image_.reset(); resources_->unregister_model(ModelType::Image); }
            try {
                if (restore) {
                    const auto reload = to_lower(strip_config_value(ezl_getenv_fallback("IMAGE_RELOAD_LLM_AFTER_GENERATION", "true")));
                    restore(reload != "0" && reload != "false" && reload != "no" && reload != "off");
                }
            } catch (...) { failure = std::current_exception(); }
            if (unload) { std::lock_guard<std::mutex> resource_lock(resource_mutex_); llm_temporarily_unavailable_ = false; llm_idle_cv_.notify_all(); }
            if (handoff.owns_lock()) handoff.unlock();
            const auto warm = video_auxiliary_restore({auxiliary[0], auxiliary[1], auxiliary[2], false}, sources, true);
            for (size_t j = 0; j < 3; ++j) if (warm[j]) try {
                if (j == 0 && tts_service_) tts_service_->warm();
                else if (j == 1 && stt_service_) stt_service_->warm();
                else if (j == 2) backend_->warm_aux_model("embedding");
            } catch (const std::exception &e) { std::cerr << "[IMG] Failed to restore auxiliary model: " << e.what() << '\n'; }
            if (auxiliary[3] && has_local_capability("video")) {
                std::lock_guard<std::mutex> video_lock(video_mutex_);
                if (!video_ && !ltx_video_) try {
                    const auto device = visible_cuda_device_count() ? "cuda" : "cpu";
                    const std::string model = ezl_getenv_default("VIDEO_MODEL");
                    load_video_backend(model, device);
                } catch (const std::exception &e) { std::cerr << "[IMG] Failed to restore video: " << e.what() << '\n'; }
            }
            if (failure) std::rethrow_exception(failure);
            if (edit && output.empty()) return failure_response(503, "Image editing produced no output");
            data.push_back(std::string("{\"") + (as_url ? "url" : "b64_json") + "\":\"" + json_escape(output) + "\"}");
        }
        return {200, "application/json", "{\"created\":" + std::to_string(unix_seconds()) + ",\"data\":[" + join_strings(data, ",") + "]}"};
    } catch (const std::system_error &) { return plain_internal_server_error(); }
    catch (const std::invalid_argument &e) { return edit ? failure_response(422, e.what()) : plain_internal_server_error(); }
    catch (const std::runtime_error &e) { return edit ? failure_response(503, e.what()) : plain_internal_server_error(); }
    catch (...) { return plain_internal_server_error(); }
}

static RgbImage decode_video_input_image(const std::string &image_input) {
    if (starts_with(image_input, "http://") || starts_with(image_input, "https://")) {
        // app.py rejects private/internal URL targets before image decoding.
        ParsedHttpUrl parsed;
        if (!parse_http_url(image_input, &parsed)) throw std::runtime_error("Invalid image URL");
        addrinfo hints{}, *addresses = nullptr; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(parsed.host.c_str(), nullptr, &hints, &addresses) != 0) throw std::runtime_error("Cannot resolve image URL");
        bool private_address = false;
        for (auto *address = addresses; address; address = address->ai_next) {
            if (address->ai_family == AF_INET) {
                const auto ip = ntohl(reinterpret_cast<sockaddr_in *>(address->ai_addr)->sin_addr.s_addr);
                private_address |= (ip >> 24) == 0 || (ip >> 24) == 10 || (ip >> 24) == 127 || (ip >> 20) == 0xac1
                    || (ip >> 16) == 0xa9fe || (ip >> 16) == 0xc0a8 || (ip >> 24) >= 240
                    || ((ip >> 8) == 0xc00000 && ip != 0xc0000009 && ip != 0xc000000a)
                    || (ip >> 8) == 0xc00002 || (ip >> 17) == 0x6309 || (ip >> 8) == 0xc63364 || (ip >> 8) == 0xcb0071;
            } else if (address->ai_family == AF_INET6) {
                const auto &ip = reinterpret_cast<sockaddr_in6 *>(address->ai_addr)->sin6_addr;
                private_address |= IN6_IS_ADDR_LOOPBACK(&ip) || IN6_IS_ADDR_UNSPECIFIED(&ip) || IN6_IS_ADDR_LINKLOCAL(&ip)
                    || IN6_IS_ADDR_V4MAPPED(&ip) || (ip.s6_addr[0] & 0xfe) == 0xfc;
            }
        }
        freeaddrinfo(addresses);
        if (private_address) throw std::runtime_error("Private image URL");
        const auto fetched = http_json_request("GET", image_input, "", "", "", 30);
        if (fetched.status < 200 || fetched.status >= 300) throw std::runtime_error("Image download failed");
        return decode_rgb_image(fetched.body);
    }
    return decode_base64_rgb_image(image_input);
}

namespace {
struct VideoSessionReservation {
    std::unique_lock<std::mutex> lock;
    std::atomic<bool> &active;
    VideoSessionReservation(std::mutex &mutex, std::atomic<bool> &state) : lock(mutex), active(state) { active.store(true); }
    ~VideoSessionReservation() { active.store(false); }
};
}

HttpResponse NativeServer::local_music_video_generation(const std::string &payload) {
#ifndef EZLOCALAI_STORYBOARD
    return {501, "application/json", make_error(501, "music-video requires the native storyboard dependencies")};
#else
    using Json = nlohmann::json;
    const auto local_uri = ezl_getenv_default("EZLOCALAI_URL");
    const auto model = strip_config_value(ezl_getenv_default("VIDEO_MODEL"));
    auto request = Json::parse(payload);
    auto binary = [](const std::string &bytes) { return Json::binary(std::vector<uint8_t>(bytes.begin(), bytes.end())); };
    auto bytes = [](const Json &image) -> std::string {
        if (image.is_null() || (image.is_string() && image.get_ref<const std::string &>().empty())) return {};
        const auto &data = image.get_binary(); return {data.begin(), data.end()};
    };
    // app.py decodes and filters images before entering the pipeline's error
    // mapping, even when conditions override those images during generation.
    try {
        if (request.contains("image") && !request["image"].is_null() && request["image"] != "")
            request["image"] = binary(encode_rgb_png(decode_video_input_image(request["image"].get<std::string>())));
        if (request.contains("scene_images") && request["scene_images"].is_array()) {
            Json decoded = Json::array();
            for (const auto &image : request["scene_images"])
                if (!image.is_null() && image != "") decoded.push_back(binary(encode_rgb_png(decode_video_input_image(image.get<std::string>()))));
            request["scene_images"] = std::move(decoded);
        }
        if (request.contains("conditions") && request["conditions"].is_array())
            for (auto &condition : request["conditions"])
                condition["image"] = binary(encode_rgb_png(decode_video_input_image(condition.at("image").get<std::string>())));
    } catch (const std::exception &) { return {500, "text/plain; charset=utf-8", "Internal Server Error"}; }
    try {
        MusicVideoOperations operations;
        operations.music = [&](const Json &music) { return Json::parse(generate_ace_music(music.dump()).body); };
        operations.resolve = [&](const std::string &reference) { return resolve_music_video_output(reference, local_uri); };
        operations.video_session = [&](const std::function<void()> &scenes) {
            VideoSessionReservation reservation(video_mutex_, video_session_active_);
            const auto secondary = secondary_cuda_device(visible_cuda_devices());
            const auto loaded = backend_->loaded_models_json();
            const bool resident_llm = json_object_field(loaded, "llm", "null") != "null"
                || json_object_field(loaded, "vision_llm", "null") != "null";
            const bool unload = video_should_unload_llm(resident_llm, secondary, [this] { return !image_client().url.empty(); });
            std::unique_lock<std::mutex> handoff(llm_handoff_mutex_, std::defer_lock);
            if (unload) handoff.lock();
            if (!run_video_session(model, secondary, resident_llm, unload, scenes))
                throw std::runtime_error("Video scene 1 failed to generate");
        };
        operations.scene = [&](const Json &scene) {
            if (is_ltx_model(model) && (!ltx_video_ || !ltx_video_->loaded())) {
                LtxVideoSettings hint; hint.size = scene.at("size"); hint.frames = scene.at("num_frames");
                load_video_backend(model, video_device(secondary_cuda_device(visible_cuda_devices())), hint);
            }
            ScopedCapabilityUse video_use(this, "video");
            const auto image = bytes(scene.at("image"));
            const bool has_conditions = !scene.at("conditions").is_null() && !scene.at("conditions").empty();
            // An explicit empty image survives app.py's truthiness check. It
            // selects I2V (is not None), where the reference rejects it.
            if (!has_conditions && !scene.at("image").is_null() && !scene.at("image").is_binary()) return std::string{};
            if (ltx_video_) {
                LtxVideoSettings settings;
                settings.size = scene.at("size"); settings.frames = scene.at("num_frames");
                settings.fps = scene.at("frame_rate"); settings.steps = scene.at("num_inference_steps");
                settings.guidance = scene.at("guidance_scale");
                std::vector<LtxCondition> conditions;
                if (has_conditions) for (const auto &condition : scene.at("conditions"))
                    conditions.push_back({bytes(condition.at("image")), condition.value("index", int64_t(0)), condition.value("strength", 1.0)});
                return ltx_video_->generate(scene.at("prompt"), settings, local_uri, image, conditions, scene.at("include_audio"));
            }
            if (video_) {
                WanVideoSettings settings;
                settings.size = scene.at("size"); settings.frames = scene.at("num_frames");
                settings.fps = scene.at("frame_rate"); settings.steps = scene.at("num_inference_steps");
                settings.guidance = scene.at("guidance_scale");
                return video_->generate(scene.at("prompt"), settings, local_uri, image, has_conditions);
            }
            return std::string{};
        };
        operations.storyboard = [&](const Json &scene) {
            return binary(encode_storyboard_png(music_video_storyboard(scene.at("prompt"), scene.at("lyrics"),
                scene.at("scene_index"), scene.at("scene_count"), scene.at("size"), scene.at("keyscale"))));
        };
        operations.overlay = [&](const std::string &size) { return binary(encode_storyboard_png(pythagorean_equation_overlay(size))); };
        operations.stitch = [&](const std::vector<std::string> &scenes, const std::string &audio, double duration,
            const std::string &format, const Json &overlay) {
            return stitch_music_video(scenes, audio, duration, format, local_uri, bytes(overlay));
        };
        operations.video_model = [&] { return model; };
        operations.now = [] { return unix_seconds(); };
        return {200, "application/json", generate_music_video(request, operations).dump()};
    } catch (const std::invalid_argument &error) {
        return {400, "application/json", "{\"detail\":\"" + json_escape(error.what()) + "\"}"};
    } catch (const std::exception &error) {
        return {502, "application/json", "{\"detail\":\"Music video generation failed: " + json_escape(error.what()) + "\"}"};
    }
#endif
}

HttpResponse NativeServer::local_video_generation(const std::string &body) {
    const auto model = strip_config_value(ezl_getenv_default("VIDEO_MODEL"));
    if (!is_wan_model(model) && !is_ltx_model(model)) return {501, "application/json", make_error(501, "this native backend does not implement the configured video model yet")};
    auto bad_request = [](const std::string &message) -> HttpResponse { return {400, "application/json", "{\"detail\":\"" + json_escape(message) + "\"}"}; };
    const auto no_output = HttpResponse{500, "application/json", "{\"detail\":\"Video generation produced no output\"}"};
    try {
        bool invalid_settings = false;
        auto integer = [&](const std::string &name, int fallback) {
            const auto raw = json_object_field(body, name, std::to_string(fallback));
            if (raw == "null") { invalid_settings = true; return static_cast<int64_t>(fallback); }
            return static_cast<int64_t>(coerce_python_int_value(raw, fallback, true));
        };
        if (json_object_field(body, "n", "1") == "null") return {500, "text/plain; charset=utf-8", "Internal Server Error"};
        const auto count = std::max<int64_t>(1, integer("n", 1));
        const auto format = json_object_field(body, "response_format", "\"url\"");
        const bool as_url = format == "\"url\"";
        WanVideoSettings settings;
        settings.size = python_str_from_json_value(json_object_field(body, "size", "\"768x512\""));
        settings.frames = integer("num_frames", 121);
        settings.fps = integer("frame_rate", 24);
        settings.steps = integer("num_inference_steps", 18);
        const auto guidance = json_object_field(body, "guidance_scale", "3.5");
        if (guidance == "null") invalid_settings = true;
        settings.guidance = coerce_python_float_value(guidance, 3.5, true);
        // app.py decodes all inputs before entering the pipeline, including
        // conditions that the selected Wan backend subsequently rejects.
        std::string image_bytes;
        const auto image_input = extract_json_string(body, "image", "");
        if (!image_input.empty()) image_bytes = encode_rgb_png(decode_video_input_image(image_input));
        const bool conditions = !is_python_falsey_json_value(json_object_field(body, "conditions", "null"));
        std::vector<LtxCondition> ltx_conditions;
        if (conditions) for (const auto &condition : json_array_elements(json_object_field(body, "conditions", "[]"))) {
            const auto decoded = decode_video_input_image(extract_json_string(condition, "image", ""));
            if (is_ltx_model(model)) ltx_conditions.push_back({encode_rgb_png(decoded),
                static_cast<int64_t>(coerce_python_int_value(json_object_field(condition, "index", "0"), 0, true)),
                coerce_python_float_value(json_object_field(condition, "strength", "1.0"), 1.0, true)});
        }
        VideoSessionReservation reservation(video_mutex_, video_session_active_);
        const auto secondary = secondary_cuda_device(visible_cuda_devices());
        const auto loaded = backend_->loaded_models_json();
        const bool resident_llm = json_object_field(loaded, "llm", "null") != "null" || json_object_field(loaded, "vision_llm", "null") != "null";
        const bool unload = video_should_unload_llm(resident_llm, secondary, [this] { return !image_client().url.empty(); });
        std::unique_lock<std::mutex> handoff(llm_handoff_mutex_, std::defer_lock);
        if (unload) handoff.lock();
        std::vector<std::string> data;
        for (int64_t i = 0; i < count; ++i) {
            std::string output;
            LtxVideoSettings hint; hint.size = settings.size; hint.frames = settings.frames;
            const bool available = run_video_session(model, secondary, resident_llm, unload, [&] {
                ScopedCapabilityUse video_use(this, "video");
                if (invalid_settings) throw std::invalid_argument("Invalid video dimensions, frame count or generation settings.");
                if (json_object_field(body, "prompt", "\"\"") == "null") throw std::runtime_error("Video prompt is null");
                if (ltx_video_) {
                    LtxVideoSettings ltx_settings;
                    ltx_settings.size = settings.size; ltx_settings.frames = settings.frames;
                    ltx_settings.fps = settings.fps; ltx_settings.steps = settings.steps; ltx_settings.guidance = settings.guidance;
                    output = ltx_video_->generate(extract_json_string(body, "prompt", ""), ltx_settings,
                        as_url ? ezl_getenv_default("EZLOCALAI_URL") : "", image_bytes, ltx_conditions);
                } else output = video_->generate(extract_json_string(body, "prompt", ""), settings, as_url ? ezl_getenv_default("EZLOCALAI_URL") : "", image_bytes, conditions);
                if (!as_url) {
                    const auto bytes = read_file_or_empty(output);
                    output = base64_encode(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());
                }
            }, hint);
            if (!available || output.empty()) return no_output;
            data.push_back(std::string("{\"") + (as_url ? "url" : "b64_json") + "\":\"" + json_escape(output) + "\"}");
        }
        return {200, "application/json", "{\"created\":" + std::to_string(unix_seconds()) + ",\"data\":[" + join_strings(data, ",") + "]}"};
    } catch (const std::invalid_argument &error) { return bad_request(error.what()); }
    catch (const std::exception &) { return {500, "text/plain; charset=utf-8", "Internal Server Error"}; }
}

void NativeServer::load_video_backend(const std::string &model, const std::string &device,
    std::optional<LtxVideoSettings> hint) {
    if (is_ltx_model(model)) {
        if (!ltx_video_) ltx_video_ = std::make_unique<LtxVideo>(model, device, std::move(hint));
        else ltx_video_->warm();
        resources_->register_model(ModelType::Video, model, device, device.rfind("cuda", 0) == 0 ? 12.0 : 0.0);
    } else {
        if (!video_) video_ = std::make_unique<WanVideo>(model, device);
        resources_->register_model(ModelType::Video, model, device, 0.0);
    }
    std::lock_guard<std::mutex> lock(resource_mutex_);
    resource_video_loaded_ = true;
    resource_video_device_ = device;
}

bool NativeServer::run_video_session(const std::string &model, std::optional<int> secondary,
    bool resident_llm, bool unload, const std::function<void()> &operation,
    std::optional<LtxVideoSettings> hint) {
    std::function<void(bool)> restore;
    VideoAuxiliaryState aux_unloaded{};
    std::exception_ptr failure;
    bool unavailable = false;
    if (unload) wait_for_llm_idle_for_media("video", "VIDEO_WAIT_FOR_LLM_IDLE_TIMEOUT", true);
    try {
        if (unload) restore = backend_->unload_llms_for_video();
        unload_aux_models_for_video(unload, aux_unloaded);
        if (unload) {
            video_.reset(); ltx_video_.reset(); resources_->unregister_model(ModelType::Video);
            std::lock_guard<std::mutex> lock(resource_mutex_); resource_video_loaded_ = false;
        }
        if (!image_client().url.empty()) unavailable = true;
        else if ((!video_ && !ltx_video_) || (ltx_video_ && !ltx_video_->loaded())) {
            try {
                if (!is_ltx_model(model) || hint) load_video_backend(model, video_device(secondary), hint);
            }
            catch (const std::exception &e) { std::cerr << "[VIDEO] Failed to load the model: " << e.what() << '\n'; unavailable = true; }
        }
        if (!unavailable && (video_ || ltx_video_ || is_ltx_model(model))) operation();
    } catch (...) { failure = std::current_exception(); }
    if (ltx_video_ && !ltx_video_->loaded()) {
        resources_->unregister_model(ModelType::Video);
        std::lock_guard<std::mutex> lock(resource_mutex_); resource_video_loaded_ = false;
    }
    if (unload) {
        video_.reset(); ltx_video_.reset(); resources_->unregister_model(ModelType::Video);
        std::lock_guard<std::mutex> lock(resource_mutex_);
        resource_video_loaded_ = false;
    }
    try {
        if (restore) {
            restore(video_should_reload_llm());
            const auto restored = backend_->loaded_models_json();
            if (resident_llm && json_object_field(restored, "llm", "null") == "null" && json_object_field(restored, "vision_llm", "null") == "null") {
                // Pipes._restore_llms_after_video retries the saved model
                // names with configured settings if the initial restore
                // left every LLM unloaded, including reload=false.
                video_.reset(); ltx_video_.reset(); resources_->unregister_model(ModelType::Video);
                { std::lock_guard<std::mutex> lock(resource_mutex_); resource_video_loaded_ = false; }
                reclaim_auxiliary_memory_for_llm(16.0);
                restore(true);
            }
        }
        restore_aux_models_after_video(aux_unloaded);
    }
    catch (...) { failure = std::current_exception(); }
    if (unload) { std::lock_guard<std::mutex> lock(resource_mutex_); llm_temporarily_unavailable_ = false; llm_idle_cv_.notify_all(); }
    if (failure) std::rethrow_exception(failure);
    return !unavailable;
}

void NativeServer::wait_for_llm_idle_for_music(bool reserve) {
    wait_for_llm_idle_for_media("music", "MUSIC_WAIT_FOR_LLM_IDLE_TIMEOUT", reserve);
}

void NativeServer::wait_for_llm_idle_for_media(const std::string &service, const char *timeout_env, bool reserve) {
    double timeout = 0;
    const std::string raw = ezl_getenv_fallback(timeout_env, "60");
    if (!parse_strict_python_float_string(raw, &timeout)) throw std::invalid_argument("could not convert string to float: " + python_repr_string(raw));
    const auto started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(resource_mutex_);
    while (llm_slots_.total > 0) {
        if (timeout <= 0 || std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= std::max(0.0, timeout))
            throw std::runtime_error("LLM inference is active; " + service + " generation cannot unload the LLM yet");
        llm_idle_cv_.wait_for(lock, std::chrono::milliseconds(500));
    }
    if (reserve) llm_temporarily_unavailable_ = true;
}

HttpResponse NativeServer::generate_ace_music(const std::string &payload) {
    std::lock_guard<std::mutex> lock(music_mutex_);
    const bool unload = music_should_unload_llm();
    std::unique_lock<std::mutex> handoff(llm_handoff_mutex_, std::defer_lock);
    if (unload) handoff.lock();
    if (unload) wait_for_llm_idle_for_music(false);
    auto configured = [](const char *key, const char *fallback) {
        const auto raw = std::string(ezl_getenv_fallback(key, fallback));
        return strip_config_value(raw.empty() ? fallback : raw);
    };
    if (!music_initialized_) {
        ace_step_runtime().start([](const std::string &url) {
            const auto response = http_json_request("GET", url, "/health", "", "", 2);
            return !response.transport_error && response.status >= 200 && response.status < 300;
        }, [] {
            double timeout = 0;
            const std::string raw = ezl_getenv_fallback("ACE_STEP_STARTUP_TIMEOUT", "30");
            if (!parse_strict_python_float_string(raw, &timeout)) throw std::invalid_argument("could not convert string to float: " + python_repr_string(raw));
            return timeout;
        });
        music_server_url_ = ace_step_server_url();
        music_model_ = configured("MUSIC_MODEL", "Serveurperso/ACE-Step-1.5-GGUF");
        if (music_model_.empty()) music_model_ = "Serveurperso/ACE-Step-1.5-GGUF";
        music_lm_model_ = configured("ACE_STEP_LM_MODEL", "acestep-5Hz-lm-4B-Q8_0.gguf");
        music_synth_model_ = configured("ACE_STEP_DIT_MODEL", "acestep-v15-turbo-Q4_K_M.gguf");
        if (!parse_strict_python_float_string(ezl_getenv_fallback("ACE_STEP_TIMEOUT", "1800"), &music_timeout_)) throw std::invalid_argument("could not convert string to float: " + python_repr_string(ezl_getenv_fallback("ACE_STEP_TIMEOUT", "1800")));
        if (!parse_strict_python_float_string(ezl_getenv_fallback("ACE_STEP_POLL_INTERVAL", "1.0"), &music_poll_interval_)) throw std::invalid_argument("could not convert string to float: " + python_repr_string(ezl_getenv_fallback("ACE_STEP_POLL_INTERVAL", "1.0")));
        music_initialized_ = true;
    }
    // Startup can take seconds. Recheck admission before detaching engines so
    // an inference admitted during startup is allowed to finish safely.
    if (unload) wait_for_llm_idle_for_music(true);
    std::function<void(bool)> restore;
    HttpResponse response;
    std::exception_ptr failure;
    {
        ScopedCapabilityUse music_use(this, "music");
        try {
            if (unload) restore = backend_->unload_llms_for_media();
            response = request_ace_music(payload);
        } catch (...) { failure = std::current_exception(); }
    }
    try { if (restore) restore(music_should_reload_llm()); }
    catch (...) { failure = std::current_exception(); }
    if (unload) {
        std::lock_guard<std::mutex> resource_lock(resource_mutex_);
        llm_temporarily_unavailable_ = false;
        llm_idle_cv_.notify_all();
    }
    if (failure) std::rethrow_exception(failure);
    return response;
}

HttpResponse NativeServer::request_ace_music(const std::string &payload) {
    auto string_field = [&](const std::string &name, const std::string &fallback) {
        return python_truthy_string_value_or_default(json_object_field(payload, name, ""), fallback);
    };
    const auto response_format = to_lower(strip_config_value(string_field("response_format", "url")));
    if (response_format != "url" && response_format != "b64_json") throw std::invalid_argument("response_format must be one of ['b64_json', 'url']");
    const auto raw_output = parse_json_string_at(json_object_field(payload, "output_format", "\"wav16\""), 0);
    const auto output_format = to_lower(strip_config_value(raw_output.empty() ? "wav16" : raw_output));
    if (output_format != "mp3" && output_format != "wav16" && output_format != "wav24" && output_format != "wav32") throw std::invalid_argument("output_format must be one of ['mp3', 'wav16', 'wav24', 'wav32']");
    const auto prompt = strip_config_value(string_field("prompt", ""));
    if (prompt.empty()) throw std::invalid_argument("prompt is required");
    auto batch = json_object_field(payload, "n", "1");
    if (batch == "0" || starts_with(batch, "-")) batch = "1";
    std::vector<std::pair<std::string, std::string>> fields = {
        {"task_type", "\"text2music\""}, {"caption", "\"" + json_escape(prompt) + "\""},
        {"output_format", "\"" + output_format + "\""},
        {"lm_batch_size", batch}
    };
    const auto lyrics = json_object_field(payload, "lyrics", "");
    if (extract_json_bool(payload, "instrumental", false)) fields.emplace_back("lyrics", "\"[Instrumental]\"");
    else if (!lyrics.empty()) fields.emplace_back("lyrics", lyrics);
    const std::map<std::string, std::string> defaults = {{"bpm", "128"}, {"timesignature", "\"4/4\""}, {"vocal_language", "\"en\""},
        {"inference_steps", "16"}, {"guidance_scale", "1.0"}, {"shift", "3.0"}, {"solver", "\"euler\""},
        {"lm_model", "\"" + json_escape(music_lm_model_) + "\""}, {"synth_model", "\"" + json_escape(music_synth_model_) + "\""}};
    for (const auto *name : {"duration", "seed", "bpm", "keyscale", "timesignature", "vocal_language", "inference_steps", "guidance_scale", "shift", "solver", "lm_model", "synth_model", "audio_codes"}) {
        const std::string key = name;
        const auto fallback = defaults.count(key) ? defaults.at(key) : "";
        const auto raw = json_object_field(payload, key, fallback);
        if (!raw.empty() && raw != "null" && raw != "\"\"") fields.emplace_back(key, raw);
    }
    std::vector<std::pair<std::string, std::string>> extra;
    for (const auto &entry : json_object_ordered_entries(json_object_field(payload, "extra_parameters", "{}"))) if (trim(entry.second) != "null") extra.push_back(entry);
    const auto ace_request = merge_json_objects(json_object_from_ordered_entries(fields), json_object_from_ordered_entries(extra));
    auto get = [&](const std::string &path, int timeout) { return http_json_request("GET", music_server_url_, path, "", "", timeout); };
    auto run_job = [&](const std::string &endpoint, const std::string &request) {
        auto submitted = http_json_request("POST", music_server_url_, endpoint, "", request, 30);
        if (submitted.status != 200) throw std::runtime_error("ACE-Step " + endpoint + " submit failed with HTTP " + std::to_string(submitted.status) + ": " + submitted.body.substr(0, 300));
        if (!is_json_value_token(trim(submitted.body))) throw std::runtime_error("ACE-Step " + endpoint + " returned invalid JSON: " + submitted.body.substr(0, 300));
        if (!is_valid_top_level_json_object(submitted.body)) throw std::runtime_error("'" + std::string(starts_with(trim(submitted.body), "[") ? "list" : trim(submitted.body) == "null" ? "NoneType" : "str") + "' object has no attribute 'get'");
        const auto job = python_truthy_string_value_or_default(json_object_field(submitted.body, "id", ""), "");
        if (job.empty()) throw std::runtime_error("ACE-Step " + endpoint + " response did not include a job id");
        const auto path = "/job?id=" + url_encode_component(job);
        const auto started = std::chrono::steady_clock::now();
        bool done = false;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < music_timeout_) {
            const auto poll = get(path, 30);
            if (poll.status != 200) throw std::runtime_error("ACE-Step job " + job + " poll failed with HTTP " + std::to_string(poll.status) + ": " + poll.body.substr(0, 300));
            if (!is_json_value_token(trim(poll.body))) throw std::runtime_error("ACE-Step job " + job + " returned invalid status JSON: " + poll.body.substr(0, 300));
            if (!is_valid_top_level_json_object(poll.body)) throw std::runtime_error("'" + std::string(starts_with(trim(poll.body), "[") ? "list" : trim(poll.body) == "null" ? "NoneType" : "str") + "' object has no attribute 'get'");
            const auto state = to_lower(python_str_from_json_value(json_object_field(poll.body, "status", "\"\"")));
            if (state == "done") { done = true; break; }
            if (state == "failed" || state == "cancelled") {
                const auto detail = python_truthy_string_value_or_default(json_object_field(poll.body, "error", ""), python_truthy_string_value_or_default(json_object_field(poll.body, "message", ""), poll.body));
                throw std::runtime_error("ACE-Step job " + job + " ended with status " + state + ": " + detail);
            }
            if (music_poll_interval_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(music_poll_interval_));
        }
        if (!done) {
            std::ostringstream seconds; seconds << std::fixed << std::setprecision(0) << music_timeout_;
            throw std::runtime_error("ACE-Step job " + job + " timed out after " + seconds.str() + "s");
        }
        auto result = get(path + "&result=1", 120);
        if (result.status != 200) throw std::runtime_error("ACE-Step job " + job + " result failed with HTTP " + std::to_string(result.status) + ": " + result.body.substr(0, 300));
        return result;
    };
    std::vector<std::string> requests;
    if (!is_python_falsey_json_value(json_object_field(ace_request, "audio_codes", ""))) requests.push_back(ace_request);
    else {
        const auto lm = run_job("/lm", ace_request);
        const auto raw = trim(lm.body);
        if (!is_json_value_token(raw)) throw std::runtime_error("ACE-Step LM result was not valid JSON");
        if (is_valid_top_level_json_object(raw)) requests.push_back(raw);
        else if (!raw.empty() && raw.front() == '[') {
            for (const auto &item : json_array_elements(raw)) if (is_valid_top_level_json_object(item)) requests.push_back(item);
        } else throw std::runtime_error("ACE-Step LM result must be a JSON object or array");
        if (requests.empty()) throw std::runtime_error("ACE-Step LM returned no synth requests");
    }
    const auto synth = run_job("/synth", requests.size() == 1 ? requests.front() : "[" + join_strings(requests, ",") + "]");
    std::vector<std::pair<std::string, std::string>> audio;
    if (starts_with(to_lower(synth.content_type), "audio/")) audio.emplace_back(synth.body, synth.content_type);
    else if (starts_with(to_lower(synth.content_type), "multipart/")) {
        const auto boundary = extract_multipart_boundary(synth.content_type);
        if (!boundary.empty()) {
            const std::string marker = "--" + boundary;
            size_t position = synth.body.find(marker);
            while (position != std::string::npos) {
                position += marker.size();
                if (synth.body.compare(position, 2, "--") == 0) break;
                const auto headers_end = synth.body.find("\r\n\r\n", position);
                if (headers_end == std::string::npos) break;
                const auto next = synth.body.find("\r\n" + marker, headers_end + 4);
                if (next == std::string::npos) break;
                const auto headers = synth.body.substr(position, headers_end - position);
                const auto raw_type = multipart_header_value(headers, "content-type");
                const auto mime = to_lower(trim(raw_type.substr(0, raw_type.find(';'))));
                auto bytes = synth.body.substr(headers_end + 4, next - headers_end - 4);
                const auto encoding = to_lower(multipart_header_value(headers, "content-transfer-encoding"));
                if (encoding == "base64") bytes = base64_decode_string(bytes);
                if (starts_with(mime, "audio/") && !bytes.empty()) audio.emplace_back(std::move(bytes), mime);
                position = next + 2;
            }
        }
        if (audio.empty()) throw std::runtime_error("ACE-Step synth multipart result contained no audio parts");
    } else throw std::runtime_error("ACE-Step synth result was " + (synth.content_type.empty() ? std::string("missing content-type") : synth.content_type) + ", not audio or multipart");
    std::vector<std::string> data;
    for (size_t i = 0; i < audio.size(); ++i) {
        const auto &part = audio[i];
        std::vector<std::pair<std::string, std::string>> item = {{"content_type", "\"" + json_escape(part.second) + "\""}, {"format", "\"" + json_escape(raw_output) + "\""}};
        const auto &metadata = requests[std::min(i, requests.size() - 1)];
        if (!json_object_ordered_entries(metadata).empty()) {
            item.emplace_back("lyrics", json_object_field(metadata, "lyrics", "\"\""));
            item.emplace_back("caption", json_object_field(metadata, "caption", "\"\""));
            for (const auto *name : {"duration", "bpm", "keyscale", "timesignature", "vocal_language", "seed"}) {
                const auto value = json_object_field(metadata, name, "");
                if (!value.empty()) item.emplace_back(name, value);
            }
        }
        if (response_format == "b64_json") item.emplace_back("b64_json", "\"" + base64_encode(reinterpret_cast<const uint8_t *>(part.first.data()), part.first.size()) + "\"");
        else {
            std::filesystem::create_directories("outputs");
            const auto extension = part.second.find("mpeg") != std::string::npos || raw_output == "mp3" ? ".mp3" : ".wav";
            const auto filename = "outputs/" + uuid_v4() + "-" + std::to_string(i) + extension;
            std::ofstream output(filename, std::ios::binary);
            output.write(part.first.data(), static_cast<std::streamsize>(part.first.size()));
            if (!output) throw std::runtime_error("could not write " + filename);
            const std::string base = ezl_getenv_default("EZLOCALAI_URL");
            item.emplace_back("url", "\"" + json_escape(base.empty() ? filename : base + "/" + filename) + "\"");
        }
        data.push_back(json_object_from_ordered_entries(item));
    }
    return {200, "application/json", "{\"created\":" + std::to_string(unix_seconds()) + ",\"model\":\"" + json_escape(music_model_) + "\",\"data\":[" + join_strings(data, ",") + "]}"};
}

bool NativeServer::forward_image_server_request(const std::string &path, const std::string &body, HttpResponse *out) const {
    if (!out) {
        return false;
    }
    const std::string image_server = image_client().url;
    if (image_server.empty()) {
        return false;
    }
    const std::string api_key = image_client().api_key;
    if (!image_server_available_cached(image_server, api_key)) {
        return false;
    }

    std::string payload;
    std::string target_path = path;
    if (path == "/v1/videos/generations") {
        payload = video_forward_payload(body);
    } else {
        const bool edit_with_image = path == "/v1/images/edits";
        target_path = edit_with_image ? "/v1/images/edits" : "/v1/images/generations";
        payload = image_forward_payload(body, edit_with_image);
    }
    const auto response = fallback_json_request("POST", image_server, target_path, api_key, payload, path == "/v1/videos/generations" ? 1800 : 660);
    return service_json_response(response, out);
}

bool NativeServer::image_server_available_cached(const std::string &image_server, const std::string &api_key) const {
    const double now = monotonic_epoch_seconds();
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        if (image_cache_valid_
            && image_cache_url_ == image_server
            && now - image_cache_last_check_ < 30.0) {
            return image_cache_available_;
        }
    }
    const auto available = fallback_json_request("GET", image_server, "/v1/models", api_key, "", 5, false, false);
    const bool ok = available.status == 200;
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        image_cache_valid_ = true;
        image_cache_url_ = image_server;
        image_cache_last_check_ = now;
        image_cache_available_ = ok;
    }
    return ok;
}

bool NativeServer::forward_fallback_image_video_request(const std::string &path, const std::string &body, HttpResponse *out) const {
    if (!out) {
        return false;
    }
    const std::string fallback_server = fallback_client().url;
    if (fallback_server.empty()) {
        return false;
    }

    bool no_local_model = false;
    if (path == "/v1/images/generations" || path == "/v1/images/edits") {
        no_local_model = !has_local_capability("image");
    } else if (path == "/v1/videos/generations") {
        no_local_model = !has_local_capability("video");
    }
    const auto fallback_decision = native_fallback_decision(resources_.get(), fallback_client().url);
    if (!no_local_model && (path == "/v1/images/edits" || !fallback_decision.should_use)) {
        return false;
    }

    const std::string fallback_api_key = fallback_client().api_key;
    if (!fallback_server_available_cached(fallback_server, fallback_api_key)) {
        return false;
    }

    std::string target_path = path;
    std::string payload;
    if (path == "/v1/videos/generations") {
        payload = video_forward_payload(body, false);
    } else if (path == "/v1/images/edits") {
        payload = image_forward_payload(body, true);
    } else {
        target_path = "/v1/images/generations";
        payload = fallback_image_payload(body);
    }
    auto response = fallback_json_request("POST", fallback_server, target_path, fallback_api_key, payload, path == "/v1/videos/generations" ? 600 : 660);
    if (response.status != 200) {
        return false;
    }
    if (path == "/v1/videos/generations") {
        std::string value;
        if (!json_response_value(response, &value)) return false;
        response = {200, "application/json", value};
    }
    *out = response;
    return true;
}

HttpResponse NativeServer::upload_voice(const std::string &body, const std::string &content_type) const {
    const auto boundary = extract_multipart_boundary(content_type);
    const auto media_type = to_lower(trim(content_type.substr(0, content_type.find(';'))));
    if (media_type == "multipart/form-data" && boundary.empty())
        return {400, "application/json", "{\"detail\":\"Missing boundary in multipart.\"}"};
    auto parts = media_type == "multipart/form-data" ? parse_multipart_parts(body, boundary) : std::vector<MultipartPart>{};
    if (media_type == "application/x-www-form-urlencoded") {
        std::istringstream input(body);std::string field;
        while (std::getline(input,field,'&')) {
            const auto equal=field.find('=');
            parts.push_back({url_decode(field.substr(0,equal),true),"",false,"",equal==std::string::npos?"":url_decode(field.substr(equal+1),true),""});
        }
    }
    const MultipartPart *file=nullptr,*voice_field=nullptr;
    for (const auto &part:parts) {
        if(part.name=="file")file=&part;
        if(part.name=="voice")voice_field=&part;
    }
    auto decode_field=[](const std::string &raw) {
        std::string result;
        if(!fallback_decode(raw,"utf-8",&result))fallback_decode(raw,"latin-1",&result);
        return result;
    };
    std::vector<std::string> errors;
    if(voice_field&&voice_field->has_filename) {
        std::vector<std::pair<std::string,std::string>> headers;
        std::istringstream lines(voice_field->raw_headers);std::string line;
        while(std::getline(lines,line)) {
            if(!line.empty()&&line.back()=='\r')line.pop_back();const auto colon=line.find(':');
            if(colon!=std::string::npos)headers.emplace_back(to_lower(line.substr(0,colon)),'"'+json_escape(trim(line.substr(colon+1)))+'"');
        }
        // FastAPI exposes jsonable_encoder(UploadFile) in string-type errors.
        const auto spool=voice_field->content.empty()?std::string("{}"):
            "{\"_file\":{},\"_max_size\":1048576,\"_rolled\":false,\"_TemporaryFileArgs\":{\"mode\":\"w+b\",\"buffering\":-1,\"suffix\":null,\"prefix\":null,\"encoding\":null,\"newline\":null,\"dir\":null,\"errors\":null}}";
        const auto value="{\"filename\":\""+json_escape(voice_field->filename)+"\",\"file\":"+spool+",\"size\":"+std::to_string(voice_field->content.size())+",\"headers\":"+json_object_from_ordered_entries(headers)+",\"_max_mem_size\":1048576}";
        append_pydantic_error(&errors,"string_type","voice","Input should be a valid string",value);
    }
    if(!file||(!file->has_filename&&file->content.empty()))errors.push_back(pydantic_missing_file_error_object());
    else if(!file->has_filename)errors.push_back(pydantic_upload_file_type_error_object(decode_field(file->content)));
    if(!errors.empty())return {422,"application/json",pydantic_error_response_body(errors)};
    if (llm_unicode_lower(ezl_getenv_default("TTS_ENABLED")) == "false") {
        return {404, "application/json", "{\"detail\":\"Text to speech is disabled.\"}"};
    }
    try {
        std::string voice=sanitize_voice_name(voice_field?decode_field(voice_field->content):"default");
        const auto directory=std::filesystem::path(ctts_real_path((std::filesystem::current_path()/"voices").string()));
        std::filesystem::create_directories(directory);
        const auto base=voice;std::filesystem::path path;
        for(uint64_t i=0;;++i) {
            voice=i==0?base:base+"-"+std::to_string(i);path=directory/(voice+".wav");
            // os.path.exists follows symlinks and returns false on stat errors.
            struct stat metadata{};if(::stat(path.c_str(),&metadata)!=0)break;
        }
        std::ofstream out(path,std::ios::binary);if(!out)return plain_internal_server_error();
        out.write(file->content.data(),static_cast<std::streamsize>(file->content.size()));out.close();
        if(!out)return plain_internal_server_error();
        return {200,"application/json","{\"detail\":\"Voice "+json_escape(voice)+" has been uploaded.\"}"};
    } catch (...) {return plain_internal_server_error();}
}

double NativeServer::router_ttl_seconds() const {
    const char *configured = getenv("ROUTER_WORKER_TTL");
    const auto raw = trim(configured ? configured : "60");
    if (raw.empty()) {
        return 60.0;
    }
    double parsed = 0.0;
    return parse_strict_python_float_string(raw, &parsed) ? parsed : 60.0;
}

std::vector<NativeServer::RouterWorker> NativeServer::router_workers_in_registration_order_locked() const {
    std::vector<RouterWorker> workers;
    std::set<std::string> emitted;
    for (const auto &worker_id : router_worker_order_) {
        const auto found = router_workers_.find(worker_id);
        if (found == router_workers_.end()) {
            continue;
        }
        workers.push_back(found->second);
        emitted.insert(worker_id);
    }
    std::vector<RouterWorker> leftovers;
    for (const auto &entry : router_workers_) {
        if (emitted.find(entry.first) == emitted.end()) {
            leftovers.push_back(entry.second);
        }
    }
    std::sort(leftovers.begin(), leftovers.end(), [](const RouterWorker &lhs, const RouterWorker &rhs) {
        if (lhs.registered_at != rhs.registered_at) {
            return lhs.registered_at < rhs.registered_at;
        }
        return lhs.worker_id < rhs.worker_id;
    });
    workers.insert(workers.end(), leftovers.begin(), leftovers.end());
    return workers;
}

uint64_t NativeServer::router_tunnel_attach(const std::string &worker_id, int fd) {
    const double now = monotonic_epoch_seconds();
    std::lock_guard<std::mutex> lock(router_mutex_);
    RouterTunnelState &state = router_tunnels_[worker_id];
    state.connected = true;
    state.fd = fd;
    state.connected_at = now;
    state.last_recv = now;
    state.last_close_reason.clear();
    state.connect_count += 1;
    state.generation = ++router_tunnel_generation_;
    return state.generation;
}

void NativeServer::router_tunnel_note_recv(const std::string &worker_id, uint64_t generation) {
    std::lock_guard<std::mutex> lock(router_mutex_);
    auto it = router_tunnels_.find(worker_id);
    if (it != router_tunnels_.end() && it->second.generation == generation && it->second.connected) {
        it->second.last_recv = monotonic_epoch_seconds();
    }
}

void NativeServer::router_tunnel_detach(const std::string &worker_id, uint64_t generation, const std::string &reason) {
    const double now = monotonic_epoch_seconds();
    bool detached = false;
    std::string close_reason;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        auto it = router_tunnels_.find(worker_id);
        if (it == router_tunnels_.end() || it->second.generation != generation || !it->second.connected) {
            return;
        }
        RouterTunnelState &state = it->second;
        state.connected = false;
        state.fd = -1;
        state.last_close_reason = reason.empty() ? "ws closed" : reason;
        close_reason = state.last_close_reason;
        state.disconnect_history.push_back({
            now,
            std::max(0.0, now - state.connected_at),
            state.last_close_reason
        });
        if (state.disconnect_history.size() > 20) {
            state.disconnect_history.erase(
                state.disconnect_history.begin(),
                state.disconnect_history.begin() + static_cast<std::ptrdiff_t>(state.disconnect_history.size() - 20)
            );
        }
        detached = true;
    }
    if (detached) {
        std::lock_guard<std::mutex> pending_lock(router_tunnel_pending_mutex_);
        for (auto &entry : router_tunnel_pending_) {
            RouterTunnelPending &pending = entry.second;
            if (pending.tunnel_worker_id == worker_id && pending.generation == generation && !pending.done) {
                pending.error = "Tunnel " + worker_id + " disconnected: " + close_reason;
                pending.started = true;
                pending.done = true;
            }
        }
        router_tunnel_pending_cv_.notify_all();
    }
}

bool NativeServer::router_tunnel_connected(const std::string &worker_id) const {
    std::lock_guard<std::mutex> lock(router_mutex_);
    auto it = router_tunnels_.find(worker_id);
    return it != router_tunnels_.end() && it->second.connected;
}

void NativeServer::router_tunnel_handle_message(const std::string &message) {
    const std::string type = extract_json_string(message, "t", "");
    if (type == "ping" || type == "pong") {
        return;
    }
    const std::string id = extract_json_string(message, "id", "");
    if (id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(router_tunnel_pending_mutex_);
    auto pending_it = router_tunnel_pending_.find(id);
    if (pending_it == router_tunnel_pending_.end()) {
        return;
    }
    RouterTunnelPending &pending = pending_it->second;
    if (type == "resp_start") {
        pending.started = true;
        pending.headers_received = true;
        pending.status = extract_json_int_value(message, "status", 0);
        const std::string headers = extract_json_value(message, "headers", "{}");
        for (const auto &entry : json_object_ordered_entries(headers)) {
            if (to_lower(entry.first) == "content-type") {
                const std::string value = trim(entry.second);
                pending.content_type = value.size() >= 2 && value.front() == '"'
                    ? parse_json_string_at(value, 0)
                    : value;
                break;
            }
        }
        router_tunnel_pending_cv_.notify_all();
    } else if (type == "resp_chunk") {
        const std::string chunk = base64_decode_string(extract_json_string(message, "data_b64", ""));
        pending.body += chunk;
        pending.chunks.push_back(chunk);
        router_tunnel_pending_cv_.notify_all();
    } else if (type == "resp_end") {
        pending.done = true;
        router_tunnel_pending_cv_.notify_all();
    } else if (type == "resp_err") {
        pending.error = extract_json_string(message, "error", "tunnel error");
        pending.done = true;
        pending.started = true;
        router_tunnel_pending_cv_.notify_all();
    }
}

HttpResponse NativeServer::router_tunnel_request(
    const RouterWorker &worker,
    const std::string &method,
    const std::string &path,
    const std::string &api_key,
    const std::string &content_type,
    const std::string &body,
    bool stream,
    int timeout_ms,
    bool allow_deferred_streaming
) {
    const std::string tunnel_id = tunnel_worker_id(worker.url);
    int fd = -1;
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        auto tunnel_it = router_tunnels_.find(tunnel_id);
        if (tunnel_it == router_tunnels_.end() || !tunnel_it->second.connected || tunnel_it->second.fd < 0) {
            return worker_transport_error_response(
                "Tunnel for worker " + worker.label + " (" + tunnel_id + ") is not connected"
            );
        }
        fd = tunnel_it->second.fd;
        generation = tunnel_it->second.generation;
    }

    const std::string request_id = uuid_v4();
    {
        std::lock_guard<std::mutex> lock(router_tunnel_pending_mutex_);
        RouterTunnelPending pending;
        pending.tunnel_worker_id = tunnel_id;
        pending.generation = generation;
        router_tunnel_pending_[request_id] = pending;
    }

    std::ostringstream payload;
    payload << "{"
        << "\"t\":\"req\","
        << "\"id\":\"" << json_escape(request_id) << "\","
        << "\"method\":\"" << json_escape(method) << "\","
        << "\"path\":\"" << json_escape(path) << "\","
        << "\"headers\":{";
    bool wrote_header = false;
    if (!content_type.empty()) {
        payload << "\"Content-Type\":\"" << json_escape(content_type) << "\"";
        wrote_header = true;
    }
    if (!api_key.empty() && api_key != "none") {
        if (wrote_header) {
            payload << ",";
        }
        payload << "\"Authorization\":\"Bearer " << json_escape(api_key) << "\"";
    }
    payload << "},"
        << "\"stream\":" << (stream ? "true" : "false");
    if (!body.empty()) {
        payload << ",\"body_b64\":\""
            << base64_encode(reinterpret_cast<const unsigned char *>(body.data()), body.size())
            << "\"";
    }
    payload << "}";

    const std::string frame = websocket_text_frame(payload.str());
    bool sent_ok = true;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        auto tunnel_it = router_tunnels_.find(tunnel_id);
        if (tunnel_it == router_tunnels_.end()
            || !tunnel_it->second.connected
            || tunnel_it->second.generation != generation
            || tunnel_it->second.fd != fd) {
            sent_ok = false;
        } else {
            size_t sent = 0;
            while (sent < frame.size()) {
                ssize_t n = send(fd, frame.data() + sent, frame.size() - sent, 0);
                if (n <= 0) {
                    sent_ok = false;
                    break;
                }
                sent += static_cast<size_t>(n);
            }
        }
    }
    if (!sent_ok) {
        std::lock_guard<std::mutex> lock(router_tunnel_pending_mutex_);
        router_tunnel_pending_.erase(request_id);
        return worker_transport_error_response("Failed to send tunnel request");
    }

    const auto timeout = std::chrono::milliseconds(
        timeout_ms > 0 ? timeout_ms : (config_.request_timeout_ms > 0 ? config_.request_timeout_ms : 300000)
    );
    if (stream && allow_deferred_streaming) {
        int status = 0;
        std::string response_content_type;
        {
            std::unique_lock<std::mutex> lock(router_tunnel_pending_mutex_);
            const bool ready = router_tunnel_pending_cv_.wait_for(lock, timeout, [&]() {
                auto it = router_tunnel_pending_.find(request_id);
                return it == router_tunnel_pending_.end()
                    || it->second.started
                    || it->second.done
                    || !it->second.error.empty();
            });
            auto it = router_tunnel_pending_.find(request_id);
            if (!ready || it == router_tunnel_pending_.end()) {
                router_tunnel_pending_.erase(request_id);
                return worker_transport_error_response("Tunnel request to " + tunnel_id + path + " timed out before headers");
            }
            if (!it->second.error.empty() && !it->second.headers_received) {
                const std::string error = it->second.error;
                router_tunnel_pending_.erase(it);
                return worker_transport_error_response(error);
            }
            status = it->second.status > 0 ? it->second.status : 503;
            response_content_type = it->second.content_type;
        }

        HttpResponse response{status, response_content_type, ""};
        response.chunked = true;
        response.stream_capture = std::make_shared<std::string>();
        auto stream_capture = response.stream_capture;
        response.stream_writer = [this, request_id, timeout, stream_capture](int client_fd) {
            size_t next_chunk = 0;
            bool done = false;
            bool ok = true;
            while (ok && !done) {
                std::vector<std::string> chunks;
                {
                    std::unique_lock<std::mutex> lock(router_tunnel_pending_mutex_);
                    const bool ready = router_tunnel_pending_cv_.wait_for(lock, timeout, [&]() {
                        auto it = router_tunnel_pending_.find(request_id);
                        return it == router_tunnel_pending_.end()
                            || it->second.chunks.size() > next_chunk
                            || it->second.done
                            || !it->second.error.empty();
                    });
                    auto it = router_tunnel_pending_.find(request_id);
                    if (!ready || it == router_tunnel_pending_.end()) {
                        if (it != router_tunnel_pending_.end()) {
                            router_tunnel_pending_.erase(it);
                        }
                        break;
                    }
                    while (next_chunk < it->second.chunks.size()) {
                        chunks.push_back(it->second.chunks[next_chunk++]);
                    }
                    done = it->second.done || !it->second.error.empty();
                    if (done) {
                        router_tunnel_pending_.erase(it);
                    }
                }
                for (const auto &chunk : chunks) {
                    append_stream_capture(stream_capture, chunk);
                    ok = ok && send_http_chunk(client_fd, chunk);
                }
            }
            {
                std::lock_guard<std::mutex> lock(router_tunnel_pending_mutex_);
                router_tunnel_pending_.erase(request_id);
            }
            send_all_fd(client_fd, "0\r\n\r\n");
        };
        return response;
    }

    RouterTunnelPending completed;
    {
        std::unique_lock<std::mutex> lock(router_tunnel_pending_mutex_);
        const bool ready = router_tunnel_pending_cv_.wait_for(lock, timeout, [&]() {
            auto it = router_tunnel_pending_.find(request_id);
            return it == router_tunnel_pending_.end() || it->second.done || !it->second.error.empty();
        });
        auto it = router_tunnel_pending_.find(request_id);
        if (!ready || it == router_tunnel_pending_.end()) {
            std::string detail = "Tunnel request to " + tunnel_id + path + " timed out before headers";
            if (it != router_tunnel_pending_.end() && it->second.headers_received) {
                detail = "Tunnel response from " + tunnel_id + path + " timed out";
            }
            router_tunnel_pending_.erase(request_id);
            return worker_transport_error_response(detail);
        }
        completed = it->second;
        router_tunnel_pending_.erase(it);
    }
    if (!completed.error.empty() && !completed.headers_received) {
        return worker_transport_error_response(completed.error);
    }
    const int status = completed.status > 0 ? completed.status : 503;
    return {status, completed.content_type.empty() ? "application/json" : completed.content_type, completed.body};
}

std::string router_formatless_model(std::string name) {
    name = strip_config_value(name);
    const auto at = name.rfind('@');
    if (at != std::string::npos) name.resize(at);
    const auto replica = name.find('#');
    if (replica != std::string::npos) name.resize(replica);
    const auto slash = name.rfind('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    name = to_lower(name);
    for (bool changed = true; changed;) {
        changed = false;
        for (const std::string suffix : {"-gguf", ".gguf", "-tee"}) {
            if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                name.resize(name.size() - suffix.size()); changed = true; break;
            }
        }
    }
    return name;
}

bool router_model_name_matches(const std::string &requested, const std::string &served) {
    if (requested.empty()) return true;
    if (served.empty()) return false;
    const auto request = router_formatless_model(requested), service = router_formatless_model(served);
    if (to_lower(strip_config_value(requested)) == to_lower(strip_config_value(served)) || (!request.empty() && request == service)) return true;
    auto family_request = request, family_service = service;
    replace_all(&family_request, "-mtp", "");
    replace_all(&family_service, "-mtp", "");
    if (!family_request.empty() && family_request == family_service) {
        auto mtp = [](const std::string &name) { return name.find("-mtp") != std::string::npos || name.find("qwen3.8-27b") != std::string::npos; };
        return !mtp(request) || mtp(service);
    }
    return !request.empty() && to_lower(strip_config_value(served)).find(request) != std::string::npos;
}

std::vector<std::string> router_slot_names(const std::map<std::string, RouterSlotState> &slots, const std::vector<std::string> &order) {
    std::vector<std::string> names;
    for (const auto &key : order) if (slots.count(key)) names.push_back(key);
    for (const auto &entry : slots) if (std::find(names.begin(), names.end(), entry.first) == names.end()) names.push_back(entry.first);
    return names;
}

std::string router_match_name(const std::string &name, const std::vector<std::string> &names) {
    if (name.empty()) return "";
    if (std::find(names.begin(), names.end(), name) != names.end()) return name;
    const auto base = to_lower(name.substr(name.rfind('/') == std::string::npos ? 0 : name.rfind('/') + 1));
    for (const auto &key : names) {
        const auto key_base = to_lower(key.substr(key.rfind('/') == std::string::npos ? 0 : key.rfind('/') + 1));
        if (to_lower(key) == to_lower(name) || base == key_base) return key;
    }
    for (const auto &key : names) if (router_model_name_matches(name, key)) return key;
    for (const auto &key : names) if (!base.empty() && to_lower(key).find(base) != std::string::npos) return key;
    return "";
}

std::set<std::string> router_llm_dependencies(const std::string &extra) {
    bool ok = true;
    const auto values = python_list_strings_from_json_value(json_object_field(extra, "llm_unload_dependent_capabilities", "[]"), true, &ok);
    return {values.begin(), values.end()};
}

int NativeServer::router_reserved(const RouterWorker &worker, const std::string &capability, const std::string &model) const {
    const auto now = std::chrono::steady_clock::now();
    std::string model_key;
    if (!model.empty()) {
        std::vector<std::string> keys;
        for (const auto &request : worker.active_requests) if (request.expires_at > now && !request.model.empty()) keys.push_back(request.model);
        model_key = router_match_name(model, keys);
        if (model_key.empty()) return 0;
    }
    int count = 0;
    for (const auto &request : worker.active_requests) {
        if (request.expires_at > now && (capability.empty() || request.capability == capability) && (model_key.empty() || request.model == model_key)) ++count;
    }
    return count;
}

RouterSlotState NativeServer::router_slot_state(
    const RouterWorker &worker,
    const std::string &capability,
    const std::string &model
) const {
    if (!model.empty()) {
        const std::string model_key = router_match_name(model, router_slot_names(worker.model_slots, worker.model_slot_order));
        if (!model_key.empty()) {
            return normalize_slot(worker.model_slots.at(model_key), worker.model_slots.at(model_key).capacity);
        }
    }

    if (!capability.empty()) {
        auto cap = worker.cap_slots.find(capability);
        if (cap != worker.cap_slots.end()) {
            return normalize_slot(cap->second, cap->second.capacity);
        }
        if (capability == "vision") {
            auto text = worker.cap_slots.find("text");
            if (text != worker.cap_slots.end()) {
                return normalize_slot(text->second, text->second.capacity);
            }
        }
    }

    int fallback_capacity = std::max(1, worker.queue_capacity);
    if (capability == "image" || capability == "stt" || capability == "video" || capability == "tts" || capability == "music" || capability == "music_video") {
        fallback_capacity = 1;
    }
    const int busy = router_effective_busy(worker);
    RouterSlotState fallback;
    fallback.capacity = fallback_capacity;
    fallback.in_flight = std::min(busy, fallback_capacity);
    fallback.queued = std::max(0, busy - fallback_capacity);
    return normalize_slot(fallback, fallback_capacity);
}

int NativeServer::router_effective_busy(const RouterWorker &worker, const std::string &capability, const std::string &model) const {
    if (!capability.empty() || !model.empty()) {
        const auto state = router_slot_state(worker, capability, model);
        int reported = state.in_flight + state.queued;
        int reserved = 0;
        const auto model_key = router_match_name(model, router_slot_names(worker.model_slots, worker.model_slot_order));
        const auto cap_key = worker.cap_slots.count(capability) ? capability : capability == "vision" && worker.cap_slots.count("text") ? "text" : "";
        auto cap_busy = [&](const std::string &cap) {
            const auto it = worker.cap_slots.find(cap);
            return it == worker.cap_slots.end() ? 0 : it->second.in_flight + it->second.queued;
        };
        if (!model_key.empty()) {
            reserved = router_reserved(worker, "", model);
            if (json_object_field(worker.extra_json, "llm_model_residency", "") == "\"swap\"") {
                reported = std::max(reported, cap_busy("text"));
                reserved = std::max(reserved, router_reserved(worker, "text"));
            }
        } else if (!cap_key.empty()) reserved = router_reserved(worker, cap_key);
        else reserved = router_reserved(worker);
        std::vector<std::string> conflicts;
        if (capability == "video" || capability == "music") conflicts.push_back("music_video");
        else if (capability == "music_video") conflicts = {"video", "music"};
        for (const auto &cap : conflicts) {
            reported = std::max(reported, cap_busy(cap));
            reserved = std::max(reserved, router_reserved(worker, cap));
        }
        auto dependencies = router_llm_dependencies(worker.extra_json);
        const bool handoff = dependencies.count(capability);
        if (handoff || !model_key.empty() || capability == "text" || capability == "vision") {
            if (handoff) dependencies.insert("text");
            for (const auto &cap : dependencies) {
                if (handoff && cap == capability) continue;
                reported = std::max({reported, cap_busy(cap), router_reserved(worker, cap)});
            }
        }
        return std::max(reported, reserved);
    }
    return std::max({worker.queue_depth, worker.in_flight, router_reserved(worker)});
}

int NativeServer::router_capacity_for(const RouterWorker &worker, const std::string &capability, const std::string &model) const {
    return router_slot_state(worker, capability, model).capacity;
}

int NativeServer::router_slots_left(const RouterWorker &worker, const std::string &capability, const std::string &model) const {
    if (router_worker_draining(worker)) return 0;
    const auto state = router_slot_state(worker, capability, model);
    return std::max(0, state.capacity - router_effective_busy(worker, capability, model));
}

int NativeServer::router_total_capacity(const RouterWorker &worker) const {
    if (worker.cap_slots.empty()) {
        return std::max(1, worker.queue_capacity);
    }
    int total = 0;
    std::set<std::string> shared;
    for (const auto &cap : router_llm_dependencies(worker.extra_json)) if (worker.cap_slots.count(cap)) shared.insert(cap);
    if (!shared.empty() && worker.cap_slots.count("text")) shared.insert("text");
    for (const auto &cap : shared) total = std::max(total, worker.cap_slots.at(cap).capacity);
    for (const auto &entry : worker.cap_slots) {
        if (shared.count(entry.first) || (entry.first == "vision" && worker.cap_slots.find("text") != worker.cap_slots.end())) {
            continue;
        }
        total += normalize_slot(entry.second, entry.second.capacity).capacity;
    }
    return total;
}

int NativeServer::router_total_busy(const RouterWorker &worker) const {
    if (worker.cap_slots.empty()) {
        return router_effective_busy(worker);
    }
    int total = 0;
    std::set<std::string> shared;
    for (const auto &cap : router_llm_dependencies(worker.extra_json)) if (worker.cap_slots.count(cap)) shared.insert(cap);
    if (!shared.empty() && worker.cap_slots.count("text")) shared.insert("text");
    for (const auto &cap : shared) total = std::max(total, router_effective_busy(worker, cap));
    for (const auto &entry : worker.cap_slots) {
        if (shared.count(entry.first) || (entry.first == "vision" && worker.cap_slots.find("text") != worker.cap_slots.end())) {
            continue;
        }
        const auto state = normalize_slot(entry.second, entry.second.capacity);
        total += std::max(state.in_flight + state.queued, router_reserved(worker, entry.first));
    }
    return std::max(total, router_reserved(worker));
}

int NativeServer::router_total_slots_left(const RouterWorker &worker) const {
    return std::max(0, router_total_capacity(worker) - router_total_busy(worker));
}

double NativeServer::router_score(const RouterWorker &worker, const std::string &capability, const std::string &model) const {
    const int busy = router_effective_busy(worker, capability, model);
    const int slots_left = router_slots_left(worker, capability, model);
    const double failure_penalty = std::min(worker.connection_failures, 3) * 5.0;
    return worker.best_tier * 10.0
        + slots_left * 5.0
        + worker.free_vram_gb
        - busy * 4.0
        - failure_penalty;
}

bool NativeServer::router_reserve_request(const std::string &worker_id, const std::string &capability, const std::string &model, bool require_capacity, std::string *token, std::string *error) {
    std::lock_guard<std::mutex> lock(router_mutex_);
    auto found = router_workers_.find(worker_id);
    if (found == router_workers_.end()) return false;
    auto &worker = found->second;
    if (router_worker_draining(worker)) {
        if (!require_capacity) *error = "Worker " + worker.label + " is draining";
        return false;
    }
    if (require_capacity && router_slots_left(worker, capability, model) <= 0) return false;
    const bool scoped = capability != "text" && capability != "vision";
    if (!scoped && router_reservation_ttl_ <= 0) return false;
    const auto cap_key = capability == "vision" && worker.cap_slots.count("text") ? "text" : capability;
    auto model_key = router_match_name(model, router_slot_names(worker.model_slots, worker.model_slot_order));
    if (model_key.empty()) model_key = model;
    *token = uuid_v4();
    const auto expiry = scoped ? std::chrono::steady_clock::time_point::max()
        : std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(router_reservation_ttl_));
    worker.active_requests.push_back({*token, cap_key, model_key, expiry});
    return true;
}

void NativeServer::router_release_request(const std::string &worker_id, const std::string &token) {
    if (token.empty()) return;
    std::lock_guard<std::mutex> lock(router_mutex_);
    const auto found = router_workers_.find(worker_id);
    if (found == router_workers_.end()) return;
    auto &requests = found->second.active_requests;
    requests.erase(std::remove_if(requests.begin(), requests.end(), [&](const auto &request) { return request.token == token; }), requests.end());
}

std::string NativeServer::router_effective_slots_json(const RouterWorker &worker, bool models) const {
    const auto &slots = models ? worker.model_slots : worker.cap_slots;
    std::vector<std::pair<std::string, std::string>> fields;
    for (const auto &name : router_slot_names(slots, models ? worker.model_slot_order : worker.cap_slot_order)) {
        auto state = normalize_slot(slots.at(name), slots.at(name).capacity);
        const auto cap = models ? "" : name, model = models ? name : "";
        state.available = router_slots_left(worker, cap, model);
        auto raw = json_object_field(slot_map_json({{name, state}}), name, "{}");
        // slot_map_json recalculates available from the reported counters.
        raw = merge_json_objects(raw, "{\"available\":" + std::to_string(state.available) + ",\"effective_busy\":" + std::to_string(router_effective_busy(worker, cap, model)) + "}");
        fields.emplace_back(name, raw);
    }
    return json_object_from_ordered_entries(fields);
}

void NativeServer::router_record_error(const std::string &worker_id, const std::string &kind, const std::string &path, const std::string &message, int status) {
    auto env_or = [](const char *name, const char *fallback) -> std::string {
        const char *value = getenv(name);
        return value ? value : fallback;
    };
    double window = 60.0;
    int threshold = 3;
    double cooldown = 30.0;
    int history_max = 50;
    double parsed_window = 0.0;
    int parsed_threshold = 0;
    double parsed_cooldown = 0.0;
    int parsed_history_max = 0;
    if (parse_strict_python_float_string(env_or("ROUTER_ERROR_WINDOW_SECONDS", "60"), &parsed_window)
        && parse_strict_int_string(env_or("ROUTER_ERROR_THRESHOLD", "3"), &parsed_threshold)
        && parse_strict_python_float_string(env_or("ROUTER_CIRCUIT_COOLDOWN", "30"), &parsed_cooldown)
        && parse_strict_int_string(env_or("ROUTER_ERROR_HISTORY_MAX", "50"), &parsed_history_max)) {
        window = parsed_window;
        threshold = parsed_threshold;
        cooldown = parsed_cooldown;
        history_max = parsed_history_max;
    }

    const double now = monotonic_epoch_seconds();
    std::lock_guard<std::mutex> lock(router_mutex_);
    auto found = router_workers_.find(worker_id);
    if (found == router_workers_.end()) {
        return;
    }
    auto &worker = found->second;
    RouterWorker::RecentError error;
    error.ts = now;
    error.kind = kind;
    error.status = status;
    error.path = path;
    error.message = message.size() > 500 ? message.substr(0, 500) : message;
    worker.recent_errors.push_back(error);
    worker.total_errors++;
    if (history_max <= 0) {
        worker.recent_errors.clear();
    } else if (static_cast<int>(worker.recent_errors.size()) > history_max) {
        worker.recent_errors.erase(
            worker.recent_errors.begin(),
            worker.recent_errors.begin() + (worker.recent_errors.size() - static_cast<size_t>(history_max))
        );
    }
    int recent_in_window = 0;
    for (const auto &entry : worker.recent_errors) {
        if (now - entry.ts <= window) {
            recent_in_window++;
        }
    }
    if (recent_in_window >= threshold && now >= worker.circuit_open_until) {
        worker.circuit_open_until = now + cooldown;
    }
}

std::string NativeServer::router_recent_errors_json(const RouterWorker &worker) const {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < worker.recent_errors.size(); ++i) {
        if (i) {
            out << ",";
        }
        const auto &entry = worker.recent_errors[i];
        out << "{"
            << "\"ts\":" << json_number_literal(entry.ts) << ","
            << "\"kind\":\"" << json_escape(entry.kind) << "\",";
        if (entry.status > 0) {
            out << "\"status\":" << entry.status << ",";
        } else {
            out << "\"status\":null,";
        }
        out << "\"path\":\"" << json_escape(entry.path) << "\","
            << "\"message\":\"" << json_escape(entry.message) << "\""
            << "}";
    }
    out << "]";
    return out.str();
}

std::string NativeServer::router_aggregate_errors_json(const std::vector<RouterWorker> &workers) const {
    struct Aggregate {
        double ts = 0.0;
        std::string worker_id;
        std::string label;
        std::string kind;
        int status = 0;
        std::string path;
        std::string message;
    };
    std::vector<Aggregate> errors;
    for (const auto &worker : workers) {
        for (const auto &entry : worker.recent_errors) {
            errors.push_back({entry.ts, worker.worker_id, worker.label, entry.kind, entry.status, entry.path, entry.message});
        }
    }
    std::sort(errors.begin(), errors.end(), [](const Aggregate &lhs, const Aggregate &rhs) {
        return lhs.ts > rhs.ts;
    });
    if (errors.size() > 100) {
        errors.resize(100);
    }
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i) {
            out << ",";
        }
        const auto &entry = errors[i];
        out << "{"
            << "\"ts\":" << json_number_literal(entry.ts) << ","
            << "\"worker_id\":\"" << json_escape(entry.worker_id) << "\","
            << "\"label\":\"" << json_escape(entry.label) << "\","
            << "\"kind\":\"" << json_escape(entry.kind) << "\",";
        if (entry.status > 0) {
            out << "\"status\":" << entry.status << ",";
        } else {
            out << "\"status\":null,";
        }
        out << "\"path\":\"" << json_escape(entry.path) << "\","
            << "\"message\":\"" << json_escape(entry.message) << "\""
            << "}";
    }
    out << "]";
    return out.str();
}

std::string NativeServer::router_worker_public_json(const RouterWorker &worker) const {
    const double now = monotonic_epoch_seconds();
    const bool circuit_open = now < worker.circuit_open_until;
    auto counts_json = [&](bool models) {
        std::vector<std::pair<std::string, std::string>> fields;
        const auto now = std::chrono::steady_clock::now();
        for (const auto &request : worker.active_requests) {
            const auto &key = models ? request.model : request.capability;
            if (request.expires_at <= now || key.empty()) continue;
            auto existing = std::find_if(fields.begin(), fields.end(), [&](const auto &entry) { return entry.first == key; });
            if (existing == fields.end()) fields.emplace_back(key, "1");
            else existing->second = std::to_string(std::stoi(existing->second) + 1);
        }
        return json_object_from_ordered_entries(fields);
    };
    std::ostringstream out;
    out << "{"
        << "\"worker_id\":\"" << json_escape(worker.worker_id) << "\","
        << "\"label\":\"" << json_escape(worker.label) << "\","
        << "\"url\":\"" << json_escape(worker.url) << "\","
        << "\"capabilities\":" << worker.capabilities_json << ","
        << "\"models\":" << worker.models_json << ","
        << "\"free_vram_gb\":" << json_number_literal(worker.free_vram_gb) << ","
        << "\"total_vram_gb\":" << json_number_literal(worker.total_vram_gb) << ","
        << "\"free_ram_gb\":" << json_number_literal(worker.free_ram_gb) << ","
        << "\"total_ram_gb\":" << json_number_literal(worker.total_ram_gb) << ","
        << "\"queue_depth\":" << worker.queue_depth << ","
        << "\"queue_capacity\":" << worker.queue_capacity << ","
        << "\"in_flight\":" << router_effective_busy(worker) << ","
        << "\"reported_in_flight\":" << worker.in_flight << ","
        << "\"router_in_flight\":" << router_reserved(worker) << ","
        << "\"router_cap_in_flight\":" << counts_json(false) << ","
        << "\"router_model_in_flight\":" << counts_json(true) << ","
        << "\"cap_slots\":" << slot_map_json(worker.cap_slots, worker.cap_slot_order) << ","
        << "\"model_slots\":" << slot_map_json(worker.model_slots, worker.model_slot_order) << ","
        << "\"effective_cap_slots\":" << router_effective_slots_json(worker, false) << ","
        << "\"effective_model_slots\":" << router_effective_slots_json(worker, true) << ","
        << "\"slot_total_capacity\":" << router_total_capacity(worker) << ","
        << "\"slot_total_busy\":" << router_total_busy(worker) << ","
        << "\"slot_total_available\":" << router_total_slots_left(worker) << ","
        << "\"gpus\":" << worker.gpus_json << ","
        << "\"best_tier\":" << worker.best_tier << ","
        << "\"priority_tier\":" << worker.best_tier - (is_tunnel_url(worker.url) ? 5 : 0) << ","
        << "\"model_context\":" << worker.model_context_json << ","
        << "\"model_quant\":" << worker.model_quant_json << ","
        << "\"cap_models\":" << worker.cap_models_json << ","
        << "\"age_seconds\":" << json_number_literal(std::max(0.0, now - worker.registered_at)) << ","
        << "\"last_heartbeat_age\":" << json_number_literal(std::max(0.0, now - worker.last_heartbeat)) << ","
        << "\"version\":\"" << json_escape(worker.version) << "\","
        << "\"connection_failures\":" << worker.connection_failures << ","
        << "\"total_errors\":" << worker.total_errors << ","
        << "\"recent_errors\":" << router_recent_errors_json(worker) << ","
        << "\"circuit_open_until\":" << json_number_literal(worker.circuit_open_until) << ","
        << "\"circuit_open\":" << (circuit_open ? "true" : "false") << ","
        << "\"draining\":" << (router_worker_draining(worker) ? "true" : "false") << ","
        << "\"drain_until\":" << (router_worker_draining(worker) ? json_number_literal(worker.drain_until) : "null") << ","
        << "\"active_requests\":" << worker.active_requests.size() << ","
        << "\"maintenance_busy\":" << router_maintenance_busy(worker) << ","
        << "\"external_provider\":\"\",\"external_fallback\":false,\"persistent\":false,\"external_balance_usd\":null,\"external_balance_updated_at\":0.0,"
        << "\"extra\":" << worker.extra_json
        << "}";
    return out.str();
}

std::string NativeServer::router_worker_dashboard_json(const RouterWorker &worker) const {
    std::string pub = router_worker_public_json(worker);
    if (!pub.empty() && pub.back() == '}') {
        pub.pop_back();
    }
    pub += ",\"tunnel\":";
    pub += is_tunnel_url(worker.url) ? "true," : "false,";
    if (is_tunnel_url(worker.url)) {
        pub += "\"tunnel_connected\":";
        pub += router_tunnel_connected(tunnel_worker_id(worker.url)) ? "true" : "false";
    } else {
        pub += "\"tunnel_connected\":null";
    }
    pub += "}";
    return pub;
}

HttpResponse NativeServer::router_register(const std::string &body, const std::string &client_host) {
    if (!is_valid_top_level_json_object(body)) {
        return invalid_json_object_response(body);
    }
    const auto request_values = top_level_json_object_values(body);
    auto field = [&request_values](const std::string &key, const std::string &fallback = "") -> std::string {
        const auto found = request_values.find(key);
        return found == request_values.end() ? fallback : found->second;
    };
    const auto worker_id = extract_router_worker_id(body);
    if (worker_id.empty()) {
        return {400, "application/json", "{\"detail\":\"Missing field: worker_id\"}"};
    }
    for (const auto &key : {"port", "queue_depth", "queue_capacity", "in_flight", "best_tier"}) {
        const auto raw = field(key);
        if (!raw.empty() && json_value_would_fail_python_int(raw, true)) {
            return plain_internal_server_error();
        }
    }
    for (const auto &key : {"free_vram_gb", "total_vram_gb", "free_ram_gb", "total_ram_gb"}) {
        const auto raw = field(key);
        if (!raw.empty() && json_value_would_fail_python_float(raw, true)) {
            return plain_internal_server_error();
        }
    }
    bool list_ok = true;
    const auto capabilities_raw = field("capabilities");
    const auto models_raw = field("models");
    const auto capabilities = python_list_strings_from_json_value(capabilities_raw, true, &list_ok);
    if (!list_ok) {
        return plain_internal_server_error();
    }
    const auto models = python_list_strings_from_json_value(models_raw, true, &list_ok);
    if (!list_ok) {
        return plain_internal_server_error();
    }
    const auto capabilities_public_json = python_public_list_json_from_value(capabilities_raw, true, &list_ok);
    if (!list_ok) {
        return plain_internal_server_error();
    }
    const auto models_public_json = python_public_list_json_from_value(models_raw, true, &list_ok);
    if (!list_ok) {
        return plain_internal_server_error();
    }
    const auto gpus_raw = field("gpus");
    const auto best_tier_raw = field("best_tier");
    if (router_register_gpus_would_fail_python(gpus_raw, best_tier_raw)) {
        return plain_internal_server_error();
    }
    for (const auto &key : {"cap_slots", "model_slots", "model_quant", "cap_models", "extra"}) {
        if (json_truthy_non_dict(field(key))) {
            return plain_internal_server_error();
        }
    }
    if (json_slot_map_values_would_fail_python(field("cap_slots"))
        || json_slot_map_values_would_fail_python(field("model_slots"))) {
        return plain_internal_server_error();
    }
    if (json_object_int_values_would_fail_python(field("model_context"), true)) {
        return plain_internal_server_error();
    }

    RouterWorker worker;
    worker.worker_id = worker_id;
    worker.label = python_truthy_string_value_or_default(field("label"), worker_id);
    worker.url = strip_trailing_slashes(trim(python_truthy_string_value_or_default(field("url"), "")));
    if (worker.url.empty() || (!is_tunnel_url(worker.url) && is_loopback_url(worker.url))) {
        if (client_host.empty()) {
            return {400, "application/json", "{\"detail\":\"Worker did not provide a public 'url' and the router could not determine the source IP.\"}"};
        }
        const int port = coerce_python_int_value(field("port"), 8091, true);
        worker.url = "http://" + client_host + ":" + std::to_string(port);
    }
    worker.api_key = python_truthy_string_value_or_default(field("api_key"), "");
    worker.capabilities = capabilities;
    worker.models = models;
    worker.capabilities_json = capabilities_public_json;
    worker.models_json = models_public_json;
    worker.free_vram_gb = coerce_python_float_value(field("free_vram_gb"), 0.0, true);
    worker.total_vram_gb = coerce_python_float_value(field("total_vram_gb"), 0.0, true);
    worker.free_ram_gb = coerce_python_float_value(field("free_ram_gb"), 0.0, true);
    worker.total_ram_gb = coerce_python_float_value(field("total_ram_gb"), 0.0, true);
    worker.queue_depth = coerce_python_int_value(field("queue_depth"), 0, true);
    worker.queue_capacity = coerce_python_int_value(field("queue_capacity"), 1, true);
    worker.in_flight = coerce_python_int_value(field("in_flight"), 0, true);
    const auto cap_slots_raw = field("cap_slots", "{}");
    const auto model_slots_raw = field("model_slots", "{}");
    worker.cap_slots = parse_slot_map_json(cap_slots_raw);
    worker.model_slots = parse_slot_map_json(model_slots_raw);
    worker.cap_slot_order = parse_slot_map_order_json(cap_slots_raw);
    worker.model_slot_order = parse_slot_map_order_json(model_slots_raw);
    worker.gpus_json = python_list_json_from_value(field("gpus", "[]"));
    worker.best_tier = coerce_python_int_value(field("best_tier"), 0, true);
    if (worker.best_tier == 0) {
        worker.best_tier = best_gpu_tier_from_json(worker.gpus_json);
    }
    worker.model_context_json = python_int_map_json_from_value(field("model_context", "{}"));
    worker.model_quant_json = python_string_map_json_from_value(field("model_quant", "{}"));
    worker.cap_models_json = python_string_map_json_from_value(field("cap_models", "{}"));
    worker.version = python_truthy_string_value_or_default(field("version"), "");
    worker.extra_json = router_register_extra_json(body);
    worker.registered_at = monotonic_epoch_seconds();
    worker.last_heartbeat = worker.registered_at;

    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        auto existing = router_workers_.find(worker.worker_id);
        if (existing != router_workers_.end()) {
            worker.registered_at = existing->second.registered_at;
            worker.active_requests = existing->second.active_requests;
            worker.recent_errors = existing->second.recent_errors;
            worker.total_errors = existing->second.total_errors;
            worker.circuit_open_until = existing->second.circuit_open_until;
        } else if (std::find(router_worker_order_.begin(), router_worker_order_.end(), worker.worker_id) == router_worker_order_.end()) {
            router_worker_order_.push_back(worker.worker_id);
        }
        for (const auto &drain : router_drains_) {
            if (drain.label == worker.label) { worker.drain_owner = drain.owner; worker.drain_until = drain.until; break; }
        }
        router_workers_[worker.worker_id] = worker;
        worker = router_workers_[worker.worker_id];
    }

    return {200, "application/json", "{\"ok\":true,\"worker\":" + router_worker_public_json(worker) + "}"};
}

HttpResponse NativeServer::router_heartbeat(const std::string &body) {
    if (!is_valid_top_level_json_object(body)) {
        return invalid_json_object_response(body);
    }
    const auto request_values = top_level_json_object_values(body);
    auto field = [&request_values](const std::string &key, const std::string &fallback = "") -> std::string {
        const auto found = request_values.find(key);
        return found == request_values.end() ? fallback : found->second;
    };
    const auto worker_id = extract_router_worker_id(body);
    if (worker_id.empty()) {
        return {400, "application/json", "{\"detail\":\"Missing worker_id\"}"};
    }
    for (const auto &key : {"queue_depth", "queue_capacity", "in_flight"}) {
        const auto raw = field(key);
        if (!raw.empty() && json_value_would_fail_python_int(raw, false)) {
            return plain_internal_server_error();
        }
    }
    for (const auto &key : {"free_vram_gb", "total_vram_gb", "free_ram_gb", "total_ram_gb"}) {
        const auto raw = field(key);
        if (!raw.empty() && json_value_would_fail_python_float(raw, false)) {
            return plain_internal_server_error();
        }
    }
    bool list_ok = true;
    std::vector<std::string> heartbeat_models;
    std::string heartbeat_models_json;
    const auto models_raw = field("models");
    if (!models_raw.empty()) {
        heartbeat_models = python_list_strings_from_json_value(models_raw, false, &list_ok);
        if (!list_ok) {
            return plain_internal_server_error();
        }
        heartbeat_models_json = python_public_list_json_from_value(models_raw, false, &list_ok);
        if (!list_ok) {
            return plain_internal_server_error();
        }
    }
    std::vector<std::string> heartbeat_capabilities;
    std::string heartbeat_capabilities_json;
    const auto capabilities_raw = field("capabilities");
    if (!capabilities_raw.empty()) {
        heartbeat_capabilities = python_list_strings_from_json_value(capabilities_raw, false, &list_ok);
        if (!list_ok) {
            return plain_internal_server_error();
        }
        heartbeat_capabilities_json = python_public_list_json_from_value(capabilities_raw, false, &list_ok);
        if (!list_ok) {
            return plain_internal_server_error();
        }
    }
    const auto gpus_raw = field("gpus");
    if (!gpus_raw.empty() && trim(gpus_raw).front() == '[' && json_array_contains_non_object(gpus_raw)) {
        return plain_internal_server_error();
    }
    if (json_object_int_values_would_fail_python(field("model_context"), false)) {
        return plain_internal_server_error();
    }
    if (json_slot_map_values_would_fail_python(field("cap_slots"))
        || json_slot_map_values_would_fail_python(field("model_slots"))) {
        return plain_internal_server_error();
    }

    RouterWorker worker;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        auto found = router_workers_.find(worker_id);
        if (found == router_workers_.end()) {
            return {410, "application/json", "{\"ok\":false,\"reason\":\"unknown_worker\",\"should_reregister\":true}"};
        }
        worker = found->second;
        worker.free_vram_gb = coerce_python_float_value(field("free_vram_gb"), worker.free_vram_gb, false);
        worker.total_vram_gb = coerce_python_float_value(field("total_vram_gb"), worker.total_vram_gb, false);
        worker.free_ram_gb = coerce_python_float_value(field("free_ram_gb"), worker.free_ram_gb, false);
        worker.total_ram_gb = coerce_python_float_value(field("total_ram_gb"), worker.total_ram_gb, false);
        worker.queue_depth = coerce_python_int_value(field("queue_depth"), worker.queue_depth, false);
        worker.queue_capacity = coerce_python_int_value(field("queue_capacity"), worker.queue_capacity, false);
        worker.in_flight = coerce_python_int_value(field("in_flight"), worker.in_flight, false);
        const auto cap_slots_raw = field("cap_slots");
        if (!cap_slots_raw.empty() && trim(cap_slots_raw).front() == '{') {
            worker.cap_slots = parse_slot_map_json(cap_slots_raw);
            worker.cap_slot_order = parse_slot_map_order_json(cap_slots_raw);
        }
        const auto model_slots_raw = field("model_slots");
        if (!model_slots_raw.empty() && trim(model_slots_raw).front() == '{') {
            worker.model_slots = parse_slot_map_json(model_slots_raw);
            worker.model_slot_order = parse_slot_map_order_json(model_slots_raw);
        }
        if (!models_raw.empty()) {
            worker.models = heartbeat_models;
            worker.models_json = heartbeat_models_json;
        }
        if (!capabilities_raw.empty()) {
            worker.capabilities = heartbeat_capabilities;
            worker.capabilities_json = heartbeat_capabilities_json;
        }
        const auto heartbeat_gpus_raw = field("gpus");
        if (!heartbeat_gpus_raw.empty() && trim(heartbeat_gpus_raw).front() == '[') {
            worker.gpus_json = heartbeat_gpus_raw;
            worker.best_tier = best_gpu_tier_from_json(worker.gpus_json);
        }
        const auto model_context_raw = field("model_context");
        if (!model_context_raw.empty() && trim(model_context_raw).front() == '{') {
            worker.model_context_json = python_int_map_json_from_value(model_context_raw);
        }
        const auto model_quant_raw = field("model_quant");
        if (!model_quant_raw.empty() && trim(model_quant_raw).front() == '{') {
            worker.model_quant_json = python_string_map_json_from_value(model_quant_raw);
        }
        const auto cap_models_raw = field("cap_models");
        if (!cap_models_raw.empty() && trim(cap_models_raw).front() == '{') {
            worker.cap_models_json = python_string_map_json_from_value(cap_models_raw);
        }
        const auto version_raw = field("version");
        if (!is_python_falsey_json_value(version_raw)) {
            worker.version = python_str_from_json_value(version_raw);
        }
        const auto heartbeat_extra = field("extra");
        if (!heartbeat_extra.empty() && !top_level_json_object_values(heartbeat_extra).empty()) {
            worker.extra_json = merge_json_objects(worker.extra_json, heartbeat_extra);
        }
        worker.last_heartbeat = monotonic_epoch_seconds();
        worker.connection_failures = 0;
        found->second = worker;
    }

    return {200, "application/json", "{\"ok\":true,\"worker\":" + router_worker_public_json(worker) + "}"};
}

bool NativeServer::ensure_router_drains_loaded() {
    std::lock_guard<std::mutex> lock(router_mutex_);
    if (router_drains_loaded_) return true;
    try {
        const std::string path = ezl_getenv_fallback("ROUTER_DRAIN_FILE", "/data/router-drains.json");
        std::vector<RouterDrain> drains;
        if (!path.empty() && std::filesystem::exists(path)) {
            std::ifstream input(path, std::ios::binary);
            if (!input) return false;
            const std::string raw((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (!is_valid_top_level_json_object(raw)) return false;
            for (const auto &entry : json_object_ordered_entries(raw)) {
                if (!is_valid_top_level_json_object(entry.second)) return false;
                const auto owner = json_object_field(entry.second, "owner", "\"\"");
                const auto until = json_object_field(entry.second, "until", "0");
                double expires = 0;
                if (!parse_strict_python_float_string(until, &expires)) return false;
                drains.push_back({entry.first, python_truthy_string_value_or_default(owner, ""), expires});
            }
        }
        router_reservation_ttl_ = std::max(0.0, parse_double_env_explicit("ROUTER_RESERVATION_TTL", 15.0));
        router_drain_path_ = path;
        router_drains_ = std::move(drains);
        router_drains_loaded_ = true;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

void NativeServer::save_router_drains_locked() const {
    if (router_drain_path_.empty()) return;
    const auto target = std::filesystem::absolute(router_drain_path_);
    std::filesystem::create_directories(target.parent_path());
    std::string temporary = (target.parent_path() / ".router-drains-XXXXXX").string();
    int fd = mkstemp(temporary.data());
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "create drain state temporary");
    try {
        std::vector<std::pair<std::string, std::string>> fields;
        for (const auto &drain : router_drains_) {
            fields.emplace_back(drain.label, "{\"owner\":\"" + json_escape(drain.owner) + "\",\"until\":" + json_number_literal(drain.until) + "}");
        }
        const auto data = json_object_from_ordered_entries(fields);
        size_t written = 0;
        while (written < data.size()) {
            const auto count = write(fd, data.data() + written, data.size() - written);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::system_error(errno ? errno : EIO, std::generic_category(), "write drain state");
            written += static_cast<size_t>(count);
        }
        if (fsync(fd) != 0) throw std::system_error(errno, std::generic_category(), "sync drain state");
        if (close(fd) != 0) { fd = -1; throw std::system_error(errno, std::generic_category(), "close drain state"); }
        fd = -1;
        std::filesystem::rename(temporary, target);
    } catch (...) {
        if (fd >= 0) close(fd);
        unlink(temporary.c_str());
        throw;
    }
}

bool NativeServer::router_worker_draining(const RouterWorker &worker) const {
    return !worker.drain_owner.empty() && monotonic_epoch_seconds() < worker.drain_until;
}

int NativeServer::router_maintenance_busy(const RouterWorker &worker) const {
    return std::max({static_cast<int>(worker.active_requests.size()), router_effective_busy(worker), router_total_busy(worker), worker.queue_depth});
}

HttpResponse NativeServer::router_drain(const std::string &body, bool resume) {
    auto error = [](int status, const std::string &detail) -> HttpResponse {
        return {status, "application/json", "{\"detail\":\"" + json_escape(detail) + "\"}"};
    };
    const auto label = python_truthy_string_value_or_default(json_object_field(body, "label", ""), "");
    const auto owner = python_truthy_string_value_or_default(json_object_field(body, "owner", ""), "");
    const auto idle_raw = trim(json_object_field(body, "require_idle", "false"));
    if (resume && (label.empty() || owner.empty())) return error(400, "label and owner are required");
    if (!resume && idle_raw != "true" && idle_raw != "false") return error(400, "require_idle must be a boolean");
    int ttl = 120;
    if (!resume) {
        const auto raw = trim(json_object_field(body, "ttl_seconds", "120"));
        if (raw == "true" || raw == "false") ttl = raw == "true" ? 1 : 0;
        else if (!raw.empty() && raw.front() == '"') {
            const auto value = parse_json_string_at(raw, 0);
            size_t digits = 0;
            if (!parse_python_int_saturated(value, &ttl, &digits)) return error(400, "invalid literal for int() with base 10: " + python_repr_string(value));
            if (digits > 4300) return error(400, "Exceeds the limit (4300 digits) for integer string conversion: value has " + std::to_string(digits) + " digits; use sys.set_int_max_str_digits() to increase the limit");
        } else if (is_json_number_token(raw)) {
            try {
                const auto value = std::stold(raw);
                ttl = value < 0 ? 0 : value > 301 ? 301 : static_cast<int>(value);
            } catch (...) { ttl = 301; }
        } else {
            const auto type = raw == "null" ? "NoneType" : !raw.empty() && raw.front() == '[' ? "list" : "dict";
            return error(400, "int() argument must be a string, a bytes-like object or a real number, not '" + std::string(type) + "'");
        }
    }
    if (!ensure_router_drains_loaded()) return plain_internal_server_error();
    if (!resume && (label.empty() || owner.empty() || ttl < 30 || ttl > 300)) {
        return error(400, "label, owner and a TTL between 30 and 300 seconds are required");
    }
    std::lock_guard<std::mutex> lock(router_mutex_);
    const double now = monotonic_epoch_seconds();
    std::vector<RouterDrain> next;
    if (resume) {
        for (const auto &drain : router_drains_) {
            if (drain.label == label) {
                if (drain.owner != owner) return error(409, "Maintenance lease belongs to another rollout");
            } else next.push_back(drain);
        }
    } else {
        bool renewing = false;
        for (const auto &drain : router_drains_) {
            if (drain.until <= now) continue;
            if (drain.label != label || drain.owner != owner) return error(409, "Another worker rollout holds the maintenance lease");
            renewing = true;
        }
        int live = 0;
        for (const auto &entry : router_workers_) {
            const auto &worker = entry.second;
            if (worker.label == label && now - worker.last_heartbeat <= router_ttl_seconds()) {
                ++live;
            }
        }
        if (!renewing && live != 1) return error(409, "Expected exactly one live worker with this label");
        for (const auto &entry : router_workers_) {
            const auto &worker = entry.second;
            if (worker.label == label && now - worker.last_heartbeat <= router_ttl_seconds() && idle_raw == "true" && router_maintenance_busy(worker)) {
                return error(409, "Worker is still serving or queueing requests");
            }
        }
        next.push_back({label, owner, now + ttl});
    }
    auto previous = router_drains_;
    router_drains_ = std::move(next);
    try {
        if (!resume || previous.size() != router_drains_.size()) save_router_drains_locked();
    } catch (...) {
        router_drains_ = std::move(previous);
        return plain_internal_server_error();
    }
    int busy = 0;
    for (auto &entry : router_workers_) {
        auto &worker = entry.second;
        if (worker.label != label) continue;
        worker.drain_owner = resume ? "" : owner;
        worker.drain_until = resume ? 0.0 : now + ttl;
        if (now - worker.last_heartbeat <= router_ttl_seconds()) busy = std::max(busy, router_maintenance_busy(worker));
    }
    if (resume) return {200, "application/json", "{\"ok\":true}"};
    return {200, "application/json", "{\"ok\":true,\"label\":\"" + json_escape(label) + "\",\"expires_at\":" + json_number_literal(now + ttl) + ",\"busy\":" + std::to_string(busy) + "}"};
}

HttpResponse NativeServer::router_deregister(const std::string &body) {
    if (!is_valid_top_level_json_object(body)) {
        return invalid_json_object_response(body);
    }
    const auto worker_id = extract_router_worker_id(body);
    if (worker_id.empty()) {
        return {400, "application/json", "{\"detail\":\"Missing worker_id\"}"};
    }
    bool removed = false;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        removed = router_workers_.erase(worker_id) > 0;
        if (removed) {
            router_worker_order_.erase(
                std::remove(router_worker_order_.begin(), router_worker_order_.end(), worker_id),
                router_worker_order_.end()
            );
        }
    }
    return {200, "application/json", std::string("{\"ok\":") + (removed ? "true" : "false") + "}"};
}

HttpResponse NativeServer::router_workers() const {
    std::vector<RouterWorker> workers;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        workers = router_workers_in_registration_order_locked();
    }
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < workers.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << router_worker_public_json(workers[i]);
    }
    out << "],\"ttl_seconds\":" << json_number_literal(router_ttl_seconds()) << ",\"maintenance_protocol\":1}";
    return {200, "application/json", out.str()};
}

HttpResponse NativeServer::router_health() const {
    const double now = monotonic_epoch_seconds();
    const double ttl = router_ttl_seconds();
    int alive = 0;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        for (const auto &entry : router_workers_) {
            if (now - entry.second.last_heartbeat <= ttl) {
                alive++;
            }
        }
    }
    std::ostringstream out;
    out << "{\"status\":\"healthy\",\"alive_workers\":" << alive << ",\"ttl_seconds\":" << json_number_literal(ttl) << "}";
    return {200, "application/json", out.str()};
}

HttpResponse NativeServer::router_usage() const {
    return {200, "application/json", usage_snapshot_json()};
}

HttpResponse NativeServer::router_history(int limit) const {
    return {200, "application/json", "{\"history\":" + usage_history_json(limit) + "}"};
}

HttpResponse NativeServer::router_tunnels() const {
    const double now = monotonic_epoch_seconds();
    const double ttl = router_ttl_seconds();
    std::vector<RouterWorker> workers;
    std::map<std::string, RouterTunnelState> tunnel_states;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        for (const auto &worker : router_workers_in_registration_order_locked()) {
            if (is_tunnel_url(worker.url)) {
                workers.push_back(worker);
            }
        }
        for (const auto &entry : router_tunnels_) {
            tunnel_states[entry.first] = entry.second;
        }
    }

    auto disconnect_history_json = [](const std::vector<RouterTunnelDisconnect> &history) {
        std::ostringstream history_out;
        history_out << "[";
        for (size_t i = 0; i < history.size(); ++i) {
            if (i) {
                history_out << ",";
            }
            history_out << "{"
                << "\"at\":" << json_number_literal(history[i].at) << ","
                << "\"duration\":" << json_number_literal(history[i].duration) << ","
                << "\"reason\":\"" << json_escape(history[i].reason) << "\""
                << "}";
        }
        history_out << "]";
        return history_out.str();
    };

    std::ostringstream out;
    std::set<std::string> registered_tunnel_ids;
    out << "{\"workers\":[";
    for (size_t i = 0; i < workers.size(); ++i) {
        if (i) {
            out << ",";
        }
        const auto &worker = workers[i];
        const std::string tunnel_id = tunnel_worker_id(worker.url);
        registered_tunnel_ids.insert(tunnel_id);
        const auto tunnel_it = tunnel_states.find(tunnel_id);
        const bool connected = tunnel_it != tunnel_states.end() && tunnel_it->second.connected;
        const int connect_count = tunnel_it == tunnel_states.end() ? 0 : tunnel_it->second.connect_count;
        out << "{"
            << "\"worker_id\":\"" << json_escape(worker.worker_id) << "\","
            << "\"label\":\"" << json_escape(worker.label) << "\","
            << "\"tunnel_worker_id\":\"" << json_escape(tunnel_id) << "\","
            << "\"connected\":" << (connected ? "true" : "false") << ",";
        if (connected) {
            out << "\"connected_at\":" << json_number_literal(tunnel_it->second.connected_at) << ","
                << "\"connected_for_seconds\":" << json_number_literal(std::max(0.0, now - tunnel_it->second.connected_at)) << ","
                << "\"last_recv_age_seconds\":"
                << json_number_literal(std::round(std::max(0.0, now - tunnel_it->second.last_recv) * 100.0) / 100.0) << ","
                << "\"last_close_reason\":\"" << json_escape(tunnel_it->second.last_close_reason) << "\",";
        } else {
            out << "\"connected_at\":null,"
                << "\"connected_for_seconds\":null,"
                << "\"last_recv_age_seconds\":null,"
                << "\"last_close_reason\":\"\",";
        }
        out << "\"connect_count\":" << connect_count << ","
            << "\"disconnect_history\":"
            << (tunnel_it == tunnel_states.end() ? "[]" : disconnect_history_json(tunnel_it->second.disconnect_history)) << ","
            << "\"alive\":" << ((now - worker.last_heartbeat <= ttl) ? "true" : "false") << ","
            << "\"last_heartbeat_age\":" << std::max(0.0, now - worker.last_heartbeat)
            << "}";
    }
    out << "],\"orphan_tunnels\":[";
    bool first_orphan = true;
    for (const auto &entry : tunnel_states) {
        if (!entry.second.connected || registered_tunnel_ids.find(entry.first) != registered_tunnel_ids.end()) {
            continue;
        }
        if (!first_orphan) {
            out << ",";
        }
        first_orphan = false;
        out << "{\"tunnel_worker_id\":\"" << json_escape(entry.first) << "\","
            << "\"connected_at\":" << json_number_literal(entry.second.connected_at)
            << "}";
    }
    out << "]}";
    return {200, "application/json", out.str()};
}

HttpResponse NativeServer::router_errors() const {
    std::vector<RouterWorker> workers;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        workers = router_workers_in_registration_order_locked();
    }
    std::ostringstream out;
    out << "{\"workers\":[";
    for (size_t i = 0; i < workers.size(); ++i) {
        if (i) {
            out << ",";
        }
        const bool circuit_open = monotonic_epoch_seconds() < workers[i].circuit_open_until;
        out << "{"
            << "\"worker_id\":\"" << json_escape(workers[i].worker_id) << "\","
            << "\"label\":\"" << json_escape(workers[i].label) << "\","
            << "\"total_errors\":" << workers[i].total_errors << ","
            << "\"circuit_open\":" << (circuit_open ? "true" : "false") << ","
            << "\"circuit_open_until\":" << workers[i].circuit_open_until << ","
            << "\"recent_errors\":" << router_recent_errors_json(workers[i])
            << "}";
    }
    out << "],\"errors\":" << router_aggregate_errors_json(workers) << "}";
    return {200, "application/json", out.str()};
}

HttpResponse NativeServer::router_dashboard() const {
    const double now = monotonic_epoch_seconds();
    const double ttl = router_ttl_seconds();
    std::vector<RouterWorker> alive;
    std::vector<RouterWorker> stale;
    {
        std::lock_guard<std::mutex> lock(router_mutex_);
        for (const auto &worker : router_workers_in_registration_order_locked()) {
            if (now - worker.last_heartbeat <= ttl) {
                alive.push_back(worker);
            } else {
                stale.push_back(worker);
            }
        }
    }

    int total_capacity = 0;
    int total_in_flight = 0;
    int total_queue_depth = 0;
    int total_available_slots = 0;
    double total_free_vram = 0.0;
    double total_vram = 0.0;
    struct DashboardWorkerRef {
        std::string label;
        std::string worker_id;
        int best_tier = 0;
        int context = 0;
        int slots_left = 0;
        int queue_capacity = 0;
        std::string quant;
    };
    struct DashboardModelRollup {
        std::string model;
        std::string type = "text";
        int worker_count = 0;
        int total_capacity = 0;
        int available_slots = 0;
        int max_context = 0;
        int best_tier = 0;
        std::vector<DashboardWorkerRef> workers;
        std::set<std::string> quants;
    };
    struct DashboardCapRollup {
        std::string capability;
        int worker_count = 0;
        int total_capacity = 0;
        int available_slots = 0;
        std::vector<std::string> worker_labels;
    };
    std::map<std::string, DashboardModelRollup> model_rollup;
    std::map<std::string, DashboardCapRollup> cap_rollup;

    auto cap_label = [](const std::string &cap) {
        if (cap == "text") return std::string("Text");
        if (cap == "vision") return std::string("Vision");
        if (cap == "text+vision") return std::string("Text + Vision");
        if (cap == "image") return std::string("Image Generation");
        if (cap == "tts") return std::string("Text-to-Speech");
        if (cap == "stt") return std::string("Speech-to-Text");
        if (cap == "embedding") return std::string("Embedding");
        if (cap == "video") return std::string("Video");
        return cap;
    };

    for (const auto &worker : alive) {
        const int capacity = router_total_capacity(worker);
        total_capacity += capacity;
        total_in_flight += router_total_busy(worker);
        total_queue_depth += worker.queue_depth;
        total_available_slots += router_total_slots_left(worker);
        total_free_vram += worker.free_vram_gb;
        total_vram += worker.total_vram_gb;
        for (const auto &model : worker.models) {
            auto &entry = model_rollup[model];
            if (entry.model.empty()) {
                entry.model = model;
            }
            entry.worker_count++;
            const int model_capacity = std::max(1, router_capacity_for(worker, "text", model));
            const int slots_left = router_slots_left(worker, "text", model);
            entry.total_capacity += model_capacity;
            entry.available_slots += slots_left;
            const auto context_values = top_level_json_object_values(worker.model_context_json);
            const auto context_it = context_values.find(model);
            const int context = context_it == context_values.end()
                ? 0
                : extract_json_int_value("{\"x\":" + context_it->second + "}", "x", 0);
            entry.max_context = std::max(entry.max_context, context);
            entry.best_tier = std::max(entry.best_tier, worker.best_tier);
            if (vector_contains(worker.capabilities, "vision") && entry.type == "text") {
                entry.type = "text+vision";
            }
            const auto quant_values = top_level_json_object_values(worker.model_quant_json);
            const auto quant_it = quant_values.find(model);
            std::string quant;
            if (quant_it != quant_values.end()) {
                const std::string raw_quant = trim(quant_it->second);
                quant = raw_quant.size() >= 2 && raw_quant.front() == '"'
                    ? parse_json_string_at(raw_quant, 0)
                    : raw_quant;
            }
            if (!quant.empty()) {
                entry.quants.insert(quant);
            }
            entry.workers.push_back({worker.label, worker.worker_id, worker.best_tier, context, slots_left, model_capacity, quant});
        }
        for (const auto &capability : worker.capabilities) {
            if (capability == "image" || capability == "tts" || capability == "stt" || capability == "video") {
                const std::string model_name = extract_json_string(worker.cap_models_json, capability, cap_label(capability));
                const std::string key = capability + "::" + model_name;
                auto &entry = model_rollup[key];
                if (entry.model.empty()) {
                    entry.model = model_name;
                    entry.type = capability;
                }
                entry.worker_count++;
                const int cap_capacity = std::max(1, router_capacity_for(worker, capability));
                const int slots_left = router_slots_left(worker, capability);
                entry.total_capacity += cap_capacity;
                entry.available_slots += slots_left;
                entry.best_tier = std::max(entry.best_tier, worker.best_tier);
                entry.workers.push_back({worker.label, worker.worker_id, worker.best_tier, 0, slots_left, cap_capacity, ""});
            }
        }
        const bool has_text = vector_contains(worker.capabilities, "text");
        const bool has_vision = vector_contains(worker.capabilities, "vision");
        for (const auto &capability : worker.capabilities) {
            if (capability == "vision" && has_text) {
                continue;
            }
            const std::string merged_cap = (capability == "text" && has_vision) ? "text+vision" : capability;
            const std::string cap_key = merged_cap == "text+vision" ? "text" : merged_cap;
            auto &entry = cap_rollup[merged_cap];
            if (entry.capability.empty()) {
                entry.capability = merged_cap;
            }
            entry.worker_count++;
            entry.total_capacity += std::max(1, router_capacity_for(worker, cap_key));
            entry.available_slots += router_slots_left(worker, cap_key);
            entry.worker_labels.push_back(worker.label);
        }
    }

    const std::string pool_health = alive.empty() ? "offline" : (!stale.empty() ? "degraded" : "healthy");
    std::vector<DashboardCapRollup> ordered_caps;
    for (const auto &entry : cap_rollup) {
        ordered_caps.push_back(entry.second);
    }
    auto cap_display_index = [](const std::string &capability) {
        if (capability == "text+vision") return 0;
        if (capability == "text") return 1;
        if (capability == "embedding") return 2;
        if (capability == "image") return 3;
        if (capability == "tts") return 4;
        if (capability == "stt") return 5;
        if (capability == "video") return 6;
        return 99;
    };
    std::sort(ordered_caps.begin(), ordered_caps.end(), [&cap_display_index](const DashboardCapRollup &lhs, const DashboardCapRollup &rhs) {
        const int left_index = cap_display_index(lhs.capability);
        const int right_index = cap_display_index(rhs.capability);
        if (left_index != right_index) {
            return left_index < right_index;
        }
        return lhs.capability < rhs.capability;
    });

    std::vector<DashboardModelRollup> ordered_models;
    for (const auto &entry : model_rollup) {
        ordered_models.push_back(entry.second);
    }
    auto model_sort_bucket = [](const std::string &type) {
        return (type == "text" || type == "text+vision" || type == "vision" || type == "embedding") ? 0 : 1;
    };
    std::sort(ordered_models.begin(), ordered_models.end(), [&model_sort_bucket](const DashboardModelRollup &lhs, const DashboardModelRollup &rhs) {
        const int left_bucket = model_sort_bucket(lhs.type);
        const int right_bucket = model_sort_bucket(rhs.type);
        if (left_bucket != right_bucket) {
            return left_bucket < right_bucket;
        }
        if (lhs.type != rhs.type) {
            return lhs.type < rhs.type;
        }
        return to_lower(lhs.model) < to_lower(rhs.model);
    });

    std::vector<RouterWorker> alive_for_output = alive;
    std::vector<RouterWorker> stale_for_output = stale;
    auto tier_desc = [](const RouterWorker &lhs, const RouterWorker &rhs) {
        return lhs.best_tier > rhs.best_tier;
    };
    std::stable_sort(alive_for_output.begin(), alive_for_output.end(), tier_desc);
    std::stable_sort(stale_for_output.begin(), stale_for_output.end(), tier_desc);

    std::ostringstream out;
    double wait_timeout = 120.0;
    const char *raw_wait_timeout = getenv("ROUTER_WAIT_TIMEOUT");
    double parsed_wait_timeout = 0.0;
    if (parse_strict_python_float_string(raw_wait_timeout ? raw_wait_timeout : "120", &parsed_wait_timeout)) {
        wait_timeout = parsed_wait_timeout;
    }
    out << "{"
        << "\"generated_at\":" << json_number_literal(now) << ","
        << "\"pool_health\":\"" << pool_health << "\","
        << "\"router\":{\"ttl_seconds\":" << json_number_literal(ttl) << ",\"wait_timeout\":" << json_number_literal(wait_timeout) << "},"
        << "\"totals\":{"
        << "\"alive_workers\":" << alive.size() << ","
        << "\"stale_workers\":" << stale.size() << ","
        << "\"total_parallel_capacity\":" << total_capacity << ","
        << "\"total_in_flight\":" << total_in_flight << ","
        << "\"total_queue_depth\":" << total_queue_depth << ","
        << "\"total_available_slots\":" << total_available_slots << ","
        << "\"total_free_vram_gb\":" << json_number_literal(std::round(total_free_vram * 100.0) / 100.0) << ","
        << "\"total_vram_gb\":" << json_number_literal(std::round(total_vram * 100.0) / 100.0) << ","
        << "\"unique_models\":" << model_rollup.size()
        << "},";

    out << "\"capabilities\":[";
    bool first_cap = true;
    for (const auto &capability : ordered_caps) {
        if (!first_cap) {
            out << ",";
        }
        first_cap = false;
        out << "{"
            << "\"capability\":\"" << json_escape(capability.capability) << "\","
            << "\"worker_count\":" << capability.worker_count << ","
            << "\"total_capacity\":" << capability.total_capacity << ","
            << "\"available_slots\":" << capability.available_slots << ","
            << "\"worker_labels\":" << string_array_json(capability.worker_labels)
            << "}";
    }
    out << "],\"models\":[";
    bool first_model = true;
    for (const auto &model : ordered_models) {
        if (!first_model) {
            out << ",";
        }
        first_model = false;
        out << "{"
            << "\"model\":\"" << json_escape(model.model) << "\","
            << "\"type\":\"" << json_escape(model.type) << "\","
            << "\"worker_count\":" << model.worker_count << ","
            << "\"total_capacity\":" << model.total_capacity << ","
            << "\"available_slots\":" << model.available_slots << ","
            << "\"max_context\":" << model.max_context << ","
            << "\"best_tier\":" << model.best_tier << ","
            << "\"workers\":[";
        for (size_t i = 0; i < model.workers.size(); ++i) {
            if (i) {
                out << ",";
            }
            const auto &worker_ref = model.workers[i];
            out << "{"
                << "\"label\":\"" << json_escape(worker_ref.label) << "\","
                << "\"worker_id\":\"" << json_escape(worker_ref.worker_id) << "\","
                << "\"best_tier\":" << worker_ref.best_tier << ","
                << "\"context\":" << worker_ref.context << ","
                << "\"slots_left\":" << worker_ref.slots_left << ","
                << "\"queue_capacity\":" << worker_ref.queue_capacity << ","
                << "\"quant\":\"" << json_escape(worker_ref.quant) << "\""
                << "}";
        }
        out << "],\"quants\":[";
        bool first_quant = true;
        for (const auto &quant : model.quants) {
            if (!first_quant) {
                out << ",";
            }
            first_quant = false;
            out << "\"" << json_escape(quant) << "\"";
        }
        out << "]}";
    }
    out << "],\"workers\":[";
    for (size_t i = 0; i < alive_for_output.size(); ++i) {
        if (i) {
            out << ",";
        }
        out << router_worker_dashboard_json(alive_for_output[i]);
    }
    for (size_t i = 0; i < stale_for_output.size(); ++i) {
        if (!alive_for_output.empty() || i) {
            out << ",";
        }
        std::string pub = router_worker_dashboard_json(stale_for_output[i]);
        if (!pub.empty() && pub.back() == '}') {
            pub.pop_back();
            pub += ",\"stale\":true}";
        }
        out << pub;
    }
    std::vector<RouterWorker> all_for_errors = alive;
    all_for_errors.insert(all_for_errors.end(), stale.begin(), stale.end());
    out << "],\"usage\":" << usage_snapshot_json()
        << ",\"history\":" << usage_history_json()
        << ",\"errors\":" << router_aggregate_errors_json(all_for_errors) << "}";
    return {200, "application/json", out.str()};
}

std::string dashboard_json_string(const std::string &json, const std::string &key, const std::string &fallback = "") {
    const std::string raw = trim(extract_json_value(json, key, ""));
    if (raw.empty() || raw == "null" || raw.front() == '{' || raw.front() == '[') {
        return fallback;
    }
    if (raw.front() == '"') {
        return parse_json_string_at(raw, 0);
    }
    return raw;
}

std::string dashboard_top_level_value(const std::string &json, const std::string &key, const std::string &fallback) {
    const auto entries = top_level_json_object_values(json);
    const auto found = entries.find(key);
    return found == entries.end() ? fallback : found->second;
}

std::vector<std::string> dashboard_top_level_array_elements(const std::string &json, const std::string &key) {
    const std::string raw = trim(json);
    if (!raw.empty() && raw.front() == '[') {
        return json_array_elements(raw);
    }
    return json_array_elements(dashboard_top_level_value(json, key, "[]"));
}

std::vector<std::string> dashboard_json_string_array(const std::string &raw_array) {
    std::vector<std::string> out;
    for (const auto &element : json_array_elements(raw_array)) {
        const std::string value = trim(element);
        out.push_back(value.size() >= 2 && value.front() == '"' ? parse_json_string_at(value, 0) : value);
    }
    return out;
}

std::string dashboard_html_text(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

std::string dashboard_join(const std::vector<std::string> &values, const std::string &separator) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out << separator;
        }
        out << values[i];
    }
    return out.str();
}

std::string dashboard_commas(long long value) {
    const bool negative = value < 0;
    std::string digits = std::to_string(negative ? -value : value);
    for (int pos = static_cast<int>(digits.size()) - 3; pos > 0; pos -= 3) {
        digits.insert(static_cast<size_t>(pos), ",");
    }
    return negative ? "-" + digits : digits;
}

std::string dashboard_fixed(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string dashboard_tps(double value) {
    if (value <= 0.0) {
        return "—";
    }
    return dashboard_fixed(value, 1);
}

std::string dashboard_time_of_day(double ts, bool utc) {
    if (ts <= 0.0) {
        return "—";
    }
    std::time_t seconds = static_cast<std::time_t>(ts);
    std::tm tm_value{};
    if (utc) {
        gmtime_r(&seconds, &tm_value);
    } else {
        localtime_r(&seconds, &tm_value);
    }
    char buffer[16];
    if (std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &tm_value) == 0) {
        return "—";
    }
    return buffer;
}

std::string dashboard_cap_label(const std::string &cap) {
    if (cap == "text") return "Text";
    if (cap == "vision") return "Vision";
    if (cap == "text+vision") return "Text + Vision";
    if (cap == "image") return "Image Generation";
    if (cap == "tts") return "Text-to-Speech";
    if (cap == "stt") return "Speech-to-Text";
    if (cap == "embedding") return "Embedding";
    if (cap == "video") return "Video";
    std::string label = cap;
    std::replace(label.begin(), label.end(), '-', ' ');
    bool new_word = true;
    for (char &ch : label) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            new_word = true;
        } else if (new_word) {
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            new_word = false;
        }
    }
    return label;
}

std::string dashboard_cap_class(const std::string &cap) {
    if (cap == "text") return "cap-text";
    if (cap == "vision" || cap == "text+vision") return "cap-vision";
    if (cap == "image") return "cap-image";
    if (cap == "tts") return "cap-tts";
    if (cap == "stt") return "cap-stt";
    if (cap == "video") return "cap-video";
    if (cap == "embedding") return "cap-embedding";
    return "";
}

std::string dashboard_cap_pill(const std::string &cap) {
    return "<span class=\"pill " + dashboard_cap_class(cap) + "\">" + dashboard_cap_label(cap) + "</span>";
}

std::string dashboard_usage_bar(double used_pct, const std::string &width = "100%") {
    const std::string fill_cls = used_pct >= 85.0 ? "crit" : (used_pct >= 60.0 ? "warn" : "good");
    std::ostringstream out;
    out << "<div class=\"usage-bar\" style=\"width:" << width << "\">"
        << "<div class=\"usage-fill " << fill_cls << "\" style=\"width:" << dashboard_fixed(used_pct, 0) << "%\"></div>"
        << "</div>";
    return out.str();
}

std::string dashboard_tier_badge(int tier) {
    std::string cls = "tier-muted";
    std::string star;
    if (tier >= 80) {
        cls = "tier-gold";
        star = "⭐ ";
    } else if (tier >= 50) {
        cls = "tier-blue";
    } else if (tier >= 20) {
        cls = "tier-green";
    } else if (tier >= 5) {
        cls = "tier-warn";
    }
    return "<span class=\"" + cls + "\">" + star + "tier " + std::to_string(tier) + "</span>";
}

std::string dashboard_version_link(const std::string &version) {
    if (version.empty()) {
        return "—";
    }
    return "<a href=\"https://github.com/DevXT-LLC/ezlocalai/commit/" + version
        + "\" target=\"_blank\" rel=\"noopener\">" + version + "</a>";
}

std::string dashboard_cap_cards(const std::string &data) {
    std::ostringstream out;
    for (const auto &capability : dashboard_top_level_array_elements(data, "capabilities")) {
        const std::string cap = dashboard_json_string(capability, "capability");
        const int worker_count = extract_json_int_value(capability, "worker_count", 0);
        const int total_capacity = extract_json_int_value(capability, "total_capacity", 0);
        const int available_slots = extract_json_int_value(capability, "available_slots", 0);
        std::ostringstream workers;
        for (const auto &label : dashboard_json_string_array(extract_json_value(capability, "worker_labels", "[]"))) {
            const std::string text = label;
            workers << "<span class=\"pill cap-worker\" title=\"" << text << "\">" << text << "</span>";
        }
        const double used_pct = 100.0 * (1.0 - static_cast<double>(available_slots) / std::max(total_capacity, 1));
        out << R"(
        <div class="card cap">
          <div class="cap-header">)" << dashboard_cap_pill(cap) << R"(</div>
          <div class="cap-stats">
            <span><b>)" << worker_count << "</b> worker" << (worker_count != 1 ? "s" : "") << R"(</span>
            <span><b>)" << available_slots << "</b>/" << total_capacity << R"( slots free</span>
          </div>
          )" << dashboard_usage_bar(used_pct, "100%")
            << "\n          "
            << (workers.str().empty() ? "" : "<div class='cap-workers'>" + workers.str() + "</div>")
            << R"(
        </div>
        )";
    }
    return out.str().empty() ? "<div class=\"muted\">No capabilities advertised yet.</div>" : out.str();
}

std::string dashboard_model_rows(const std::string &data) {
    std::ostringstream out;
    for (const auto &model : dashboard_top_level_array_elements(data, "models")) {
        const std::string type = dashboard_json_string(model, "type", "text");
        const int max_context = extract_json_int_value(model, "max_context", 0);
        const std::string ctx = max_context ? dashboard_commas(max_context) : "—";
        std::vector<std::string> quant_values;
        for (const auto &quant : dashboard_json_string_array(extract_json_value(model, "quants", "[]"))) {
            quant_values.push_back(quant);
        }
        const std::string quants = quant_values.empty() ? "—" : dashboard_join(quant_values, ", ");
        std::ostringstream worker_pills;
        for (const auto &worker : json_array_elements(extract_json_value(model, "workers", "[]"))) {
            const std::string label = dashboard_json_string(worker, "label");
            const int best_tier = extract_json_int_value(worker, "best_tier", 0);
            const int slots_left = extract_json_int_value(worker, "slots_left", 0);
            const int queue_capacity = extract_json_int_value(worker, "queue_capacity", 0);
            const int context = extract_json_int_value(worker, "context", 0);
            const std::string quant = dashboard_json_string(worker, "quant");
            worker_pills << "<span class=\"pill\" title=\"tier " << best_tier
                << " · " << slots_left << "/" << queue_capacity << " slots free · ctx "
                << (context ? std::to_string(context) : "?") << " · quant "
                << (quant.empty() ? "?" : quant) << "\">" << label << "</span>";
        }
        const bool text_type = type == "text" || type == "text+vision" || type == "vision" || type == "embedding";
        out << R"(
        <tr>
          <td class="mono">)" << dashboard_cap_pill(type) << " " << dashboard_json_string(model, "model") << R"(</td>
          <td class="num">)" << extract_json_int_value(model, "worker_count", 0) << R"(</td>
          <td class="num">)" << extract_json_int_value(model, "total_capacity", 0) << R"(</td>
          <td class="num">)" << extract_json_int_value(model, "available_slots", 0) << R"(</td>
          <td class="num">)" << (text_type ? ctx : "—") << R"(</td>
          <td class="mono small">)" << (text_type ? quants : "—") << R"(</td>
          <td class="num">)" << extract_json_int_value(model, "best_tier", 0) << R"(</td>
          <td>)" << worker_pills.str() << R"(</td>
        </tr>
        )";
    }
    return out.str().empty() ? "<tr><td colspan=\"8\" class=\"muted\">No models loaded across the pool.</td></tr>" : out.str();
}

std::string dashboard_worker_rows(const std::string &data) {
    std::ostringstream out;
    for (const auto &worker : dashboard_top_level_array_elements(data, "workers")) {
        const bool stale = extract_json_bool(worker, "stale", false);
        const auto raw_gpus = json_array_elements(extract_json_value(worker, "gpus", "[]"));
        std::vector<std::string> display_gpus;
        std::string cpu_entry;
        for (const auto &gpu : raw_gpus) {
            if (extract_json_int_value(gpu, "index", 0) >= 0) {
                display_gpus.push_back(gpu);
            } else if (dashboard_json_string(gpu, "backend") == "cpu" && cpu_entry.empty()) {
                cpu_entry = gpu;
            }
        }
        if (display_gpus.empty() && !cpu_entry.empty()) {
            display_gpus.push_back(cpu_entry);
        }
        std::vector<std::string> gpu_labels;
        for (const auto &gpu : display_gpus) {
            const std::string name = dashboard_json_string(gpu, "name", "?");
            const double total_vram = extract_json_double_value(gpu, "total_vram_gb", 0.0);
            const std::string backend = dashboard_json_string(gpu, "backend");
            std::ostringstream label;
            label << name;
            if (total_vram > 0.0) {
                label << " (" << dashboard_fixed(total_vram, 0) << "GB VRAM)";
            } else if (backend == "cpu") {
                const double ram = extract_json_double_value(worker, "total_ram_gb", 0.0) > 0.0
                    ? extract_json_double_value(worker, "total_ram_gb", 0.0)
                    : extract_json_double_value(worker, "free_ram_gb", 0.0);
                if (ram > 0.0) {
                    label << " (" << dashboard_fixed(ram, 0) << "GB RAM)";
                }
            }
            if (!backend.empty() && backend != "cuda") {
                label << " [" << backend << "]";
            }
            gpu_labels.push_back(label.str());
        }
        const std::string gpus = gpu_labels.empty() ? "CPU" : dashboard_join(gpu_labels, ", ");

        const double free_vram = extract_json_double_value(worker, "free_vram_gb", 0.0);
        const double total_vram = extract_json_double_value(worker, "total_vram_gb", 0.0);
        const double free_ram = extract_json_double_value(worker, "free_ram_gb", 0.0);
        const double total_ram = extract_json_double_value(worker, "total_ram_gb", 0.0);
        std::string mem_cell;
        if (total_vram > 0.0) {
            const double used_pct = std::max(0.0, std::min(100.0, (1.0 - free_vram / total_vram) * 100.0));
            mem_cell = dashboard_fixed(free_vram, 1) + "/" + dashboard_fixed(total_vram, 0) + " GB VRAM" + dashboard_usage_bar(used_pct, "80px");
        } else if (total_ram > 0.0) {
            const double used_pct = std::max(0.0, std::min(100.0, (1.0 - free_ram / total_ram) * 100.0));
            mem_cell = dashboard_fixed(free_ram, 1) + "/" + dashboard_fixed(total_ram, 0) + " GB RAM" + dashboard_usage_bar(used_pct, "80px");
        } else {
            mem_cell = free_ram > 0.0 ? dashboard_fixed(free_ram, 1) + " GB RAM free" : "—";
        }

        const auto model_quant = top_level_json_object_values(extract_json_value(worker, "model_quant", "{}"));
        const auto model_context = top_level_json_object_values(extract_json_value(worker, "model_context", "{}"));
        std::vector<std::string> model_lines;
        for (const auto &model_name : dashboard_json_string_array(extract_json_value(worker, "models", "[]"))) {
            const int ctx = extract_json_int_value("{" + std::string("\"x\":") + (model_context.count(model_name) ? model_context.at(model_name) : "0") + "}", "x", 0);
            std::string quant;
            const auto quant_it = model_quant.find(model_name);
            if (quant_it != model_quant.end()) {
                quant = dashboard_json_string("{\"x\":" + quant_it->second + "}", "x");
            }
            std::string line = model_name;
            if (ctx) {
                line += " @ " + dashboard_commas(ctx);
            }
            if (!quant.empty()) {
                line += " (" + quant + ")";
            }
            model_lines.push_back(line);
        }
        const std::string models = model_lines.empty() ? "—" : dashboard_join(model_lines, "<br>");

        int slots_left = extract_json_int_value(worker, "slot_total_available", 0);
        int slots_total = extract_json_int_value(worker, "slot_total_capacity", 0);
        if (slots_total <= 0) {
            slots_total = extract_json_int_value(worker, "queue_capacity", 1);
            slots_left = std::max(0, slots_total - extract_json_int_value(worker, "queue_depth", 0));
        }
        const double slot_pct = slots_total == 0 ? 0.0 : (1.0 - static_cast<double>(slots_left) / slots_total) * 100.0;
        std::ostringstream slot_bar;
        slot_bar << "<div class=\"bar\"><div class=\"bar-fill\" style=\"width:" << dashboard_fixed(slot_pct, 0) << "%\"></div></div>";

        const std::string tunnel_raw = trim(extract_json_value(worker, "tunnel_connected", "null"));
        const bool tunnel_connected_false = tunnel_raw == "false";
        std::string status;
        if (stale) {
            status = "🔴 stale";
        } else if (tunnel_connected_false) {
            status = "<span style=\"color:#f85149\">🔌 tunnel-off</span>";
        } else if (slots_left > 0) {
            status = "🟢 ready";
        } else {
            status = "🟡 full";
        }

        const bool circuit_open = extract_json_bool(worker, "circuit_open", false);
        const int total_errors = extract_json_int_value(worker, "total_errors", 0);
        std::string err_cell;
        if (circuit_open) {
            err_cell = "<span style=\"color:#f85149\">🔴 OPEN</span><div class=\"muted small\">" + std::to_string(total_errors) + " total</div>";
        } else if (total_errors) {
            err_cell = "<span style=\"color:#d29922\">" + std::to_string(total_errors) + "</span><div class=\"muted small\">recent ok</div>";
        } else {
            err_cell = "<span class=\"muted\">0</span>";
        }

        const auto raw_caps = dashboard_json_string_array(extract_json_value(worker, "capabilities", "[]"));
        const bool has_text = std::find(raw_caps.begin(), raw_caps.end(), "text") != raw_caps.end();
        const bool has_vision = std::find(raw_caps.begin(), raw_caps.end(), "vision") != raw_caps.end();
        const std::string cap_slots = extract_json_value(worker, "cap_slots", "{}");
        std::vector<std::string> cap_pills;
        for (const auto &cap : raw_caps) {
            if (cap == "vision" && has_text) {
                continue;
            }
            const std::string display_cap = (cap == "text" && has_vision) ? "text+vision" : cap;
            const std::string lookup_cap = display_cap == "text+vision" ? "text" : display_cap;
            const std::string slot = extract_json_value(cap_slots, lookup_cap, "{}");
            const int capacity = extract_json_int_value(slot, "capacity", 0);
            const int available = extract_json_int_value(slot, "available", 0);
            std::string pill = dashboard_cap_pill(display_cap);
            if (capacity) {
                const std::string title = std::to_string(available) + "/" + std::to_string(capacity) + " slots free";
                const size_t pos = pill.find("<span ");
                if (pos != std::string::npos) {
                    pill.replace(pos, 6, "<span title=\"" + title + "\" ");
                }
            }
            cap_pills.push_back(pill);
        }
        const std::string caps = cap_pills.empty() ? "—" : dashboard_join(cap_pills, " ");
        const bool tunnel = extract_json_bool(worker, "tunnel", false);
        const std::string worker_url = dashboard_json_string(worker, "url");
        const std::string url_cell = tunnel
            ? worker_url + "<div class=\"muted small\">🔌 " + (tunnel_connected_false ? "disconnected" : "connected") + "</div>"
            : worker_url;

        out << "\n        <tr class=\"" << (stale ? "stale" : "") << "\">\n"
            << "          <td><b>" << dashboard_json_string(worker, "label") << "</b><div class=\"muted small\">"
            << dashboard_json_string(worker, "worker_id") << "</div></td>\n"
            << "          <td>" << status << "</td>\n"
            << "          <td class=\"mono small\">" << url_cell << "</td>\n"
            << "          <td>" << gpus << "<div class=\"muted small\">" << dashboard_tier_badge(extract_json_int_value(worker, "best_tier", 0)) << "</div></td>\n"
            << "          <td class=\"num small\">" << mem_cell << "</td>\n"
            << "          <td>" << slots_left << "/" << slots_total << " " << slot_bar.str() << "<div class=\"muted small\">"
            << extract_json_int_value(worker, "slot_total_busy", extract_json_int_value(worker, "in_flight", 0)) << " active/queued</div></td>\n"
            << "          <td class=\"small\">" << caps << "</td>\n"
            << "          <td class=\"small mono\">" << models << "</td>\n"
            << "          <td class=\"num small\">" << dashboard_fixed(extract_json_double_value(worker, "last_heartbeat_age", 0.0), 0) << "s</td>\n"
            << "          <td class=\"mono small\">" << dashboard_version_link(dashboard_json_string(worker, "version")) << "</td>\n"
            << "          <td class=\"small\">" << err_cell << "</td>\n"
            << "        </tr>\n        ";
    }
    return out.str().empty() ? "<tr><td colspan=\"11\" class=\"muted\">No workers registered.</td></tr>" : out.str();
}

struct DashboardSpeedStats {
    double prompt_tps_sum = 0.0;
    int prompt_tps_n = 0;
    double predicted_tps_sum = 0.0;
    int predicted_tps_n = 0;
};

std::string dashboard_usage_summary_rows(const std::string &usage_json) {
    std::ostringstream out;
    auto entries = json_object_ordered_entries(usage_json);
    std::sort(entries.begin(), entries.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &entry : entries) {
        const std::string &label = entry.first;
        const std::string &wdata = entry.second;
        int llm_reqs = 0;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        for (const auto &model : json_object_ordered_entries(extract_json_value(wdata, "llm", "{}"))) {
            llm_reqs += extract_json_int_value(model.second, "requests", 0);
            prompt_tokens += extract_json_int_value(model.second, "prompt_tokens", 0);
            completion_tokens += extract_json_int_value(model.second, "completion_tokens", 0);
        }
        const int embedding_reqs = extract_json_int_value(extract_json_value(wdata, "embedding", "{}"), "requests", 0);
        const int tts_reqs = extract_json_int_value(extract_json_value(wdata, "tts", "{}"), "requests", 0);
        const int stt_reqs = extract_json_int_value(extract_json_value(wdata, "stt", "{}"), "requests", 0);
        const int image_reqs = extract_json_int_value(extract_json_value(wdata, "image", "{}"), "requests", 0);
        const int video_reqs = extract_json_int_value(extract_json_value(wdata, "video", "{}"), "requests", 0);
        const int total_tokens = prompt_tokens + completion_tokens;
        out << R"(
        <tr>
          <td><b>)" << label << R"(</b></td>
          <td class="num">)" << dashboard_commas(llm_reqs) << R"(</td>
          <td class="num">)" << dashboard_commas(prompt_tokens) << R"(</td>
          <td class="num">)" << dashboard_commas(completion_tokens) << R"(</td>
          <td class="num )" << (total_tokens > 0 ? "tok-hi" : "") << R"(">)" << dashboard_commas(total_tokens) << R"(</td>
          <td class="num">)" << dashboard_commas(embedding_reqs) << R"(</td>
          <td class="num">)" << dashboard_commas(tts_reqs) << R"(</td>
          <td class="num">)" << dashboard_commas(stt_reqs) << R"(</td>
          <td class="num">)" << dashboard_commas(image_reqs) << R"(</td>
          <td class="num">)" << dashboard_commas(video_reqs) << R"(</td>
        </tr>)";
    }
    return out.str().empty() ? "<tr><td colspan=\"10\" class=\"muted\">No usage recorded yet.</td></tr>" : out.str();
}

std::string dashboard_usage_model_rows(const std::string &usage_json, const std::string &history_json) {
    std::map<std::string, DashboardSpeedStats> speed_by_worker_model;
    for (const auto &history : json_array_elements(history_json)) {
        const std::string key = dashboard_json_string(history, "worker", "unknown") + "\n" + normalize_usage_model_name(dashboard_json_string(history, "model"));
        auto &stats = speed_by_worker_model[key];
        const double prompt_tps = extract_json_double_value(history, "prompt_tps", 0.0);
        const double predicted_tps = extract_json_double_value(history, "predicted_tps", 0.0);
        if (prompt_tps > 0.0) {
            stats.prompt_tps_sum += prompt_tps;
            stats.prompt_tps_n++;
        }
        if (predicted_tps > 0.0) {
            stats.predicted_tps_sum += predicted_tps;
            stats.predicted_tps_n++;
        }
    }

    struct ModelUsage {
        std::string model;
        int requests = 0;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        double prompt_tps_sum = 0.0;
        int prompt_tps_n = 0;
        double predicted_tps_sum = 0.0;
        int predicted_tps_n = 0;
    };
    std::ostringstream out;
    auto workers = json_object_ordered_entries(usage_json);
    std::sort(workers.begin(), workers.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &worker : workers) {
        const auto llm_entries = json_object_ordered_entries(extract_json_value(worker.second, "llm", "{}"));
        if (llm_entries.empty()) {
            continue;
        }
        std::vector<ModelUsage> merged;
        for (const auto &model : llm_entries) {
            const std::string canon = normalize_usage_model_name(model.first);
            auto it = std::find_if(merged.begin(), merged.end(), [&canon](const ModelUsage &item) { return item.model == canon; });
            if (it == merged.end()) {
                merged.push_back({canon});
                it = std::prev(merged.end());
            }
            it->requests += extract_json_int_value(model.second, "requests", 0);
            it->prompt_tokens += extract_json_int_value(model.second, "prompt_tokens", 0);
            it->completion_tokens += extract_json_int_value(model.second, "completion_tokens", 0);
            it->prompt_tps_sum += extract_json_double_value(model.second, "prompt_tps_sum", 0.0);
            it->prompt_tps_n += extract_json_int_value(model.second, "prompt_tps_n", 0);
            it->predicted_tps_sum += extract_json_double_value(model.second, "predicted_tps_sum", 0.0);
            it->predicted_tps_n += extract_json_int_value(model.second, "predicted_tps_n", 0);
        }
        std::stable_sort(merged.begin(), merged.end(), [](const ModelUsage &lhs, const ModelUsage &rhs) { return lhs.requests > rhs.requests; });
        for (const auto &model : merged) {
            double avg_prompt = model.prompt_tps_n ? model.prompt_tps_sum / model.prompt_tps_n : 0.0;
            double avg_predicted = model.predicted_tps_n ? model.predicted_tps_sum / model.predicted_tps_n : 0.0;
            if (!avg_prompt || !avg_predicted) {
                const auto speed_it = speed_by_worker_model.find(worker.first + "\n" + model.model);
                if (speed_it != speed_by_worker_model.end()) {
                    const auto &speed = speed_it->second;
                    if (!avg_prompt && speed.prompt_tps_n) {
                        avg_prompt = speed.prompt_tps_sum / speed.prompt_tps_n;
                    }
                    if (!avg_predicted && speed.predicted_tps_n) {
                        avg_predicted = speed.predicted_tps_sum / speed.predicted_tps_n;
                    }
                }
            }
          out << R"(
        <tr>
          <td class="muted small">)" << worker.first << R"(</td>
          <td class="mono small">)" << model.model << R"(</td>
          <td class="num small">)" << dashboard_commas(model.requests) << R"(</td>
          <td class="num small">)" << dashboard_commas(model.prompt_tokens) << R"(</td>
          <td class="num small">)" << dashboard_commas(model.completion_tokens) << R"(</td>
          <td class="num small">)" << dashboard_commas(model.prompt_tokens + model.completion_tokens) << R"(</td>
          <td class="num small">)" << dashboard_tps(avg_prompt) << R"(</td>
          <td class="num small">)" << dashboard_tps(avg_predicted) << R"(</td>
        </tr>)";
        }
    }
    return out.str().empty() ? "<tr><td colspan=\"8\" class=\"muted\">No LLM usage recorded yet.</td></tr>" : out.str();
}

std::string dashboard_history_rows(const std::string &history_json) {
    const auto history = json_array_elements(history_json);
    std::ostringstream out;
    size_t rendered = 0;
    for (auto it = history.rbegin(); it != history.rend() && rendered < 50; ++it, ++rendered) {
        const double total_ms = extract_json_double_value(*it, "total_ms", 0.0);
        std::string total_cell = "<span class=\"muted\">—</span>";
        if (total_ms > 0.0) {
            total_cell = total_ms < 60000.0
                ? dashboard_fixed(total_ms / 1000.0, 2) + "s"
                : dashboard_fixed(total_ms / 60000.0, 2) + "m";
        }
        out << R"(
        <tr>
          <td class="muted small">)" << dashboard_time_of_day(extract_json_double_value(*it, "ts", 0.0), false) << R"(</td>
          <td class="small"><b>)" << dashboard_json_string(*it, "worker", "—") << R"(</b></td>
          <td class="mono small">)" << dashboard_json_string(*it, "model", "—") << R"(</td>
          <td class="num small">)" << dashboard_commas(extract_json_int_value(*it, "prompt_tokens", 0)) << R"(</td>
          <td class="num small">)" << dashboard_commas(extract_json_int_value(*it, "completion_tokens", 0)) << R"(</td>
          <td class="num small">)" << dashboard_tps(extract_json_double_value(*it, "prompt_tps", 0.0)) << R"(</td>
          <td class="num small">)" << dashboard_tps(extract_json_double_value(*it, "predicted_tps", 0.0)) << R"(</td>
          <td class="num small">)" << total_cell << R"(</td>
        </tr>)";
    }
    return out.str().empty() ? "<tr><td colspan=\"8\" class=\"muted\">No requests yet.</td></tr>" : out.str();
}

std::string dashboard_error_rows(const std::string &data) {
    std::ostringstream out;
    size_t rendered = 0;
    for (const auto &error : dashboard_top_level_array_elements(data, "errors")) {
        if (rendered++ >= 50) {
            break;
        }
        std::string kind = dashboard_json_string(error, "kind");
        std::replace(kind.begin(), kind.end(), '_', ' ');
        const std::string status_raw = trim(extract_json_value(error, "status", ""));
        const bool show_status = !(status_raw.empty() || status_raw == "null" || status_raw == "0");
        const std::string status_cell = show_status ? "<span class=\"mono small\">" + dashboard_json_string(error, "status") + "</span>" : "";
        std::string msg = dashboard_json_string(error, "message");
        replace_all(&msg, "<", "&lt;");
        replace_all(&msg, ">", "&gt;");
        out << R"(
        <tr>
          <td class="muted small">)" << dashboard_time_of_day(extract_json_double_value(error, "ts", 0.0), true) << R"(</td>
          <td class="small"><b>)" << dashboard_json_string(error, "label", "—") << R"(</b></td>
          <td class="small">)" << kind << R"(</td>
          <td class="small">)" << status_cell << R"(</td>
          <td class="mono small">)" << dashboard_json_string(error, "path", "—") << R"(</td>
          <td class="mono small">)" << msg << R"(</td>
        </tr>)";
    }
    return out.str().empty() ? "<tr><td colspan=\"6\" class=\"muted\">No errors recorded.</td></tr>" : out.str();
}

HttpResponse NativeServer::router_dashboard_html() const {
    const auto dashboard = router_dashboard();
    const std::string data = dashboard.body;
    const int alive_workers = extract_json_int_value(data, "alive_workers", 0);
    const int stale_workers = extract_json_int_value(data, "stale_workers", 0);
    const int total_capacity = extract_json_int_value(data, "total_parallel_capacity", 0);
    const int total_available = extract_json_int_value(data, "total_available_slots", 0);
    const int in_flight = extract_json_int_value(data, "total_in_flight", 0);
    const int queued = extract_json_int_value(data, "total_queue_depth", 0);
    const int unique_models = extract_json_int_value(data, "unique_models", 0);
    const double total_free_vram = extract_json_double_value(data, "total_free_vram_gb", 0.0);
    const double total_vram = extract_json_double_value(data, "total_vram_gb", 0.0);
    const double ttl = extract_json_double_value(data, "ttl_seconds", router_ttl_seconds());
    const double wait_timeout = extract_json_double_value(data, "wait_timeout", parse_double_env("ROUTER_WAIT_TIMEOUT", 120.0));
    const std::string pool_health = extract_json_string(data, "pool_health", alive_workers > 0 ? "healthy" : "offline");
    std::string health_class = "health-ok";
    std::string health_label = "● Healthy";
    if (pool_health == "degraded") {
        health_class = "health-warn";
        health_label = "◐ Degraded";
    } else if (pool_health == "offline") {
        health_class = "health-crit";
        health_label = "○ Offline";
    }
    const std::string usage_json = dashboard_top_level_value(data, "usage", "{}");
    const std::string history_json = dashboard_top_level_value(data, "history", "[]");

    std::ostringstream html;
    html << R"(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ezlocalai Router</title>
<style>
  :root {
    --bg: #0f1419; --fg: #e6edf3; --muted: #8b949e; --card: #161b22;
    --border: #30363d; --accent: #58a6ff; --warn: #f0883e; --ok: #3fb950;
    --crit: #f85149;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--fg); font: 14px/1.45 -apple-system,
         BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0;
         padding: 24px; }
  h1 { margin: 0 0 4px; font-size: 22px; }
  h2 { margin: 32px 0 12px; font-size: 16px; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.05em; }
  .muted { color: var(--muted); }
  .small { font-size: 12px; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  code { background: rgba(110,118,129,0.2); padding: 1px 6px; border-radius: 4px; }
  .banner { padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; }
  .banner.warn { background: rgba(240,136,62,0.15); border: 1px solid var(--warn); }
  /* Layout */
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 12px; }
  .card { background: var(--card); border: 1px solid var(--border);
           border-radius: 8px; padding: 14px 16px; }
  /* Stat cards */
  .stat .label { color: var(--muted); font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.05em; }
  .stat .value { font-size: 28px; font-weight: 700; margin-top: 4px;
                  background: linear-gradient(135deg, var(--fg), var(--accent));
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                  background-clip: text; }
  .stat .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  /* Capability cards */
  .cap .cap-header { margin-bottom: 8px; }
  .cap .cap-stats { display: flex; gap: 14px; margin-top: 6px;
                     color: var(--muted); font-size: 12px; }
  /* Tables */
  table { width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: rgba(110,118,129,0.1); font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--muted); }
  tr:last-child td { border-bottom: none; }
  tr.stale { opacity: 0.5; }
  tr:not(.stale):hover { background: rgba(88,166,255,0.04); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  /* Capability pills */
  .pill { display: inline-block; padding: 2px 9px; margin: 2px;
           border-radius: 999px; font-size: 11px; border: 1px solid; }
  .cap-text    { background: rgba(88,166,255,0.15);  border-color: rgba(88,166,255,0.45); }
  .cap-vision  { background: rgba(160,110,255,0.15); border-color: rgba(160,110,255,0.45); }
  .cap-image   { background: rgba(240,136,62,0.15);  border-color: rgba(240,136,62,0.45); }
  .cap-tts     { background: rgba(63,185,80,0.15);   border-color: rgba(63,185,80,0.45); }
  .cap-stt     { background: rgba(50,200,180,0.15);  border-color: rgba(50,200,180,0.45); }
  .cap-video   { background: rgba(245,80,80,0.15);   border-color: rgba(245,80,80,0.45); }
  .cap-embedding { background: rgba(130,130,130,0.15); border-color: rgba(130,130,130,0.45); }
  /* Tier badges */
  .tier-gold  { color: #ffd700; font-weight: 600; }
  .tier-blue  { color: #58a6ff; }
  .tier-green { color: #3fb950; }
  .tier-warn  { color: #f0883e; }
  .tier-muted { color: #8b949e; }
  /* Usage / slot bars */
  .bar { display: inline-block; width: 80px; height: 6px; margin-left: 6px;
          background: rgba(110,118,129,0.25); border-radius: 3px;
          vertical-align: middle; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--ok), var(--warn)); }
  .usage-bar { width: 100%; height: 4px; background: rgba(110,118,129,0.2);
                border-radius: 2px; margin-top: 8px; overflow: hidden;
                display: block; }
  .usage-fill { height: 100%; border-radius: 2px; transition: width 0.3s; }
  .usage-fill.good { background: var(--ok); }
  .usage-fill.warn { background: var(--warn); }
  .usage-fill.crit { background: var(--crit); }
  /* Capability card worker labels */
  .cap-workers { margin-top: 8px; }
  .cap-worker { background: rgba(110,118,129,0.12) !important; border-color: rgba(110,118,129,0.3) !important;
                 font-size: 10px !important; }
  /* Header */
  .header { display: flex; justify-content: space-between; align-items: baseline;
             flex-wrap: wrap; gap: 12px; }
  .header .meta { color: var(--muted); font-size: 12px; }
  /* Pool health badge */
  .health-ok   { font-size: 14px; font-weight: 600; color: var(--ok); }
  .health-warn { font-size: 14px; font-weight: 600; color: var(--warn); }
  .health-crit { font-size: 14px; font-weight: 600; color: var(--crit); }
  /* Usage table */
  .tok-hi { color: var(--accent); font-weight: 600; }
</style>
</head>
<body>
  <div id="dash">
  <div class="header">
    <div>
      <h1>ezlocalai router</h1>
      <div class="meta">)" << alive_workers << " worker" << (alive_workers == 1 ? "" : "s")
        << R"( online
      · TTL )" << dashboard_fixed(ttl, 0)
        << "s · wait timeout " << dashboard_fixed(wait_timeout, 0) << R"(s
      · auto-refresh 5s</div>
    </div>
    <span class=")" << health_class << "\">" << health_label << R"(</span>
  </div>

    <div class="grid">
    <div class="card stat"><div class="label">Workers</div>
      <div class="value">)" << alive_workers << R"(</div>
      <div class="sub">)" << stale_workers << R"( stale</div></div>
    <div class="card stat"><div class="label">Parallel capacity</div>
      <div class="value">)" << total_capacity << R"(</div>
      <div class="sub">)" << total_available << R"( slots free now</div></div>
    <div class="card stat"><div class="label">In flight</div>
      <div class="value">)" << in_flight << R"(</div>
      <div class="sub">)" << queued << R"( queued</div></div>
    <div class="card stat">
      <div class="label">Free VRAM</div>
      <div class="value">)" << dashboard_fixed(total_free_vram, 0) << R"( GB</div>
      )" << dashboard_usage_bar(100.0 * (1.0 - total_free_vram / std::max(total_vram, 0.1))) << R"(
      <div class="sub">of )" << dashboard_fixed(total_vram, 0) << R"( GB total</div></div>
    <div class="card stat"><div class="label">Unique models</div>
      <div class="value">)" << unique_models << R"(</div>
      <div class="sub">across the pool</div></div>
  </div>

  <h2>Capabilities</h2>
  <div class="grid">)" << dashboard_cap_cards(data) << R"(</div>

  <h2>Models</h2>
  <table>
    <thead><tr>
      <th>Model</th><th class="num">Workers</th><th class="num">Total parallel</th>
      <th class="num">Available now</th><th class="num">Max context</th>
      <th>Quant(s)</th><th class="num">Best tier</th><th>Served by</th>
    </tr></thead>
    <tbody>)" << dashboard_model_rows(data) << R"(</tbody>
  </table>

  <h2>Workers</h2>
  <table>
    <thead><tr>
      <th>Label</th><th>Status</th><th>URL</th><th>GPUs</th>
      <th>Free VRAM</th><th>Slots free</th><th>Capabilities</th>
      <th>Models</th><th class="num">Last hb</th><th>Version</th><th>Errors</th>
    </tr></thead>
    <tbody>)" << dashboard_worker_rows(data) << R"(</tbody>
  </table>

  <h2>Usage history</h2>
  <table>
    <thead><tr>
      <th>Worker</th>
      <th class="num">LLM reqs</th><th class="num">Prompt tok</th>
      <th class="num">Completion tok</th><th class="num">Total tokens</th>
      <th class="num">Embedding reqs</th>
      <th class="num">TTS reqs</th><th class="num">STT reqs</th>
      <th class="num">Image reqs</th><th class="num">Video reqs</th>
    </tr></thead>
    <tbody>)" << dashboard_usage_summary_rows(usage_json) << R"(</tbody>
  </table>

  <h2>LLM model breakdown</h2>
  <table>
    <thead><tr>
      <th>Worker</th><th>Model</th>
      <th class="num">Requests</th><th class="num">Prompt tokens</th>
      <th class="num">Completion tokens</th><th class="num">Total tokens</th>
      <th class="num">Avg prompt t/s</th><th class="num">Avg output t/s</th>
    </tr></thead>
    <tbody>)" << dashboard_usage_model_rows(usage_json, history_json) << R"(</tbody>
  </table>

  <h2>Recent requests</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Worker</th><th>Model</th>
      <th class="num">Prompt tok</th><th class="num">Output tok</th>
      <th class="num">Prompt t/s</th><th class="num">Output t/s</th>
      <th class="num">Total time</th>
    </tr></thead>
    <tbody>)" << dashboard_history_rows(history_json) << R"(</tbody>
  </table>

  <h2>Recent errors</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Worker</th><th>Kind</th><th>Status</th><th>Path</th><th>Message</th>
    </tr></thead>
    <tbody>)" << dashboard_error_rows(data) << R"(</tbody>
  </table>
  </div>
<script>
(function() {
  // In-place dashboard refresh: fetch the same page, parse it, and swap the
  // contents of #dash. Keeps scroll position and avoids the white flash you
  // get from <meta refresh>.
  var INTERVAL_MS = 5000;
  var inFlight = false;
  function tick() {
    if (inFlight || document.hidden) return;
    inFlight = true;
    fetch(window.location.pathname + '?_=' + Date.now(), {
      cache: 'no-store',
      credentials: 'same-origin',
      headers: { 'X-Dashboard-Refresh': '1' }
    })
      .then(function(r) { return r.ok ? r.text() : null; })
      .then(function(html) {
        if (!html) return;
        var doc = new DOMParser().parseFromString(html, 'text/html');
        var fresh = doc.getElementById('dash');
        var current = document.getElementById('dash');
        if (fresh && current) current.innerHTML = fresh.innerHTML;
      })
      .catch(function() {})
      .finally(function() { inFlight = false; });
  }
  setInterval(tick, INTERVAL_MS);
})();
</script>
</body>
</html>)";
    return {200, "text/html", html.str()};
}

bool NativeServer::has_alive_router_worker(const std::string &capability, const std::string &model) const {
    RouterWorker worker;
    return find_alive_router_worker(capability, model, &worker, {}, false);
}

bool NativeServer::find_alive_router_worker(
    const std::string &capability,
    const std::string &model,
    RouterWorker *out,
    const std::vector<std::string> &exclude_worker_ids,
    bool allow_cross_model
) const {
    const double now = monotonic_epoch_seconds();
    const double ttl = router_ttl_seconds();
    std::vector<RouterWorker> all_alive;
    auto model_matches = [](const RouterWorker &worker, const std::string &requested_model) {
        if (requested_model.empty() || requested_model == "unknown") {
            return true;
        }
        if (worker.models.empty()) {
            return true;
        }
        if (vector_contains(worker.models, requested_model)) {
            return true;
        }
        const auto slash = requested_model.find_last_of('/');
        const std::string base = to_lower(slash == std::string::npos ? requested_model : requested_model.substr(slash + 1));
        for (const auto &available_model : worker.models) {
            if (!base.empty() && to_lower(available_model).find(base) != std::string::npos) {
                return true;
            }
        }
        return false;
    };
    std::lock_guard<std::mutex> lock(router_mutex_);
    for (const auto &worker : router_workers_in_registration_order_locked()) {
        if (now - worker.last_heartbeat > ttl) {
            continue;
        }
        if (vector_contains(exclude_worker_ids, worker.worker_id)) {
            continue;
        }
        if (is_tunnel_url(worker.url)) {
            const auto tunnel = router_tunnels_.find(tunnel_worker_id(worker.url));
            if (tunnel == router_tunnels_.end() || !tunnel->second.connected) {
                continue;
            }
        }
        if (now < worker.circuit_open_until || router_worker_draining(worker)) {
            continue;
        }
        if (!capability.empty() && !vector_contains(worker.capabilities, capability)) {
            continue;
        }
        all_alive.push_back(worker);
    }

    auto has_capacity = [this, &capability, &model](const RouterWorker &worker, bool use_model) {
        return router_slots_left(worker, capability, use_model ? model : "") > 0;
    };
    auto rank = [this, &capability, &model](const std::vector<RouterWorker> &candidates, bool use_model) -> RouterWorker {
        std::vector<RouterWorker> ranked = candidates;
        std::stable_sort(ranked.begin(), ranked.end(), [this, &capability, &model, use_model](const RouterWorker &lhs, const RouterWorker &rhs) {
            return router_score(lhs, capability, use_model ? model : "") > router_score(rhs, capability, use_model ? model : "");
        });

        int fastest_tier = 0;
        for (const auto &worker : candidates) {
            fastest_tier = std::max(fastest_tier, worker.best_tier);
        }
        int fastest_busy = std::numeric_limits<int>::max();
        for (const auto &worker : candidates) {
            if (worker.best_tier == fastest_tier) {
                fastest_busy = std::min(fastest_busy, router_effective_busy(worker, capability, use_model ? model : ""));
            }
        }
        int window = 20;
        int parsed_window = 0;
        window = parse_strict_int_string(ezl_getenv_default("ROUTER_IDLE_TIER_WINDOW"), &parsed_window)
            ? std::max(0, parsed_window)
            : 20;
        if (fastest_busy > 0 && window > 0) {
            std::vector<RouterWorker> idle_near;
            for (const auto &worker : candidates) {
                if (router_effective_busy(worker, capability, use_model ? model : "") == 0
                    && worker.best_tier >= fastest_tier - window) {
                    idle_near.push_back(worker);
                }
            }
            if (!idle_near.empty()) {
                std::stable_sort(idle_near.begin(), idle_near.end(), [this, &capability, &model, use_model](const RouterWorker &lhs, const RouterWorker &rhs) {
                    if (lhs.best_tier != rhs.best_tier) {
                        return lhs.best_tier > rhs.best_tier;
                    }
                    return router_score(lhs, capability, use_model ? model : "") > router_score(rhs, capability, use_model ? model : "");
                });
                return idle_near.front();
            }
        }
        return ranked.front();
    };

    std::vector<RouterWorker> candidates;
    if (!is_model_strict_capability(capability)) {
        for (const auto &worker : all_alive) {
            if (has_capacity(worker, false)) {
                candidates.push_back(worker);
            }
        }
        if (candidates.empty()) {
            return false;
        }
        if (out) {
            *out = rank(candidates, false);
        }
        return true;
    }

    if (!model.empty() && model != "unknown") {
        std::vector<RouterWorker> model_servers;
        for (const auto &worker : all_alive) {
            if (model_matches(worker, model)) {
                model_servers.push_back(worker);
                if (has_capacity(worker, true)) {
                    candidates.push_back(worker);
                }
            }
        }
        if (!candidates.empty()) {
            if (out) {
                *out = rank(candidates, true);
            }
            return true;
        }
        if (!allow_cross_model) {
            return false;
        }
        for (const auto &worker : all_alive) {
            if (has_capacity(worker, false)) {
                candidates.push_back(worker);
            }
        }
        if (!candidates.empty()) {
            if (out) {
                *out = rank(candidates, false);
            }
            return true;
        }
        return false;
    }

    for (const auto &worker : all_alive) {
        if (has_capacity(worker, true)) {
            candidates.push_back(worker);
        }
    }
    if (candidates.empty()) {
        return false;
    }
    if (out) {
        *out = rank(candidates, true);
    }
    return true;
}

bool NativeServer::wait_for_alive_router_worker(
    const std::string &capability,
    const std::string &model,
    RouterWorker *out,
    const std::vector<std::string> &exclude_worker_ids
) const {
    auto parse_seconds = [](const char *name, const char *fallback) {
        double parsed = 0.0;
        const char *raw = getenv(name);
        if (parse_strict_python_float_string(raw ? raw : fallback, &parsed)) {
            return std::max(0.0, parsed);
        }
        return std::max(0.0, std::strtod(fallback, nullptr));
    };

    const double wait_timeout = parse_seconds("ROUTER_WAIT_TIMEOUT", "120");
    const double cross_model_grace = parse_seconds("ROUTER_CROSS_MODEL_GRACE", "8");
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::duration<double>(wait_timeout);
    const auto grace_deadline = start + std::chrono::duration<double>(cross_model_grace);

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        const bool allow_cross_model = now >= grace_deadline;
        if (find_alive_router_worker(capability, model, out, exclude_worker_ids, allow_cross_model)) {
            return true;
        }
        if (now >= deadline) {
            return false;
        }
        const auto remaining = deadline - now;
        const auto sleep_for = std::min<std::chrono::duration<double>>(remaining, std::chrono::duration<double>(0.5));
        std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(sleep_for));
    }
}

HttpResponse NativeServer::router_proxy_request(
    const std::string &capability,
    const std::string &model,
    const std::string &path,
    const std::string &body,
    const std::string &content_type,
    const std::string &request_base_url,
    int timeout_seconds,
    bool allow_deferred_streaming
) {
    const bool retry_llm = path == "/v1/chat/completions" || path == "/v1/completions";
    const std::string stream_value = retry_llm ? extract_json_value(body, "stream", "") : "";
    const bool llm_stream = retry_llm && !is_python_falsey_json_value(stream_value);
    int max_attempts = 1;
    if (retry_llm && !llm_stream) {
        int retries = 0;
        max_attempts += parse_strict_int_string(ezl_getenv_default("ROUTER_MAX_RETRIES"), &retries)
            ? std::max(0, retries)
            : 2;
    }
    std::vector<std::string> tried;
    HttpResponse response;
    RouterWorker worker;
    bool have_response = false;
    std::string pick_error;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        worker = RouterWorker{};
        if (!wait_for_alive_router_worker(capability, model, &worker, tried)) {
            std::ostringstream detail;
            detail << "No worker available for capability='" << capability << "'";
            if (!model.empty()) {
                detail << " model='" << model << "'";
            }
            if (retry_llm && !tried.empty()) {
                pick_error = "HTTPException: 503: " + detail.str();
                continue;
            }
            return {503, "application/json", "{\"detail\":\"" + json_escape(detail.str()) + "\"}"};
        }

        const std::string api_key = worker.api_key.empty() ? config_.api_key : worker.api_key;
        std::string reservation_token, admission_error;
        if (!router_reserve_request(worker.worker_id, capability, model, retry_llm, &reservation_token, &admission_error)) {
            if (!admission_error.empty()) return {503, "application/json", "{\"detail\":\"" + json_escape(admission_error) + "\"}"};
            --attempt;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        auto admission = std::shared_ptr<int>(new int(0), [this, id = worker.worker_id, reservation_token](int *value) {
            delete value;
            router_release_request(id, reservation_token);
        });
        const bool deferred_stream =
            allow_deferred_streaming
            && !is_tunnel_url(worker.url)
            && (llm_stream || path == "/v1/audio/speech/stream");
        if (deferred_stream) {
            response = http_worker_stream_request(
                "POST",
                worker.url,
                path,
                api_key,
                content_type.empty() ? "application/json" : content_type,
                body,
                timeout_seconds
            );
            have_response = true;
            if (!response.transport_error) {
                if (llm_stream) {
                    response.status = 200;
                    response.content_type = "text/event-stream";
                    response.chunked = true;
                }
                if (path == "/v1/audio/speech/stream") {
                    response.content_type = "text/event-stream";
                    response.chunked = true;
                }
                auto upstream_writer = response.stream_writer;
                auto stream_capture = response.stream_capture;
                const auto stream_started = std::chrono::steady_clock::now();
                const std::string worker_id = worker.worker_id;
                const std::string usage_capability = path == "/v1/embeddings" ? "embedding" : capability;
                const int usage_status = response.status;
                const std::string usage_content_type = response.content_type;
                response.stream_writer = [this,
                                          upstream_writer,
                                          worker,
                                          worker_id,
                                          reservation_token,
                                          admission,
                                          capability,
                                          model,
                                          usage_capability,
                                          usage_status,
                                          usage_content_type,
                                          stream_capture,
                                          stream_started](int client_fd) mutable {
                    auto active_request = std::move(admission);
                    if (upstream_writer) {
                        upstream_writer(client_fd);
                    } else {
                        send_all_fd(client_fd, "0\r\n\r\n");
                    }
                    router_release_request(worker_id, reservation_token);
                    active_request.reset();
                    std::string usage_body = stream_capture ? *stream_capture : "";
                    if (usage_body.find("\"total_ms\"") == std::string::npos) {
                        const auto elapsed = std::chrono::steady_clock::now() - stream_started;
                        const double total_ms = std::chrono::duration<double, std::milli>(elapsed).count();
                        usage_body += "\n{\"total_ms\":" + json_number_literal(total_ms) + "}";
                    }
                    HttpResponse usage_response{usage_status, usage_content_type, usage_body};
                    record_router_usage(worker, usage_capability, model, usage_response);
                };
                return response;
            }
        } else if (is_tunnel_url(worker.url)) {
            const int timeout_ms = timeout_seconds > 0
                ? static_cast<int>(std::min<long long>(
                    static_cast<long long>(timeout_seconds) * 1000LL,
                    std::numeric_limits<int>::max()
                ))
                : 0;
            response = router_tunnel_request(
                worker,
                "POST",
                path,
                api_key,
                content_type.empty() ? "application/json" : content_type,
                body,
                llm_stream || path == "/v1/audio/speech/stream",
                timeout_ms,
                allow_deferred_streaming
            );
        } else {
            response = http_worker_request("POST", worker.url, path, api_key, content_type.empty() ? "application/json" : content_type, body, timeout_seconds);
        }
        have_response = true;
        if (response.stream_writer && !response.transport_error) {
            if (llm_stream) {
                response.status = 200;
                response.content_type = "text/event-stream";
                response.chunked = true;
            }
            if (path == "/v1/audio/speech/stream") {
                if (response.content_type.empty()) {
                    response.content_type = "text/event-stream";
                }
                response.chunked = true;
            }
            auto upstream_writer = response.stream_writer;
            auto stream_capture = response.stream_capture;
            const auto stream_started = std::chrono::steady_clock::now();
            const std::string worker_id = worker.worker_id;
            const std::string usage_capability = path == "/v1/embeddings" ? "embedding" : capability;
            const int usage_status = response.status;
            const std::string usage_content_type = response.content_type;
            response.stream_writer = [this,
                                      upstream_writer,
                                      worker,
                                      worker_id,
                                      reservation_token,
                                      admission,
                                      capability,
                                      model,
                                      usage_capability,
                                      usage_status,
                                      usage_content_type,
                                      stream_capture,
                                      stream_started](int client_fd) mutable {
                auto active_request = std::move(admission);
                if (upstream_writer) {
                    upstream_writer(client_fd);
                } else {
                    send_all_fd(client_fd, "0\r\n\r\n");
                }
                router_release_request(worker_id, reservation_token);
                active_request.reset();
                std::string usage_body = stream_capture ? *stream_capture : "";
                if (usage_body.find("\"total_ms\"") == std::string::npos) {
                    const auto elapsed = std::chrono::steady_clock::now() - stream_started;
                    const double total_ms = std::chrono::duration<double, std::milli>(elapsed).count();
                    usage_body += "\n{\"total_ms\":" + json_number_literal(total_ms) + "}";
                }
                HttpResponse usage_response{usage_status, usage_content_type, usage_body};
                record_router_usage(worker, usage_capability, model, usage_response);
            };
            return response;
        }
        admission.reset();
        if (response.transport_error) {
            std::string message = extract_json_string(response.body, "detail", "");
            if (message.empty()) {
                message = response.body.empty() ? "Worker request failed" : response.body;
            }
            if (is_tunnel_url(worker.url)) {
                const std::string lower_message = to_lower(message);
                const bool not_connected = lower_message.find("not connected") != std::string::npos;
                if (not_connected) {
                    const std::string tunnel_id = tunnel_worker_id(worker.url);
                    const std::string record_message = tunnel_id.empty()
                        ? message
                        : "Tunnel " + tunnel_id + " not connected";
                    router_record_error(
                        worker.worker_id,
                        "tunnel_disconnect",
                        path,
                        record_message,
                        response.status
                    );
                }
                pick_error = std::string(not_connected ? "HTTPException: " : "RuntimeError: ")
                    + (not_connected ? std::to_string(response.status) + ": " : "")
                    + message;
            } else {
                {
                    std::lock_guard<std::mutex> lock(router_mutex_);
                    auto found = router_workers_.find(worker.worker_id);
                    if (found != router_workers_.end()) {
                        found->second.connection_failures += 1;
                    }
                }
                router_record_error(
                    worker.worker_id,
                    "connection",
                    path,
                    message,
                    0
                );
            }
            if (retry_llm) {
                if (!is_tunnel_url(worker.url)) {
                    pick_error = message;
                }
                if (attempt + 1 < max_attempts) {
                    tried.push_back(worker.worker_id);
                    continue;
                }
            }
            if (!retry_llm) {
                if (is_tunnel_url(worker.url) && to_lower(message).find("not connected") != std::string::npos) {
                    return response;
                }
                return plain_internal_server_error();
            }
            continue;
        }
        const bool transient = !llm_stream && (response.status == 408 || (response.status >= 500 && response.status <= 599));
        if (transient && retry_llm && attempt + 1 < max_attempts) {
            router_record_error(
                worker.worker_id,
                "http_5xx",
                path,
                response.body,
                response.status
            );
            tried.push_back(worker.worker_id);
            continue;
        }
        break;
    }

    if (!pick_error.empty()) {
        std::ostringstream detail;
        detail << "All " << max_attempts << " worker attempts failed for "
            << path << ": " << pick_error;
        return {502, "application/json", "{\"detail\":\"" + json_escape(detail.str()) + "\"}"};
    }

    if (!have_response) {
        return {503, "application/json", "{\"detail\":\"No worker available\"}"};
    }

    const std::string usage_capability = path == "/v1/embeddings" ? "embedding" : capability;
    record_router_usage(worker, usage_capability, model, response);
    if (llm_stream) {
        response.status = 200;
        response.content_type = "text/event-stream";
        response.chunked = true;
    }
    if (path == "/v1/audio/speech/stream") {
        response.content_type = "text/event-stream";
        response.chunked = true;
    }
    if (path == "/v1/audio/speech"
        || path == "/v1/audio/music"
        || path == "/v1/videos/music"
        || path == "/v1/images/generations"
        || path == "/v1/images/edits"
        || path == "/v1/videos/generations") {
        return persist_response_assets(worker, response, request_base_url);
    }
    return response;
}

std::string NativeServer::usage_snapshot_json() const {
    std::lock_guard<std::mutex> lock(usage_mutex_);
    return usage_snapshot_json_locked();
}

std::string NativeServer::usage_history_json(int limit) const {
    std::lock_guard<std::mutex> lock(usage_mutex_);
    return usage_history_json_locked(limit);
}

void NativeServer::ensure_usage_loaded_locked() const {
    if (usage_loaded_) {
        return;
    }
    usage_loaded_ = true;

    auto append_unique = [](std::vector<std::string> *values, const std::string &value) {
        if (std::find(values->begin(), values->end(), value) == values->end()) {
            values->push_back(value);
        }
    };
    const auto usage_entries = json_object_ordered_entries(read_file_or_empty(usage_file_path()));
    for (const auto &worker_entry : usage_entries) {
        append_unique(&usage_worker_order_, worker_entry.first);
        std::map<std::string, std::string> worker_values;
        for (const auto &section : json_object_ordered_entries(worker_entry.second)) {
            worker_values[section.first] = section.second;
            append_unique(&usage_section_order_[worker_entry.first], section.first);
        }
        usage_loaded_worker_values_[worker_entry.first] = worker_values;

        auto llm_it = worker_values.find("llm");
        if (llm_it != worker_values.end()) {
            for (const auto &model_entry : json_object_ordered_entries(llm_it->second)) {
                UsageLlmStats stats;
                stats.requests = extract_json_int_value(model_entry.second, "requests", 0);
                stats.prompt_tokens = extract_json_int_value(model_entry.second, "prompt_tokens", 0);
                stats.completion_tokens = extract_json_int_value(model_entry.second, "completion_tokens", 0);
                stats.prompt_tps_sum = extract_json_double_value(model_entry.second, "prompt_tps_sum", 0.0);
                stats.prompt_tps_n = extract_json_int_value(model_entry.second, "prompt_tps_n", 0);
                stats.predicted_tps_sum = extract_json_double_value(model_entry.second, "predicted_tps_sum", 0.0);
                stats.predicted_tps_n = extract_json_int_value(model_entry.second, "predicted_tps_n", 0);
                usage_llm_[worker_entry.first][model_entry.first] = stats;
                append_unique(&usage_llm_model_order_[worker_entry.first], model_entry.first);
            }
        }

        for (const auto *capability : {"tts", "stt", "image", "video", "embedding", "music", "music_video"}) {
            auto cap_it = worker_values.find(capability);
            if (cap_it != worker_values.end()) {
                usage_caps_[worker_entry.first][capability] = extract_json_int_value(cap_it->second, "requests", 0);
            }
        }
    }

    const int max_history = usage_history_max();
    auto history_elements = json_array_elements(read_file_or_empty(usage_history_file_path()));
    if (static_cast<int>(history_elements.size()) > max_history) {
        history_elements.erase(history_elements.begin(), history_elements.end() - max_history);
    }
    for (const auto &element : history_elements) {
        UsageHistoryEntry entry;
        entry.ts = extract_json_double_value(element, "ts", 0.0);
        entry.worker = extract_json_string(element, "worker", "");
        entry.model = extract_json_string(element, "model", "unknown");
        entry.prompt_tokens = extract_json_int_value(element, "prompt_tokens", 0);
        entry.completion_tokens = extract_json_int_value(element, "completion_tokens", 0);
        entry.prompt_ms = extract_json_double_value(element, "prompt_ms", 0.0);
        entry.predicted_ms = extract_json_double_value(element, "predicted_ms", 0.0);
        entry.total_ms = extract_json_double_value(element, "total_ms", 0.0);
        entry.prompt_tps = extract_json_double_value(element, "prompt_tps", 0.0);
        entry.predicted_tps = extract_json_double_value(element, "predicted_tps", 0.0);
        entry.raw_json = element;
        usage_history_.push_back(entry);
    }
}

std::string NativeServer::usage_snapshot_json_locked() const {
    ensure_usage_loaded_locked();

    auto append_unique = [](std::vector<std::string> *values, const std::string &value) {
        if (std::find(values->begin(), values->end(), value) == values->end()) {
            values->push_back(value);
        }
    };
    auto llm_json = [](const std::map<std::string, UsageLlmStats> &models, const std::vector<std::string> &order) {
        std::ostringstream out;
        out << "{";
        bool first_model = true;
        std::set<std::string> emitted;
        auto emit_model = [&out, &first_model, &emitted, &models](const std::string &model) {
            const auto found = models.find(model);
            if (found == models.end() || emitted.find(model) != emitted.end()) {
                return;
            }
            if (!first_model) {
                out << ",";
            }
            first_model = false;
            emitted.insert(model);
            const auto &stats = found->second;
            out << "\"" << json_escape(found->first) << "\":{"
                << "\"requests\":" << stats.requests << ","
                << "\"prompt_tokens\":" << stats.prompt_tokens << ","
                << "\"completion_tokens\":" << stats.completion_tokens << ","
                << "\"prompt_tps_sum\":" << json_number_literal(stats.prompt_tps_sum) << ","
                << "\"prompt_tps_n\":" << stats.prompt_tps_n << ","
                << "\"predicted_tps_sum\":" << json_number_literal(stats.predicted_tps_sum) << ","
                << "\"predicted_tps_n\":" << stats.predicted_tps_n
                << "}";
        };
        for (const auto &model : order) {
            emit_model(model);
        }
        for (const auto &model_entry : models) {
            emit_model(model_entry.first);
        }
        out << "}";
        return out.str();
    };

    std::ostringstream out;
    out << "{";
    bool first_worker = true;
    std::vector<std::string> workers = usage_worker_order_;
    for (const auto &entry : usage_loaded_worker_values_) {
        append_unique(&workers, entry.first);
    }
    for (const auto &entry : usage_llm_) {
        append_unique(&workers, entry.first);
    }
    for (const auto &entry : usage_caps_) {
        append_unique(&workers, entry.first);
    }
    for (const auto &label : workers) {
        if (!first_worker) {
            out << ",";
        }
        first_worker = false;
        std::map<std::string, std::string> sections;
        std::vector<std::string> section_order;
        auto order_it = usage_section_order_.find(label);
        if (order_it != usage_section_order_.end()) {
            section_order = order_it->second;
        }
        auto loaded_it = usage_loaded_worker_values_.find(label);
        if (loaded_it != usage_loaded_worker_values_.end()) {
            sections = loaded_it->second;
        }
        auto llm_it = usage_llm_.find(label);
        if (llm_it != usage_llm_.end()) {
            std::vector<std::string> model_order;
            auto model_order_it = usage_llm_model_order_.find(label);
            if (model_order_it != usage_llm_model_order_.end()) {
                model_order = model_order_it->second;
            }
            sections["llm"] = llm_json(llm_it->second, model_order);
            append_unique(&section_order, "llm");
        }
        auto caps_it = usage_caps_.find(label);
        if (caps_it != usage_caps_.end()) {
            for (const auto &cap_entry : caps_it->second) {
                sections[cap_entry.first] = "{\"requests\":" + std::to_string(cap_entry.second) + "}";
                append_unique(&section_order, cap_entry.first);
            }
        }
        std::vector<std::pair<std::string, std::string>> ordered_sections;
        std::set<std::string> emitted_sections;
        for (const auto &section : section_order) {
            const auto found = sections.find(section);
            if (found == sections.end() || emitted_sections.find(section) != emitted_sections.end()) {
                continue;
            }
            emitted_sections.insert(section);
            ordered_sections.push_back(*found);
        }
        for (const auto &section : sections) {
            if (emitted_sections.find(section.first) == emitted_sections.end()) {
                ordered_sections.push_back(section);
            }
        }
        out << "\"" << json_escape(label) << "\":" << json_object_from_ordered_entries(ordered_sections);
    }
    out << "}";
    return out.str();
}

std::string NativeServer::usage_history_json_locked(int limit) const {
    ensure_usage_loaded_locked();
    size_t start = 0;
    if (limit > 0 && static_cast<size_t>(limit) < usage_history_.size()) {
        start = usage_history_.size() - static_cast<size_t>(limit);
    } else if (limit < 0) {
        start = std::min(usage_history_.size(), static_cast<size_t>(-limit));
    }
    std::ostringstream out;
    out << "[";
    for (size_t i = start; i < usage_history_.size(); ++i) {
        if (i != start) {
            out << ",";
        }
        const auto &entry = usage_history_[i];
        if (!entry.raw_json.empty()) {
            out << entry.raw_json;
            continue;
        }
        out << "{"
            << "\"ts\":" << json_number_literal(entry.ts) << ","
            << "\"worker\":\"" << json_escape(entry.worker) << "\","
            << "\"model\":\"" << json_escape(entry.model) << "\","
            << "\"prompt_tokens\":" << entry.prompt_tokens << ","
            << "\"completion_tokens\":" << entry.completion_tokens << ","
            << "\"prompt_ms\":" << json_number_literal(entry.prompt_ms) << ","
            << "\"predicted_ms\":" << json_number_literal(entry.predicted_ms) << ","
            << "\"total_ms\":" << json_number_literal(entry.total_ms) << ","
            << "\"prompt_tps\":" << json_number_literal(entry.prompt_tps) << ","
            << "\"predicted_tps\":" << json_number_literal(entry.predicted_tps)
            << "}";
    }
    out << "]";
    return out.str();
}

void NativeServer::record_router_usage(const RouterWorker &worker, const std::string &capability, const std::string &model, const HttpResponse &response) {
    if (response.transport_error) {
        return;
    }
    if ((capability == "music" || capability == "music_video") && response.status >= 400) return;
    if (capability == "text" || capability == "vision") {
        if (response.status < 200 || response.status >= 300) {
            return;
        }
        int prompt_tokens = extract_json_int_value(response.body, "prompt_tokens", 0);
        int completion_tokens = extract_json_int_value(response.body, "completion_tokens", 0);
        if (prompt_tokens == 0) {
            prompt_tokens = extract_json_int_value(response.body, "prompt_n", 0);
        }
        if (completion_tokens == 0) {
            completion_tokens = extract_json_int_value(response.body, "predicted_n", 0);
        }
        if (prompt_tokens == 0 && completion_tokens == 0) {
            return;
        }
        const double prompt_ms = extract_json_double_value(response.body, "prompt_ms", 0.0);
        const double predicted_ms = extract_json_double_value(response.body, "predicted_ms", 0.0);
        const double total_ms = extract_json_double_value(response.body, "total_ms", 0.0);
        const double prompt_tps = prompt_ms > 0.0 ? prompt_tokens / (prompt_ms / 1000.0) : 0.0;
        const double predicted_tps = predicted_ms > 0.0 ? completion_tokens / (predicted_ms / 1000.0) : 0.0;
        std::lock_guard<std::mutex> lock(usage_mutex_);
        ensure_usage_loaded_locked();
        const std::string normalized_model = normalize_usage_model_name(model);
        if (std::find(usage_worker_order_.begin(), usage_worker_order_.end(), worker.label) == usage_worker_order_.end()) {
            usage_worker_order_.push_back(worker.label);
        }
        if (std::find(usage_section_order_[worker.label].begin(), usage_section_order_[worker.label].end(), "llm") == usage_section_order_[worker.label].end()) {
            usage_section_order_[worker.label].push_back("llm");
        }
        if (std::find(usage_llm_model_order_[worker.label].begin(), usage_llm_model_order_[worker.label].end(), normalized_model) == usage_llm_model_order_[worker.label].end()) {
            usage_llm_model_order_[worker.label].push_back(normalized_model);
        }
        auto &stats = usage_llm_[worker.label][normalized_model];
        stats.requests++;
        stats.prompt_tokens += prompt_tokens;
        stats.completion_tokens += completion_tokens;
        if (prompt_tps > 0.0) {
            stats.prompt_tps_sum += prompt_tps;
            stats.prompt_tps_n++;
        }
        if (predicted_tps > 0.0) {
            stats.predicted_tps_sum += predicted_tps;
            stats.predicted_tps_n++;
        }
        usage_history_.push_back({
            monotonic_epoch_seconds(),
            worker.label,
            normalized_model,
            prompt_tokens,
            completion_tokens,
            prompt_ms,
            predicted_ms,
            total_ms,
            prompt_tps,
            predicted_tps,
            "",
        });
        const int max_history = usage_history_max();
        if (static_cast<int>(usage_history_.size()) > max_history) {
            usage_history_.erase(usage_history_.begin(), usage_history_.begin() + (usage_history_.size() - static_cast<size_t>(max_history)));
        }
        mark_usage_dirty_locked();
        return;
    }
    if (capability == "tts" || capability == "stt" || capability == "image" || capability == "video" || capability == "embedding" || capability == "music" || capability == "music_video") {
        std::lock_guard<std::mutex> lock(usage_mutex_);
        ensure_usage_loaded_locked();
        if (std::find(usage_worker_order_.begin(), usage_worker_order_.end(), worker.label) == usage_worker_order_.end()) {
            usage_worker_order_.push_back(worker.label);
        }
        if (std::find(usage_section_order_[worker.label].begin(), usage_section_order_[worker.label].end(), capability) == usage_section_order_[worker.label].end()) {
            usage_section_order_[worker.label].push_back(capability);
        }
        usage_caps_[worker.label][capability]++;
        mark_usage_dirty_locked();
    }
}

HttpResponse NativeServer::persist_response_assets(const RouterWorker &worker, const HttpResponse &response, const std::string &request_base_url) {
    if (env_false("ROUTER_PERSIST_ASSETS") || response.status < 200 || response.status >= 300) {
        return response;
    }
    if (response.body.empty() || response.body.find("/outputs/") == std::string::npos || !is_valid_utf8(response.body)) {
        return response;
    }
    auto urls = extract_asset_urls(response.body);
    if (urls.empty()) {
        return response;
    }

    std::string rewritten = response.body;
    std::string public_base = strip_trailing_slashes(request_base_url);
    if (public_base.empty()) {
        public_base = strip_trailing_slashes(ezl_getenv_default("ROUTER_PUBLIC_URL"));
    }
    const std::string api_key = worker.api_key.empty() ? config_.api_key : worker.api_key;
    bool changed = false;
    for (const auto &url : urls) {
        const auto idx = to_lower(url).find("/outputs/");
        if (idx == std::string::npos) {
            continue;
        }
        const std::string worker_path = url.substr(idx);
        std::string relative;
        if (!safe_output_relative_path(worker_path, &relative)) {
            continue;
        }
        const std::string local_target = outputs_dir() + "/" + relative;
        std::ifstream existing(local_target, std::ios::binary);
        if (!existing.good()) {
            auto asset = is_tunnel_url(worker.url)
                ? router_tunnel_request(worker, "GET", worker_path, api_key, "", "", false)
                : http_worker_request("GET", worker.url, worker_path, api_key, "", "");
            if (asset.status != 200) {
                continue;
            }
            write_file(local_target, asset.body);
        }
        const std::string replacement = public_base.empty() ? worker_path : public_base + worker_path;
        replace_all(&rewritten, url, replacement);
        changed = true;
    }

    if (!changed) {
        return response;
    }
    return {response.status, response.content_type, rewritten};
}

void NativeServer::usage_flush_loop() {
    while (running_) {
        for (int i = 0; i < 100 && running_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (running_) {
            flush_usage_if_dirty();
        }
    }
}

void NativeServer::flush_usage_if_dirty() const {
    std::lock_guard<std::mutex> lock(usage_mutex_);
    if (!usage_dirty_) {
        return;
    }
    usage_dirty_ = false;
    persist_usage_locked();
}

void NativeServer::mark_usage_dirty_locked() const {
    usage_dirty_ = true;
}

void NativeServer::persist_usage_locked() const {
    write_file(usage_file_path(), usage_snapshot_json_locked());
    write_file(usage_history_file_path(), usage_history_json_locked());
}

void NativeServer::queue_processor(const char *payload, char *result, size_t result_size, void *user_data) {
    auto *server = static_cast<NativeServer *>(user_data);
    bool counted_model = false;
    std::string slot_model, slot_replica;
    std::shared_ptr<QueuedForwardResponse> internal_target;
    try {
        std::string value = payload ? payload : "";
        const auto newline = value.find('\n');
        const auto kind = newline == std::string::npos ? value : value.substr(0, newline);
        const auto body_start = newline == std::string::npos ? std::string::npos : value.find('\n',newline+1);
        const auto response_key = body_start == std::string::npos ? "" : value.substr(newline+1,body_start-newline-1);
        { std::lock_guard<std::mutex> lock(server->queued_forward_mutex_); const auto found = server->queued_forward_responses_.find(response_key); if (found != server->queued_forward_responses_.end() && found->second->internal) internal_target = found->second; }
        auto body = body_start == std::string::npos ? std::string() : value.substr(body_start+1);
        const auto return_forwarded = [&](HttpResponse response) {
            std::shared_ptr<QueuedForwardResponse> target;
            { std::lock_guard<std::mutex> lock(server->queued_forward_mutex_); const auto found=server->queued_forward_responses_.find(response_key); if(found!=server->queued_forward_responses_.end())target=found->second; }
            if (target) { std::lock_guard<std::mutex> lock(target->mutex); target->response=std::move(response); }
            if (result_size) result[0]='\0';
        };
        const auto return_stream = [&](std::shared_ptr<LlmInferenceStream> stream) {
            std::shared_ptr<QueuedForwardResponse> target;
            { std::lock_guard<std::mutex> lock(server->queued_forward_mutex_);
              server->local_streams_.erase(std::remove_if(server->local_streams_.begin(), server->local_streams_.end(), [](const auto &handle) { return handle.first.expired(); }), server->local_streams_.end());
              server->local_streams_.push_back(stream->owner_handle());
              const auto found = server->queued_forward_responses_.find(response_key);
              if (found != server->queued_forward_responses_.end()) target = found->second; }
            if (target) { std::lock_guard<std::mutex> lock(target->mutex); target->stream = std::move(stream); }
            if (result_size) result[0] = '\0';
        };
        const auto llm_path = kind == "chat" ? "/v1/chat/completions" : "/v1/completions";
        // FALLBACK_SERVER keeps the source behavior; without it, text that
        // cannot be served locally goes to ROUTER_URL when one is configured.
        const auto forward_to_router = [&](const std::string &request) {
            if (internal_target || !server->fallback_client().url.empty()) return false;
            HttpResponse routed; std::shared_ptr<LlmInferenceStream> stream;
            if (!server->forward_llm_to_router(llm_path, request, &routed, &stream)) return false;
            if (stream) return_stream(std::move(stream)); else return_forwarded(std::move(routed));
            return true;
        };
        const bool llm_request = kind == "chat" || kind == "completion";
        bool disable_fallback = false;
        if (llm_request) {
            disable_fallback = llm_fallback_disabled(body);
            auto fields = json_object_ordered_entries(body);
            fields.erase(std::remove_if(fields.begin(), fields.end(), [](const auto &entry) { return entry.first == "disable_fallback"; }), fields.end());
            body = json_object_from_ordered_entries(fields);
            bool unavailable;
            { std::lock_guard<std::mutex> lock(server->resource_mutex_); unavailable = server->llm_temporarily_unavailable_; }
            if (!unavailable && !disable_fallback) {
                bool use_fallback;
                try { use_fallback = server->should_use_llm_fallback(); }
                catch (const std::exception &) {
                    // A precheck failure escapes Pipes.get_response and the
                    // API maps the queue exception to this HTTPException.
                    return_forwarded({500, "application/json", "{\"detail\":\"Internal server error\"}"});
                    return;
                }
                HttpResponse fallback;
                if (use_fallback && server->forward_fallback_llm(kind == "chat" ? "/v1/chat/completions" : "/v1/completions", body, &fallback)) {
                    return_forwarded(std::move(fallback));
                    return;
                }
                if (use_fallback && forward_to_router(body)) return;
            }
            if (unavailable) {
                HttpResponse fallback;
                std::string response = "{\"error\":{\"message\":\"LLM temporarily unavailable while this worker is using its shared GPU slot.\",\"type\":\"temporarily_unavailable\"}}";
                if (!disable_fallback && server->forward_fallback_llm(kind == "chat" ? "/v1/chat/completions" : "/v1/completions", body, &fallback)) {
                    return_forwarded(std::move(fallback));
                    return;
                }
                if (!disable_fallback && forward_to_router(body)) return;
                if (internal_target) { std::lock_guard<std::mutex> lock(internal_target->mutex); internal_target->local_body = response; }
                else if (llm_output_falsey(json_object_field(body, "stream", "false"))) {
                    // Python answers 200 with this error dictionary; the router
                    // then returns it as a completion. 503 lets it fail over.
                    // Streams already end without text, which it retries.
                    return_forwarded({503, "application/json", response});
                    return;
                }
                snprintf(result, result_size, "%s", response.c_str());
                return;
            }
            slot_model = server->resolve_request_slot_model(body);
            slot_replica = server->begin_llm_inference(slot_model, body);
            counted_model = true;
        }
        std::string response;
        if (llm_request) {
            InferenceResult inference("");
            LlmInputOperations input;
            input.transcribe = [server](const std::string &audio, const std::string &format) {
                try { return server->transcribe_llm_audio(audio, format); }
                catch (const SttError &error) { throw LlmResponseError{error.category,error.what()}; }
                catch (const LlmResponseError &) { throw; }
                catch (const std::exception &error) { throw LlmResponseError{"RuntimeError", error.what()}; }
            };
            input.transcribe_segments = [server](const std::string &audio, const std::string &format) {
                try { return server->transcribe_llm_audio(audio, format, true); }
                catch (const SttError &error) { throw LlmResponseError{error.category,error.what()}; }
                catch (const LlmResponseError &) { throw; }
                catch (const std::exception &error) { throw LlmResponseError{"RuntimeError", error.what()}; }
            };
            try { inference = server->backend_->pipeline_text_result(body, slot_replica, kind == "chat", input); }
            catch (const LlmResponseError &) {
                server->end_llm_inference(slot_model, slot_replica); counted_model = false;
                return_forwarded({500, "application/json", "{\"detail\":\"Internal server error\"}"});
                return;
            }
            if (inference.stream) {
                inference.stream->on_close([server, slot_model, slot_replica] { server->end_llm_inference(slot_model, slot_replica); });
                counted_model = false;
                std::shared_ptr<QueuedForwardResponse> target;
                { std::lock_guard<std::mutex> lock(server->queued_forward_mutex_);
                  server->local_streams_.erase(std::remove_if(server->local_streams_.begin(), server->local_streams_.end(), [](const auto &handle) { return handle.first.expired(); }), server->local_streams_.end());
                  server->local_streams_.push_back(inference.stream->owner_handle());
                  const auto found = server->queued_forward_responses_.find(response_key);
                  if (found != server->queued_forward_responses_.end()) target = found->second; }
                if (target) { std::lock_guard<std::mutex> lock(target->mutex); target->stream = inference.stream; }
                if (result_size) result[0] = '\0';
                return;
            }
            response = inference.body;
            if (inference.failure && !inference.lazy_chat_stream && !internal_target
                && server->fallback_client().url.empty()) {
                // Release the local slot first so the router does not wait on
                // this worker while it holds the request.
                server->end_llm_inference(slot_model, slot_replica); counted_model = false;
                if (!disable_fallback && forward_to_router(body)) return;
                return_forwarded(llm_failure_response(inference.failure->message, inference.body));
                return;
            }
            if (inference.failure && !inference.lazy_chat_stream) {
                try { response = server->post_inference_fallback(inference.prepared_request.empty() ? body : inference.prepared_request, kind == "chat"); }
                catch (const std::exception &) {
                    server->end_llm_inference(slot_model, slot_replica); counted_model = false;
                    return_forwarded({500, "application/json", "{\"detail\":\"Internal server error\"}"});
                    return;
                }
            }
            if (inference.from_engine) {
                std::shared_ptr<QueuedForwardResponse> target;
                { std::lock_guard<std::mutex> lock(server->queued_forward_mutex_); const auto found = server->queued_forward_responses_.find(response_key); if (found != server->queued_forward_responses_.end()) target = found->second; }
                if (target) {
                    std::lock_guard<std::mutex> lock(target->mutex);
                    target->source_response = true;
                    target->fallback_dictionary = (inference.failure && !inference.lazy_chat_stream)
                        || (kind == "completion" && !starts_with(trim(response), "data:"));
                    // The C ABI's bounded output buffer cannot represent an
                    // arbitrary Python result. Keep the owned C++ result until
                    // the waiter consumes it (or discards a late response).
                    target->local_body = response;
                }
            }
            if (inference.failure && inference.lazy_chat_stream) {
                server->end_llm_inference(slot_model, slot_replica);
                counted_model = false;
                if (!extract_json_bool(body, "stream", false)) {
                    // FastAPI consumes a raw-truthy, Pydantic-false generator
                    // while encoding the response; the lazy error yields 500.
                    return_forwarded({500, "text/plain; charset=utf-8", "Internal Server Error"});
                } else {
                    HttpResponse streamed{200, "text/event-stream; charset=utf-8",
                        inference.stream_prefix + llm_stream_error_event(inference.failure->message)};
                    streamed.chunked = true;
                    return_forwarded(std::move(streamed));
                }
                return;
            }
        } else if (kind == "embedding") {
            ScopedCapabilityUse embedding_use(server, "embedding");
            response = server->backend_->embedding(body);
        } else {
            response = make_error(400, "unknown inference kind");
        }
        if (counted_model) {
            server->end_llm_inference(slot_model, slot_replica);
            counted_model = false;
        }
        if (internal_target) { std::lock_guard<std::mutex> lock(internal_target->mutex); internal_target->local_body = response; }
        snprintf(result, result_size, "%s", response.c_str());
    } catch (const std::exception &ex) {
        if (counted_model) {
            server->end_llm_inference(slot_model, slot_replica);
        }
        const auto response = make_error(500, ex.what());
        if (internal_target) { std::lock_guard<std::mutex> lock(internal_target->mutex); internal_target->failure = std::current_exception(); }
        snprintf(result, result_size, "%s", response.c_str());
    }
}

void NativeServer::accept_loop() {
    while (running_) {
        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);
        int client_fd = accept(server_fd_, reinterpret_cast<sockaddr *>(&client_addr), &len);
        if (client_fd < 0) {
            if (running_) {
                continue;
            }
            break;
        }
        { std::lock_guard<std::mutex> lock(client_mutex_); client_fds_.insert(client_fd); }
        std::thread(&NativeServer::handle_client, this, client_fd).detach();
    }
}

void NativeServer::handle_router_tunnel_websocket(int client_fd, const std::string &worker_id) {
    const uint64_t generation = router_tunnel_attach(worker_id, client_fd);
    std::string close_reason = "ws closed";
    char buffer[4096];
    std::string frame_buffer;
    const double ping_interval = std::max(0.01, parse_double_env_explicit("ROUTER_TUNNEL_PING_INTERVAL", 15.0));
    const double pong_timeout = std::max(0.01, parse_double_env_explicit("ROUTER_TUNNEL_PONG_TIMEOUT", 45.0));
    double last_recv = monotonic_epoch_seconds();
    double last_ping = last_recv;
    timeval read_timeout{};
    read_timeout.tv_sec = 0;
    read_timeout.tv_usec = 20000;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &read_timeout, sizeof(read_timeout));
    auto send_tunnel_text = [&](const std::string &payload) {
        const std::string frame = websocket_text_frame(payload);
        std::lock_guard<std::mutex> lock(router_mutex_);
        auto tunnel_it = router_tunnels_.find(worker_id);
        if (tunnel_it == router_tunnels_.end()
            || !tunnel_it->second.connected
            || tunnel_it->second.generation != generation
            || tunnel_it->second.fd != client_fd) {
            return false;
        }
        size_t sent = 0;
        while (sent < frame.size()) {
            ssize_t n = send(client_fd, frame.data() + sent, frame.size() - sent, 0);
            if (n <= 0) {
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    };
    while (running_) {
        const double now = monotonic_epoch_seconds();
        if (now - last_recv > pong_timeout) {
            close_reason = "no frames in " + std::to_string(static_cast<int>(now - last_recv)) + "s (pong timeout)";
            break;
        }
        if (now - last_ping >= ping_interval) {
            if (!send_tunnel_text("{\"t\":\"ping\"}")) {
                close_reason = "keepalive send failed";
                break;
            }
            last_ping = now;
        }
        ssize_t n = recv(client_fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
                continue;
            }
            break;
        }
        last_recv = monotonic_epoch_seconds();
        router_tunnel_note_recv(worker_id, generation);
        frame_buffer.append(buffer, static_cast<size_t>(n));
        WebSocketFrame frame;
        while (pop_websocket_frame(&frame_buffer, &frame)) {
            if (frame.opcode == 0x8U) {
                const std::string close_frame = websocket_close_frame(1000);
                send(client_fd, close_frame.data(), close_frame.size(), 0);
                router_tunnel_detach(worker_id, generation, close_reason);
                close(client_fd);
                return;
            }
            if (frame.opcode == 0x9U) {
                const std::string pong = websocket_pong_frame();
                send(client_fd, pong.data(), pong.size(), 0);
            } else if (frame.opcode == 0x1U) {
                const std::string message_type = extract_json_string(frame.payload, "t", "");
                if (message_type == "ping") {
                    if (!send_tunnel_text("{\"t\":\"pong\"}")) {
                        close_reason = "json pong send failed";
                        break;
                    }
                } else {
                    router_tunnel_handle_message(frame.payload);
                }
            }
        }
        if (close_reason != "ws closed") {
            break;
        }
    }
    router_tunnel_detach(worker_id, generation, close_reason);
    close(client_fd);
}

void NativeServer::handle_tts_websocket(int client_fd,const std::string &initial) {
    // handle_client owns the descriptor. Closing it here would let its final
    // cleanup close an unrelated fd reused by a concurrent model/connection.
    const int flags=fcntl(client_fd,F_GETFL,0);
    if(flags<0||fcntl(client_fd,F_SETFL,flags|O_NONBLOCK)<0)return;
    std::mutex send_mutex;std::atomic<bool> interrupted{false};
    const auto send_frame=[&](const std::string &frame,bool cancellable=true){
        HttpDeadline deadline(30,false);std::lock_guard<std::mutex> lock(send_mutex);
        for(size_t sent=0;sent<frame.size();){
            if(cancellable&&interrupted)throw TtsError("CancelledError","");
            if(deadline.expired())throw TtsError("TimeoutError","");
            const auto count=send(client_fd,frame.data()+sent,frame.size()-sent,MSG_NOSIGNAL|MSG_DONTWAIT);
            if(count>0){sent+=static_cast<size_t>(count);continue;}
            if(count<0&&errno==EINTR)continue;
            if(count<0&&(errno==EAGAIN||errno==EWOULDBLOCK)){pollfd ready{client_fd,POLLOUT,0};poll(&ready,1,20);continue;}
            throw TtsError("WebSocketDisconnect","");
        }
    };
    const auto close_session=[&](int code,const std::string &message){
        size_t end=0;for(unsigned count=0;count<100&&end<message.size();++count){int32_t at=static_cast<int32_t>(end);UChar32 c;U8_NEXT(message.data(),at,static_cast<int32_t>(message.size()),c);end=static_cast<size_t>(at);}
        const auto reason=message.substr(0,end);
        // The source WebSocket library rejects oversized close control frames;
        // it does not silently cut a UTF-8 sequence to fit 123 bytes.
        if(reason.size()<=123)try{send_frame(websocket_close_frame(code,reason),false);}catch(...){}
    };
    if(llm_unicode_lower(ezl_getenv_default("TTS_ENABLED"))=="false"){close_session(1008,"TTS disabled");return;}
    std::string buffered=initial,fragment;unsigned char fragment_opcode=0;
    const auto valid_utf8=[](const std::string &value){for(int32_t at=0;at<static_cast<int32_t>(value.size());){UChar32 c;U8_NEXT(value.data(),at,static_cast<int32_t>(value.size()),c);if(c<0)return false;}return true;};
    const auto protocol=[](int code){throw TtsError("WebSocketProtocol"+std::to_string(code),"");};
    unsigned utf8_remaining=0,utf8_min=0x80,utf8_max=0xbf;
    const auto text_fragment=[&](const std::string &bytes,bool final){
        // websockets validates UTF-8 incrementally, including unfinished text
        // messages. Only a valid incomplete suffix may await continuation.
        for(unsigned char byte:bytes){
            if(utf8_remaining){if(byte<utf8_min||byte>utf8_max)protocol(1007);--utf8_remaining;utf8_min=0x80;utf8_max=0xbf;}
            else if(byte<0x80)continue;
            else if(byte>=0xc2&&byte<=0xdf)utf8_remaining=1;
            else if(byte>=0xe0&&byte<=0xef){utf8_remaining=2;if(byte==0xe0)utf8_min=0xa0;if(byte==0xed)utf8_max=0x9f;}
            else if(byte>=0xf0&&byte<=0xf4){utf8_remaining=3;if(byte==0xf0)utf8_min=0x90;if(byte==0xf4)utf8_max=0x8f;}
            else protocol(1007);
        }
        if(final&&utf8_remaining)protocol(1007);
    };
    TtsWebSocketOperations operations;
    operations.receive=[&]{
        HttpDeadline deadline(30,false);
        for(;;){
            WebSocketFrame frame;bool complete=false,final=true;
            if(buffered.size()>=2){
                const auto first=static_cast<unsigned char>(buffered[0]),second=static_cast<unsigned char>(buffered[1]);
                const unsigned opcode=first&15;final=(first&128)!=0;
                if((first&112)||!(second&128)||(opcode>2&&opcode<8)||opcode>10)protocol(1002);
                uint64_t length=second&127;size_t at=2;bool length_ready=true;
                if(length==126||length==127){const size_t width=length==126?2:8;if(buffered.size()<at+width)length_ready=false;else{length=0;for(size_t i=0;i<width;++i)length=(length<<8)|static_cast<unsigned char>(buffered[at++]);}}
                if(length_ready){
                    if(opcode>=8&&(!final||length>125))protocol(1002);
                    if(length>16*1024*1024||(opcode<8&&length+fragment.size()>16*1024*1024))protocol(1009);
                    complete=pop_websocket_frame(&buffered,&frame);
                }
            }
            if(!complete){char bytes[4096];const auto count=http_read(client_fd,nullptr,bytes,sizeof(bytes),deadline);if(count<=0){if(deadline.expired())throw TtsError("TimeoutError","");throw TtsError("WebSocketDisconnect","");}buffered.append(bytes,count);continue;}
            if(frame.opcode==8){
                if(frame.payload.size()==1)protocol(1002);
                if(frame.payload.size()>=2){const auto code=(static_cast<unsigned char>(frame.payload[0])<<8)|static_cast<unsigned char>(frame.payload[1]);if(!((code>=3000&&code<5000)||code==1000||code==1001||code==1002||code==1003||(code>=1007&&code<=1014)))protocol(1002);if(!valid_utf8(frame.payload.substr(2)))protocol(1007);}
                send_frame(websocket_data_frame(8,frame.payload));throw TtsError("WebSocketDisconnect","");
            }
            if(frame.opcode==9){send_frame(websocket_data_frame(10,frame.payload));continue;}
            if(frame.opcode==10)continue;
            if(frame.opcode==0){if(!fragment_opcode)protocol(1002);if(fragment_opcode==1)text_fragment(frame.payload,final);fragment+=frame.payload;if(!final)continue;frame.opcode=fragment_opcode;frame.payload=std::move(fragment);fragment.clear();fragment_opcode=0;}
            else {if(fragment_opcode)protocol(1002);if(frame.opcode==1)text_fragment(frame.payload,final);if(!final){fragment_opcode=frame.opcode;fragment=std::move(frame.payload);continue;}}
            if(frame.opcode==2)throw TtsError("KeyError","'text'");
            if(!valid_utf8(frame.payload))protocol(1007);
            return frame.payload;
        }
    };
    operations.send=[&](const auto &bytes){send_frame(websocket_binary_frame(bytes));};
    operations.interrupt_receive=[&]{interrupted=true;shutdown(client_fd,SHUT_RD);};
    operations.acquire=[&](const auto &cancellation)->std::unique_ptr<TtsWebSocketLease>{if(!tts_service_)throw TtsError("RuntimeError","Native TTS service is unavailable");return tts_service_->websocket_lease(cancellation);};
    operations.state_changed=[this]{notify_worker_state_changed();};
    try{run_tts_websocket(operations);close_session(1000,"");}
    catch(const TtsError &error){
        if(error.category=="WebSocketDisconnect"||error.category=="CancelledError")return;
        if(starts_with(error.category,"WebSocketProtocol")){close_session(std::stoi(error.category.substr(17)),"");return;}
        const bool policy=error.category=="ValueError"||error.category=="JSONDecodeError"||error.category=="TimeoutError";
        auto message=std::string(error.what());if(policy&&message.empty())message="TTS session idle timeout";
        close_session(policy?1008:1011,message);
    }catch(const VoiceCancelled &){}catch(const std::exception &error){close_session(1011,error.what());}catch(...){close_session(1011,"");}
}

void NativeServer::handle_client(int client_fd) {
    bool publish_completion = false;
    auto client_cleanup = std::shared_ptr<int>(new int(client_fd), [this, &publish_completion](int *value) {
        // HttpResponse and its deferred stream ownership are destroyed first.
        // Keep this client registered until the notification has finished so
        // stop() cannot destroy the server while this deleter still uses it.
        if (publish_completion) notify_worker_state_changed();
        {
            std::lock_guard<std::mutex> lock(client_mutex_);
            client_fds_.erase(*value); close(*value); delete value; client_cv_.notify_all();
        }
    });
    HttpResponse response;
    bool head_request = false;
    std::string request_origin;
    bool request_has_cookie = false;
    try {
        std::string client_host;
        sockaddr_in peer_addr{};
        socklen_t peer_len = sizeof(peer_addr);
        if (getpeername(client_fd, reinterpret_cast<sockaddr *>(&peer_addr), &peer_len) == 0) {
            char host_buffer[INET_ADDRSTRLEN] = {0};
            if (inet_ntop(AF_INET, &peer_addr.sin_addr, host_buffer, sizeof(host_buffer))) {
                client_host = host_buffer;
            }
        }
        const std::string raw = read_socket_request(client_fd);
        const auto header_end = raw.find("\r\n\r\n");
        if (header_end == std::string::npos) {
            response = invalid_http_request_response();
        } else {
            std::istringstream stream(raw.substr(0, header_end + 2));
            std::string method;
            std::string path;
            std::string version;
            stream >> method >> path >> version;
            if (method.empty() || path.empty() || (version != "HTTP/1.0" && version != "HTTP/1.1")) {
                response = invalid_http_request_response();
            } else {
            head_request = method == "HEAD";
            std::string ignored;
            std::getline(stream, ignored);
            auto headers = parse_headers(stream);
            std::string authorization;
            auto auth_it = headers.find("authorization");
            if (auth_it != headers.end()) {
                authorization = auth_it->second;
            }
            auto origin_it = headers.find("origin");
            auto request_method_it = headers.find("access-control-request-method");
            if (origin_it != headers.end()) {
                request_origin = origin_it->second;
            }
            request_has_cookie = headers.find("cookie") != headers.end();
            if (method == "OPTIONS" && origin_it != headers.end() && request_method_it != headers.end()) {
                const auto request_headers_it = headers.find("access-control-request-headers");
                response = cors_preflight_response(
                    origin_it->second,
                    request_headers_it == headers.end() ? "" : request_headers_it->second
                );
                const std::string serialized = serialize_response(response);
                size_t sent = 0;
                while (sent < serialized.size()) {
                    ssize_t n = send(client_fd, serialized.data() + sent, serialized.size() - sent, MSG_NOSIGNAL);
                    if (n <= 0) {
                        break;
                    }
                    sent += static_cast<size_t>(n);
                }
                return;
            }
            const auto websocket_query_pos = path.find('?');
            const std::string websocket_route_path = websocket_query_pos == std::string::npos ? path : path.substr(0, websocket_query_pos);
            const std::string websocket_query = websocket_query_pos == std::string::npos ? "" : path.substr(websocket_query_pos + 1);
            publish_completion = authorized(authorization) && ((method == "POST" && (
                websocket_route_path == "/v1/chat/completions" || websocket_route_path == "/v1/completions" ||
                websocket_route_path == "/v1/embeddings" || starts_with(websocket_route_path, "/v1/audio/") ||
                starts_with(websocket_route_path, "/v1/images/") || starts_with(websocket_route_path, "/v1/video") ||
                starts_with(websocket_route_path, "/v1/music"))) ||
                websocket_route_path == "/v1/audio/speech/ws");
            if (is_websocket_upgrade(headers)) {
                bool websocket_protocol_error = false;
                if (method != "GET") {
                    response = websocket_bad_request_response("did not receive a valid HTTP request.");
                    websocket_protocol_error = true;
                } else {
                    auto key_it = headers.find("sec-websocket-key");
                    if (key_it == headers.end()) {
                        response = websocket_bad_request_response("missing Sec-WebSocket-Key header.");
                        websocket_protocol_error = true;
                    } else if (!valid_websocket_key(key_it->second)) {
                        response = websocket_bad_request_response("invalid Sec-WebSocket-Key header: " + key_it->second + ".");
                        websocket_protocol_error = true;
                    } else {
                        auto version_it = headers.find("sec-websocket-version");
                        if (version_it == headers.end()) {
                            response = websocket_bad_request_response("missing Sec-WebSocket-Version header.");
                            websocket_protocol_error = true;
                        } else if (trim(version_it->second) != "13") {
                            response = websocket_bad_request_response("invalid Sec-WebSocket-Version header: " + version_it->second + ".");
                            websocket_protocol_error = true;
                        } else if (!websocket_route_available(websocket_route_path, native_router_mode_enabled())) {
                            response = websocket_plain_response(403, "");
                            websocket_protocol_error = true;
                        } else if(websocket_route_path=="/v1/audio/speech/ws"&&!authorized(authorization)) {
                            response=websocket_plain_response(403,"");websocket_protocol_error=true;
                        } else {
                            const std::string accept = websocket_accept_value(key_it->second);
                            std::ostringstream handshake;
                            handshake << "HTTP/1.1 101 Switching Protocols\r\n"
                                << "Upgrade: websocket\r\n"
                                << "Connection: Upgrade\r\n"
                                << "Sec-WebSocket-Accept: " << accept << "\r\n\r\n";
                            const std::string raw_handshake = handshake.str();
                            if (websocket_route_path == "/v1/router/tunnel") {
                                int close_code = 0;
                                const std::string worker_id = query_param_value(websocket_query, "worker_id", "");
                                if (!authorized_for_worker_websocket(authorization)) {
                                    close_code = 4401;
                                } else if (worker_id.empty()) {
                                    close_code = 4400;
                                }
                                send(client_fd, raw_handshake.data(), raw_handshake.size(), MSG_NOSIGNAL);
                                if (close_code != 0) {
                                    const std::string close_frame = websocket_close_frame(close_code);
                                    send(client_fd, close_frame.data(), close_frame.size(), MSG_NOSIGNAL);
                                } else {
                                    handle_router_tunnel_websocket(client_fd, worker_id);
                                }
                                return;
                            }
                            send(client_fd, raw_handshake.data(), raw_handshake.size(), MSG_NOSIGNAL);
                            handle_tts_websocket(client_fd,raw.substr(header_end+4));
                            return;
                        }
                    }
                }
                if (websocket_protocol_error) {
                    const std::string serialized = serialize_response(response);
                    size_t sent = 0;
                    while (sent < serialized.size()) {
                        ssize_t n = send(client_fd, serialized.data() + sent, serialized.size() - sent, MSG_NOSIGNAL);
                        if (n <= 0) {
                            break;
                        }
                        sent += static_cast<size_t>(n);
                    }
                    return;
                }
            }
            std::string content_type;
            auto content_type_it = headers.find("content-type");
            if (content_type_it != headers.end()) {
                content_type = content_type_it->second;
            }
            std::string host_header;
            auto host_it = headers.find("host");
            if (host_it != headers.end()) {
                host_header = host_it->second;
            }
            const std::string body = raw.substr(header_end + 4);
            response = handle_request(method, path, authorization, body, content_type, client_host, host_header, headers, true);
            }
        }
    } catch (const std::exception &ex) {
        response = {500, "application/json", make_error(500, ex.what())};
    }
    if (!head_request && response.chunked && response.stream_writer)
        http_socket::low_latency(client_fd);
    const std::string serialized = serialize_response(response, head_request, request_origin, request_has_cookie);
    size_t sent = 0;
    while (sent < serialized.size()) {
        ssize_t n = send(client_fd, serialized.data() + sent, serialized.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            break;
        }
        sent += static_cast<size_t>(n);
    }
    if (response.stream_writer && !head_request && sent == serialized.size()) {
        response.stream_writer(client_fd);
    }
}

} // namespace ezlocalai
