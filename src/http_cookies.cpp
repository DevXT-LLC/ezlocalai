// Cookie grammar and behavior adapted from aiohttp 3.13.2 and 3.14.5.
// Copyright aio-libs contributors. Apache-2.0; see licenses/aiohttp.txt.
#include "ezlocalai/http_cookies.hpp"
#include "ezlocalai/configuration.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <optional>
#include <regex>
#include <stdexcept>
#include <unicode/utf8.h>

namespace ezlocalai {
namespace {
std::string lower(std::string value) {
    for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return value;
}
void validate_cookie_text(const std::string &value) {
    for (unsigned char c : value) if (c < 32 || c == 127)
        throw std::invalid_argument("Control characters are not allowed in cookies");
}
bool ip_host(const std::string &host) {
    if (host.find(':') != std::string::npos) return true;
    bool digit = false;
    for (char c : host) {
        if (c >= '0' && c <= '9') digit = true;
        else if (c != '.') return false;
    }
    return digit;
}
bool domain_match(const std::string &domain, const std::string &host) {
    if (host == domain) return true;
    return !domain.empty() && host.size() > domain.size()
        && host.compare(host.size() - domain.size(), domain.size(), domain) == 0
        && host[host.size() - domain.size() - 1] == '.' && !ip_host(host);
}
void append_utf8(std::string &out, unsigned code) {
    char buffer[4]; int32_t length = 0;
    U8_APPEND_UNSAFE(buffer, length, code);
    out.append(buffer, length);
}
std::string unquote_cookie(const std::string &value) {
    if (value.size() < 2 || value.front() != '"' || value.back() != '"') return value;
    std::string out;
    for (size_t i = 1; i + 1 < value.size(); ++i) {
        if (value[i] != '\\' || i + 2 >= value.size() || value[i + 1] == '\n') { out += value[i]; continue; }
        if (i + 4 < value.size() && value[i + 1] >= '0' && value[i + 1] <= '3'
            && value[i + 2] >= '0' && value[i + 2] <= '7' && value[i + 3] >= '0' && value[i + 3] <= '7') {
            append_utf8(out, (value[i + 1] - '0') * 64 + (value[i + 2] - '0') * 8 + value[i + 3] - '0');
            i += 3;
        } else out += value[++i];
    }
    return out;
}
std::string quote_cookie(const std::string &value) {
    static const std::string legal = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'*+-.^_`|~:";
    if (!value.empty() && value.find_first_not_of(legal) == std::string::npos) return value;
    std::string out = "\"";
    for (int32_t pos = 0; pos < static_cast<int32_t>(value.size());) {
        const auto begin = pos; UChar32 code;
        U8_NEXT(value.data(), pos, static_cast<int32_t>(value.size()), code);
        if (code < 0) { out.append(value, begin, pos - begin); continue; }
        if (code == '"' || code == '\\') { out += '\\'; out += static_cast<char>(code); }
        else if (code < 32 || code == ',' || code == ';' || (code >= 127 && code <= 255)) {
            out += '\\'; out += static_cast<char>('0' + (code >> 6));
            out += static_cast<char>('0' + ((code >> 3) & 7)); out += static_cast<char>('0' + (code & 7));
        } else out.append(value, begin, pos - begin);
    }
    return out + '"';
}
std::string wire_utf8(const std::string &value) {
    std::string out;
    for (int32_t pos = 0; pos < static_cast<int32_t>(value.size());) {
        const auto begin = pos; UChar32 code;
        U8_NEXT(value.data(), pos, static_cast<int32_t>(value.size()), code);
        if (code >= 0) out.append(value, begin, pos - begin);
    }
    return out;
}
std::optional<double> max_age(const std::string &raw) {
    // Python int accepts Unicode decimal digits and underscores, but rejects
    // the extra C0 characters accepted by str.strip(). Its float conversion can
    // overflow even though positive expiration timestamps are later clamped.
    for (unsigned char c : raw) if (c >= 0x1c && c <= 0x1f) return {};
    auto value = normalize_decimal_digits(strip_config_value(raw));
    static const std::regex integer("[+-]?[0-9]+(_[0-9]+)*");
    if (!std::regex_match(value, integer)) return {};
    value.erase(std::remove(value.begin(), value.end(), '_'), value.end());
    const auto digits = value.size() - (!value.empty() && (value[0] == '+' || value[0] == '-'));
    if (digits > 4300) return {};
    return std::stod(value);
}
std::optional<double> cookie_date(const std::string &raw) {
    // RFC 6265 token extraction, including the source's permissive prefix
    // matches, two-digit years, ignored zones and normalized month overflow.
    const auto value = normalize_decimal_digits(raw);
    std::vector<std::string> tokens; std::string token;
    for (int32_t pos = 0; pos < static_cast<int32_t>(value.size());) {
        const auto begin = pos; UChar32 code;
        U8_NEXT(value.data(), pos, static_cast<int32_t>(value.size()), code);
        const bool accepted = (code >= 0 && code <= 8) || (code >= 10 && code <= 31)
            || (code >= '0' && code <= '9') || code == ':' || (code >= 'a' && code <= 'z')
            || (code >= 'A' && code <= 'Z') || (code >= 127 && code <= 255);
        if (accepted) token.append(value, begin, pos - begin);
        else if (!token.empty()) { tokens.push_back(std::move(token)); token.clear(); }
    }
    if (!token.empty()) tokens.push_back(std::move(token));
    static const std::regex time("([0-9]{1,2}):([0-9]{1,2}):([0-9]{1,2})"), day_re("[0-9]{1,2}"), year_re("[0-9]{2,4}");
    static const std::vector<std::string> months{"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"};
    bool found_time = false, found_day = false, found_month = false, found_year = false;
    int hour = 0, minute = 0, second = 0, day = 0, month = 0, year = 0;
    for (const auto &part : tokens) {
        std::smatch match;
        if (!found_time && std::regex_search(part, match, time, std::regex_constants::match_continuous)) {
            found_time = true; hour = std::stoi(match[1]); minute = std::stoi(match[2]); second = std::stoi(match[3]); continue;
        }
        if (!found_day && std::regex_search(part, match, day_re, std::regex_constants::match_continuous)) {
            found_day = true; day = std::stoi(match[0]); continue;
        }
        if (!found_month) {
            const auto prefix = lower(part.substr(0, 3)); const auto it = std::find(months.begin(), months.end(), prefix);
            if (it != months.end()) { month = 1 + static_cast<int>(it - months.begin()); found_month = true; continue; }
        }
        if (!found_year && std::regex_search(part, match, year_re, std::regex_constants::match_continuous)) {
            found_year = true; year = std::stoi(match[0]);
        }
    }
    if (year >= 70 && year <= 99) year += 1900;
    else if (year >= 0 && year <= 69) year += 2000;
    if (!found_time || !found_day || !found_month || !found_year || day < 1 || day > 31
        || year < 1601 || hour > 23 || minute > 59 || second > 59) return {};
    std::tm date{}; date.tm_year = year - 1900; date.tm_mon = month - 1; date.tm_mday = day;
    date.tm_hour = hour; date.tm_min = minute; date.tm_sec = second;
    return static_cast<double>(timegm(&date));
}
} // namespace

HttpCookieJar::HttpCookieJar(std::function<double()> now) : now_(std::move(now)) {
    if (!now_) now_ = [] { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
}

std::vector<HttpCookieJar::Cookie> HttpCookieJar::parse(const std::vector<std::string> &headers) {
    // This ASCII grammar retains aiohttp's recovery for unmatched quotes and
    // missing semicolons, extended cookie names, and special date formats.
    static const std::string word = R"([a-zA-Z0-9_!#%&'~`><@,:/$*+\-.^|)(?}{=\[\]])";
    static const std::regex pattern("\\s*(" + word + "+?)(?:\\s*=\\s*("
        R"cookie("(?:[^\\"]|\\[^\n])*"|"[^";]*|(?:\w{3,6}day|\w{3}),\s[\w\d\s-]{9,11}\s[\d:]{8}\s(?:GMT|[+-]\d{4})|\w{3}\s+\w{3}\s+[\s\d]\d\s+\d{2}:\d{2}:\d{2}\s+\d{4}|)cookie"
        + word + "*))?\\s*(?:\\s+|;|$)");
    static const std::string names = "!#$%&'()*+-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    static const std::set<std::string> known{"path","domain","max-age","expires","secure","httponly","samesite","partitioned","version","comment"};
    std::vector<Cookie> parsed;
    for (const auto &header : headers) {
        size_t pos = 0; std::optional<size_t> current; bool seen = false;
        while (pos < header.size()) {
            std::match_results<std::string::const_iterator> match;
            if (!std::regex_search(header.begin() + pos, header.end(), match, pattern, std::regex_constants::match_continuous)) break;
            const auto key = match[1].str(), value = match[2].str(), name = lower(key); const bool has_value = match[2].matched;
            pos += match.length();
            if (key.front() == '$') {
                if (!seen) continue;
                const auto attr = name.substr(1);
                if (current && known.count(attr)) {
                    // Python 3.12's Morsel does not define partitioned. The
                    // normal boolean form is ignored; the dollar form raises.
                    if (attr == "partitioned") throw std::invalid_argument("Invalid cookie attribute");
                    validate_cookie_text(value);
                    parsed[*current].attributes[attr] = value;
                }
            } else if (known.count(name)) {
                if (!seen) break;
                if (name == "secure" || name == "httponly" || name == "partitioned") {
                    if (current && name != "partitioned") parsed[*current].attributes[name] = "true";
                } else if (!has_value) break;
                else if (current) {
                    const auto decoded = unquote_cookie(value); validate_cookie_text(decoded);
                    parsed[*current].attributes[name] = decoded;
                }
            } else if (has_value) {
                if (key.find_first_not_of(names) != std::string::npos) current.reset();
                else {
                    const auto decoded = unquote_cookie(value); validate_cookie_text(decoded); validate_cookie_text(value);
                    parsed.push_back({key, decoded, value, {}}); current = parsed.size() - 1; seen = true;
                }
            } else break;
        }
    }
    return parsed;
}

// aiohttp 3.14 parses one cookie per Set-Cookie field. Attribute-like
// cookie names are valid here; unknown attributes never become new cookies.
std::vector<HttpCookieJar::Cookie>
HttpCookieJar::parse_response_headers(const std::vector<std::string> &headers) {
    static const std::string word = R"([a-zA-Z0-9_!#%&'~`><@,:/$*+\-.^|)(?}{=\[\]])";
    static const std::regex pattern(
        "\\s*(" + word +
        "+?)(?:\\s*=\\s*("
        R"cookie("(?:[^\\"]|\\[^\n])*"|"[^";]*|(?:\w{3,6}day|\w{3}),\s[\w\d\s-]{9,11}\s[\d:]{8}\s(?:GMT|[+-]\d{4})|\w{3}\s+\w{3}\s+[\s\d]\d\s+\d{2}:\d{2}:\d{2}\s+\d{4}|)cookie" +
        word + "*))?\\s*(?:\\s+|;|$)");
    static const std::string names =
        "!#$%&'()*+-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    static const std::set<std::string> attributes{"path",     "domain",   "max-age", "expires", "secure",
                                                  "httponly", "samesite", "version", "comment"};
    const auto trim_pair = [](const std::string &value) {
        const auto first = value.find_first_not_of(" \t"), last = value.find_last_not_of(" \t");
        return first == std::string::npos ? std::string{} : value.substr(first, last - first + 1);
    };
    const auto controls = [](const std::string &value, bool allow_tab) {
        return std::any_of(value.begin(), value.end(), [=](unsigned char c) {
            return (c < 32 && !(allow_tab && c == '\t')) || c == 127;
        });
    };
    std::vector<Cookie> result;
    for (const auto &header : headers) {
        if (result.size() == 50)
            break;
        if (header.empty() || controls(header, true) || wire_utf8(header) != header)
            continue;
        auto end = header.find(';');
        size_t attributes_start = end == std::string::npos ? header.size() : end + 1;
        if (end == std::string::npos)
            end = header.size();
        std::string name, coded;
        bool quoted = false;
        if (header.find('"') < end) {
            std::smatch match;
            if (std::regex_search(header, match, pattern, std::regex_constants::match_continuous)) {
                const auto value = match[2].str();
                if (value.size() >= 2 && value.front() == '"' && value.back() == '"' &&
                    value.find(';') != std::string::npos) {
                    const auto after_value = static_cast<size_t>(match.position(2) + match.length(2));
                    const auto tail = header.find_first_not_of(" \t", after_value);
                    if (tail == std::string::npos || header[tail] == ';') {
                        name = match[1];
                        coded = value;
                        attributes_start = tail == std::string::npos ? header.size() : tail + 1;
                        quoted = true;
                    }
                }
            }
        }
        if (!quoted) {
            const auto equals = header.find('=');
            if (equals == std::string::npos || equals >= end)
                continue;
            name = header.substr(0, equals);
            coded = header.substr(equals + 1, end - equals - 1);
        }
        name = trim_pair(name);
        coded = trim_pair(coded);
        if (name.empty() || name.find_first_not_of(names) != std::string::npos ||
            name.size() + coded.size() > 4096)
            continue;
        const auto value = unquote_cookie(coded);
        if (controls(value, false))
            continue;
        Cookie cookie{name, value, coded, {}};
        size_t index = attributes_start;
        for (unsigned count = 0; count < 32 && index < header.size(); ++count) {
            std::match_results<std::string::const_iterator> match;
            if (!std::regex_search(header.begin() + index, header.end(), match, pattern,
                                   std::regex_constants::match_continuous)) {
                const auto semi = header.find(';', index);
                if (semi == std::string::npos)
                    break;
                index = header.find_first_not_of("; \t\r\n\f\v", semi + 1);
                if (index == std::string::npos)
                    break;
                continue;
            }
            index += match.length();
            const auto attr = lower(match[1]);
            if (!attributes.count(attr))
                continue;
            if (attr == "secure" || attr == "httponly") {
                cookie.attributes[attr] = "true";
                continue;
            }
            const auto raw = match[2].str();
            if (raw.size() > 1024)
                continue;
            const auto decoded = unquote_cookie(raw);
            if (!controls(decoded, false))
                cookie.attributes[attr] = decoded;
        }
        result.push_back(std::move(cookie));
    }
    return result;
}

void HttpCookieJar::expire() {
    if (expirations_.empty())
        return;
    const auto now = now_();
    for (auto it = expirations_.begin(); it != expirations_.end();) {
        if (it->second > now) {
            ++it;
            continue;
        }
        host_only_.erase({std::get<0>(it->first), std::get<2>(it->first)});
        response_host_only_.erase(it->first);
        cookies_.erase(it->first);
        it = expirations_.erase(it);
    }
}

void HttpCookieJar::update(const std::vector<std::string> &headers, const HttpUrl &url) {
    store(parse(headers), url);
}
void HttpCookieJar::update_from_headers(const std::vector<std::string> &headers, const HttpUrl &url) {
    response_headers_ = true;
    if (!ip_host(url.host)) store(parse_response_headers(headers), url);
}
void HttpCookieJar::store(const std::vector<Cookie> &parsed, const HttpUrl &url) {
    if (parsed.empty() || ip_host(url.host)) return;
    for (auto cookie : parsed) {
        auto &domain = cookie.attributes["domain"];
        if (!domain.empty() && domain.back() == '.') domain.clear();
        const bool host_only = domain.empty() && !url.host.empty();
        if (host_only) { domain = url.host; if (!response_headers_) host_only_.insert({domain, cookie.name}); }
        validate_cookie_text(domain);
        if (!domain.empty() && domain.front() == '.') domain.erase(0, 1);
        if (!url.host.empty() && !domain_match(domain, url.host)) continue;
        auto &path = cookie.attributes["path"];
        if (path.empty() || path.front() != '/') {
            path = url.decoded_path();
            if (path.empty() || path.front() != '/') path = "/";
            else path = path.substr(0, path.rfind('/'));
            if (path.empty()) path = "/";
            validate_cookie_text(path);
        }
        auto path_key = path; while (!path_key.empty() && path_key.back() == '/') path_key.pop_back();
        const Key key{domain, path_key, cookie.name};
        if (response_headers_) {
            if (host_only) response_host_only_.insert(key); else response_host_only_.erase(key);
            expirations_.erase(key); // Replacements do not inherit an old deadline.
        }
        const auto &age = cookie.attributes["max-age"];
        if (!age.empty()) {
            if (response_headers_) {
                if (const auto seconds = parse_configuration_integer(age)) {
                    if (!configuration_integer_less("0", *seconds)) expirations_[key] = 0;
                    else if (!configuration_integer_less(*seconds, "253402300799")) expirations_[key] = 253402300799.0;
                    else expirations_[key] = std::min(now_() + std::stod(*seconds), 253402300799.0);
                }
            } else if (const auto seconds = max_age(age)) expirations_[key] = std::min(now_() + *seconds, 253402300799.0);
        } else if (const auto date = cookie_date(cookie.attributes["expires"])) {
            if (*date != 0) expirations_[key] = *date;
        }
        cookies_[key] = std::move(cookie);
    }
    expire();
}

std::string HttpCookieJar::header(const HttpUrl &url) {
    if (cookies_.empty()) return {};
    expire();
    if (ip_host(url.host)) return {};
    std::vector<std::string> domains, paths{ "" };
    for (size_t pos = url.host.size();;) {
        const auto dot = pos ? url.host.rfind('.', pos - 1) : std::string::npos;
        domains.push_back(url.host.substr(dot == std::string::npos ? 0 : dot + 1));
        if (dot == std::string::npos) break;
        pos = dot;
    }
    const auto path = url.decoded_path();
    for (size_t pos = 0; pos < path.size();) {
        const auto slash = path.find('/', pos + 1);
        paths.push_back(path.substr(0, slash));
        if (slash == std::string::npos) break;
        pos = slash;
    }
    std::map<std::string, std::string> selected;
    for (const auto &domain : domains) for (const auto &prefix : paths) {
        for (auto it = cookies_.lower_bound(Key{domain, prefix, ""}); it != cookies_.end(); ++it) {
            if (std::get<0>(it->first) != domain || std::get<1>(it->first) != prefix) break;
            const auto &cookie = it->second;
            if ((response_headers_ ? response_host_only_.count(it->first) : host_only_.count({domain, cookie.name})) && domain != url.host) continue;
            if (cookie.attributes.at("path").size() > path.size()) continue;
            const auto secure = cookie.attributes.find("secure");
            if (url.scheme != "https" && url.scheme != "wss" && secure != cookie.attributes.end() && !secure->second.empty()) continue;
            selected[cookie.name] = !cookie.coded_value.empty() && cookie.coded_value.front() == '"'
                ? cookie.coded_value : quote_cookie(cookie.value);
        }
    }
    std::string result;
    for (const auto &cookie : selected) { if (!result.empty()) result += "; "; result += cookie.first + "=" + cookie.second; }
    return wire_utf8(strip_config_value(result));
}
} // namespace ezlocalai
