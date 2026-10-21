// Protocol behavior follows installed edge-tts 7.2.8; see licenses/edge-tts.txt.
#include "ezlocalai/edge_tts.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <regex>
#include <sstream>
namespace ezlocalai {
namespace {
constexpr auto token = "6A5AA1D4EAFF4E9FB37E23D68491D6F4";
bool whitespace(unsigned char c) { return c == ' ' || (c >= 9 && c <= 13); }
std::string strip(const std::string &text) {
    size_t first = 0, last = text.size(); while (first < last && whitespace(text[first])) ++first;
    while (last > first && whitespace(text[last - 1])) --last; return text.substr(first, last - first);
}
std::pair<std::map<std::string, std::string>, std::string> headers(const std::string &data, size_t length) {
    const auto head = data.substr(0, length); std::map<std::string, std::string> fields;
    for (size_t at = 0;;) {
        auto end = head.find("\r\n", at); if (end == std::string::npos) end = head.size();
        const auto colon = head.find(':', at);
        if (colon == std::string::npos || colon >= end) throw TtsError("ValueError", "not enough values to unpack (expected 2, got 1)");
        fields[head.substr(at, colon - at)] = head.substr(colon + 1, end - colon - 1);
        if (end == head.size()) break; at = end + 2;
    }
    return {fields, length + 2 < data.size() ? data.substr(length + 2) : ""};
}
std::string date(double unix_time) {
    const auto value = time_t(std::floor(unix_time)); std::tm utc{}; gmtime_r(&value, &utc);
    // Fixed English names, independent of the worker process locale.
    static const char *days[]{"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
    static const char *months[]{"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    std::ostringstream out; out << days[utc.tm_wday] << ' ' << months[utc.tm_mon] << ' ' << std::setfill('0') << std::setw(2) << utc.tm_mday << ' ' << utc.tm_year + 1900 << ' ' << std::setw(2) << utc.tm_hour << ':' << std::setw(2) << utc.tm_min << ':' << std::setw(2) << utc.tm_sec << " GMT+0000 (Coordinated Universal Time)"; return out.str();
}
}
std::vector<std::string> edge_tts_chunks(const std::string &input) {
    std::string text;
    for (unsigned char c : input) {
        if (c <= 8 || c == 11 || c == 12 || (c >= 14 && c <= 31)) text += ' ';
        else if (c == '&') text += "&amp;"; else if (c == '<') text += "&lt;"; else if (c == '>') text += "&gt;"; else text += char(c);
    }
    std::vector<std::string> result;
    while (text.size() > 4096) {
        auto split = text.rfind('\n', 4095); if (split == std::string::npos) split = text.rfind(' ', 4095);
        // The source passes the full valid UTF-8 string to its fallback helper,
        // so an unbroken long word stays whole instead of being cut at 4096.
        if (split == std::string::npos) split = text.size();
        while (split > 0) {
            const auto amp = text.rfind('&', split - 1); if (amp == std::string::npos) break;
            const auto end = text.find(';', amp); if (end != std::string::npos && end < split) break; split = amp;
        }
        const auto chunk = strip(text.substr(0, split)); if (!chunk.empty()) result.push_back(chunk);
        text.erase(0, split > 0 ? split : 1);
    }
    text = strip(text); if (!text.empty()) result.push_back(text); return result;
}
std::string edge_tts_ssml(const std::string &text, const std::string &voice) {
    std::string resolved = voice; std::smatch match;
    if (std::regex_match(voice, match, std::regex("([a-z]{2,})-([A-Z]{2,})-(.+Neural)"))) {
        auto region = match[2].str(), name = match[3].str(); const auto hyphen = name.find('-');
        if (hyphen != std::string::npos) { region += "-" + name.substr(0, hyphen); name.erase(0, hyphen + 1); }
        resolved = "Microsoft Server Speech Text to Speech Voice (" + match[1].str() + "-" + region + ", " + name + ")";
    }
    if (!std::regex_match(resolved, std::regex("Microsoft Server Speech Text to Speech Voice \\(.+,.+\\)"))) throw TtsError("ValueError", "Invalid voice '" + resolved + "'.");
    return "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='" + resolved + "'><prosody pitch='+0Hz' rate='+0%' volume='+0%'>" + text + "</prosody></voice></speak>";
}
std::string edge_tts_token(double unix_time) {
    double ticks = unix_time + 11644473600.0; ticks -= ticks - 300 * std::floor(ticks / 300); ticks *= 10000000;
    std::ostringstream input; input << std::fixed << std::setprecision(0) << ticks << token; const auto value = input.str();
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    std::ostringstream output; for (auto byte : digest) output << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return output.str();
}
EdgeTtsRequest edge_tts_request(const std::string &text, const std::string &voice, double time, double skew,
    const std::string &connection, const std::string &request, const std::string &muid) {
    EdgeTtsRequest result; result.url = std::string("wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=") + token + "&ConnectionId=" + connection + "&Sec-MS-GEC=" + edge_tts_token(time + skew) + "&Sec-MS-GEC-Version=1-143.0.3650.75";
    result.headers = {{"Pragma", "no-cache"}, {"Cache-Control", "no-cache"}, {"Origin", "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold"}, {"Sec-WebSocket-Version", "13"},
        {"User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"},
        {"Accept-Encoding", "gzip, deflate, br, zstd"}, {"Accept-Language", "en-US,en;q=0.9"}, {"Cookie", "muid=" + muid + ";"}};
    const auto timestamp = date(time);
    result.messages.push_back("X-Timestamp:" + timestamp + "\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n{\"context\":{\"synthesis\":{\"audio\":{\"metadataoptions\":{\"sentenceBoundaryEnabled\":\"true\",\"wordBoundaryEnabled\":\"false\"},\"outputFormat\":\"audio-24khz-48kbitrate-mono-mp3\"}}}}\r\n");
    result.messages.push_back("X-RequestId:" + request + "\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:" + timestamp + "Z\r\nPath:ssml\r\n\r\n" + edge_tts_ssml(text, voice)); return result;
}
bool EdgeTtsStream::accept(int type, const std::string &data) {
    if (type == 1) {
        auto end = data.find("\r\n\r\n"); if (end == std::string::npos) end = data.empty() ? 0 : data.size() - 1;
        const auto [fields, body] = headers(data, end); const auto path = fields.find("Path"); const auto value = path == fields.end() ? "" : path->second;
        if (value == "turn.end") return false;
        if (value == "response" || value == "turn.start") return true;
        if (value != "audio.metadata") throw TtsError("UnknownResponse", "Unknown path received");
        const auto metadata = nlohmann::json::parse(body);
        for (const auto &item : metadata.at("Metadata")) {
            const auto kind = item.at("Type").get<std::string>();
            if (kind == "SessionEnd") continue;
            if (kind != "WordBoundary" && kind != "SentenceBoundary") throw TtsError("UnknownResponse", "Unknown metadata type: " + kind);
            const auto &value = item.at("Data"); (void)value.at("Offset").get<double>(); (void)value.at("Duration").get<double>(); (void)value.at("text").at("Text").get<std::string>(); return true;
        }
        throw TtsError("UnexpectedResponse", "No WordBoundary metadata found");
    }
    if (type != 2) return true;
    if (data.size() < 2) throw TtsError("UnexpectedResponse", "We received a binary message, but it is missing the header length.");
    const size_t length = (size_t(uint8_t(data[0])) << 8) | uint8_t(data[1]);
    if (length > data.size()) throw TtsError("UnexpectedResponse", "The header length is greater than the length of the data.");
    const auto [fields, body] = headers(data, length);
    auto it = fields.find("Path"); if (it == fields.end() || it->second != "audio") throw TtsError("UnexpectedResponse", "Received binary message, but the path is not audio.");
    it = fields.find("Content-Type");
    if (it != fields.end() && it->second != "audio/mpeg") throw TtsError("UnexpectedResponse", "Received binary message, but with an unexpected Content-Type.");
    if (it == fields.end()) { if (body.empty()) return true; throw TtsError("UnexpectedResponse", "Received binary message with no Content-Type, but with data."); }
    if (body.empty()) throw TtsError("UnexpectedResponse", "Received binary message, but it is missing the audio data.");
    received_ = true; audio_ += body; return true;
}
std::string EdgeTtsStream::finish() const { if (!received_) throw TtsError("NoAudioReceived", "No audio was received. Please verify that your parameters are correct."); return audio_; }
}
