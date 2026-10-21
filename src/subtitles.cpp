#include "ezlocalai/subtitles.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/stt_service.hpp"
#include <nlohmann/json.hpp>
#include <unicode/utf8.h>
#include <algorithm>
#include <future>
#include <map>
#include <vector>

namespace ezlocalai {
namespace {
using Json = nlohmann::ordered_json;
size_t words(const std::string &text) {
    size_t count = 0; bool in_word = false;
    for (int32_t i = 0; i < static_cast<int32_t>(text.size());) {
        const auto start = i; UChar32 cp; U8_NEXT(text.data(),i,static_cast<int32_t>(text.size()),cp);
        const bool space = strip_config_value(text.substr(start,i-start)).empty();
        if (!space && !in_word) ++count; in_word = !space;
    }
    return count;
}
std::map<std::string,std::string> numbered(const Json &response) {
    std::string text;
    if (response.is_object() && response.contains("choices")) text = strip_config_value(response.at("choices").at(0).at("message").at("content").get<std::string>());
    std::map<std::string,std::string> result;
    for (size_t start = 0; start <= text.size();) {
        const auto end = text.find('\n',start); const auto line = strip_config_value(text.substr(start,end == std::string::npos ? end : end-start));
        const auto close = line.find(']');
        if (!line.empty() && line.front() == '[' && close != std::string::npos) {
            const auto number = normalize_decimal_digits(line.substr(1,close-1));
            if (!number.empty() && std::all_of(number.begin(),number.end(),[](char c) { return c >= '0' && c <= '9'; })) {
                const auto value = strip_config_value(line.substr(close+1));
                if (!value.empty()) { const auto parsed = parse_configuration_integer(number); if (!parsed) throw std::runtime_error("Invalid subtitle line number"); result[*parsed] = value; }
            }
        }
        if (end == std::string::npos) break; start = end+1;
    }
    return result;
}
}
std::string translate_subtitle_segments(const std::string &encoded,const std::string &source,
    const std::string &target,const std::string &model,const SubtitleChat &chat,size_t max_words) {
    const auto segments = Json::parse(encoded); Json translated = Json::array();
    if (!segments.is_array()) throw std::invalid_argument("Subtitle segments must be an array");
    for (size_t first = 0; first < segments.size();) {
        size_t last = first, count = 0;
        while (last < segments.size()) { const auto next = words(segments[last].at("text").get<std::string>()); if (last > first && count+next > max_words) break; count += next; ++last; }
        std::string block; for (size_t i = first; i < last; ++i) { if (i != first) block += '\n'; block += '[' + std::to_string(i-first+1) + "] " + segments[i].at("text").get<std::string>(); }
        const auto prompt = "You are a professional subtitle translator. Translate the numbered subtitle lines from " + source + " to " + target + ".\nKeep translations concise and natural for subtitles.\nMaintain the exact same numbering format [N] for each line.\nReturn ONLY the translated lines with their numbers, nothing else.";
        const Json request{{"model",model},{"messages",Json::array({Json{{"role","system"},{"content",prompt}},Json{{"role","user"},{"content",block}}})}};
        const auto translations = numbered(Json::parse(chat(request.dump())));
        for (size_t i = first; i < last; ++i) {
            const auto &seg = segments[i]; const auto found = translations.find(std::to_string(i-first+1));
            translated.push_back(Json{{"id",seg.at("id")},{"start",seg.at("start")},{"end",seg.at("end")},{"text",found == translations.end() ? seg.at("text") : Json(found->second)}});
        }
        first = last;
    }
    return translated.dump();
}
std::string generate_native_subtitles(const std::string &encoded,const std::string &source,
    const std::string &targets,const std::string &format,const std::string &model,const SubtitleChat &chat) {
    const auto transcription = Json::parse(encoded); auto detected = transcription.value("language",Json());
    if (detected.is_null() || detected == "" || detected == false) detected = source.empty() ? "en" : source;
    const auto language = detected.get<std::string>(); const auto segments = transcription.at("segments").dump();
    std::vector<std::string> languages;
    for (size_t start = 0; start <= targets.size();) { const auto end = targets.find(',',start); languages.push_back(strip_config_value(targets.substr(start,end == std::string::npos ? end : end-start))); if (end == std::string::npos) break; start = end+1; }
    std::vector<std::future<std::string>> tasks;
    for (const auto &target : languages) tasks.push_back(std::async(std::launch::async,[&,target] { return target == language ? segments : translate_subtitle_segments(segments,language,target,model,chat); }));
    std::vector<std::string> translated; std::exception_ptr failure;
    for (auto &task : tasks) { try { translated.push_back(task.get()); } catch (...) { if (!failure) failure = std::current_exception(); translated.emplace_back(); } }
    if (failure) std::rethrow_exception(failure);
    Json outputs = Json::object();
    for (size_t i = 0; i < languages.size(); ++i) {
        if (format == "json") outputs[languages[i]] = Json::parse(translated[i]);
        else if (format == "srt" || format == "vtt") outputs[languages[i]] = stt_segments_subtitles(translated[i],format == "vtt");
    }
    return Json{{"source_language",detected},{"subtitles",outputs}}.dump();
}
}
