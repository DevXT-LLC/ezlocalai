#pragma once
#include <optional>
#include <string>
#include <vector>
namespace ezlocalai {
// Current CTTS text policy; limits count Unicode code points, as Python len().
std::string tts_number_to_words(const std::string &decimal);
std::string tts_normalize_english(const std::string &text);
std::string tts_clean_text(const std::string &text);
bool tts_contains_cyrillic(const std::string &text);
std::string tts_resolve_language(const std::string &language,const std::string &text);
std::string tts_prepare_text(const std::string &text,const std::string &language);
std::string tts_safe_file_stem(const std::optional<std::string> &value,const std::string &fallback="default");
std::string tts_cache_normalize_text(const std::string &text);
std::vector<std::string> tts_split_text(const std::string &text,double maximum=350,bool keep_short=true);
std::vector<std::string> tts_split_stream_text(const std::string &text,double maximum=280,std::optional<double> first={},std::optional<double> minimum={});
}
