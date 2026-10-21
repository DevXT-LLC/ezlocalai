#pragma once
#include "ezlocalai/whisper_transcription.hpp"
#include "whisper_test_data.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <map>
#include <unistd.h>
namespace transcription_test {
using namespace whisper_test;
using namespace ezlocalai;
using Json = nlohmann::json;
inline std::vector<Json> records(const fs::path &path) {
    std::istringstream input(read(path)); std::vector<Json> result;
    for (std::string line; std::getline(input,line);) if (!line.empty()) result.push_back(Json::parse(line));
    return result;
}
inline void compare_json(const Json &actual,const Json &expected,const std::string &path,double metric_tolerance = 1e-12) {
    if (actual.is_number() && expected.is_number()) {
        const bool metric = path.find("prob") != std::string::npos || path.find("score") != std::string::npos || path.find("all_language_probs") != std::string::npos;
        if (std::abs(actual.get<double>() - expected.get<double>()) <= (metric ? metric_tolerance : 1e-12)) return;
    } else if (actual.is_object() && expected.is_object() && actual.size() == expected.size()) {
        for (const auto &item : expected.items()) { expect(actual.contains(item.key()),path + " missing " + item.key()); compare_json(actual[item.key()],item.value(),path + "." + item.key(),metric_tolerance); } return;
    } else if (actual.is_array() && expected.is_array() && actual.size() == expected.size()) {
        for (size_t i = 0; i < expected.size(); ++i) compare_json(actual[i],expected[i],path + "[" + std::to_string(i) + "]",metric_tolerance); return;
    } else if (actual == expected) return;
    throw std::runtime_error(path + " actual=" + actual.dump() + " expected=" + expected.dump());
}
template<class T> void option(const Json &input,const char *key,T &value) { if (input.contains(key)) value = input.at(key).get<T>(); }
template<class T> void optional(const Json &input,const char *key,std::optional<T> &value) { if (input.contains(key)) { if (input.at(key).is_null()) value.reset(); else value = input.at(key).get<T>(); } }
inline WhisperTranscriptionOptions options(const Json &input,bool legacy) {
    WhisperTranscriptionOptions out;
#define FIELD(name) option(input,#name,out.name)
#define OPTIONAL(name) optional(input,#name,out.name)
    FIELD(beam_size); FIELD(best_of); FIELD(patience); FIELD(length_penalty); FIELD(repetition_penalty); FIELD(no_repeat_ngram_size);
    OPTIONAL(log_prob_threshold); OPTIONAL(no_speech_threshold); OPTIONAL(compression_ratio_threshold);
    FIELD(condition_on_previous_text); FIELD(prompt_reset_on_temperature); FIELD(temperatures);
    FIELD(suppress_blank); OPTIONAL(suppress_tokens); FIELD(without_timestamps); FIELD(word_timestamps); FIELD(multilingual);
    FIELD(max_initial_timestamp); FIELD(prepend_punctuations); FIELD(append_punctuations); OPTIONAL(max_new_tokens);
    OPTIONAL(hallucination_silence_threshold); OPTIONAL(language); FIELD(task); FIELD(vad_filter);
    OPTIONAL(language_detection_segments); OPTIONAL(language_detection_threshold);
#undef FIELD
#undef OPTIONAL
    for (auto item : {std::pair{"prefix",&out.prefix},std::pair{"hotwords",&out.hotwords}})
        if (input.contains(item.first) && !input.at(item.first).is_null()) *item.second = input.at(item.first).get<std::string>();
    if (input.contains("temperature")) { const auto &value = input.at("temperature"); out.temperatures = value.is_array() ? value.get<std::vector<double>>() : std::vector<double>{value.get<double>()}; }
    if (input.contains("initial_prompt") && !input.at("initial_prompt").is_null()) {
        const auto &value = input.at("initial_prompt"); if (value.is_string()) out.initial_prompt = value.get<std::string>(); else out.initial_prompt = value.get<std::vector<uint32_t>>();
    }
    if (input.contains("clip_timestamps")) { const auto &value = input.at("clip_timestamps"); if (value.is_string()) out.clip_timestamps = value.get<std::string>(); else out.clip_timestamps = value.get<std::vector<double>>(); }
    if (input.contains("chunk_length") && !input.at("chunk_length").is_null()) out.chunk_seconds = input.at("chunk_length").get<int64_t>();
    if (input.contains("vad_parameters") && !input.at("vad_parameters").is_null()) {
        auto vad = whisper_vad_options(legacy ? WhisperVadProfile::faster_whisper_1_0_3 : WhisperVadProfile::faster_whisper_1_2_0);
        const auto &params = input.at("vad_parameters"); option(params,"min_silence_duration_ms",vad.min_silence_duration_ms); option(params,"speech_pad_ms",vad.speech_pad_ms);
        option(params,"threshold",vad.threshold); optional(params,"neg_threshold",vad.neg_threshold); option(params,"min_speech_duration_ms",vad.min_speech_duration_ms); option(params,"max_speech_duration_s",vad.max_speech_duration_s); out.vad_options = vad;
    }
    return out;
}
inline WhisperBatchedOptions batch_options(Json input) {
    WhisperBatchedOptions out; const auto clips = input.value("clip_timestamps",Json()); input.erase("clip_timestamps");
    out.transcription = options(input,false); out.transcription.without_timestamps = input.value("without_timestamps",true); out.transcription.vad_filter = input.value("vad_filter",true);
    out.batch_size = input.value("batch_size",int64_t{8});
    if (!clips.is_null()) for (const auto &chunk : clips) out.clip_timestamps.push_back({chunk.at("start"),chunk.at("end")});
    return out;
}
inline Json segments(const std::vector<WhisperSegment> &input) {
    Json result = Json::array();
    for (const auto &segment : input) {
        Json words;
        if (segment.words) { words = Json::array(); for (const auto &word : *segment.words) words.push_back({{"start",word.start},{"end",word.end},{"word",word.word},{"probability",word.probability}}); }
        result.push_back({{"id",segment.id},{"seek",segment.seek},{"start",segment.start},{"end",segment.end},{"text",segment.text},{"tokens",segment.tokens},
            {"temperature",segment.temperature},{"avg_logprob",segment.avg_logprob},{"compression_ratio",segment.compression_ratio},{"no_speech_prob",segment.no_speech_prob},{"words",words}});
    }
    return result;
}
inline Json info(const WhisperTranscription &input) {
    return {{"language",input.language},{"language_probability",input.language_probability},{"duration",input.duration},{"duration_after_vad",input.duration_after_vad},
        {"all_language_probs",input.all_language_probs ? Json(*input.all_language_probs) : Json()}};
}
struct Temporary {
    fs::path path;
    Temporary() { auto pattern = (fs::temp_directory_path()/"ezlocalai-transcription-XXXXXX").string(); expect(mkdtemp(pattern.data()),"temporary tokenizer directory"); path = pattern; }
    ~Temporary() { std::error_code ignored; fs::remove_all(path,ignored); }
};
inline std::map<std::string,WhisperTokenizer> tokenizers(const fs::path &fixtures,const fs::path &directory) {
    std::map<std::string,WhisperTokenizer> result;
    for (const auto &row : rows(fixtures/"whisper_tokenizer_cases.tsv")) if (row[0] == "H") {
        const auto compressed = read(fixtures/"whisper_tokenizer_assets"/row[3]); const auto size = std::stoul(row[4]); std::string raw(size + 1,'\0'); uLongf length = raw.size();
        expect(uncompress(reinterpret_cast<Bytef *>(raw.data()),&length,reinterpret_cast<const Bytef *>(compressed.data()),compressed.size()) == Z_OK && length == size,"compressed tokenizer bytes");
        raw.resize(size); expect(hash(raw) == row[2],"tokenizer identity"); const auto path = directory/(row[1] + ".json");
        std::ofstream file(path,std::ios::binary); file.write(raw.data(),raw.size()); file.close(); result.emplace(row[1],WhisperTokenizer(path.string(),row[1] == "multi"));
    }
    expect(result.size() == 2,"both tokenizer profiles"); return result;
}
}
