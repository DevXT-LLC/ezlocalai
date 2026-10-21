// Ported from faster-whisper 1.0.3/1.2.0. See licenses/faster-whisper.txt.
#include "ezlocalai/whisper_transcription.hpp"
#include "ezlocalai/configuration.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <zlib.h>

namespace ezlocalai {
namespace {
bool old(WhisperTranscriptionProfile profile) { return profile == WhisperTranscriptionProfile::faster_whisper_1_0_3; }
WhisperFeatureConfiguration feature_configuration(const WhisperEngineOperations &engine,std::optional<int64_t> chunk = {}) {
    auto config = engine.feature_configuration.value_or(WhisperFeatureConfiguration{});
    if (!engine.feature_configuration) config.feature_size = engine.bands;
    if (chunk) config.chunk_length = *chunk;
    if (config.sampling_rate <= 0 || config.hop_length <= 0 || config.n_fft <= 0 || config.feature_size <= 0)
        throw std::invalid_argument("Invalid Whisper feature dimensions");
    return config;
}
WhisperFeatures extract_features(const std::vector<float> &audio,const WhisperFeatureConfiguration &config,bool legacy) {
    return whisper_log_mel(audio,config.feature_size,legacy ? config.chunk_length * config.sampling_rate : 160,
        config.sampling_rate,config.n_fft,config.hop_length,legacy ? WhisperFeatureProfile::faster_whisper_1_0_3 : WhisperFeatureProfile::faster_whisper_1_2_0);
}
int64_t integer_round(double value) {
    if (!std::isfinite(value) || value >= static_cast<double>(INT64_MAX) || value < static_cast<double>(INT64_MIN)) throw std::invalid_argument("Invalid Whisper frame timestamp");
    const auto lower = std::floor(value), fraction = value - lower;
    return static_cast<int64_t>(lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower,2) != 0)));
}
double round_time(double value,int precision = 2) {
    if (!std::isfinite(value)) return value;
    char data[512]; const auto result = std::to_chars(data,data + sizeof(data),value,std::chars_format::fixed,precision);
    if (result.ec != std::errc{}) throw std::runtime_error("Invalid Whisper word timestamp");
    return std::stod(std::string(data,result.ptr));
}
// Alignment times are NumPy float64 scalars: their round method scales before
// ties-to-even rounding, unlike Python float's decimal rounding above.
double round_alignment_time(double value) { return std::nearbyint(value * 100) / 100; }
double compression_ratio(const std::string &text) {
    uLongf size = compressBound(text.size()); std::vector<Bytef> output(size);
    if (compress2(output.data(),&size,reinterpret_cast<const Bytef *>(text.data()),text.size(),Z_DEFAULT_COMPRESSION) != Z_OK) throw std::runtime_error("Whisper text compression failed");
    return static_cast<double>(text.size()) / size;
}
int64_t slice_index(int64_t index,int64_t length) { return std::clamp(index < 0 ? length + index : index,int64_t{0},length); }
WhisperFeatures slice_features(const WhisperFeatures &features,int64_t start,int64_t end,size_t padding = 0) {
    const auto first = slice_index(start,features.frames), last = slice_index(end,features.frames);
    const auto count = std::max<int64_t>(0,last - first);
    return whisper_feature_window(features,first,count,padding ? padding : count);
}
std::vector<double> clips(const WhisperTranscriptionOptions &options) {
    if (const auto *values = std::get_if<std::vector<double>>(&options.clip_timestamps)) return *values;
    const auto &text = std::get<std::string>(options.clip_timestamps); std::vector<double> values;
    if (text.empty()) return values;
    for (size_t at = 0;;) {
        const auto end = text.find(',',at); const auto item = text.substr(at,end == std::string::npos ? end : end - at); double value;
        if (!parse_configuration_float(item,&value)) throw std::invalid_argument("could not convert string to float: '" + item + "'");
        values.push_back(value); if (end == std::string::npos) break; at = end + 1;
    }
    return values;
}
std::string language_code(const std::string &token) { return token.size() >= 4 ? token.substr(2,token.size() - 4) : ""; }
struct AlignedWord { std::string word; std::vector<uint32_t> tokens; double start, end, probability; };
std::vector<AlignedWord> alignment_words(const WhisperAlignment &result,const WhisperTokenizer &tokenizer,const std::vector<uint32_t> &tokens,const WhisperFeatureConfiguration &config) {
    auto input = tokens; input.push_back(tokenizer.eot()); const auto words = tokenizer.split_to_word_tokens(input);
    if (words.size() <= 1) return {};
    std::vector<double> jumps;
    for (size_t i = 0; i < result.indices.size(); ++i) if (!i || result.indices[i].first != result.indices[i - 1].first) jumps.push_back(result.indices[i].second / double(config.tokens_per_second()));
    std::vector<AlignedWord> output; size_t begin = 0;
    for (size_t i = 0; i + 1 < words.size(); ++i) {
        const auto end = begin + words[i].tokens.size(); const auto available = std::min(end,result.probabilities.size());
        double sum = 0; for (size_t j = begin; j < available; ++j) sum += result.probabilities[j];
        const auto probability = available > begin ? sum / (available - begin) : std::numeric_limits<double>::quiet_NaN();
        output.push_back({words[i].word,words[i].tokens,jumps.at(begin),jumps.at(end),probability}); begin = end;
    }
    return output;
}
void merge_punctuation(std::vector<AlignedWord> &words,const std::string &prepended,const std::string &appended) {
    if (words.empty()) return;
    int64_t j = words.size() - 1;
    for (int64_t i = static_cast<int64_t>(words.size()) - 2; i >= 0; --i) {
        auto &previous = words[i]; auto &following = words[j];
        if (!previous.word.empty() && previous.word.front() == ' ' && prepended.find(strip_config_value(previous.word)) != std::string::npos) {
            following.word = previous.word + following.word; following.tokens.insert(following.tokens.begin(),previous.tokens.begin(),previous.tokens.end()); previous.word.clear(); previous.tokens.clear();
        } else j = i;
    }
    size_t i = 0;
    for (size_t next = 1; next < words.size(); ++next) {
        auto &previous = words[i]; auto &following = words[next];
        if ((previous.word.empty() || previous.word.back() != ' ') && appended.find(following.word) != std::string::npos) {
            previous.word += following.word; previous.tokens.insert(previous.tokens.end(),following.tokens.begin(),following.tokens.end()); following.word.clear(); following.tokens.clear();
        } else i = next;
    }
}
double add_aligned_words(std::vector<WhisperSegment> &segments,const WhisperTokenizer &tokenizer,
    std::vector<AlignedWord> alignment,const WhisperTranscriptionOptions &options,bool legacy,double last_speech,const WhisperFeatureConfiguration &config) {
    if (segments.empty()) return last_speech;
    std::vector<size_t> counts;
    for (const auto &segment : segments) counts.push_back(std::count_if(segment.tokens.begin(),segment.tokens.end(),[&](auto id) { return id < tokenizer.eot(); }));
    std::vector<double> durations; for (const auto &word : alignment) if (word.end - word.start != 0) durations.push_back(word.end - word.start);
    std::sort(durations.begin(),durations.end()); double median = 0;
    if (!durations.empty()) { const auto middle = durations.size() / 2; median = durations.size() % 2 ? durations[middle] : (durations[middle - 1] + durations[middle]) / 2; }
    median = std::min(0.7,median); const auto maximum = median * 2;
    if (!durations.empty()) for (size_t i = 1; i < alignment.size(); ++i) if (alignment[i].end - alignment[i].start > maximum) {
        if (std::string(".。!！?？").find(alignment[i].word) != std::string::npos) alignment[i].end = alignment[i].start + maximum;
        else if (std::string(".。!！?？").find(alignment[i - 1].word) != std::string::npos) alignment[i].start = alignment[i].end - maximum;
    }
    merge_punctuation(alignment,options.prepend_punctuations,options.append_punctuations);
    const auto offset = legacy ? segments[0].seek * double(config.hop_length) / config.sampling_rate : segments[0].seek / double(config.frames_per_second());
    size_t position = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        auto &segment = segments[i]; size_t saved = 0; std::vector<WhisperWord> words;
        while (position < alignment.size() && saved < counts[i]) {
            const auto &word = alignment[position++];
            if (!word.word.empty()) words.push_back({round_alignment_time(offset + word.start),round_alignment_time(offset + word.end),word.word,word.probability});
            saved += word.tokens.size();
        }
        if (!words.empty()) {
            if (words[0].end - last_speech > median * 4 && (words[0].end - words[0].start > maximum || (words.size() > 1 && words[1].end - words[0].start > maximum * 2))) {
                if (words.size() > 1 && words[1].end - words[1].start > maximum) { const auto boundary = std::max(words[1].end / 2,words[1].end - maximum); words[0].end = words[1].start = boundary; }
                words[0].start = std::max(0.0,words[0].end - maximum);
            }
            if (segment.start < words[0].end && segment.start - 0.5 > words[0].start) words[0].start = std::max(0.0,std::min(words[0].end - median,segment.start));
            else segment.start = words[0].start;
            if (segment.end > words.back().start && segment.end + 0.5 < words.back().end) words.back().end = std::max(words.back().start + median,segment.end);
            else segment.end = words.back().end;
            last_speech = segment.end;
        }
        segment.words = std::move(words);
    }
    return last_speech;
}
void add_words(std::vector<WhisperSegment> &segments,const WhisperTokenizer &tokenizer,const WhisperEncoded &encoded,
    size_t frames,const WhisperTranscriptionOptions &options,const WhisperEngineOperations &engine,bool legacy,double last_speech) {
    if (segments.empty()) return;
    std::vector<uint32_t> text; for (const auto &segment : segments) for (auto token : segment.tokens) if (token < tokenizer.eot()) text.push_back(token);
    const auto config = feature_configuration(engine);
    auto alignment = text.empty() && legacy ? std::vector<AlignedWord>{} : alignment_words(engine.align(encoded,tokenizer.sot_sequence(),text,frames,7),tokenizer,text,config);
    add_aligned_words(segments,tokenizer,std::move(alignment),options,legacy,last_speech,config);
}
std::optional<double> last_end(const std::vector<WhisperSegment> &segments) {
    for (auto i = segments.rbegin(); i != segments.rend(); ++i) if (i->words && !i->words->empty()) return i->words->back().end;
    return segments.empty() ? std::optional<double>{} : segments.back().end;
}
bool anomaly(const WhisperSegment *segment) {
    if (!segment || !segment->words || segment->words->empty()) return false;
    const std::string punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"; size_t count = 0; double score = 0;
    for (const auto &word : *segment->words) {
        if (punctuation.find(word.word) != std::string::npos) continue;
        if (count++ == 8) { --count; break; }
        const auto duration = word.end - word.start;
        if (word.probability < 0.15) score += 1;
        if (duration < 0.133) score += (0.133 - duration) * 15;
        if (duration > 2) score += duration - 2;
    }
    return score >= 3 || score + 0.01 >= count;
}
const WhisperSegment *next_words(const std::vector<WhisperSegment> &segments,size_t begin = 0) {
    for (size_t i = begin; i < segments.size(); ++i) if (segments[i].words && !segments[i].words->empty()) return &segments[i]; return nullptr;
}
}
WhisperSegmentSplit whisper_split_segments(const std::vector<uint32_t> &tokens,uint32_t timestamp_begin,double time_offset,int64_t segment_size,double segment_duration,int64_t seek) {
    WhisperSegmentSplit result{{},seek,tokens.size() >= 2 && tokens[tokens.size() - 2] < timestamp_begin && tokens.back() >= timestamp_begin};
    std::vector<size_t> slices; for (size_t i = 1; i < tokens.size(); ++i) if (tokens[i] >= timestamp_begin && tokens[i - 1] >= timestamp_begin) slices.push_back(i);
    if (!slices.empty()) {
        if (result.single_timestamp_ending) slices.push_back(tokens.size()); size_t previous = 0;
        for (auto end : slices) {
            WhisperSegment segment; segment.seek = seek; segment.tokens.assign(tokens.begin() + previous,tokens.begin() + end);
            segment.start = time_offset + (static_cast<int64_t>(segment.tokens.front()) - timestamp_begin) * 0.02;
            segment.end = time_offset + (static_cast<int64_t>(segment.tokens.back()) - timestamp_begin) * 0.02;
            result.segments.push_back(std::move(segment)); previous = end;
        }
        result.seek += result.single_timestamp_ending ? segment_size : (static_cast<int64_t>(tokens[previous - 1]) - timestamp_begin) * 2;
    } else {
        double duration = segment_duration; std::optional<uint32_t> last;
        for (auto token : tokens) if (token >= timestamp_begin) last = token;
        if (last && *last != timestamp_begin) duration = (static_cast<int64_t>(*last) - timestamp_begin) * 0.02;
        WhisperSegment segment; segment.seek = seek; segment.start = time_offset; segment.end = time_offset + duration; segment.tokens = tokens;
        result.segments.push_back(std::move(segment)); result.seek += segment_size;
    }
    return result;
}
WhisperDecodeResult whisper_decode_with_fallback(const WhisperEncoded &encoded,const std::vector<uint32_t> &prompt,const WhisperTokenizer &tokenizer,
    const WhisperTranscriptionOptions &options,const WhisperEngineOperations &engine) {
    const int64_t length = options.max_new_tokens ? static_cast<int64_t>(prompt.size()) + *options.max_new_tokens : 448;
    if (length > 448) throw std::invalid_argument("The length of the prompt is " + std::to_string(prompt.size()) + ", and the `max_new_tokens` " + std::to_string(length - prompt.size()) + ". Thus, the combined length of the prompt and `max_new_tokens` is: " + std::to_string(length) + ". This exceeds the `max_length` of the Whisper model: 448. You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, so that their combined length is less that 448.");
    std::vector<WhisperDecodeResult> all, below;
    for (double temperature : options.temperatures) {
        ezl_whisper_options request; request.max_length = length; request.length_penalty = options.length_penalty; request.repetition_penalty = options.repetition_penalty;
        request.no_repeat_ngram_size = options.no_repeat_ngram_size; request.suppress_blank = options.suppress_blank;
        request.suppress_tokens = options.suppress_tokens ? options.suppress_tokens->data() : nullptr;
        request.suppress_count = options.suppress_tokens ? options.suppress_tokens->size() : 0; request.suppress_tokens_is_set = true;
        request.max_initial_timestamp_index = integer_round(options.max_initial_timestamp / 0.02);
        if (temperature > 0) { request.beam_size = 1; request.num_hypotheses = options.best_of; request.sampling_topk = 0; request.sampling_temperature = temperature; }
        else { request.beam_size = options.beam_size; request.patience = options.patience; }
        WhisperDecodeResult decoded; decoded.result = engine.generate(encoded,prompt,request); decoded.temperature = temperature;
        const auto count = decoded.result.tokens.size(); decoded.avg_logprob = decoded.result.score * std::pow(static_cast<double>(count),options.length_penalty) / (count + 1);
        decoded.compression_ratio = compression_ratio(strip_config_value(tokenizer.decode(decoded.result.tokens))); all.push_back(decoded);
        bool fallback = false;
        if (options.compression_ratio_threshold) { if (decoded.compression_ratio > *options.compression_ratio_threshold) fallback = true; else below.push_back(decoded); }
        if (options.log_prob_threshold && decoded.avg_logprob < *options.log_prob_threshold) fallback = true;
        if (options.no_speech_threshold && decoded.result.no_speech_prob > *options.no_speech_threshold && options.log_prob_threshold && decoded.avg_logprob < *options.log_prob_threshold) fallback = false;
        if (!fallback) return decoded;
    }
    if (all.empty()) throw std::invalid_argument("max() iterable argument is empty");
    const auto &choices = below.empty() ? all : below; size_t best = 0;
    for (size_t i = 1; i < choices.size(); ++i) if (choices[i].avg_logprob > choices[best].avg_logprob) best = i;
    auto selected = choices[best]; selected.temperature = options.temperatures.back(); return selected;
}
std::vector<WhisperSegment> whisper_generate_segments(const WhisperFeatures &features,WhisperTokenizer tokenizer,const WhisperTranscriptionOptions &options,
    const WhisperEngineOperations &engine,WhisperTranscriptionProfile profile,WhisperEncoded encoded) {
    const bool legacy = old(profile); const auto config = feature_configuration(engine,options.chunk_seconds);
    const int64_t maximum_frames = config.maximum_frames(), fps = config.frames_per_second(); const double frame_time = config.time_per_frame();
    if (maximum_frames <= 0) throw std::invalid_argument("Invalid Whisper chunk length");
    const int64_t content_frames = static_cast<int64_t>(features.frames) - (legacy ? maximum_frames : 1);
    const auto duration = content_frames * frame_time; const auto times = clips(options); std::vector<int64_t> points;
    for (auto time : times) points.push_back(integer_round(time * fps)); if (points.empty()) points.push_back(0); if (points.size() % 2) points.push_back(content_frames);
    size_t clip = 0, prompt_reset = 0; int64_t seek = points[0], index = 0; double last_speech = 0;
    std::vector<uint32_t> all_tokens; std::vector<WhisperSegment> output;
    if (options.initial_prompt) {
        if (const auto *text = std::get_if<std::string>(&*options.initial_prompt)) all_tokens = tokenizer.encode(" " + strip_config_value(*text));
        else all_tokens = std::get<std::vector<uint32_t>>(*options.initial_prompt);
    }
    while (clip < points.size() / 2) {
        const auto start = points[clip * 2], end = std::min(points[clip * 2 + 1],content_frames);
        if (seek < start) seek = start;
        if (seek >= end) { if (++clip < points.size() / 2) seek = points[clip * 2]; continue; }
        const double offset = seek * frame_time, window_end = (seek + maximum_frames) * frame_time;
        const auto size = std::min({maximum_frames,content_frames - seek,end - seek});
        const auto window = slice_features(features,seek,seek + size,legacy ? maximum_frames : 3000);
        std::vector<uint32_t> previous(all_tokens.begin() + prompt_reset,all_tokens.end()); std::vector<uint32_t> prompt;
        if (legacy) prompt = whisper_prompt(tokenizer,previous,options.without_timestamps,seek == 0 ? options.prefix : "",options.hotwords);
        if (seek > 0 || !encoded) encoded = engine.encode(window);
        if (!legacy && options.multilingual) {
            const auto detected = engine.detect_language(encoded); if (detected.empty()) throw std::out_of_range("Missing detected Whisper language");
            tokenizer = tokenizer.configured(true,options.task,language_code(detected[0].first));
        }
        if (!legacy) prompt = whisper_prompt(tokenizer,previous,options.without_timestamps,seek == 0 ? options.prefix : "",options.hotwords);
        const auto decoded = whisper_decode_with_fallback(encoded,prompt,tokenizer,options,engine);
        if (options.no_speech_threshold) {
            bool skip = decoded.result.no_speech_prob > *options.no_speech_threshold;
            if (options.log_prob_threshold && decoded.avg_logprob > *options.log_prob_threshold) skip = false;
            if (skip) { seek += size; continue; }
        }
        const auto previous_seek = seek; auto split = whisper_split_segments(decoded.result.tokens,tokenizer.timestamp_begin(),offset,size,size * frame_time,seek); seek = split.seek;
        auto &segments = split.segments;
        if (options.word_timestamps) {
            add_words(segments,tokenizer,encoded,size,options,engine,legacy,last_speech);
            if (!split.single_timestamp_ending) { const auto last = last_end(segments); if (last && *last > offset) seek = integer_round(*last * fps); }
            if (options.hallucination_silence_threshold) {
                const auto threshold = *options.hallucination_silence_threshold; const auto *first = next_words(segments);
                if (first && anomaly(first) && first->start - offset > threshold) { seek = previous_seek + integer_round((first->start - offset) * fps); continue; }
                auto previous_end = last_speech;
                for (size_t i = 0; i < segments.size(); ++i) {
                    auto &segment = segments[i]; if (!segment.words || segment.words->empty()) continue;
                    if (anomaly(&segment)) {
                        const auto *next = next_words(segments,i + 1); const auto next_start = next ? next->words->front().start : offset + size * frame_time;
                        const bool before = segment.start - previous_end > threshold || segment.start < threshold || segment.start - offset < 2;
                        const bool after = next_start - segment.end > threshold || anomaly(next) || window_end - segment.end < 2;
                        if (before && after) { seek = integer_round(std::max(offset + 1,segment.start) * fps); if (duration - segment.end < threshold) seek = content_frames; segments.erase(segments.begin() + i,segments.end()); break; }
                    }
                    previous_end = segment.end;
                }
            }
            const auto last = last_end(segments); if (last) last_speech = *last;
        }
        for (auto &segment : segments) {
            segment.text = tokenizer.decode(segment.tokens); if (segment.start == segment.end || strip_config_value(segment.text).empty()) continue;
            all_tokens.insert(all_tokens.end(),segment.tokens.begin(),segment.tokens.end()); segment.id = ++index; segment.seek = legacy ? seek : previous_seek;
            segment.temperature = decoded.temperature; segment.avg_logprob = decoded.avg_logprob; segment.compression_ratio = decoded.compression_ratio; segment.no_speech_prob = decoded.result.no_speech_prob;
            output.push_back(std::move(segment));
        }
        if (!options.condition_on_previous_text || decoded.temperature > options.prompt_reset_on_temperature) prompt_reset = all_tokens.size();
    }
    return output;
}
void whisper_restore_speech_timestamps(std::vector<WhisperSegment> &segments,const std::vector<WhisperSpeechChunk> &chunks,WhisperTranscriptionProfile profile,int64_t sampling_rate) {
    const WhisperSpeechTimestampsMap map(chunks,sampling_rate,2,old(profile) ? WhisperVadProfile::faster_whisper_1_0_3 : WhisperVadProfile::faster_whisper_1_2_0);
    for (auto &segment : segments) {
        if (segment.words && !segment.words->empty()) {
            for (auto &word : *segment.words) { const auto chunk = map.chunk_index((word.start + word.end) / 2); word.start = map.original_time(word.start,chunk); word.end = map.original_time(word.end,chunk); }
            segment.start = segment.words->front().start; segment.end = segment.words->back().end;
        } else { segment.start = map.original_time(segment.start); segment.end = map.original_time(segment.end,{},true); }
    }
}
WhisperTranscription whisper_transcribe(const std::vector<float> &input,const WhisperTokenizer &base,WhisperTranscriptionOptions options,
    const WhisperEngineOperations &engine,WhisperTranscriptionProfile profile,const std::function<WhisperVadAudio(const std::vector<float> &,const WhisperVadOptions &)> &vad) {
    const bool legacy = old(profile); const auto config = feature_configuration(engine,options.chunk_seconds);
    WhisperTranscription out; out.duration = input.size() / double(config.sampling_rate);
    if (!engine.multilingual) options.multilingual = false;
    auto audio = input; std::vector<WhisperSpeechChunk> speech;
    const auto *clip_string = std::get_if<std::string>(&options.clip_timestamps);
    if (options.vad_filter && clip_string && *clip_string == "0") {
        if (!vad) throw std::runtime_error("Native Whisper VAD is not configured");
        const auto settings = options.vad_options.value_or(whisper_vad_options(legacy ? WhisperVadProfile::faster_whisper_1_0_3 : WhisperVadProfile::faster_whisper_1_2_0));
        options.vad_options = settings; auto filtered = vad(audio,settings); audio = std::move(filtered.audio); speech = std::move(filtered.chunks);
    }
    out.duration_after_vad = audio.size() / double(config.sampling_rate);
    if (config.chunk_length < 0) throw std::invalid_argument("index can't contain negative values");
    const auto features = extract_features(audio,config,legacy);
    WhisperEncoded encoded;
    if (!options.language && engine.multilingual) {
        const auto times = clips(options); if (times.empty()) throw std::invalid_argument("Missing initial language-detection timestamp");
        const int64_t maximum_frames = config.maximum_frames(); if (maximum_frames <= 0) throw std::invalid_argument("Invalid Whisper chunk length");
        const auto content_frames = static_cast<int64_t>(features.frames) - (legacy ? maximum_frames : 1);
        const auto initial = times[0] * config.frames_per_second() < content_frames ? static_cast<int64_t>(times[0] * config.frames_per_second()) : 0;
        auto count = options.language_detection_segments;
        if (legacy && (!count || *count < 1)) count = 1;
        if (!count) throw std::invalid_argument("Invalid language detection segment count");
        std::vector<std::pair<std::string,std::vector<double>>> votes;
        bool accepted = false; WhisperFeatures detection;
        if (!legacy) { detection = slice_features(features,initial,features.frames); detection = slice_features(detection,0,*count * maximum_frames); }
        const int64_t end = legacy ? std::min(initial + maximum_frames * *count,content_frames) : detection.frames;
        for (int64_t seek = legacy ? initial : 0; seek < end; seek += maximum_frames) {
            const auto window = legacy ? slice_features(features,seek,seek + maximum_frames) : slice_features(detection,seek,seek + maximum_frames,3000);
            auto current = engine.encode(window); if (legacy) encoded = current;
            const auto result = engine.detect_language(current); if (result.empty()) throw std::out_of_range("Missing detected Whisper language");
            WhisperLanguages languages; for (const auto &item : result) languages.emplace_back(language_code(item.first),item.second); out.all_language_probs = languages;
            out.language = languages[0].first; out.language_probability = languages[0].second;
            if (!options.language_detection_threshold && !legacy) throw std::invalid_argument("Invalid language detection threshold");
            if (!options.language_detection_threshold || out.language_probability > *options.language_detection_threshold) { accepted = true; break; }
            auto found = std::find_if(votes.begin(),votes.end(),[&](const auto &item) { return item.first == out.language; });
            if (found == votes.end()) votes.push_back({out.language,{out.language_probability}}); else found->second.push_back(out.language_probability);
        }
        if (!accepted) {
            if (votes.empty()) throw std::invalid_argument("max() iterable argument is empty"); size_t best = 0;
            for (size_t i = 1; i < votes.size(); ++i) if (votes[i].second.size() > votes[best].second.size()) best = i;
            out.language = votes[best].first; out.language_probability = *std::max_element(votes[best].second.begin(),votes[best].second.end());
        }
    } else out.language = engine.multilingual ? options.language.value_or("en") : "en";
    auto tokenizer = base.configured(engine.multilingual,options.task,out.language);
    if (options.suppress_tokens && !options.suppress_tokens->empty()) options.suppress_tokens = whisper_suppressed_tokens(tokenizer,options.suppress_tokens,legacy ? WhisperTokenizerProfile::faster_whisper_1_0_3 : WhisperTokenizerProfile::faster_whisper_1_2_0);
    out.options = options; out.segments = whisper_generate_segments(features,std::move(tokenizer),options,engine,profile,encoded);
    if (!speech.empty()) whisper_restore_speech_timestamps(out.segments,speech,profile,config.sampling_rate);
    return out;
}

std::vector<WhisperSegment> WhisperBatchedPipeline::generate_segments(const std::vector<WhisperFeatures> &features,const WhisperTokenizer &tokenizer,
    const std::vector<WhisperAudioChunk> &chunks,const WhisperTranscriptionOptions &options,int64_t batch_size) {
    return generate_segments_impl(features,tokenizer,chunks,options,batch_size,nullptr);
}
std::vector<WhisperSegment> WhisperBatchedPipeline::generate_segments_impl(const std::vector<WhisperFeatures> &features,const WhisperTokenizer &tokenizer,
    const std::vector<WhisperAudioChunk> &chunks,const WhisperTranscriptionOptions &options,int64_t batch_size,const std::vector<WhisperSpeechChunk> *speech) {
    const auto config = feature_configuration(engine_); const auto fps = config.frames_per_second();
    if (batch_size == 0) throw std::invalid_argument("range() arg 3 must not be zero");
    std::vector<WhisperSegment> output; int64_t index = 0;
    for (size_t begin = 0; batch_size > 0 && begin < features.size(); begin += batch_size) {
        const auto end = std::min(features.size(),begin + static_cast<size_t>(batch_size));
        std::vector<uint32_t> previous;
        if (options.initial_prompt) {
            const auto *text = std::get_if<std::string>(&*options.initial_prompt);
            if (!text) throw std::invalid_argument("TextInputSequence must be str");
            previous = tokenizer.encode(*text);
        }
        const auto prompt = whisper_prompt(tokenizer,previous,options.without_timestamps,{},options.hotwords);
        const int64_t length = options.max_new_tokens ? static_cast<int64_t>(prompt.size()) + *options.max_new_tokens : 448;
        if (length > 448) throw std::invalid_argument("The length of the prompt is " + std::to_string(prompt.size()) + ", and the `max_new_tokens` " + std::to_string(length - prompt.size()) + ". Thus, the combined length of the prompt and `max_new_tokens` is: " + std::to_string(length) + ". This exceeds the `max_length` of the Whisper model: 448. You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, so that their combined length is less that 448.");
        const auto encoded = engine_.encode_batch(std::vector<WhisperFeatures>(features.begin() + begin,features.begin() + end));
        std::vector<std::vector<uint32_t>> prompts(end - begin,prompt);
        if (options.multilingual) {
            const auto languages = engine_.detect_language_batch(encoded);
            const auto language = tokenizer.sot_sequence().size() > 1 ? tokenizer.token_id("<|" + tokenizer.language_code() + "|>") : std::optional<uint32_t>{};
            const auto position = language ? std::find(prompt.begin(),prompt.end(),*language) : prompt.end();
            if (position == prompt.end()) throw std::invalid_argument(language ? std::to_string(*language) + " is not in list" : "None is not in list");
            for (size_t i = 0; i < languages.size(); ++i) prompts.at(i)[position - prompt.begin()] = tokenizer.token_id(languages.at(i).at(0).first).value();
        }
        if (options.temperatures.empty()) throw std::out_of_range("list index out of range");
        ezl_whisper_options request; request.beam_size = options.beam_size; request.patience = options.patience;
        request.length_penalty = options.length_penalty; request.repetition_penalty = options.repetition_penalty;
        request.no_repeat_ngram_size = options.no_repeat_ngram_size; request.max_length = length;
        request.suppress_blank = options.suppress_blank; request.suppress_tokens = options.suppress_tokens ? options.suppress_tokens->data() : nullptr;
        request.suppress_count = options.suppress_tokens ? options.suppress_tokens->size() : 0; request.suppress_tokens_is_set = true; request.sampling_temperature = options.temperatures[0];
        const auto generated = engine_.generate_batch(encoded,prompts,request);
        std::vector<std::vector<WhisperSegment>> groups; std::vector<size_t> frames; std::vector<std::vector<uint32_t>> text;
        for (size_t i = 0; i < generated.size() && begin + i < chunks.size(); ++i) {
            const auto &chunk = chunks[begin + i]; const auto &result = generated[i];
            const auto size = static_cast<int64_t>(std::ceil(chunk.duration) * fps); frames.push_back(size);
            auto split = whisper_split_segments(result.tokens,tokenizer.timestamp_begin(),chunk.offset,size,chunk.duration,0);
            std::vector<uint32_t> words;
            for (auto &segment : split.segments) {
                segment.text = tokenizer.decode(segment.tokens); segment.seek = static_cast<int64_t>(chunk.offset * fps);
                segment.avg_logprob = result.score * std::pow(static_cast<double>(result.tokens.size()),options.length_penalty) / (result.tokens.size() + 1);
                segment.no_speech_prob = result.no_speech_prob; segment.compression_ratio = compression_ratio(segment.text);
                segment.temperature = options.temperatures[0];
                for (auto token : segment.tokens) if (token < tokenizer.eot()) words.push_back(token);
            }
            groups.push_back(std::move(split.segments)); text.push_back(std::move(words));
        }
        if (options.word_timestamps) {
            const auto alignments = engine_.align_batch(encoded,tokenizer.sot_sequence(),text,frames,7);
            auto last = last_speech_timestamp_;
            for (size_t i = 0; i < groups.size(); ++i) last = add_aligned_words(groups[i],tokenizer,alignment_words(alignments.at(i),tokenizer,text[i],config),options,false,last,config);
            last_speech_timestamp_ = last;
        }
        for (auto &group : groups) {
            for (auto &segment : group) { segment.id = ++index; segment.start = round_time(segment.start,3); segment.end = round_time(segment.end,3); }
            if (speech) whisper_restore_speech_timestamps(group,*speech,WhisperTranscriptionProfile::faster_whisper_1_2_0,config.sampling_rate);
            output.insert(output.end(),std::make_move_iterator(group.begin()),std::make_move_iterator(group.end()));
        }
    }
    last_speech_timestamp_ = 0; return output;
}
WhisperTranscription WhisperBatchedPipeline::transcribe(const std::vector<float> &audio,const WhisperTokenizer &base,WhisperBatchedOptions requested,
    const std::function<std::vector<WhisperSpeechChunk>(const std::vector<float> &,const WhisperVadOptions &)> &vad) {
    auto options = std::move(requested.transcription); const auto config = feature_configuration(engine_);
    WhisperTranscription out; out.duration = audio.size() / double(config.sampling_rate);
    if (!engine_.multilingual) options.multilingual = false;
    const auto chunk_seconds = options.chunk_seconds && *options.chunk_seconds ? *options.chunk_seconds : config.chunk_length;
    std::vector<WhisperSpeechChunk> speech;
    if (requested.clip_timestamps.empty()) {
        if (options.vad_filter) {
            if (!vad) throw std::runtime_error("Native Whisper VAD is not configured");
            auto settings = options.vad_options.value_or(whisper_vad_options(WhisperVadProfile::faster_whisper_1_2_0));
            if (!options.vad_options) settings.min_silence_duration_ms = 160;
            if (!options.vad_options || requested.vad_options_from_dictionary) settings.max_speech_duration_s = chunk_seconds;
            options.vad_options = settings; speech = vad(audio,settings);
        } else if (out.duration < chunk_seconds) speech.push_back({0,static_cast<int64_t>(audio.size())});
        else throw std::runtime_error("No clip timestamps found. Set 'vad_filter' to True or provide 'clip_timestamps'.");
    } else for (const auto &chunk : requested.clip_timestamps) speech.push_back({static_cast<int64_t>(chunk.start * config.sampling_rate),static_cast<int64_t>(chunk.end * config.sampling_rate)});
    const auto chunks = whisper_collect_chunks(audio,speech,16000,chunk_seconds);
    int64_t kept = 0; for (const auto &chunk : speech) kept += chunk.end - chunk.start; out.duration_after_vad = kept / double(config.sampling_rate);
    std::vector<WhisperFeatures> features;
    if (kept) for (const auto &chunk : chunks) { const auto mel = extract_features(chunk.audio,config,false); features.push_back(slice_features(mel,0,static_cast<int64_t>(mel.frames) - 1)); }
    if (!options.language && engine_.multilingual) {
        WhisperFeatures combined; combined.bands = config.feature_size; combined.frames = 1;
        for (const auto &feature : features) combined.frames += feature.frames;
        combined.values.assign(combined.bands * combined.frames,-1.5f);
        size_t offset = 0;
        for (const auto &feature : features) { for (size_t band = 0; band < combined.bands; ++band) std::copy_n(feature.values.begin() + band * feature.frames,feature.frames,combined.values.begin() + band * combined.frames + offset); offset += feature.frames; }
        if (!options.language_detection_segments) throw std::invalid_argument("Invalid language detection segment count");
        combined = slice_features(combined,0,*options.language_detection_segments * config.maximum_frames());
        std::vector<std::pair<std::string,std::vector<double>>> votes; bool accepted = false;
        for (size_t seek = 0; seek < combined.frames; seek += config.maximum_frames()) {
            const auto encoded = engine_.encode(slice_features(combined,seek,seek + config.maximum_frames(),3000)); const auto detected = engine_.detect_language(encoded);
            WhisperLanguages languages; for (const auto &item : detected) languages.emplace_back(language_code(item.first),item.second); out.all_language_probs = languages;
            out.language = languages.at(0).first; out.language_probability = languages[0].second;
            if (!options.language_detection_threshold) throw std::invalid_argument("Invalid language detection threshold");
            if (out.language_probability > *options.language_detection_threshold) { accepted = true; break; }
            auto found = std::find_if(votes.begin(),votes.end(),[&](const auto &item) { return item.first == out.language; });
            if (found == votes.end()) votes.push_back({out.language,{out.language_probability}}); else found->second.push_back(out.language_probability);
        }
        if (!accepted) {
            if (votes.empty()) throw std::invalid_argument("max() iterable argument is empty"); size_t best = 0;
            for (size_t i = 1; i < votes.size(); ++i) if (votes[i].second.size() > votes[best].second.size()) best = i;
            out.language = votes[best].first; out.language_probability = *std::max_element(votes[best].second.begin(),votes[best].second.end());
        }
    } else out.language = engine_.multilingual ? options.language.value_or("en") : "en";
    const auto tokenizer = base.configured(engine_.multilingual,options.task,out.language);
    for (auto &feature : features) feature = whisper_feature_window(feature,0,feature.frames);
    if (options.temperatures.size() > 1) options.temperatures.resize(1);
    if (options.suppress_tokens && !options.suppress_tokens->empty()) options.suppress_tokens = whisper_suppressed_tokens(tokenizer,options.suppress_tokens);
    options.hallucination_silence_threshold.reset(); options.condition_on_previous_text = false; options.prompt_reset_on_temperature = 0.5; options.max_initial_timestamp = 0;
    out.options = options; out.segments = generate_segments_impl(features,tokenizer,chunks,options,requested.batch_size,&speech); return out;
}
}
