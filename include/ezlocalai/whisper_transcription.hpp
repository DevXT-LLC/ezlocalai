#pragma once
#include "ezlocalai/whisper_features.hpp"
#include "ezlocalai/whisper_kernel.h"
#include "ezlocalai/whisper_tokenizer.hpp"
#include "ezlocalai/whisper_vad.hpp"
#include <functional>
#include <variant>

namespace ezlocalai {
enum class WhisperTranscriptionProfile { faster_whisper_1_2_0, faster_whisper_1_0_3 };
struct WhisperWord { double start = 0, end = 0; std::string word; double probability = 0; };
struct WhisperSegment {
    int64_t id = 0, seek = 0;
    double start = 0, end = 0;
    std::string text;
    std::vector<uint32_t> tokens;
    double temperature = 0, avg_logprob = 0, compression_ratio = 0, no_speech_prob = 0;
    std::optional<std::vector<WhisperWord>> words;
};
struct WhisperGeneration { std::vector<uint32_t> tokens; double score = 0, no_speech_prob = 0; };
struct WhisperAlignment { std::vector<std::pair<int64_t,int64_t>> indices; std::vector<float> probabilities; };
using WhisperLanguages = std::vector<std::pair<std::string,double>>;
using WhisperEncoded = std::shared_ptr<void>;
struct WhisperEngineOperations {
    size_t bands = 80, languages = 0;
    std::optional<WhisperFeatureConfiguration> feature_configuration;
    bool multilingual = false;
    std::function<WhisperEncoded(const WhisperFeatures &)> encode;
    std::function<WhisperGeneration(const WhisperEncoded &,const std::vector<uint32_t> &,const ezl_whisper_options &)> generate;
    std::function<WhisperLanguages(const WhisperEncoded &)> detect_language;
    std::function<WhisperAlignment(const WhisperEncoded &,const std::vector<uint32_t> &,const std::vector<uint32_t> &,size_t,int64_t)> align;
    std::function<WhisperEncoded(const std::vector<WhisperFeatures> &)> encode_batch;
    std::function<std::vector<WhisperGeneration>(const WhisperEncoded &,const std::vector<std::vector<uint32_t>> &,const ezl_whisper_options &)> generate_batch;
    std::function<std::vector<WhisperLanguages>(const WhisperEncoded &)> detect_language_batch;
    std::function<std::vector<WhisperAlignment>(const WhisperEncoded &,const std::vector<uint32_t> &,const std::vector<std::vector<uint32_t>> &,const std::vector<size_t> &,int64_t)> align_batch;
};
struct WhisperTranscriptionOptions {
    int64_t beam_size = 5, best_of = 5, no_repeat_ngram_size = 0;
    double patience = 1, length_penalty = 1, repetition_penalty = 1;
    std::optional<double> log_prob_threshold = -1, no_speech_threshold = 0.6, compression_ratio_threshold = 2.4;
    bool condition_on_previous_text = true;
    double prompt_reset_on_temperature = 0.5;
    std::vector<double> temperatures{0,0.2,0.4,0.6,0.8,1};
    std::optional<std::variant<std::string,std::vector<uint32_t>>> initial_prompt;
    std::string prefix, hotwords;
    bool suppress_blank = true;
    std::optional<std::vector<int>> suppress_tokens = std::vector<int>{-1};
    bool without_timestamps = false, word_timestamps = false, multilingual = false;
    double max_initial_timestamp = 1;
    std::string prepend_punctuations = "\"'“¿([{-", append_punctuations = "\"'.。,，!！?？:：”)]}、";
    std::optional<int64_t> max_new_tokens;
    std::variant<std::string,std::vector<double>> clip_timestamps = std::string("0");
    std::optional<double> hallucination_silence_threshold;
    std::optional<std::string> language;
    std::string task = "transcribe";
    bool vad_filter = false;
    std::optional<WhisperVadOptions> vad_options;
    std::optional<int64_t> chunk_seconds;
    std::optional<int64_t> language_detection_segments = 1;
    std::optional<double> language_detection_threshold = 0.5;
};
struct WhisperDecodeResult {
    WhisperGeneration result;
    double avg_logprob = 0, temperature = 0, compression_ratio = 0;
};
struct WhisperSegmentSplit { std::vector<WhisperSegment> segments; int64_t seek; bool single_timestamp_ending; };
struct WhisperTranscription {
    std::vector<WhisperSegment> segments;
    std::string language;
    double language_probability = 1, duration = 0, duration_after_vad = 0;
    std::optional<WhisperLanguages> all_language_probs;
    WhisperTranscriptionOptions options;
};
WhisperSegmentSplit whisper_split_segments(const std::vector<uint32_t> &tokens,uint32_t timestamp_begin,
    double time_offset,int64_t segment_size,double segment_duration,int64_t seek);
WhisperDecodeResult whisper_decode_with_fallback(const WhisperEncoded &encoded,const std::vector<uint32_t> &prompt,
    const WhisperTokenizer &tokenizer,const WhisperTranscriptionOptions &options,const WhisperEngineOperations &engine);
std::vector<WhisperSegment> whisper_generate_segments(const WhisperFeatures &features,WhisperTokenizer tokenizer,
    const WhisperTranscriptionOptions &options,const WhisperEngineOperations &engine,
    WhisperTranscriptionProfile profile = WhisperTranscriptionProfile::faster_whisper_1_2_0,WhisperEncoded encoded = {});
void whisper_restore_speech_timestamps(std::vector<WhisperSegment> &segments,const std::vector<WhisperSpeechChunk> &chunks,
    WhisperTranscriptionProfile profile = WhisperTranscriptionProfile::faster_whisper_1_2_0,int64_t sampling_rate = 16000);
WhisperTranscription whisper_transcribe(const std::vector<float> &audio,const WhisperTokenizer &tokenizer,
    WhisperTranscriptionOptions options,const WhisperEngineOperations &engine,WhisperTranscriptionProfile profile,
    const std::function<WhisperVadAudio(const std::vector<float> &,const WhisperVadOptions &)> &vad = {});

struct WhisperTimeChunk { double start = 0, end = 0; };
struct WhisperBatchedOptions {
    WhisperTranscriptionOptions transcription;
    int64_t batch_size = 8;
    std::vector<WhisperTimeChunk> clip_timestamps;
    bool vad_options_from_dictionary = true;
    WhisperBatchedOptions() { transcription.without_timestamps = true; transcription.vad_filter = true; }
};
class WhisperBatchedPipeline {
public:
    explicit WhisperBatchedPipeline(WhisperEngineOperations engine) : engine_(std::move(engine)) {}
    WhisperTranscription transcribe(const std::vector<float> &audio,const WhisperTokenizer &tokenizer,
        WhisperBatchedOptions options = {},
        const std::function<std::vector<WhisperSpeechChunk>(const std::vector<float> &,const WhisperVadOptions &)> &vad = {});
    std::vector<WhisperSegment> generate_segments(const std::vector<WhisperFeatures> &features,const WhisperTokenizer &tokenizer,
        const std::vector<WhisperAudioChunk> &chunks,const WhisperTranscriptionOptions &options,int64_t batch_size = 8);
    double last_speech_timestamp() const { return last_speech_timestamp_; }
private:
    WhisperEngineOperations engine_;
    double last_speech_timestamp_ = 0;
    std::vector<WhisperSegment> generate_segments_impl(const std::vector<WhisperFeatures> &features,const WhisperTokenizer &tokenizer,
        const std::vector<WhisperAudioChunk> &chunks,const WhisperTranscriptionOptions &options,int64_t batch_size,
        const std::vector<WhisperSpeechChunk> *speech);
};

// Native implementation is supplied by the optional CTranslate2 target.
class WhisperNativeEngine {
public:
    WhisperNativeEngine(const std::string &model,const std::string &device = "cpu",const std::string &compute = "int8",size_t threads = 4);
    WhisperEngineOperations operations() const;
private:
    std::shared_ptr<void> model_;
};
}
