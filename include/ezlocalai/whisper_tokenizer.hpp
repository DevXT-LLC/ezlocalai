#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace ezlocalai {
enum class WhisperTokenizerProfile { faster_whisper_1_2_0, faster_whisper_1_0_3 };
struct WhisperWordTokens { std::string word; std::vector<uint32_t> tokens; };
// The raw text pieces selected by the tokenizer's ByteLevel pre-tokenizer.
std::vector<std::string> whisper_pre_tokenize(const std::string &text);
class WhisperTokenizer {
public:
    WhisperTokenizer(const std::string &path, bool multilingual,
        const std::string &task = "transcribe", const std::string &language = "en");
    WhisperTokenizer configured(bool multilingual,const std::string &task,const std::string &language) const;
    std::optional<uint32_t> token_id(const std::string &text) const;
    std::vector<uint32_t> encode(const std::string &text) const;
    std::string decode(const std::vector<uint32_t> &tokens) const;
    std::string decode_with_timestamps(const std::vector<uint32_t> &tokens) const;
    std::vector<uint32_t> sot_sequence() const;
    std::vector<int> non_speech_tokens() const;
    std::vector<WhisperWordTokens> split_tokens_on_unicode(const std::vector<uint32_t> &tokens) const;
    std::vector<WhisperWordTokens> split_tokens_on_spaces(const std::vector<uint32_t> &tokens) const;
    std::vector<WhisperWordTokens> split_to_word_tokens(const std::vector<uint32_t> &tokens) const;
    uint32_t sot() const, sot_lm() const, sot_prev() const, eot() const;
    uint32_t transcribe() const, translate() const, no_timestamps() const;
    uint32_t timestamp_begin() const { return no_timestamps() + 1; }
    const std::string &language_code() const { return language_code_; }
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    std::optional<uint32_t> task_, language_;
    std::string language_code_;
    void configure(bool multilingual,const std::string &task,const std::string &language);
};
std::vector<uint32_t> whisper_prompt(const WhisperTokenizer &tokenizer,const std::vector<uint32_t> &previous_tokens,
    bool without_timestamps = false,const std::string &prefix = {},const std::string &hotwords = {},int64_t max_length = 448);
std::optional<std::vector<int>> whisper_suppressed_tokens(const WhisperTokenizer &tokenizer,
    std::optional<std::vector<int>> tokens,WhisperTokenizerProfile profile = WhisperTokenizerProfile::faster_whisper_1_2_0);
}
