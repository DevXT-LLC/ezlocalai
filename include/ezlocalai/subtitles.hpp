#pragma once
#include <functional>
#include <string>

namespace ezlocalai {
using SubtitleChat = std::function<std::string(const std::string &)>;
// app.py's numbered, word-count-bounded translation, preserving only the four
// fields selected by translate_segments_batch for translated segments.
std::string translate_subtitle_segments(const std::string &segments, const std::string &source,
    const std::string &target, const std::string &model, const SubtitleChat &chat, size_t max_words = 4000);
std::string generate_native_subtitles(const std::string &transcription, const std::string &source,
    const std::string &targets, const std::string &format, const std::string &model, const SubtitleChat &chat);
}
