#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ezlocalai {
struct LlmInputVideo {
    std::vector<std::string> frames;
    std::vector<double> timestamps;
    double duration = 0;
    // The caller owns this file, including when no frames were extracted.
    std::string audio_path;
};
std::vector<std::pair<int64_t, double>> llm_video_frame_plan(double video_fps,
    int64_t total_frames, int64_t max_frames, const std::vector<double> &scenes,
    bool scene_detect = true);
std::vector<double> llm_video_scenes(const std::string &path, double threshold = 30);
LlmInputVideo extract_llm_video(const std::string &source, double fps, int64_t max_frames,
    const std::function<std::string(const std::string &)> &download, bool scene_detect = true);
// Pipes._format_transcript_with_timestamps; JSON input preserves Python errors.
std::string llm_video_transcript(const std::string &transcript);
}
