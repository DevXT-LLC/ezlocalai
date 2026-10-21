#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ezlocalai {
int64_t music_video_frame_count(double duration, int64_t frame_rate = 24);
nlohmann::json plan_music_video_scenes(double duration, std::optional<double> scene_duration = {},
    int64_t frame_rate = 24, std::optional<double> max_scene_duration = {});
bool is_pythagorean_request(const std::string &prompt, const std::string &lyrics);
std::string music_video_lyric_excerpt(const std::string &lyrics, int64_t index, int64_t count);
std::string ffconcat_quote(const std::string &path);
std::string resolve_music_video_output(const std::string &reference, const std::string &local_uri);
nlohmann::json stitch_music_video(const std::vector<std::string> &scenes, const std::string &audio,
    double duration, const std::string &format, const std::string &local_uri,
    const std::string &overlay_png = {});

// Images are opaque values here. The HTTP adapter validates/decodes them
// before orchestration; the renderer and video engine own their pixel format.
struct MusicVideoOperations {
    using Json = nlohmann::json;
    std::function<Json(const Json &)> music;
    std::function<std::string(const std::string &)> resolve;
    // Retain one video lock and memory handoff across every scene, restoring
    // displaced models on success and failure, before final FFmpeg stitching.
    std::function<void(const std::function<void()> &)> video_session;
    std::function<std::string(const Json &)> scene;
    std::function<Json(const Json &)> storyboard;
    std::function<Json(const std::string &)> overlay;
    std::function<Json(const std::vector<std::string> &, const std::string &, double,
                       const std::string &, const Json &)> stitch;
    std::function<std::string()> video_model;
    std::function<int64_t()> now;
};
nlohmann::json generate_music_video(const nlohmann::json &request, const MusicVideoOperations &operations);
}
