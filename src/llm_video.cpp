#include "ezlocalai/llm_video.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/process.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <unistd.h>

namespace ezlocalai {
namespace {
std::string temporary(const char *suffix) {
    auto pattern = (std::filesystem::temp_directory_path() / (std::string("ezlocalai-video-XXXXXX") + suffix)).string();
    const auto fd = mkstemps(pattern.data(), std::char_traits<char>::length(suffix));
    if (fd < 0) throw std::runtime_error("Cannot create video temporary file");
    close(fd); return pattern;
}
struct Remove { std::string path; ~Remove() { std::error_code error; if (!path.empty()) std::filesystem::remove(path, error); } };
std::string encode(const std::string &bytes) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t n = static_cast<unsigned char>(bytes[i]) << 16;
        if (i + 1 < bytes.size()) n |= static_cast<unsigned char>(bytes[i + 1]) << 8;
        if (i + 2 < bytes.size()) n |= static_cast<unsigned char>(bytes[i + 2]);
        output += alphabet[n >> 18]; output += alphabet[(n >> 12) & 63];
        output += i + 1 < bytes.size() ? alphabet[(n >> 6) & 63] : '='; output += i + 2 < bytes.size() ? alphabet[n & 63] : '=';
    }
    return output;
}
std::string quote(const std::string &value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c < 32) { const char digits[] = "0123456789abcdef"; result += "\\u00"; result += digits[c >> 4]; result += digits[c & 15]; }
        else result += c;
    }
    return result + '"';
}
double rounded_timestamp(double value) {
    char output[512]; const auto result = std::to_chars(output, output + sizeof(output), value, std::chars_format::fixed, 2);
    if (result.ec != std::errc{}) throw std::runtime_error("Invalid video timestamp");
    return std::stod(std::string(output, result.ptr));
}
std::string extract_audio(const std::string &path) {
    Remove audio{temporary(".wav")}, errors{temporary(".log")};
    try {
        const auto result = run_process_to_files({"ffmpeg", "-i", path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio.path}, "/dev/null", errors.path, 120);
        if (result.status || result.timed_out || std::filesystem::file_size(audio.path) < 1000) return {};
        auto output = audio.path; audio.path.clear(); return output;
    } catch (...) { return {}; }
}
}
std::vector<std::pair<int64_t, double>> llm_video_frame_plan(double video_fps, int64_t total_frames,
    int64_t max_frames, const std::vector<double> &scenes, bool scene_detect) {
    if (!total_frames || video_fps <= 0) return {};
    const auto duration = total_frames / video_fps;
    std::set<double> timestamps;
    if (scene_detect && duration > 2.0) {
        auto selected = scenes;
        const auto budget = max_frames / 2 - (max_frames < 0 && max_frames % 2 != 0);
        if (static_cast<int64_t>(selected.size()) > budget) {
            if (!budget) throw std::runtime_error("division by zero");
            const auto step = static_cast<double>(selected.size()) / budget;
            selected.clear();
            for (int64_t i = 0; i < budget; ++i) selected.push_back(scenes.at(static_cast<size_t>(i * step)));
        }
        for (double timestamp : selected) if (timestamp >= 0 && timestamp <= duration) timestamps.insert(rounded_timestamp(timestamp));
    }
    const auto remaining = max_frames - static_cast<int64_t>(timestamps.size());
    const auto step = duration <= 0 ? 1 : duration / remaining;
    for (int64_t i = 0; i < remaining; ++i) {
        const auto timestamp = rounded_timestamp(i * step);
        if (timestamp <= duration) timestamps.insert(timestamp);
    }
    std::vector<double> selected(timestamps.begin(), timestamps.end());
    const auto end = max_frames < 0 ? std::max<int64_t>(0, static_cast<int64_t>(selected.size()) + max_frames) : std::min<int64_t>(selected.size(), max_frames);
    selected.resize(end);
    std::map<int64_t, double> frames;
    for (double timestamp : selected) {
        const auto position = timestamp * video_fps;
        if (!std::isfinite(position) || position > static_cast<double>(INT64_MAX)) throw std::runtime_error("Invalid video frame position");
        frames.emplace(std::min<int64_t>(static_cast<int64_t>(position), total_frames - 1), timestamp);
    }
    return {frames.begin(), frames.end()};
}
std::vector<double> llm_video_scenes(const std::string &path, double threshold) {
    cv::VideoCapture capture(path);
    if (!capture.isOpened()) return {};
    const auto fps = capture.get(cv::CAP_PROP_FPS);
    if (fps <= 0) return {};
    std::vector<double> scenes{0}; cv::Mat frame, previous; int64_t index = 0;
    const auto interval = std::max<int64_t>(1, static_cast<int64_t>(fps * 0.5));
    while (capture.read(frame)) {
        if (index % interval == 0) {
            cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); cv::resize(gray, gray, cv::Size(160, 90));
            if (!previous.empty()) { cv::Mat diff; cv::absdiff(previous, gray, diff); if (cv::mean(diff)[0] > threshold) scenes.push_back(index / fps); }
            previous = gray;
        }
        ++index;
    }
    return scenes;
}
LlmInputVideo extract_llm_video(const std::string &source, double fps, int64_t max_frames,
    const std::function<std::string(const std::string &)> &download, bool scene_detect) {
    (void)fps; // The source parses VIDEO_FPS but uses the decoded rate for sampling.
    LlmInputVideo result; Remove video;
    try {
        std::string bytes;
        if (source.rfind("data:", 0) == 0) {
            const auto comma = source.find(','); if (comma == std::string::npos) return result;
            bytes = stt_audio_input(quote(source.substr(comma + 1)), "\"wav\"").bytes;
        } else if (source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0) bytes = download(source);
        else return result;
        video.path = temporary(".mp4"); std::ofstream file(video.path, std::ios::binary); file.write(bytes.data(), bytes.size()); file.close();
        if (!file) return result;
    } catch (...) { return result; }
    try {
        result.audio_path = extract_audio(video.path);
        cv::VideoCapture capture(video.path); if (!capture.isOpened()) return result;
        const auto video_fps = capture.get(cv::CAP_PROP_FPS);
        const auto total_frames = static_cast<int64_t>(capture.get(cv::CAP_PROP_FRAME_COUNT));
        result.duration = video_fps > 0 ? total_frames / video_fps : 0;
        if (!total_frames || video_fps <= 0) return result;
        const auto scenes = scene_detect && result.duration > 2 ? llm_video_scenes(video.path) : std::vector<double>{};
        const auto plan = llm_video_frame_plan(video_fps, total_frames, max_frames, scenes, scene_detect);
        std::vector<std::string> frames;
        std::vector<double> timestamps;
        for (const auto &[number, timestamp] : plan) {
            capture.set(cv::CAP_PROP_POS_FRAMES, number); cv::Mat frame;
            if (!capture.read(frame)) continue;
            try {
                cv::Mat rgb; cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
                RgbImage image{rgb.cols, rgb.rows, {}}; image.pixels.reserve(static_cast<size_t>(rgb.cols) * rgb.rows * 3);
                for (int y = 0; y < rgb.rows; ++y) image.pixels.insert(image.pixels.end(), rgb.ptr<uint8_t>(y), rgb.ptr<uint8_t>(y) + rgb.cols * 3);
                const auto maximum = std::max(image.width, image.height);
                if (maximum > 1024) { const auto scale = 1024.0 / maximum; image = resize_rgb_lanczos(image, int(image.width * scale), int(image.height * scale)); }
                frames.push_back("data:image/jpeg;base64," + encode(encode_rgb_jpeg(image, 85)));
                timestamps.push_back(timestamp);
            } catch (...) { /* The source skips a frame that cannot be converted. */ }
        }
        result.frames = std::move(frames); result.timestamps = std::move(timestamps);
    } catch (...) { /* Extraction errors return the accumulated metadata/audio. */ }
    return result;
}
}
