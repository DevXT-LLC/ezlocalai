#include "ezlocalai/music_video.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ctts_paths.hpp"
#include "ezlocalai/llama_tts.hpp"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/llm_startup.hpp"
#include "ezlocalai/process.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <regex>
#include <unicode/unistr.h>

namespace ezlocalai {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
bool truth(const Json &v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0;
    return !v.empty() && (!v.is_string() || !v.get_ref<const std::string &>().empty());
}
Json value(const Json &request, const char *key, Json fallback = nullptr) {
    return request.contains(key) ? request.at(key) : fallback;
}
Json first(std::initializer_list<Json> values) {
    for (const auto &v : values) if (truth(v)) return v;
    return *std::prev(values.end());
}
std::string str(const Json &v) { return tts_python_string(v.dump()); }
double number(const Json &v) {
    if (v.is_number()) return v.get<double>();
    if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
    double result;
    if (!v.is_string() || !parse_configuration_float(v.get<std::string>(), &result))
        throw std::invalid_argument("Invalid music-video numeric setting");
    return result;
}
int64_t integer(const Json &v) {
    const auto n = number(v);
    if (!std::isfinite(n) || n < double(std::numeric_limits<int64_t>::min())
        || n >= -double(std::numeric_limits<int64_t>::min()))
        throw std::overflow_error("Music-video integer exceeds native range");
    return static_cast<int64_t>(n);
}
double environment_number(const char *name, double fallback) {
    const auto raw = std::getenv(name);
    if (!raw) return fallback;
    double result;
    if (!parse_configuration_float(raw, &result)) throw std::invalid_argument("could not convert string to float: " + python_string_repr(raw));
    return result;
}
std::string fixed(double n, int precision) {
    char bytes[512];
    const auto result = std::to_chars(bytes, bytes + sizeof(bytes), n, std::chars_format::fixed, precision);
    if (result.ec != std::errc()) throw std::overflow_error("Music-video decimal exceeds native range");
    return {bytes, result.ptr};
}
double round3(double n) { return std::stod(fixed(n, 3)); }
std::string read(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), {}};
}
void write(const fs::path &path, const std::string &data) {
    std::ofstream out(path, std::ios::binary); out.write(data.data(), data.size());
    if (!out) throw std::runtime_error("Cannot write " + path.string());
}
struct Temporary {
    fs::path path;
    explicit Temporary(const std::string &suffix) {
        auto pattern = (fs::temp_directory_path() / ("music-video-XXXXXX" + suffix)).string();
        const int fd = mkstemps(pattern.data(), suffix.size());
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "music-video temporary file");
        close(fd); path = pattern;
    }
    ~Temporary() { std::error_code ignored; fs::remove(path, ignored); }
};
std::string uuid() {
    unsigned char bytes[16];
    if (RAND_bytes(bytes, sizeof(bytes)) != 1) throw std::runtime_error("Music-video output identifier");
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const char *hex = "0123456789abcdef"; std::string out;
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out += '-';
        out += hex[bytes[i] >> 4]; out += hex[bytes[i] & 15];
    }
    return out;
}
}

int64_t music_video_frame_count(double duration, int64_t frame_rate) {
    const auto fps = std::max<int64_t>(1, frame_rate ? frame_rate : 24);
    const auto frames = std::ceil(duration * fps);
    if (std::isnan(frames)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (!std::isfinite(frames) || frames >= double(std::numeric_limits<int64_t>::max() - 8))
        throw std::overflow_error("Music-video frame count exceeds native range");
    if (frames <= 9) return 9;
    return ((static_cast<int64_t>(frames) + 6) / 8) * 8 + 1;
}
Json plan_music_video_scenes(double duration, std::optional<double> scene_duration,
    int64_t frame_rate, std::optional<double> max_scene_duration) {
    const double total = std::max(0.0, duration);
    Json scenes = Json::array();
    if (total <= 0) return scenes;
    if (!std::isfinite(total)) throw std::overflow_error("Infinite music-video duration");
    const auto requested = scene_duration ? *scene_duration : environment_number("MUSIC_VIDEO_SCENE_DURATION", 5);
    const auto maximum = max_scene_duration ? *max_scene_duration : environment_number("MUSIC_VIDEO_MAX_SCENE_DURATION", 20);
    const double length = std::min(std::max(1.0, requested), std::max(1.0, maximum));
    double start = 0;
    for (int64_t i = 0; start < total - .001; ++i) {
        const double current = std::min(length, total - start);
        scenes.push_back({{"index", i}, {"start", round3(start)}, {"duration", round3(current)},
                          {"num_frames", music_video_frame_count(current, frame_rate)}});
        if (start + current == start) throw std::overflow_error("Music-video timeline cannot advance");
        start += current;
    }
    return scenes;
}
bool is_pythagorean_request(const std::string &prompt, const std::string &lyrics) {
    const auto combined = llm_unicode_lower(prompt + " " + lyrics);
    for (const char *term : {"pythagorean", "hypotenuse", "a squared", "a²", "b squared", "b²", "c squared", "c²", "right triangle"})
        if (combined.find(term) != std::string::npos) return true;
    return false;
}
std::string music_video_lyric_excerpt(const std::string &lyrics, int64_t index, int64_t count) {
    const auto text = icu::UnicodeString::fromUTF8(lyrics);
    std::vector<std::string> lines;
    static const std::regex tags(R"(\[[^\]]+\])");
    int start = 0;
    auto append = [&](int end) {
        std::string line; text.tempSubStringBetween(start, end).toUTF8String(line);
        line = strip_config_value(std::regex_replace(line, tags, ""));
        if (!line.empty()) lines.push_back(std::move(line));
    };
    for (int i = 0; i < text.length(); ++i) {
        const auto ch = text.charAt(i);
        if (ch == '\n' || ch == '\r' || ch == '\v' || ch == '\f' || ch == 0x1c || ch == 0x1d
            || ch == 0x1e || ch == 0x85 || ch == 0x2028 || ch == 0x2029) {
            append(i);
            if (ch == '\r' && i + 1 < text.length() && text.charAt(i + 1) == '\n') ++i;
            start = i + 1;
        }
    }
    append(text.length());
    if (lines.empty()) return "";
    if (count <= 1) return lines.front();
    const double position = double(index) / std::max<int64_t>(1, count - 1) * (lines.size() - 1);
    const auto selected = std::clamp(std::nearbyint(position), 0.0, double(lines.size() - 1));
    return lines[static_cast<size_t>(selected)];
}
std::string ffconcat_quote(const std::string &path) {
    std::string result = "'";
    for (char ch : path) { if (ch == '\'') result += "'\\''"; else result += ch; }
    return result + "'";
}
std::string resolve_music_video_output(const std::string &reference, const std::string &local_uri) {
    if (reference.empty()) throw std::invalid_argument("Missing output reference");
    if (reference.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
    auto ref = reference, base = local_uri;
    while (!base.empty() && base.back() == '/') base.pop_back();
    if (!local_uri.empty() && ref.rfind(base + '/', 0) == 0) ref.erase(0, base.size() + 1);
    else if (ref.rfind("http://", 0) == 0 || ref.rfind("https://", 0) == 0) {
        const auto authority_end = ref.find_first_of("/?#", ref.find("://") + 3);
        auto path = authority_end == std::string::npos || ref[authority_end] != '/' ? "" : ref.substr(authority_end);
        path = path.substr(0, path.find_first_of("?#"));
        const auto last_slash = path.rfind('/'), params = path.find(';', last_slash == std::string::npos ? 0 : last_slash);
        if (params != std::string::npos) path.resize(params);
        const auto marker = path.find("/outputs/");
        if (marker == std::string::npos) throw std::invalid_argument("Cannot resolve non-output URL: " + ref);
        ref = "outputs/" + path.substr(marker + 9);
    }
    const auto candidate = ctts_real_path(fs::absolute(ref).lexically_normal().string());
    const auto outputs = ctts_real_path((fs::current_path() / "outputs").string());
    if (candidate.rfind(outputs + '/', 0) != 0)
        throw std::invalid_argument("Output reference resolves outside outputs directory");
    std::error_code error;
    if (!fs::exists(candidate, error)) throw std::runtime_error(candidate);
    return candidate;
}
Json stitch_music_video(const std::vector<std::string> &scenes, const std::string &audio,
    double duration, const std::string &format, const std::string &local_uri, const std::string &overlay_png) {
    fs::create_directories("outputs");
    const auto output = (fs::path("outputs") / (uuid() + ".mp4")).string();
    Temporary concat(".ffconcat"), log(".log");
    std::string lines;
    for (const auto &scene : scenes) lines += "file " + ffconcat_quote(ctts_real_path(scene)) + '\n';
    write(concat.path, lines);
    std::unique_ptr<Temporary> overlay;
    if (!overlay_png.empty()) { overlay = std::make_unique<Temporary>(".png"); write(overlay->path, overlay_png); }
    std::vector<std::string> command{"ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", concat.path.string(), "-i", audio};
    if (overlay) command.insert(command.end(), {"-loop", "1", "-i", overlay->path.string(), "-filter_complex",
        "[0:v:0][2:v:0]overlay=0:0:format=auto:shortest=0[v]", "-map", "[v]", "-map", "1:a:0"});
    else command.insert(command.end(), {"-map", "0:v:0", "-map", "1:a:0"});
    command.insert(command.end(), {"-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-af", "apad",
        "-t", fixed(duration, 3), "-movflags", "+faststart", output});
    const auto result = run_process_to_files(command, "/dev/null", log.path.string(), std::max(300.0, std::trunc(duration) * 20));
    if (result.timed_out) throw std::runtime_error("Music-video stitching timed out");
    if (result.status) {
        const auto message = strip_config_value(read(log.path));
        throw std::runtime_error(message.empty() ? "ffmpeg failed" : message);
    }
    if (format == "b64_json") {
        const auto bytes = read(output);
        if (bytes.size() > (std::string().max_size() - 4) / 4 * 3)
            throw std::length_error("Music-video base64 output too large");
        std::string encoded(4 * ((bytes.size() + 2) / 3) + 1, '\0');
        size_t position = 0;
        // EVP takes an int length and returns an int. Encode aligned chunks
        // so long videos do not overflow either count or insert midstream padding.
        for (size_t offset = 0; offset < bytes.size();) {
            const int chunk = static_cast<int>(std::min<size_t>(3 * 1024 * 1024, bytes.size() - offset));
            position += EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data() + position),
                reinterpret_cast<const unsigned char *>(bytes.data() + offset), chunk);
            offset += chunk;
        }
        encoded.resize(position); return {{"b64_json", encoded}};
    }
    return {{"url", local_uri.empty() ? output : local_uri + '/' + output}};
}

Json generate_music_video(const Json &request, const MusicVideoOperations &op) {
    const auto format = llm_unicode_lower(strip_config_value(str(first({value(request, "response_format"), "url"}))));
    if (format != "url" && format != "b64_json") throw std::invalid_argument("response_format must be 'url' or 'b64_json'");
    const auto duration = number(first({value(request, "duration"), 0}));
    if (duration <= 0) throw std::invalid_argument("duration must be greater than 0");
    const auto fps = integer(first({value(request, "frame_rate"), 24}));
    const auto scene_duration = value(request, "scene_duration");
    const auto scenes = plan_music_video_scenes(duration, scene_duration.is_null() ? std::nullopt : std::optional<double>(number(scene_duration)), fps);
    if (scenes.empty()) throw std::invalid_argument("No music-video scenes could be planned");
    const auto prompt = first({value(request, "prompt"), ""}), lyrics = first({value(request, "lyrics"), ""});
    const auto video_prompt = first({value(request, "video_prompt"), "cinematic music video visuals for: " + str(prompt)});
    const auto prompts = first({value(request, "scene_prompts"), Json::array()});
    const auto images = first({value(request, "scene_images"), Json::array()});
    const auto storyboard = llm_unicode_lower(strip_config_value(str(value(request, "storyboard", true))));
    const bool storyboard_enabled = storyboard != "0" && storyboard != "false" && storyboard != "no" && storyboard != "off";
    const bool math_overlay = storyboard_enabled && is_pythagorean_request(str(prompt), str(lyrics));
    Json audio_url = value(request, "audio_url"), music_result;
    if (!truth(audio_url)) {
        Json payload = {{"prompt", prompt}, {"lyrics", lyrics}, {"duration", duration}, {"keyscale", value(request, "keyscale")},
            {"model", first({value(request, "music_model"), value(request, "model")})}, {"instrumental", value(request, "instrumental", false)},
            {"n", 1}, {"response_format", "url"}, {"output_format", first({value(request, "music_output_format"), "wav16"})},
            {"seed", first({value(request, "music_seed"), value(request, "seed")})},
            {"inference_steps", first({value(request, "music_inference_steps"), value(request, "inference_steps")})},
            {"guidance_scale", value(request, "music_guidance_scale")}};
        for (const auto *key : {"bpm", "timesignature", "vocal_language", "shift", "solver", "lm_model", "synth_model", "audio_codes"})
            payload[key] = value(request, key);
        for (auto it = payload.begin(); it != payload.end();) { if (it->is_null()) it = payload.erase(it); else ++it; }
        music_result = op.music(payload);
        const auto items = first({value(music_result, "data"), Json::array()});
        if (items.empty() || !truth(value(items[0], "url"))) throw std::runtime_error("Music generation did not return a URL output");
        audio_url = items[0]["url"];
    }
    const auto audio_path = op.resolve(str(audio_url));
    std::vector<std::string> urls, paths;
    op.video_session([&] {
        for (const auto &scene : scenes) {
            const auto index = scene.at("index").get<size_t>();
            const auto start = scene.at("start").get<double>(), end = start + scene.at("duration").get<double>();
            Json scene_prompt = index < prompts.size() && !strip_config_value(str(prompts[index])).empty() ? prompts[index]
                : Json(str(video_prompt) + ". Scene " + std::to_string(index + 1) + " of " + std::to_string(scenes.size())
                    + ", covering seconds " + fixed(start, 1) + "-" + fixed(end, 1)
                    + "; match the song energy, rhythm, stage lighting, camera motion, and lyrical mood.");
            Json image = index < images.size() && !images[index].is_null() ? images[index] : value(request, "image");
            const auto conditions = value(request, "conditions");
            const auto size = first({value(request, "size"), "768x512"});
            if (image.is_null() && !truth(conditions) && storyboard_enabled)
                image = op.storyboard({{"prompt", prompt}, {"lyrics", lyrics}, {"scene_index", index}, {"scene_count", scenes.size()},
                    {"size", size}, {"keyscale", first({value(request, "keyscale"), ""})}});
            if (!image.is_null() && !truth(conditions))
                scene_prompt = str(scene_prompt) + " Start from the supplied storyboard frame; preserve the visible lyric/theorem graphics on the LED screen, especially the superscript exponent notation for a squared plus b squared equals c squared; do not use caret notation. Animate the band, lights, camera, and crowd around those visual anchors. Do not replace it with generic unrelated concert footage.";
            auto ref = op.scene({{"prompt", scene_prompt}, {"response_format", "url"}, {"size", size}, {"num_frames", scene.at("num_frames")},
                {"num_inference_steps", integer(first({value(request, "num_inference_steps"), 18}))},
                {"guidance_scale", number(first({value(request, "video_guidance_scale"), value(request, "guidance_scale"), 3.5}))},
                {"frame_rate", fps}, {"image", image}, {"conditions", conditions}, {"include_audio", truth(value(request, "include_scene_audio", false))}});
            if (ref.empty()) throw std::runtime_error("Video scene " + std::to_string(index + 1) + " failed to generate");
            urls.push_back(ref); paths.push_back(op.resolve(ref));
        }
    });
    const auto overlay = math_overlay ? op.overlay(str(first({value(request, "size"), "768x512"}))) : Json();
    auto item = op.stitch(paths, audio_path, duration, format, overlay);
    item.update({{"content_type", "video/mp4"}, {"format", "mp4"}, {"audio_url", audio_url}, {"scene_urls", urls},
        {"scene_plan", scenes}, {"scene_count", urls.size()}, {"scene_duration", number(first({scene_duration, 5.0}))},
        {"storyboard", storyboard_enabled}, {"math_overlay", math_overlay}});
    return {{"created", op.now()}, {"model", {{"music", first({value(music_result, "model"), value(request, "music_model"), value(request, "model")})},
        {"video", op.video_model()}}}, {"data", Json::array({item})}};
}
}
