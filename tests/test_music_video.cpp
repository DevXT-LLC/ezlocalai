#include "ezlocalai/music_video.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/process.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sstream>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using namespace ezlocalai;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), {}}; }
std::string sha(const std::string &value) {
    unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned count;
    EVP_Digest(value.data(), value.size(), bytes, &count, EVP_sha256(), nullptr);
    std::ostringstream out;
    for (unsigned i = 0; i < count; ++i) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
    return out.str();
}
struct Environment {
    std::map<std::string, std::optional<std::string>> previous;
    explicit Environment(const Json &values) {
        for (const auto *key : {"MUSIC_VIDEO_SCENE_DURATION", "MUSIC_VIDEO_MAX_SCENE_DURATION"}) {
            const char *old = std::getenv(key); previous[key] = old ? std::optional<std::string>(old) : std::nullopt;
            unsetenv(key);
        }
        for (const auto &[key, value] : values.items()) setenv(key.c_str(), value.get<std::string>().c_str(), 1);
    }
    ~Environment() { for (const auto &[key, value] : previous) { if (value) setenv(key.c_str(), value->c_str(), 1); else unsetenv(key.c_str()); } }
};
std::optional<double> optional(const Json &value) { return value.is_null() ? std::nullopt : std::optional<double>(value.get<double>()); }
void orchestration(const Json &row) {
    Json trace = Json::array(), result, error;
    const auto failure = row.at("failure").get<std::string>();
    int counter = 0;
    auto record = [&](const std::string &kind, const Json &value = nullptr) { trace.push_back(Json::array({kind, value})); };
    MusicVideoOperations op;
    op.music = [&](const Json &p) -> Json {
        record("music", p);
        if (failure == "music") throw std::runtime_error("fixture music failure");
        return failure == "music-empty" ? Json::object() : Json{{"model", "fixture-music"}, {"data", Json::array({{{"url", "outputs/song.wav"}}})}};
    };
    op.resolve = [&](const std::string &ref) {
        record("resolve", ref);
        if (failure == "resolve") throw std::invalid_argument("fixture path failure");
        return "/fixture/" + ref;
    };
    op.video_session = [&](const std::function<void()> &work) {
        record("begin");
        try { work(); } catch (...) { record("end"); throw; }
        record("end");
    };
    op.scene = [&](const Json &p) {
        record("scene", p); ++counter;
        if (failure == "scene" && counter == 2) throw std::runtime_error("fixture scene failure");
        return failure == "scene-empty" && counter == 2 ? "" : "outputs/scene-" + std::to_string(counter) + ".mp4";
    };
    op.storyboard = [&](const Json &p) -> Json {
        record("storyboard", p);
        if (failure == "storyboard") throw std::runtime_error("fixture storyboard failure");
        return "storyboard-" + std::to_string(p.at("scene_index").get<int>());
    };
    op.overlay = [&](const std::string &size) -> Json { record("overlay", size); return "math-overlay"; };
    op.stitch = [&](const std::vector<std::string> &scenes, const std::string &audio, double duration,
                    const std::string &format, const Json &overlay) -> Json {
        record("stitch", {{"scene_paths", scenes}, {"audio_path", audio}, {"duration", duration},
                          {"response_format", format}, {"overlay_image", overlay}});
        if (failure == "stitch") throw std::runtime_error("fixture stitch failure");
        return format == "b64_json" ? Json{{"b64_json", "fixture-base64"}} : Json{{"url", "outputs/final.mp4"}};
    };
    op.video_model = [] { return "fixture-video"; };
    op.now = [] { return 1234567890; };
    Environment environment(row.at("environment"));
    try { result = generate_music_video(row.at("request"), op); }
    catch (const std::invalid_argument &e) { error = {{"type", "ValueError"}, {"message", e.what()}}; }
    catch (const std::exception &e) { error = {{"type", "RuntimeError"}, {"message", e.what()}}; }
    check(error == row.value("error", Json()), "orchestration error: " + row.at("request").dump() + " actual " + error.dump());
    check(trace == row.at("trace"), "orchestration trace mismatch: " + row.at("request").dump() + "\nactual " + trace.dump() + "\nexpected " + row.at("trace").dump());
    if (error.is_null()) check(result == row.at("result"), "orchestration result mismatch: " + result.dump());
}
void media_and_paths() {
    auto pattern = (fs::temp_directory_path() / "ezlocalai-music-video-XXXXXX").string();
    check(mkdtemp(pattern.data()) != nullptr, "temporary workspace");
    const auto previous = fs::current_path();
    struct Cleanup {
        fs::path previous, directory;
        ~Cleanup() { fs::current_path(previous); std::error_code ignored; fs::remove_all(directory, ignored); }
    } cleanup{previous, pattern};
    fs::current_path(pattern); fs::create_directory("outputs");
    auto command = [&](const std::vector<std::string> &args) {
        const auto result = run_process_to_files(args, "/dev/null", "ffmpeg.log", 30);
        check(!result.status && !result.timed_out, "fixture FFmpeg: " + read("ffmpeg.log"));
    };
    const std::string red = "outputs/red 'quoted'.mp4", blue = "outputs/blue.mp4", audio = "outputs/music.wav";
    for (const auto &[color, path] : std::vector<std::pair<std::string, std::string>>{{"red", red}, {"blue", blue}})
        command({"ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=" + color + ":s=64x64:r=24:d=0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", path});
    command({"ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=24000:duration=0.25", audio});
    const auto absolute = (fs::current_path() / blue).string();
    for (const auto &ref : {blue, absolute, std::string("http://worker/outputs/blue.mp4"), std::string("https://router/prefix/outputs/blue.mp4?download=1#part")})
        check(resolve_music_video_output(ref, "http://worker/") == absolute, "resolve output " + ref);
    fs::create_symlink("blue.mp4", "outputs/link");
    check(resolve_music_video_output("outputs/link", "") == absolute, "resolve contained symlink");
    std::ofstream("outside") << "data";
    fs::create_symlink("../outside", "outputs/escape");
    fs::create_symlink("../../missing", "outputs/dangling");
    for (const auto *ref : {"", "outputs", "outside", "outputs/escape", "outputs/dangling", "http://other/non-output", "http://other?next=/outputs/blue.mp4"}) {
        bool rejected = false;
        try { resolve_music_video_output(ref, ""); } catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "reject uncontained output " + std::string(ref));
    }
    check(ffconcat_quote("a'b") == "'a'\\''b'", "concat quoting");
    auto probe = [&](const std::string &path) {
        const auto result = run_process({"ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path}, true);
        check(result.status == 0, "probe output"); return Json::parse(result.output);
    };
    const auto result = stitch_music_video({red, blue}, audio, .75, "url", "http://worker");
    const auto path = resolve_music_video_output(result.at("url"), "http://worker");
    const auto info = probe(path);
    const auto &video = info.at("streams").at(0), &sound = info.at("streams").at(1);
    check(video.at("nb_frames") == "18" && sound.at("sample_rate") == "24000", "stitch frame count and audio");
    check(std::abs(std::stod(info.at("format").at("duration").get<std::string>()) - .75) < .05, "trim final duration");
    command({"ffmpeg", "-v", "error", "-y", "-i", path, "-vf", "select=eq(n\\,15)", "-frames:v", "1", "last.png"});
    const auto last = decode_rgb_image(read("last.png"));
    check(last.pixels[2] > 200 && last.pixels[0] < 20, "scene ordering: blue follows red");
    RgbImage image; image.width = image.height = 64; image.pixels.assign(64 * 64 * 3, 0);
    for (size_t i = 0; i < image.pixels.size(); i += 3) image.pixels[i + 1] = 255;
    const auto encoded = stitch_music_video({red}, audio, .25, "b64_json", "", encode_rgb_png(image));
    const auto base64 = encoded.at("b64_json").get<std::string>();
    std::string bytes(base64.size(), '\0');
    int count = EVP_DecodeBlock(reinterpret_cast<unsigned char *>(bytes.data()), reinterpret_cast<const unsigned char *>(base64.data()), base64.size());
    check(count > 0, "base64 video");
    if (base64.back() == '=') --count;
    if (base64.size() > 1 && base64[base64.size() - 2] == '=') --count;
    bytes.resize(count); std::ofstream("overlay.mp4", std::ios::binary).write(bytes.data(), bytes.size());
    command({"ffmpeg", "-v", "error", "-y", "-i", "overlay.mp4", "-frames:v", "1", "overlay.png"});
    const auto overlay = decode_rgb_image(read("overlay.png"));
    check(overlay.pixels[1] > 240 && overlay.pixels[0] < 20, "overlay encoded into final video");
    bool failed = false;
    try { stitch_music_video({"outputs/missing.mp4"}, audio, 1, "url", ""); }
    catch (const std::runtime_error &) { failed = true; }
    check(failed, "stitch failure propagates");
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--stitch") {
            const auto request = Json::parse(argv[2]);
            const auto overlay = request.value("overlay_path", "");
            std::cout << stitch_music_video(request.at("scene_paths").get<std::vector<std::string>>(),
                request.at("audio_path"), request.at("duration"), request.at("response_format"),
                request.value("local_uri", ""), overlay.empty() ? "" : read(overlay)).dump() << '\n';
            return 0;
        }
        check(argc == 3, "reference root and fixture required");
        const auto fixture = Json::parse(read(argv[2]));
        for (const auto &[path, hash] : fixture.at("sources").items())
            check(sha(read(fs::path(argv[1]) / path)) == hash, "reference source changed: " + path);
        size_t count = 0;
        for (const auto &row : fixture.at("planning")) {
            Environment environment(row.at("environment"));
            const auto &args = row.at("args"); Json result; std::string error;
            try { result = plan_music_video_scenes(args.at("duration"), optional(args.at("scene_duration")), args.at("frame_rate"), optional(args.at("max_scene_duration"))); }
            catch (const std::invalid_argument &) { error = "ValueError"; }
            check(error == row.value("error", ""), "planning error: " + row.dump());
            if (error.empty()) check(result == row.at("result"), "planning mismatch: " + row.dump() + " actual " + result.dump());
            ++count;
        }
        for (const auto &row : fixture.at("frames")) { check(music_video_frame_count(row.at("duration"), row.at("fps")) == row.at("result"), "frames"); ++count; }
        for (const auto &row : fixture.at("lyrics")) { check(music_video_lyric_excerpt(row.at("lyrics"), row.at("index"), row.at("count")) == row.at("result"), "lyrics: " + row.dump()); ++count; }
        for (const auto &row : fixture.at("theorem")) { check(is_pythagorean_request(row.at("prompt"), row.at("lyrics")) == row.at("result"), "theorem"); ++count; }
        for (const auto &row : fixture.at("orchestration")) { orchestration(row); ++count; }
        media_and_paths();
        std::cout << "Verified " << count << " source cases plus real FFmpeg, output containment and overlay checks\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
