#include "ezlocalai/wan_video.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/image_processing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <set>
#include <sstream>

namespace ezlocalai {
std::string python_float_string(double value);
namespace {
const std::string t2v = "QuantStack/Wan2.2-T2V-A14B-GGUF", i2v = "QuantStack/Wan2.2-I2V-A14B-GGUF";
std::string env(const char *name, const char *fallback = "") { const auto value = std::getenv(name); return value ? value : fallback; }
std::string lower(std::string value) { for (auto &c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return value; }
int64_t integer(const std::string &raw) {
    auto value = normalize_decimal_digits(strip_config_value(raw));
    if (value.empty()) throw std::invalid_argument("invalid integer");
    size_t start = value[0] == '+' || value[0] == '-' ? 1 : 0;
    bool digit = false;
    for (size_t i = start; i < value.size(); ++i) {
        if (value[i] >= '0' && value[i] <= '9') digit = true;
        else if (value[i] != '_' || !digit || i + 1 >= value.size() || value[i + 1] < '0' || value[i + 1] > '9') throw std::invalid_argument("invalid integer");
    }
    if (!digit) throw std::invalid_argument("invalid integer");
    value.erase(std::remove(value.begin(), value.end(), '_'), value.end());
    return std::stoll(value);
}
struct WorkDirectory {
    std::filesystem::path path;
    WorkDirectory() {
        auto name = (std::filesystem::temp_directory_path() / "wan-XXXXXX").string();
        if (!mkdtemp(name.data())) throw std::system_error(errno, std::generic_category(), "Wan temporary directory");
        path = name;
    }
    ~WorkDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
std::string uuid() {
    std::random_device random;
    unsigned char bytes[16]; for (auto &byte : bytes) byte = static_cast<unsigned char>(random());
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    std::ostringstream out;
    for (int i = 0; i < 16; ++i) { if (i == 4 || i == 6 || i == 8 || i == 10) out << '-'; out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]); }
    return out.str();
}
} // namespace

bool is_wan_model(const std::string &model) {
    const auto name = lower(model);
    return name == lower(t2v) || name == lower(i2v) || name == "wan2.2" || name == "wan2.2-t2v" || name == "wan2.2-i2v";
}
std::vector<HubAsset> wan_model_assets(bool image_to_video, const std::string &quant) {
    auto selected = quant.empty() ? env("VIDEO_QUANT_TYPE", "Q4_K_M") : quant;
    for (auto &c : selected) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    static const std::set<std::string> allowed = {"Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"};
    if (!allowed.count(selected)) throw std::invalid_argument("Unsupported Wan quantization: " + selected);
    const std::string mode = image_to_video ? "I2V" : "T2V", repo = image_to_video ? i2v : t2v;
    const std::string revision = image_to_video ? "6c6717459277b9cd1f72579d78a0fd62a79e57dc" : "73eafba53a1a8f29254e4c77f92e74ea27d7cd6f";
    return {{"high", repo, "HighNoise/Wan2.2-" + mode + "-A14B-HighNoise-" + selected + ".gguf", revision},
        {"low", repo, "LowNoise/Wan2.2-" + mode + "-A14B-LowNoise-" + selected + ".gguf", revision},
        {"encoder", "city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q8_0.gguf", "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7"},
        {"vae", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d"}};
}
WanVideoSettings normalize_wan_settings(WanVideoSettings settings) {
    int64_t width, height;
    try {
        const auto size = lower(settings.size);
        const auto x = size.find('x');
        if (x == std::string::npos || size.find('x', x + 1) != std::string::npos) throw std::invalid_argument("size");
        width = integer(size.substr(0, x)); height = integer(size.substr(x + 1));
    } catch (const std::exception &) { throw std::invalid_argument("Invalid video dimensions, frame count or generation settings."); }
    if (width < 64 || height < 64 || settings.frames < 5 || settings.fps < 1 || settings.steps < 2)
        throw std::invalid_argument("Video needs dimensions >=64, frames >=5, fps >=1 and at least two steps.");
    if (!std::isfinite(settings.guidance) || settings.guidance < 0) throw std::invalid_argument("Video guidance scale must be finite and non-negative.");
    settings.width = static_cast<int>(std::min<int64_t>(width, 1280) / 16 * 16);
    settings.height = static_cast<int>(std::min<int64_t>(height, 1280) / 16 * 16);
    settings.frames = (settings.frames - 1) / 4 * 4 + 1;
    return settings;
}
std::vector<std::string> wan_video_command(const std::string &binary, const std::string &device,
    const std::map<std::string, std::string> &assets, const std::string &output,
    const std::string &prompt, const std::string &negative_prompt,
    const WanVideoSettings &s, const std::string &image_path) {
    // Integer round-to-nearest is exact for the source's 4/9 expert split.
    const auto high = std::max<int64_t>(1, std::min(s.steps - 1, (s.steps / 9) * 4 + ((s.steps % 9) * 4 + 4) / 9));
    const auto guidance = python_float_string(s.guidance);
    std::vector<std::string> args = {binary, "-M", "vid_gen", "--diffusion-model", assets.at("low"),
        "--high-noise-diffusion-model", assets.at("high"), "--t5xxl", assets.at("encoder"), "--vae", assets.at("vae"),
        "-p", prompt, "-n", negative_prompt, "-W", std::to_string(s.width), "-H", std::to_string(s.height),
        "--video-frames", std::to_string(s.frames), "--fps", std::to_string(s.fps), "--steps", std::to_string(s.steps - high),
        "--high-noise-steps", std::to_string(high), "--cfg-scale", guidance, "--high-noise-cfg-scale", guidance,
        "--sampling-method", "euler", "--high-noise-sampling-method", "euler", "--flow-shift", "3.0", "--diffusion-fa",
        "--vae-tiling", "--temporal-tiling", "--seed", "42", "--output-begin-idx", "0", "-o", output};
    args.push_back("--backend");
    if (device == "cpu") args.push_back("cpu");
    else {
        const auto colon = device.find(':');
        args.push_back("cuda" + std::to_string(integer(colon == std::string::npos ? env("MAIN_GPU", "0") : device.substr(colon + 1))));
    }
    if (lower(env("VIDEO_GPU_RESIDENCY", "auto")) != "full") args.push_back("--offload-to-cpu");
    if (!image_path.empty()) { args.push_back("-i"); args.push_back(image_path); }
    return args;
}
WanVideo::WanVideo(std::string model, std::string device) : model_(std::move(model)), device_(std::move(device)),
    binary_(env("SDCPP_BIN", "/opt/stable-diffusion.cpp/build/bin/sd-cli")) {
    if (!is_wan_model(model_)) throw std::invalid_argument("Unsupported Wan model: " + model_);
    if (!std::filesystem::is_regular_file(binary_)) throw std::runtime_error("Wan requires the stable-diffusion.cpp sd-cli binary: " + binary_);
}
std::string WanVideo::generate(const std::string &prompt, WanVideoSettings settings, const std::string &local_uri,
                              const std::string &image_bytes, bool conditions, const std::string &negative_prompt) {
    if (conditions) throw std::invalid_argument("Wan 2.2 supports text-to-video and first-image conditioning; use image instead of conditions.");
    if ((lower(model_) == lower(i2v) || lower(model_) == "wan2.2-i2v") && image_bytes.empty()) throw std::invalid_argument("The configured Wan I2V model requires an image.");
    settings = normalize_wan_settings(settings);
    std::filesystem::create_directories("outputs");
    const auto output = std::filesystem::path("outputs") / (uuid() + ".mp4");
    WorkDirectory work;
    std::string image_path;
    if (!image_bytes.empty()) {
        image_path = (work.path / "input.png").string();
        const auto png = encode_rgb_png(fit_rgb_lanczos(decode_rgb_image(image_bytes), settings.width, settings.height));
        std::ofstream file(image_path, std::ios::binary); file.write(png.data(), png.size());
        if (!file) throw std::runtime_error("Cannot write video conditioning image");
    }
    std::map<std::string, std::string> assets;
    for (const auto &asset : wan_model_assets(!image_bytes.empty(), "")) assets[asset.role] = download_hub_asset(asset, env("HF_HUB_CACHE", "models"));
    const auto pattern = (work.path / "frame_%06d.png").string();
    const auto command = wan_video_command(binary_, device_, assets, pattern, prompt, negative_prompt, settings, image_path);
    try {
        const auto raw_timeout = strip_config_value(env("VIDEO_GENERATION_TIMEOUT", "3600"));
        size_t used = 0; const double timeout = std::stod(raw_timeout, &used);
        if (used != raw_timeout.size()) throw std::invalid_argument("Invalid VIDEO_GENERATION_TIMEOUT");
        const auto generation = run_process_to_files(command, (work.path / "generation.log").string(), "", std::max(0.0, timeout), true);
        if (generation.timed_out) throw std::runtime_error("Wan generation timed out");
        if (generation.status) throw std::runtime_error("Wan generation failed (sd-cli exit " + std::to_string(generation.status) + ")");
        int64_t frames = 0;
        for (const auto &entry : std::filesystem::directory_iterator(work.path)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("frame_", 0) == 0 && entry.path().extension() == ".png") ++frames;
        }
        if (frames != settings.frames) throw std::runtime_error("Wan did not produce the requested number of frames");
        const auto encoding = run_process_to_files({"ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", std::to_string(settings.fps),
            "-start_number", "0", "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output.string()},
            "/dev/null", (work.path / "encoding.log").string(), 180);
        if (encoding.timed_out) throw std::runtime_error("Video encoding timed out");
        if (encoding.status) throw std::runtime_error("Video encoding failed");
        if (!std::filesystem::is_regular_file(output) || std::filesystem::file_size(output) == 0) throw std::runtime_error("Video encoding produced no output");
    } catch (...) { std::error_code ignored; std::filesystem::remove(output, ignored); throw; }
    auto base = local_uri;
    while (!base.empty() && base.back() == '/') base.pop_back();
    return local_uri.empty() ? output.generic_string() : base + "/" + output.generic_string();
}
} // namespace ezlocalai
