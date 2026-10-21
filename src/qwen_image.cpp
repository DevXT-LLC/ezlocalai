#include "ezlocalai/qwen_image.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/video_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <sys/file.h>

namespace ezlocalai {
std::string python_float_string(double value);
namespace {
namespace fs = std::filesystem;
std::string env(const char *name, const char *fallback) { const auto p = std::getenv(name); return p ? p : fallback; }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), {}}; }
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary); out.write(bytes.data(), bytes.size());
    if (!out) throw std::system_error(errno ? errno : EIO, std::generic_category(), "Cannot write image file");
}
struct WorkDirectory {
    fs::path path;
    WorkDirectory() {
        auto name = (fs::temp_directory_path() / "qwen-image-XXXXXX").string();
        if (!mkdtemp(name.data())) throw std::system_error(errno, std::generic_category(), "Qwen image temporary directory");
        path = name;
    }
    ~WorkDirectory() { std::error_code ignored; fs::remove_all(path, ignored); }
};
struct InferenceLock {
    int fd;
    explicit InferenceLock(const fs::path &path) : fd(open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "Image inference lock");
        while (flock(fd, LOCK_EX) != 0) if (errno != EINTR) {
            const int error = errno; close(fd); throw std::system_error(error, std::generic_category(), "Image inference lock");
        }
    }
    ~InferenceLock() { close(fd); }
};
int64_t integer(std::string value) {
    const auto parsed = parse_configuration_integer(value);
    if (!parsed) throw std::invalid_argument("integer");
    return std::stoll(*parsed);
}
// Pillow's image decoding errors derive from OSError, not RuntimeError; the
// edit endpoint lets those become a generic 500 instead of its explicit 503.
struct OutputDecodeError : std::exception {
    std::string message;
    explicit OutputDecodeError(std::string value) : message(std::move(value)) {}
    const char *what() const noexcept override { return message.c_str(); }
};
std::string base64(const std::string &bytes) {
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("Image too large");
    std::string result(4 * ((bytes.size() + 2) / 3), '\0');
    EVP_EncodeBlock(reinterpret_cast<unsigned char *>(result.data()), reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
    return result;
}
RgbImage load_image(const std::string &source, const fs::path &temporary) {
    if (source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0) {
        const auto result = run_process_to_files({"curl", "--silent", "--show-error", "--fail", "--location",
            "--proto", "=http,https", "--proto-redir", "=http,https", "--max-time", "30", "--max-filesize", "33554432",
            "--output", temporary.string(), "--", source}, "/dev/null", "/dev/null", 30);
        if (result.timed_out || result.status || !fs::is_regular_file(temporary) || fs::file_size(temporary) > 32 * 1024 * 1024)
            throw std::invalid_argument("Image download failed");
        return decode_rgb_image(read(temporary));
    }
    auto encoded = source;
    if (encoded.rfind("data:", 0) == 0) {
        const auto comma = encoded.find(',');
        const auto header = encoded.substr(0, comma);
        if (comma == std::string::npos || header.rfind("data:image/", 0) != 0 || header.size() < 7
            || header.substr(header.size() - 7) != ";base64") throw std::invalid_argument("Image data URL");
        encoded.erase(0, comma + 1);
    }
    if (encoded.size() > 45 * 1024 * 1024) throw std::invalid_argument("Encoded image exceeds size limit");
    // Python uses validate=True here, unlike the permissive video decoder.
    size_t digits = 0, padding = 0;
    for (const unsigned char c : encoded) {
        if (c == '=') { ++padding; continue; }
        if (padding || !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' || c == '/'))
            throw std::invalid_argument("Invalid base64 image");
        ++digits;
    }
    // b64decode accepts trailing padding after a complete quartet, but rejects
    // excess padding after a partial one and padding without any data.
    if ((!digits && padding) || digits % 4 == 1 || (digits % 4 && padding != 4 - digits % 4))
        throw std::invalid_argument("Invalid base64 image padding");
    return decode_base64_rgb_image(encoded);
}
std::string uuid() {
    unsigned char bytes[16]; if (RAND_bytes(bytes, sizeof(bytes)) != 1) throw std::runtime_error("Image output identifier");
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const char *hex = "0123456789abcdef"; std::string result;
    for (int i = 0; i < 16; ++i) { if (i == 4 || i == 6 || i == 8 || i == 10) result += '-'; result += hex[bytes[i] >> 4]; result += hex[bytes[i] & 15]; }
    return result;
}
} // namespace
std::vector<HubAsset> qwen_image_assets() {
    return {{"diffusion", "leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q4_K.gguf", "main"},
        {"vae", "Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", "main"},
        {"llm", "Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "main"},
        {"mmproj", "Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-F16.gguf", "main"}};
}
std::pair<int64_t, int64_t> qwen_image_dimensions(const std::string &size) {
    int64_t width, height;
    try {
        const auto x = size.find('x');
        if (x == std::string::npos || size.find('x', x + 1) != std::string::npos) throw std::invalid_argument("size");
        width = integer(size.substr(0, x)); height = integer(size.substr(x + 1));
    } catch (const std::exception &) { throw std::invalid_argument("size must be WIDTHxHEIGHT"); }
    if (width <= 0 || height <= 0 || width % 32 || height % 32) throw std::invalid_argument("Image dimensions must be positive multiples of 32");
    return {width, height};
}
QwenImage::QwenImage(std::string device) : device_(std::move(device)),
    binary_(env("SDCPP_BIN", "/opt/stable-diffusion.cpp/build/bin/sd-cli")), models_(env("SDCPP_MODELS_DIR", "models/qwen-image")) {
    if (!fs::is_regular_file(binary_)) return;
    try {
        for (const auto &asset : qwen_image_assets()) if (!fs::is_regular_file(fs::path(models_) / asset.filename))
            download_hub_file(asset.repository, asset.filename, models_, asset.revision);
        available_ = true;
    } catch (const std::exception &e) { std::cerr << "[IMG] Failed to download models: " << e.what() << '\n'; }
}
std::string QwenImage::generate(const std::string &prompt, const QwenImageSettings &s, const std::string &local_uri) {
    if (!available_) throw std::runtime_error("Qwen-Image-2.1 is unavailable (binary or models missing)");
    const auto [width, height] = qwen_image_dimensions(s.size);
    if (!(s.strength >= 0 && s.strength <= 1)) throw std::invalid_argument("strength must be between 0 and 1");
    if (s.images.size() > 10) throw std::invalid_argument("At most 10 additional reference images are supported");
    if (s.steps <= 0) throw std::invalid_argument("num_inference_steps must be positive");
    InferenceLock lock(fs::path(models_) / ".inference.lock");
    WorkDirectory work;
    auto sources = s.images; if (s.image) sources.insert(sources.begin(), *s.image);
    std::vector<std::string> refs;
    for (size_t i = 0; i < sources.size(); ++i) {
        RgbImage image;
        try { image = load_image(sources[i], work.path / "download"); }
        catch (const std::exception &) { throw std::invalid_argument("Invalid reference image at index " + std::to_string(i)); }
        const auto path = work.path / ("ref_" + std::to_string(i) + ".png");
        write(path, encode_rgb_png(image)); refs.push_back(path.string());
    }
    const auto assets = qwen_image_assets();
    auto asset = [&](size_t i) { return (fs::path(models_) / assets[i].filename).string(); };
    std::vector<std::string> args{binary_, "--diffusion-model", asset(0), "--vae", asset(1), "--llm", asset(2), "-p", prompt,
        "--cfg-scale", python_float_string(s.guidance), "--sampling-method", "euler", "--steps", std::to_string(s.steps),
        "-H", std::to_string(height), "-W", std::to_string(width), "--diffusion-fa"};
    if (!s.negative_prompt.empty()) args.insert(args.end(), {"--negative-prompt", s.negative_prompt});
    if (s.image) args.insert(args.end(), {"--init-img", refs.front(), "--strength", python_float_string(s.strength)});
    if (!refs.empty()) {
        if (!fs::is_regular_file(asset(3))) throw std::runtime_error("Qwen image editing requires mmproj vision weights");
        args.insert(args.end(), {"--llm_vision", asset(3)});
        for (const auto &ref : refs) args.insert(args.end(), {"-r", ref});
    }
    if (device_ == "cpu") args.insert(args.end(), {"--backend", "cpu"});
    else if (device_.rfind("cuda:", 0) == 0) { auto backend = device_; backend.erase(std::remove(backend.begin(), backend.end(), ':'), backend.end()); args.insert(args.end(), {"--backend", backend}); }
    if (device_ != "cpu") {
        bool offload = true;
        try { const int ordinal = device_.rfind("cuda:", 0) == 0 ? std::stoi(device_.substr(5)) : 0;
            offload = cuda_memory_info(ordinal).free_bytes < 16ULL * 1024 * 1024 * 1024; } catch (...) {}
        if (offload) args.push_back("--offload-to-cpu");
    }
    const auto output = work.path / "output.png", log = work.path / "generation.log";
    args.insert(args.end(), {"-o", output.string()});
    for (const auto &argument : args) if (argument.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
    const auto result = run_process_to_files(args, log.string(), "", 600, true);
    if (result.timed_out) throw std::runtime_error("sd-cli timed out after 600 seconds");
    if (result.status) {
        // Read only the diagnostic tail; a verbose engine must not consume RAM.
        std::ifstream in(log, std::ios::binary); in.seekg(0, std::ios::end); const auto length = in.tellg();
        in.seekg(std::max<std::streamoff>(0, static_cast<std::streamoff>(length) - 4000));
        const std::string diagnostics{std::istreambuf_iterator<char>(in), {}};
        std::cerr << "[IMG] sd-cli exit " << result.status << ": " << diagnostics << '\n';
        throw std::runtime_error("sd-cli failed (exit " + std::to_string(result.status) + "); see server logs");
    }
    if (!fs::is_regular_file(output)) throw std::runtime_error("sd-cli completed without an output image");
    const auto bytes = read(output);
    RgbImage generated;
    try { generated = decode_rgb_image(bytes); }
    catch (const std::exception &e) { throw OutputDecodeError(e.what()); }
    if (bytes.compare(0, 8, std::string("\x89PNG\r\n\x1a\n", 8)) || generated.width != width || generated.height != height)
        throw std::runtime_error("sd-cli returned an invalid PNG or unexpected dimensions");
    if (local_uri.empty()) return base64(bytes);
    fs::create_directories("outputs"); const auto destination = fs::path("outputs") / (uuid() + ".png");
    write(destination, reencode_png(bytes));
    auto base = local_uri; while (!base.empty() && base.back() == '/') base.pop_back();
    return base + "/" + destination.generic_string();
}
} // namespace ezlocalai
