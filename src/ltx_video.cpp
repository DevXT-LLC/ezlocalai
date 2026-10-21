#include "ezlocalai/ltx_video.hpp"
#include "ezlocalai/configuration.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>

namespace ezlocalai {
namespace {
using Integer = __int128;
std::string lower(std::string value) {
    for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return value;
}
Integer integer(const std::string &value) {
    const auto parsed = parse_configuration_integer(value);
    if (!parsed) throw std::invalid_argument("invalid integer: " + value);
    return std::stoll(*parsed);
}
Integer dimension(const std::string &value) {
    const auto parsed = parse_configuration_integer(value);
    if (!parsed) throw std::invalid_argument("invalid integer: " + value);
    // The source clamps large positive dimensions after alignment. Clamp
    // before narrowing so arbitrarily large positive Python ints still work.
    if (configuration_integer_less("1280", *parsed)) return 1280;
    return integer(*parsed);
}
Integer floor_multiple(Integer value, int divisor) {
    Integer quotient = value / divisor;
    if (value < 0 && value % divisor != 0) --quotient;
    return quotient * divisor;
}
int64_t narrow(const Integer &value) {
    if (value < std::numeric_limits<int64_t>::min() || value > std::numeric_limits<int64_t>::max())
        throw std::overflow_error("LTX dimension or frame count exceeds native integer range");
    return static_cast<int64_t>(value);
}
}
LtxVideoSettings normalize_ltx_settings(LtxVideoSettings settings) {
    const auto separator = settings.size.find('x');
    if (separator == std::string::npos || settings.size.find('x', separator + 1) != std::string::npos)
        throw std::invalid_argument("LTX size must contain two integers separated by x");
    const auto width = floor_multiple(dimension(settings.size.substr(0, separator)), 32);
    const auto height = floor_multiple(dimension(settings.size.substr(separator + 1)), 32);
    settings.width = narrow(std::min(width, Integer(1280)));
    settings.height = narrow(std::min(height, Integer(1280)));
    settings.frames = narrow(floor_multiple(Integer(settings.frames) - 1, 8) + 1);
    settings.steps = std::max<int64_t>(settings.steps, 2);
    return settings;
}
LtxVideoSettings reduced_ltx_settings(LtxVideoSettings settings) {
    settings.width = std::min<int64_t>(settings.width, 512);
    settings.height = std::min<int64_t>(settings.height, 320);
    settings.frames = narrow(floor_multiple(Integer(std::min<int64_t>(settings.frames, 41)) - 1, 8) + 1);
    return settings;
}
std::string ltx_generation_mode(bool conditions, bool condition_pipeline, bool image, bool image_pipeline) {
    if (conditions && condition_pipeline) return "condition";
    if (image && image_pipeline) return "i2v";
    return "t2v";
}
std::string choose_video_gpu_residency(const VideoResidencyPolicy &p) {
    auto mode = lower(strip_config_value(p.configured_mode.empty() ? "auto" : p.configured_mode));
    static const std::map<std::string, std::string> aliases = {{"gpu", "full"}, {"cuda", "full"}, {"resident", "full"},
        {"model", "model_offload"}, {"model_cpu_offload", "model_offload"}, {"offload", "model_offload"},
        {"seq", "sequential"}, {"sequential_cpu_offload", "sequential"}};
    if (const auto alias = aliases.find(mode); alias != aliases.end()) mode = alias->second;
    if (mode == "full" || mode == "model_offload" || mode == "sequential") return mode;
    if (p.text_encoder_on_gpu && p.free_gb >= p.full_min_free_gb) return "full";
    const Integer pixels = Integer(p.width.value_or(0)) * p.height.value_or(0);
    const auto frames = p.frames.value_or(0);
    const bool short_clip = frames > 0 && frames <= p.max_frames && (pixels <= 0 || pixels <= p.max_pixels);
    if (short_clip && p.free_gb >= p.short_min_free_gb) return "model_offload";
    if (p.free_gb >= p.model_min_free_gb || p.total_gb >= 40.0) return "model_offload";
    return "sequential";
}
VideoResidencyPolicy ltx_residency_policy_from_environment(VideoResidencyPolicy p) {
    const auto floating = [](const char *key, double fallback) {
        double result;
        const auto raw = std::getenv(key);
        return raw && parse_configuration_float(raw, &result) ? result : fallback;
    };
    const auto integral = [](const char *key, int64_t fallback) {
        const auto raw = std::getenv(key);
        const auto parsed = raw ? parse_configuration_integer(raw) : std::nullopt;
        if (!parsed) return fallback;
        // Native limits are bounded to int64_t. Python's arbitrary-precision
        // limits beyond this range are not represented by this policy type.
        if (configuration_integer_less(std::to_string(std::numeric_limits<int64_t>::max()), *parsed)) return std::numeric_limits<int64_t>::max();
        if (configuration_integer_less(*parsed, std::to_string(std::numeric_limits<int64_t>::min()))) return std::numeric_limits<int64_t>::min();
        return static_cast<int64_t>(std::stoll(*parsed));
    };
    const auto mode = std::getenv("VIDEO_GPU_RESIDENCY");
    p.configured_mode = mode ? mode : "auto";
    p.model_min_free_gb = floating("VIDEO_MODEL_OFFLOAD_MIN_FREE_GB",16);
    p.full_min_free_gb = floating("VIDEO_FULL_GPU_MIN_FREE_GB",30);
    p.short_min_free_gb = floating("VIDEO_SHORT_MODEL_OFFLOAD_MIN_FREE_GB",10);
    p.max_frames = integral("VIDEO_MODEL_OFFLOAD_MAX_FRAMES",129);
    p.max_pixels = integral("VIDEO_MODEL_OFFLOAD_MAX_PIXELS",262144);
    return p;
}
LtxFailureAction ltx_failure_action(bool runtime_error, const std::string &message,
                                  const std::string &residency, const std::string &retry_setting) {
    const auto error = lower(message);
    if (!runtime_error || (error.find("out of memory") == std::string::npos && error.find("cuda") == std::string::npos
        && error.find("model manager cannot make enough memory available") == std::string::npos))
        return LtxFailureAction::propagate;
    const auto retry = lower(strip_config_value(retry_setting));
    if (residency != "sequential" && (retry == "1" || retry == "true" || retry == "yes" || retry == "on"))
        return LtxFailureAction::reload_sequential;
    return LtxFailureAction::reduce_on_gpu;
}
LtxDevicePolicy ltx_device_policy(const std::string &device, bool available, int major, double free_gb) {
    LtxDevicePolicy result;
    result.cuda = device.rfind("cuda", 0) == 0;
    const auto colon = device.find(':');
    if (colon != std::string::npos) {
        const auto next = device.find(':', colon + 1);
        try { result.gpu_index = narrow(integer(device.substr(colon + 1, next - colon - 1))); }
        catch (const std::invalid_argument &) {}
    }
    result.dtype = device == "cpu" || (available && major >= 8) ? "bfloat16" : "float16";
    result.try_bnb = result.cuda && free_gb >= 12.0;
    return result;
}
}
