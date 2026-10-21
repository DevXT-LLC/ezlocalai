#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace ezlocalai {
// Model-independent LTX policy. Tensor execution and conditioning must use
// these settings; this does not advertise an implemented inference backend.
struct LtxVideoSettings {
    std::string size = "768x512";
    int64_t width = 768, height = 512, frames = 121, fps = 24, steps = 40;
    double guidance = 4.0;
};
LtxVideoSettings normalize_ltx_settings(LtxVideoSettings settings);
LtxVideoSettings reduced_ltx_settings(LtxVideoSettings settings);
std::string ltx_generation_mode(bool has_conditions, bool condition_pipeline,
                                bool has_image, bool image_pipeline);

struct VideoResidencyPolicy {
    std::string configured_mode = "auto";
    double total_gb = 0, free_gb = 0;
    bool text_encoder_on_gpu = false;
    std::optional<int64_t> frames, width, height;
    double model_min_free_gb = 16, full_min_free_gb = 30, short_min_free_gb = 12;
    int64_t max_frames = 129, max_pixels = 262144;
};
std::string choose_video_gpu_residency(const VideoResidencyPolicy &policy);
// LTX_VIDEO.py uses raw environment values and a 10 GiB short-clip default;
// VIDEO_UTILS.py's standalone helper defaults that threshold to 12 GiB.
VideoResidencyPolicy ltx_residency_policy_from_environment(VideoResidencyPolicy observation);

enum class LtxFailureAction { propagate, reload_sequential, reduce_on_gpu };
LtxFailureAction ltx_failure_action(bool runtime_error, const std::string &message,
                                  const std::string &residency, const std::string &retry_setting = "true");
struct LtxDevicePolicy {
    bool cuda = false;
    int64_t gpu_index = 0;
    std::string dtype;
    bool try_bnb = false;
};
LtxDevicePolicy ltx_device_policy(const std::string &device, bool cuda_available,
                                int compute_major, double free_gb);
}
