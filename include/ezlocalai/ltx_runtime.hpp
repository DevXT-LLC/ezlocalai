#pragma once
#include "ezlocalai/ltx_video.hpp"
#include "ezlocalai/media_engine_process.hpp"
#include <string>
#include <string_view>
#include <vector>

namespace ezlocalai {
// All three installed LTX-2.3 pipelines calculate mu using max_image_seq_len,
// yielding max_shift=2.05 for the reference checkpoint, regardless of clip size.
inline constexpr std::string_view ltx_reference_scheduler_args = "base_shift=2.05,max_shift=2.05";
bool is_ltx_model(const std::string &model);
struct LtxCondition {
    std::string image_bytes;
    int64_t index = 0;
    double strength = 1;
};
struct LtxAssets { std::string transformer, components, tokenizer; };
struct LtxTextEncoderSettings {
    std::string weight_type = "bf16";
    int nf4_sm_count = 0, nf4_threads_per_sm = 0;
};
LtxAssets prepare_ltx_assets(const std::string &model);
std::vector<std::string> ltx_video_command(const std::string &binary, const std::string &device,
    const LtxAssets &assets, const std::string &output, const std::string &prompt,
    const LtxVideoSettings &settings, const std::string &residency,
    const std::string &image_path = "", const std::string &conditions_json = "",
    const LtxTextEncoderSettings &text_encoder = {});
class LtxVideo {
public:
    LtxVideo(std::string model, std::string device, std::optional<LtxVideoSettings> hint = {});
    void warm();
    bool loaded() const { return loaded_; }
    const std::string &residency() const { return residency_; }
    std::string generate(const std::string &prompt, LtxVideoSettings settings,
        const std::string &local_uri, const std::string &image_bytes = "",
        const std::vector<LtxCondition> &conditions = {}, bool include_audio = true);
private:
    FileProcessResult execute(const std::vector<std::string> &arguments, bool preload,
                              const std::string &log, double timeout);
    std::string model_, device_, binary_;
    LtxAssets assets_;
    LtxTextEncoderSettings text_encoder_;
    VideoResidencyPolicy placement_observation_;
    std::string residency_;
    MediaEngineProcess engine_;
    bool loaded_ = false;
    bool placement_selected_ = false;
};
}
