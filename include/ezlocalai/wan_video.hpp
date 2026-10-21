#pragma once
#include "ezlocalai/hub_assets.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ezlocalai {
bool is_wan_model(const std::string &model);
std::vector<HubAsset> wan_model_assets(bool image_to_video, const std::string &quant);

struct WanVideoSettings {
    std::string size = "768x512";
    int64_t frames = 121, fps = 24, steps = 18;
    double guidance = 3.5;
    int width = 768, height = 512;
};
WanVideoSettings normalize_wan_settings(WanVideoSettings settings);
std::vector<std::string> wan_video_command(const std::string &binary, const std::string &device,
    const std::map<std::string, std::string> &assets, const std::string &output,
    const std::string &prompt, const std::string &negative_prompt,
    const WanVideoSettings &settings, const std::string &image_path = "");

class WanVideo {
public:
    explicit WanVideo(std::string model, std::string device);
    std::string generate(const std::string &prompt, WanVideoSettings settings,
                         const std::string &local_uri, const std::string &image_bytes = "",
                         bool conditions = false,
                         const std::string &negative_prompt = "low quality, blurry, distorted, static");
private:
    std::string model_, device_, binary_;
};
} // namespace ezlocalai
