#pragma once
#include "ezlocalai/hub_assets.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ezlocalai {
std::vector<HubAsset> qwen_image_assets();
std::pair<int64_t, int64_t> qwen_image_dimensions(const std::string &size);
struct QwenImageSettings {
    std::string size = "1024x1024";
    std::string negative_prompt;
    int64_t steps = 20;
    double guidance = 6.0, strength = 0.75;
    std::optional<std::string> image;
    std::vector<std::string> images;
};

// The model argument in Python IMG is an identity for lifecycle reporting;
// its engine and four weights always select Qwen-Image-2.1.
class QwenImage {
public:
    explicit QwenImage(std::string device = "cpu");
    bool available() const { return available_; }
    std::string generate(const std::string &prompt, const QwenImageSettings &settings,
                         const std::string &local_uri = "");
private:
    std::string device_, binary_, models_;
    bool available_ = false;
};
} // namespace ezlocalai
