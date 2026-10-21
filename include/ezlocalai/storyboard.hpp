#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace ezlocalai {
struct StoryboardImage {
    int width = 0, height = 0;
    bool alpha = false;
    std::vector<uint8_t> pixels;
};
std::string encode_storyboard_png(const StoryboardImage &image);
StoryboardImage music_video_storyboard(const std::string &prompt, const std::string &lyrics,
    int64_t scene_index, int64_t scene_count, const std::string &size = "768x512", const std::string &keyscale = "");
StoryboardImage pythagorean_equation_overlay(const std::string &size = "768x512");
}
