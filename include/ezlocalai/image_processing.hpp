#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ezlocalai {
struct RgbImage {
    int width = 0, height = 0;
    std::vector<uint8_t> pixels;
};

// Image.open(...).convert("RGB"): discard alpha without compositing, and do
// not apply EXIF orientation or ICC/gamma transforms.
RgbImage decode_rgb_image(const std::string &bytes);
RgbImage decode_base64_rgb_image(std::string input);
std::string encode_rgb_png(const RgbImage &image);
// Save a generated PNG without textual metadata, retaining its alpha channel,
// palette and sample depth. Reference images separately use RGB conversion.
std::string reencode_png(const std::string &bytes);
std::string encode_rgb_jpeg(const RgbImage &image, int quality = 85);
// Pillow BILINEAR for LTX image-to-video (distinct from float conditions).
RgbImage resize_rgb_bilinear(const RgbImage &image, int width, int height);
RgbImage resize_rgb_lanczos(const RgbImage &image, int width, int height);
// ImageOps.fit(..., method=Image.Resampling.LANCZOS), default centering/bleed.
RgbImage fit_rgb_lanczos(const RgbImage &image, int width, int height);
} // namespace ezlocalai
