// The fit/resampling and CMYK conversion algorithms are adapted from Pillow
// 11.3.0 (ImageOps.py and libImaging). See licenses/Pillow.txt.
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/process.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <png.h>
#include <jpeglib.h>
#include <jerror.h>
#include <webp/decode.h>
#include <webp/demux.h>

namespace ezlocalai {
namespace {
size_t pixel_bytes(int width, int height) {
    if (width <= 0 || height <= 0 || static_cast<uint64_t>(width) * height > 178956970)
        throw std::runtime_error("Image size exceeds the reference decompression limit");
    return static_cast<size_t>(width) * height * 3;
}
void validate(const RgbImage &image) {
    if (image.pixels.size() != pixel_bytes(image.width, image.height)) throw std::invalid_argument("Invalid RGB image storage");
}
struct PngState {
    const std::string *input = nullptr;
    size_t offset = 0;
    std::string output, error;
    RgbImage image;
    std::vector<uint8_t> raw;
    std::vector<png_bytep> rows;
    int depth = 8, color = PNG_COLOR_TYPE_RGB;
    std::vector<png_color> palette;
    std::vector<png_byte> alpha;
    png_color_16 transparent{};
    bool has_transparent = false;
};
void png_failure(png_structp png, png_const_charp message) {
    auto *state = static_cast<PngState *>(png_get_error_ptr(png));
    state->error = message;
    png_longjmp(png, 1);
}
void png_warning_ignored(png_structp, png_const_charp) {}
void png_read_bytes(png_structp png, png_bytep output, png_size_t size) {
    auto *state = static_cast<PngState *>(png_get_io_ptr(png));
    if (size > state->input->size() - state->offset) png_error(png, "Truncated PNG image");
    std::memcpy(output, state->input->data() + state->offset, size); state->offset += size;
}
void png_write_bytes(png_structp png, png_bytep bytes, png_size_t size) {
    static_cast<PngState *>(png_get_io_ptr(png))->output.append(reinterpret_cast<const char *>(bytes), size);
}
RgbImage decode_png(const std::string &bytes) {
    // Containers live on the heap before setjmp: no C++ objects are skipped by
    // libpng's error jump, and modified automatic variables are not inspected.
    auto state = std::make_unique<PngState>(); state->input = &bytes;
    auto *png = png_create_read_struct(PNG_LIBPNG_VER_STRING, state.get(), png_failure, png_warning_ignored);
    if (!png) throw std::bad_alloc();
    auto *info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); throw std::bad_alloc(); }
    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, nullptr); throw std::runtime_error(state->error); }
    try {
        png_set_read_fn(png, state.get(), png_read_bytes);
        png_read_info(png, info);
        const auto width = png_get_image_width(png, info), height = png_get_image_height(png, info);
        if (width > INT32_MAX || height > INT32_MAX) throw std::runtime_error("Image dimensions are too large");
        state->image.width = width; state->image.height = height;
        state->image.pixels.resize(pixel_bytes(width, height));
        const int type = png_get_color_type(png, info), depth = png_get_bit_depth(png, info);
        const bool gray16 = type == PNG_COLOR_TYPE_GRAY && depth == 16;
        if (type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
        if (type == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
        if (depth == 16 && !gray16) png_set_strip_16(png);
        // Palette expansion also expands tRNS into an alpha channel. Pillow's
        // convert("RGB") discards that channel without compositing its colors.
        png_set_strip_alpha(png);
        if (!gray16 && (type == PNG_COLOR_TYPE_GRAY || type == PNG_COLOR_TYPE_GRAY_ALPHA)) png_set_gray_to_rgb(png);
        png_set_interlace_handling(png);
        png_read_update_info(png, info);
        const auto stride = png_get_rowbytes(png, info);
        state->raw.resize(stride * height); state->rows.resize(height);
        for (size_t y = 0; y < height; ++y) state->rows[y] = state->raw.data() + y * stride;
        png_read_image(png, state->rows.data());
        for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
            auto *target = state->image.pixels.data() + (y * width + x) * 3;
            if (gray16) { const auto *p = state->rows[y] + x * 2; target[0] = target[1] = target[2] = p[0] ? 255 : p[1]; }
            else std::memcpy(target, state->rows[y] + x * 3, 3);
        }
    } catch (...) { png_destroy_read_struct(&png, &info, nullptr); throw; }
    png_destroy_read_struct(&png, &info, nullptr);
    return std::move(state->image);
}
struct JpegState {
    jpeg_decompress_struct decoder{};
    jpeg_error_mgr error_manager{};
    jmp_buf jump;
    char error[JMSG_LENGTH_MAX]{};
    RgbImage image;
    std::vector<uint8_t> row;
    bool truncated = false;
};
void jpeg_failure(j_common_ptr decoder) {
    auto *state = static_cast<JpegState *>(decoder->client_data);
    (*decoder->err->format_message)(decoder, state->error);
    longjmp(state->jump, 1);
}
void jpeg_message(j_common_ptr decoder, int level) {
    if (level < 0 && decoder->err->msg_code == JWRN_JPEG_EOF)
        static_cast<JpegState *>(decoder->client_data)->truncated = true;
}
RgbImage decode_jpeg(const std::string &bytes) {
    auto state = std::make_unique<JpegState>();
    state->decoder.err = jpeg_std_error(&state->error_manager);
    state->error_manager.error_exit = jpeg_failure; state->error_manager.emit_message = jpeg_message;
    state->decoder.client_data = state.get();
    if (setjmp(state->jump)) { jpeg_destroy_decompress(&state->decoder); throw std::runtime_error(state->error); }
    jpeg_create_decompress(&state->decoder);
    try {
        jpeg_mem_src(&state->decoder, reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
        jpeg_read_header(&state->decoder, TRUE);
        const bool cmyk = state->decoder.num_components == 4;
        state->decoder.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
        pixel_bytes(state->decoder.image_width, state->decoder.image_height);
        jpeg_start_decompress(&state->decoder);
        state->image.width = state->decoder.output_width; state->image.height = state->decoder.output_height;
        state->image.pixels.resize(pixel_bytes(state->image.width, state->image.height));
        state->row.resize(static_cast<size_t>(state->image.width) * state->decoder.output_components);
        while (state->decoder.output_scanline < state->decoder.output_height) {
            const auto y = state->decoder.output_scanline;
            auto *row = state->row.data(); jpeg_read_scanlines(&state->decoder, &row, 1);
            auto *target = state->image.pixels.data() + static_cast<size_t>(y) * state->image.width * 3;
            if (!cmyk) std::memcpy(target, row, static_cast<size_t>(state->image.width) * 3);
            else for (int x = 0; x < state->image.width; ++x) for (int channel = 0; channel < 3; ++channel) {
                // Pillow decodes JPEG to CMYK;I, then applies cmyk2rgb.
                const int k = row[x * 4 + 3], ink = 255 - row[x * 4 + channel];
                const int product = ink * k + 128;
                target[x * 3 + channel] = k - (((product >> 8) + product) >> 8);
            }
        }
        jpeg_finish_decompress(&state->decoder);
        if (state->truncated) throw std::runtime_error("Truncated JPEG image");
    } catch (...) { jpeg_destroy_decompress(&state->decoder); throw; }
    jpeg_destroy_decompress(&state->decoder);
    return std::move(state->image);
}
RgbImage decode_webp(const std::string &bytes) {
    WebPBitstreamFeatures features{};
    const auto *input = reinterpret_cast<const uint8_t *>(bytes.data());
    if (WebPGetFeatures(input, bytes.size(), &features) != VP8_STATUS_OK) throw std::runtime_error("Invalid WebP image");
    RgbImage image{features.width, features.height, std::vector<uint8_t>(pixel_bytes(features.width, features.height))};
    WebPData data{input, bytes.size()}; WebPAnimDecoderOptions options{}; WebPAnimDecoderOptionsInit(&options);
    options.color_mode = MODE_RGBA;
    std::unique_ptr<WebPAnimDecoder, decltype(&WebPAnimDecoderDelete)> decoder(WebPAnimDecoderNew(&data, &options), WebPAnimDecoderDelete);
    uint8_t *rgba = nullptr; int timestamp = 0;
    if (!decoder || !WebPAnimDecoderGetNext(decoder.get(), &rgba, &timestamp)) throw std::runtime_error("Cannot decode WebP image");
    for (size_t i = 0; i < image.pixels.size() / 3; ++i) std::memcpy(image.pixels.data() + i * 3, rgba + i * 4, 3);
    return image;
}
RgbImage decode_other_image(const std::string &bytes) {
    // Preserve the previously supported formats while their native decoder
    // parity is audited. Resampling below is always the Pillow algorithm.
    auto name = (std::filesystem::temp_directory_path() / "ezlocalai-image-XXXXXX").string();
    if (!mkdtemp(name.data())) throw std::system_error(errno, std::generic_category(), "image temporary directory");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); } } cleanup{name};
    const auto input = (cleanup.path / "input").string(), output = (cleanup.path / "image.png").string();
    std::ofstream file(input, std::ios::binary); file.write(bytes.data(), bytes.size()); file.close();
    const auto result = run_process_to_files({"ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input, "-vf", "format=rgb24", "-frames:v", "1", output}, "/dev/null", (cleanup.path / "error").string(), 180);
    if (result.status || result.timed_out) throw std::runtime_error("Cannot decode video input image");
    std::ifstream encoded(output, std::ios::binary);
    return decode_png({std::istreambuf_iterator<char>(encoded), std::istreambuf_iterator<char>()});
}

struct Coefficients { int first; std::vector<int32_t> weights; };
std::vector<Coefficients> coefficients(int input, float begin, float end, int output, bool bilinear) {
    constexpr int precision = 22;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double scale = static_cast<double>(end - begin) / output;
    const double filter_scale = std::max(1.0, scale), support = (bilinear ? 1.0 : 3.0) * filter_scale;
    std::vector<Coefficients> result(output);
    auto sinc = [](double value) { if (value == 0.0) return 1.0; value *= pi; return std::sin(value) / value; };
    for (int i = 0; i < output; ++i) {
        const double center = begin + (i + 0.5) * scale;
        const int first = std::max(0, static_cast<int>(center - support + 0.5));
        const int last = std::min(input, static_cast<int>(center + support + 0.5));
        std::vector<double> weights(last - first); double sum = 0.0;
        for (int x = first; x < last; ++x) {
            const double distance = (x - center + 0.5) * (1.0 / filter_scale);
            const double weight = bilinear ? std::max(0.0, 1.0 - std::abs(distance))
                : (-3.0 <= distance && distance < 3.0 ? sinc(distance) * sinc(distance / 3.0) : 0.0);
            weights[x - first] = weight; sum += weight;
        }
        result[i].first = first;
        for (auto weight : weights) {
            if (sum != 0.0) weight /= sum;
            result[i].weights.push_back(static_cast<int32_t>((weight < 0 ? -0.5 : 0.5) + weight * (1 << precision)));
        }
    }
    return result;
}
RgbImage resample(const RgbImage &input, int width, int height, const std::array<float, 4> &box, bool bilinear = false) {
    const bool horizontal = width != input.width || box[0] != 0 || box[2] != width;
    const bool vertical = height != input.height || box[1] != 0 || box[3] != height;
    RgbImage work = input;
    if (horizontal) {
        const auto table = coefficients(input.width, box[0], box[2], width, bilinear);
        RgbImage output{width, input.height, std::vector<uint8_t>(pixel_bytes(width, input.height))};
        for (int y = 0; y < input.height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
            int32_t sum = 1 << 21;
            for (size_t k = 0; k < table[x].weights.size(); ++k) sum += input.pixels[(static_cast<size_t>(y) * input.width + table[x].first + k) * 3 + c] * table[x].weights[k];
            output.pixels[(static_cast<size_t>(y) * width + x) * 3 + c] = static_cast<uint8_t>(std::clamp(sum >> 22, 0, 255));
        }
        work = std::move(output);
    }
    if (!vertical) return work;
    const auto table = coefficients(input.height, box[1], box[3], height, bilinear);
    RgbImage output{width, height, std::vector<uint8_t>(pixel_bytes(width, height))};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        int32_t sum = 1 << 21;
        for (size_t k = 0; k < table[y].weights.size(); ++k) sum += work.pixels[((table[y].first + k) * width + x) * 3 + c] * table[y].weights[k];
        output.pixels[(static_cast<size_t>(y) * width + x) * 3 + c] = static_cast<uint8_t>(std::clamp(sum >> 22, 0, 255));
    }
    return output;
}
} // namespace

RgbImage decode_rgb_image(const std::string &bytes) {
    if (bytes.size() >= 8 && png_sig_cmp(reinterpret_cast<const unsigned char *>(bytes.data()), 0, 8) == 0) return decode_png(bytes);
    if (bytes.size() >= 3 && static_cast<uint8_t>(bytes[0]) == 0xff && static_cast<uint8_t>(bytes[1]) == 0xd8 && static_cast<uint8_t>(bytes[2]) == 0xff) return decode_jpeg(bytes);
    if (bytes.size() >= 12 && bytes.compare(0, 4, "RIFF") == 0 && bytes.compare(8, 4, "WEBP") == 0) return decode_webp(bytes);
    return decode_other_image(bytes);
}
RgbImage decode_base64_rgb_image(std::string input) {
    if (input.rfind("data:", 0) == 0) {
        const auto comma = input.find(',');
        if (comma == std::string::npos) throw std::runtime_error("Invalid image data URL");
        input.erase(0, comma + 1);
    }
    // Match base64.b64decode's default permissive ASCII mode: punctuation is
    // ignored, but incomplete quartets and non-ASCII strings are rejected.
    for (unsigned char c : input) if (c > 127) throw std::runtime_error("Non-ASCII base64 image");
    std::string bytes;
    uint32_t accumulator = 0; int digits = 0, padding = 0;
    for (unsigned char c : input) {
        if (c == '=') {
            if (digits >= 2 && digits + ++padding >= 4) return decode_rgb_image(bytes);
            continue;
        }
        const int value = c >= 'A' && c <= 'Z' ? c - 'A'
            : c >= 'a' && c <= 'z' ? c - 'a' + 26
            : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' ? 62 : c == '/' ? 63 : -1;
        if (value < 0) continue;
        padding = 0; accumulator = (accumulator << 6) | value;
        if (++digits >= 2) bytes.push_back(static_cast<char>((accumulator >> (8 - digits * 2)) & 255));
        if (digits == 4) digits = 0;
    }
    if (digits) throw std::runtime_error("Invalid base64 image padding");
    return decode_rgb_image(bytes);
}
std::string reencode_png(const std::string &bytes) {
    auto state = std::make_unique<PngState>(); state->input = &bytes;
    auto *png = png_create_read_struct(PNG_LIBPNG_VER_STRING, state.get(), png_failure, png_warning_ignored);
    if (!png) throw std::bad_alloc();
    auto *info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); throw std::bad_alloc(); }
    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, nullptr); throw std::runtime_error(state->error); }
    try {
        png_set_read_fn(png, state.get(), png_read_bytes); png_read_info(png, info);
        const auto width = png_get_image_width(png, info), height = png_get_image_height(png, info);
        if (width > INT32_MAX || height > INT32_MAX) throw std::runtime_error("Image dimensions are too large");
        pixel_bytes(width, height); state->image.width = width; state->image.height = height;
        state->color = png_get_color_type(png, info); state->depth = png_get_bit_depth(png, info);
        png_colorp palette = nullptr; int palette_size = 0;
        if (png_get_PLTE(png, info, &palette, &palette_size)) state->palette.assign(palette, palette + palette_size);
        png_bytep alpha = nullptr; int alpha_size = 0; png_color_16p transparent = nullptr;
        if (png_get_tRNS(png, info, &alpha, &alpha_size, &transparent)) {
            if (alpha_size && alpha) state->alpha.assign(alpha, alpha + alpha_size);
            if (transparent) { state->transparent = *transparent; state->has_transparent = true; }
        }
        png_set_interlace_handling(png); png_read_update_info(png, info);
        const auto stride = png_get_rowbytes(png, info);
        state->raw.resize(stride * height); state->rows.resize(height);
        for (size_t y = 0; y < height; ++y) state->rows[y] = state->raw.data() + y * stride;
        png_read_image(png, state->rows.data()); png_read_end(png, info);
    } catch (...) { png_destroy_read_struct(&png, &info, nullptr); throw; }
    png_destroy_read_struct(&png, &info, nullptr);
    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, state.get(), png_failure, png_warning_ignored);
    if (!png) throw std::bad_alloc();
    info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); throw std::bad_alloc(); }
    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); throw std::runtime_error(state->error); }
    try {
        png_set_write_fn(png, state.get(), png_write_bytes, nullptr);
        png_set_IHDR(png, info, state->image.width, state->image.height, state->depth, state->color,
            PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        if (!state->palette.empty()) png_set_PLTE(png, info, state->palette.data(), state->palette.size());
        if (!state->alpha.empty() || state->has_transparent) png_set_tRNS(png, info,
            state->alpha.empty() ? nullptr : state->alpha.data(), state->alpha.size(), state->has_transparent ? &state->transparent : nullptr);
        png_set_compression_level(png, 6); png_write_info(png, info);
        png_write_image(png, state->rows.data()); png_write_end(png, info);
    } catch (...) { png_destroy_write_struct(&png, &info); throw; }
    png_destroy_write_struct(&png, &info);
    return std::move(state->output);
}
std::string encode_rgb_png(const RgbImage &image) {
    validate(image);
    auto state = std::make_unique<PngState>();
    auto *png = png_create_write_struct(PNG_LIBPNG_VER_STRING, state.get(), png_failure, png_warning_ignored);
    if (!png) throw std::bad_alloc();
    auto *info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); throw std::bad_alloc(); }
    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); throw std::runtime_error(state->error); }
    try {
        png_set_write_fn(png, state.get(), png_write_bytes, nullptr);
        png_set_IHDR(png, info, image.width, image.height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_set_compression_level(png, 6);
        png_write_info(png, info);
        for (int y = 0; y < image.height; ++y) png_write_row(png, const_cast<png_bytep>(image.pixels.data() + static_cast<size_t>(y) * image.width * 3));
        png_write_end(png, info);
    } catch (...) { png_destroy_write_struct(&png, &info); throw; }
    png_destroy_write_struct(&png, &info);
    return std::move(state->output);
}
namespace {
struct JpegWriteState {
    jpeg_compress_struct encoder{};
    jpeg_error_mgr errors{};
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX]{};
    unsigned char *data = nullptr;
    unsigned long size = 0;
};
void jpeg_write_failure(j_common_ptr encoder) {
    auto *state = static_cast<JpegWriteState *>(encoder->client_data);
    (*encoder->err->format_message)(encoder, state->message);
    longjmp(state->jump, 1);
}
}
std::string encode_rgb_jpeg(const RgbImage &image, int quality) {
    validate(image);
    auto state = std::make_unique<JpegWriteState>();
    state->encoder.err = jpeg_std_error(&state->errors);
    state->errors.error_exit = jpeg_write_failure;
    state->encoder.client_data = state.get();
    if (setjmp(state->jump)) { jpeg_destroy_compress(&state->encoder); std::free(state->data); throw std::runtime_error(state->message); }
    jpeg_create_compress(&state->encoder);
    jpeg_mem_dest(&state->encoder, &state->data, &state->size);
    state->encoder.image_width = image.width; state->encoder.image_height = image.height;
    state->encoder.input_components = 3; state->encoder.in_color_space = JCS_RGB;
    jpeg_set_defaults(&state->encoder); jpeg_set_quality(&state->encoder, quality, TRUE);
    jpeg_start_compress(&state->encoder, TRUE);
    while (state->encoder.next_scanline < state->encoder.image_height) {
        auto *row = const_cast<JSAMPLE *>(image.pixels.data() + static_cast<size_t>(state->encoder.next_scanline) * image.width * 3);
        jpeg_write_scanlines(&state->encoder, &row, 1);
    }
    jpeg_finish_compress(&state->encoder);
    std::string output(reinterpret_cast<char *>(state->data), state->size);
    jpeg_destroy_compress(&state->encoder); std::free(state->data);
    return output;
}
RgbImage resize_rgb_bilinear(const RgbImage &image, int width, int height) {
    validate(image); pixel_bytes(width, height);
    return resample(image, width, height, {0, 0, static_cast<float>(image.width), static_cast<float>(image.height)}, true);
}
RgbImage resize_rgb_lanczos(const RgbImage &image, int width, int height) {
    validate(image); pixel_bytes(width, height);
    return resample(image, width, height, {0, 0, static_cast<float>(image.width), static_cast<float>(image.height)});
}
RgbImage fit_rgb_lanczos(const RgbImage &image, int width, int height) {
    validate(image); pixel_bytes(width, height);
    const double ratio = static_cast<double>(image.width) / image.height, target_ratio = static_cast<double>(width) / height;
    double crop_width = image.width, crop_height = image.height;
    if (ratio > target_ratio) crop_width = target_ratio * image.height;
    else if (ratio < target_ratio) crop_height = image.width / target_ratio;
    const double left = (image.width - crop_width) * 0.5, top = (image.height - crop_height) * 0.5;
    return resample(image, width, height, {static_cast<float>(left), static_cast<float>(top), static_cast<float>(left + crop_width), static_cast<float>(top + crop_height)});
}
} // namespace ezlocalai
