#include "ezlocalai/image_processing.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); expect(bool(in), "Cannot open " + path.string()); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &value) {
    std::vector<std::string> out; size_t start = 0;
    do { const auto end = value.find('\t', start); out.push_back(value.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) break; start = end + 1; } while (true);
    return out;
}
std::string sha256(const uint8_t *data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH]; SHA256(data, size, hash);
    std::ostringstream out; for (auto byte : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return out.str();
}
std::string sha256(const std::string &value) { return sha256(reinterpret_cast<const uint8_t *>(value.data()), value.size()); }
std::string sha256(const ezlocalai::RgbImage &image) { return sha256(image.pixels.data(), image.pixels.size()); }
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
}
int main(int argc, char **argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--decode") {
            const auto image = ezlocalai::decode_rgb_image(read(argv[2]));
            std::ofstream out(argv[3], std::ios::binary); out.write(reinterpret_cast<const char *>(image.pixels.data()), image.pixels.size());
            std::cout << image.width << 'x' << image.height << '\n'; return 0;
        }
        expect(argc == 4, "reference checkout and image/input fixtures required");
        const fs::path fixture = argv[2];
        std::istringstream lines(read(fixture));
        std::map<std::string, ezlocalai::RgbImage> decoded;
        size_t count = 0, failures = 0;
        for (std::string line; std::getline(lines, line);) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                const auto marker = line.find(" sha256=");
                if (line.find("# ezlocalai/") == 0 && marker != std::string::npos)
                    expect(sha256(read(fs::path(argv[1]) / line.substr(2, marker - 2))) == line.substr(marker + 8), "Reference image processing call changed");
                continue;
            }
            const auto c = split(line); expect(c.size() == 8, "image fixture columns"); ++count;
            try {
                const auto bytes = read(fixture.parent_path() / "images" / c[0]);
                expect(sha256(bytes) == c[5], c[0] + " fixture data changed");
                if (c[6] == "ERROR") {
                    bool caught = false; try { ezlocalai::decode_rgb_image(bytes); } catch (const std::exception &) { caught = true; }
                    expect(caught, c[0] + " should fail to decode"); continue;
                }
                if (!decoded.count(c[0])) {
                    auto image = ezlocalai::decode_rgb_image(bytes);
                    expect(image.width == std::stoi(c[3]) && image.height == std::stoi(c[4]), c[0] + " decoded dimensions");
                    expect(sha256(image) == c[6], c[0] + " decoded RGB pixels differ");
                    const auto roundtrip = ezlocalai::decode_rgb_image(ezlocalai::encode_rgb_png(image));
                    expect(roundtrip.pixels == image.pixels && roundtrip.width == image.width && roundtrip.height == image.height, c[0] + " lossless PNG encoding");
                    decoded[c[0]] = std::move(image);
                }
                const auto fitted = ezlocalai::fit_rgb_lanczos(decoded.at(c[0]), std::stoi(c[1]), std::stoi(c[2]));
                expect(fitted.width == std::stoi(c[1]) && fitted.height == std::stoi(c[2]), "fit dimensions");
                expect(sha256(fitted) == c[7], c[0] + " fit " + c[1] + "x" + c[2] + " pixels differ");
            } catch (const std::exception &e) { if (++failures <= 20) std::cerr << e.what() << '\n'; }
        }
        expect(failures == 0, std::to_string(failures) + " image differential failures");
        expect(count == 979, "all image fixture cases");
        std::istringstream input_lines(read(argv[3])); size_t input_count = 0;
        for (std::string line; std::getline(input_lines, line);) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (line.rfind("# app.py sha256=", 0) == 0)
                    expect(sha256(read(fs::path(argv[1]) / "app.py")) == line.substr(16), "Reference image input decoder changed");
                continue;
            }
            const auto c = split(line); expect(c.size() == 5, "image input columns"); ++input_count;
            if (c[2] == "ERROR") {
                bool caught = false; try { ezlocalai::decode_base64_rgb_image(unhex(c[1])); } catch (const std::exception &) { caught = true; }
                expect(caught, c[0] + " input should fail");
            } else {
                const auto image = ezlocalai::decode_base64_rgb_image(unhex(c[1]));
                expect(image.width == std::stoi(c[2]) && image.height == std::stoi(c[3]) && sha256(image) == c[4], c[0] + " decoded base64 image pixels");
            }
        }
        expect(input_count == 181, "all video image input cases");
        std::cout << "Verified " << count << " Pillow image decode/fit cases pixel-for-pixel and " << input_count << " video image inputs\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
