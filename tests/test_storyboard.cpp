#include "../src/pillow_canvas.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
namespace {
using Json = nlohmann::json;
using namespace ezlocalai::pillow;
Color color(const Json &v) { return {v.at(0).get<uint8_t>(), v.at(1).get<uint8_t>(), v.at(2).get<uint8_t>(), v.size() == 4 ? v.at(3).get<uint8_t>() : uint8_t(255)}; }
std::optional<Color> optional_color(const Json &v) { return v.is_null() ? std::nullopt : std::optional<Color>(color(v)); }
std::string sha(const std::vector<uint8_t> &data) {
    unsigned char hash[EVP_MAX_MD_SIZE]; unsigned length;
    EVP_Digest(data.data(), data.size(), hash, &length, EVP_sha256(), nullptr);
    std::ostringstream out;
    for (unsigned i = 0; i < length; ++i) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(hash[i]);
    return out.str();
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--render") {
            std::ifstream input(argv[2]); Json args; input >> args;
            const auto rendered = ezlocalai::music_video_storyboard(args.at("prompt"), args.at("lyrics"),
                args.at("scene_index"), args.at("scene_count"), args.at("size"), args.value("keyscale", ""));
            const auto png = ezlocalai::encode_storyboard_png(rendered);
            std::ofstream output(argv[3], std::ios::binary); output.write(png.data(), png.size());
            std::ofstream(argv[4]) << sha(rendered.pixels) << '\n';
            if (!output) throw std::runtime_error("Cannot write rendered storyboard");
            return 0;
        }
        if (argc != 2) throw std::runtime_error("fixture required");
        std::ifstream file(argv[1]); Json fixture; file >> fixture;
        size_t index = 0;
        for (const auto &row : fixture.at("cases")) {
            if (row.contains("kind")) {
                const auto &args = row.at("args");
                const auto rendered = row.at("kind") == "overlay"
                    ? ezlocalai::pythagorean_equation_overlay(args.at("size"))
                    : ezlocalai::music_video_storyboard(args.at("prompt"), args.at("lyrics"),
                        args.at("scene_index"), args.at("scene_count"), args.at("size"), args.value("keyscale", ""));
                if (rendered.width != row.at("width") || rendered.height != row.at("height")
                    || rendered.alpha != row.at("alpha") || sha(rendered.pixels) != row.at("sha256")) {
                    std::ofstream image("/tmp/ezlocalai-storyboard-mismatch.png", std::ios::binary);
                    const auto png = ezlocalai::encode_storyboard_png(rendered); image.write(png.data(), png.size());
                    throw std::runtime_error("composition case " + std::to_string(index) + " mismatch; hash "
                        + sha(rendered.pixels) + " expected " + row.at("sha256").get<std::string>());
                }
                ++index;
                continue;
            }
            Canvas canvas(row.at("width"), row.at("height"), color(row.at("background")), row.value("alpha", false));
            Json boxes = Json::array();
            for (const auto &op : row.at("operations")) {
                const auto kind = op.at("kind").get<std::string>();
                const auto fill = optional_color(op.value("fill", Json())), outline = optional_color(op.value("outline", Json()));
                const int width = op.value("stroke", 1);
                if (kind == "line" || kind == "polygon") {
                    std::vector<Point> points; for (const auto &p : op.at("points")) points.emplace_back(p.at(0), p.at(1));
                    if (kind == "line") canvas.line(points, *fill, width); else canvas.polygon(points, *fill);
                } else if (kind == "rectangle") canvas.rectangle(op.at("box").get<Box>(), fill, outline, width);
                else if (kind == "ellipse") canvas.ellipse(op.at("box").get<Box>(), fill, outline, width);
                else if (kind == "rounded") canvas.rounded_rectangle(op.at("box").get<Box>(), op.at("radius"), *fill, *outline, width);
                else if (kind == "text") {
                    boxes.push_back(canvas.text_bbox(op.at("text"), op.at("size"), op.value("bold", false)));
                    canvas.text(op.at("x"), op.at("y"), op.at("text"), op.at("size"), op.value("bold", false), *fill,
                        op.value("stroke", 0), color(op.value("stroke_fill", Json::array({0,0,0,255}))));
                } else throw std::runtime_error("unknown operation");
            }
            const auto rendered = canvas.image();
            if (boxes != row.at("bounds") || sha(rendered.pixels) != row.at("sha256")) {
                std::ofstream image("/tmp/ezlocalai-storyboard-mismatch.png", std::ios::binary);
                const auto png = ezlocalai::encode_storyboard_png(rendered); image.write(png.data(), png.size());
                throw std::runtime_error("case " + std::to_string(index) + " mismatch; boxes " + boxes.dump() + " expected "
                    + row.at("bounds").dump() + "; hash " + sha(rendered.pixels) + " expected " + row.at("sha256").get<std::string>());
            }
            ++index;
        }
        std::cout << "Verified " << index << " installed Pillow rendering cases\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
