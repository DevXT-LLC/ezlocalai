#pragma once
#include "ezlocalai/storyboard.hpp"
#include <array>
#include <memory>
#include <optional>
#include <utility>
namespace ezlocalai::pillow {
using Color = std::array<uint8_t, 4>;
using Box = std::array<int, 4>;
using Point = std::pair<int, int>;
class Canvas {
public:
    Canvas(int width, int height, Color background, bool alpha = false);
    ~Canvas();
    void line(const std::vector<Point> &points, Color color, int width = 1);
    void polygon(const std::vector<Point> &points, Color color);
    void rectangle(Box box, std::optional<Color> fill, std::optional<Color> outline = {}, int width = 1);
    void ellipse(Box box, std::optional<Color> fill, std::optional<Color> outline = {}, int width = 1);
    void rounded_rectangle(Box box, int radius, Color fill, Color outline, int width = 1);
    Box text_bbox(const std::string &text, int size, bool bold = false);
    void text(int x, int y, const std::string &text, int size, bool bold, Color color,
              int stroke_width = 0, Color stroke_color = {0, 0, 0, 255});
    StoryboardImage image() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
