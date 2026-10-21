// Composition follows ezlocalai/VIDEO_UTILS.py; raster/text behavior is in
// pillow_canvas.cpp. The Python runtime is not used by these renderers.
#include "ezlocalai/storyboard.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/music_video.hpp"
#include "pillow_canvas.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unicode/locid.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>

namespace ezlocalai {
namespace {
using pillow::Canvas;
using pillow::Color;
Color rgb(int r, int g, int b, int a = 255) { return {uint8_t(r), uint8_t(g), uint8_t(b), uint8_t(a)}; }
std::pair<int, int> dimensions(const std::string &size) {
    auto text = size; for (auto &ch : text) if (ch == 'X') ch = 'x';
    const auto split = text.find('x');
    if (split == std::string::npos) return {768, 512};
    const auto w = parse_configuration_integer(text.substr(0, split)), h = parse_configuration_integer(text.substr(split + 1));
    if (!w || !h) return {768, 512};
    auto dimension = [](const std::string &value) {
        if (configuration_integer_less(value, "64")) return 64;
        if (configuration_integer_less(std::to_string(std::numeric_limits<int>::max()), value))
            throw std::overflow_error("Storyboard dimension exceeds native range");
        return std::stoi(value);
    };
    return {dimension(*w), dimension(*h)};
}
int mod(int64_t value, int divisor) { const int result = value % divisor; return result < 0 ? result + divisor : result; }
int half_floor(int value) { return value / 2 - (value < 0 && value % 2 ? 1 : 0); }
std::string title(const std::string &prompt) {
    auto text = icu::UnicodeString::fromUTF8(prompt.empty() ? "MUSIC VIDEO" : prompt);
    text.truncate(text.moveIndex32(0, 40)); text.toUpper(icu::Locale::getRoot());
    std::string result; text.toUTF8String(result); return result;
}
std::u32string points(const std::string &text) {
    const auto unicode = icu::UnicodeString::fromUTF8(text); std::u32string result;
    for (int i = 0; i < unicode.length();) { const auto ch = unicode.char32At(i); result.push_back(ch); i += U16_LENGTH(ch); }
    return result;
}
std::string utf8(const std::u32string &text) {
    icu::UnicodeString unicode; for (auto ch : text) unicode.append(UChar32(ch));
    std::string result; unicode.toUTF8String(result); return result;
}
bool blank(const std::u32string &value) { return strip_config_value(utf8(value)).empty(); }
std::vector<std::string> wrapped(const std::string &input) {
    // CPython textwrap's default word splitter, with Python's Unicode \w
    // and letter-like classes expanded explicitly for ICU's regex engine.
    static const auto pattern = [] {
        UErrorCode status = U_ZERO_ERROR;
        auto result = std::unique_ptr<icu::RegexPattern>(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(R"re(
            ( [\t\n\v\f\r\x20]+
            | (?<=[\p{L}\p{N}_!"'&.,?]) -{2,} (?=[\p{L}\p{N}_])
            | [^\t\n\v\f\r\x20]+? (?:
                 - (?:(?<=[\p{L}\p{Nl}\p{No}_]{2}-)|(?<=[\p{L}\p{Nl}\p{No}_]-[\p{L}\p{Nl}\p{No}_]-))
                    (?=[\p{L}\p{Nl}\p{No}_]-?[\p{L}\p{Nl}\p{No}_])
               | (?=[\t\n\v\f\r\x20]|\Z)
               | (?<=[\p{L}\p{N}_!"'&.,?]) (?=-{2,}[\p{L}\p{N}_])
            ) )
        )re"), UREGEX_COMMENTS, status));
        if (U_FAILURE(status)) throw std::runtime_error("Storyboard wrapping expression failed");
        return result;
    }();
    std::u32string expanded; size_t column = 0;
    for (const auto ch : points(input)) {
        if (ch == '\t') { const auto count = 8 - column % 8; expanded.append(count, U' '); column += count; }
        else {
            expanded.push_back(ch == '\n' || ch == '\v' || ch == '\f' || ch == '\r' ? U' ' : ch);
            if (ch == '\n' || ch == '\r') column = 0; else ++column;
        }
    }
    const auto text = icu::UnicodeString::fromUTF8(utf8(expanded));
    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(text, status));
    if (U_FAILURE(status)) throw std::runtime_error("Storyboard wrapping matcher failed");
    std::vector<std::u32string> chunks;
    while (matcher->find(status)) { std::string chunk; matcher->group(status).toUTF8String(chunk); chunks.push_back(points(chunk)); }
    if (U_FAILURE(status)) throw std::runtime_error("Storyboard wrapping failed");
    std::reverse(chunks.begin(), chunks.end());
    std::vector<std::string> result;
    constexpr size_t width = 28;
    while (!chunks.empty()) {
        std::vector<std::u32string> line; size_t length = 0;
        if (blank(chunks.back()) && !result.empty()) chunks.pop_back();
        while (!chunks.empty() && length + chunks.back().size() <= width) {
            length += chunks.back().size(); line.push_back(chunks.back()); chunks.pop_back();
        }
        if (!chunks.empty() && chunks.back().size() > width) {
            const auto space = width - length;
            size_t end = space; const auto &chunk = chunks.back();
            const auto hyphen = chunk.substr(0, space).rfind(U'-');
            if (hyphen != std::u32string::npos && hyphen > 0
                && chunk.substr(0, hyphen).find_first_not_of(U'-') != std::u32string::npos) end = hyphen + 1;
            line.push_back(chunk.substr(0, end)); chunks.back().erase(0, end);
        }
        if (!line.empty() && blank(line.back())) line.pop_back();
        if (!line.empty()) { std::u32string joined; for (const auto &part : line) joined += part; result.push_back(utf8(joined)); }
    }
    return result;
}
void equation(Canvas &draw, int center, int y, int base_size, int exponent_size, Color fill,
    int stroke = 0, Color stroke_fill = {0, 0, 0, 255}) {
    const std::vector<std::pair<std::string, bool>> parts = {{"a", true}, {" + ", false}, {"b", true}, {" = ", false}, {"c", true}};
    std::vector<int> widths; int total = 0;
    for (const auto &[text, power] : parts) {
        const auto base = draw.text_bbox(text, base_size, true); int width = base[2] - base[0];
        if (power) { const auto exp = draw.text_bbox("2", exponent_size, true); width += exp[2] - exp[0]; }
        widths.push_back(width); total += width;
    }
    int x = static_cast<int>(center - total / 2.0);
    for (size_t i = 0; i < parts.size(); ++i) {
        const auto &[text, power] = parts[i];
        draw.text(x, y, text, base_size, true, fill, stroke, stroke_fill);
        const auto base = draw.text_bbox(text, base_size, true);
        if (power) draw.text(x + base[2] - base[0] + 1, y - std::max(4, exponent_size / 3), "2", exponent_size, true, fill, stroke, stroke_fill);
        x += widths[i];
    }
}
}
StoryboardImage music_video_storyboard(const std::string &prompt, const std::string &lyrics,
    int64_t index, int64_t count, const std::string &size, const std::string &keyscale) {
    const auto [w, h] = dimensions(size);
    Canvas draw(w, h, rgb(5, 5, 10));
    const int large = std::max(18, h / 13), middle = std::max(14, h / 18), small = std::max(10, h / 30), tiny = std::max(9, h / 38);
    for (int y = 0; y < h; ++y)
        draw.line({{0, y}, {w, y}}, rgb(int(12 + 38.0 * y / std::max(1, h)), 8, int(18 + 28 * (1 - double(y) / std::max(1, h)))));
    for (double x : {w * .15, w * .35, w * .66, w * .85})
        draw.polygon({{int(x), 0}, {int(x - w * .18), int(h * .72)}, {int(x + w * .18), int(h * .72)}}, rgb(55, 10, 22));
    draw.rectangle({0, int(h * .68), w, h}, rgb(8, 8, 12), rgb(90, 90, 110));
    draw.rounded_rectangle({int(w * .08), int(h * .09), int(w * .92), int(h * .62)}, 8, rgb(3, 5, 12), rgb(170, 35, 45), 3);
    const int phase = mod(index, 6);
    const bool theorem = is_pythagorean_request(prompt, lyrics);
    const auto lyric = music_video_lyric_excerpt(lyrics, index, count);
    auto centered = [&](const std::string &text, int y, int font, Color color) {
        const auto box = draw.text_bbox(text, font, true);
        draw.text(half_floor(w - (box[2] - box[0])), y, text, font, true, color);
    };
    centered(theorem ? "PYTHAGOREAN THEOREM" : title(prompt), int(h * .12), large, rgb(245, 245, 245));
    if (theorem) equation(draw, w / 2, int(h * .22), large, middle, rgb(255, 225, 110));
    else centered("LYRIC / PERFORMANCE SCENE", int(h * .22), large, rgb(255, 225, 110));
    const int left = int(w * .18), top = int(h * .35), tw = int(w * .25), th = int(h * .18);
    const pillow::Point p1{left, top + th}, p2{left + tw, top + th}, p3{left, top};
    const auto proof = phase == 1 || phase == 2 || phase == 5 ? rgb(95, 155, 255) : rgb(240, 70, 70);
    if (theorem) {
        draw.line({p1, p2, p3, p1}, rgb(230, 245, 255), 4);
        draw.line({p1, {left + int(tw * .18), top + th}, {left, top + int(th * .82)}}, rgb(255, 255, 255), 2);
        draw.text(left + tw / 2, top + th + 5, "a", middle, true, rgb(255, 120, 120));
        draw.text(left - 24, top + th / 2, "b", middle, true, rgb(130, 190, 255));
        draw.text(left + tw / 2, top + th / 2 - 8, "c", middle, true, rgb(255, 235, 120));
    } else for (int bar = 0; bar < 10; ++bar) {
        const int bx = left + bar * std::max(8, tw / 10), bh = int((.25 + mod(bar * 7 + index, 9) / 12.0) * th);
        draw.rectangle({bx, top + th - bh, bx + 5, top + th}, rgb(90, 150, 255));
    }
    if (theorem && (phase == 1 || phase == 2 || phase == 4)) {
        draw.rectangle({left, top + th + 35, left + tw, top + th + 35 + std::min(70, tw)}, {}, rgb(255, 90, 90), 3);
        draw.rectangle({left - std::min(70, th) - 12, top, left - 12, top + th}, {}, rgb(90, 150, 255), 3);
        draw.text(left + tw + 20, top + th - 5, "area proof", small, false, proof);
    }
    const int right = int(w * .52);
    const char *theorem_titles[] = {"RIGHT TRIANGLE RIFF", "LEG A + LEG B", "AREA SQUARES", "HYPOTENUSE SOLO", "CHORUS: C SQUARED", "THEOREM ON FIRE"};
    const char *generic_titles[] = {"VERSE VISUAL", "PERFORMANCE CLOSE-UP", "LYRIC SCREEN", "INSTRUMENT BREAK", "CHORUS HIT", "FINAL HOOK"};
    draw.text(right, int(h * .35), theorem ? theorem_titles[phase] : generic_titles[phase], middle, true, rgb(255, 245, 210));
    const auto fallback = theorem ? "The squares on legs a and b equal the square on hypotenuse c" : "Visible lyric and performance anchor";
    const auto lines = wrapped(lyric.empty() ? fallback : lyric);
    for (size_t row = 0; row < std::min<size_t>(3, lines.size()); ++row)
        draw.text(right, int(h * .44) + row * (small + 6), lines[row], small, false, rgb(225, 235, 255));
    draw.text(right, int(h * .58), strip_config_value("Scene " + std::to_string(index + 1) + "/" + std::to_string(count) + "   " + keyscale), tiny, false, rgb(180, 190, 210));
    const int base = int(h * .78);
    for (int cx : {int(w * .22), int(w * .50), int(w * .76)}) {
        draw.ellipse({cx - 10, base - 48, cx + 10, base - 28}, rgb(20, 20, 24));
        draw.rectangle({cx - 7, base - 28, cx + 7, base + 18}, rgb(18, 18, 22));
    }
    draw.ellipse({int(w * .44), base - 22, int(w * .56), base + 38}, {}, rgb(150, 150, 170), 3);
    return draw.image();
}
StoryboardImage pythagorean_equation_overlay(const std::string &size) {
    const auto [w, h] = dimensions(size);
    Canvas draw(w, h, rgb(0, 0, 0, 0), true);
    const int panel_w = std::min(int(w * .62), std::max(260, w - 48)), panel_h = std::max(44, int(h * .135));
    const int x = (w - panel_w) / 2, y = h - panel_h - std::max(10, int(h * .04));
    draw.rounded_rectangle({x, y, x + panel_w, y + panel_h}, 6, rgb(2, 4, 10, 190), rgb(255, 205, 75, 235), 2);
    const int base_font = std::max(22, panel_h / 2), exponent = std::max(12, panel_h / 3);
    equation(draw, w / 2, y + std::max(9, (panel_h - base_font) / 2), base_font, exponent,
        rgb(255, 225, 95, 255), 2, rgb(0, 0, 0, 230));
    return draw.image();
}
}
