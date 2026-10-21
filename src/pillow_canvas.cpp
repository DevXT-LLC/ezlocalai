// Pillow 12.3 ImageDraw/_imagingft algorithms adapted to C++, using native
// FreeType and Raqm directly. See licenses/Pillow.txt.
#include "pillow_canvas.hpp"
#include "pillow_drawing.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include FT_STROKER_H
#include <raqm.h>
#include <png.h>
#include <unicode/unistr.h>

namespace ezlocalai::pillow {
namespace {
void ft_check(FT_Error error) { if (error) throw std::runtime_error("FreeType error " + std::to_string(error)); }
int pixel(long value) { return ((value + 32) & -64L) >> 6; }
unsigned divide255(unsigned value) { value += 128; return ((value >> 8) + value) >> 8; }
std::vector<std::string> split(const std::string &text) {
    std::vector<std::string> lines; size_t begin = 0;
    for (;;) { auto end = text.find('\n', begin); lines.push_back(text.substr(begin, end - begin)); if (end == std::string::npos) break; begin = end + 1; }
    return lines;
}
struct Font {
    FT_Face face = nullptr;
    ~Font() { if (face) FT_Done_Face(face); }
};
struct Glyph { unsigned index; int advance_x, advance_y, offset_x, offset_y; };
std::vector<Glyph> layout(FT_Face face, const std::string &text) {
    if (text.empty()) return {};
    std::unique_ptr<raqm_t, decltype(&raqm_destroy)> rq(raqm_create(), raqm_destroy);
    if (!rq) throw std::bad_alloc();
    const auto unicode = icu::UnicodeString::fromUTF8(text);
    std::vector<uint32_t> points;
    for (int i = 0; i < unicode.length();) { const auto ch = unicode.char32At(i); points.push_back(ch); i += U16_LENGTH(ch); }
    if (!raqm_set_text(rq.get(), points.data(), points.size())
        || !raqm_set_par_direction(rq.get(), RAQM_DIRECTION_DEFAULT)
        || !raqm_set_freetype_face(rq.get(), face) || !raqm_layout(rq.get()))
        throw std::runtime_error("Raqm text layout failed");
    size_t count = 0; const auto glyphs = raqm_get_glyphs(rq.get(), &count);
    if (!glyphs) throw std::runtime_error("Raqm glyph retrieval failed");
    std::vector<Glyph> result;
    for (size_t i = 0; i < count; ++i)
        result.push_back({glyphs[i].index, glyphs[i].x_advance, glyphs[i].y_advance, glyphs[i].x_offset, glyphs[i].y_offset});
    return result;
}
Box bounds(FT_Face face, const std::vector<Glyph> &glyphs, int flags = FT_LOAD_DEFAULT) {
    long position = 0;
    int xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    for (const auto &g : glyphs) {
        const auto px = pixel(position + g.offset_x), py = pixel(g.offset_y);
        position += g.advance_x; xmax = std::max(xmax, pixel(position));
        ft_check(FT_Load_Glyph(face, g.index, flags));
        FT_Glyph glyph; ft_check(FT_Get_Glyph(face->glyph, &glyph));
        FT_BBox box; FT_Glyph_Get_CBox(glyph, FT_GLYPH_BBOX_PIXELS, &box); FT_Done_Glyph(glyph);
        xmin = std::min<long>(xmin, box.xMin + px); xmax = std::max<long>(xmax, box.xMax + px);
        ymin = std::min<long>(ymin, box.yMin + py); ymax = std::max<long>(ymax, box.yMax + py);
    }
    const int ascender = glyphs.empty() ? 0 : pixel(face->size->metrics.ascender);
    return {xmin, ascender - ymax, xmax, ascender - ymin};
}
}
struct Canvas::Impl {
    int width, height; bool alpha;
    std::vector<uint8_t> data;
    std::vector<int32_t *> rows32;
    std::vector<char *> rows;
    ImagingMemoryInstance view;
    FT_Library library = nullptr;
    std::map<std::pair<int, bool>, std::unique_ptr<Font>> fonts;
    Impl(int w, int h, Color color, bool a) : width(w), height(h), alpha(a) {
        if (w <= 0 || h <= 0 || size_t(w) > std::vector<uint8_t>().max_size() / 4 / size_t(h))
            throw std::invalid_argument("Invalid storyboard dimensions");
        data.resize(size_t(w) * h * 4);
        if (!alpha) color[3] = 255;
        for (size_t i = 0; i < data.size(); i += 4) std::copy(color.begin(), color.end(), data.begin() + i);
        for (int y = 0; y < h; ++y) {
            rows32.push_back(reinterpret_cast<int32_t *>(data.data() + size_t(y) * w * 4));
            rows.push_back(reinterpret_cast<char *>(data.data() + size_t(y) * w * 4));
        }
        view.xsize = w; view.ysize = h; view.image32 = rows32.data(); view.image = rows.data();
        ft_check(FT_Init_FreeType(&library));
    }
    ~Impl() { fonts.clear(); if (library) FT_Done_FreeType(library); }
    FT_Face font(int size, bool bold) {
        auto &entry = fonts[{size, bold}];
        if (entry) return entry->face;
        auto candidate = std::make_unique<Font>();
        const auto name = bold ? "DejaVuSans-Bold.ttf" : "DejaVuSans.ttf";
        for (const auto *directory : {"/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"}) {
            if (FT_New_Face(library, (std::filesystem::path(directory) / name).c_str(), 0, &candidate->face) == 0) break;
        }
        if (!candidate->face) throw std::runtime_error("Storyboard DejaVu font unavailable");
        FT_Size_RequestRec request{}; request.type = FT_SIZE_REQUEST_TYPE_NOMINAL;
        request.width = request.height = size * 64;
        ft_check(FT_Request_Size(candidate->face, &request));
        entry = std::move(candidate); return entry->face;
    }
    void render(int origin_x, int origin_y, const std::string &text, FT_Face face, Color color, int stroke) {
        const auto glyphs = layout(face, text);
        if (glyphs.empty()) return;
        int flags = stroke ? FT_LOAD_NO_BITMAP : FT_LOAD_DEFAULT;
        const auto box = bounds(face, glyphs, flags);
        const int w = box[2] - box[0] + 2 * stroke, h = box[3] - box[1] + 2 * stroke;
        if (w == 0 || h == 0) return;
        std::vector<uint8_t> mask(size_t(w) * h, 0);
        int x = 0, y = 0, xmin = 0, ymax = 0;
        for (const auto &g : glyphs) {
            const int px = pixel(x + g.offset_x), py = pixel(y + g.offset_y);
            ft_check(FT_Load_Glyph(face, g.index, flags | FT_LOAD_RENDER));
            ymax = std::max(ymax, face->glyph->bitmap_top + py);
            xmin = std::min(xmin, face->glyph->bitmap_left + px);
            x += g.advance_x; y += g.advance_y;
        }
        x = (-xmin + stroke) * 64; y = (-ymax - stroke) * 64;
        FT_Stroker raw_stroker = nullptr;
        if (stroke) {
            ft_check(FT_Stroker_New(library, &raw_stroker));
            FT_Stroker_Set(raw_stroker, stroke * 64, FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0);
        } else flags |= FT_LOAD_RENDER;
        std::unique_ptr<std::remove_pointer_t<FT_Stroker>, decltype(&FT_Stroker_Done)> stroker(raw_stroker, FT_Stroker_Done);
        for (const auto &g : glyphs) {
            const int px = pixel(x + g.offset_x), py = pixel(y + g.offset_y);
            ft_check(FT_Load_Glyph(face, g.index, flags));
            FT_Bitmap bitmap; int xx, yy; FT_Glyph glyph = nullptr;
            if (stroke) {
                ft_check(FT_Get_Glyph(face->glyph, &glyph));
                ft_check(FT_Glyph_StrokeBorder(&glyph, stroker.get(), 0, 1));
                FT_Vector zero{0, 0}; ft_check(FT_Glyph_To_Bitmap(&glyph, FT_RENDER_MODE_NORMAL, &zero, 1));
                const auto rendered = reinterpret_cast<FT_BitmapGlyph>(glyph);
                bitmap = rendered->bitmap; xx = px + rendered->left; yy = -(py + rendered->top);
            } else {
                bitmap = face->glyph->bitmap; xx = px + face->glyph->bitmap_left; yy = -(py + face->glyph->bitmap_top);
            }
            if (bitmap.buffer) {
                if (bitmap.pixel_mode != FT_PIXEL_MODE_GRAY) {
                    if (glyph) FT_Done_Glyph(glyph);
                    throw std::runtime_error("Unsupported storyboard glyph bitmap");
                }
                for (unsigned by = 0; by < bitmap.rows; ++by) {
                    if (yy + int(by) < 0 || yy + int(by) >= h) continue;
                    for (unsigned bx = 0; bx < bitmap.width; ++bx) {
                        if (xx + int(bx) < 0 || xx + int(bx) >= w) continue;
                        const unsigned coverage = bitmap.buffer[by * bitmap.pitch + bx];
                        auto &target = mask[size_t(yy + by) * w + xx + bx];
                        target = std::min(255u, coverage + divide255(target * (255 - coverage)));
                    }
                }
            }
            if (glyph) FT_Done_Glyph(glyph);
            x += g.advance_x; y += g.advance_y;
        }
        const int left = origin_x + box[0] - stroke, top = origin_y + box[1] - stroke;
        for (int my = 0; my < h; ++my) {
            if (top + my < 0 || top + my >= height) continue;
            for (int mx = 0; mx < w; ++mx) {
                if (left + mx < 0 || left + mx >= width) continue;
                const unsigned coverage = mask[size_t(my) * w + mx];
                auto *target = data.data() + (size_t(top + my) * width + left + mx) * 4;
                for (int c = 0; c < (alpha ? 4 : 3); ++c)
                    target[c] = divide255(target[c] * (255 - coverage) + color[c] * coverage);
            }
        }
    }
};
Canvas::Canvas(int w, int h, Color color, bool alpha) : impl_(std::make_unique<Impl>(w, h, color, alpha)) {}
Canvas::~Canvas() = default;
void Canvas::line(const std::vector<Point> &points, Color color, int width) {
    for (size_t i = 1; i < points.size(); ++i) {
        const auto [x0, y0] = points[i - 1]; const auto [x1, y1] = points[i];
        if (width == 1) ImagingDrawLine(&impl_->view, x0, y0, x1, y1, color.data(), 0);
        else ImagingDrawWideLine(&impl_->view, x0, y0, x1, y1, color.data(), width, 0, nullptr);
    }
    if (width == 1 && points.size() > 1) ImagingDrawPoint(&impl_->view, points.back().first, points.back().second, color.data(), 0);
}
void Canvas::polygon(const std::vector<Point> &points, Color color) {
    std::vector<int> xy; for (const auto &[x, y] : points) { xy.push_back(x); xy.push_back(y); }
    ImagingDrawPolygon(&impl_->view, points.size(), xy.data(), color.data(), 1, 1, 0, nullptr);
}
void Canvas::rectangle(Box b, std::optional<Color> fill, std::optional<Color> outline, int width) {
    if (b[2] < b[0] || b[3] < b[1]) throw std::invalid_argument("Invalid rectangle bounds");
    if (fill) ImagingDrawRectangle(&impl_->view, b[0], b[1], b[2], b[3], fill->data(), 1, 1, 0);
    if (outline && outline != fill && width) ImagingDrawRectangle(&impl_->view, b[0], b[1], b[2], b[3], outline->data(), 0, width, 0);
}
void Canvas::ellipse(Box b, std::optional<Color> fill, std::optional<Color> outline, int width) {
    if (b[2] < b[0] || b[3] < b[1]) throw std::invalid_argument("Invalid ellipse bounds");
    if (fill) ImagingDrawEllipse(&impl_->view, b[0], b[1], b[2], b[3], fill->data(), 1, 1, 0);
    if (outline && outline != fill && width) ImagingDrawEllipse(&impl_->view, b[0], b[1], b[2], b[3], outline->data(), 0, width, 0);
}
void Canvas::rounded_rectangle(Box b, int radius, Color fill, Color outline, int width) {
    const auto [x0, y0, x1, y1] = b;
    if (x1 < x0 || y1 < y0) throw std::invalid_argument("Invalid rounded rectangle bounds");
    int d = std::min({x1 - x0, y1 - y0, radius * 2});
    const bool full_x = d >= x1 - x0 - 1;
    if (full_x) d = x1 - x0;
    const bool full_y = d >= y1 - y0 - 1;
    if (full_y) d = y1 - y0;
    if (full_x && full_y) { ellipse(b, fill, outline, width); return; }
    if (!d) { rectangle(b, fill, outline, width); return; }
    const int r = d / 2;
    struct Arc { Box box; int start, end; }; std::vector<Arc> parts;
    if (full_x) parts = {{{x0, y0, x0 + d, y0 + d}, 180, 360}, {{x0, y1 - d, x0 + d, y1}, 0, 180}};
    else if (full_y) parts = {{{x0, y0, x0 + d, y0 + d}, 90, 270}, {{x1 - d, y0, x1, y0 + d}, 270, 90}};
    else parts = {{{x0, y0, x0 + d, y0 + d}, 180, 270}, {{x1 - d, y0, x1, y0 + d}, 270, 360},
                  {{x1 - d, y1 - d, x1, y1}, 0, 90}, {{x0, y1 - d, x0 + d, y1}, 90, 180}};
    for (const auto &part : parts) {
        const auto &a = part.box;
        ImagingDrawPieslice(&impl_->view, a[0], a[1], a[2], a[3], part.start, part.end, fill.data(), 1, 1, 0);
    }
    if (full_x) rectangle({x0, y0 + r + 1, x1, y1 - r - 1}, fill);
    else if (x1 - r - 1 >= x0 + r + 1) rectangle({x0 + r + 1, y0, x1 - r - 1, y1}, fill);
    if (!full_x && !full_y) {
        rectangle({x0, y0 + r + 1, x0 + r, y1 - r - 1}, fill);
        rectangle({x1 - r, y0 + r + 1, x1, y1 - r - 1}, fill);
    }
    if (outline != fill && width) {
        for (const auto &part : parts) {
            const auto &a = part.box;
            ImagingDrawArc(&impl_->view, a[0], a[1], a[2], a[3], part.start, part.end, outline.data(), width, 0);
        }
        if (!full_x) {
            rectangle({x0 + r + 1, y0, x1 - r - 1, y0 + width - 1}, outline);
            rectangle({x0 + r + 1, y1 - width + 1, x1 - r - 1, y1}, outline);
        }
        if (!full_y) {
            rectangle({x0, y0 + r + 1, x0 + width - 1, y1 - r - 1}, outline);
            rectangle({x1 - width + 1, y0 + r + 1, x1, y1 - r - 1}, outline);
        }
    }
}
Box Canvas::text_bbox(const std::string &text, int size, bool bold) {
    const auto face = impl_->font(size, bold); const auto lines = split(text);
    const int spacing = bounds(face, layout(face, "A"))[3] + 4;
    Box result{}; bool first = true; int y = 0;
    for (const auto &line : lines) {
        auto b = bounds(face, layout(face, line)); b[1] += y; b[3] += y;
        if (first) { result = b; first = false; }
        else { result[0] = std::min(result[0], b[0]); result[1] = std::min(result[1], b[1]); result[2] = std::max(result[2], b[2]); result[3] = std::max(result[3], b[3]); }
        y += spacing;
    }
    return result;
}
void Canvas::text(int x, int y, const std::string &text, int size, bool bold, Color color, int stroke, Color stroke_color) {
    const auto face = impl_->font(size, bold);
    const int spacing = bounds(face, layout(face, "A"))[3] + 2 * stroke + 4;
    for (const auto &line : split(text)) {
        if (stroke) impl_->render(x, y, line, face, stroke_color, stroke);
        if (!stroke || stroke_color != color) impl_->render(x, y, line, face, color, 0);
        y += spacing;
    }
}
StoryboardImage Canvas::image() const {
    StoryboardImage result{impl_->width, impl_->height, impl_->alpha, {}};
    if (result.alpha) result.pixels = impl_->data;
    else {
        result.pixels.reserve(size_t(result.width) * result.height * 3);
        for (size_t i = 0; i < impl_->data.size(); i += 4) result.pixels.insert(result.pixels.end(), impl_->data.begin() + i, impl_->data.begin() + i + 3);
    }
    return result;
}
}

namespace ezlocalai {
std::string encode_storyboard_png(const StoryboardImage &image) {
    png_image png{}; png.version = PNG_IMAGE_VERSION; png.width = image.width; png.height = image.height;
    png.format = image.alpha ? PNG_FORMAT_RGBA : PNG_FORMAT_RGB;
    png_alloc_size_t bytes = 0;
    if (!png_image_write_to_memory(&png, nullptr, &bytes, 0, image.pixels.data(), 0, nullptr))
        throw std::runtime_error(png.message);
    std::string result(bytes, '\0');
    if (!png_image_write_to_memory(&png, result.data(), &bytes, 0, image.pixels.data(), 0, nullptr))
        throw std::runtime_error(png.message);
    result.resize(bytes); return result;
}
}
