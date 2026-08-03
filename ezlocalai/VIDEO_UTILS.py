import math
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

DEFAULT_VIDEO_MODEL = "unsloth/LTX-2.3-GGUF"
DEFAULT_MUSIC_VIDEO_SCENE_DURATION = 5.0
DEFAULT_MUSIC_VIDEO_MAX_SCENE_DURATION = 20.0
DEFAULT_VIDEO_MODEL_OFFLOAD_MIN_FREE_GB = 16.0
DEFAULT_VIDEO_FULL_GPU_MIN_FREE_GB = 30.0
DEFAULT_VIDEO_SHORT_MODEL_OFFLOAD_MIN_FREE_GB = 12.0
DEFAULT_VIDEO_MODEL_OFFLOAD_MAX_FRAMES = 129
DEFAULT_VIDEO_MODEL_OFFLOAD_MAX_PIXELS = 512 * 512

PYTHAGOREAN_TERMS = (
    "pythagorean",
    "hypotenuse",
    "a squared",
    "a²",
    "b squared",
    "b²",
    "c squared",
    "c²",
    "right triangle",
)


def ltx_frame_count_for_duration(duration: float, frame_rate: int = 24) -> int:
    """Return an LTX-compatible frame count for a target duration."""
    fps = max(1, int(frame_rate or 24))
    frames = max(9, int(math.ceil(float(duration) * fps)))
    return max(9, (((frames - 1) + 7) // 8) * 8 + 1)


def plan_music_video_scenes(
    duration: float,
    scene_duration: Optional[float] = None,
    frame_rate: int = 24,
    max_scene_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Split a song duration into short LTX scene specs."""
    total = max(0.0, float(duration or 0))
    if total <= 0:
        return []

    default_scene = float(
        scene_duration
        if scene_duration is not None
        else os.getenv(
            "MUSIC_VIDEO_SCENE_DURATION", str(DEFAULT_MUSIC_VIDEO_SCENE_DURATION)
        )
    )
    max_scene = float(
        max_scene_duration
        if max_scene_duration is not None
        else os.getenv(
            "MUSIC_VIDEO_MAX_SCENE_DURATION",
            str(DEFAULT_MUSIC_VIDEO_MAX_SCENE_DURATION),
        )
    )
    scene_len = min(max(1.0, default_scene), max(1.0, max_scene))

    scenes = []
    start = 0.0
    index = 0
    while start < total - 0.001:
        length = min(scene_len, total - start)
        scenes.append(
            {
                "index": index,
                "start": round(start, 3),
                "duration": round(length, 3),
                "num_frames": ltx_frame_count_for_duration(length, frame_rate),
            }
        )
        start += length
        index += 1
    return scenes


def ffconcat_quote(path: str) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _parse_size(size: str, default_width: int = 768, default_height: int = 512):
    try:
        width, height = str(size).lower().split("x", 1)
        return max(64, int(width)), max(64, int(height))
    except (TypeError, ValueError):
        return default_width, default_height


def _pil_font(size_px: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size_px)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_power_equation(
    draw,
    center_x: int,
    y: int,
    base_font,
    exp_font,
    fill,
    stroke_width: int = 0,
    stroke_fill=None,
):
    parts = [
        ("a", "2"),
        (" + ", None),
        ("b", "2"),
        (" = ", None),
        ("c", "2"),
    ]
    widths = []
    total_width = 0
    for text, exponent in parts:
        text_bbox = draw.textbbox((0, 0), text, font=base_font)
        part_width = text_bbox[2] - text_bbox[0]
        if exponent:
            exp_bbox = draw.textbbox((0, 0), exponent, font=exp_font)
            part_width += exp_bbox[2] - exp_bbox[0]
        widths.append(part_width)
        total_width += part_width

    x = int(center_x - total_width / 2)
    exp_size = getattr(exp_font, "size", 12)
    for (text, exponent), part_width in zip(parts, widths):
        draw.text(
            (x, y),
            text,
            font=base_font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        text_bbox = draw.textbbox((0, 0), text, font=base_font)
        text_width = text_bbox[2] - text_bbox[0]
        if exponent:
            draw.text(
                (x + text_width + 1, y - max(4, exp_size // 3)),
                exponent,
                font=exp_font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        x += part_width


def _clean_lyric_lines(lyrics: str) -> List[str]:
    lines = []
    for line in str(lyrics or "").splitlines():
        cleaned = re.sub(r"\[[^\]]+\]", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _scene_lyric_excerpt(lyrics: str, scene_index: int, scene_count: int) -> str:
    lines = _clean_lyric_lines(lyrics)
    if not lines:
        return ""
    if scene_count <= 1:
        return lines[0]
    idx = min(
        len(lines) - 1,
        max(0, int(round((scene_index / max(1, scene_count - 1)) * (len(lines) - 1)))),
    )
    return lines[idx]


def is_pythagorean_theorem_request(prompt: str = "", lyrics: str = "") -> bool:
    combined = f"{prompt or ''} {lyrics or ''}".lower()
    return any(term in combined for term in PYTHAGOREAN_TERMS)


def make_music_video_storyboard_image(
    prompt: str,
    lyrics: str,
    scene_index: int,
    scene_count: int,
    size: str = "768x512",
    keyscale: str = "",
):
    """Create a deterministic first-frame storyboard image for music-video scenes."""
    from PIL import Image, ImageDraw

    width, height = _parse_size(size)
    image = Image.new("RGB", (width, height), (5, 5, 10))
    draw = ImageDraw.Draw(image)

    title_font = _pil_font(max(18, height // 13), True)
    mid_font = _pil_font(max(14, height // 18), True)
    small_font = _pil_font(max(10, height // 30), False)
    tiny_font = _pil_font(max(9, height // 38), False)

    # Stage wash and light beams.
    for y in range(height):
        red = int(12 + 38 * y / max(1, height))
        blue = int(18 + 28 * (1 - y / max(1, height)))
        draw.line([(0, y), (width, y)], fill=(red, 8, blue))
    for x in (width * 0.15, width * 0.35, width * 0.66, width * 0.85):
        draw.polygon(
            [
                (int(x), 0),
                (int(x - width * 0.18), int(height * 0.72)),
                (int(x + width * 0.18), int(height * 0.72)),
            ],
            fill=(55, 10, 22),
        )
    draw.rectangle(
        [0, int(height * 0.68), width, height],
        fill=(8, 8, 12),
        outline=(90, 90, 110),
    )

    # LED screen.
    screen = (
        int(width * 0.08),
        int(height * 0.09),
        int(width * 0.92),
        int(height * 0.62),
    )
    draw.rounded_rectangle(screen, radius=8, fill=(3, 5, 12), outline=(170, 35, 45), width=3)

    phase = scene_index % 6
    header = "PYTHAGOREAN THEOREM"
    lyric = _scene_lyric_excerpt(lyrics, scene_index, scene_count)
    theorem_mode = is_pythagorean_theorem_request(prompt, lyrics)

    def center_text(text: str, y: int, text_font, fill=(245, 245, 245)):
        bbox = draw.textbbox((0, 0), text, font=text_font)
        draw.text(((width - (bbox[2] - bbox[0])) // 2, y), text, font=text_font, fill=fill)

    title_text = header if theorem_mode else str(prompt or "MUSIC VIDEO")[:40].upper()
    center_text(title_text, int(height * 0.12), title_font)
    if theorem_mode:
        _draw_power_equation(
            draw,
            width // 2,
            int(height * 0.22),
            title_font,
            mid_font,
            fill=(255, 225, 110),
        )
    else:
        center_text(
            "LYRIC / PERFORMANCE SCENE",
            int(height * 0.22),
            title_font,
            fill=(255, 225, 110),
        )

    left = int(width * 0.18)
    top = int(height * 0.35)
    tri_w = int(width * 0.25)
    tri_h = int(height * 0.18)
    p1 = (left, top + tri_h)
    p2 = (left + tri_w, top + tri_h)
    p3 = (left, top)
    triangle_color = (230, 245, 255)
    proof_color = (95, 155, 255) if phase in {1, 2, 5} else (240, 70, 70)
    if theorem_mode:
        draw.line([p1, p2, p3, p1], fill=triangle_color, width=4)
        draw.line(
            [
                p1,
                (left + int(tri_w * 0.18), top + tri_h),
                (left, top + int(tri_h * 0.82)),
            ],
            fill=(255, 255, 255),
            width=2,
        )
        draw.text(
            (left + tri_w // 2, top + tri_h + 5),
            "a",
            font=mid_font,
            fill=(255, 120, 120),
        )
        draw.text(
            (left - 24, top + tri_h // 2),
            "b",
            font=mid_font,
            fill=(130, 190, 255),
        )
        draw.text(
            (left + tri_w // 2, top + tri_h // 2 - 8),
            "c",
            font=mid_font,
            fill=(255, 235, 120),
        )
    else:
        for bar in range(10):
            bar_x = left + bar * max(8, tri_w // 10)
            bar_h = int((0.25 + ((bar * 7 + scene_index) % 9) / 12) * tri_h)
            draw.rectangle(
                [bar_x, top + tri_h - bar_h, bar_x + 5, top + tri_h],
                fill=(90, 150, 255),
            )

    if theorem_mode and phase in {1, 2, 4}:
        draw.rectangle(
            [left, top + tri_h + 35, left + tri_w, top + tri_h + 35 + min(70, tri_w)],
            outline=(255, 90, 90),
            width=3,
        )
        draw.rectangle(
            [left - min(70, tri_h) - 12, top, left - 12, top + tri_h],
            outline=(90, 150, 255),
            width=3,
        )
        draw.text((left + tri_w + 20, top + tri_h - 5), "area proof", font=small_font, fill=proof_color)

    right_x = int(width * 0.52)
    theorem_titles = [
        "RIGHT TRIANGLE RIFF",
        "LEG A + LEG B",
        "AREA SQUARES",
        "HYPOTENUSE SOLO",
        "CHORUS: C SQUARED",
        "THEOREM ON FIRE",
    ]
    generic_titles = [
        "VERSE VISUAL",
        "PERFORMANCE CLOSE-UP",
        "LYRIC SCREEN",
        "INSTRUMENT BREAK",
        "CHORUS HIT",
        "FINAL HOOK",
    ]
    scene_title = (theorem_titles if theorem_mode else generic_titles)[phase]
    draw.text((right_x, int(height * 0.35)), scene_title, font=mid_font, fill=(255, 245, 210))
    fallback_line = (
        "The squares on legs a and b equal the square on hypotenuse c"
        if theorem_mode
        else "Visible lyric and performance anchor"
    )
    for row, line in enumerate(textwrap.wrap(lyric or fallback_line, width=28)[:3]):
        draw.text((right_x, int(height * 0.44) + row * (small_font.size + 6)), line, font=small_font, fill=(225, 235, 255))
    draw.text(
        (right_x, int(height * 0.58)),
        f"Scene {scene_index + 1}/{scene_count}   {keyscale}".strip(),
        font=tiny_font,
        fill=(180, 190, 210),
    )

    # Simple band silhouettes at the bottom so the storyboard still reads as a concert.
    base_y = int(height * 0.78)
    for cx in (int(width * 0.22), int(width * 0.50), int(width * 0.76)):
        draw.ellipse([cx - 10, base_y - 48, cx + 10, base_y - 28], fill=(20, 20, 24))
        draw.rectangle([cx - 7, base_y - 28, cx + 7, base_y + 18], fill=(18, 18, 22))
    draw.ellipse(
        [
            int(width * 0.44),
            base_y - 22,
            int(width * 0.56),
            base_y + 38,
        ],
        outline=(150, 150, 170),
        width=3,
    )

    return image


def make_pythagorean_equation_overlay_image(size: str = "768x512"):
    """Create a transparent overlay with exact superscript theorem notation."""
    from PIL import Image, ImageDraw

    width, height = _parse_size(size)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    panel_w = min(int(width * 0.62), max(260, width - 48))
    panel_h = max(44, int(height * 0.135))
    panel_x = (width - panel_w) // 2
    panel_y = height - panel_h - max(10, int(height * 0.04))
    draw.rounded_rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        radius=6,
        fill=(2, 4, 10, 190),
        outline=(255, 205, 75, 235),
        width=2,
    )

    equation_font = _pil_font(max(22, panel_h // 2), True)
    exponent_font = _pil_font(max(12, panel_h // 3), True)

    _draw_power_equation(
        draw,
        width // 2,
        panel_y + max(9, (panel_h - getattr(equation_font, "size", 22)) // 2),
        equation_font,
        exponent_font,
        fill=(255, 225, 95, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 230),
    )
    return image


def choose_video_gpu_residency(
    configured_mode: Optional[str],
    total_gb: float,
    free_gb: float,
    text_encoder_on_gpu: bool,
    num_frames: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    model_offload_min_free_gb: float = DEFAULT_VIDEO_MODEL_OFFLOAD_MIN_FREE_GB,
    full_gpu_min_free_gb: float = DEFAULT_VIDEO_FULL_GPU_MIN_FREE_GB,
    short_model_offload_min_free_gb: float = (
        DEFAULT_VIDEO_SHORT_MODEL_OFFLOAD_MIN_FREE_GB
    ),
    model_offload_max_frames: int = DEFAULT_VIDEO_MODEL_OFFLOAD_MAX_FRAMES,
    model_offload_max_pixels: int = DEFAULT_VIDEO_MODEL_OFFLOAD_MAX_PIXELS,
) -> str:
    """Choose how aggressively LTX should use the GPU."""
    mode = (configured_mode or "auto").strip().lower()
    aliases = {
        "gpu": "full",
        "cuda": "full",
        "resident": "full",
        "model": "model_offload",
        "model_cpu_offload": "model_offload",
        "offload": "model_offload",
        "seq": "sequential",
        "sequential_cpu_offload": "sequential",
    }
    mode = aliases.get(mode, mode)
    if mode in {"full", "model_offload", "sequential"}:
        return mode

    try:
        requested_frames = int(num_frames or 0)
    except (TypeError, ValueError):
        requested_frames = 0
    try:
        requested_pixels = int(width or 0) * int(height or 0)
    except (TypeError, ValueError):
        requested_pixels = 0

    if text_encoder_on_gpu and free_gb >= full_gpu_min_free_gb:
        return "full"
    short_clip = (
        requested_frames > 0
        and requested_frames <= int(model_offload_max_frames)
        and (
            requested_pixels <= 0
            or requested_pixels <= int(model_offload_max_pixels)
        )
    )
    if short_clip and free_gb >= short_model_offload_min_free_gb:
        return "model_offload"
    if free_gb >= model_offload_min_free_gb or total_gb >= 40.0:
        return "model_offload"
    return "sequential"
