"""Native-resolution teaching targets, without detector blur or interpolation.

The extended target keeps the original 256x256 phantom byte-for-byte in its
upper-left panel. Its other panels are a vector-derived Pomeranian illustration
and exact equal-width line/gap groups. They are synthetic, non-anatomical test
patterns; they are not animal radiographs or a measured detector resolution.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from PIL import Image, ImageDraw

from simulator.imaging import make_phantom


PATTERN_VERSION = "extended-pomeranian-linepairs-v1"
LEGACY_ROI = (0, 256, 0, 256)
DOG_ROI = (0, 256, 256, 512)
DOG_DETAIL_ROIS = {
    "body_fur": (146, 188, 390, 450),
    "face_and_muzzle": (77, 126, 317, 386),
}
LINE_PAIR_GROUPS = tuple(
    {
        "id": f"line_gap_{width}px",
        "bbox": (280 + 128 * (index // 3), 360 + 128 * (index // 3),
                 16 + 168 * (index % 3), 160 + 168 * (index % 3)),
        "line_width_px": width,
        "gap_width_px": width,
        "period_px": 2 * width,
        "periods": 144 // (2 * width),
        "cycles_per_pixel": 1.0 / (2 * width),
        "orientation": "vertical_lines_vary_along_x",
        "dark_signal": 0.30,
        "bright_signal": 0.90,
    }
    for index, width in enumerate((8, 6, 4, 3, 2, 1))
)
PATTERN_SPECS = {
    "legacy": {
        "label": "従来の幾何学ファントム",
        "version": "legacy-geometry-v1",
        "shape": (256, 256),
        "signal_unit": "dimensionless",
        "physical_pixel_spacing_mm": None,
        "synthetic_only": True,
        "dog_roi": None,
        "line_pair_groups": (),
    },
    "extended": {
        "label": "ポメラニアンと段階的な線対パターン",
        "version": PATTERN_VERSION,
        "shape": (512, 512),
        "signal_unit": "dimensionless",
        "physical_pixel_spacing_mm": None,
        "synthetic_only": True,
        "legacy_roi": LEGACY_ROI,
        "dog_roi": DOG_ROI,
        "dog_detail_rois": DOG_DETAIL_ROIS,
        "dog_vector_asset": "assets/pomeranian_test.svg",
        "line_pair_groups": LINE_PAIR_GROUPS,
        "line_pair_mean_signal": 0.60,
        "blur_applied": False,
        "interpolation_applied": False,
    },
}


def _dog_primitives() -> list[dict]:
    """One shared vector description for native masks and the editable SVG."""
    shapes = []

    def polygon(points, signal, outline=None, width=1):
        shapes.append({"kind": "polygon", "points": points, "signal": signal,
                       "outline": outline, "width": width})

    def ellipse(box, signal, outline=None, width=1):
        shapes.append({"kind": "ellipse", "box": box, "signal": signal,
                       "outline": outline, "width": width})

    def line(points, signal, width=1):
        shapes.append({"kind": "line", "points": points, "signal": signal, "width": width})

    # A fluffy tail rises from the rump and curls over the short back.
    polygon([(165, 148), (158, 131), (151, 118), (144, 108), (148, 99),
             (142, 89), (151, 83), (150, 73), (162, 71), (166, 61),
             (178, 63), (187, 55), (197, 61), (210, 60), (215, 71),
             (226, 75), (225, 87), (236, 95), (230, 106), (233, 118),
             (221, 123), (216, 135), (204, 133), (193, 140), (182, 133),
             (177, 143)], 0.82, 0.53)
    polygon([(164, 126), (155, 107), (158, 88), (172, 75), (191, 70),
             (207, 77), (220, 91), (223, 109), (211, 122), (198, 125),
             (190, 117), (194, 108), (206, 108), (209, 101), (202, 92),
             (188, 91), (179, 103), (181, 126)], 0.92)
    ellipse((181, 88, 207, 110), 1.0)
    for points in (
        [(159, 101), (160, 88), (170, 79)], [(166, 110), (166, 94), (177, 82)],
        [(178, 74), (191, 72), (205, 78)], [(200, 81), (212, 91), (216, 102)],
        [(218, 108), (209, 119), (201, 120)], [(151, 94), (153, 83), (161, 78)],
        [(178, 124), (173, 114), (174, 101)],
    ):
        line(points, 0.65, 1)

    # The compact standing body and four short legs remain separate from the
    # large head and chest ruff; none of these shapes depicts internal anatomy.
    polygon([(112, 177), (134, 180), (131, 213), (139, 224), (133, 232),
             (113, 232), (108, 224), (113, 209)], 0.48, 0.42)
    polygon([(183, 178), (201, 182), (203, 210), (213, 222), (209, 230),
             (187, 231), (182, 223), (184, 207)], 0.51, 0.42)
    polygon([(88, 135), (102, 122), (120, 117), (138, 123), (156, 120),
             (174, 126), (191, 133), (204, 148), (211, 162), (208, 174),
             (211, 184), (201, 191), (200, 202), (184, 211), (168, 214),
             (149, 210), (133, 213), (114, 205), (99, 192), (91, 173)], 0.63, 0.46)
    polygon([(115, 137), (137, 130), (160, 133), (183, 140), (196, 154),
             (197, 169), (178, 166), (164, 158), (144, 163), (127, 154)], 0.74)
    polygon([(120, 175), (139, 181), (160, 183), (181, 176), (199, 180),
             (192, 199), (175, 204), (154, 202), (136, 204), (124, 193)], 0.56)
    polygon([(164, 184), (184, 188), (183, 211), (193, 223), (188, 232),
             (165, 232), (160, 225), (165, 211)], 0.67, 0.46)
    ellipse((160, 219, 192, 234), 0.77, 0.52)
    polygon([(80, 170), (105, 176), (103, 208), (111, 221), (108, 232),
             (83, 235), (76, 226), (81, 209)], 0.72, 0.46)
    ellipse((75, 221, 110, 236), 0.85, 0.54)
    for x, y in ((84, 227), (94, 227), (169, 225), (179, 225), (119, 225), (198, 224)):
        line([(x, y), (x + 1, y + 5)], 0.51, 1)

    # Fine fur strokes on broader mid-gray body shading; native widths are
    # one or two pixels, deliberately retained without smoothing or blur.
    for row in range(5):
        for column in range(7):
            x = 134 + 8 * column + (row % 2) * 3
            y = 145 + 8 * row
            line([(x, y), (x + 3, y + 3), (x + 2, y + 6)],
                 (0.61, 0.79, 0.68, 0.83)[(row + column) % 4], 1)
            if (row + column) % 3 == 0:
                line([(x + 4, y + 2), (x + 6, y + 6)], 0.49, 1)

    # Rich chest ruff with an irregular fluffy outline and layered shading.
    polygon([(61, 70), (78, 57), (98, 51), (117, 59), (130, 70),
             (145, 78), (144, 94), (155, 105), (148, 117), (158, 130),
             (145, 138), (152, 151), (137, 155), (140, 172), (125, 168),
             (121, 185), (109, 179), (104, 195), (92, 185), (81, 196),
             (74, 179), (60, 183), (58, 166), (44, 167), (48, 151),
             (35, 146), (42, 132), (30, 124), (38, 110), (31, 98),
             (43, 86), (44, 72)], 0.83, 0.53)
    polygon([(48, 116), (63, 104), (81, 116), (91, 141), (106, 151),
             (122, 137), (138, 123), (140, 144), (126, 157), (114, 173),
             (99, 183), (86, 169), (74, 176), (68, 157), (52, 154)], 0.92)
    polygon([(45, 122), (59, 136), (62, 155), (78, 170), (76, 181),
             (63, 171), (59, 154), (47, 145)], 0.70)
    polygon([(124, 124), (143, 114), (136, 135), (143, 148), (129, 150),
             (129, 167), (117, 166), (110, 183), (108, 163)], 0.76)
    for angle in np.linspace(0.20, math.pi - 0.20, 19):
        x, y = 94 + 34 * math.cos(angle), 119 + 35 * math.sin(angle)
        outward_x, outward_y = 9 * math.cos(angle), 14 * math.sin(angle)
        line([(x, y), (x + outward_x * 0.5 - 2, y + outward_y * 0.5),
              (x + outward_x, y + outward_y)], 0.65, 1)
        line([(x + 3, y - 3), (x + outward_x + 3, y + outward_y - 4)], 0.87, 1)
    for x, y in ((52, 110), (44, 121), (54, 137), (68, 150), (82, 166),
                 (99, 168), (114, 156), (129, 141), (139, 119)):
        line([(x, y), (x - 2, y + 5), (x + 1, y + 9)], 0.59, 1)

    # Upright triangular ears, round fluffy head, small eyes and short muzzle.
    polygon([(49, 60), (51, 35), (57, 16), (66, 22), (84, 47)], 0.65, 0.43)
    polygon([(56, 48), (59, 27), (74, 46)], 0.39)
    polygon([(105, 47), (120, 23), (132, 14), (139, 40), (137, 66)], 0.66, 0.43)
    polygon([(115, 48), (129, 26), (132, 52)], 0.40)
    polygon([(46, 70), (51, 53), (63, 45), (76, 44), (91, 39), (106, 45),
             (122, 46), (133, 57), (140, 74), (136, 89), (144, 100),
             (133, 112), (125, 124), (108, 132), (91, 132), (76, 127),
             (60, 120), (51, 108), (42, 99), (48, 86)], 0.78, 0.55)
    polygon([(62, 58), (80, 49), (96, 47), (115, 55), (126, 71),
             (117, 82), (99, 77), (83, 82), (65, 78)], 0.86)
    polygon([(51, 87), (64, 77), (79, 78), (86, 91), (78, 102),
             (61, 108), (50, 101)], 0.69)
    polygon([(106, 80), (123, 77), (135, 88), (132, 104), (118, 108),
             (107, 98)], 0.70)
    polygon([(55, 104), (72, 99), (88, 106), (96, 120), (86, 128),
             (70, 122), (60, 116)], 0.91)
    polygon([(107, 105), (122, 101), (134, 106), (126, 119),
             (113, 128), (100, 122)], 0.90)
    ellipse((75, 103, 119, 129), 0.95)
    ellipse((67, 85, 77, 96), 0.28)
    ellipse((111, 84, 121, 95), 0.28)
    ellipse((69, 86, 71, 88), 0.98)
    ellipse((113, 85, 115, 87), 0.98)
    polygon([(87, 108), (99, 106), (107, 110), (101, 117), (94, 118),
             (88, 114)], 0.29)
    line([(91, 109), (98, 108)], 0.60, 1)
    line([(97, 117), (96, 122), (90, 124), (85, 122)], 0.42, 1)
    line([(96, 122), (102, 125), (109, 122)], 0.42, 1)
    for points in (
        [(58, 70), (63, 64), (66, 63)], [(71, 61), (77, 57), (80, 59)],
        [(87, 55), (91, 51), (96, 55)], [(104, 59), (108, 57), (113, 63)],
        [(121, 64), (127, 70)], [(57, 83), (63, 80), (67, 81)],
        [(80, 82), (84, 86)], [(106, 82), (109, 80)],
        [(124, 82), (129, 85)], [(51, 97), (57, 100), (60, 104)],
        [(63, 108), (68, 111), (72, 111)], [(68, 116), (75, 119)],
        [(128, 107), (123, 112), (119, 113)], [(120, 119), (114, 122)],
        [(80, 102), (83, 104)], [(110, 101), (106, 105)],
    ):
        line(points, 0.60, 1)
    return shapes


def _paint_pomeranian() -> np.ndarray:
    """Rasterize primitive masks at native 256px, without supersampling."""
    signal = np.ones((256, 256), dtype=np.float64)
    for primitive in _dog_primitives():
        mask = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        kind = primitive["kind"]
        if kind == "polygon":
            draw.polygon(primitive["points"], fill=255)
        elif kind == "ellipse":
            draw.ellipse(primitive["box"], fill=255)
        else:
            draw.line(primitive["points"], fill=255, width=primitive["width"])
        signal[np.asarray(mask) != 0] = primitive["signal"]
        if primitive.get("outline") is not None:
            outline_mask = Image.new("L", (256, 256), 0)
            outline_draw = ImageDraw.Draw(outline_mask)
            if kind == "polygon":
                points = primitive["points"]
                outline_draw.line(points + [points[0]], fill=255, width=primitive["width"])
            else:
                outline_draw.ellipse(primitive["box"], outline=255, width=primitive["width"])
            signal[np.asarray(outline_mask) != 0] = primitive["outline"]
    return signal


def render_pomeranian_svg() -> str:
    """Return an editable SVG from the same primitives, using window 0..1.2.

    Browser vector edge rendering can differ from native pixel masks. The SVG
    is a schematic reference; the arrays, not SVG pixels, define calculations.
    """
    def color(signal):
        percent = signal / 1.2 * 100
        return f"rgb({percent:.6f}%,{percent:.6f}%,{percent:.6f}%)"

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256" role="img" aria-labelledby="dog-title dog-desc">',
        '  <title id="dog-title">ポメラニアンの教育用合成テストパターン</title>',
        '  <desc id="dog-desc">立った小型犬の模式図。立ち耳、丸い頭、短い口吻、豊かな胸毛、背中へ巻いた尾を中間調と細い毛の線で表す。解剖構造や実撮影画像ではない。数値モデルと共通のベクター図形を用い、表示窓0から1.2に合わせた灰色で描く。</desc>',
        f'  <rect width="256" height="256" fill="{color(1.0)}"/>',
    ]
    for primitive in _dog_primitives():
        kind = primitive["kind"]
        if kind == "line":
            points = " ".join(f"{x:.4f},{y:.4f}" for x, y in primitive["points"])
            parts.append(f'  <polyline points="{points}" fill="none" stroke="{color(primitive["signal"])}" stroke-width="{primitive["width"]}"/>')
            continue
        outline = primitive.get("outline")
        style = f'fill="{color(primitive["signal"])}"'
        if outline is not None:
            style += f' stroke="{color(outline)}" stroke-width="{primitive["width"]}"'
        if kind == "polygon":
            points = " ".join(f"{x:.4f},{y:.4f}" for x, y in primitive["points"])
            parts.append(f'  <polygon points="{points}" {style}/>')
        else:
            x0, y0, x1, y1 = primitive["box"]
            parts.append(f'  <ellipse cx="{(x0+x1)/2:g}" cy="{(y0+y1)/2:g}" rx="{(x1-x0)/2:g}" ry="{(y1-y0)/2:g}" {style}/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def make_extended_phantom() -> np.ndarray:
    """Return a new float64 512x512 signal array, with no physical pixel size."""
    phantom = np.ones((512, 512), dtype=np.float64)
    phantom[:256, :256] = make_phantom()
    phantom[:256, 256:] = _paint_pomeranian()
    for group in LINE_PAIR_GROUPS:
        y0, y1, x0, x1 = group["bbox"]
        width = group["line_width_px"]
        dark = (np.arange(x1 - x0) % (2 * width)) < width
        row = np.where(dark, group["dark_signal"], group["bright_signal"])
        phantom[y0:y1, x0:x1] = row
    return phantom


def sample_extended_image(mean_quanta: float, seed: int = 42) -> np.ndarray:
    """Sample Poisson(N*S)/N, with the unchanged legacy validation domain.

    This is a new 512x512 random field. Identical N/seed reproduce this field;
    its top-left noise realization need not equal the separate legacy field.
    No blur, interpolation, or display enhancement is applied here.
    """
    if isinstance(mean_quanta, bool) or not isinstance(mean_quanta, numbers.Real):
        raise ValueError("mean_quanta must be a finite number in [1e-9, 1e12]")
    try:
        count = float(mean_quanta)
    except (ValueError, OverflowError):
        raise ValueError("mean_quanta must be a finite number in [1e-9, 1e12]") from None
    if not math.isfinite(count) or not 1e-9 <= count <= 1e12:
        raise ValueError("mean_quanta must be a finite number in [1e-9, 1e12]")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    phantom = make_extended_phantom()
    rng = np.random.default_rng(int(seed))
    return rng.poisson(count * phantom).astype(np.float64) / count
