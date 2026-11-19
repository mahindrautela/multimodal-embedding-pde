"""Draw the image given to the models: one "sheet" of five snapshots.

Layout: two rows (fields u and v) by five columns (observed times). Each
64 x 64 field is enlarged 2x by repeating pixels and colored with viridis on a
fixed per-field range, so every sheet uses the same scale. x increases to the
right and y upward. The only text is the time labels and the field names; the
PDE, BC and problem name are never drawn. The arrays themselves are not changed.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

PANEL_SCALE = 2          # 64 x 64 field -> 128 x 128 pixel panel
GAP = 4                  # pixels between panels
LEFT = 28                # margin for the field names
TOP = 22                 # margin for the time labels
PAD = 4                  # right and bottom margin
COLORMAP = "viridis"
FONT_SIZE = 14
# 256-entry color table, as 8-bit RGB.
_LUT = (np.asarray(colormaps[COLORMAP](np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(np.uint8)


def _font() -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet_size(n: int = 64, columns: int = 5) -> tuple[int, int]:
    """(width, height) of a sheet in pixels."""
    panel = n * PANEL_SCALE
    width = LEFT + columns * panel + (columns - 1) * GAP + PAD
    height = TOP + 2 * panel + GAP + PAD
    return width, height


def colorize(field: np.ndarray, vmin: float, vmax: float) -> tuple[np.ndarray, int]:
    """Color one (x, y) field. Returns the RGB panel and the number of values outside [vmin, vmax].

    Out-of-range values are shown with the end colors; they are only counted.
    """
    scaled = (np.asarray(field, dtype=np.float64) - vmin) / (vmax - vmin)
    out_of_range = int(np.count_nonzero((scaled < 0) | (scaled > 1)))
    index = np.clip(np.floor(scaled * 255 + 0.5), 0, 255).astype(np.int64)
    rgb = _LUT[index.T[::-1]]                       # image rows = y from top to bottom
    rgb = np.repeat(np.repeat(rgb, PANEL_SCALE, axis=0), PANEL_SCALE, axis=1)
    return rgb, out_of_range


def render_sheet(frames: np.ndarray, times: list[float],
                 ranges: list[tuple[float, float]]) -> tuple[Image.Image, int]:
    """frames: [Q, 2, n, n] snapshots; times: Q labels; ranges: (vmin, vmax) per field.

    Returns the image and the total count of out-of-range values.
    """
    n_times, n_fields, n, _ = frames.shape
    width, height = sheet_size(n, n_times)
    panel = n * PANEL_SCALE
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    out_of_range = 0
    for c in range(n_fields):
        for q in range(n_times):
            rgb, bad = colorize(frames[q, c], *ranges[c])
            out_of_range += bad
            y0 = TOP + c * (panel + GAP)
            x0 = LEFT + q * (panel + GAP)
            canvas[y0:y0 + panel, x0:x0 + panel] = rgb
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font()
    for q, t in enumerate(times):                   # time labels, centered above each column
        label = f"t={t:.2f}"
        x0 = LEFT + q * (panel + GAP)
        w = draw.textlength(label, font=font)
        draw.text((x0 + (panel - w) / 2, 3), label, fill=(0, 0, 0), font=font)
    for c, name in enumerate(["u", "v"]):           # field names, left of each row
        y0 = TOP + c * (panel + GAP)
        draw.text((9, y0 + panel / 2 - FONT_SIZE / 2 - 1), name, fill=(0, 0, 0), font=font)
    return image, out_of_range
