#!/usr/bin/env python3
"""Common contact-sheet renderer (two channel rows x five time columns).

Every sheet has the same layout, labels, colormap and fixed per-channel value
range. Fields are displayed with x increasing to the right and y increasing
upward. Each 64x64 field is enlarged by nearest-neighbour replication only.
No PDE, boundary-condition or configuration information is drawn.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps

PANEL_SCALE = 2          # 64 -> 128 pixels per panel
GAP = 4
LEFT = 28
TOP = 22
PAD = 4
COLORMAP = "viridis"
FONT_SIZE = 14
_LUT = (np.asarray(colormaps[COLORMAP](np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(np.uint8)


def _font() -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet_size(n: int = 64, columns: int = 5) -> tuple[int, int]:
    panel = n * PANEL_SCALE
    width = LEFT + columns * panel + (columns - 1) * GAP + PAD
    height = TOP + 2 * panel + GAP + PAD
    return width, height


def colorize(field: np.ndarray, vmin: float, vmax: float) -> tuple[np.ndarray, int]:
    """Map a [n, n] (x, y) field to an RGB panel; returns (rgb, out-of-range count)."""
    scaled = (np.asarray(field, dtype=np.float64) - vmin) / (vmax - vmin)
    out_of_range = int(np.count_nonzero((scaled < 0) | (scaled > 1)))
    index = np.clip(np.floor(scaled * 255 + 0.5), 0, 255).astype(np.int64)
    rgb = _LUT[index.T[::-1]]  # rows = y descending, columns = x ascending
    rgb = np.repeat(np.repeat(rgb, PANEL_SCALE, axis=0), PANEL_SCALE, axis=1)
    return rgb, out_of_range


def render_sheet(frames: np.ndarray, times: list[float], ranges: list[tuple[float, float]]) -> tuple[Image.Image, int]:
    """frames: [Q, 2, n, n]; times: Q labels; ranges: per-channel (vmin, vmax)."""
    Q, C, n, _ = frames.shape
    width, height = sheet_size(n, Q)
    panel = n * PANEL_SCALE
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    out_of_range = 0
    for c in range(C):
        for q in range(Q):
            rgb, bad = colorize(frames[q, c], *ranges[c])
            out_of_range += bad
            y0 = TOP + c * (panel + GAP)
            x0 = LEFT + q * (panel + GAP)
            canvas[y0:y0 + panel, x0:x0 + panel] = rgb
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font()
    for q, t in enumerate(times):
        label = f"t={t:.2f}"
        x0 = LEFT + q * (panel + GAP)
        w = draw.textlength(label, font=font)
        draw.text((x0 + (panel - w) / 2, 3), label, fill=(0, 0, 0), font=font)
    for c, name in enumerate(["u", "v"]):
        y0 = TOP + c * (panel + GAP)
        draw.text((9, y0 + panel / 2 - FONT_SIZE / 2 - 1), name, fill=(0, 0, 0), font=font)
    return image, out_of_range
