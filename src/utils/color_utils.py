"""Colour analysis of text pixels, independent of the text/background colours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from src.core.models import Point, RGB
from src.utils.image_utils import polygon_mask

_RING_INNER = 5    # px: kernel used to skip the polygon edge / anti-aliasing
_RING_OUTER = 21   # px: kernel giving a background ring ~8 px wide
_MARGIN = 12       # px of context around the polygons


@dataclass
class TextPixels:
    """Result of ``analyze_text_pixels``.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` of the analysed crop in page coordinates.
        ink: Boolean mask (crop coordinates) of the pixels that differ from the background.
        region: Boolean mask (crop coordinates) of the text polygons.
        ink_bgr: Median colour of the text core (BGR).
        bg_bgr: Median background colour (BGR).
        coverage: Fraction of the polygon area covered by ink.
    """

    bbox: Tuple[int, int, int, int]
    ink: np.ndarray
    region: np.ndarray
    ink_bgr: Tuple[int, int, int]
    bg_bgr: Tuple[int, int, int]
    coverage: float

    @property
    def ink_rgb(self) -> RGB:
        """Text colour as RGB."""
        b, g, r = self.ink_bgr
        return int(r), int(g), int(b)

    @property
    def bg_rgb(self) -> RGB:
        """Background colour as RGB."""
        b, g, r = self.bg_bgr
        return int(r), int(g), int(b)


def luminance(rgb: Sequence[int]) -> float:
    """Perceived brightness (0-255) of an RGB colour."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def contrast_color(rgb: Sequence[int]) -> RGB:
    """Black or white, whichever contrasts better with ``rgb``."""
    return (0, 0, 0) if luminance(rgb) > 140 else (255, 255, 255)


def colors_too_similar(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether two RGB colours would be hard to tell apart as text on background."""
    dist = float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))
    return dist < 70 or abs(luminance(a) - luminance(b)) < 40 and dist < 110


def analyze_text_pixels(image_bgr: np.ndarray, polygons: Sequence[Sequence[Point]]) -> Optional[TextPixels]:
    """Separate text ink from background using colour distance in Lab space.

    The background colour is taken from a thin ring just outside the text polygons
    (not from "the majority colour"), so bold text, light-on-dark text and text whose
    brightness equals the background's brightness are all handled.

    Args:
        image_bgr: Page image (BGR ``uint8``).
        polygons: Text polygons (e.g. one per OCR line) in page coordinates.

    Returns:
        ``TextPixels`` or ``None`` if no ink can be distinguished.
    """
    polys = [p for p in polygons if len(p) >= 3]
    if not polys:
        return None
    h, w = image_bgr.shape[:2]
    pts = np.array([pt for p in polys for pt in p], dtype=np.float32)
    x1 = max(0, int(pts[:, 0].min()) - _MARGIN)
    y1 = max(0, int(pts[:, 1].min()) - _MARGIN)
    x2 = min(w, int(np.ceil(pts[:, 0].max())) + _MARGIN)
    y2 = min(h, int(np.ceil(pts[:, 1].max())) + _MARGIN)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    shifted = [[(px - x1, py - y1) for px, py in p] for p in polys]
    region = polygon_mask(crop.shape[:2], shifted) > 0
    if region.sum() < 10:
        return None
    grown = cv2.dilate(region.astype(np.uint8), np.ones((_RING_INNER, _RING_INNER), np.uint8)) > 0
    outer = cv2.dilate(region.astype(np.uint8), np.ones((_RING_OUTER, _RING_OUTER), np.uint8)) > 0
    ring = outer & ~grown

    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    if ring.sum() >= 30:
        bg_lab = np.median(lab[ring], axis=0)
    else:  # text fills the crop: assume the most common colour inside is the background
        bg_lab = np.median(lab[region], axis=0)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    values = dist[region]
    p95 = float(np.percentile(values, 95))
    if p95 < 8.0:
        return None  # ink indistinguishable from the background
    ink = region & (dist > max(8.0, 0.35 * p95))
    if ink.sum() < 6:
        return None
    core = ink & (dist >= np.percentile(dist[ink], 40))  # skip blended anti-aliased edges
    ink_bgr = np.median(crop[core], axis=0)
    # Keep only pixels on the bg -> ink colour line (solid or anti-aliased text). Other
    # colours inside the polygon (a dark stripe, a border, art) are NOT text: leave them.
    ink_lab = np.median(lab[core], axis=0)
    axis = ink_lab - bg_lab
    span = float(np.linalg.norm(axis))
    if span > 1e-3:
        rel = lab - bg_lab
        t = (rel @ axis) / (span * span)
        perp = np.linalg.norm(rel - t[..., None] * axis, axis=2)
        on_line = (t > 0.25) & (perp < 0.30 * span + 6.0)
        refined = ink & on_line
        if refined.sum() >= 6:
            ink = refined
    if ring.sum() >= 30:
        bg_bgr = np.median(crop[ring], axis=0)
    else:
        bg_bgr = np.median(crop[region & ~ink], axis=0) if (region & ~ink).any() else ink_bgr
    return TextPixels(
        (x1, y1, x2, y2), ink, region,
        (int(ink_bgr[0]), int(ink_bgr[1]), int(ink_bgr[2])),
        (int(bg_bgr[0]), int(bg_bgr[1]), int(bg_bgr[2])),
        float(ink.sum() / region.sum()),
    )
