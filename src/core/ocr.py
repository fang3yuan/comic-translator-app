"""OCR wrapper (RapidOCR), text cleaning and source-style estimation."""

from __future__ import annotations

import importlib.util
import logging
import re
from typing import Any, Callable, List, Optional, Sequence

import cv2
import numpy as np

from src.core.models import OCRLine, SourceStyle, TextRegion
from src.utils.color_utils import analyze_text_pixels, contrast_color
from src.utils.image_utils import bbox_iou
from src.utils.sysutils import worker_threads

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
def clean_ocr_text(text: str) -> str:
    """Normalise raw OCR output for translation.

    Joins hyphen-broken words, collapses whitespace, repairs a few common OCR
    confusions and converts SHOUTED ALL-CAPS text to sentence case, which
    machine translators handle far better.

    Args:
        text: Raw OCR text (lines joined by spaces or newlines).

    Returns:
        Cleaned text.
    """
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "I").replace("¡", "!")
    text = re.sub(r"\bl\b(?=\s+[a-z])", "I", text)
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        lowered = text.lower()
        lowered = re.sub(r"\bi\b", "I", lowered)
        lowered = re.sub(r"(^|[.!?…]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), lowered)
        text = lowered
    return text


# --------------------------------------------------------------------------
# Style estimation
# --------------------------------------------------------------------------
def estimate_style(image_bgr: np.ndarray, region: TextRegion) -> SourceStyle:
    """Estimate colour, size, angle and weight of a region's source text.

    Args:
        image_bgr: Full page (BGR).
        region: Region with ``line_polygons`` / ``polygon`` / ``bbox`` set.

    Returns:
        Estimated ``SourceStyle``.
    """
    h, w = image_bgr.shape[:2]
    polys = region.line_polygons or [region.polygon]
    x1, y1, x2, y2 = region.bbox
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    style = SourceStyle()
    if x2 - x1 < 2 or y2 - y1 < 2:
        return style

    # Font size: median line height (shorter side of each minimum-area rectangle).
    heights, angles = [], []
    for poly in polys:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
        heights.append(min(rw, rh))
        if len(pts) == 4:
            dx, dy = pts[1] - pts[0]
            angles.append(float(np.degrees(np.arctan2(dy, dx))))
    if heights:
        style.font_size = float(np.median(heights))
    if angles:
        angle = float(np.median(angles))
        style.angle = angle if abs(angle) >= 3 and abs(angle) <= 45 else 0.0
    bw, bh = x2 - x1, y2 - y1
    style.vertical = bh > 2.5 * bw and len(polys) <= 2

    # Colours: colour distance from the surrounding background (any text colour works).
    tp = analyze_text_pixels(image_bgr, polys)
    if tp is not None:
        style.color = tp.ink_rgb
        style.bg_color = tp.bg_rgb
        style.stroke_color = contrast_color(tp.ink_rgb)
        style.bold = bool(tp.coverage > 0.28)
    return style


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class OCRUnavailableError(RuntimeError):
    """Raised when the OCR engine is not installed."""


class OCREngine:
    """Lazy wrapper over RapidOCR (light, CPU-friendly, models bundled in the package)."""

    def __init__(
        self,
        min_confidence: float = 0.25,
        tile_height: int = 1500,
        log: Optional[Callable[[str], None]] = None,
        max_side: int = 1280,
    ) -> None:
        """Create the engine (the model loads on first use).

        Args:
            min_confidence: Drop lines below this confidence.
            tile_height: Pages taller than this are processed in vertical tiles.
            log: Optional logging callback.
            max_side: The text finder works on a copy whose long side is at most this many
                pixels (big speed-up on large scans; text is still found at this size).
        """
        self.max_side = max(640, int(max_side))
        self.min_confidence = min_confidence
        self.tile_height = max(600, tile_height)
        self.log = log or (lambda msg: logger.info(msg))
        self._reader: Any = None

    @staticmethod
    def is_installed() -> bool:
        """Whether RapidOCR can be imported."""
        return importlib.util.find_spec("rapidocr") is not None

    def _load(self) -> None:
        """Instantiate RapidOCR."""
        if self._reader is not None:
            return
        importlib.invalidate_caches()
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OCRUnavailableError(
                "The text reader (OCR) is not installed. Open Setup and press 'Install everything'. "
                "/ قارئ النصوص غير مثبّت، افتح Setup واضغط Install."
            ) from exc
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        self.log("Loading text reader...")
        self._reader = RapidOCR(params={
            "Det.limit_side_len": self.max_side, "Det.limit_type": "max", "Global.use_cls": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": worker_threads(),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })

    def _read_once(self, image_bgr: np.ndarray) -> List[OCRLine]:
        """Run OCR on one image (no tiling)."""
        self._load()
        out = self._reader(image_bgr)
        if out is None or getattr(out, "boxes", None) is None:
            return []
        return [
            OCRLine([(float(x), float(y)) for x, y in box], str(text), float(score))
            for box, text, score in zip(out.boxes, out.txts, out.scores)
        ]

    def read_lines(self, image_bgr: np.ndarray) -> List[OCRLine]:
        """Recognise all text lines on a page.

        Very tall pages (webtoon strips) are split into overlapping vertical tiles and the
        results de-duplicated.

        Args:
            image_bgr: Page image (BGR).

        Returns:
            Filtered ``OCRLine`` list in page coordinates.
        """
        h = image_bgr.shape[0]
        if h <= self.tile_height * 1.3:
            return self._filter(self._read_once(image_bgr))
        raw: List[OCRLine] = []
        y, overlap = 0, 120
        while y < h:
            y2 = min(h, y + self.tile_height)
            for ln in self._read_once(image_bgr[y:y2]):
                ln.polygon = [(px, py + y) for px, py in ln.polygon]
                raw.append(ln)
            if y2 >= h:
                break
            y = y2 - overlap
        return self._filter(self._dedupe(raw))

    @staticmethod
    def _dedupe(lines: Sequence[OCRLine]) -> List[OCRLine]:
        """Drop duplicates produced by overlapping tiles (keep higher confidence)."""
        kept: List[OCRLine] = []
        for ln in sorted(lines, key=lambda l: -l.confidence):
            if all(bbox_iou(ln.bbox, k.bbox) < 0.5 for k in kept):
                kept.append(ln)
        return kept

    def _filter(self, lines: Sequence[OCRLine]) -> List[OCRLine]:
        """Remove low-confidence, empty or symbol-only lines."""
        out = []
        for ln in lines:
            x1, y1, x2, y2 = ln.bbox
            if ln.confidence < self.min_confidence or not re.search(r"[A-Za-z0-9]", ln.text):
                continue
            if (y2 - y1) < 6 or (x2 - x1) < 6:
                continue
            out.append(ln)
        return out
