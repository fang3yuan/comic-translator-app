"""OCR wrapper (RapidOCR), text cleaning and source-style estimation."""

from __future__ import annotations

import importlib.util
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.core.models import OCRLine, SourceStyle, TextRegion
from src.utils.color_utils import analyze_text_pixels, contrast_color
from src.utils.image_utils import bbox_iou
from src.utils.sysutils import pool_workers, worker_threads

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
        workers: int = 0,
    ) -> None:
        """Create the engine (the model loads on first use).

        Args:
            min_confidence: Drop lines below this confidence.
            tile_height: Pages taller than this are processed in vertical tiles.
            log: Optional logging callback.
            max_side: The text finder works on a copy whose long side is at most this many
                pixels (big speed-up on large scans; text is still found at this size).
            workers: Threads used to read the found lines (``0`` = automatic, a few at most).
        """
        self.workers = pool_workers(workers)
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

    # -- step 1: where is the text? (no reading yet) -----------------------
    def _detect_once(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Find text-line polygons in one image without recognising them."""
        self._load()
        out = self._reader(image_bgr, use_det=True, use_cls=False, use_rec=False)
        boxes = getattr(out, "boxes", None)
        if boxes is None:
            return []
        return [np.asarray(b, np.float32).reshape(-1, 2) for b in boxes]

    def detect(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Locate every text line on a page (tall pages are searched in overlapping bands).

        A line touching the inside edge of a band may be cut in half there, so such boxes are
        only used when no other band found the same line whole (the bands overlap by more
        than a line's height).

        Args:
            image_bgr: Page image (BGR).

        Returns:
            Polygons (4x2 float arrays) in page coordinates.
        """
        h = image_bgr.shape[0]
        if h <= self.tile_height * 1.3:
            return self._detect_once(image_bgr)
        overlap = 240
        found: List[Tuple[np.ndarray, int, bool]] = []  # (polygon, band number, cut by band edge)
        y, band = 0, 0
        while y < h:
            y2 = min(h, y + self.tile_height)
            for poly in self._detect_once(image_bgr[y:y2]):
                cut = (y > 0 and poly[:, 1].min() < 4) or (y2 < h and poly[:, 1].max() > (y2 - y) - 4)
                found.append((poly + np.array([0, y], np.float32), band, bool(cut)))
            if y2 >= h:
                break
            y, band = y2 - overlap, band + 1

        def box(p: np.ndarray) -> Tuple[int, int, int, int]:
            return int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())

        def area(p: np.ndarray) -> float:
            b = box(p)
            return float((b[2] - b[0]) * (b[3] - b[1]))

        # Lines of one band are all real (tilted neighbours may overlap); only the same line seen
        # by two different bands is a duplicate. Whole lines win, cut ones only fill real gaps.
        kept: List[Tuple[np.ndarray, int]] = []
        for want_cut in (False, True):
            for poly, b, cut in sorted((f for f in found if f[2] == want_cut), key=lambda f: -area(f[0])):
                if all(kb == b or bbox_iou(box(poly), box(kp)) < 0.3 for kp, kb in kept):
                    kept.append((poly, b))
        return [kp for kp, _ in kept]

    # -- step 2: what does it say? (full-resolution crops, small thread pool) -
    @staticmethod
    def _crop_line(image_bgr: np.ndarray, poly: np.ndarray) -> Optional[np.ndarray]:
        """Straighten one text line into a crop (perspective warp, vertical text turned)."""
        pts = np.asarray(poly, np.float32)
        if len(pts) != 4:
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        if w < 2 or h < 2:
            return None
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        crop = cv2.warpPerspective(image_bgr, cv2.getPerspectiveTransform(pts, dst), (w, h),
                                   borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        return np.ascontiguousarray(np.rot90(crop)) if h / w >= 1.5 else crop

    def _recognise(self, image_bgr: np.ndarray, poly: np.ndarray) -> Optional[OCRLine]:
        """Read one line; ``None`` if nothing could be read."""
        crop = self._crop_line(image_bgr, poly)
        if crop is None:
            return None
        try:
            out = self._reader(crop, use_det=False, use_cls=False, use_rec=True)
            txts, scores = getattr(out, "txts", None), getattr(out, "scores", None)
            if not txts:
                return None
            return OCRLine([(float(x), float(y)) for x, y in poly], str(txts[0]), float(scores[0]))
        except Exception as exc:  # noqa: BLE001 - one bad crop must not lose the page
            logger.debug("line skipped: %s", exc)
            return None

    def read_lines(self, image_bgr: np.ndarray) -> List[OCRLine]:
        """Find and read all text lines on a page.

        First every line is located (cheap, whole page or overlapping bands), then each one is
        read from the full-resolution image, so no text is ever split between two crops. Lines
        are read by a small thread pool (``workers``) to keep the computer responsive.

        Args:
            image_bgr: Page image (BGR).

        Returns:
            Filtered ``OCRLine`` list in page coordinates.
        """
        polys = self.detect(image_bgr)
        if self.workers <= 1 or len(polys) < 3:
            lines = [self._recognise(image_bgr, p) for p in polys]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                lines = list(pool.map(lambda p: self._recognise(image_bgr, p), polys))
        return self._filter([ln for ln in lines if ln is not None])

    @staticmethod
    def _dedupe(lines: Sequence[OCRLine]) -> List[OCRLine]:
        """Drop duplicates produced by overlapping tiles (keep higher confidence)."""
        kept: List[OCRLine] = []
        for ln in sorted(lines, key=lambda l: -l.confidence):
            if all(bbox_iou(ln.bbox, k.bbox) < 0.5 for k in kept):
                kept.append(ln)
        return kept

    def _filter(self, lines: Sequence[OCRLine]) -> List[OCRLine]:
        """Remove low-confidence, empty or symbol-only lines (a lone character needs 0.75: round heads read as "O")."""
        out = []
        for ln in lines:
            x1, y1, x2, y2 = ln.bbox
            if ln.confidence < self.min_confidence or not re.search(r"[A-Za-z0-9]", ln.text):
                continue
            if len(ln.text.strip()) == 1 and ln.confidence < max(self.min_confidence, 0.75):
                continue
            if (y2 - y1) < 6 or (x2 - x1) < 6:
                continue
            out.append(ln)
        return out
