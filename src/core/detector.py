"""Text-block grouping, free-room measurement and erase-mask generation (no extra model needed)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.core.models import BBox, OCRLine, TextRegion
from src.utils.color_utils import analyze_text_pixels
from src.utils.image_utils import (
    bbox_union,
    clamp_bbox,
    expand_bbox,
    polygon_mask,
)


class Detector:
    """Turns OCR lines into text blocks (bubbles / captions / SFX) and erase masks."""

    # -- grouping --------------------------------------------------------
    @staticmethod
    def _should_merge(a: OCRLine, b: OCRLine) -> bool:
        """Heuristic: do two lines belong to the same block?"""
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        ha, hb = max(1, ay2 - ay1), max(1, by2 - by1)
        if not 0.55 <= ha / hb <= 1.8:
            return False
        v_overlap = min(ay2, by2) - max(ay1, by1)
        h_gap = max(bx1 - ax2, ax1 - bx2)
        if v_overlap >= 0.6 * min(ha, hb) and 0 <= h_gap <= 1.5 * max(ha, hb):
            return True  # one text row split by the OCR (wide gap, colour change...)
        if max(by1 - ay2, ay1 - by2) > 0.9 * min(ha, hb):
            return False
        overlap = min(ax2, bx2) - max(ax1, bx1)
        narrow = max(1, min(ax2 - ax1, bx2 - bx1))
        centre_dx = abs((ax1 + ax2) / 2 - (bx1 + bx2) / 2)
        return overlap / narrow >= 0.3 or centre_dx < 0.6 * max(ax2 - ax1, bx2 - bx1)

    def group_lines(self, lines: Sequence[OCRLine]) -> List[List[OCRLine]]:
        """Cluster OCR lines into text blocks (union-find over ``_should_merge``).

        Args:
            lines: OCR lines.

        Returns:
            List of line groups.
        """
        parent = list(range(len(lines)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if self._should_merge(lines[i], lines[j]):
                    parent[find(i)] = find(j)
        groups: dict[int, List[OCRLine]] = {}
        for i, line in enumerate(lines):
            groups.setdefault(find(i), []).append(line)
        return list(groups.values())

    @staticmethod
    def order_lines(lines: Sequence[OCRLine]) -> List[OCRLine]:
        """Order lines top-to-bottom, and left-to-right inside one text row.

        Args:
            lines: Lines of one block.

        Returns:
            Lines in reading order (a plain ``(y, x)`` sort breaks when two words on one
            row differ by a few pixels in height).
        """
        rows: List[List[OCRLine]] = []
        for ln in sorted(lines, key=lambda l: l.center[1]):
            for row in rows:
                ref = row[0]
                limit = 0.5 * min(ln.bbox[3] - ln.bbox[1], ref.bbox[3] - ref.bbox[1])
                if abs(ln.center[1] - ref.center[1]) <= limit:
                    row.append(ln)
                    break
            else:
                rows.append([ln])
        return [ln for row in rows for ln in sorted(row, key=lambda l: l.bbox[0])]

    # -- bubble estimation -----------------------------------------------
    @staticmethod
    def find_room(image_bgr: np.ndarray, bbox: BBox) -> Tuple[BBox, bool]:
        """Grow the text box into the empty area around it (never across an outline).

        The box is extended side by side, 2 px at a time, as long as the strip it would
        add has the same colour as the background right around the text. It therefore
        stops at a bubble outline, at artwork or at a neighbouring text, so a translation
        fitted in the result can never leave the bubble. Textured surroundings (text drawn
        on artwork) are not grown beyond a small margin.

        Args:
            image_bgr: Page image (BGR).
            bbox: Box around the source text.

        Returns:
            ``(box, flat)``: the box the translation may use, and whether the text sits on
            a flat background (bubble / caption box) rather than on artwork.
        """
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad, step = 3, 2
        fallback = clamp_bbox(expand_bbox(bbox, 0.06, w, h), w, h)
        cap_x, cap_y = max(16, int(0.9 * bw)), max(16, int(1.1 * bh))
        wx1, wy1 = max(0, x1 - cap_x - pad - 4), max(0, y1 - cap_y - pad - 4)
        wx2, wy2 = min(w, x2 + cap_x + pad + 5), min(h, y2 + cap_y + pad + 5)
        lab = cv2.cvtColor(image_bgr[wy1:wy2, wx1:wx2].astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        ring = np.zeros(lab.shape[:2], bool)
        ox1, oy1, ox2, oy2 = x1 - wx1, y1 - wy1, x2 - wx1, y2 - wy1
        r_out = (max(0, oy1 - pad - 4), max(0, ox1 - pad - 4), min(ring.shape[0], oy2 + pad + 4), min(ring.shape[1], ox2 + pad + 4))
        ring[r_out[0]:r_out[2], r_out[1]:r_out[3]] = True
        ring[max(0, oy1 - pad):oy2 + pad, max(0, ox1 - pad):ox2 + pad] = False
        if ring.sum() < 40:
            return fallback, False
        bg = np.median(lab[ring], axis=0)
        dist = np.linalg.norm(lab - bg, axis=2)
        if float(np.percentile(dist[ring], 92)) > 10.0:  # textured / not a flat bubble
            return fallback, False
        integral = cv2.integral((dist < 14.0).astype(np.uint8)).astype(np.int64)

        def clear(ax1: int, ay1: int, ax2: int, ay2: int) -> bool:
            """Is the strip (window coords, half-open) background-coloured?"""
            if ax1 < 0 or ay1 < 0 or ax2 > ring.shape[1] or ay2 > ring.shape[0] or ax2 <= ax1 or ay2 <= ay1:
                return False
            tot = integral[ay2, ax2] - integral[ay1, ax2] - integral[ay2, ax1] + integral[ay1, ax1]
            return tot >= 0.985 * (ax2 - ax1) * (ay2 - ay1)

        L, T, R, B = ox1 - pad, oy1 - pad, ox2 + pad, oy2 + pad
        grown = {"L": 0, "T": 0, "R": 0, "B": 0}
        active = {"L", "T", "R", "B"}
        while active:
            for side in sorted(active):
                cap = cap_x if side in "LR" else cap_y
                ok = grown[side] + step <= cap
                if ok and side == "L":
                    ok = clear(L - step, T, L, B)
                elif ok and side == "R":
                    ok = clear(R, T, R + step, B)
                elif ok and side == "T":
                    ok = clear(L, T - step, R, T)
                elif ok and side == "B":
                    ok = clear(L, B, R, B + step)
                if not ok:
                    active.discard(side)
                    continue
                grown[side] += step
                L, T, R, B = (L - step if side == "L" else L, T - step if side == "T" else T,
                              R + step if side == "R" else R, B + step if side == "B" else B)
        # Grow symmetrically (the translation stays centred on the original text) and keep a
        # small margin so it never touches the outline.
        gx, gy = min(grown["L"], grown["R"]), min(grown["T"], grown["B"])
        m = 4
        gx, gy = (max(0, gx - m) if gx else 0), (max(0, gy - m) if gy else 0)
        box = (x1 - pad - gx, y1 - pad - gy, x2 + pad + gx, y2 + pad + gy)
        return clamp_bbox(bbox_union([bbox, box]), w, h), True

    # -- region construction -----------------------------------------------
    def build_regions(self, image_bgr: np.ndarray, lines: Sequence[OCRLine], reading_order: str = "rtl") -> List[TextRegion]:
        """Turn OCR lines into ordered ``TextRegion`` objects with fit boxes.

        Args:
            image_bgr: Page image (BGR).
            lines: OCR lines from ``OCREngine.read_lines``.
            reading_order: ``"rtl"`` (manga) or ``"ltr"`` (western comics).

        Returns:
            Ordered regions (``text`` is raw; clean it in the pipeline).
        """
        h, w = image_bgr.shape[:2]
        regions: List[TextRegion] = []
        for group in self.group_lines(lines):
            group = self.order_lines(group)
            bbox = clamp_bbox(bbox_union([l.bbox for l in group]), w, h)
            pts = np.array([p for l in group for p in l.polygon], dtype=np.float32)
            hull = cv2.convexHull(pts).reshape(-1, 2)
            region = TextRegion(
                id=0,
                bbox=bbox,
                polygon=[(float(x), float(y)) for x, y in hull],
                line_polygons=[list(l.polygon) for l in group],
                text=" ".join(l.text for l in group),
                confidence=float(np.mean([l.confidence for l in group])),
            )
            region.target_bbox, region.flat_bg = self.find_room(image_bgr, bbox)
            regions.append(region)
        return self.sort_reading_order(regions, reading_order)

    @staticmethod
    def sort_reading_order(regions: List[TextRegion], order: str = "rtl") -> List[TextRegion]:
        """Sort regions into reading order and renumber their ids.

        Args:
            regions: Regions to sort.
            order: ``"rtl"`` or ``"ltr"``.

        Returns:
            Sorted regions with ``id`` = position.
        """
        rows: List[List[TextRegion]] = []
        bounds: List[Tuple[int, int]] = []
        for r in sorted(regions, key=lambda r: r.bbox[1]):
            cy = (r.bbox[1] + r.bbox[3]) / 2
            for i, (top, bottom) in enumerate(bounds):
                if top <= cy <= bottom:
                    rows[i].append(r)
                    bounds[i] = (min(top, r.bbox[1]), max(bottom, r.bbox[3]))
                    break
            else:
                rows.append([r])
                bounds.append((r.bbox[1], r.bbox[3]))
        ordered: List[TextRegion] = []
        for row in sorted(rows, key=lambda row: min(r.bbox[1] for r in row)):
            row.sort(key=lambda r: (r.bbox[0] + r.bbox[2]) / 2, reverse=(order == "rtl"))
            ordered.extend(row)
        for i, r in enumerate(ordered):
            r.id = i
        return ordered

    # -- mask ------------------------------------------------------------
    @staticmethod
    def build_mask(image_bgr: np.ndarray, regions: Sequence[TextRegion], dilate: int = 3) -> np.ndarray:
        """Build a precise binary mask (0/255) covering only the text ink.

        The background colour is sampled from a ring around each line; pixels whose colour
        differs from it are ink, whatever the text colour (black, white, red...). The mask is
        dilated to swallow anti-aliasing and outlines (more for large lettering). If the ink
        cannot be separated the whole grown polygon is used.

        Args:
            image_bgr: Page image (BGR).
            regions: Regions whose source text must be erased.
            dilate: Minimum dilation radius in pixels.

        Returns:
            ``HxW`` ``uint8`` mask.
        """
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            if not region.enabled:
                continue
            for poly in region.line_polygons or [region.polygon]:
                if len(poly) < 3:
                    continue
                ys = [pt[1] for pt in poly]
                grow = max(1, dilate, int(round(0.06 * (max(ys) - min(ys)))))
                kernel = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
                tp = analyze_text_pixels(image_bgr, [poly])
                if tp is not None and 0.02 <= tp.coverage <= 0.85:
                    x1, y1, x2, y2 = tp.bbox
                    local = cv2.dilate(tp.ink.astype(np.uint8) * 255, kernel)
                    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], local)
                else:
                    mask = np.maximum(mask, cv2.dilate(polygon_mask((h, w), [poly]), kernel))
        return mask
