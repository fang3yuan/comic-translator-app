"""Shared dataclasses used across detection, OCR, rendering and the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2
Point = Tuple[float, float]
RGB = Tuple[int, int, int]


class FontRole(str, Enum):
    """Typographic role of a text block inside a comic page."""

    DIALOGUE = "dialogue"
    NARRATION = "narration"
    SFX = "sfx"
    WHISPER = "whisper"


@dataclass
class OCRLine:
    """A single recognised text line.

    Attributes:
        polygon: Four (or more) corner points in image coordinates.
        text: Recognised string.
        confidence: Engine confidence in ``[0, 1]``.
    """

    polygon: List[Point]
    text: str
    confidence: float = 1.0

    @property
    def bbox(self) -> BBox:
        """Axis-aligned bounding box of the polygon."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    @property
    def center(self) -> Point:
        """Centre of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class SourceStyle:
    """Style estimated from the original (English) text."""

    font_size: float = 20.0
    color: RGB = (0, 0, 0)
    stroke_color: RGB = (255, 255, 255)
    bg_color: RGB = (255, 255, 255)
    angle: float = 0.0
    bold: bool = False
    vertical: bool = False


@dataclass
class TextRegion:
    """A text block (usually one speech bubble / caption / SFX).

    Attributes:
        id: Reading-order index.
        bbox: Tight axis-aligned box around the source text.
        polygon: Convex outline of the source text.
        line_polygons: Per-line polygons, used to build precise erase masks.
        target_bbox: Box into which the Arabic text is fitted (editable).
        flat_bg: ``True`` if the text sits on a flat background (bubble / caption), ``False`` on artwork.
        text: Source English text.
        translation: Arabic translation.
        role: Typographic role.
        source_style: Estimated source style.
        confidence: Mean OCR/detector confidence.
        enabled: If ``False`` the block is neither erased nor rendered.
        ignored: Credit / watermark / website text: erased from the page but never translated.
        style_override: Optional per-block render-style overrides.
    """

    id: int
    bbox: BBox
    polygon: List[Point] = field(default_factory=list)
    line_polygons: List[List[Point]] = field(default_factory=list)
    target_bbox: Optional[BBox] = None
    flat_bg: bool = True
    text: str = ""
    translation: str = ""
    role: FontRole = FontRole.DIALOGUE
    source_style: SourceStyle = field(default_factory=SourceStyle)
    confidence: float = 1.0
    enabled: bool = True
    ignored: bool = False
    style_override: Dict[str, Any] = field(default_factory=dict)

    @property
    def fit_bbox(self) -> BBox:
        """Box used for typesetting (``target_bbox`` or the source ``bbox``)."""
        return self.target_bbox or self.bbox

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        data = asdict(self)
        data["role"] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextRegion":
        """Rebuild a region from ``to_dict`` output.

        Args:
            data: Dictionary produced by ``to_dict``.

        Returns:
            Reconstructed ``TextRegion``.
        """
        d = dict(data)
        d["role"] = FontRole(d.get("role", "dialogue"))
        style = d.get("source_style") or {}
        for key in ("color", "stroke_color", "bg_color"):
            if key in style:
                style[key] = tuple(style[key])
        d["source_style"] = SourceStyle(**style)
        d["bbox"] = tuple(d["bbox"])
        if d.get("target_bbox") is not None:
            d["target_bbox"] = tuple(d["target_bbox"])
        d["polygon"] = [tuple(p) for p in d.get("polygon", [])]
        d["line_polygons"] = [[tuple(p) for p in poly] for poly in d.get("line_polygons", [])]
        known = {f.name for f in fields(cls)}  # projects saved by older versions may hold extra keys
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RenderStyle:
    """Fully-resolved style used to draw one translated block."""

    font_path: str
    color: RGB = (0, 0, 0)
    stroke_enabled: bool = False
    stroke_color: RGB = (255, 255, 255)
    stroke_width: int = 2
    shadow_enabled: bool = False
    shadow_color: RGB = (0, 0, 0)
    shadow_offset: int = 2
    opacity: float = 1.0
    max_font_ratio: float = 1.4
