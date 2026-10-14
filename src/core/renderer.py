"""PIL-based Arabic typesetting with RTL shaping and binary-search font fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from src.config import Settings
from src.core.models import RenderStyle, TextRegion
from src.utils.arabic_utils import shape_arabic, split_words
from src.utils.color_utils import colors_too_similar, contrast_color
from src.utils.font_matcher import FontMatcher
from src.utils.image_utils import hex_to_rgb

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load (and cache) a font.

    ``Layout.BASIC`` is used on purpose: the text is already shaped by
    ``arabic_reshaper`` + ``bidi`` and must not be shaped a second time by Raqm.

    Args:
        path: Font file path.
        size: Pixel size.

    Returns:
        FreeType font object.
    """
    return ImageFont.truetype(path, max(1, size), layout_engine=ImageFont.Layout.BASIC)


@dataclass
class FitResult:
    """Outcome of the font-size search.

    Attributes:
        font_size: Chosen pixel size.
        lines: Logical-order lines (before shaping).
        line_height: Pixel distance between baselines.
        width: Widest shaped line in pixels.
        height: Total text height in pixels.
        fits: ``False`` if even the minimum size overflows the box.
    """

    font_size: int
    lines: List[str] = field(default_factory=list)
    line_height: int = 0
    width: int = 0
    height: int = 0
    fits: bool = True


def text_width(font: ImageFont.FreeTypeFont, logical_line: str) -> float:
    """Pixel width of a logical Arabic line after shaping."""
    return font.getlength(shape_arabic(logical_line))


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    """Wrap logical text into balanced lines that do not exceed ``max_width``.

    A greedy pass finds the minimum line count; a binary search on the wrap
    width then shrinks the lines to roughly equal length (nicer in bubbles).
    Words are never split, so a single over-wide word yields an over-wide line
    (the caller reduces the font size in that case).

    Args:
        text: Logical text; explicit ``\\n`` are respected.
        font: Font used for measuring.
        max_width: Maximum line width in pixels.

    Returns:
        List of logical lines.
    """

    def greedy(width: float) -> List[str]:
        lines: List[str] = []
        for paragraph in text.split("\n"):
            words = split_words(paragraph)
            if not words:
                lines.append("")
                continue
            cur = words[0]
            for word in words[1:]:
                trial = f"{cur} {word}"
                if text_width(font, trial) <= width:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    base = greedy(max_width)
    if len(base) <= 1:
        return base
    widest_word = max((text_width(font, w) for w in split_words(text)), default=0.0)
    lo, hi = max(widest_word, 1.0), max_width
    best = base
    for _ in range(14):
        mid = (lo + hi) / 2
        cand = greedy(mid)
        if len(cand) <= len(base):
            best, hi = cand, mid
        else:
            lo = mid
        if hi - lo < 1.0:
            break
    return best


def fit_text(
    text: str,
    font_path: str,
    box_w: int,
    box_h: int,
    min_size: int = 6,
    max_size: Optional[int] = None,
    line_spacing: float = 1.0,
) -> FitResult:
    """Find the largest font size whose wrapped text fits inside a box.

    A binary search over integer sizes is used: for every candidate the text is
    wrapped to ``box_w`` and the resulting block must satisfy both the width
    and height limits. If nothing fits, the minimum size is returned with
    ``fits=False`` so callers can warn instead of silently clipping.

    Args:
        text: Logical Arabic text.
        font_path: Font file.
        box_w: Available width in pixels.
        box_h: Available height in pixels.
        min_size: Smallest allowed size.
        max_size: Largest allowed size (defaults to ``box_h``).
        line_spacing: Multiplier on font metrics for line height.
    Returns:
        ``FitResult`` describing the layout.
    """
    box_w, box_h = max(1, int(box_w)), max(1, int(box_h))
    upper = max(min_size, min(int(max_size or box_h), box_h, 400))

    def layout(size: int) -> FitResult:
        font = load_font(font_path, size)
        ascent, descent = font.getmetrics()
        lh = max(1, int((ascent + descent) * line_spacing))
        lines = wrap_text(text, font, box_w)
        width = int(max((text_width(font, l) for l in lines), default=0))
        height = lh * len(lines)
        return FitResult(size, lines, lh, width, height, width <= box_w and height <= box_h)

    lo, hi = min_size, upper
    best: Optional[FitResult] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        res = layout(mid)
        if res.fits:
            best, lo = res, mid + 1
        else:
            hi = mid - 1
    return best or layout(min_size)


class Typesetter:
    """Draws translated Arabic text onto a cleaned page."""

    def __init__(
        self,
        settings: Settings,
        font_matcher: Optional[FontMatcher] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Create the typesetter.

        Args:
            settings: Application settings (styles, fonts).
            font_matcher: Font resolver; created from ``settings`` if omitted.
            log: Optional logging callback.
        """
        self.settings = settings
        self.fonts = font_matcher or FontMatcher(settings)
        self.log = log or (lambda msg: logger.info(msg))

    def style_for(self, region: TextRegion) -> RenderStyle:
        """Resolve the render style of a region (role style + overrides).

        Args:
            region: Region to style.

        Returns:
            Fully resolved ``RenderStyle``.
        """
        raw: Dict = dict(self.settings.role_style(region.role.value))
        raw.update(region.style_override or {})
        font_path = raw.get("font_path") or str(self.fonts.font_for(region.role))
        src = region.source_style
        if raw.get("match_source_color"):
            color = tuple(src.color)
            if colors_too_similar(color, src.bg_color):  # estimate failed: stay readable
                color = contrast_color(src.bg_color)
        else:
            color = hex_to_rgb(raw["color"])
        stroke_color = contrast_color(color) if raw.get("stroke_auto", True) else hex_to_rgb(raw["stroke_color"])
        return RenderStyle(
            font_path=font_path,
            color=color,  # type: ignore[arg-type]
            stroke_enabled=bool(raw["stroke_enabled"]),
            stroke_color=stroke_color,
            stroke_width=int(raw["stroke_width"]),
            shadow_enabled=bool(raw["shadow_enabled"]),
            shadow_color=hex_to_rgb(raw["shadow_color"]),
            shadow_offset=int(raw["shadow_offset"]),
            opacity=float(raw["opacity"]),
            max_font_ratio=float(raw["max_font_ratio"]),
        )

    def render_region(self, base: Image.Image, region: TextRegion) -> None:
        """Draw one region onto ``base`` (RGBA) in place.

        Args:
            base: RGBA page image.
            region: Region with a non-empty ``translation``.
        """
        text = region.translation.strip()
        if not text:
            return
        style = self.style_for(region)
        x1, y1, x2, y2 = region.fit_bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            return
        margin = (style.stroke_width if style.stroke_enabled else 0) + (
            style.shadow_offset if style.shadow_enabled else 0
        )
        avail_w, avail_h = max(1, bw - 2 * margin), max(1, bh - 2 * margin)
        cap = int(max(region.source_style.font_size, 10) * style.max_font_ratio)
        fit = fit_text(text, style.font_path, avail_w, avail_h, max_size=cap)
        if not fit.fits:
            self.log(f"Block #{region.id}: text does not fit even at {fit.font_size}px; shorten the translation.")

        font = load_font(style.font_path, fit.font_size)
        pad = margin + 2
        layer = Image.new("RGBA", (bw + 2 * pad, bh + 2 * pad), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        alpha = int(255 * max(0.0, min(1.0, style.opacity)))
        top = pad + (bh - fit.height) / 2
        for i, line in enumerate(fit.lines):
            shaped = shape_arabic(line)
            lw = font.getlength(shaped)
            x = pad + (bw - lw) / 2
            y = top + i * fit.line_height
            if style.shadow_enabled:
                off = style.shadow_offset
                draw.text((x + off, y + off), shaped, font=font, fill=(*style.shadow_color, alpha),
                          stroke_width=style.stroke_width if style.stroke_enabled else 0,
                          stroke_fill=(*style.shadow_color, alpha))
            draw.text(
                (x, y), shaped, font=font, fill=(*style.color, alpha),
                stroke_width=style.stroke_width if style.stroke_enabled else 0,
                stroke_fill=(*style.stroke_color, alpha),
            )
        angle = region.source_style.angle
        if abs(angle) >= 1.0:
            layer = layer.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        px, py = cx - layer.width // 2, cy - layer.height // 2
        self._paste_clipped(base, layer, px, py)

    @staticmethod
    def _paste_clipped(base: Image.Image, layer: Image.Image, px: int, py: int) -> None:
        """Alpha-composite ``layer`` at ``(px, py)``, clipping to the page bounds."""
        sx, sy = max(0, -px), max(0, -py)
        dx, dy = max(0, px), max(0, py)
        w = min(layer.width - sx, base.width - dx)
        h = min(layer.height - sy, base.height - dy)
        if w <= 0 or h <= 0:
            return
        base.alpha_composite(layer, dest=(dx, dy), source=(sx, sy, sx + w, sy + h))

    def render_page(self, cleaned: Image.Image, regions: Sequence[TextRegion]) -> Image.Image:
        """Typeset all enabled regions onto a cleaned page.

        Args:
            cleaned: Page with the original text removed.
            regions: Regions with translations.

        Returns:
            New RGB image; ``cleaned`` is not modified.
        """
        canvas = cleaned.convert("RGBA")
        for region in regions:
            if region.enabled:
                self.render_region(canvas, region)
        return canvas.convert("RGB")
