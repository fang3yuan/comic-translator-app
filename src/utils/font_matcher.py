"""Font discovery, role classification and role -> font mapping."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.config import FONTS_DIR, Settings, resolve_path
from src.core.models import FontRole, TextRegion

FONT_EXTENSIONS = {".ttf", ".otf"}
_SFX_HINTS = ("bold", "black", "heavy", "display", "marker", "brush", "impact", "comic")
_NARRATION_HINTS = ("naskh", "kufi", "regular", "text", "serif")
_WHISPER_HINTS = ("light", "thin", "hand", "script", "italic", "ruqaa", "nastaliq")


def classify_roles(regions: Sequence[TextRegion]) -> None:
    """Assign a ``FontRole`` to each region from text and geometry heuristics.

    Rules (in order): SFX = a few words drawn on artwork, or shouted ALL-CAPS text;
    narration = one wide line on a flat box (a caption); whisper/thought = parenthesised,
    ellipsis-led, or noticeably small text; else dialogue.

    Args:
        regions: Regions to classify (modified in place).
    """
    sizes = [r.source_style.font_size for r in regions if r.source_style.font_size > 0]
    median = statistics.median(sizes) if sizes else 20.0
    for r in regions:
        text = r.text.strip()
        letters = [c for c in text if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
        words = len(text.split())
        size = r.source_style.font_size
        bx1, by1, bx2, by2 = r.bbox
        caption = r.flat_bg and (bx2 - bx1) > 8 * max(1, by2 - by1) and words >= 3

        if not r.flat_bg and words <= 3:
            r.role = FontRole.SFX
        elif upper_ratio > 0.85 and (text.count("!") >= 2 or size > 1.6 * median):
            r.role = FontRole.SFX
        elif caption:
            r.role = FontRole.NARRATION
        elif (text.startswith("(") and text.endswith(")")) or text.startswith("...") or size < 0.7 * median:
            r.role = FontRole.WHISPER
        else:
            r.role = FontRole.DIALOGUE


class FontMatcher:
    """Maps comic roles to Arabic font files located in ``fonts/``."""

    def __init__(self, settings: Settings, fonts_dir: Optional[Path] = None) -> None:
        """Create a matcher.

        Args:
            settings: Application settings (``fonts.*``).
            fonts_dir: Directory scanned for fonts. Defaults to ``fonts/``.
        """
        self.settings = settings
        self.fonts_dir = fonts_dir or FONTS_DIR

    def available_fonts(self) -> List[Path]:
        """Return all ``.ttf``/``.otf`` files in the fonts directory (sorted)."""
        if not self.fonts_dir.exists():
            return []
        return sorted(
            p for p in self.fonts_dir.iterdir() if p.suffix.lower() in FONT_EXTENSIONS
        )

    def default_font(self) -> Path:
        """Return the configured default font, or the first available one.

        Raises:
            FileNotFoundError: If no font can be found at all.
        """
        configured = resolve_path(self.settings.get("fonts.default_font", "fonts/default_arabic.ttf"))
        if configured.exists():
            return configured
        fonts = self.available_fonts()
        if fonts:
            return fonts[0]
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ):
            if Path(candidate).exists():
                return Path(candidate)
        raise FileNotFoundError("No Arabic-capable font found. Put a .ttf in the fonts/ folder.")

    def _auto_pick(self, role: FontRole) -> Optional[Path]:
        """Pick a font whose file name hints at the role, if any."""
        hints = {
            FontRole.SFX: _SFX_HINTS,
            FontRole.NARRATION: _NARRATION_HINTS,
            FontRole.WHISPER: _WHISPER_HINTS,
            FontRole.DIALOGUE: (),
        }[role]
        for font in self.available_fonts():
            if font.name == "default_arabic.ttf":
                continue
            if any(h in font.stem.lower() for h in hints):
                return font
        return None

    def font_for(self, role: FontRole | str) -> Path:
        """Resolve the font for a role: explicit mapping, name hint, default.

        Args:
            role: Comic role.

        Returns:
            Path to a font file.
        """
        role = FontRole(role)
        mapped = (self.settings.get("fonts.mappings", {}) or {}).get(role.value, "")
        if mapped:
            path = resolve_path(mapped)
            if not path.exists():
                path = self.fonts_dir / Path(mapped).name
            if path.exists():
                return path
        return self._auto_pick(role) or self.default_font()

    def mapping(self) -> Dict[str, str]:
        """Return the resolved role -> font path mapping for display."""
        return {r.value: str(self.font_for(r)) for r in FontRole}
