"""Arabic text shaping, RTL handling and translation post-processing."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import arabic_reshaper

try:  # python-bidi < 0.5 and >= 0.5 expose get_display differently
    from bidi.algorithm import get_display
except ImportError:  # pragma: no cover - depends on installed version
    from bidi import get_display  # type: ignore[no-redef]

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_TATWEEL = "ـ"
_DIACRITICS_RE = re.compile(r"[ً-ٰٟ]")
_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

_PUNCT_MAP = {"?": "؟", ",": "،", ";": "؛"}

_CONCISE_REPLACEMENTS = {
    "في الوقت الحالي": "حاليًا",
    "في هذه اللحظة": "الآن",
    "على الرغم من ذلك": "مع ذلك",
    "من أجل أن": "كي",
    "بسبب حقيقة أن": "لأن",
    "في حالة أن": "إن",
    "في الواقع،": "",
    "في الواقع": "",
    "بشكل كبير جدًا": "جدًا",
    "بشكل كبير": "كثيرًا",
}


def contains_arabic(text: str) -> bool:
    """Return ``True`` if ``text`` contains any Arabic-script character."""
    return bool(_ARABIC_RE.search(text))


def shape_arabic(text: str) -> str:
    """Prepare logical Arabic text for drawing with PIL.

    Each line is passed through ``arabic_reshaper.reshape()`` (letter joining)
    followed by ``bidi.algorithm.get_display()`` (visual RTL ordering).

    Args:
        text: Logical-order Arabic text; may contain ``\\n``.

    Returns:
        Visual-order string ready for ``ImageDraw.text``.
    """
    return "\n".join(_shape_line(line) for line in text.split("\n"))


@lru_cache(maxsize=8192)
def _shape_line(line: str) -> str:
    """Shape one line (cached: font fitting measures the same lines many times)."""
    return get_display(arabic_reshaper.reshape(line), base_dir="R")


def split_words(text: str) -> List[str]:
    """Split on whitespace while ignoring empty tokens."""
    return [w for w in text.split() if w]


def to_arabic_digits(text: str) -> str:
    """Convert ASCII digits to Arabic-Indic digits."""
    return text.translate(_ARABIC_DIGITS)


def localize_punctuation(text: str) -> str:
    """Replace Latin punctuation by Arabic equivalents and fix ellipses.

    Args:
        text: Arabic text.

    Returns:
        Text with ``? , ;`` replaced and ``...`` collapsed to ``…``.
    """
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"…{2,}", "…", text)
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"!{3,}", "!!", text)
    text = re.sub(r"؟{3,}", "؟؟", text)
    return text


def make_concise(text: str) -> str:
    """Apply rule-based shortening so Arabic fits the original bubble.

    Args:
        text: Arabic translation.

    Returns:
        Shortened text with normalised whitespace.
    """
    for long_form, short_form in _CONCISE_REPLACEMENTS.items():
        text = text.replace(long_form, short_form)
    return re.sub(r"[ \t]+", " ", text).strip()


def postprocess_translation(text: str, concise: bool = True, arabic_digits: bool = False) -> str:
    """Normalise a raw machine translation for comic typesetting.

    Args:
        text: Raw translation.
        concise: Apply ``make_concise``.
        arabic_digits: Convert digits to Arabic-Indic numerals.

    Returns:
        Cleaned Arabic text.
    """
    text = text.replace(_TATWEEL, "")
    text = _DIACRITICS_RE.sub("", text)
    text = localize_punctuation(text)
    if concise:
        text = make_concise(text)
    if arabic_digits:
        text = to_arabic_digits(text)
    return re.sub(r"\s+", " ", text).strip()
