"""Detect text that must be erased but not translated (credits, watermarks, website ads)."""

from __future__ import annotations

import re

_TLDS = r"(?:com|net|org|io|me|co|cc|tv|to|xyz|site|info|app|top|club|online|live|fun|vip|ws|ru|fr|es|id|jp)"
_URL = re.compile(
    rf"(?:https?\s*:\s*//|www\s*\.|\b[a-z0-9][a-z0-9\-]{{1,}}\.\s?{_TLDS}\b)",
    re.IGNORECASE,
)
_PHRASES = re.compile(
    r"|".join([
        r"\bread\b.{0,40}\b(?:on|at|online|here)\b.{0,40}\b(?:site|website|web|page|app|\.\w{2,4})\b",
        r"\bread\s+(?:this|the|our)\s+(?:work|series|manga|manhwa|comic|chapter)s?\s+(?:on|at|online|only)\b",
        r"\bscan(?:s|lat\w*)\b",
        r"\b(?:translated|scanned|cleaned|redrawn|typeset(?:ted)?|proofread|edited|raws?|uploaded|released)\s+(?:by|from|on)\b",
        r"\b(?:join|follow|support)\s+(?:us|our)\b",
        r"\b(?:discord|patreon|facebook|twitter|instagram|telegram|ko-?fi)\b",
        r"\ball\s+rights\s+reserved\b",
        r"©|\(c\)\s*\d{4}|\bcopyright\b",
        r"\bofficial\s+(?:release|translation|source)\b",
        r"\bvisit\s+(?:us|our|www|http)",
        r"\bmanga\s*(?:dex|plus|park|kakalot|bat|town)\b",
    ]),
    re.IGNORECASE,
)


def is_credit_text(text: str) -> bool:
    """Whether ``text`` looks like a credit line, watermark or website advertisement.

    Such blocks should be erased from the page without being translated.

    Args:
        text: OCR text of one block.

    Returns:
        ``True`` for URLs, "read this work on ...", scanlation credits, social links,
        copyright notices and similar.
    """
    t = " ".join(text.split())
    return bool(t) and bool(_URL.search(t) or _PHRASES.search(t))
