"""Series memory: what the translator should remember between pages of the same story."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

MAX_GLOSSARY = 60
MAX_SUMMARY = 600
KEEP_RECENT = 5


@dataclass
class SeriesMemory:
    """Story summary, name glossary and the last translated bubbles.

    Sent with every request so names, tone and pronouns stay consistent from page to page;
    the model returns the updated summary and glossary with each answer.

    Attributes:
        summary: Short story-so-far.
        glossary: ``{English name/term: Arabic rendering}``.
        recent: The last translated ``(English, Arabic)`` pairs.
    """

    summary: str = ""
    glossary: Dict[str, str] = field(default_factory=dict)
    recent: List[Tuple[str, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Whether nothing has been remembered yet."""
        return not (self.summary or self.glossary or self.recent)

    def clear(self) -> None:
        """Forget everything (a new series)."""
        self.summary, self.glossary, self.recent = "", {}, []

    def payload(self) -> Optional[Dict[str, Any]]:
        """The part of the request describing what is known (``None`` when empty)."""
        if self.is_empty():
            return None
        out: Dict[str, Any] = {}
        if self.summary:
            out["summary"] = self.summary
        if self.glossary:
            out["glossary"] = dict(self.glossary)
        if self.recent:
            out["last_bubbles"] = [{"en": e, "ar": a} for e, a in self.recent]
        return out

    def update(self, summary: str, glossary: Dict[str, str], pairs: Sequence[Tuple[str, str]]) -> None:
        """Merge what the model reported and remember the newest bubbles.

        Args:
            summary: New summary (kept as is when empty).
            glossary: Names/terms to add or correct.
            pairs: ``(English, Arabic)`` of the page just translated, in reading order.
        """
        if summary.strip():
            self.summary = summary.strip()[:MAX_SUMMARY]
        for src, ar in glossary.items():
            if src.strip() and ar.strip():
                self.glossary.pop(src.strip(), None)  # re-insert: newest entries survive trimming
                self.glossary[src.strip()] = ar.strip()
        while len(self.glossary) > MAX_GLOSSARY:
            self.glossary.pop(next(iter(self.glossary)))
        if pairs:
            self.recent = [(e[:160], a[:160]) for e, a in [*self.recent, *pairs]][-KEEP_RECENT:]
