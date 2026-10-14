"""Translation: Gemini only (free API keys), plus factory and diagnostics."""

from __future__ import annotations

from typing import List

from src.config import Settings
from src.translators.base import ConfigurationError, TranslationError, explain_error
from src.translators.gemini import (
    DEFAULT_MODEL,
    KEY_URL,
    GeminiTranslator,
    KeyCheck,
    TranslateItem,
    parse_keys,
    suggest_max_chars,
)

__all__ = [
    "ConfigurationError",
    "DEFAULT_MODEL",
    "GeminiTranslator",
    "KEY_URL",
    "KeyCheck",
    "TranslateItem",
    "TranslationError",
    "create_translator",
    "diagnose",
    "explain_error",
    "parse_keys",
    "suggest_max_chars",
]


def create_translator(settings: Settings) -> GeminiTranslator:
    """Build the Gemini translator from the settings.

    Args:
        settings: Application settings (``translation.*``).

    Returns:
        Configured ``GeminiTranslator`` (it raises ``ConfigurationError`` on first use if no key is set).
    """
    return GeminiTranslator(
        keys=settings.get("translation.gemini_keys", []),
        model=settings.get("translation.model", DEFAULT_MODEL),
        min_interval=settings.get("translation.min_interval", 1.0),
        timeout=settings.get("translation.timeout", 90),
        retries=settings.get("translation.retries", 3),
        notes=settings.get("translation.notes", ""),
    )


def diagnose(settings: Settings) -> List[KeyCheck]:
    """Test every configured key with a tiny request.

    Args:
        settings: Application settings.

    Returns:
        One ``KeyCheck`` per key (or a single failed check when no key is set).
    """
    return create_translator(settings).check_keys()
