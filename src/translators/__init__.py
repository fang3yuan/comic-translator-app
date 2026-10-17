"""Translation: Gemini only (free API keys), plus factory and diagnostics."""

from __future__ import annotations

from typing import List, Optional

from src.config import Settings
from src.translators.base import ConfigurationError, TranslationError, explain_error
from src.translators.memory import SeriesMemory
from src.translators.gemini import (
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    KEY_URL,
    KNOWN_MODELS,
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
    "FALLBACK_MODELS",
    "KEY_URL",
    "KNOWN_MODELS",
    "KeyCheck",
    "SeriesMemory",
    "TranslateItem",
    "TranslationError",
    "create_translator",
    "diagnose",
    "explain_error",
    "parse_keys",
    "suggest_max_chars",
]


def create_translator(settings: Settings, memory: Optional[SeriesMemory] = None) -> GeminiTranslator:
    """Build the Gemini translator from the settings.

    Args:
        settings: Application settings (``translation.*``).
        memory: Series memory to share between pages (used only if ``translation.memory`` is on).

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
        fallback_models=settings.get("translation.fallback_models", list(FALLBACK_MODELS)) or (),
        memory=memory if settings.get("translation.memory", True) else None,
    )


def diagnose(settings: Settings) -> List[KeyCheck]:
    """Test every configured key with a tiny request.

    Args:
        settings: Application settings.

    Returns:
        One ``KeyCheck`` per key (or a single failed check when no key is set).
    """
    return create_translator(settings).check_keys()
