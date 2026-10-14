"""Shared translation errors and human-readable error explanations."""

from __future__ import annotations


class TranslationError(Exception):
    """Translation failed (after retries / after every key was tried)."""


class ConfigurationError(TranslationError):
    """Needs the user's action: no API key, invalid key, unknown model..."""


def explain_error(exc: BaseException) -> str:
    """Turn a low-level exception into a short, human-readable cause (English + Arabic).

    Args:
        exc: The exception raised while talking to the service.

    Returns:
        One-line explanation with a suggested fix.
    """
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if isinstance(exc, TranslationError):
        return str(exc)
    if any(k in low for k in ("name resolution", "getaddrinfo", "connection", "max retries", "timed out",
                              "timeout", "unreachable", "offline", "proxy")):
        return ("Cannot reach Google - check your internet / VPN / firewall. "
                "/ لا يوجد اتصال، تأكد من الإنترنت أو الـVPN أو الجدار الناري.")
    if "ssl" in low or "certificate" in low:
        return ("SSL/certificate problem - check date/time or antivirus HTTPS scanning. "
                "/ مشكلة شهادة SSL، تحقق من الوقت والتاريخ ومضاد الفيروسات.")
    return text[:300]
