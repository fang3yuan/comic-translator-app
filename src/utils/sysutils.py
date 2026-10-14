"""Small operating-system helpers."""

from __future__ import annotations

import os
import sys


def lower_priority() -> None:
    """Run below normal priority so heavy work (OCR, LaMa) never makes the PC feel stuck."""
    try:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        elif hasattr(os, "nice"):
            os.nice(5)
    except Exception:  # noqa: BLE001 - purely a courtesy, never fatal
        pass


def worker_threads(heavy: bool = False) -> int:
    """Threads for ONNX inference, leaving cores free for the user.

    Args:
        heavy: ``True`` for the big eraser model (half of the cores); text reading uses a bit more.

    Returns:
        Thread count (at least 1 for heavy, 2 otherwise).
    """
    cores = os.cpu_count() or 2
    return max(1, cores // 2) if heavy else max(2, cores // 2 + 1)
