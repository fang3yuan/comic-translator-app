"""Environment checks and one-click installation of missing components."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from src.config import FONTS_DIR, Settings
from src.utils import modelstore

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]


@dataclass
class Component:
    """One installable/checkable part of the application.

    Attributes:
        key: Stable identifier (``core``, ``ocr``, ``model:lama``, ...).
        title: Display name.
        ok: Whether it is ready.
        detail: Status or hint text.
        required: Needed for the app to work at all.
        size: Rough download size shown to the user.
        packages: pip requirements installed by ``install_component``.
    """

    key: str
    title: str
    ok: bool
    detail: str
    required: bool = True
    size: str = ""
    packages: List[str] = field(default_factory=list)


def has_modules(*modules: str) -> bool:
    """Whether every module can be found (without importing it)."""
    importlib.invalidate_caches()
    return all(importlib.util.find_spec(m) is not None for m in modules)


MODEL_PREFIX = "model:"


def check_environment(settings: Optional[Settings] = None) -> List[Component]:
    """Inspect the current environment.

    Args:
        settings: Unused (kept so callers can pass the current settings).

    Returns:
        Status of each component: the required ones, then drag & drop, fonts and one entry per eraser model.
    """
    fonts = [p for p in FONTS_DIR.glob("*") if p.suffix.lower() in {".ttf", ".otf"}] if FONTS_DIR.exists() else []
    core_ok = has_modules("PIL", "numpy", "cv2", "arabic_reshaper", "bidi", "requests", "customtkinter")
    ocr_ok = has_modules("rapidocr")
    return [
        Component("core", "Core libraries (window, images, Arabic text)", core_ok,
                  "OK" if core_ok else "Missing - press Install", True, "~150 MB",
                  ["customtkinter", "Pillow", "numpy", "opencv-python-headless", "arabic-reshaper",
                   "python-bidi", "requests"]),
        Component("ocr", "Text reader (OCR)", ocr_ok,
                  "OK" if ocr_ok else "Missing - press Install", True, "~60 MB",
                  ["onnxruntime", "rapidocr"]),
        Component("dnd", "Drag & drop support", has_modules("tkinterdnd2"),
                  "OK" if has_modules("tkinterdnd2") else "Missing - restart the app after installing",
                  False, "~1 MB", ["tkinterdnd2"]),
        Component("fonts", "Arabic font", bool(fonts),
                  f"{len(fonts)} font(s) in fonts/" if fonts else "No font in fonts/ - add a .ttf file",
                  True, ""),
        *[Component(MODEL_PREFIX + m.key, m.title, modelstore.is_installed(m.key),
                    "Installed" if modelstore.is_installed(m.key)
                    else ("Not installed" if m.url else "Not installed - import the .onnx file"),
                    False, m.size, ["onnxruntime"]) for m in modelstore.CATALOG.values()],
    ]


def pip_install(packages: Sequence[str], log: LogFn, extra_args: Sequence[str] = ()) -> bool:
    """Run ``pip install`` with the current interpreter, streaming output to ``log``.

    Args:
        packages: Requirement strings.
        log: Receives each output line.
        extra_args: Extra pip arguments (e.g. ``--index-url``).

    Returns:
        ``True`` on success.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *extra_args, *packages]
    log("$ " + " ".join(cmd[2:]))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
    except OSError as exc:
        log(f"Could not start pip: {exc}")
        return False
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    code = proc.wait()
    importlib.invalidate_caches()
    if code != 0:
        log(f"pip failed with exit code {code}. Check your internet connection and try again.")
    return code == 0


def install_component(comp: Component, settings: Settings, log: LogFn, progress: Optional[ProgressFn] = None) -> bool:
    """Install or download a single component.

    Args:
        comp: Component to install.
        settings: Application settings.
        log: Log callback.
        progress: Optional ``(fraction, text)`` callback.

    Returns:
        ``True`` if the component is ready afterwards.
    """
    prog = progress or (lambda f, t: None)
    log(f"== {comp.title} ==")
    ok = True
    if comp.key.startswith(MODEL_PREFIX):
        if comp.packages and not has_modules("onnxruntime"):
            ok = pip_install(comp.packages, log)
        try:
            modelstore.download(comp.key[len(MODEL_PREFIX):], lambda f: prog(f, f"Downloading {comp.title}... {f:.0%}"), log)
            log(f"{comp.title} installed.")
        except RuntimeError as exc:
            log(str(exc))
            ok = False
    elif comp.key == "fonts":
        log("Put an Arabic .ttf/.otf file into the 'fonts' folder (see fonts/README.md).")
        ok = comp.ok
    elif comp.packages:
        ok = pip_install(comp.packages, log)
    log("Done." if ok else "FAILED.")
    return ok


def install_missing(
    settings: Settings,
    log: LogFn,
    progress: Optional[ProgressFn] = None,
    only: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
) -> bool:
    """Install what is missing: the required parts, drag & drop and ONE eraser model.

    Args:
        settings: Application settings.
        log: Log callback.
        progress: Optional ``(fraction, text)`` callback.
        only: Install exactly these component keys instead.
        model: Eraser model key to install with the essentials (``None`` = none).

    Returns:
        ``True`` if all attempted components succeeded.
    """
    prog = progress or (lambda f, t: None)

    def wanted(c: Component) -> bool:
        if c.ok or c.key == "fonts":
            return False
        if only is not None:
            return c.key in only
        if c.key.startswith(MODEL_PREFIX):
            return model is not None and c.key == MODEL_PREFIX + model
        return True

    todo = [c for c in check_environment(settings) if wanted(c)]
    if not todo:
        log("Everything is already installed.")
        prog(1.0, "Everything is installed")
        return True
    all_ok = True
    for i, comp in enumerate(todo):
        prog(i / len(todo), f"Installing: {comp.title}")
        all_ok &= install_component(comp, settings, log, lambda f, t, i=i: prog((i + f) / len(todo), t))
    prog(1.0, "Finished" if all_ok else "Finished with errors - see the log")
    return all_ok


def missing_required(settings: Optional[Settings] = None) -> List[Component]:
    """Required components that are not ready."""
    return [c for c in check_environment(settings) if c.required and not c.ok]


def open_folder(path: Path) -> None:
    """Open a folder in the system file manager.

    Args:
        path: Directory to open.
    """
    import os

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass
