"""Application settings: defaults, persistence and path constants."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

FROZEN: bool = bool(getattr(sys, "frozen", False))  # running as the packaged .exe/.app
RESOURCE_DIR: Path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))  # read-only bundle


def _user_dir() -> Path:
    """Writable folder for settings, models, fonts and results."""
    env = os.environ.get("COMIC_TRANSLATOR_HOME")
    if env:
        return Path(env)
    if not FROZEN:
        return Path(__file__).resolve().parents[1]
    exe_dir = Path(sys.executable).resolve().parent  # portable: next to the program
    try:
        probe = exe_dir / ".write_test"
        probe.write_text("x")
        probe.unlink()
        return exe_dir
    except OSError:
        return Path.home() / "ComicTranslator"


PROJECT_ROOT: Path = _user_dir()
FONTS_DIR: Path = PROJECT_ROOT / "fonts"
MODELS_DIR: Path = PROJECT_ROOT / "models"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "settings.json"
ROLES = ["dialogue", "narration", "sfx", "whisper"]

_ROLE_STYLE_BASE: Dict[str, Any] = {
    "color": "#000000",
    "match_source_color": True,
    "stroke_enabled": False,
    "stroke_color": "#FFFFFF",
    "stroke_auto": True,
    "stroke_width": 2,
    "shadow_enabled": False,
    "shadow_color": "#000000",
    "shadow_offset": 2,
    "opacity": 1.0,
    "max_font_ratio": 1.4,
}


def _role_style(**overrides: Any) -> Dict[str, Any]:
    style = copy.deepcopy(_ROLE_STYLE_BASE)
    style.update(overrides)
    return style


SETTINGS_VERSION = 6

DEFAULT_SETTINGS: Dict[str, Any] = {
    "version": SETTINGS_VERSION,
    "general": {
        "theme": "dark",
        "reading_order": "rtl",
        "output_format": "png",
        "jpg_quality": 95,
        "save_project_json": True,
        "first_run_done": False,
    },
    "detector": {
        "mask_dilate": 3,
    },
    "ocr": {
        "min_confidence": 0.25,
        "tile_height": 1500,
        "max_side": 1280,
        "workers": 0,  # 0 = automatic (a few threads at most)
    },
    "inpainting": {
        "backend": "auto",  # auto | lama_manga | lama | migan | telea | ns
        "backup": "telea",  # used when the first choice cannot run
        "workers": 0,  # blocks erased at once; 0 = automatic (a few at most)
        "radius": 5,
        "flat_fill": True,  # plane fill on smooth paper
        "patch_fill": True,  # copy clean patches to keep screentone
    },
    "translation": {
        "gemini_keys": [],
        "model": "gemini-3.5-flash-lite",
        "fallback_models": ["gemini-flash-latest", "gemini-2.5-flash"],
        "memory": True,  # remember names/summary between pages
        "vision": False,  # send the page picture so Gemini can fix OCR mistakes
        "notes": "",
        "retries": 3,
        "min_interval": 1.0,
        "timeout": 90,
        "concise": True,
        "arabic_digits": False,
        "erase_credits": True,
        "on_failure": "raise",
    },
    "fonts": {
        "default_font": "fonts/default_arabic.ttf",
        "mappings": {"dialogue": "", "narration": "", "sfx": "", "whisper": ""},
    },
    "styles": {
        "dialogue": _role_style(),
        "narration": _role_style(max_font_ratio=1.2),
        "sfx": _role_style(
            stroke_enabled=True,
            stroke_width=3,
            max_font_ratio=3.0,
        ),
        "whisper": _role_style(max_font_ratio=1.2),
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Args:
        base: Dictionary providing default values.
        override: Dictionary whose values take precedence.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a possibly project-relative path to an absolute one.

    Relative paths are looked up in the user folder first, then in the read-only bundle of
    the packaged app (so bundled fonts are found even before they are copied).

    Args:
        path: Absolute path or path relative to ``PROJECT_ROOT``.

    Returns:
        Absolute ``Path``.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    user = PROJECT_ROOT / p
    bundled = RESOURCE_DIR / p
    return bundled if (not user.exists() and bundled.exists()) else user


def prepare_user_dirs() -> None:
    """Create the writable folders and copy the bundled fonts there (packaged app only)."""
    for d in (FONTS_DIR, MODELS_DIR, CONFIG_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
    bundled = RESOURCE_DIR / "fonts"
    if FROZEN and bundled.is_dir() and bundled != FONTS_DIR:
        for f in bundled.iterdir():
            target = FONTS_DIR / f.name
            if f.is_file() and not target.exists():
                shutil.copy2(f, target)


class Settings:
    """Nested settings store with dotted-key access and JSON persistence."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Create settings, merging ``data`` over the defaults.

        Args:
            data: Optional partial settings dictionary.
        """
        data = dict(data or {})
        if data and int(data.get("version", 1)) < SETTINGS_VERSION:
            if int(data.get("version", 1)) < 2:  # v2: translated text takes the original colour
                for style in (data.get("styles") or {}).values():
                    if isinstance(style, dict):
                        style["match_source_color"] = True
            if int(data.get("version", 1)) < 4 and isinstance(data.get("ocr"), dict):  # v4: faster OCR defaults
                data["ocr"].pop("tile_height", None)
            tr = data.get("translation")  # v5: Gemini is the only translator
            if isinstance(tr, dict):
                for old in ("provider", "use_fallback", "fallback_providers", "deepl_api_key", "retry_backoff"):
                    tr.pop(old, None)
                tr["min_interval"] = max(float(tr.get("min_interval", 1.0)), 1.0)
                tr["timeout"] = max(float(tr.get("timeout", 90)), 90.0)
                tr["retries"] = max(int(tr.get("retries", 3)), 3)
            inp = data.get("inpainting")  # v6: models live in the catalogue, backup replaces fallback
            if isinstance(inp, dict):
                for old in ("lama_model", "migan_model"):
                    inp.pop(old, None)
                if "fallback" in inp:
                    inp["backup"] = inp.pop("fallback")
            data["version"] = SETTINGS_VERSION
        self.data: Dict[str, Any] = _deep_merge(DEFAULT_SETTINGS, data)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from disk, falling back to defaults on any error.

        Args:
            path: JSON file path. Defaults to ``config/settings.json``.

        Returns:
            Loaded ``Settings`` instance.
        """
        path = path or CONFIG_PATH
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls(json.load(fh))
        except (OSError, json.JSONDecodeError):
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        """Persist settings as pretty-printed UTF-8 JSON.

        Args:
            path: Destination file. Defaults to ``config/settings.json``.
        """
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a value using ``"section.key"`` notation.

        Args:
            dotted: Dotted key path.
            default: Value returned if the key is missing.

        Returns:
            The stored value or ``default``.
        """
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        """Write a value using ``"section.key"`` notation, creating sections.

        Args:
            dotted: Dotted key path.
            value: Value to store.
        """
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def role_style(self, role: str) -> Dict[str, Any]:
        """Return the merged style dictionary for a comic role.

        Args:
            role: One of ``ROLES``.

        Returns:
            Style dictionary (copy).
        """
        return _deep_merge(_ROLE_STYLE_BASE, self.data.get("styles", {}).get(role, {}))
