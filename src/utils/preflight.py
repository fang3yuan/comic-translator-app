"""Checks run before translating: every problem is explained and points to the tab that fixes it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import Settings, resolve_path
from src.translators.gemini import parse_keys
from src.utils import modelstore
from src.utils.installer import has_modules, check_environment


@dataclass
class Issue:
    """A problem found by ``preflight``.

    Attributes:
        error: ``True`` blocks the run, ``False`` is only a warning.
        text: What is wrong and how to fix it (English / Arabic).
        tab: Tab of the main window where it is fixed (``""`` = Setup window).
    """

    error: bool
    text: str
    tab: str = ""


def preflight(settings: Settings, out_dir: Optional[Path] = None) -> List[Issue]:
    """List everything that would stop or degrade a translation run.

    Args:
        settings: Current settings.
        out_dir: Output folder that must be writable.

    Returns:
        Issues, errors first.
    """
    issues: List[Issue] = []
    keys = parse_keys(settings.get("translation.gemini_keys", []))
    if not keys:
        issues.append(Issue(True, "No Gemini key. Paste a free key in the Translation tab. / "
                                  "لا يوجد مفتاح Gemini، الصقه في تبويب Translation.", "Translation"))
    elif any(len(k) < 30 for k in keys):
        issues.append(Issue(False, "One of the keys looks too short - copy it completely. / أحد المفاتيح قصير، "
                                   "انسخه كاملًا.", "Translation"))
    for comp in check_environment(settings):
        if comp.required and not comp.ok:
            issues.append(Issue(True, f"{comp.title} is missing. Open Setup and press Install. / "
                                      f"{comp.title} غير موجود، افتح Setup واضغط Install."))
    backend, backup = settings.get("inpainting.backend", "auto"), settings.get("inpainting.backup", "telea")
    have = modelstore.installed_models()
    if backend in modelstore.CATALOG and backend not in have:
        issues.append(Issue(False, f"{modelstore.CATALOG[backend].title} is not installed; "
                                   f"'{modelstore.chain(backend, backup)[0]}' will erase instead. / "
                                   "النموذج المختار غير مثبّت، سيُستخدم البديل.", "Eraser"))
    elif backend == "auto" and not have:
        issues.append(Issue(False, "No AI eraser installed: OpenCV will be used (install one in the Eraser tab for "
                                   "cleaner results). / لا يوجد نموذج تبييض، ثبّت واحدًا من تبويب Eraser.", "Eraser"))
    if have and not has_modules("onnxruntime"):
        issues.append(Issue(False, "onnxruntime is missing, the AI eraser cannot run. / ثبّت onnxruntime من Setup."))
    if not resolve_path(settings.get("fonts.default_font", "fonts/default_arabic.ttf")).exists():
        issues.append(Issue(True, "The default Arabic font is missing. Add a .ttf in the Fonts & Style tab. / "
                                  "الخط العربي الافتراضي مفقود.", "Fonts & Style"))
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            probe = out_dir / ".write_test"
            probe.write_text("x")
            probe.unlink()
        except OSError as exc:
            issues.append(Issue(True, f"Cannot write to the output folder ({exc}). Choose another one. / "
                                      "لا يمكن الكتابة في مجلد الإخراج.", ""))
    return sorted(issues, key=lambda i: not i.error)
