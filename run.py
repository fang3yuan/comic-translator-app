#!/usr/bin/env python3
"""Launcher: ``python run.py`` (GUI) or ``python run.py cli ...``.

On start it checks the essential libraries and installs any that are missing, so
the app also works when started directly (without ``Start.bat`` / ``start.sh``).
"""

import importlib
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# import name -> pip name (the minimum needed to open the window and process a page)
ESSENTIAL = {
    "customtkinter": "customtkinter",
    "PIL": "Pillow",
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "arabic_reshaper": "arabic-reshaper",
    "bidi": "python-bidi",
    "requests": "requests",
    "onnxruntime": "onnxruntime",
    "rapidocr": "rapidocr",
    "tkinterdnd2": "tkinterdnd2",
}


def ensure_dependencies() -> None:
    """Install missing essential packages with pip (once), then continue."""
    if getattr(sys, "frozen", False):  # packaged app: everything is already inside
        return
    if "--no-install" in sys.argv:
        sys.argv.remove("--no-install")
        return
    missing = [pip for mod, pip in ESSENTIAL.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print("First run: installing required packages (this can take a few minutes)...")
    print("  " + ", ".join(missing))
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing]
    if subprocess.call(cmd) != 0:
        print("\nAutomatic installation failed. Check your internet connection and run:\n"
              f"  {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)
    importlib.invalidate_caches()


def ensure_models() -> None:
    """Download the smart eraser (LaMa, ~200 MB) before the first start.

    Without internet the app still starts and uses the simple eraser instead.
    """
    if "--download-models" in sys.argv or getattr(sys, "frozen", False):
        return  # the packaged app downloads it from the Setup window instead
    from src.config import Settings, resolve_path
    from src.core.inpainter import ensure_lama_model

    path = resolve_path(Settings.load().get("inpainting.lama_model", "models/lama_fp32.onnx"))
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    print("First run: downloading the smart text eraser (~200 MB), please wait...")
    last = [-1]

    def show(frac: float) -> None:
        pct = int(frac * 100)
        if pct // 5 != last[0]:
            last[0] = pct // 5
            print(f"  {pct}%", flush=True)

    try:
        ensure_lama_model(path, show)
        print("  done.")
    except Exception as exc:  # noqa: BLE001 - offline etc.: keep going with the simple eraser
        print(f"  could not download ({exc}).\n  The app will start with the simple eraser; "
              "open Setup later to download it.")


def check_tk() -> None:
    """Explain how to get Tk if it is missing (Linux/macOS Homebrew)."""
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        return
    if importlib.util.find_spec("tkinter") is None:
        print("Python was installed without Tk (needed for the window).\n"
              "  Windows/macOS: reinstall Python from python.org.\n"
              "  Ubuntu/Debian: sudo apt install python3-tk\n"
              "  Fedora: sudo dnf install python3-tkinter\n"
              "  Homebrew: brew install python-tk")
        sys.exit(1)


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    check_tk()
    ensure_dependencies()
    ensure_models()
    from src.main import main

    sys.exit(main())
