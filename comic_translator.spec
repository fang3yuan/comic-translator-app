# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe. Build with:  pyinstaller comic_translator.spec --noconfirm
import sys
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("customtkinter", "rapidocr", "tkinterdnd2", "onnxruntime", "arabic_reshaper"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:  # optional package missing: build without it
        continue
    datas += d; binaries += b; hiddenimports += h

datas += [("fonts", "fonts"), ("assets", "assets"), ("samples", "samples")]

ICON = "assets/icon.ico" if sys.platform == "win32" else None

a = Analysis(
    ["packaging/launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["PIL._tkinter_finder"],
    excludes=["torch", "torchvision", "matplotlib", "scipy", "pandas", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ComicTranslator",
    console=False,
    icon=ICON,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ComicTranslator")
