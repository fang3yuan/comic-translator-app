"""Compare inpainting backends on your own pages (speed + side-by-side picture).

Usage:
    python tools/bench_inpaint.py page.png [more pages...] [--out bench_out]

It erases every text block found by OCR with each backend that is available:
OpenCV Telea, LaMa (models/lama_fp32.onnx) and MI-GAN (models/migan_512.onnx),
prints the time of each and saves  <page>_compare.png  (original | telea | lama | migan).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings, resolve_path  # noqa: E402
from src.core.detector import Detector  # noqa: E402
from src.core.inpainter import Inpainter  # noqa: E402
from src.core.ocr import OCREngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()
    s = Settings.load()
    lama = resolve_path(s.get("inpainting.lama_model", "models/lama_fp32.onnx"))
    migan = resolve_path(s.get("inpainting.migan_model", "models/migan_512.onnx"))
    backends = [("telea", None)]
    backends += [("lama", lama)] if lama.exists() else []
    backends += [("migan", migan)] if migan.exists() else []
    print("Backends:", ", ".join(b for b, _ in backends))
    for name, path in (("LaMa", lama), ("MI-GAN", migan)):
        if not path.exists():
            print(f"  ({name} skipped: {path} not found)")
    Path(args.out).mkdir(exist_ok=True)
    ocr, det = OCREngine(0.25), Detector()
    for page in args.pages:
        img = cv2.imdecode(np.fromfile(page, np.uint8), cv2.IMREAD_COLOR)
        regions = det.build_regions(img, ocr.read_lines(img), "ltr")
        mask = det.build_mask(img, regions)
        print(f"\n{page}: {len(regions)} text block(s)")
        tiles = [img]
        for name, path in backends:
            inp = Inpainter(name, lama_model=lama, migan_model=migan, log=lambda m: None)
            inp.inpaint(img, mask)  # warm-up (model load)
            t = time.perf_counter()
            res = inp.inpaint(img, mask)
            print(f"  {name:6s} {time.perf_counter() - t:6.2f} s")
            tiles.append(res)
        h = 700
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        out = Path(args.out) / (Path(page).stem + "_compare.png")
        ok, buf = cv2.imencode(".png", np.hstack(tiles))
        buf.tofile(str(out))
        print("  saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
