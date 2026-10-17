"""Compare inpainting backends on your own pages (speed + side-by-side picture).

Usage:
    python tools/bench_inpaint.py page.png [more pages...] [--out bench_out]

It erases every text block found by OCR with each backend that is available:
OpenCV Telea and every eraser model found in models/ (LaMa Manga, LaMa, MI-GAN),
prints the time of each and saves  <page>_compare.png  (original | telea | models...).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.detector import Detector  # noqa: E402
from src.core.inpainter import Inpainter  # noqa: E402
from src.core.ocr import OCREngine  # noqa: E402
from src.utils import modelstore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()
    backends = ["telea", *modelstore.installed_models()]
    print("Backends:", ", ".join(backends))
    for key in modelstore.CATALOG:
        if not modelstore.is_installed(key):
            print(f"  ({modelstore.CATALOG[key].title} skipped: not installed)")
    Path(args.out).mkdir(exist_ok=True)
    ocr, det = OCREngine(0.25), Detector()
    for page in args.pages:
        img = cv2.imdecode(np.fromfile(page, np.uint8), cv2.IMREAD_COLOR)
        regions = det.build_regions(img, ocr.read_lines(img), "ltr")
        mask = det.build_mask(img, regions)
        print(f"\n{page}: {len(regions)} text block(s)")
        tiles = [img]
        for name in backends:
            inp = Inpainter(name, "none", log=lambda m: None)
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
