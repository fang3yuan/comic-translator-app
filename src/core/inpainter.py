"""Text removal: a ladder of erasers, each checked for quality.

For every block of text the cheapest faithful method is used first:

1. **Plane fill** - when the paper around the letters is smooth (white, grey or a gradient),
   paint a fitted plane (plus matching grain). The mask is first grown just enough to cover
   the faint JPEG halo, without touching outlines or art.
2. **Patch copy** - screentone/hatching: copy a clean patch from nearby.
3. **Neural eraser** - LaMa (manga or general) or MI-GAN for whatever is left.
4. **OpenCV** (Telea / Navier-Stokes) - the backup, also used to undo a bad neural result.

A result whose interior is much busier or darker/brighter than its surroundings is rejected
and the next rung is tried (the idea comes from the quality checks of libre-manga-translator).
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils import modelstore

logger = logging.getLogger(__name__)

MODEL_SIDE = 512
FLAT_DEV = 3.5  # paper noise (levels, about the fitted plane) up to which a plane fill is faithful
STRONG_EDGE = 200.0  # Sobel magnitude of a real stroke; the mask never grows into these
PAPER_DIST = 60.0  # colour distance from the paper still counted as paper
RING_WIDTH = 5
GROW_STEPS = (0, 2, 4, 6, 9, 12)


def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _edges(gray: np.ndarray) -> np.ndarray:
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))


def _grow(mask: np.ndarray, k: int) -> np.ndarray:
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))


class Paper:
    """What the paper around a mask looks like: fitted plane per channel and its roughness."""

    def __init__(self, crop: np.ndarray, mask: np.ndarray, edges: np.ndarray) -> None:
        """Measure the ring just outside ``mask`` (strong edges excluded).

        Args:
            crop: BGR crop.
            mask: ``uint8`` mask (0/255) inside the crop.
            edges: Sobel magnitude of the crop.
        """
        ring = (_grow(mask, RING_WIDTH) > 0) & (mask == 0) & (edges < STRONG_EDGE)
        ys, xs = np.nonzero(ring)
        self.count = int(ys.size)
        self.dev = 999.0
        self.multimodal = False
        self.median = np.zeros(3, np.float32)
        self.coef: Optional[np.ndarray] = None
        self._origin = (float(xs.mean()), float(ys.mean())) if self.count else (0.0, 0.0)
        if self.count < 30:
            return
        if self.count > 4000:  # plenty of samples; keep it fast
            keep = np.arange(0, self.count, self.count // 4000 + 1)
            ys, xs = ys[keep], xs[keep]
        px = crop[ys, xs].astype(np.float32)
        self.median = np.median(px, axis=0)
        design = self._design(xs, ys)
        self.coef = np.linalg.lstsq(design, px, rcond=None)[0]
        resid = px - design @ self.coef
        self.dev = float(resid.std(axis=0).max())
        luma = np.sort(px @ np.array([0.114, 0.587, 0.299], np.float32))
        self.multimodal = bool(luma.size > 20 and (np.diff(luma[luma.size // 20: -luma.size // 20]) > 40).any())

    def _design(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        x0, y0 = self._origin
        return np.stack([np.ones(xs.size, np.float32), (xs - x0) / 50.0, (ys - y0) / 50.0], axis=1).astype(np.float32)

    @property
    def flat(self) -> bool:
        """Smooth enough for a plane fill to be indistinguishable from the real paper."""
        return self.coef is not None and self.dev <= FLAT_DEV and not self.multimodal

    def paint(self, where: np.ndarray, grain: bool) -> np.ndarray:
        """Plane values for the pixels of ``where`` (``N x 3``), with optional matching grain."""
        ys, xs = np.nonzero(where)
        vals = self._design(xs, ys) @ self.coef  # type: ignore[operator]
        if grain and self.dev > 0.8:
            rng = np.random.default_rng(int(xs.sum() + 31 * ys.sum()) & 0xFFFFFFFF)  # same page, same result
            vals = vals + rng.normal(0.0, min(self.dev, 2.5) * 0.7, vals.shape).astype(np.float32)
        return np.clip(vals, 0, 255)


def fit_mask(crop: np.ndarray, mask: np.ndarray, max_grow: int) -> Tuple[np.ndarray, Paper]:
    """Grow the mask until the paper around it is clean (JPEG halo covered), never into strokes.

    Args:
        crop: BGR crop.
        mask: ``uint8`` mask (0/255).
        max_grow: Largest growth in pixels.

    Returns:
        ``(mask, paper)``: the first mask whose surrounding paper is flat, or the original mask
        (with its measurement) when none is - textured areas must keep their exact mask.
    """
    edges = _edges(_luma(crop))
    base = Paper(crop, mask, edges)
    if base.coef is None:
        return mask, base
    paper_like = (edges < STRONG_EDGE) & (np.linalg.norm(crop.astype(np.float32) - base.median, axis=2) < PAPER_DIST)
    for k in GROW_STEPS:
        if k > max_grow:
            break
        cand = mask if k == 0 else np.where((_grow(mask, k) > 0) & paper_like, 255, mask).astype(np.uint8)
        paper = base if k == 0 else Paper(crop, cand, edges)
        if paper.flat:
            return cand, paper
    return mask, base


def reject_score(new: np.ndarray, orig: np.ndarray, mask: np.ndarray, reach: int) -> float:
    """How suspicious a reconstruction is: ``> 1`` means "worse than the page around it".

    Compares the edge energy inside the mask with the surroundings (up to twice as busy is
    fine) and checks that the tone stays within the range of the surroundings.

    Args:
        new: Reconstructed BGR crop.
        orig: The crop before erasing (the surroundings are measured here, they must not move).
        mask: ``uint8`` mask (0/255).
        reach: Width in pixels of the surrounding band to compare with.

    Returns:
        Score; ``0`` when there is nothing to compare with.
    """
    inside = mask > 0
    ring = (_grow(mask, max(RING_WIDTH + 3, reach)) > 0) & (_grow(mask, RING_WIDTH) == 0)
    if inside.sum() < 4 or ring.sum() < 30:
        return 0.0
    gray, ref = _luma(new), _luma(orig)
    mag, ref_mag = _edges(gray), _edges(ref)
    score = float(mag[inside].mean()) / (2.0 * float(ref_mag[ring].mean()) + 8.0)
    lo, hi = np.percentile(ref[ring], (5, 95))
    mean_in = float(gray[inside].mean())
    if mean_in < lo - 12 or mean_in > hi + 12:
        score = max(score, 2.0)
    return score


class Inpainter:
    """Removes masked pixels and reconstructs the background.

    Work is done per connected mask cluster on a cropped window, so pixels outside the mask
    are never altered and big pages stay fast. ``last_stats`` counts which rung handled
    each block in the latest call.
    """

    def __init__(
        self,
        backend: str = "auto",
        backup: str = "telea",
        radius: int = 5,
        flat_fill: bool = True,
        log: Optional[Callable[[str], None]] = None,
        patch_fill: bool = True,
        workers: int = 0,
        models: Optional[Dict[str, Path]] = None,
    ) -> None:
        """Create the inpainter.

        Args:
            backend: ``auto`` (best installed model), a model key from ``modelstore.CATALOG``,
                or ``telea`` / ``ns``.
            backup: Second choice when the first cannot load: another model or ``telea`` / ``ns``.
            radius: OpenCV inpainting radius.
            flat_fill: Allow the plane fill on smooth paper.
            log: Optional logging callback.
            patch_fill: Copy clean patches to restore textures.
            workers: Blocks erased at the same time (``0`` = automatic, a few at most).
            models: Model file locations by key (default: the ``models/`` folder).
        """
        from src.utils.sysutils import pool_workers

        self.workers = pool_workers(workers)
        self.backend, self.backup = backend, backup
        self.models = {k: Path(v) for k, v in (models or {}).items()}
        self.radius = radius
        self.flat_fill = flat_fill
        self.patch_fill = patch_fill
        self.log = log or (lambda msg: logger.info(msg))
        self.last_stats: Dict[str, int] = {}
        self.active: Optional[str] = None  # model in use (``None`` = OpenCV only)
        self._session: Any = None
        self._loaded = False
        self._cv_flag = cv2.INPAINT_NS if "ns" in (backend, backup) else cv2.INPAINT_TELEA
        self._model_lock = threading.Lock()  # the neural model runs one block at a time

    # -- model loading ---------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.models.get(key) or modelstore.model_path(key)

    def _available(self) -> List[str]:
        return [k for k in modelstore.CATALOG if self._path(k).is_file()]

    def _load(self) -> None:
        """Load the first model of the chain that works (once); OpenCV is the last resort."""
        if self._loaded:
            return
        self._loaded = True
        order = modelstore.chain(self.backend, self.backup, self._available())
        for key in order:
            if key in modelstore.OPENCV_BACKENDS:
                self._cv_flag = cv2.INPAINT_NS if key == "ns" else cv2.INPAINT_TELEA
                break
            title = modelstore.CATALOG[key].title
            try:
                import onnxruntime as ort

                from src.utils.sysutils import worker_threads

                wanted = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider",
                          "CPUExecutionProvider")
                providers = [p for p in wanted if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = worker_threads(heavy=True)  # leave half of the CPU to the user
                opts.inter_op_num_threads = 1
                opts.enable_mem_pattern = False
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(str(self._path(key)), sess_options=opts, providers=providers)
                self.active = key
                self.log(f"Eraser: {title} ({providers[0]}, {opts.intra_op_num_threads} thread(s)).")
                return
            except Exception as exc:  # noqa: BLE001
                self.log(f"{title} could not be loaded ({exc}); trying the next option.")
        if self.backend not in modelstore.OPENCV_BACKENDS and not self._available():
            self.log("No eraser model installed; using OpenCV (install one in the Eraser tab for better results).")

    # -- neural eraser ---------------------------------------------------
    def _run_model(self, crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Run the loaded network on a crop (padded to a square, resized to 512)."""
        h, w = crop.shape[:2]
        side = max(h, w)
        pad_b, pad_r = side - h, side - w
        img = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)
        msk = cv2.copyMakeBorder(crop_mask, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (MODEL_SIDE, MODEL_SIDE), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (MODEL_SIDE, MODEL_SIDE), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        names = [i.name for i in self._session.get_inputs()]
        if modelstore.CATALOG[self.active or "lama"].kind == "migan":  # input [0.5-mask, image*(1-mask)], RGB in [-1, 1]
            mk = (msk > 120).astype(np.float32)[None, None]
            erased = np.transpose(rgb * 2.0 - 1.0, (2, 0, 1))[None] * (1.0 - mk)
            out = self._session.run(None, {names[0]: np.concatenate([0.5 - mk, erased], axis=1).astype(np.float32)})[0][0]
            out = np.transpose(out, (1, 2, 0)) * 127.5 + 127.5
        else:  # LaMa: image [0, 1] and mask
            mk = (msk > 127).astype(np.float32)[None, None]
            out = self._session.run(None, {names[0]: np.transpose(rgb, (2, 0, 1))[None], names[1]: mk})[0][0]
            out = np.transpose(out, (1, 2, 0))
            if out.max() <= 1.5:
                out = out * 255.0
        out = cv2.cvtColor(np.clip(np.round(out), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)[:h, :w]

    def _run_opencv(self, crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(crop, crop_mask, self.radius, self._cv_flag)

    def _reconstruct(self, crop: np.ndarray, cmask: np.ndarray) -> Tuple[np.ndarray, str]:
        """Neural result if a model is ready (OpenCV if it fails); returns the rung used."""
        if self._session is not None:
            try:
                with self._model_lock:
                    return self._run_model(crop, cmask), "model"
            except Exception as exc:  # noqa: BLE001
                self.log(f"{self.active} failed on a block ({exc}); using OpenCV for it.")
        return self._run_opencv(crop, cmask), "opencv"

    # -- texture copy ----------------------------------------------------
    @staticmethod
    def _patch_fill(image: np.ndarray, mask: np.ndarray, box: Tuple[int, int, int, int]
                    ) -> Optional[Tuple[int, int, np.ndarray]]:
        """Fill a text cluster by copying the best-matching clean patch from nearby.

        Keeps screentones, hatching and textures that a smooth inpainting would blur away.
        The patch must match the pixels around the text closely; otherwise ``None`` is
        returned and the caller falls back to inpainting.

        Args:
            image: Full page (BGR).
            mask: Full-page text mask (0/255).
            box: ``(x, y, w, h)`` of the cluster.

        Returns:
            ``(x1, y1, patch)``: replacement pixels for the window ``box`` grown by a
            margin and its page position, or ``None`` if nothing matches well.
        """
        H, W = mask.shape
        x, y, bw, bh = box
        m = 6
        x1, y1, x2, y2 = max(0, x - m), max(0, y - m), min(W, x + bw + m), min(H, y + bh + m)
        tw, th = x2 - x1, y2 - y1
        if tw < 8 or th < 8 or tw * th > 400_000:
            return None
        sx1, sy1 = max(0, x1 - 2 * tw), max(0, y1 - 3 * th)
        sx2, sy2 = min(W, x2 + 2 * tw), min(H, y2 + 3 * th)
        area = image[sy1:sy2, sx1:sx2]
        if area.shape[0] < th or area.shape[1] < tw:
            return None
        tmpl = image[y1:y2, x1:x2]
        known = (mask[y1:y2, x1:x2] == 0).astype(np.uint8)
        score = cv2.matchTemplate(area, tmpl, cv2.TM_SQDIFF, mask=cv2.merge([known] * 3))
        # candidate windows must not contain any text pixel
        text_sum = cv2.boxFilter((mask[sy1:sy2, sx1:sx2] > 0).astype(np.float32), -1, (tw, th),
                                 normalize=False, anchor=(0, 0))[: score.shape[0], : score.shape[1]]
        score = np.where(text_sum > 0, np.inf, score)
        # never copy the window onto itself
        score[max(0, y1 - sy1 - 2): y1 - sy1 + 3, max(0, x1 - sx1 - 2): x1 - sx1 + 3] = np.inf
        if not np.isfinite(score).any():
            return None
        my, mx = np.unravel_index(int(np.argmin(score)), score.shape)
        per_px = float(score[my, mx]) / max(1, int(known.sum()) * 3)
        if per_px > 200.0:  # RMS difference ~14 grey levels on the known pixels
            return None
        cand = area[my: my + th, mx: mx + tw].astype(np.float32)
        sel = known > 0  # match the brightness of the surroundings (gradients, shading)
        cand += (tmpl.astype(np.float32) - cand)[sel].mean(axis=0)
        return x1, y1, np.clip(cand, 0, 255).astype(np.uint8)

    # -- public ----------------------------------------------------------
    @staticmethod
    def _clusters(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Group nearby mask pixels and return their bounding rectangles."""
        h, w = mask.shape
        k = max(9, int(min(h, w) * 0.012) | 1)
        merged = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=2)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours]

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Erase masked pixels and reconstruct the background.

        Args:
            image_bgr: Page image (BGR ``uint8``).
            mask: ``HxW`` ``uint8`` mask; non-zero pixels are removed.

        Returns:
            New image (the input is not modified).
        """
        result = image_bgr.copy()
        mask = (mask > 0).astype(np.uint8) * 255
        self.last_stats = {}
        if not mask.any():
            return result
        self._load()
        h, w = mask.shape
        clusters = self._clusters(mask)
        grow = max(5, int(w * 0.011))  # JPEG ringing reaches this far around letters
        stats: Counter = Counter()

        def work(box: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int, np.ndarray, np.ndarray]]:
            """Reconstruct one block from the ORIGINAL page (read-only, so blocks can run in parallel)."""
            x, y, cw, ch = box
            pad = max(24, int(0.6 * max(cw, ch)))
            x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(w, x + cw + pad), min(h, y + ch + pad)
            crop = image_bgr[y1:y2, x1:x2]
            cmask = np.zeros((y2 - y1, x2 - x1), np.uint8)  # only THIS cluster's pixels
            cmask[y - y1: y - y1 + ch, x - x1: x - x1 + cw] = mask[y: y + ch, x: x + cw]
            if not cmask.any():
                return None
            cmask, paper = fit_mask(crop, cmask, grow)
            if self.flat_fill and paper.flat:
                new = crop.copy()
                new[cmask > 0] = paper.paint(cmask > 0, grain=True)
                stats["plane"] += 1
                return x1, y1, x2, y2, new, cmask
            new, rest, rung = crop.copy(), cmask.copy(), "opencv"
            if self.patch_fill:  # textures first: copy clean patches tile by tile
                for tx in range(x, x + cw, 56):
                    tile = (tx, y, min(56, x + cw - tx), ch)
                    if not mask[y: y + ch, tile[0]: tile[0] + tile[2]].any():
                        continue
                    got = self._patch_fill(image_bgr, mask, tile)
                    if got is None:
                        continue
                    px1, py1, patch = got
                    ph, pw = patch.shape[:2]
                    oy, ox = py1 - y1, px1 - x1
                    sel = rest[oy: oy + ph, ox: ox + pw] > 0
                    new[oy: oy + ph, ox: ox + pw][sel] = patch[sel]
                    rest[oy: oy + ph, ox: ox + pw][sel] = 0
                    rung = "patch"
            if rest.any():
                new, rung = self._reconstruct(new, rest)
            reach = max(24, int(0.5 * max(cw, ch)))
            bad = reject_score(new, crop, cmask, reach)
            if bad > 1.0:  # worse than its surroundings: undo with the plain method
                alt = self._run_opencv(crop, cmask)
                if reject_score(alt, crop, cmask, reach) < bad:
                    new, rung = alt, "rollback"
            stats[rung] += 1
            return x1, y1, x2, y2, new, cmask

        if self.workers > 1 and len(clusters) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                done = list(pool.map(work, clusters))
        else:
            done = [work(c) for c in clusters]
        for item in done:  # blend in page order
            if item is None:
                continue
            x1, y1, x2, y2, new, cmask = item
            soft = cv2.GaussianBlur((cmask > 0).astype(np.float32), (0, 0), 1.2)[..., None]
            soft = np.where(cmask[..., None] > 0, 1.0, soft)
            current = result[y1:y2, x1:x2].astype(np.float32)  # keeps what earlier blocks produced
            result[y1:y2, x1:x2] = np.clip(new.astype(np.float32) * soft + current * (1 - soft), 0, 255).astype(np.uint8)
        self.last_stats = dict(stats)
        return result
