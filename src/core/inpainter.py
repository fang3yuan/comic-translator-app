"""Background restoration after text removal: LaMa (ONNX) with OpenCV fallback."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
LAMA_SIZE = 512


def ensure_lama_model(
    path: Path,
    progress: Optional[Callable[[float], None]] = None,
    url: str = LAMA_URL,
) -> Path:
    """Download the LaMa ONNX weights if they are not present.

    Args:
        path: Destination ``.onnx`` file.
        progress: Optional callback receiving a fraction in ``[0, 1]``.
        url: Download URL.

    Returns:
        Path to the model file.

    Raises:
        RuntimeError: If the download fails.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(done / total)
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download LaMa model from {url}: {exc}") from exc
    return path


class Inpainter:
    """Removes masked pixels and reconstructs the background.

    Processing is done per connected mask cluster on a cropped context window,
    so pixels outside the mask are never altered and large pages stay fast.
    """

    def __init__(
        self,
        backend: str = "auto",
        lama_model: Optional[Path] = None,
        fallback: str = "telea",
        radius: int = 5,
        flat_fill: bool = True,
        log: Optional[Callable[[str], None]] = None,
        patch_fill: bool = True,
        migan_model: Optional[Path] = None,
        workers: int = 0,
    ) -> None:
        """Create the inpainter.

        Args:
            backend: ``"auto"`` (LaMa if available), ``"lama"``, ``"migan"`` (lighter ONNX
                generator, needs ``migan_model``), ``"telea"`` or ``"ns"``.
            lama_model: Path to ``lama_fp32.onnx``.
            fallback: OpenCV algorithm used when LaMa is unavailable (``telea``/``ns``).
            radius: OpenCV inpainting radius.
            flat_fill: Fill clusters surrounded by a uniform colour with that colour.
            log: Optional logging callback.
            patch_fill: Restore textures (screentone, hatching) by copying clean nearby patches.
            migan_model: Path of the optional MI-GAN ONNX file.
            workers: Text blocks erased at the same time (``0`` = automatic, a few at most).
        """
        from src.utils.sysutils import pool_workers

        self.workers = pool_workers(workers)
        self._model_lock = threading.Lock()  # the neural model runs one block at a time
        self.backend = backend
        self.lama_model = Path(lama_model) if lama_model else None
        self.migan_model = Path(migan_model) if migan_model else None
        self.fallback = fallback if fallback in ("telea", "ns") else "telea"
        self.radius = radius
        self.flat_fill = flat_fill
        self.patch_fill = patch_fill
        self.log = log or (lambda msg: logger.info(msg))
        self._session: Any = None
        self._lama_failed = False

    # -- LaMa ------------------------------------------------------------
    def _lama_ready(self) -> bool:
        """Load the ONNX session if possible; remember failures."""
        if self.backend in ("telea", "ns") or self._lama_failed:
            return False
        if self._session is not None:
            return True
        model = self.migan_model if self.backend == "migan" else self.lama_model
        label = "MI-GAN" if self.backend == "migan" else "LaMa"
        if model is None or not model.exists():
            self.log(f"{label} model not found; using OpenCV inpainting"
                     + (f" (put the .onnx at {model})." if self.backend == "migan" else " (run with --download-models)."))
            self._lama_failed = True
            return False
        try:
            import onnxruntime as ort

            from src.utils.sysutils import worker_threads

            wanted = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
            providers = [p for p in wanted if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = worker_threads(heavy=True)  # leave half of the CPU to the user
            opts.inter_op_num_threads = 1
            opts.enable_mem_pattern = False  # lower peak memory
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(model), sess_options=opts, providers=providers)
            self.log(f"{label} loaded ({providers[0]}, {opts.intra_op_num_threads} thread(s)).")
            return True
        except Exception as exc:  # noqa: BLE001
            self.log(f"{label} could not be loaded ({exc}); using OpenCV inpainting.")
            self._lama_failed = True
            return False

    def _run_lama(self, crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Run LaMa on a crop and return an inpainted crop of the same size."""
        h, w = crop.shape[:2]
        side = max(h, w)
        pad_b, pad_r = side - h, side - w
        img = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)
        msk = cv2.copyMakeBorder(crop_mask, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None]
        mk = (msk > 127).astype(np.float32)[None, None]
        names = [i.name for i in self._session.get_inputs()]
        out = self._session.run(None, {names[0]: inp, names[1]: mk})[0][0]
        out = np.transpose(out, (1, 2, 0))
        if out.max() <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)
        return out[:h, :w]

    def _run_migan(self, crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Run the MI-GAN 512 generator (input ``[1,4,512,512]`` = ``[0.5-mask, image*(1-mask)]``,
        image in [-1, 1], RGB; output RGB in [-1, 1])."""
        h, w = crop.shape[:2]
        side = max(h, w)
        pad_b, pad_r = side - h, side - w
        img = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)
        msk = cv2.copyMakeBorder(crop_mask, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 * 2.0 - 1.0
        mk = (msk > 120).astype(np.float32)[None, None]
        erased = np.transpose(rgb, (2, 0, 1))[None] * (1.0 - mk)
        inp = np.concatenate([0.5 - mk, erased], axis=1).astype(np.float32)
        out = self._session.run(None, {self._session.get_inputs()[0].name: inp})[0][0]
        out = np.clip(np.round(np.transpose(out, (1, 2, 0)) * 127.5 + 127.5), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)
        return out[:h, :w]

    # -- OpenCV / flat fill ----------------------------------------------
    def _run_opencv(self, crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Inpaint a crop with Telea or Navier-Stokes."""
        flag = cv2.INPAINT_TELEA if self.fallback == "telea" else cv2.INPAINT_NS
        if self.backend == "ns":
            flag = cv2.INPAINT_NS
        elif self.backend == "telea":
            flag = cv2.INPAINT_TELEA
        return cv2.inpaint(crop, crop_mask, self.radius, flag)

    @staticmethod
    def _flat_colour(crop: np.ndarray, crop_mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Return the surrounding colour if the ring around the mask is uniform."""
        near = cv2.dilate(crop_mask, np.ones((3, 3), np.uint8), iterations=1)
        far = cv2.dilate(crop_mask, np.ones((3, 3), np.uint8), iterations=6)
        ring = (far > 0) & (near == 0)
        if ring.sum() < 30:
            return None
        px = crop[ring].astype(np.float32)
        if float(px.std(axis=0).max()) < 4.5:  # truly flat; gradients must go through inpainting
            med = np.median(px, axis=0)
            return int(med[0]), int(med[1]), int(med[2])
        return None

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

    def _reconstruct(self, crop: np.ndarray, cmask: np.ndarray, use_lama: bool) -> np.ndarray:
        """LaMa (if ready) or OpenCV reconstruction of the masked pixels."""
        if use_lama:
            try:
                with self._model_lock:
                    return (self._run_migan if self.backend == "migan" else self._run_lama)(crop, cmask)
            except Exception as exc:  # noqa: BLE001
                self.log(f"LaMa failed on a block ({exc}); using OpenCV for it.")
        return self._run_opencv(crop, cmask)

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
        if not mask.any():
            return result
        h, w = mask.shape
        use_lama = self._lama_ready()
        clusters = self._clusters(mask)
        halo = max(5, int(w * 0.011))  # JPEG ringing reaches this far around letters

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
            fill = self._flat_colour(crop, cmask) if self.flat_fill else None
            if fill is not None:
                new = crop.copy()
                new[cmask > 0] = fill
                # Flat bubble: also wipe the faint halo around the letters (JPEG noise), but never
                # touch strong colours such as the bubble outline.
                near = cv2.dilate(cmask, np.ones((2 * halo + 1, 2 * halo + 1), np.uint8)) > 0
                close = np.linalg.norm(crop.astype(np.float32) - np.array(fill, np.float32), axis=2) < 60.0
                new[near & close] = fill
                cmask = np.where(near & close, 255, cmask).astype(np.uint8)
            else:
                new, rest = crop.copy(), cmask.copy()
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
                if rest.any():  # whatever could not be copied is reconstructed
                    new = self._reconstruct(new, rest, use_lama)
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
        return result
