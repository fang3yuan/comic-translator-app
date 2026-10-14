"""PIL / OpenCV conversion and geometry helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.core.models import BBox, Point, RGB

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PathLike = Union[str, Path]


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR ``uint8`` array.

    Args:
        image: Any PIL image.

    Returns:
        ``HxWx3`` BGR array.
    """
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(array: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image.

    Args:
        array: ``HxWx3`` BGR array.

    Returns:
        PIL RGB image.
    """
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def load_image(path: PathLike) -> Image.Image:
    """Load an image as RGB, honouring EXIF orientation.

    Args:
        path: Image file path.

    Returns:
        PIL RGB image.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def save_image(image: Image.Image, path: PathLike, quality: int = 95) -> Path:
    """Save an image, choosing parameters from the file extension.

    Args:
        image: PIL image.
        path: Destination path (extension decides the format).
        quality: JPEG/WebP quality.

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, quality=quality, subsampling=0)
    elif ext == ".webp":
        image.save(path, quality=quality)
    else:
        image.save(path)
    return path


def natural_key(value: str) -> List[Union[int, str]]:
    """Sort key giving natural ordering (``page2`` before ``page10``)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", value)]


def list_images(folder: PathLike, recursive: bool = False) -> List[Path]:
    """List supported images in a folder using natural sort order.

    Args:
        folder: Directory to scan.
        recursive: Whether to descend into sub-directories.

    Returns:
        Sorted list of image paths.
    """
    folder = Path(folder)
    it: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: natural_key(str(p)))


def hex_to_rgb(value: str) -> RGB:
    """Convert ``#rrggbb`` to an RGB tuple."""
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb_to_hex(rgb: Sequence[int]) -> str:
    """Convert an RGB tuple to ``#rrggbb``."""
    return "#{:02x}{:02x}{:02x}".format(*[int(max(0, min(255, c))) for c in rgb[:3]])


def polygon_to_bbox(polygon: Sequence[Point]) -> BBox:
    """Axis-aligned bounding box of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """Clamp a box into ``[0, width] x [0, height]``."""
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, ratio: float, width: int, height: int) -> BBox:
    """Grow a box by ``ratio`` of its size on every side and clamp it.

    Args:
        bbox: Source box.
        ratio: Fractional growth per side (0.1 = 10 %).
        width: Image width.
        height: Image height.

    Returns:
        Expanded, clamped box.
    """
    x1, y1, x2, y2 = bbox
    dw, dh = (x2 - x1) * ratio, (y2 - y1) * ratio
    return clamp_bbox((int(x1 - dw), int(y1 - dh), int(x2 + dw), int(y2 + dh)), width, height)


def shrink_bbox(bbox: BBox, factor: float) -> BBox:
    """Scale a box about its centre (``factor < 1`` shrinks it)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    return int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)


def bbox_union(boxes: Sequence[BBox]) -> BBox:
    """Smallest box containing all ``boxes``."""
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def bbox_area(bbox: BBox) -> int:
    """Area of a box (0 if degenerate)."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def polygon_mask(shape: Tuple[int, int], polygons: Sequence[Sequence[Point]]) -> np.ndarray:
    """Rasterise polygons into a binary ``uint8`` mask (0/255).

    Args:
        shape: ``(height, width)``.
        polygons: Polygons to fill.

    Returns:
        Mask array.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask
