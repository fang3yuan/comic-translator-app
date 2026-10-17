"""Catalogue of the eraser models: status, download, import from a file, removal."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from src.config import MODELS_DIR

ProgressFn = Callable[[float], None]
MIN_BYTES = 1_000_000


@dataclass(frozen=True)
class ModelInfo:
    """One downloadable (or importable) inpainting network.

    Attributes:
        key: Value used in ``inpainting.backend``.
        title: Display name.
        about: One-line description shown in the UI.
        filename: File name inside ``models/``.
        url: Direct download link (``None`` = import from a file only).
        size: Rough size shown to the user.
        kind: ``"lama"`` or ``"migan"`` (selects the pre/post-processing).
        sha256: Published checksum, checked after a download (a mismatch only warns).
    """

    key: str
    title: str
    about: str
    filename: str
    url: Optional[str]
    size: str
    kind: str
    sha256: str = ""


CATALOG: Dict[str, ModelInfo] = {m.key: m for m in (
    ModelInfo("lama_manga", "LaMa Manga (recommended)",
              "LaMa fine-tuned on manga: cleanest result on screentone and line art.",
              "lama-manga-dynamic.onnx",
              "https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx",
              "~207 MB", "lama",
              "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9"),
    ModelInfo("lama", "LaMa (general)", "General-purpose LaMa: good for colour comics and artwork.",
              "lama_fp32.onnx", "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
              "~208 MB", "lama"),
    ModelInfo("migan", "MI-GAN 512 (light)", "Smaller and lighter; better on textures, slower.",
              "migan_512.onnx", "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan.onnx",
              "~30 MB", "migan"),
)}

OPENCV_BACKENDS = ("telea", "ns")
AUTO_ORDER = ("lama_manga", "lama", "migan")  # what "auto" tries first
RECOMMENDED = "lama_manga"


def model_path(key: str) -> Path:
    """Where the model file lives."""
    return MODELS_DIR / CATALOG[key].filename


def is_installed(key: str) -> bool:
    """Whether the model file exists and is not a leftover fragment."""
    path = model_path(key)
    return path.is_file() and path.stat().st_size > MIN_BYTES


def installed_models() -> List[str]:
    """Keys of the installed models, best first."""
    return [k for k in AUTO_ORDER if is_installed(k)]


def sha256_of(path: Path) -> str:
    """SHA-256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(key: str, progress: Optional[ProgressFn] = None, log: Callable[[str], None] = lambda m: None) -> Path:
    """Download a model (resumable-by-restart: a partial file is never used).

    Args:
        key: Catalogue key.
        progress: Receives a fraction in ``[0, 1]``.
        log: Receives status lines.

    Returns:
        Path of the installed file.

    Raises:
        RuntimeError: No link, network failure or truncated file.
    """
    info = CATALOG[key]
    path = model_path(key)
    if is_installed(key):
        return path
    if not info.url:
        raise RuntimeError(f"{info.title} has no download link - use 'Import file...' with the .onnx file.")
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        with requests.get(info.url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(done / total)
        if total and done != total:
            raise RuntimeError(f"incomplete download ({done} of {total} bytes)")
        if done < MIN_BYTES:
            raise RuntimeError("the server did not return a model file")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {info.title}: {exc}") from exc
    if info.sha256 and sha256_of(path) != info.sha256:
        log(f"Warning: {info.filename} differs from the published checksum (the file may have been updated upstream).")
    return path


def import_file(key: str, source: Path, log: Callable[[str], None] = lambda m: None) -> Path:
    """Copy a model file chosen by the user into ``models/`` under the expected name.

    Args:
        key: Catalogue key the file is for.
        source: The ``.onnx`` file.
        log: Receives status lines.

    Returns:
        Path of the installed copy.

    Raises:
        RuntimeError: Not an ONNX model, or too small.
    """
    source = Path(source)
    if source.suffix.lower() != ".onnx" or not source.is_file() or source.stat().st_size < MIN_BYTES:
        raise RuntimeError(f"{source.name} is not a valid .onnx model file.")
    dest = model_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    log(f"Imported {source.name} as {CATALOG[key].title}.")
    return dest


def remove(key: str) -> bool:
    """Delete an installed model; returns whether a file was removed."""
    path = model_path(key)
    if path.exists():
        path.unlink()
        return True
    return False



def chain(backend: str, backup: str, available: Optional[Iterable[str]] = None) -> List[str]:
    """Order in which erasers are tried: chosen one, backup, then OpenCV.

    Args:
        backend: ``auto`` or a catalogue key / OpenCV name.
        backup: Second choice (a catalogue key or OpenCV name, ``none`` to skip).
        available: Model keys that can be loaded (default: the installed ones).

    Returns:
        Keys to try, with unavailable models skipped and duplicates removed.
    """
    have = set(installed_models() if available is None else available)
    first = list(AUTO_ORDER) if backend == "auto" else [backend]
    seen: List[str] = []
    for k in first + [backup, "telea"]:
        if k not in seen and (k in OPENCV_BACKENDS or k in have):
            seen.append(k)
    return seen or ["telea"]
