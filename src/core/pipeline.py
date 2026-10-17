"""End-to-end orchestration shared by the GUI and the CLI."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from PIL import Image

from src.config import Settings
from src.core.detector import Detector
from src.core.inpainter import Inpainter
from src.core.models import TextRegion
from src.core.ocr import OCREngine, clean_ocr_text, estimate_style
from src.core.renderer import Typesetter
from src.translators import (
    GeminiTranslator,
    SeriesMemory,
    TranslateItem,
    TranslationError,
    create_translator,
    suggest_max_chars,
)
from src.utils.arabic_utils import postprocess_translation
from src.utils.font_matcher import FontMatcher, classify_roles
from src.utils.text_filters import is_credit_text
from src.utils.image_utils import bgr_to_pil, load_image, numbered_page_jpeg, pil_to_bgr, save_image

logger = logging.getLogger(__name__)

ProgressCb = Callable[[float, str], None]


@dataclass
class PageResult:
    """All data produced for one page.

    Attributes:
        source_path: Original file (``None`` for in-memory images).
        original: Untouched page.
        cleaned: Page with source text erased.
        result: Final typeset page.
        regions: Editable text blocks.
    """

    source_path: Optional[Path]
    original: Image.Image
    cleaned: Image.Image
    result: Image.Image
    regions: List[TextRegion] = field(default_factory=list)


class ComicPipeline:
    """Detect -> OCR -> erase -> translate -> typeset."""

    def __init__(self, settings: Settings, log: Optional[Callable[[str], None]] = None) -> None:
        """Create the pipeline (heavy models load lazily).

        Args:
            settings: Application settings.
            log: Optional logging callback.
        """
        self.settings = settings
        self.log = log or (lambda msg: logger.info(msg))
        self._ocr: Optional[OCREngine] = None
        self._detector: Optional[Detector] = None
        self._inpainter: Optional[Inpainter] = None
        self._translator: Optional[GeminiTranslator] = None
        self.memory = SeriesMemory()  # names / summary shared by the pages of one series
        self.fonts = FontMatcher(settings)
        self.typesetter = Typesetter(settings, self.fonts, self.log)

    # -- lazy components -------------------------------------------------
    @property
    def ocr(self) -> OCREngine:
        """OCR engine (created on first access)."""
        if self._ocr is None:
            s = self.settings
            self._ocr = OCREngine(s.get("ocr.min_confidence", 0.25), s.get("ocr.tile_height", 1500), self.log,
                                  s.get("ocr.max_side", 1280), s.get("ocr.workers", 0))
        return self._ocr

    @property
    def detector(self) -> Detector:
        """Text/bubble grouping logic."""
        if self._detector is None:
            self._detector = Detector()
        return self._detector

    @property
    def inpainter(self) -> Inpainter:
        """Inpainting engine."""
        if self._inpainter is None:
            s = self.settings
            self._inpainter = Inpainter(
                s.get("inpainting.backend", "auto"), s.get("inpainting.backup", "telea"),
                s.get("inpainting.radius", 5), s.get("inpainting.flat_fill", True), self.log,
                s.get("inpainting.patch_fill", True), s.get("inpainting.workers", 0),
            )
        return self._inpainter

    @property
    def translator(self) -> GeminiTranslator:
        """Gemini translator built from the current settings."""
        if self._translator is None:
            self._translator = create_translator(self.settings, self.memory)
        return self._translator

    def reload_settings(self, full: bool = False) -> None:
        """Drop cached components so changed settings take effect.

        Args:
            full: Also rebuild OCR/detector (reloads their models).
                Translator, inpainter, fonts and typesetter are always rebuilt (cheap until used).
        """
        self._translator = self._inpainter = None
        if full:
            self._ocr = self._detector = None
        self.fonts = FontMatcher(self.settings)
        self.typesetter = Typesetter(self.settings, self.fonts, self.log)

    # -- stages ----------------------------------------------------------
    def analyze(self, image: Image.Image) -> List[TextRegion]:
        """Detect text blocks, run OCR and estimate style/role for each.

        Args:
            image: Page image.

        Returns:
            Regions with source text, style and role (no translation yet).
        """
        bgr = pil_to_bgr(image)
        lines = self.ocr.read_lines(bgr)
        regions = self.detector.build_regions(bgr, lines, self.settings.get("general.reading_order", "rtl"))
        regions = [r for r in regions if clean_ocr_text(r.text)]
        erase_credits = bool(self.settings.get("translation.erase_credits", True))
        for r in regions:
            r.text = clean_ocr_text(r.text)
            r.source_style = estimate_style(bgr, r)
            r.ignored = erase_credits and is_credit_text(r.text)
        classify_roles(regions)
        return Detector.sort_reading_order(regions, self.settings.get("general.reading_order", "rtl"))

    @staticmethod
    def length_budget(region: TextRegion) -> int:
        """Characters the Arabic text may use: the English length, capped by what fits the box."""
        n = len(region.text.strip())
        x1, y1, x2, y2 = region.fit_bbox
        fs = max(8.0, region.source_style.font_size)
        fits = int(((x2 - x1) / (0.52 * fs)) * max(1.0, (y2 - y1) / (1.3 * fs)))
        return max(int(0.7 * n) + 3, min(suggest_max_chars(region.text), fits))

    def translate_regions(self, regions: Sequence[TextRegion], image: Optional[Image.Image] = None) -> None:
        """Translate (or re-translate) the source text of all enabled regions in ONE request.

        Blocks Gemini did not return keep their English text (and are logged) instead of failing the page.
        With ``translation.vision`` on and a page ``image``, Gemini also sees the numbered boxes and may
        correct OCR mistakes in ``region.text``.

        Args:
            regions: Regions modified in place.
            image: The page (needed for picture mode).

        Raises:
            TranslationError: If Gemini fails and ``on_failure == "raise"``.
        """
        targets = [r for r in regions if r.enabled and not r.ignored and r.text.strip()]
        for r in regions:
            if r.ignored:
                r.translation = ""
        if not targets:
            return
        items = [TranslateItem(r.id, r.text, r.role.value, self.length_budget(r)) for r in targets]
        picture = None
        if image is not None and self.settings.get("translation.vision", False):
            picture = numbered_page_jpeg(image, {r.id: r.bbox for r in targets})
        single = len(targets) == 1 and image is None  # re-translating one block: do not disturb the memory
        try:
            raw = self.translator.translate_items(items, picture, remember=not single)
        except TranslationError:
            if self.settings.get("translation.on_failure", "raise") == "keep_source":
                self.log("Translation failed; keeping source text.")
                raw = {r.id: r.text for r in targets}
            else:
                raise
        concise = self.settings.get("translation.concise", True)
        digits = self.settings.get("translation.arabic_digits", False)
        fixed = 0
        for r in targets:
            fix = self.translator.last_sources.get(r.id, "")
            if fix and fix != r.text and 0.4 <= len(fix) / max(1, len(r.text)) <= 2.5:  # a sane OCR correction
                r.text, fixed = fix, fixed + 1
            r.translation = postprocess_translation(raw.get(r.id, r.text), concise, digits)
        left = [i for i in self.translator.last_missing if i in {r.id for r in targets}]
        if left:
            self.log(f"Gemini skipped {len(left)} block(s) {left}: kept in English - select one and press Re-translate.")
        if self.translator.last_used:
            self.log(f"Translated {len(targets) - len(left)} block(s) with {self.translator.last_used}"
                     + (f", fixed {fixed} OCR mistake(s)" if fixed else "")
                     + f" ({self.translator.requests_sent} request(s) so far).")

    def clean(self, image: Image.Image, regions: Sequence[TextRegion]) -> Image.Image:
        """Erase the source text of all enabled regions.

        Args:
            image: Original page.
            regions: Regions to erase.

        Returns:
            Cleaned page.
        """
        bgr = pil_to_bgr(image)
        mask = Detector.build_mask(bgr, regions, self.settings.get("detector.mask_dilate", 3))
        out = self.inpainter.inpaint(bgr, mask)
        if self.inpainter.last_stats:
            names = {"plane": "smooth fill", "patch": "texture copy", "model": "AI eraser", "opencv": "OpenCV",
                     "rollback": "redone with OpenCV"}
            self.log("Erased: " + ", ".join(f"{n} {names.get(k, k)}" for k, n in self.inpainter.last_stats.items()))
        return bgr_to_pil(out)

    def render(self, page: PageResult) -> Image.Image:
        """Re-typeset a page from its cleaned image and current regions.

        Args:
            page: Page to render (``page.result`` is updated).

        Returns:
            The new result image.
        """
        page.result = self.typesetter.render_page(page.cleaned, page.regions)
        return page.result

    def reclean(self, page: PageResult) -> None:
        """Re-run erase + render, e.g. after toggling a block on/off."""
        page.cleaned = self.clean(page.original, page.regions)
        self.render(page)

    # -- full page -------------------------------------------------------
    def process_image(
        self,
        image: Image.Image,
        source_path: Optional[Path] = None,
        progress: Optional[ProgressCb] = None,
    ) -> PageResult:
        """Run the whole pipeline on one image.

        Args:
            image: Page image (RGB).
            source_path: Original file path, if any.
            progress: Callback ``(fraction, message)``.

        Returns:
            ``PageResult`` with all intermediate images.
        """
        step = progress or (lambda f, m: None)
        step(0.05, "Detecting text and running OCR...")
        regions = self.analyze(image)
        self.log(f"Found {len(regions)} text block(s).")
        # Translation (network) and erasing (CPU) are independent: run them side by side.
        step(0.40, "Erasing text and translating...")
        with ThreadPoolExecutor(max_workers=1) as pool:
            translation = pool.submit(self.translate_regions, regions, image)
            try:
                cleaned = self.clean(image, regions)
            finally:
                translation.result()  # re-raises TranslationError after the erase finished
        step(0.85, "Typesetting Arabic...")
        page = PageResult(source_path, image, cleaned, cleaned, regions)
        self.render(page)
        step(1.0, "Done")
        return page

    def process_file(self, path: Union[str, Path], progress: Optional[ProgressCb] = None) -> PageResult:
        """Load and process an image file.

        Args:
            path: Image path.
            progress: Progress callback.

        Returns:
            ``PageResult``.
        """
        path = Path(path)
        return self.process_image(load_image(path), path, progress)

    def export_page(self, page: PageResult, out_dir: Path, stem: Optional[str] = None) -> Path:
        """Save the final image (and optional project JSON) to ``out_dir``.

        Args:
            page: Page to save.
            out_dir: Destination directory.
            stem: File name without extension (defaults to the source stem).

        Returns:
            Path of the saved image.
        """
        s = self.settings
        stem = stem or (page.source_path.stem if page.source_path else "page")
        fmt = s.get("general.output_format", "png").lower().lstrip(".")
        out = save_image(page.result, Path(out_dir) / f"{stem}_ar.{fmt}", s.get("general.jpg_quality", 95))
        if s.get("general.save_project_json", True):
            self.save_project(page, Path(out_dir), stem)
        return out

    def process_batch(
        self,
        paths: Sequence[Path],
        out_dir: Path,
        progress: Optional[ProgressCb] = None,
        cancel: Optional[threading.Event] = None,
        on_page: Optional[Callable[[Path, PageResult], None]] = None,
    ) -> List[Path]:
        """Process many pages, exporting each as soon as it is done.

        Per-page errors are logged and skipped; a ``TranslationError`` (every
        provider failed, e.g. IP ban) aborts the batch so the caller can show
        an error dialog instead of producing untranslated pages.

        Args:
            paths: Input image paths.
            out_dir: Output directory.
            progress: Callback ``(overall_fraction, message)``.
            cancel: Set to stop after the current page.
            on_page: Called with each finished page.

        Returns:
            Paths of the exported images.

        Raises:
            TranslationError: If translation is impossible.
        """
        outputs: List[Path] = []
        total = max(1, len(paths))
        for i, path in enumerate(paths):
            if cancel is not None and cancel.is_set():
                self.log("Batch cancelled.")
                break
            base = i / total

            def sub(f: float, m: str, base: float = base, name: str = Path(path).name) -> None:
                if progress:
                    progress(base + f / total, f"[{i + 1}/{total}] {name}: {m}")

            try:
                page = self.process_file(path, sub)
                outputs.append(self.export_page(page, out_dir))
                if on_page:
                    on_page(Path(path), page)
            except TranslationError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log(f"Failed on {Path(path).name}: {exc}")
                logger.exception("Page failed")
        if progress:
            progress(1.0, f"Finished {len(outputs)}/{total} page(s)")
        return outputs

    # -- project files ---------------------------------------------------
    def save_project(self, page: PageResult, out_dir: Path, stem: str) -> Path:
        """Save regions to JSON plus the cleaned background for later editing.

        Args:
            page: Page to save.
            out_dir: Destination directory.
            stem: File name without extension.

        Returns:
            Path of the JSON file.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_path = save_image(page.cleaned, out_dir / f"{stem}_clean.png")
        data = {
            "version": 1,
            "source": str(page.source_path) if page.source_path else None,
            "cleaned": clean_path.name,
            "size": list(page.original.size),
            "regions": [r.to_dict() for r in page.regions],
        }
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return json_path

    def load_project(self, json_path: Path) -> PageResult:
        """Load a project saved by ``save_project`` and re-render it.

        Args:
            json_path: Project JSON path.

        Returns:
            Re-rendered ``PageResult``.
        """
        json_path = Path(json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        cleaned = load_image(json_path.parent / data["cleaned"])
        src = Path(data["source"]) if data.get("source") else None
        original = load_image(src) if src and src.exists() else cleaned
        page = PageResult(src, original, cleaned, cleaned, [TextRegion.from_dict(r) for r in data["regions"]])
        self.render(page)
        return page
