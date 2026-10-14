"""Tests: text of any colour is detected, erased and re-typeset in the same colour."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.config import SETTINGS_VERSION, Settings, resolve_path
from src.core.detector import Detector
from src.core.models import FontRole, OCRLine, SourceStyle, TextRegion
from src.core.ocr import estimate_style
from src.core.pipeline import ComicPipeline
from src.utils.color_utils import analyze_text_pixels, colors_too_similar, contrast_color

FONT = ImageFont.truetype(str(resolve_path("fonts/default_arabic.ttf")), 34)
CASES = [
    ((0, 0, 0), (255, 255, 255)),
    ((255, 255, 255), (20, 20, 20)),
    ((200, 20, 20), (255, 230, 120)),
    ((255, 220, 40), (20, 30, 90)),
    ((30, 170, 60), (120, 120, 120)),   # same brightness as the background
    ((190, 60, 60), (110, 110, 110)),
]


def _page(fg, bg):
    im = Image.new("RGB", (500, 120), bg)
    d = ImageDraw.Draw(im)
    d.text((20, 35), "HELLO WORLD", font=FONT, fill=fg)
    x1, y1, x2, y2 = d.textbbox((20, 35), "HELLO WORLD", font=FONT)
    line = OCRLine([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], "HELLO WORLD", 0.9)
    return np.array(im)[:, :, ::-1].copy(), line


@pytest.mark.parametrize("fg,bg", CASES)
def test_ink_and_background_colours_are_estimated(fg, bg):
    img, line = _page(fg, bg)
    region = Detector().build_regions(img, [line], "ltr")[0]
    style = estimate_style(img, region)
    assert np.abs(np.array(style.color) - fg).max() <= 12
    assert np.abs(np.array(style.bg_color) - bg).max() <= 12


@pytest.mark.parametrize("fg,bg", CASES)
def test_mask_erases_ink_of_any_colour(fg, bg):
    img, line = _page(fg, bg)
    region = Detector().build_regions(img, [line], "ltr")[0]
    mask = Detector.build_mask(img, [region], 2)
    ink = np.abs(img[:, :, ::-1].astype(int) - np.array(bg)).sum(axis=2) > 60
    assert (mask > 0)[ink].mean() > 0.98


def test_no_ink_when_text_equals_background():
    img = np.full((100, 200, 3), 128, np.uint8)
    poly = [(20, 30), (180, 30), (180, 70), (20, 70)]
    assert analyze_text_pixels(img, [poly]) is None


def test_defaults_use_source_colour_and_old_settings_are_migrated():
    s = Settings()
    assert all(s.role_style(r)["match_source_color"] for r in ("dialogue", "narration", "sfx", "whisper"))
    old = Settings({"styles": {"dialogue": {"match_source_color": False}}})  # no version key = v1 file
    assert old.role_style("dialogue")["match_source_color"] is True
    assert old.get("version") == SETTINGS_VERSION


@pytest.mark.parametrize("fg,bg", CASES[1:4])
def test_arabic_is_drawn_in_the_source_colour(fg, bg):
    pipe = ComicPipeline(Settings())
    base = Image.new("RGB", (400, 200), bg)
    region = TextRegion(id=0, bbox=(50, 50, 350, 150), target_bbox=(50, 50, 350, 150),
                        translation="مرحبا بالعالم", role=FontRole.DIALOGUE,
                        source_style=SourceStyle(font_size=30, color=fg, bg_color=bg))
    out = np.array(pipe.typesetter.render_page(base, [region])).reshape(-1, 3)
    diff = np.abs(out.astype(int) - np.array(bg)).sum(axis=1) > 150
    assert diff.any()
    assert np.abs(np.median(out[diff], axis=0) - fg).max() <= 20


def test_unreadable_estimate_falls_back_to_contrast():
    pipe = ComicPipeline(Settings())
    region = TextRegion(id=0, bbox=(0, 0, 100, 50), translation="x",
                        source_style=SourceStyle(color=(250, 250, 250), bg_color=(255, 255, 255)))
    assert pipe.typesetter.style_for(region).color == (0, 0, 0)
    assert colors_too_similar((250, 250, 250), (255, 255, 255)) and contrast_color((255, 255, 255)) == (0, 0, 0)


def test_same_row_fragments_are_merged_in_reading_order():
    img = np.full((200, 700, 3), 255, np.uint8)
    lines = [OCRLine([(235, 62), (415, 62), (415, 98), (235, 98)], "ARE YOU", 0.9),
             OCRLine([(40, 60), (200, 60), (200, 100), (40, 100)], "WHAT", 0.9),
             OCRLine([(440, 61), (590, 61), (590, 99), (440, 99)], "DOING", 0.9)]
    regions = Detector().build_regions(img, lines, "ltr")
    assert len(regions) == 1 and regions[0].text == "WHAT ARE YOU DOING"
