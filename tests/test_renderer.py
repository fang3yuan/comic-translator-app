"""Tests for font fitting and Arabic rendering."""

import numpy as np
from PIL import Image

from src.config import Settings, resolve_path
from src.core.models import FontRole, SourceStyle, TextRegion
from src.core.pipeline import ComicPipeline
from src.core.renderer import fit_text, load_font, text_width, wrap_text

FONT = str(resolve_path("fonts/default_arabic.ttf"))
TEXT = "مرحبا بك يا صديقي العزيز في هذا العالم الواسع الجميل"


def test_fit_text_never_overflows():
    for w, h in ((120, 60), (300, 100), (80, 200), (400, 40)):
        fit = fit_text(TEXT, FONT, w, h)
        if fit.fits:
            assert fit.width <= w and fit.height <= h


def test_fit_text_is_maximal():
    fit = fit_text(TEXT, FONT, 200, 120, max_size=300)
    bigger = fit_text(TEXT, FONT, 200, 120, min_size=fit.font_size + 1, max_size=fit.font_size + 1)
    assert fit.fits and not bigger.fits


def test_bigger_box_gives_bigger_font():
    small = fit_text(TEXT, FONT, 150, 60)
    large = fit_text(TEXT, FONT, 450, 180)
    assert large.font_size > small.font_size


def test_wrap_keeps_words_and_width():
    font = load_font(FONT, 24)
    lines = wrap_text(TEXT, font, 220)
    assert " ".join(lines).split() == TEXT.split()
    assert all(text_width(font, l) <= 220 for l in lines)


def test_impossible_box_reports_not_fit():
    assert not fit_text(TEXT, FONT, 20, 10).fits


def test_render_draws_inside_box_only():
    s = Settings()
    pipe = ComicPipeline(s)
    base = Image.new("RGB", (400, 300), (255, 255, 255))
    region = TextRegion(id=0, bbox=(100, 100, 300, 200), target_bbox=(100, 100, 300, 200),
                        translation=TEXT, role=FontRole.DIALOGUE, source_style=SourceStyle(font_size=24))
    out = np.array(pipe.typesetter.render_page(base, [region]))
    dark = np.argwhere(out.sum(axis=2) < 300)
    assert dark.size
    y, x = dark[:, 0], dark[:, 1]
    assert x.min() >= 100 and x.max() <= 300 and y.min() >= 100 and y.max() <= 200


def test_disabled_region_not_rendered():
    pipe = ComicPipeline(Settings())
    base = Image.new("RGB", (200, 100), (255, 255, 255))
    region = TextRegion(id=0, bbox=(10, 10, 190, 90), translation=TEXT, enabled=False)
    assert np.array(pipe.typesetter.render_page(base, [region])).min() == 255
