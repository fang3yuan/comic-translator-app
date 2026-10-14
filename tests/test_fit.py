"""Translations are fitted inside the text box and never drawn outside it."""

from PIL import Image

from src.config import Settings, resolve_path
from src.core.models import TextRegion
from src.core.renderer import Typesetter, fit_text, load_font, text_width

FONT = str(resolve_path("fonts/default_arabic.ttf"))
LONG = "لقد انتظرت هذه اللحظة طويلا ولن أسمح لأحد بأن يقف في طريقي مهما كان الثمن الذي سأدفعه"


def test_fit_keeps_every_line_inside_the_box():
    fit = fit_text(LONG, FONT, 200, 120)
    font = load_font(FONT, fit.font_size)
    assert fit.fits and fit.height <= 120
    assert all(text_width(font, line) <= 200 for line in fit.lines)


def test_longer_text_gets_a_smaller_font():
    short = fit_text("نعم", FONT, 200, 120)
    long = fit_text(LONG, FONT, 200, 120)
    assert long.font_size < short.font_size


def test_render_never_paints_outside_the_target_box():
    page = Image.new("RGBA", (400, 300), (255, 255, 255, 255))
    region = TextRegion(id=0, bbox=(100, 100, 300, 200), target_bbox=(90, 90, 310, 210), translation=LONG)
    Typesetter(Settings()).render_region(page, region)
    px = page.convert("L").load()
    for y in range(300):
        for x in range(400):
            if px[x, y] < 200:
                assert 90 - 3 <= x <= 310 + 3 and 90 - 3 <= y <= 210 + 3, (x, y)
