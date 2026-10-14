"""End-to-end: OCR -> Gemini (fake) -> erase -> typeset, on a generated page."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.config import Settings, resolve_path
from src.core.models import FontRole, TextRegion
from src.core.pipeline import ComicPipeline
from src.translators import GeminiTranslator


class FakeGoogle:
    def __init__(self):
        self.payloads = []

    def __call__(self, url, headers=None, data=None, timeout=None):
        text = json.loads(data)["contents"][0]["parts"][0]["text"]
        self.payloads.append(json.loads(text))
        rows = [{"id": i["id"], "text": "ترجمة تجريبية"} for i in self.payloads[-1]["items"]]

        class R:
            status_code, headers, text = 200, {}, ""

            @staticmethod
            def json():
                return {"candidates": [{"content": {"parts": [{"text": json.dumps({"translations": rows})}]}}]}

        return R()


def _pipeline(fake):
    p = ComicPipeline(Settings(), lambda m: None)
    p._translator = GeminiTranslator("KEY_FOR_TESTS", poster=fake, min_interval=0, sleep=lambda s: None)
    return p


def test_translate_regions_skips_credits_and_sends_one_request():
    fake = FakeGoogle()
    p = _pipeline(fake)
    regions = [TextRegion(0, (0, 0, 200, 40), text="Hello there, friend"),
               TextRegion(1, (0, 50, 200, 90), text="Read at mangasite.com", ignored=True),
               TextRegion(2, (0, 100, 200, 140), text="BOOM", role=FontRole.SFX)]
    p.translate_regions(regions)
    assert len(fake.payloads) == 1  # the whole page in a single request
    assert [i["id"] for i in fake.payloads[0]["items"]] == [0, 2]  # the credit is never sent
    assert regions[0].translation and regions[1].translation == "" and regions[2].translation


def test_length_budget_is_capped_by_the_box():
    roomy = TextRegion(0, (0, 0, 600, 200), text="Hello there, friend")
    tight = TextRegion(1, (0, 0, 60, 30), text="Hello there, friend")
    assert ComicPipeline.length_budget(tight) < ComicPipeline.length_budget(roomy)
    assert ComicPipeline.length_budget(tight) >= int(0.7 * len(tight.text))


def test_full_page_with_a_fake_gemini():
    pytest.importorskip("rapidocr")
    im = Image.new("RGB", (700, 400), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(str(resolve_path("fonts/default_arabic.ttf")), 36)
    d.ellipse((60, 40, 640, 220), fill="white", outline="black", width=3)
    d.text((140, 100), "WHAT ARE YOU DOING", font=f, fill="black")
    fake = FakeGoogle()
    page = _pipeline(fake).process_image(im)
    assert len(fake.payloads) == 1 and "WHAT ARE YOU" in fake.payloads[0]["items"][0]["text"].upper()
    assert page.regions[0].translation
    assert np.abs(np.array(page.result, int) - np.array(im, int)).sum() > 0  # something was drawn
