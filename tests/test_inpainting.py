"""Tests for mask generation and inpainting (OpenCV path; LaMa if available)."""

import cv2
import numpy as np
import pytest

from src.config import resolve_path
from src.core.detector import Detector
from src.core.inpainter import Inpainter
from src.core.models import OCRLine


def _page() -> np.ndarray:
    img = np.full((300, 400, 3), (90, 140, 200), np.uint8)
    cv2.rectangle(img, (100, 80), (300, 160), (255, 255, 255), -1)
    cv2.putText(img, "HELLO", (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


def _regions(img):
    line = OCRLine([(115, 95), (260, 95), (260, 145), (115, 145)], "HELLO", 0.9)
    return Detector().build_regions(img, [line], "ltr")


def test_mask_covers_ink_only():
    img = _page()
    mask = Detector.build_mask(img, _regions(img), dilate=2)
    assert mask.dtype == np.uint8 and mask.any()
    assert not mask[200:].any()  # nothing outside the text area
    assert mask[95:145, 115:260].mean() < 255 * 0.8  # ink, not the whole polygon


def test_opencv_inpaint_removes_text_and_preserves_outside():
    img = _page()
    mask = Detector.build_mask(img, _regions(img), dilate=3)
    out = Inpainter("telea", flat_fill=False).inpaint(img, mask)
    assert (out[95:145, 115:260].sum(axis=2) < 150).sum() == 0
    outside = mask == 0
    far = cv2.erode(outside.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    assert np.array_equal(out[far], img[far])


def test_flat_fill_makes_clean_bubble():
    img = _page()
    mask = Detector.build_mask(img, _regions(img), dilate=3)
    out = Inpainter("telea", flat_fill=True).inpaint(img, mask)
    patch = out[100:140, 120:255]
    assert patch.std() < 3


def test_empty_mask_returns_copy():
    img = _page()
    out = Inpainter("telea").inpaint(img, np.zeros(img.shape[:2], np.uint8))
    assert np.array_equal(out, img) and out is not img


def test_lama_if_model_present():
    model = resolve_path("models/lama_fp32.onnx")
    if not model.exists():
        pytest.skip("LaMa model not downloaded")
    pytest.importorskip("onnxruntime")
    img = _page()
    mask = Detector.build_mask(img, _regions(img), dilate=3)
    out = Inpainter("lama", model, flat_fill=False).inpaint(img, mask)
    assert out.shape == img.shape


def test_texture_is_restored_and_clusters_do_not_overwrite_each_other():
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from src.config import resolve_path
    from src.core.inpainter import Inpainter

    h, w = 300, 600
    yy, xx = np.mgrid[0:h, 0:w]
    clean = np.stack([90 + xx * 0.1, 120 + yy * 0.1, 150 - xx * 0.05], -1)
    clean[((xx + yy) % 20) < 3] -= 30  # hatching
    clean = np.clip(clean, 0, 255).astype(np.uint8)
    im = Image.fromarray(clean)
    font = ImageFont.truetype(str(resolve_path("fonts/default_arabic.ttf")), 40)
    d = ImageDraw.Draw(im)
    d.text((40, 40), "FIRST LINE", font=font, fill=(255, 255, 255))
    d.text((40, 200), "SECOND LINE", font=font, fill=(255, 255, 255))
    page = np.array(im)
    mask = (np.abs(page.astype(int) - clean.astype(int)).sum(axis=2) > 0).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    out = Inpainter(backend="telea", flat_fill=True, patch_fill=True).inpaint(page[:, :, ::-1].copy(), mask)[:, :, ::-1]
    err = np.abs(out.astype(int) - clean.astype(int)).mean(axis=2)[mask > 0].mean()
    assert err < 8.0, err  # hatching preserved (Telea alone leaves a blurry band)


def test_migan_backend_plumbing(tmp_path):
    """The MI-GAN path packs [0.5-mask, image*(1-mask)] and decodes [-1,1] RGB (stand-in model)."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto as T
    from onnx import helper

    g = helper.make_graph(
        [helper.make_node("Slice", ["x", "s", "e", "a"], ["y"])], "stub",
        [helper.make_tensor_value_info("x", T.FLOAT, [1, 4, 512, 512])],
        [helper.make_tensor_value_info("y", T.FLOAT, [1, 3, 512, 512])],
        [helper.make_tensor("s", T.INT64, [1], [1]), helper.make_tensor("e", T.INT64, [1], [4]),
         helper.make_tensor("a", T.INT64, [1], [1])])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "migan_512.onnx"
    onnx.save(model, str(path))

    img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    mask = np.zeros((300, 400), np.uint8)
    mask[100:140, 150:260] = 255
    inp = Inpainter("migan", migan_model=path, flat_fill=False, patch_fill=False, log=lambda m: None)
    out = inp.inpaint(img, mask)
    assert inp._session is not None  # the model really ran (no OpenCV fallback)
    far = np.ones_like(mask, bool)
    far[60:180, 110:300] = False
    assert np.abs(out[far].astype(int) - img[far].astype(int)).max() == 0  # untouched outside the mask
    assert np.abs(out[100:140, 150:260].astype(int) - img[100:140, 150:260].astype(int)).mean() > 20  # erased


def test_flat_bubble_halo_is_wiped_but_outline_survives():
    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (5, 5), (294, 194), (0, 0, 0), 3)  # outline
    cv2.putText(img, "HELLO", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    rng = np.random.default_rng(0)
    near = cv2.dilate((img.min(axis=2) < 128).astype(np.uint8), np.ones((9, 9), np.uint8))
    noise = rng.integers(-18, 1, img.shape[:2])  # faint grey halo like JPEG ringing around the letters
    for c in range(3):
        ch = img[..., c].astype(int)
        img[..., c] = np.where((near > 0) & (ch == 255), 255 + noise, ch).clip(0, 255).astype(np.uint8)
    ink = np.zeros(img.shape[:2], np.uint8)
    ink[40:170, 40:260] = (img[40:170, 40:260].min(axis=2) < 100).astype(np.uint8) * 255
    ink = cv2.dilate(ink, np.ones((7, 7), np.uint8))  # as Detector.build_mask does
    out = Inpainter("telea", log=lambda m: None, workers=1).inpaint(img, ink)
    inside = out[30:180, 30:270]
    assert inside.min() >= 250  # letters and their halo are gone
    assert out[5:8, 100].max() < 60  # the outline was not touched
