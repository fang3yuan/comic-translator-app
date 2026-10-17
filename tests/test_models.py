"""Eraser model catalogue: install, import, remove, backup chain; settings migration; health check."""

import hashlib
import io

import pytest
from PIL import Image

from src.config import Settings
from src.utils import modelstore, preflight
from src.utils.image_utils import numbered_page_jpeg


@pytest.fixture()
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(modelstore, "MODELS_DIR", tmp_path)
    return tmp_path


class FakeGet:
    def __init__(self, payload: bytes, length=None):
        self.payload, self.length = payload, length

    def __call__(self, url, stream=False, timeout=None):
        outer = self

        class Resp:
            headers = {"content-length": str(outer.length if outer.length is not None else len(outer.payload))}

            def raise_for_status(self):
                pass

            def iter_content(self, n):
                yield outer.payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Resp()


def test_download_installs_atomically_and_warns_on_other_checksum(models_dir, monkeypatch):
    import requests

    data = b"x" * 1_500_000
    monkeypatch.setattr(requests, "get", FakeGet(data))
    logs, fractions = [], []
    path = modelstore.download("lama", fractions.append, logs.append)
    assert path.read_bytes() == data and not list(models_dir.glob("*.part"))
    assert fractions[-1] == 1.0 and modelstore.is_installed("lama")
    assert modelstore.download("lama") == path  # already there: nothing to do
    assert hashlib.sha256(data).hexdigest() == modelstore.sha256_of(path)


def test_truncated_download_is_never_kept(models_dir, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", FakeGet(b"x" * 1_500_000, length=3_000_000))
    with pytest.raises(RuntimeError, match="incomplete"):
        modelstore.download("lama")
    assert not modelstore.is_installed("lama") and not list(models_dir.iterdir())


def test_model_without_link_must_be_imported(models_dir, tmp_path):
    with pytest.raises(RuntimeError, match="Import"):
        modelstore.download("migan")
    src = tmp_path / "somewhere" / "my_migan.onnx"
    src.parent.mkdir()
    src.write_bytes(b"m" * 1_200_000)
    dest = modelstore.import_file("migan", src)
    assert dest.name == "migan_512.onnx" and modelstore.is_installed("migan")
    with pytest.raises(RuntimeError):
        modelstore.import_file("migan", tmp_path / "notes.txt")
    assert modelstore.remove("migan") and not modelstore.is_installed("migan")


def test_backup_chain(models_dir):
    assert modelstore.chain("auto", "telea") == ["telea"]  # nothing installed
    for key in ("lama", "migan"):
        (models_dir / modelstore.CATALOG[key].filename).write_bytes(b"x" * 1_200_000)
    assert modelstore.chain("auto", "telea") == ["lama", "migan", "telea"]
    assert modelstore.chain("lama_manga", "migan") == ["migan", "telea"]  # chosen one is missing
    assert modelstore.chain("ns", "none") == ["ns", "telea"]


def test_inpainter_falls_back_when_the_model_cannot_load(models_dir):
    import numpy as np

    from src.core.inpainter import Inpainter

    (models_dir / "lama_fp32.onnx").write_bytes(b"not a model" * 200_000)
    logs = []
    inp = Inpainter("lama", "ns", log=logs.append, models={"lama": models_dir / "lama_fp32.onnx"})
    inp.inpaint(np.full((60, 60, 3), 200, np.uint8), np.pad(np.full((10, 10), 255, np.uint8), 25))
    assert inp.active is None and any("could not be loaded" in m for m in logs)


def test_settings_migration_from_v5():
    s = Settings({"version": 5, "inpainting": {"backend": "lama", "fallback": "ns", "lama_model": "x",
                                               "migan_model": "y"}})
    inp = s.data["inpainting"]
    assert inp["backup"] == "ns" and "lama_model" not in inp and "fallback" not in inp and s.get("version") == 6
    assert s.get("translation.fallback_models") and s.get("translation.memory") is True


def test_preflight_explains_each_problem_and_where_to_fix_it(models_dir):
    issues = preflight.preflight(Settings())
    keys = [i for i in issues if i.tab == "Translation"]
    assert keys and keys[0].error
    assert any(i.tab == "Eraser" and not i.error for i in issues)  # no model installed: only a warning
    ok = Settings({"translation": {"gemini_keys": ["A" * 39]}})
    assert not [i for i in preflight.preflight(ok) if i.tab == "Translation"]


def test_preflight_blocks_on_unwritable_output(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    issues = preflight.preflight(Settings(), blocker / "sub")
    assert any(i.error and "output folder" in i.text for i in issues)


def test_numbered_page_picture_is_scaled_and_valid():
    page = Image.new("RGB", (2000, 3000), "white")
    jpeg = numbered_page_jpeg(page, {7: (100, 100, 600, 400)}, max_side=1000)
    pic = Image.open(io.BytesIO(jpeg))
    assert pic.format == "JPEG" and max(pic.size) == 1000
    r, g, b = pic.getpixel((34, 90))  # left edge of the box at 1/3 scale (x=100 -> 33)
    assert r > 140 and g < 90 and b < 90
