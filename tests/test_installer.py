"""Tests for environment checks and the pip installer."""

import subprocess

from src.config import Settings
from src.utils import installer


def test_check_environment_has_expected_components():
    comps = {c.key: c for c in installer.check_environment(Settings())}
    assert {"core", "ocr", "dnd", "fonts", "model:lama_manga", "model:lama", "model:migan"} <= set(comps)
    assert comps["fonts"].ok  # default font ships with the project
    assert comps["core"].required


def test_pip_install_streams_output_and_reports_failure(monkeypatch):
    class FakeProc:
        def __init__(self, code):
            self.stdout = iter(["Collecting x\n", "\n", "Installing x\n"])
            self._code = code

        def wait(self):
            return self._code

    lines = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc(0))
    assert installer.pip_install(["x"], lines.append) is True
    assert "Collecting x" in lines and "Installing x" in lines

    lines.clear()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc(1))
    assert installer.pip_install(["x"], lines.append) is False
    assert any("failed" in l for l in lines)


def test_install_missing_only_touches_missing_components(monkeypatch):
    calls = []
    comps = [
        installer.Component("core", "Core", True, "OK"),
        installer.Component("ocr", "OCR", False, "Missing", True, "", ["rapidocr"]),
    ]
    monkeypatch.setattr(installer, "check_environment", lambda s=None: comps)
    monkeypatch.setattr(installer, "pip_install", lambda pk, log, extra=(): calls.append(list(pk)) or True)
    assert installer.install_missing(Settings(), lambda m: None) is True
    assert calls == [["rapidocr"]]  # components that are already OK are skipped


def test_install_installs_only_the_chosen_model(monkeypatch):
    done = []
    comps = [installer.Component("model:lama_manga", "LaMa Manga", False, "x"),
             installer.Component("model:lama", "LaMa", False, "x")]
    monkeypatch.setattr(installer, "check_environment", lambda s=None: comps)
    monkeypatch.setattr(installer, "install_component", lambda c, s, log, prog=None: done.append(c.key) or True)
    installer.install_missing(Settings(), lambda m: None, model="lama_manga")
    assert done == ["model:lama_manga"]
    done.clear()
    installer.install_missing(Settings(), lambda m: None)  # no model chosen: none is downloaded
    assert done == []
