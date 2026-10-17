"""Eraser tab: pick the model that removes the text, install / import / remove models, backup choice."""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable, List

import customtkinter as ctk

from src.config import Settings
from src.ui.widgets import LabeledOption, section
from src.utils import modelstore
from src.utils.installer import MODEL_PREFIX, check_environment, install_component

AUTO = "auto"
NAMES = {"telea": "OpenCV Telea (built in, fast)", "ns": "OpenCV Navier-Stokes (built in)"}


class EraserPanel(ctk.CTkFrame):
    """Radio list of erasers with per-model Install / Import / Remove buttons."""

    def __init__(
        self,
        master: Any,
        settings: Settings,
        run_task: Callable[..., None],
        progress: Callable[[float, str], None],
        log: Callable[[str], None],
        on_change: Callable[[], None],
    ) -> None:
        """Create the panel.

        Args:
            master: Parent widget.
            settings: Application settings (``inpainting.*`` is read and written).
            run_task: ``App._run_task`` (background work with error dialog and busy flag).
            progress: Thread-safe progress callback ``(fraction, text)``.
            log: Log callback.
            on_change: Called after the chosen eraser or its files changed.
        """
        super().__init__(master, fg_color="transparent")
        self.settings, self._run, self._progress, self.log, self._changed = settings, run_task, progress, log, on_change
        self.choice = ctk.StringVar(value=settings.get("inpainting.backend", AUTO))
        self._rows: List[ctk.CTkBaseClass] = []
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="x")
        self.backup = LabeledOption(self, "Backup (used if the first choice cannot run)",
                                    ["telea", "ns", *modelstore.CATALOG], settings.get("inpainting.backup", "telea"),
                                    command=lambda _v: self._save())
        self.backup.pack(fill="x", pady=(8, 2))
        section(self, "Quality steps", "Cheap, faithful methods run first; the AI eraser only handles what is left.")
        self.flat = self._switch("Smooth fill on plain paper and gradients (recommended)", "inpainting.flat_fill")
        self.patch = self._switch("Copy clean patches to keep screentone", "inpainting.patch_fill")
        self.workers = LabeledOption(self, "Blocks erased at once", ["auto", "1", "2", "3", "4", "6"],
                                     "auto" if not settings.get("inpainting.workers", 0)
                                     else str(settings.get("inpainting.workers")), command=lambda _v: self._save())
        self.workers.pack(fill="x", pady=(6, 2))
        self.refresh()

    def _switch(self, text: str, key: str) -> ctk.CTkSwitch:
        sw = ctk.CTkSwitch(self, text=text, command=self._save)
        sw.pack(anchor="w", pady=3)
        if self.settings.get(key, True):
            sw.select()
        return sw

    # -- drawing -----------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the list (after an install, import or removal)."""
        for w in self._rows:
            w.destroy()
        self._rows.clear()
        self._row(AUTO, "Automatic (best installed model)", "Uses LaMa Manga, then LaMa, then MI-GAN, then OpenCV.")
        for m in modelstore.CATALOG.values():
            self._row(m.key, m.title, m.about, m)
        for k, name in NAMES.items():
            self._row(k, name, "No download needed; fine for plain bubbles, blurry on art.")

    def _row(self, key: str, title: str, about: str, info: Any = None) -> None:
        card = ctk.CTkFrame(self.body)
        card.pack(fill="x", pady=3)
        self._rows.append(card)
        installed = info is None or modelstore.is_installed(key)
        ctk.CTkRadioButton(card, text=title, variable=self.choice, value=key,
                           command=lambda: self._select(key)).pack(anchor="w", padx=8, pady=(6, 0))
        ctk.CTkLabel(card, text=about, anchor="w", justify="left", wraplength=300, text_color="gray60").pack(
            fill="x", padx=30)
        if info is None:
            return
        bar = ctk.CTkFrame(card, fg_color="transparent")
        bar.pack(fill="x", padx=30, pady=(2, 6))
        ctk.CTkLabel(bar, text=(f"✔ installed  ({info.size})" if installed else f"○ not installed  ({info.size})"),
                     text_color="#3fb950" if installed else "#d29922").pack(side="left")
        if installed:
            ctk.CTkButton(bar, text="Remove", width=70, fg_color="gray30",
                          command=lambda: self._remove(key)).pack(side="right")
            return
        ctk.CTkButton(bar, text="Import…", width=70, fg_color="gray30",
                      command=lambda: self._import(key)).pack(side="right", padx=(4, 0))
        if info.url:
            ctk.CTkButton(bar, text="Install", width=70, command=lambda: self.install(key)).pack(side="right")

    # -- actions -----------------------------------------------------------
    def _save(self) -> None:
        s = self.settings
        s.set("inpainting.backend", self.choice.get())
        s.set("inpainting.backup", self.backup.get())
        s.set("inpainting.flat_fill", bool(self.flat.get()))
        s.set("inpainting.patch_fill", bool(self.patch.get()))
        s.set("inpainting.workers", 0 if self.workers.get() == "auto" else int(self.workers.get()))
        s.save()
        self._changed()

    def _select(self, key: str) -> None:
        if key in modelstore.CATALOG and not modelstore.is_installed(key):
            info = modelstore.CATALOG[key]
            if info.url and messagebox.askyesno("Install model", f"{info.title} is not installed ({info.size}).\n"
                                                "Download it now? Until then the backup eraser is used."):
                self.install(key)
            elif not info.url:
                messagebox.showinfo("Import needed", f"{info.title} has no download link.\nPress 'Import…' and "
                                                     "choose its .onnx file.")
        self._save()

    def install(self, key: str) -> None:
        """Download a model in the background (installs onnxruntime first if needed)."""
        comp = {c.key: c for c in check_environment(self.settings)}[MODEL_PREFIX + key]

        def work() -> bool:
            return install_component(comp, self.settings, self.log, self._progress)

        def done(ok: bool) -> None:
            self.refresh()
            self._changed()
            if not ok:
                messagebox.showwarning("Install failed", "The model could not be installed - see the log below. "
                                                         "You can also download it yourself and use 'Import…'.")

        self._run(work, done, f"Installing {modelstore.CATALOG[key].title}...")

    def _import(self, key: str) -> None:
        path = filedialog.askopenfilename(title=f"Choose the .onnx file for {modelstore.CATALOG[key].title}",
                                          filetypes=[("ONNX model", "*.onnx"), ("All", "*.*")])
        if not path:
            return

        def done(_r: Any) -> None:
            self.refresh()
            self._changed()

        self._run(lambda: modelstore.import_file(key, Path(path), self.log), done, "Importing model...")

    def _remove(self, key: str) -> None:
        if messagebox.askyesno("Remove model", f"Delete {modelstore.CATALOG[key].title} from the models folder?"):
            modelstore.remove(key)
            self.refresh()
            self._changed()
