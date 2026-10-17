"""Main CustomTkinter window."""

from __future__ import annotations

import queue
import shutil
import threading
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

from src.config import FONTS_DIR, ROLES, Settings
from src.core.models import BBox, FontRole, TextRegion
from src.core.pipeline import ComicPipeline, PageResult
from src.translators import DEFAULT_MODEL, FALLBACK_MODELS, KEY_URL, KNOWN_MODELS, TranslationError, create_translator, diagnose, parse_keys
from src.ui.canvas import ImageCanvas
from src.ui.eraser_panel import EraserPanel
from src.ui.setup_dialog import SetupDialog
from src.ui.widgets import (
    install_clipboard_fix,
    ColorButton,
    ErrorModal,
    LabeledCombo,
    LabeledOption,
    LabeledSlider,
    LogBox,
    ProgressPanel,
    section,
)
from src.utils.image_utils import IMAGE_EXTENSIONS, list_images, load_image
from src.utils import modelstore
from src.utils.installer import missing_required, open_folder
from src.utils.preflight import preflight

try:  # optional: drag & drop support
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except Exception:  # noqa: BLE001 - missing package or Tk without tkdnd binaries
    HAS_DND = False

VIEW_MODES = ["Original", "Processed", "Side-by-side"]
AUTO_FONT = "(auto)"
FAILURE_CHOICES = {"raise": "Stop and show the error", "keep_source": "Keep the English text and go on"}


class ComicTranslatorApp(ctk.CTk):
    """Desktop front-end for the comic translation pipeline."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Build the window.

        Args:
            settings: Preloaded settings (loaded from disk if omitted).
        """
        self.settings = settings or Settings.load()
        ctk.set_appearance_mode(self.settings.get("general.theme", "dark"))
        ctk.set_default_color_theme("blue")
        super().__init__()
        install_clipboard_fix(self)
        self._dnd_ok = False
        if HAS_DND:
            try:
                TkinterDnD._require(self)
                self._dnd_ok = True
            except Exception:  # noqa: BLE001
                self._dnd_ok = False
        self._setup: Optional[SetupDialog] = None
        self.title("Comic Translator — English → Arabic")
        self.geometry("1500x900")
        self.minsize(1100, 700)

        self._events: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._busy = False
        self._cancel = threading.Event()
        self.files: List[Path] = []
        self.pages: Dict[Path, PageResult] = {}
        self.current: Optional[Path] = None
        self.selected_region: Optional[int] = None
        self.out_dir: Optional[Path] = None
        self._file_buttons: Dict[Path, ctk.CTkButton] = {}
        self._block_buttons: Dict[int, ctk.CTkButton] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_topbar()
        self._build_sidebar()
        self._build_center()
        self._build_right_panel()
        self._build_bottom()

        self.pipeline = ComicPipeline(self.settings, self.log)
        self._load_style_controls()
        self._refresh_status()
        self.canvas.placeholder = (
            "Drop comic pages or a folder here\nاسحب صفحات الكوميكس أو المجلد وأفلتها هنا"
            if self._dnd_ok else "Press 'Add Images' to begin\n(Drag & drop: open Setup and install it)"
        )
        self.canvas.set_layers([])
        self._setup_dnd()
        self.after(100, self._poll_events)
        self.after(700, self._first_run_check)
        self.log("Ready. Drop images or a folder into the window, then press 'Translate all'.")

    # ------------------------------------------------------------------
    # Setup / drag & drop
    # ------------------------------------------------------------------
    def open_setup(self, autostart: Optional[str] = None) -> None:
        """Open (or focus) the Setup window.

        Args:
            autostart: ``"all"`` or a component key to install immediately.
        """
        if self._setup is not None and self._setup.winfo_exists():
            self._setup.lift()
            return
        self._setup = SetupDialog(self, self.settings, on_change=self._after_setup, autostart=autostart)

    def _after_setup(self) -> None:
        """Rebuild engines after something was installed."""
        self.pipeline.reload_settings(full=True)
        self.eraser.refresh()
        self._refresh_status()

    def _first_run_check(self) -> None:
        """Open Setup on the first launch or when a required part is missing."""
        first = not self.settings.get("general.first_run_done", False)
        if first or missing_required(self.settings):
            self.settings.set("general.first_run_done", True)
            self.settings.save()
            self.open_setup()

    def _setup_dnd(self) -> None:
        """Register the main widgets as file drop targets."""
        if not self._dnd_ok:
            return
        for widget in (self, self.canvas, self.file_list, self.center_frame, self.side_frame):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:  # noqa: BLE001
                continue

    def _on_drop(self, event: Any) -> Any:
        """Handle files/folders dropped on the window."""
        paths: List[Path] = []
        for raw in self.tk.splitlist(event.data):
            p = Path(raw)
            if p.is_dir():
                paths.extend(list_images(p))
            elif p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
            elif p.suffix.lower() == ".json" and p.exists():
                self._load_project_file(p)
        if paths:
            self._add_files(paths)
        return event.action

    # ------------------------------------------------------------------
    # Threading helpers
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        """Thread-safe log line."""
        self.log_box.write(message)

    def _post(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the Tk main thread."""
        self._events.put(fn)

    def _poll_events(self) -> None:
        """Execute callbacks queued by worker threads."""
        try:
            while True:
                self._events.get_nowait()()
        except queue.Empty:
            pass
        self.after(60, self._poll_events)

    def _run_task(self, work: Callable[[], Any], done: Optional[Callable[[Any], None]] = None,
                  label: str = "Working...") -> None:
        """Run ``work`` in a background thread with error reporting.

        Args:
            work: Function executed off the UI thread; its result goes to ``done``.
            done: Called on the UI thread with the result.
            label: Initial status text.
        """
        if self._busy:
            messagebox.showinfo("Busy", "Another task is still running.")
            return
        self._busy = True
        self._cancel.clear()
        self.progress.update_progress(0.0, label)

        def runner() -> None:
            try:
                result = work()
            except TranslationError as exc:
                msg = str(exc)  # `exc` is unbound after the except block, so capture it now
                self._post(lambda: self._fail("Translation failed", msg))
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                self._post(lambda: self._fail("Error", msg))
            else:
                if done:
                    self._post(lambda: done(result))
            finally:
                self._post(self._task_finished)

        threading.Thread(target=runner, daemon=True).start()

    def _task_finished(self) -> None:
        """Reset the busy flag."""
        self._busy = False

    def _fail(self, title: str, message: str) -> None:
        """Log an error and show the error modal."""
        self.log(f"ERROR: {message}")
        self.progress.update_progress(0.0, title)
        low = message.lower()
        action = None
        if any(k in low for k in ("api key", "keys were rejected", "every key", "limit", "model not available",
                                  "none of the selected models")):
            action = ("Open Translation tab", lambda: self.tabs.set("Translation"))
        elif any(k in low for k in ("onnx", "eraser", "inpaint")):
            action = ("Open Eraser tab", lambda: self.tabs.set("Eraser"))
        elif any(k in low for k in ("setup", "install")):
            action = ("Open Setup", self.open_setup)
        ErrorModal(self, title, message, action=action)

    def _progress_cb(self, fraction: float, message: str) -> None:
        """Thread-safe progress update."""
        self._post(lambda: self.progress.update_progress(fraction, message))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        """Left column: file list and actions."""
        side = ctk.CTkFrame(self, width=270)
        self.side_frame = side
        side.grid(row=1, column=0, sticky="nsw", padx=(8, 4), pady=8)
        side.grid_propagate(False)
        side.grid_rowconfigure(5, weight=1)
        ctk.CTkButton(side, text="🛠  Setup / Install", fg_color="#6e40c9", hover_color="#5a32a3",
                      command=self.open_setup).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.grid(row=1, column=0, sticky="ew", padx=8)
        ctk.CTkButton(row, text="Add Images", width=110, command=self._add_images).pack(side="left", padx=2)
        ctk.CTkButton(row, text="Add Folder", width=110, command=self._add_folder).pack(side="left", padx=2)
        ctk.CTkButton(side, text="▶  Translate this page", command=self._process_current).grid(
            row=2, column=0, sticky="ew", padx=10, pady=(10, 2))
        ctk.CTkButton(side, text="▶▶  Translate all pages", height=38, command=self._process_all).grid(
            row=3, column=0, sticky="ew", padx=10, pady=2)
        ctk.CTkButton(side, text="Cancel", fg_color="#b33a3a", hover_color="#8f2d2d", command=self._cancel.set).grid(
            row=4, column=0, sticky="ew", padx=10, pady=2)
        self.file_list = ctk.CTkScrollableFrame(side, label_text="Pages (drop files here)")
        self.file_list.grid(row=5, column=0, sticky="nsew", padx=8, pady=6)
        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 8))
        for text, cmd in (
            ("📂 Open output folder", self._open_output),
            ("Change output folder", self._choose_out_dir),
            ("Export this page", self._export_current),
            ("Save project (edit later)", self._save_project),
            ("Open project", self._open_project),
            ("Clear list", self._clear_files),
        ):
            ctk.CTkButton(bottom, text=text, command=cmd, fg_color="gray30").pack(fill="x", pady=2)

    def _build_center(self) -> None:
        """Centre column: view controls and canvas."""
        center = ctk.CTkFrame(self)
        self.center_frame = center
        center.grid(row=1, column=1, sticky="nsew", padx=4, pady=8)
        center.grid_rowconfigure(1, weight=1)
        center.grid_columnconfigure(0, weight=1)
        bar = ctk.CTkFrame(center, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.view_mode = ctk.CTkSegmentedButton(bar, values=VIEW_MODES, command=lambda _v: self._refresh_view())
        self.view_mode.set("Processed")
        self.view_mode.pack(side="left")
        self.show_boxes = ctk.CTkSwitch(bar, text="Boxes", command=lambda: self._refresh_view(True))
        self.show_boxes.select()
        self.show_boxes.pack(side="left", padx=12)
        ctk.CTkButton(bar, text="Fit", width=60, command=lambda: self.canvas.fit_to_window()).pack(side="left")
        ctk.CTkLabel(bar, text="Wheel: zoom · Drag: pan · Drag box: move/resize",
                     text_color="gray60").pack(side="right")
        self.canvas = ImageCanvas(center, on_box_changed=self._on_box_changed, on_box_selected=self._on_box_selected)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

    def _build_right_panel(self) -> None:
        """Right column: tabs for translation, fonts, blocks and general settings."""
        tabs = ctk.CTkTabview(self, width=380)
        tabs.grid(row=1, column=2, sticky="nse", padx=(4, 8), pady=8)
        self.tabs = tabs
        for name in ("Translation", "Eraser", "Fonts & Style", "Blocks", "General"):
            tabs.add(name)
        self._build_translation_tab(self._scroll(tabs.tab("Translation")))
        self._build_eraser_tab(self._scroll(tabs.tab("Eraser")))
        self._build_style_tab(self._scroll(tabs.tab("Fonts & Style")))
        self._build_blocks_tab(tabs.tab("Blocks"))
        self._build_general_tab(self._scroll(tabs.tab("General")))

    @staticmethod
    def _scroll(tab: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
        """A scrollable body for a tab (settings never get cut off on small screens)."""
        body = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True)
        return body

    def _build_topbar(self) -> None:
        """Top strip: what is configured right now; each chip jumps to the tab that changes it."""
        bar = ctk.CTkFrame(self, height=40)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 0))
        self.chip_gemini = ctk.CTkButton(bar, text="", fg_color="gray25", height=28,
                                         command=lambda: self.tabs.set("Translation"))
        self.chip_eraser = ctk.CTkButton(bar, text="", fg_color="gray25", height=28,
                                         command=lambda: self.tabs.set("Eraser"))
        for chip in (self.chip_gemini, self.chip_eraser):
            chip.pack(side="left", padx=(8, 0), pady=6)
        ctk.CTkButton(bar, text="🛠 Setup", width=80, height=28, fg_color="#6e40c9", hover_color="#5a32a3",
                      command=self.open_setup).pack(side="right", padx=8, pady=6)
        ctk.CTkButton(bar, text="✔ Check everything", width=140, height=28, fg_color="gray30",
                      command=self._health_check).pack(side="right", pady=6)

    def _refresh_status(self) -> None:
        """Update the top chips from the saved settings."""
        s = self.settings
        n = len(parse_keys(s.get("translation.gemini_keys", [])))
        self.chip_gemini.configure(
            text=f"{'🔑' if n else '⚠'} Gemini: {n} key(s) · {s.get('translation.model', DEFAULT_MODEL)}",
            text_color="#3fb950" if n else "#ff6b6b")
        first = modelstore.chain(s.get("inpainting.backend", "auto"), s.get("inpainting.backup", "telea"))[0]
        label = modelstore.CATALOG[first].title if first in modelstore.CATALOG else f"OpenCV ({first})"
        self.chip_eraser.configure(text=f"🧽 Eraser: {label}",
                                   text_color="#3fb950" if first in modelstore.CATALOG else "#d29922")

    def _health_check(self) -> None:
        """Explain every problem with the current settings (or confirm all is fine)."""
        self._sync_translation()
        issues = preflight(self.settings, self.out_dir)
        if not issues:
            ErrorModal(self, "All good | كل شيء جاهز", "Keys, eraser, fonts and libraries are ready.", error=False)
            return
        text = "\n\n".join(f"{'✘' if i.error else '!'}  {i.text}" for i in issues)
        first = issues[0]
        action = (f"Open {first.tab}", lambda: self.tabs.set(first.tab)) if first.tab else ("Open Setup", self.open_setup)
        ErrorModal(self, "Problems found | مشاكل", text, error=any(i.error for i in issues), action=action)

    def _build_bottom(self) -> None:
        """Bottom strip: progress bar and log."""
        bottom = ctk.CTkFrame(self)
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        bottom.grid_columnconfigure(0, weight=1)
        self.progress = ProgressPanel(bottom)
        self.progress.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        self.log_box = LogBox(bottom, height=110)
        self.log_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 8))
        ctk.CTkButton(bottom, text="Clear log", width=80, height=22, fg_color="gray30",
                      command=self.log_box.clear).grid(row=0, column=1, padx=(0, 8), pady=(6, 2))

    # -- Translation tab -------------------------------------------------
    def _paste_key(self) -> None:
        """Append the clipboard text to the keys box (works with any keyboard layout)."""
        try:
            text = self.clipboard_get().strip()
        except Exception:  # noqa: BLE001 - empty clipboard
            messagebox.showinfo("Clipboard", "The clipboard is empty. Copy the key first.")
            return
        have = self.keys_box.get("1.0", "end").strip()
        self.keys_box.delete("1.0", "end")
        self.keys_box.insert("1.0", (have + "\n" if have else "") + text)

    def _build_translation_tab(self, tab: ctk.CTkFrame) -> None:
        """Gemini keys, main + backup models, memory, picture mode and options."""
        s = self.settings
        section(tab, "Gemini keys", "One per line. Several keys from different Google accounts = more free quota.")
        self.keys_box = ctk.CTkTextbox(tab, height=84, wrap="none")
        self.keys_box.pack(fill="x", pady=2)
        self.keys_box.insert("1.0", "\n".join(parse_keys(s.get("translation.gemini_keys", []))))
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkButton(row, text="📋 Paste", width=80, fg_color="gray30", command=self._paste_key).pack(side="left")
        ctk.CTkButton(row, text="🔑 Get a free key", fg_color="gray30",
                      command=lambda: webbrowser.open(KEY_URL)).pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(row, text="🔎 Test", width=70, command=self._test_translation).pack(side="left")

        section(tab, "Models", "Pick from the list or type a model name. Backups take over when the main model "
                               "is out of quota or does not exist.")
        names = list(dict.fromkeys([*KNOWN_MODELS, s.get("translation.model", DEFAULT_MODEL),
                                    *s.get("translation.fallback_models", [])]))
        backups = [*s.get("translation.fallback_models", list(FALLBACK_MODELS)), "", ""]
        self.model = LabeledCombo(tab, "Main model", names, s.get("translation.model", DEFAULT_MODEL))
        self.model.pack(fill="x", pady=2)
        self.backup1 = LabeledCombo(tab, "Backup model 1", names, backups[0])
        self.backup1.pack(fill="x", pady=2)
        self.backup2 = LabeledCombo(tab, "Backup model 2", names, backups[1])
        self.backup2.pack(fill="x", pady=2)
        ctk.CTkButton(tab, text="↻ Load the models my key can use", fg_color="gray30",
                      command=self._load_models).pack(fill="x", pady=4)

        section(tab, "Consistency", "Helps names and tone stay the same across the pages of one chapter.")
        self.memory = self._switch(tab, "Remember names and story between pages", "translation.memory", True)
        ctk.CTkButton(tab, text="Forget the story (new series)", fg_color="gray30", height=26,
                      command=self._forget_memory).pack(fill="x", pady=2)
        self.vision = self._switch(tab, "Show Gemini the page picture (fixes OCR mistakes, uses more quota)",
                                   "translation.vision", False)
        ctk.CTkLabel(tab, text="Instructions for the translator (tone, names, glossary) — optional",
                     anchor="w", wraplength=330, justify="left").pack(fill="x", pady=(8, 0))
        self.notes_box = ctk.CTkTextbox(tab, height=64, wrap="word")
        self.notes_box.pack(fill="x", pady=2)
        self.notes_box.insert("1.0", s.get("translation.notes", ""))

        section(tab, "Text")
        self.concise = self._switch(tab, "Shorten Arabic to fit the text box", "translation.concise", True)
        self.erase_credits = self._switch(tab, "Erase credit / website texts without translating them",
                                          "translation.erase_credits", True)
        self.arabic_digits = self._switch(tab, "Use Arabic-Indic digits (٠١٢)", "translation.arabic_digits", False)
        self.interval = LabeledSlider(tab, "Seconds between requests", 0, 5,
                                      s.get("translation.min_interval", 1.0), 50, "{:.1f}")
        self.interval.pack(fill="x", pady=4)
        ctk.CTkButton(tab, text="Save", command=lambda: self._sync_translation(log=True)).pack(fill="x", pady=(10, 4))

    def _switch(self, tab: ctk.CTkFrame, text: str, key: str, default: bool) -> ctk.CTkSwitch:
        """A switch initialised from settings."""
        sw = ctk.CTkSwitch(tab, text=text)
        sw.pack(anchor="w", pady=3)
        if self.settings.get(key, default):
            sw.select()
        return sw

    def _forget_memory(self) -> None:
        """Clear the remembered story/names (start of a new series)."""
        self.pipeline.memory.clear()
        self.log("Story memory cleared.")

    def _load_models(self) -> None:
        """Ask Google which models this key can use and offer them in the lists."""
        self._sync_translation()

        def done(names: List[str]) -> None:
            for combo in (self.model, self.backup1, self.backup2):
                combo.set_values(names)
            self.log(f"{len(names)} model(s) available for your key.")

        self._run_task(lambda: create_translator(self.settings).list_models(), done, "Loading models...")

    def _sync_translation(self, log: bool = False) -> None:
        """Copy the translation widgets into settings; rebuild the translator if they changed."""
        s = self.settings
        main = self.model.get() or DEFAULT_MODEL
        backups = [b for b in dict.fromkeys([self.backup1.get(), self.backup2.get()]) if b and b != "(none)" and b != main]
        new = {
            "translation.gemini_keys": parse_keys(self.keys_box.get("1.0", "end")),
            "translation.model": main,
            "translation.fallback_models": backups,
            "translation.memory": bool(self.memory.get()),
            "translation.vision": bool(self.vision.get()),
            "translation.notes": self.notes_box.get("1.0", "end").strip(),
            "translation.min_interval": round(self.interval.get(), 1),
            "translation.concise": bool(self.concise.get()),
            "translation.arabic_digits": bool(self.arabic_digits.get()),
            "translation.erase_credits": bool(self.erase_credits.get()),
        }
        changed = any(s.get(k) != v for k, v in new.items())
        for k, v in new.items():
            s.set(k, v)
        if changed:
            s.save()
            self.pipeline.reload_settings()
            self._refresh_status()
        if log:
            self.log(f"Translation settings saved ({len(new['translation.gemini_keys'])} key(s), model {main}"
                     + (f", backups {', '.join(backups)}" if backups else "") + ").")

    def _test_translation(self) -> None:
        """Test every key in the background and show a readable report."""
        self._sync_translation()

        def work() -> str:
            results = diagnose(self.settings)
            lines = [f"{'✔' if r.ok else '✘'}  {r.label}\n     {r.message}\n" for r in results]
            if not any(r.ok for r in results):
                lines.append("No working key. Get a free one: " + KEY_URL + "\n"
                             "لا يوجد مفتاح يعمل. احصل على مفتاح مجاني من الرابط أعلاه.")
            return "\n".join(lines)

        self._run_task(work, lambda text: ErrorModal(self, "Key test | اختبار المفاتيح", text, error=False),
                       "Testing keys...")

    # -- Fonts & style tab -----------------------------------------------
    def _font_names(self) -> List[str]:
        """File names in ``fonts/`` plus the auto option."""
        return [AUTO_FONT] + [p.name for p in self.pipeline.fonts.available_fonts()]

    def _build_style_tab(self, tab: ctk.CTkFrame) -> None:
        """Role → font mapping and per-role colour/stroke/shadow/opacity."""
        self.role_var = ctk.CTkSegmentedButton(tab, values=[r.capitalize() for r in ROLES],
                                               command=lambda _v: self._load_style_controls())
        self.role_var.set(ROLES[0].capitalize())
        self.role_var.pack(fill="x", pady=(2, 6))
        self.font_option = LabeledOption(tab, "Arabic font for this role", [AUTO_FONT], AUTO_FONT)
        self.font_option.pack(fill="x", pady=2)
        ctk.CTkButton(tab, text="Add font file (.ttf/.otf)…", fg_color="gray30", command=self._add_font).pack(
            fill="x", pady=2)
        self.color_btn = ColorButton(tab, "Fixed text colour (when the switch below is off)")
        self.color_btn.pack(fill="x", pady=2)
        self.match_src = ctk.CTkSwitch(tab, text="Same colour as the original text (recommended)")
        self.match_src.pack(anchor="w", pady=2)
        self.stroke_sw = ctk.CTkSwitch(tab, text="Outline")
        self.stroke_sw.pack(anchor="w", pady=2)
        self.stroke_auto = ctk.CTkSwitch(tab, text="Auto outline colour (contrasts with the text)")
        self.stroke_auto.pack(anchor="w", pady=2)
        self.stroke_color = ColorButton(tab, "Outline colour (when auto is off)", "#FFFFFF")
        self.stroke_color.pack(fill="x", pady=2)
        self.stroke_w = LabeledSlider(tab, "Outline width (px)", 1, 8, 2, 7, "{:.0f}")
        self.stroke_w.pack(fill="x", pady=2)
        self.shadow_sw = ctk.CTkSwitch(tab, text="Shadow")
        self.shadow_sw.pack(anchor="w", pady=2)
        self.opacity = LabeledSlider(tab, "Opacity", 0.1, 1.0, 1.0, 90, "{:.2f}")
        self.opacity.pack(fill="x", pady=2)
        ctk.CTkButton(tab, text="Apply Style & Re-render", command=self._apply_style).pack(fill="x", pady=(10, 2))

    def _current_role(self) -> str:
        """Role currently selected in the style tab."""
        return self.role_var.get().lower()

    def _load_style_controls(self) -> None:
        """Fill the style widgets from settings for the selected role."""
        role = self._current_role()
        st = self.settings.role_style(role)
        names = self._font_names()
        self.font_option.set_values(names)
        mapped = (self.settings.get("fonts.mappings", {}) or {}).get(role, "")
        self.font_option.set(Path(mapped).name if mapped and Path(mapped).name in names else AUTO_FONT)
        self.color_btn.set(st["color"])
        (self.match_src.select if st["match_source_color"] else self.match_src.deselect)()
        (self.stroke_sw.select if st["stroke_enabled"] else self.stroke_sw.deselect)()
        (self.stroke_auto.select if st.get("stroke_auto", True) else self.stroke_auto.deselect)()
        self.stroke_color.set(st["stroke_color"])
        self.stroke_w.set(st["stroke_width"])
        (self.shadow_sw.select if st["shadow_enabled"] else self.shadow_sw.deselect)()
        self.opacity.set(st["opacity"])

    def _apply_style(self) -> None:
        """Store the style widgets in settings and re-render the page."""
        role = self._current_role()
        style = self.settings.role_style(role)
        style.update(
            color=self.color_btn.get(), match_source_color=bool(self.match_src.get()),
            stroke_enabled=bool(self.stroke_sw.get()), stroke_auto=bool(self.stroke_auto.get()),
            stroke_color=self.stroke_color.get(),
            stroke_width=int(self.stroke_w.get()), shadow_enabled=bool(self.shadow_sw.get()),
            opacity=round(self.opacity.get(), 2),
        )
        self.settings.set(f"styles.{role}", style)
        font = self.font_option.get()
        self.settings.set(f"fonts.mappings.{role}", "" if font == AUTO_FONT else f"fonts/{font}")
        self.settings.save()
        self.pipeline.reload_settings()
        self._rerender()
        self.log(f"Style for '{role}' applied.")

    def _add_font(self) -> None:
        """Copy a user-selected font into ``fonts/``."""
        paths = filedialog.askopenfilenames(title="Select Arabic fonts",
                                            filetypes=[("Fonts", "*.ttf *.otf"), ("All", "*.*")])
        for p in paths:
            FONTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, FONTS_DIR / Path(p).name)
            self.log(f"Added font {Path(p).name}")
        if paths:
            self._load_style_controls()

    # -- Blocks tab ------------------------------------------------------
    def _build_blocks_tab(self, tab: ctk.CTkFrame) -> None:
        """List of detected blocks with a translation editor."""
        self.block_list = ctk.CTkScrollableFrame(tab, label_text="Detected blocks", height=220)
        self.block_list.pack(fill="x", pady=(2, 6))
        self.src_label = ctk.CTkLabel(tab, text="Source: —", anchor="w", justify="left", wraplength=300)
        self.src_label.pack(fill="x")
        ctk.CTkLabel(tab, text="Arabic translation (editable)", anchor="w").pack(fill="x", pady=(6, 0))
        self.trans_box = ctk.CTkTextbox(tab, height=110, wrap="word")
        self.trans_box.pack(fill="x", pady=2)
        self.block_role = LabeledOption(tab, "Role", [r.capitalize() for r in ROLES])
        self.block_role.pack(fill="x", pady=2)
        self.block_enabled = ctk.CTkSwitch(tab, text="Block enabled (erase + render)")
        self.block_enabled.select()
        self.block_enabled.pack(anchor="w", pady=4)
        self.block_ignored = ctk.CTkSwitch(tab, text="Credit / watermark (erase only, don't translate)")
        self.block_ignored.pack(anchor="w", pady=4)
        ctk.CTkButton(tab, text="Apply Edit", command=self._apply_block_edit).pack(fill="x", pady=2)
        ctk.CTkButton(tab, text="Re-translate Block", fg_color="gray30", command=self._retranslate_block).pack(
            fill="x", pady=2)

    def _region(self, rid: Optional[int]) -> Optional[TextRegion]:
        """Region with the given id on the current page."""
        page = self.pages.get(self.current) if self.current else None
        if page is None or rid is None:
            return None
        return next((r for r in page.regions if r.id == rid), None)

    def _populate_blocks(self) -> None:
        """Rebuild the block list for the current page."""
        for btn in self._block_buttons.values():
            btn.destroy()
        self._block_buttons.clear()
        page = self.pages.get(self.current) if self.current else None
        if page is None:
            return
        for r in page.regions:
            tag = "credit" if r.ignored else r.role.value
            label = f"#{r.id} [{tag}] {r.text[:28]}"
            btn = ctk.CTkButton(self.block_list, text=label, anchor="w", fg_color="gray25",
                                command=lambda rid=r.id: self._select_region(rid))
            btn.pack(fill="x", pady=1)
            self._block_buttons[r.id] = btn

    def _select_region(self, rid: Optional[int], from_canvas: bool = False) -> None:
        """Select a block in the list/editor (and on the canvas)."""
        self.selected_region = rid
        for bid, btn in self._block_buttons.items():
            btn.configure(fg_color="#1f6aa5" if bid == rid else "gray25")
        r = self._region(rid)
        if r is None:
            self.src_label.configure(text="Source: —")
            self.trans_box.delete("1.0", "end")
            return
        self.src_label.configure(text=f"Source: {r.text}")
        self.trans_box.delete("1.0", "end")
        self.trans_box.insert("1.0", r.translation)
        self.block_role.set(r.role.value.capitalize())
        (self.block_enabled.select if r.enabled else self.block_enabled.deselect)()
        (self.block_ignored.select if r.ignored else self.block_ignored.deselect)()
        if not from_canvas:
            self.canvas.select_box(rid)
        self.tabs.set("Blocks")

    def _on_box_selected(self, rid: Optional[int]) -> None:
        """Canvas -> list selection."""
        self._select_region(rid, from_canvas=True)

    def _on_box_changed(self, rid: int, bbox: BBox) -> None:
        """Canvas box edited: store and re-render."""
        r = self._region(rid)
        if r is None:
            return
        r.target_bbox = tuple(bbox)  # type: ignore[assignment]
        self._rerender()

    def _apply_block_edit(self) -> None:
        """Apply translation/role/enabled edits of the selected block."""
        r = self._region(self.selected_region)
        page = self.pages.get(self.current) if self.current else None
        if r is None or page is None:
            return
        was_enabled = r.enabled
        r.translation = self.trans_box.get("1.0", "end").strip()
        r.role = FontRole(self.block_role.get().lower())
        r.enabled = bool(self.block_enabled.get())
        r.ignored = bool(self.block_ignored.get())
        if r.ignored:
            r.translation = ""
        if was_enabled != r.enabled:
            self._run_task(lambda: self.pipeline.reclean(page), lambda _r: self._after_edit(),
                           "Re-cleaning page...")
        else:
            self._rerender()
        self._populate_blocks()
        self._select_region(r.id, from_canvas=True)

    def _after_edit(self) -> None:
        """Refresh after a background edit."""
        self._refresh_view(True)
        self.progress.update_progress(1.0, "Done")

    def _retranslate_block(self) -> None:
        """Re-translate the selected block with Gemini."""
        r = self._region(self.selected_region)
        page = self.pages.get(self.current) if self.current else None
        if r is None or page is None or not self._preflight():
            return

        def work() -> None:
            self.pipeline.translate_regions([r])

        def done(_res: Any) -> None:
            self._rerender()
            self._select_region(r.id, from_canvas=True)

        self._run_task(work, done, "Translating block...")

    # -- Eraser tab ------------------------------------------------------
    def _build_eraser_tab(self, tab: ctk.CTkFrame) -> None:
        """Choose, install, import or remove the eraser models."""
        section(tab, "Text eraser", "Choose how the original text is removed. Press Install on ONE model - "
                                    "you do not need the others.")
        self.eraser = EraserPanel(tab, self.settings, self._run_task, self._progress_cb, self.log, self._eraser_changed)
        self.eraser.pack(fill="x")

    def _eraser_changed(self) -> None:
        """A different eraser was chosen or its files changed: rebuild it and update the chip."""
        self.pipeline.reload_settings()
        self._refresh_status()

    # -- General tab -----------------------------------------------------
    def _build_general_tab(self, tab: ctk.CTkFrame) -> None:
        """Reading order, output, theme and the advanced engine settings."""
        s = self.settings
        self.reading = LabeledOption(tab, "Reading order", ["rtl", "ltr"], s.get("general.reading_order"))
        self.reading.pack(fill="x", pady=4)
        self.out_format = LabeledOption(tab, "Export format", ["png", "jpg", "webp"], s.get("general.output_format"))
        self.out_format.pack(fill="x", pady=4)
        self.jpg_quality = LabeledSlider(tab, "JPG / WebP quality", 60, 100, s.get("general.jpg_quality", 95), 40, "{:.0f}")
        self.jpg_quality.pack(fill="x", pady=4)
        self.save_json = self._switch(tab, "Save project JSON on export", "general.save_project_json", True)
        self.theme = LabeledOption(tab, "Theme", ["dark", "light", "system"], s.get("general.theme"),
                                   command=lambda v: ctk.set_appearance_mode(v))
        self.theme.pack(fill="x", pady=4)

        section(tab, "Advanced", "Defaults are fine for most pages.")
        self.ocr_conf = LabeledSlider(tab, "Text detection sensitivity (lower finds more)", 0.1, 0.8,
                                      s.get("ocr.min_confidence", 0.25), 70, "{:.2f}")
        self.ocr_conf.pack(fill="x", pady=4)
        self.ocr_workers = LabeledOption(tab, "Text reading threads", ["auto", "1", "2", "3", "4", "6"],
                                         "auto" if not s.get("ocr.workers", 0) else str(s.get("ocr.workers")))
        self.ocr_workers.pack(fill="x", pady=4)
        self.mask_grow = LabeledSlider(tab, "Extra margin around erased letters (px)", 0, 8,
                                       s.get("detector.mask_dilate", 3), 8, "{:.0f}")
        self.mask_grow.pack(fill="x", pady=4)
        self.retries = LabeledSlider(tab, "Retries when Gemini fails or answers badly", 1, 6,
                                     s.get("translation.retries", 3), 5, "{:.0f}")
        self.retries.pack(fill="x", pady=4)
        self.timeout = LabeledSlider(tab, "Wait for Gemini (seconds)", 30, 180, s.get("translation.timeout", 90), 15, "{:.0f}")
        self.timeout.pack(fill="x", pady=4)
        self.on_failure = LabeledOption(tab, "If translation fails", list(FAILURE_CHOICES.values()),
                                        FAILURE_CHOICES.get(s.get("translation.on_failure", "raise"), FAILURE_CHOICES["raise"]))
        self.on_failure.pack(fill="x", pady=4)
        ctk.CTkButton(tab, text="Save", command=self._save_general).pack(fill="x", pady=(12, 4))
        ctk.CTkButton(tab, text="Restore default settings", fg_color="gray30", command=self._reset_settings).pack(
            fill="x", pady=2)

    def _save_general(self) -> None:
        """Persist general and advanced settings and rebuild the engines."""
        s = self.settings
        failure = {v: k for k, v in FAILURE_CHOICES.items()}
        for key, value in {
            "general.reading_order": self.reading.get(),
            "general.output_format": self.out_format.get(),
            "general.jpg_quality": int(self.jpg_quality.get()),
            "general.save_project_json": bool(self.save_json.get()),
            "general.theme": self.theme.get(),
            "ocr.min_confidence": round(self.ocr_conf.get(), 2),
            "ocr.workers": 0 if self.ocr_workers.get() == "auto" else int(self.ocr_workers.get()),
            "detector.mask_dilate": int(self.mask_grow.get()),
            "translation.retries": int(self.retries.get()),
            "translation.timeout": int(self.timeout.get()),
            "translation.on_failure": failure.get(self.on_failure.get(), "raise"),
        }.items():
            s.set(key, value)
        s.save()
        self.pipeline.reload_settings(full=True)
        self.log("Settings saved.")

    def _reset_settings(self) -> None:
        """Back to the defaults (keeps the Gemini keys); the window is rebuilt on the next start."""
        if not messagebox.askyesno("Restore defaults", "Restore all settings to their defaults?\n"
                                                       "Your Gemini keys are kept. Restart the app afterwards."):
            return
        keys = self.settings.get("translation.gemini_keys", [])
        fresh = Settings()
        fresh.set("translation.gemini_keys", keys)
        fresh.set("general.first_run_done", True)
        fresh.save()
        self.settings.data = fresh.data
        self.log("Defaults restored - restart the app to see them in the controls.")

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------
    def _add_images(self) -> None:
        """Add individual image files."""
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS))
        paths = filedialog.askopenfilenames(title="Select pages", filetypes=[("Images", exts), ("All", "*.*")])
        self._add_files([Path(p) for p in paths])

    def _add_folder(self) -> None:
        """Add every image in a folder."""
        folder = filedialog.askdirectory(title="Select chapter folder")
        if folder:
            self._add_files(list_images(folder))

    def _add_files(self, paths: List[Path]) -> None:
        """Register files and show the first new one."""
        new = [p for p in paths if p not in self.files]
        for p in new:
            self.files.append(p)
            btn = ctk.CTkButton(self.file_list, text=p.name, anchor="w", fg_color="gray25",
                                command=lambda path=p: self._select_file(path))
            btn.pack(fill="x", pady=1)
            self._file_buttons[p] = btn
        if new:
            self.log(f"Added {len(new)} file(s).")
            self._select_file(new[0])

    def _clear_files(self) -> None:
        """Remove all files from the session."""
        for btn in self._file_buttons.values():
            btn.destroy()
        self._file_buttons.clear()
        self.files.clear()
        self.pages.clear()
        self.current = None
        self.canvas.set_layers([])
        self._populate_blocks()

    def _select_file(self, path: Path) -> None:
        """Make ``path`` the current page."""
        self.current = path
        for p, btn in self._file_buttons.items():
            btn.configure(fg_color="#1f6aa5" if p == path else "gray25")
        self._populate_blocks()
        self._select_region(None)
        self._refresh_view()

    def _open_project(self) -> None:
        """Load a saved project JSON chosen by the user."""
        p = filedialog.askopenfilename(title="Open project", filetypes=[("Project", "*.json")])
        if p:
            self._load_project_file(Path(p))

    def _load_project_file(self, path: Path) -> None:
        """Load ``path`` (project JSON) and show it."""
        try:
            page = self.pipeline.load_project(path)
        except Exception as exc:  # noqa: BLE001
            self._fail("Cannot open project", str(exc))
            return
        if path not in self.files:
            self._add_files([path])
        self.pages[path] = page
        self._select_file(path)

    # ------------------------------------------------------------------
    # Viewing
    # ------------------------------------------------------------------
    def _original_image(self) -> Optional[Any]:
        """Original image of the current file."""
        if self.current is None:
            return None
        page = self.pages.get(self.current)
        if page is not None:
            return page.original
        if self.current.suffix.lower() in IMAGE_EXTENSIONS:
            return load_image(self.current)
        return None

    def _refresh_view(self, keep_view: bool = False) -> None:
        """Redraw the canvas for the current page and view mode."""
        if self.current is None:
            return
        page = self.pages.get(self.current)
        orig = self._original_image()
        if orig is None:
            return
        mode = self.view_mode.get()
        if page is None or mode == "Original":
            self.canvas.set_layers([orig], keep_view)
            self.canvas.set_boxes({}, False)
            return
        if mode == "Processed":
            self.canvas.set_layers([page.result], keep_view)
        else:
            self.canvas.set_layers([page.original, page.result], keep_view)
        boxes = {r.id: r.fit_bbox for r in page.regions if r.enabled}
        self.canvas.set_boxes(boxes, bool(self.show_boxes.get()))
        self.canvas.select_box(self.selected_region)

    def _rerender(self) -> None:
        """Re-typeset the current page and refresh the view."""
        page = self.pages.get(self.current) if self.current else None
        if page is None:
            return
        self.pipeline.render(page)
        self._refresh_view(True)

    def _set_page(self, path: Path, page: PageResult) -> None:
        """Store a processed page and show it."""
        self.pages[path] = page
        if path == self.current:
            self._populate_blocks()
            self._refresh_view()
        self.progress.update_progress(1.0, f"Done: {path.name}")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def _preflight(self, out_dir: Optional[Path] = None) -> bool:
        """Check keys, eraser, fonts and folders before working; explain problems and where to fix them.

        Args:
            out_dir: Output folder that must be writable (batch runs only).

        Returns:
            ``False`` when an error blocks the run (a dialog with a shortcut was shown).
        """
        self._sync_translation()
        issues = preflight(self.settings, out_dir)
        for i in issues:
            if not i.error:
                self.log("Warning: " + i.text)
        errors = [i for i in issues if i.error]
        if not errors:
            return True
        first = errors[0]
        if first.tab:
            self.tabs.set(first.tab)
        ErrorModal(self, "Fix this first | أصلح هذا أولًا", "\n\n".join(f"✘  {i.text}" for i in errors),
                   action=(f"Open {first.tab}", lambda: self.tabs.set(first.tab)) if first.tab else ("Open Setup", self.open_setup))
        return False

    def _process_current(self) -> None:
        """Process the selected page."""
        if self.current is None or self.current.suffix.lower() not in IMAGE_EXTENSIONS:
            messagebox.showinfo("No page", "Select an image first.")
            return
        path = self.current
        if not self._preflight():
            return

        def work() -> PageResult:
            return self.pipeline.process_file(path, self._progress_cb)

        self._run_task(work, lambda page: self._set_page(path, page), "Processing...")

    def _default_out_dir(self) -> Path:
        """Output folder: the chosen one, else ``translated`` next to the first page."""
        if self.out_dir is not None:
            return self.out_dir
        base = (self.files[0].parent if self.files else Path.cwd())
        return base / "translated"

    def _choose_out_dir(self) -> None:
        """Let the user pick another output folder."""
        out = filedialog.askdirectory(title="Output folder")
        if out:
            self.out_dir = Path(out)
            self.log(f"Output folder: {self.out_dir}")

    def _open_output(self) -> None:
        """Open the output folder in the file manager."""
        open_folder(self._default_out_dir())

    def _process_all(self) -> None:
        """Translate every page and save the results in the output folder."""
        images = [p for p in self.files if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            messagebox.showinfo("No pages", "Drop images or a folder into the window first.")
            return
        out_dir = self._default_out_dir()
        if not self._preflight(out_dir):
            return
        self.log(f"Results will be saved in: {out_dir}")

        def on_page(path: Path, page: PageResult) -> None:
            self._post(lambda: self._set_page(path, page))

        def work() -> List[Path]:
            return self.pipeline.process_batch(images, out_dir, self._progress_cb, self._cancel, on_page)

        self._run_task(work, lambda outs: self.log(f"Finished: {len(outs)} page(s) saved in {out_dir}"),
                       "Translating all pages...")

    def _ensure_out_dir(self) -> Optional[Path]:
        """Output folder (created on demand)."""
        out = self._default_out_dir()
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _export_current(self) -> None:
        """Export the current page."""
        page = self.pages.get(self.current) if self.current else None
        out_dir = self._ensure_out_dir()
        if page is None or out_dir is None:
            return
        path = self.pipeline.export_page(page, out_dir, self.current.stem if self.current else None)
        self.log(f"Exported {path}")

    def _save_project(self) -> None:
        """Save regions JSON and clean background for the current page."""
        page = self.pages.get(self.current) if self.current else None
        out_dir = self._ensure_out_dir()
        if page is None or out_dir is None or self.current is None:
            return
        path = self.pipeline.save_project(page, out_dir, self.current.stem)
        self.log(f"Project saved: {path}")


def run_gui() -> None:
    """Create and run the application."""
    app = ComicTranslatorApp()
    app.mainloop()
