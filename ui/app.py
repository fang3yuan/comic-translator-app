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
from src.translators import DEFAULT_MODEL, KEY_URL, TranslationError, diagnose, parse_keys
from src.ui.canvas import ImageCanvas
from src.ui.setup_dialog import SetupDialog
from src.ui.widgets import (
    install_clipboard_fix,
    ColorButton,
    ErrorModal,
    LabeledEntry,
    LabeledOption,
    LabeledSlider,
    LogBox,
    ProgressPanel,
)
from src.utils.image_utils import IMAGE_EXTENSIONS, list_images, load_image
from src.utils.installer import missing_required, open_folder

try:  # optional: drag & drop support
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except Exception:  # noqa: BLE001 - missing package or Tk without tkdnd binaries
    HAS_DND = False

VIEW_MODES = ["Original", "Processed", "Side-by-side"]
AUTO_FONT = "(auto)"


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
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_center()
        self._build_right_panel()
        self._build_bottom()

        self.pipeline = ComicPipeline(self.settings, self.log)
        self._load_style_controls()
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
        if "api key" in low or "keys were rejected" in low:
            action = ("Open Translation tab", lambda: self.tabs.set("Translation"))
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
        side.grid(row=0, column=0, sticky="nsw", padx=(8, 4), pady=8)
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
        center.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
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
        tabs = ctk.CTkTabview(self, width=340)
        tabs.grid(row=0, column=2, sticky="nse", padx=(4, 8), pady=8)
        self.tabs = tabs
        for name in ("Translation", "Fonts & Style", "Blocks", "General"):
            tabs.add(name)
        self._build_translation_tab(tabs.tab("Translation"))
        self._build_style_tab(tabs.tab("Fonts & Style"))
        self._build_blocks_tab(tabs.tab("Blocks"))
        self._build_general_tab(tabs.tab("General"))

    def _build_bottom(self) -> None:
        """Bottom strip: progress bar and log."""
        bottom = ctk.CTkFrame(self)
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        bottom.grid_columnconfigure(0, weight=1)
        self.progress = ProgressPanel(bottom)
        self.progress.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        self.log_box = LogBox(bottom, height=110)
        self.log_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 8))

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
        """Gemini keys, model, custom instructions and options."""
        s = self.settings
        ctk.CTkLabel(tab, text="Gemini API keys (one per line)  |  مفاتيح Gemini (مفتاح في كل سطر)",
                     anchor="w", wraplength=320, justify="left").pack(fill="x", pady=(2, 0))
        self.keys_box = ctk.CTkTextbox(tab, height=84, wrap="none")
        self.keys_box.pack(fill="x", pady=2)
        self.keys_box.insert("1.0", "\n".join(parse_keys(s.get("translation.gemini_keys", []))))
        ctk.CTkButton(tab, text="📋 Paste key from clipboard", fg_color="gray30",
                      command=self._paste_key).pack(fill="x", pady=2)
        ctk.CTkButton(tab, text="🔑 Get a free key (opens Google AI Studio)", fg_color="gray30",
                      command=lambda: webbrowser.open(KEY_URL)).pack(fill="x", pady=2)
        self.model = LabeledEntry(tab, "Model", s.get("translation.model", DEFAULT_MODEL), width=240)
        self.model.pack(fill="x", pady=4)
        ctk.CTkLabel(tab, text="Instructions for the translator (tone, names, glossary) — optional",
                     anchor="w", wraplength=320, justify="left").pack(fill="x", pady=(4, 0))
        self.notes_box = ctk.CTkTextbox(tab, height=64, wrap="word")
        self.notes_box.pack(fill="x", pady=2)
        self.notes_box.insert("1.0", s.get("translation.notes", ""))
        self.interval = LabeledSlider(tab, "Seconds between requests", 0, 5,
                                      s.get("translation.min_interval", 1.0), 50, "{:.1f}")
        self.interval.pack(fill="x", pady=4)
        self.concise = ctk.CTkSwitch(tab, text="Shorten Arabic to fit the text box")
        self.concise.pack(anchor="w", pady=3)
        if s.get("translation.concise", True):
            self.concise.select()
        self.erase_credits = ctk.CTkSwitch(tab, text="Erase credit / website texts without translating them")
        self.erase_credits.pack(anchor="w", pady=3)
        if s.get("translation.erase_credits", True):
            self.erase_credits.select()
        self.arabic_digits = ctk.CTkSwitch(tab, text="Use Arabic-Indic digits (٠١٢)")
        self.arabic_digits.pack(anchor="w", pady=3)
        if s.get("translation.arabic_digits", False):
            self.arabic_digits.select()
        ctk.CTkButton(tab, text="Save settings", command=lambda: self._sync_translation(log=True)).pack(
            fill="x", pady=(10, 4))
        ctk.CTkButton(tab, text="🔎 Test my keys", fg_color="gray30",
                      command=self._test_translation).pack(fill="x", pady=4)

    def _sync_translation(self, log: bool = False) -> None:
        """Copy the translation widgets into settings; rebuild the translator if they changed."""
        s = self.settings
        new = {
            "translation.gemini_keys": parse_keys(self.keys_box.get("1.0", "end")),
            "translation.model": self.model.get().strip() or DEFAULT_MODEL,
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
        if log:
            n = len(new["translation.gemini_keys"])
            self.log(f"Translation settings saved ({n} key(s), model {new['translation.model']}).")

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
        if r is None or page is None or not self._require_key():
            return

        def work() -> None:
            self.pipeline.translate_regions([r])

        def done(_res: Any) -> None:
            self._rerender()
            self._select_region(r.id, from_canvas=True)

        self._run_task(work, done, "Translating block...")

    # -- General tab -----------------------------------------------------
    def _build_general_tab(self, tab: ctk.CTkFrame) -> None:
        """Reading order, engines, output and theme."""
        s = self.settings
        self.reading = LabeledOption(tab, "Reading order", ["rtl", "ltr"], s.get("general.reading_order"))
        self.reading.pack(fill="x", pady=4)
        self.inpaint_backend = LabeledOption(tab, "Inpainting", ["auto", "lama", "telea", "ns"],
                                             s.get("inpainting.backend"))
        self.inpaint_backend.pack(fill="x", pady=4)
        self.out_format = LabeledOption(tab, "Export format", ["png", "jpg", "webp"], s.get("general.output_format"))
        self.out_format.pack(fill="x", pady=4)
        self.save_json = ctk.CTkSwitch(tab, text="Save project JSON on export")
        self.save_json.pack(anchor="w", pady=4)
        if s.get("general.save_project_json", True):
            self.save_json.select()
        self.theme = LabeledOption(tab, "Theme", ["dark", "light", "system"], s.get("general.theme"),
                                   command=lambda v: ctk.set_appearance_mode(v))
        self.theme.pack(fill="x", pady=4)
        ctk.CTkButton(tab, text="Save General Settings", command=self._save_general).pack(fill="x", pady=(12, 4))

    def _save_general(self) -> None:
        """Persist general settings and rebuild engines."""
        s = self.settings
        s.set("general.reading_order", self.reading.get())
        s.set("inpainting.backend", self.inpaint_backend.get())
        s.set("general.output_format", self.out_format.get())
        s.set("general.save_project_json", bool(self.save_json.get()))
        s.set("general.theme", self.theme.get())
        s.save()
        self.pipeline.reload_settings(full=True)
        self.log("General settings saved.")

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
    def _require_key(self) -> bool:
        """Make sure at least one Gemini key is set; otherwise guide the user to the right tab."""
        self._sync_translation()
        if parse_keys(self.settings.get("translation.gemini_keys", [])):
            return True
        self.tabs.set("Translation")
        messagebox.showinfo(
            "Gemini key needed | مفتاح Gemini مطلوب",
            "Paste a free Gemini API key in the Translation tab first.\nPress '🔑 Get a free key' to create one in 1 minute.\n\n"
            "الصق مفتاح Gemini المجاني في تبويب Translation أولًا.\nاضغط زر 🔑 لإنشاء مفتاح خلال دقيقة.")
        return False

    def _process_current(self) -> None:
        """Process the selected page."""
        if self.current is None or self.current.suffix.lower() not in IMAGE_EXTENSIONS:
            messagebox.showinfo("No page", "Select an image first.")
            return
        path = self.current
        if not self._require_key():
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
        if not self._require_key():
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
