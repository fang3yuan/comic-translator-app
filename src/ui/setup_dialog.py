"""Setup window: environment checklist, one-click install and translation test."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from src.config import Settings
from src.translators import diagnose
from src.ui.widgets import LogBox, ProgressPanel
from src.utils.installer import Component, check_environment, install_component, install_missing


class SetupDialog(ctk.CTkToplevel):
    """Shows what is installed and installs the rest with one button."""

    def __init__(
        self,
        master: tk.Misc,
        settings: Settings,
        on_change: Optional[Callable[[], None]] = None,
        autostart: Optional[str] = None,
    ) -> None:
        """Create the dialog.

        Args:
            master: Parent window.
            settings: Application settings.
            on_change: Called (on the UI thread) after something was installed.
            autostart: ``"all"`` to install everything missing right away, or a component key.
        """
        super().__init__(master)
        self.settings = settings
        self.on_change = on_change
        self.title("Setup - install everything | التثبيت")
        self.geometry("760x680")
        self.minsize(640, 560)
        self.transient(master)  # type: ignore[arg-type]
        self._busy = False
        self._events: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._row_widgets: List[ctk.CTkBaseClass] = []

        ctk.CTkLabel(
            self, text="Press the big button once - everything needed is installed automatically.\n"
                       "اضغط الزر الكبير مرة واحدة وسيُثبَّت كل ما يلزم تلقائيًا.",
            justify="left", font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 6))

        self.rows = ctk.CTkFrame(self)
        self.rows.pack(fill="x", padx=16, pady=6)

        self.big = ctk.CTkButton(self, text="⬇  Install everything needed  |  تثبيت كل شيء", height=48,
                                 font=ctk.CTkFont(size=16, weight="bold"), command=lambda: self._start(None))
        self.big.pack(fill="x", padx=16, pady=(8, 4))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=16)
        ctk.CTkButton(bar, text="Test translation | اختبر الترجمة", fg_color="gray30",
                      command=self._test_translation).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Re-check | إعادة الفحص", fg_color="gray30", command=self.refresh).pack(side="left")
        ctk.CTkButton(bar, text="Close", width=80, command=self.destroy).pack(side="right")

        self.progress = ProgressPanel(self)
        self.progress.pack(fill="x", padx=16, pady=(10, 2))
        self.log_box = LogBox(self)
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(2, 14))

        self.refresh()
        self.after(80, self._poll)
        if autostart:
            self.after(400, lambda: self._start(None if autostart == "all" else [autostart]))

    # -- thread plumbing -------------------------------------------------
    def _poll(self) -> None:
        """Run callbacks queued by the worker thread."""
        try:
            while True:
                self._events.get_nowait()()
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(80, self._poll)

    def _post(self, fn: Callable[[], None]) -> None:
        self._events.put(fn)

    def _progress(self, fraction: float, text: str) -> None:
        self._post(lambda: self.progress.update_progress(fraction, text))

    # -- checklist ---------------------------------------------------------
    def refresh(self) -> None:
        """Re-run the environment check and rebuild the rows."""
        for w in self._row_widgets:
            w.destroy()
        self._row_widgets.clear()
        comps = check_environment(self.settings)
        self._components: Dict[str, Component] = {c.key: c for c in comps}
        for i, c in enumerate(comps):
            icon, color = ("✔", "#3fb950") if c.ok else (("✘", "#ff6b6b") if c.required else ("○", "#d29922"))
            lbl_icon = ctk.CTkLabel(self.rows, text=icon, text_color=color, width=24, font=ctk.CTkFont(size=16, weight="bold"))
            lbl_icon.grid(row=i, column=0, padx=(10, 4), pady=3)
            lbl = ctk.CTkLabel(self.rows, text=f"{c.title}\n{c.detail}" + (f"   [{c.size}]" if c.size and not c.ok else ""),
                               anchor="w", justify="left")
            lbl.grid(row=i, column=1, sticky="w", padx=4, pady=3)
            self._row_widgets += [lbl_icon, lbl]
            if not c.ok and c.key != "fonts":
                btn = ctk.CTkButton(self.rows, text="Install", width=80, command=lambda k=c.key: self._start([k]))
                btn.grid(row=i, column=2, padx=10, pady=3)
                self._row_widgets.append(btn)
        self.rows.grid_columnconfigure(1, weight=1)
        missing = [c for c in comps if not c.ok and c.key != "fonts"]
        self.big.configure(text="⬇  Install everything needed  |  تثبيت كل شيء" if missing
                           else "✔  Everything is installed  |  كل شيء مثبّت")

    # -- actions -------------------------------------------------------------
    def _start(self, only: Optional[List[str]]) -> None:
        """Install in a worker thread."""
        if self._busy:
            return
        self._busy = True
        self.big.configure(state="disabled")
        self.progress.update_progress(0.0, "Starting...")

        def work() -> None:
            log = self.log_box.write
            ok = True
            try:
                if only and len(only) == 1:
                    comp = self._components.get(only[0])
                    if comp is not None:
                        ok = install_component(comp, self.settings, log, self._progress)
                else:
                    ok = install_missing(self.settings, log, self._progress, only=only)
            except Exception as exc:  # noqa: BLE001
                log(f"Unexpected error: {exc}")
                ok = False
            self._post(lambda: self._finished(ok))

        threading.Thread(target=work, daemon=True).start()

    def _finished(self, ok: bool) -> None:
        """Re-enable controls and notify the main window."""
        self._busy = False
        self.big.configure(state="normal")
        self.progress.update_progress(1.0, "Finished ✔ - drag & drop needs an app restart"
                                      if ok else "Some steps failed - see the log below")
        self.refresh()
        if self.on_change:
            self.on_change()

    def _test_translation(self) -> None:
        """Test every translation provider and print the explained result."""
        if self._busy:
            return
        self._busy = True
        self.progress.update_progress(0.0, "Testing translation providers...")

        def work() -> None:
            self.log_box.write("== Translation test ==")
            for res in diagnose(self.settings):
                self.log_box.write(f"{'✔' if res.ok else '✘'} {res.name}: {res.message}")
            self.log_box.write("If Google and MyMemory both fail: check internet/VPN/firewall.")
            self._post(lambda: (setattr(self, "_busy", False), self.progress.update_progress(1.0, "Test finished")))

        threading.Thread(target=work, daemon=True).start()
