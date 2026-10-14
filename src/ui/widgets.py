"""Reusable CustomTkinter controls and dialogs."""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import colorchooser
from typing import Any, Callable, List, Optional, Sequence, Tuple

import customtkinter as ctk

from src.utils.image_utils import rgb_to_hex


class ErrorModal(ctk.CTkToplevel):
    """Modal error dialog (e.g. when every translation provider failed)."""

    def __init__(self, master: tk.Misc, title: str, message: str, error: bool = True,
                 action: Optional[Tuple[str, Callable[[], None]]] = None) -> None:
        """Show the dialog.

        Args:
            master: Parent window.
            title: Window title.
            message: Text to show.
            error: Red title for errors, neutral for information.
            action: Optional ``(button text, callback)`` shown next to OK.
        """
        super().__init__(master)
        self.title(title)
        self.geometry("620x380")
        self.resizable(False, False)
        self.transient(master)  # type: ignore[arg-type]
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#ff6b6b" if error else ("gray10", "gray90")).pack(pady=(16, 6), padx=16, anchor="w")
        box = ctk.CTkTextbox(self, wrap="word")
        box.pack(fill="both", expand=True, padx=16, pady=6)
        box.insert("1.0", message)
        box.configure(state="disabled")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=12)
        if action:
            ctk.CTkButton(row, text=action[0], width=140,
                          command=lambda: (self.destroy(), action[1]())).pack(side="left", padx=6)
        ctk.CTkButton(row, text="OK", width=100, fg_color="gray30", command=self.destroy).pack(side="left", padx=6)
        self.after(150, self._grab)

    def _grab(self) -> None:
        """Grab focus once the window is mapped."""
        try:
            self.grab_set()
            self.focus_force()
        except tk.TclError:
            pass


class LabeledEntry(ctk.CTkFrame):
    """Label + entry pair."""

    def __init__(self, master: tk.Misc, label: str, value: str = "", show: str = "", width: int = 200) -> None:
        """Create the control.

        Args:
            master: Parent.
            label: Caption.
            value: Initial text.
            show: Mask character (for passwords).
            width: Entry width.
        """
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, anchor="w").pack(fill="x")
        self.entry = ctk.CTkEntry(self, show=show, width=width)
        self.entry.pack(fill="x")
        self.set(value)

    def get(self) -> str:
        """Current text."""
        return self.entry.get()

    def set(self, value: str) -> None:
        """Replace the text."""
        self.entry.delete(0, "end")
        self.entry.insert(0, value)


class LabeledOption(ctk.CTkFrame):
    """Label + drop-down pair."""

    def __init__(self, master: tk.Misc, label: str, values: Sequence[str], value: Optional[str] = None,
                 command: Optional[Callable[[str], None]] = None) -> None:
        """Create the control.

        Args:
            master: Parent.
            label: Caption.
            values: Choices.
            value: Initially selected choice.
            command: Called with the new value.
        """
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, anchor="w").pack(fill="x")
        self.var = ctk.StringVar(value=value if value is not None else (values[0] if values else ""))
        self.menu = ctk.CTkOptionMenu(self, values=list(values) or [""], variable=self.var, command=command)
        self.menu.pack(fill="x")

    def get(self) -> str:
        """Selected value."""
        return self.var.get()

    def set(self, value: str) -> None:
        """Select a value."""
        self.var.set(value)

    def set_values(self, values: Sequence[str]) -> None:
        """Replace the choices."""
        self.menu.configure(values=list(values) or [""])


class LabeledSlider(ctk.CTkFrame):
    """Label + slider with a live value read-out."""

    def __init__(self, master: tk.Misc, label: str, lo: float, hi: float, value: float,
                 steps: Optional[int] = None, fmt: str = "{:.2f}") -> None:
        """Create the control.

        Args:
            master: Parent.
            label: Caption.
            lo: Minimum.
            hi: Maximum.
            value: Initial value.
            steps: Number of discrete steps.
            fmt: Format string for the read-out.
        """
        super().__init__(master, fg_color="transparent")
        self._fmt = fmt
        self._caption = label
        self.label = ctk.CTkLabel(self, text="", anchor="w")
        self.label.pack(fill="x")
        self.slider = ctk.CTkSlider(self, from_=lo, to=hi, number_of_steps=steps, command=self._update)
        self.slider.pack(fill="x")
        self.set(value)

    def _update(self, value: float) -> None:
        """Refresh the read-out."""
        self.label.configure(text=f"{self._caption}: {self._fmt.format(value)}")

    def get(self) -> float:
        """Current value."""
        return float(self.slider.get())

    def set(self, value: float) -> None:
        """Set the value."""
        self.slider.set(value)
        self._update(value)


class ColorButton(ctk.CTkFrame):
    """Button showing a colour swatch and opening the system colour chooser."""

    def __init__(self, master: tk.Misc, label: str, value: str = "#000000",
                 command: Optional[Callable[[str], None]] = None) -> None:
        """Create the control.

        Args:
            master: Parent.
            label: Caption.
            value: Initial ``#rrggbb``.
            command: Called with the new colour.
        """
        super().__init__(master, fg_color="transparent")
        self._command = command
        self.value = value
        ctk.CTkLabel(self, text=label, anchor="w").pack(fill="x")
        self.button = ctk.CTkButton(self, text=value, command=self._pick, fg_color=value, text_color="#808080")
        self.button.pack(fill="x")
        self.set(value)

    def _pick(self) -> None:
        """Open the colour chooser."""
        rgb, hex_value = colorchooser.askcolor(color=self.value, parent=self.winfo_toplevel())
        if hex_value:
            self.set(rgb_to_hex(rgb))
            if self._command:
                self._command(self.value)

    def get(self) -> str:
        """Current ``#rrggbb``."""
        return self.value

    def set(self, value: str) -> None:
        """Set the colour."""
        self.value = value
        lum = int(value[1:3], 16) * 0.299 + int(value[3:5], 16) * 0.587 + int(value[5:7], 16) * 0.114
        self.button.configure(text=value, fg_color=value, hover_color=value,
                              text_color="#000000" if lum > 140 else "#ffffff")


class LogBox(ctk.CTkTextbox):
    """Read-only log view fed from any thread through a queue."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        """Create the log box and start polling its queue."""
        super().__init__(master, state="disabled", wrap="word", **kwargs)
        self._q: "queue.Queue[str]" = queue.Queue()
        self.after(100, self._drain)

    def write(self, message: str) -> None:
        """Thread-safe append of a line."""
        self._q.put(message)

    def _drain(self) -> None:
        """Move queued lines into the widget."""
        lines: List[str] = []
        try:
            while True:
                lines.append(self._q.get_nowait())
        except queue.Empty:
            pass
        if lines:
            self.configure(state="normal")
            self.insert("end", "\n".join(lines) + "\n")
            self.see("end")
            self.configure(state="disabled")
        self.after(100, self._drain)


class ProgressPanel(ctk.CTkFrame):
    """Progress bar with a status label."""

    def __init__(self, master: tk.Misc) -> None:
        """Create the panel."""
        super().__init__(master, fg_color="transparent")
        self.label = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.label.pack(fill="x")
        self.bar = ctk.CTkProgressBar(self)
        self.bar.pack(fill="x")
        self.bar.set(0)

    def update_progress(self, fraction: float, text: str) -> None:
        """Set the bar fraction and status text."""
        self.bar.set(max(0.0, min(1.0, fraction)))
        self.label.configure(text=text)
