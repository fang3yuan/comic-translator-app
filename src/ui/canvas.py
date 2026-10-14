"""Interactive image viewer: zoom, pan, side-by-side view and box editing."""

from __future__ import annotations

import sys
import tkinter as tk
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageTk

from src.core.models import BBox

HANDLE = 6          # half-size of resize handles (canvas px)
GAP = 12            # gap between layers in side-by-side mode (image px)
COLOR_BOX = "#3da9fc"
COLOR_SELECTED = "#ff8906"


class ImageCanvas(tk.Canvas):
    """Canvas showing one or two images with zoom/pan and editable boxes.

    Controls: mouse wheel zooms around the cursor; dragging empty space (or the
    middle/right button anywhere) pans; dragging a box moves it; dragging a
    corner handle of the selected box resizes it.

    Boxes are always drawn on the *last* layer (the processed page).
    """

    def __init__(
        self,
        master: tk.Misc,
        on_box_changed: Optional[Callable[[int, BBox], None]] = None,
        on_box_selected: Optional[Callable[[Optional[int]], None]] = None,
        **kwargs: object,
    ) -> None:
        """Create the canvas.

        Args:
            master: Parent widget.
            on_box_changed: Called as ``(box_id, new_bbox)`` after an edit.
            on_box_selected: Called with the selected box id (or ``None``).
            **kwargs: Extra ``tk.Canvas`` options.
        """
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("bg", "#1e1e1e")
        super().__init__(master, **kwargs)
        self.on_box_changed = on_box_changed
        self.on_box_selected = on_box_selected

        self._layers: List[Image.Image] = []
        self._composite: Optional[Image.Image] = None
        self._box_offset_x = 0
        self._boxes: Dict[int, BBox] = {}
        self._show_boxes = True
        self._selected: Optional[int] = None
        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._photo: Optional[ImageTk.PhotoImage] = None
        self.placeholder = ""
        self._drag: Optional[Tuple[str, float, float, Optional[BBox]]] = None

        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        for seq in ("<ButtonPress-2>", "<ButtonPress-3>"):
            self.bind(seq, self._on_pan_start)
        for seq in ("<B2-Motion>", "<B3-Motion>"):
            self.bind(seq, self._on_pan_move)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.15))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.15))

    # -- public API ------------------------------------------------------
    def set_layers(self, layers: Sequence[Image.Image], keep_view: bool = False) -> None:
        """Show one image, or several side by side.

        Args:
            layers: Images to display left to right.
            keep_view: Preserve zoom/pan if the composite size is unchanged.
        """
        layers = [im.convert("RGB") for im in layers]
        old_size = self._composite.size if self._composite else None
        self._layers = list(layers)
        if not layers:
            self._composite = None
            self._redraw()
            return
        if len(layers) == 1:
            self._composite = layers[0]
            self._box_offset_x = 0
        else:
            total_w = sum(im.width for im in layers) + GAP * (len(layers) - 1)
            comp = Image.new("RGB", (total_w, max(im.height for im in layers)), (30, 30, 30))
            x = 0
            for im in layers:
                comp.paste(im, (x, 0))
                self._box_offset_x = x
                x += im.width + GAP
            self._composite = comp
        if keep_view and old_size == self._composite.size:
            self._redraw()
        else:
            self.fit_to_window()

    def set_boxes(self, boxes: Dict[int, BBox], show: bool = True) -> None:
        """Set editable boxes (image coordinates of the processed page).

        Args:
            boxes: Mapping ``id -> (x1, y1, x2, y2)``.
            show: Whether to draw them.
        """
        self._boxes = dict(boxes)
        self._show_boxes = show
        if self._selected not in self._boxes:
            self._selected = None
        self._redraw()

    def select_box(self, box_id: Optional[int], notify: bool = False) -> None:
        """Select a box programmatically.

        Args:
            box_id: Box id or ``None``.
            notify: Fire ``on_box_selected`` too.
        """
        self._selected = box_id if box_id in self._boxes else None
        self._redraw()
        if notify and self.on_box_selected:
            self.on_box_selected(self._selected)

    def fit_to_window(self) -> None:
        """Zoom so the whole composite fits and centre it."""
        if self._composite is None:
            return
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        iw, ih = self._composite.size
        self.zoom = min(cw / iw, ch / ih) * 0.98
        self.ox = (cw - iw * self.zoom) / 2
        self.oy = (ch - ih * self.zoom) / 2
        self._redraw()

    # -- coordinate helpers ----------------------------------------------
    def _to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Image coordinates (composite) -> canvas coordinates."""
        return self.ox + x * self.zoom, self.oy + y * self.zoom

    def _to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        """Canvas coordinates -> image coordinates (composite)."""
        return (cx - self.ox) / self.zoom, (cy - self.oy) / self.zoom

    def _box_canvas(self, box: BBox) -> Tuple[float, float, float, float]:
        """Canvas rectangle of an image-space box on the processed layer."""
        x1, y1 = self._to_canvas(box[0] + self._box_offset_x, box[1])
        x2, y2 = self._to_canvas(box[2] + self._box_offset_x, box[3])
        return x1, y1, x2, y2

    # -- drawing ---------------------------------------------------------
    def _redraw(self) -> None:
        """Redraw the visible part of the image and the boxes."""
        self.delete("all")
        if self._composite is None:
            if self.placeholder:
                self.create_text(self.winfo_width() / 2, self.winfo_height() / 2, text=self.placeholder,
                                 fill="#8b949e", font=("Segoe UI", 18), justify="center")
            return
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        iw, ih = self._composite.size
        ix1, iy1 = self._to_image(0, 0)
        ix2, iy2 = self._to_image(cw, ch)
        ix1, iy1 = max(0, int(ix1) - 1), max(0, int(iy1) - 1)
        ix2, iy2 = min(iw, int(ix2) + 2), min(ih, int(iy2) + 2)
        if ix2 > ix1 and iy2 > iy1:
            crop = self._composite.crop((ix1, iy1, ix2, iy2))
            size = (max(1, round((ix2 - ix1) * self.zoom)), max(1, round((iy2 - iy1) * self.zoom)))
            resample = Image.Resampling.BILINEAR if self.zoom < 1 else Image.Resampling.NEAREST
            self._photo = ImageTk.PhotoImage(crop.resize(size, resample))
            cx, cy = self._to_canvas(ix1, iy1)
            self.create_image(cx, cy, image=self._photo, anchor="nw")
        self._draw_boxes()

    def _draw_boxes(self) -> None:
        """Draw box outlines and handles."""
        self.delete("box")
        if not self._show_boxes:
            return
        for bid, box in self._boxes.items():
            x1, y1, x2, y2 = self._box_canvas(box)
            sel = bid == self._selected
            self.create_rectangle(x1, y1, x2, y2, outline=COLOR_SELECTED if sel else COLOR_BOX,
                                  width=2 if sel else 1, tags="box")
            self.create_text(x1 + 3, y1 + 2, text=str(bid), anchor="nw",
                             fill=COLOR_SELECTED if sel else COLOR_BOX, tags="box")
            if sel:
                for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE,
                                          fill=COLOR_SELECTED, outline="", tags="box")

    # -- hit testing -----------------------------------------------------
    def _hit(self, x: float, y: float) -> Tuple[Optional[int], str]:
        """Return ``(box_id, mode)`` under the cursor; mode is move/tl/tr/bl/br."""
        if not self._show_boxes:
            return None, ""
        if self._selected is not None and self._selected in self._boxes:
            x1, y1, x2, y2 = self._box_canvas(self._boxes[self._selected])
            for name, hx, hy in (("tl", x1, y1), ("tr", x2, y1), ("bl", x1, y2), ("br", x2, y2)):
                if abs(x - hx) <= HANDLE + 2 and abs(y - hy) <= HANDLE + 2:
                    return self._selected, name
        hits = []
        for bid, box in self._boxes.items():
            x1, y1, x2, y2 = self._box_canvas(box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits.append(((x2 - x1) * (y2 - y1), bid))
        if hits:
            return min(hits)[1], "move"
        return None, ""

    # -- mouse -----------------------------------------------------------
    def _on_press(self, e: tk.Event) -> None:
        """Start a move/resize/pan depending on what is under the cursor."""
        self.focus_set()
        bid, mode = self._hit(e.x, e.y)
        if bid is not None:
            if bid != self._selected:
                self._selected = bid
                if self.on_box_selected:
                    self.on_box_selected(bid)
            self._drag = (mode, e.x, e.y, self._boxes[bid])
            self._draw_boxes()
        else:
            if self._selected is not None:
                self._selected = None
                if self.on_box_selected:
                    self.on_box_selected(None)
                self._draw_boxes()
            self._drag = ("pan", e.x, e.y, None)

    def _on_drag(self, e: tk.Event) -> None:
        """Update the box or the view while dragging."""
        if not self._drag:
            return
        mode, sx, sy, start = self._drag
        if mode == "pan":
            self.ox += e.x - sx
            self.oy += e.y - sy
            self._drag = ("pan", e.x, e.y, None)
            self._redraw()
            return
        if start is None or self._selected is None:
            return
        dx, dy = (e.x - sx) / self.zoom, (e.y - sy) / self.zoom
        x1, y1, x2, y2 = start
        if mode == "move":
            x1, x2, y1, y2 = x1 + dx, x2 + dx, y1 + dy, y2 + dy
        else:
            if "l" in mode:
                x1 += dx
            if "r" in mode:
                x2 += dx
            if "t" in mode:
                y1 += dy
            if "b" in mode:
                y2 += dy
        nx1, nx2 = sorted((int(x1), int(x2)))
        ny1, ny2 = sorted((int(y1), int(y2)))
        if nx2 - nx1 >= 8 and ny2 - ny1 >= 8:
            self._boxes[self._selected] = (nx1, ny1, nx2, ny2)
            self._draw_boxes()

    def _on_release(self, e: tk.Event) -> None:
        """Commit a box edit."""
        drag, self._drag = self._drag, None
        if drag and drag[0] != "pan" and self._selected is not None and self.on_box_changed:
            if self._boxes[self._selected] != drag[3]:
                self.on_box_changed(self._selected, self._boxes[self._selected])

    def _on_pan_start(self, e: tk.Event) -> None:
        """Begin panning with the middle/right button."""
        self._drag = ("pan", e.x, e.y, None)

    def _on_pan_move(self, e: tk.Event) -> None:
        """Pan with the middle/right button."""
        self._on_drag(e)

    def _on_wheel(self, e: tk.Event) -> None:
        """Zoom with the mouse wheel (Windows/macOS delta conventions)."""
        step = e.delta if sys.platform == "darwin" else e.delta / 120
        self._zoom_at(e.x, e.y, 1.15 ** step)

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        """Zoom keeping the image point under ``(cx, cy)`` fixed."""
        if self._composite is None:
            return
        new_zoom = max(0.02, min(16.0, self.zoom * factor))
        ix, iy = self._to_image(cx, cy)
        self.zoom = new_zoom
        self.ox = cx - ix * new_zoom
        self.oy = cy - iy * new_zoom
        self._redraw()
