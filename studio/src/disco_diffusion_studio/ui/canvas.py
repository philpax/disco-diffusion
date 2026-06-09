"""The canvas area: the zoom/pan view of the generation-size image, the paint layer, and the frame.

:class:`Canvas` is the image area as one object — it owns the view transform (zoom/pan), the paint
controller (the active stroke + overlays), and the latest rendered frame surface. The App supplies
the two things that come from elsewhere: the on-screen image region (window geometry) and the
generation (canvas) size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from ..paint import PaintController
from ..view import ViewTransform

if TYPE_CHECKING:
    from ..app import App


class Canvas:
    """View transform + paint controller + rendered frame for the image area."""

    def __init__(self, app: App) -> None:
        self.app = app
        self.view = ViewTransform()
        self.paint = PaintController.for_canvas(app.width, app.height)
        self.frame_surface: pygame.Surface | None = None
        self.frame_key: tuple[int, int] | None = None  # (id(array), index) to detect new frames

    def _size(self) -> tuple[int, int]:
        return (self.app.width, self.app.height)

    def _region(self) -> pygame.Rect:
        return self.app.layout.image_region()

    # -- view transform (supplies the ViewTransform with the live region + canvas size) --
    def fit(self) -> None:
        self.view.fit(self._region(), *self._size())

    def zoom_at(self, pos: tuple[int, int], factor: float) -> None:
        self.view.zoom_at(pos, factor, self._region(), *self._size())

    def clamp_pan(self) -> None:
        self.view.clamp_pan(self._region(), *self._size())

    def screen_to_canvas(self, pos: tuple[int, int]) -> tuple[float, float] | None:
        return self.view.screen_to_canvas(pos, self._region(), *self._size())

    def canvas_screen_rect(self) -> pygame.Rect:
        return self.view.canvas_screen_rect(*self._size())

    def blit(self, surf: pygame.Surface) -> None:
        """Blit a canvas-resolution surface onto the screen under the current view transform."""
        self.view.blit(self.app.screen, surf, self._region())

    # -- frame + paint --
    def resize(self, width: int, height: int) -> None:
        """Rebuild the paint layer for a new generation size and drop the stale frame."""
        self.paint.resize(width, height)
        self.frame_surface = None
        self.frame_key = None

    def update_frame_surface(self) -> None:
        """Pull the worker's latest frame into ``frame_surface`` (+ refresh the step label)."""
        app = self.app
        if app.worker is None:
            return
        frame = app.worker.latest_frame()
        if frame is None:
            return
        # Refresh the step label every frame (cheap — pygame_gui no-ops if unchanged). A UI
        # rebuild (resize / divider drag) recreates the label as "step 0 / 0", so updating it
        # only when the *surface* changes would leave it stale at "0 / 0" once generation stops.
        app.bottom_bar.step_label.set_text(f"step {frame.index} / {frame.total}")
        key = (id(frame.image), frame.index)
        if key == self.frame_key:
            return
        self.frame_key = key
        # pygame.surfarray expects (W, H, 3), so swap the first two axes.
        self.frame_surface = pygame.surfarray.make_surface(frame.image.swapaxes(0, 1))

    def paint_at(self, pos: tuple[int, int]) -> None:
        """Paint into the layer at screen ``pos`` (no-op if outside the canvas)."""
        gen = self.screen_to_canvas(pos)
        if gen is not None:
            self.paint.paint_to(gen, self.app.brush)
