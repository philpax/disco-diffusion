"""The canvas area: the zoom/pan view of the generation-size image, the paint layer, and the frame.

:class:`Canvas` is the image area as one object — it owns the view transform (zoom/pan), the paint
controller (the active stroke + overlays), and the latest rendered frame surface. The App supplies
the two things that come from elsewhere: the on-screen image region (window geometry) and the
generation (canvas) size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from ..constants import CANVAS_BORDER, CANVAS_EMPTY_BG, DRAW_HELP, NAV_HELP
from ..paint import PaintController
from ..view import ViewTransform

if TYPE_CHECKING:
    from ..app import App
    from ..layout import Layout
    from ..state import PaintState, SharedState
    from .bottom_bar import BottomBar


class Canvas:
    """View transform + paint controller + rendered frame for the image area."""

    def __init__(
        self,
        screen: pygame.Surface,
        layout: Layout,
        state: SharedState,
        paint: PaintState,
        bottom_bar: BottomBar,
    ) -> None:
        self.screen = screen  # the window surface (App re-points this on resize)
        self.layout = layout
        self.state = state
        self.paint_state = paint  # brush settings (self.paint below is the stroke controller)
        self.bottom_bar = bottom_bar
        self.view = ViewTransform()
        self.paint = PaintController.for_canvas(state.width, state.height)
        self.frame_surface: pygame.Surface | None = None
        self.frame_key: tuple[int, int] | None = None  # (id(array), index) to detect new frames

    def _size(self) -> tuple[int, int]:
        return (self.state.width, self.state.height)

    def _region(self) -> pygame.Rect:
        return self.layout.image_region()

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
        self.view.blit(self.screen, surf, self._region())

    # -- frame + paint --
    def resize(self, width: int, height: int) -> None:
        """Rebuild the paint layer for a new generation size and drop the stale frame."""
        self.paint.resize(width, height)
        self.frame_surface = None
        self.frame_key = None

    def apply_size(self, width: int, height: int) -> None:
        """Adopt a new generation size: rebuild the paint layer, drop the frame, refit the view."""
        self.resize(width, height)
        self.fit()

    def clear_frame(self) -> None:
        """Drop the rendered frame, revealing the init preview / empty canvas underneath."""
        self.frame_surface = None
        self.frame_key = None

    def frame_for_save(self) -> pygame.Surface | None:
        """A frozen copy of the current frame to save (or None if nothing's rendered yet)."""
        return self.frame_surface.copy() if self.frame_surface is not None else None

    def update_frame_surface(self) -> None:
        """Pull the worker's latest frame into ``frame_surface`` (+ refresh the step label)."""
        if self.state.worker is None:
            return
        frame = self.state.worker.latest_frame()
        if frame is None:
            return
        # Refresh the step label every frame (cheap — pygame_gui no-ops if unchanged). A UI
        # rebuild (resize / divider drag) recreates the label as "step 0 / 0", so updating it
        # only when the *surface* changes would leave it stale at "0 / 0" once generation stops.
        self.bottom_bar.set_step_label(f"step {frame.index} / {frame.total}")
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
            self.paint.paint_to(gen, self.paint_state.brush)

    # -- rendering --
    def draw(self, app: App) -> None:
        """Render the image area: the frame (or init/empty state) + paint overlays, then the
        brush-ring cursor and the help HUD on top. Drawn into the image region (clipped)."""
        brush = self.paint_state.brush
        # Draw the canvas (and unbaked paint overlay) under the view transform, clipped to the
        # viewport so a zoomed/panned canvas never spills into the panel.
        self.screen.set_clip(self.layout.image_region())
        crect = self.canvas_screen_rect()
        surface = app.history.displayed_surface()
        if surface is not None:
            self.blit(surface)
        elif self.state.init.surface is not None:
            # No frame yet but an init image is set: preview it (dimmed) so it's clear the run
            # will seed from it.
            self.blit(self.state.init.surface)
            scrim = pygame.Surface(crect.size, pygame.SRCALPHA)
            scrim.fill((10, 12, 16, 120))
            self.screen.blit(scrim, crect.topleft)
            label = app._hud_font.render(
                f"init: {self.state.init.label} — press Play to evolve", True, (224, 228, 236)
            )
            self.screen.blit(label, label.get_rect(center=crect.center))
        else:
            # No frame yet: show the canvas bounds so the size/aspect is clear before Play.
            pygame.draw.rect(self.screen, CANVAS_EMPTY_BG, crect)
            label = app._hud_font.render(
                f"{self.state.width} × {self.state.height} — press Play", True, (140, 147, 160)
            )
            self.screen.blit(label, label.get_rect(center=crect.center))
        # Paint overlays only on the live view (hidden while previewing history): the in-progress
        # stroke plus any flushed strokes not yet baked into a published frame.
        if self.state.timeline.preview_index is None:
            for overlay, _ in self.paint.pending_overlays:
                self.blit(overlay)
            if not self.paint.layer.empty():
                self.blit(self.paint.layer.to_surface())
        pygame.draw.rect(self.screen, CANVAS_BORDER, crect, 1)  # canvas outline at any zoom
        self.screen.set_clip(None)
        # Brush ring (scaled by zoom) — only in draw mode (not navigating, not previewing, and
        # not while a dialog window is up).
        region = self.layout.image_region()
        if (
            not app._navigating
            and self.state.timeline.preview_index is None
            and not app._modal_open()
            and region.collidepoint(app._mouse_pos)
        ):
            ring = max(2, int(brush.size * self.view.zoom))
            pygame.draw.circle(self.screen, brush.color, app._mouse_pos, ring, 2)
            pygame.draw.circle(self.screen, (255, 255, 255), app._mouse_pos, ring + 1, 1)
        # Help HUD in the corner of the canvas (doesn't cost panel height), per mode.
        text = app._hud_font.render(
            NAV_HELP if app._navigating else DRAW_HELP, True, (210, 214, 222)
        )
        pad = 6
        chip = pygame.Surface(
            (text.get_width() + 2 * pad, text.get_height() + 2 * pad), pygame.SRCALPHA
        )
        chip.fill((0, 0, 0, 120))
        pos = (10, region.bottom - chip.get_height() - 10)
        self.screen.blit(chip, pos)
        self.screen.blit(text, (pos[0] + pad, pos[1] + pad))
