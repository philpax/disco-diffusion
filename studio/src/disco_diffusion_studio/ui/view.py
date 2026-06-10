"""The canvas view transform: zoom + pan of the generation-resolution canvas on screen.

The canvas is the generation-size image (``width`` x ``height`` pixels). The :class:`ViewTransform`
maps between *canvas* coordinates (generation pixels) and *screen* coordinates (window pixels)
under a zoom factor and a pan offset. It owns only the view state; the on-screen image region and
the canvas size are passed in by the caller (they live on the App, which re-flows them with the
window), so the transform stays a small, testable value object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

MIN_ZOOM, MAX_ZOOM = 0.1, 16.0  # view zoom bounds


@dataclass
class ViewTransform:
    """Zoom + pan of the canvas within the image region. Screen px <-> canvas (generation) px."""

    zoom: float = 1.0
    pan: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(0, 0))

    def canvas_screen_rect(self, canvas_w: int, canvas_h: int) -> pygame.Rect:
        """The canvas bounds in screen coords under the current view transform."""
        return pygame.Rect(
            int(self.pan.x), int(self.pan.y), int(canvas_w * self.zoom), int(canvas_h * self.zoom)
        )

    def fit(self, region: pygame.Rect, canvas_w: int, canvas_h: int) -> None:
        """Reset the view so the whole canvas fits the viewport, centred."""
        if canvas_w and canvas_h:
            self.zoom = min(region.width / canvas_w, region.height / canvas_h)
        else:
            self.zoom = 1.0
        self.pan = pygame.Vector2(
            region.x + (region.width - canvas_w * self.zoom) / 2,
            region.y + (region.height - canvas_h * self.zoom) / 2,
        )

    def zoom_at(
        self, pos: tuple[int, int], factor: float, region: pygame.Rect, canvas_w: int, canvas_h: int
    ) -> None:
        """Multiply the zoom by ``factor``, keeping the canvas point under ``pos`` fixed."""
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        ratio = new_zoom / self.zoom
        anchor = pygame.Vector2(pos)
        self.pan = anchor - (anchor - self.pan) * ratio
        self.zoom = new_zoom
        self.clamp_pan(region, canvas_w, canvas_h)

    def clamp_pan(self, region: pygame.Rect, canvas_w: int, canvas_h: int) -> None:
        """Keep the canvas centre within the viewport so the canvas can't be lost off-screen."""
        cw, ch = canvas_w * self.zoom, canvas_h * self.zoom
        cx = max(region.left, min(region.right, self.pan.x + cw / 2))
        cy = max(region.top, min(region.bottom, self.pan.y + ch / 2))
        self.pan = pygame.Vector2(cx - cw / 2, cy - ch / 2)

    def screen_to_canvas(
        self, pos: tuple[int, int], region: pygame.Rect, canvas_w: int, canvas_h: int
    ) -> tuple[float, float] | None:
        """Map a screen position to canvas-pixel coords, or None if outside the canvas."""
        if not region.collidepoint(pos):
            return None
        cx = (pos[0] - self.pan.x) / self.zoom
        cy = (pos[1] - self.pan.y) / self.zoom
        return (cx, cy) if 0 <= cx < canvas_w and 0 <= cy < canvas_h else None

    def blit(self, screen: pygame.Surface, surf: pygame.Surface, region: pygame.Rect) -> None:
        """Blit only the visible part of a canvas-resolution surface under the view transform."""
        w, h = surf.get_size()
        z = self.zoom
        cx0 = max(0, int((region.left - self.pan.x) / z))
        cy0 = max(0, int((region.top - self.pan.y) / z))
        cx1 = min(w, math.ceil((region.right - self.pan.x) / z))
        cy1 = min(h, math.ceil((region.bottom - self.pan.y) / z))
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = surf.subsurface(pygame.Rect(cx0, cy0, cx1 - cx0, cy1 - cy0))
        dest = (max(1, int((cx1 - cx0) * z)), max(1, int((cy1 - cy0) * z)))
        scaled = pygame.transform.smoothscale(sub, dest)
        screen.blit(scaled, (self.pan.x + cx0 * z, self.pan.y + cy0 * z))
