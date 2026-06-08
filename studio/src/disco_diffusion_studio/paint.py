"""Brushes, a colour palette, and a paintable RGBA layer.

The :class:`PaintLayer` is a generation-resolution RGBA buffer the user paints into. The app
shows it as an overlay on the image and hands snapshots to the worker, which injects them into
the diffusion latent (see ``disco_diffusion.Sampler.paint``).
"""

from __future__ import annotations

import math

import numpy as np
import pygame

# Brush kinds (in display order). Selected by name.
BRUSHES = ["Soft", "Hard", "Spray"]

# The colour palette now lives in studio/config.toml (loaded via presets.load_colours), so it can
# be edited and extended with recently-picked colours rather than hard-coded here.


class PaintLayer:
    """A generation-resolution RGBA buffer with brush stamping.

    ``rgb`` is ``(H, W, 3)`` in ``[0, 1]`` and ``alpha`` is ``(H, W)`` in ``[0, 1]``.
    ``dirty`` marks strokes not yet handed to the worker; ``_surf_dirty`` caches the
    overlay surface.
    """

    def __init__(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self.rgb = np.zeros((height, width, 3), dtype=np.float32)
        self.alpha = np.zeros((height, width), dtype=np.float32)
        self.tint = np.zeros((height, width), dtype=np.float32)  # per-pixel tinted-noise amount
        self.dirty = False  # new strokes awaiting hand-off to the worker
        self._nonempty = False
        self._surf_dirty = True
        self._surface: pygame.Surface | None = None

    def empty(self) -> bool:
        return not self._nonempty

    def clear(self) -> None:
        if not self._nonempty:
            return
        self.rgb.fill(0.0)
        self.alpha.fill(0.0)
        self.tint.fill(0.0)
        self._nonempty = False
        self.dirty = False
        self._surf_dirty = True

    def stamp(
        self,
        cx: float,
        cy: float,
        radius: float,
        color01: tuple[float, float, float],
        strength: float,
        brush: str,
        tint: float = 0.0,
    ) -> None:
        """Composite one brush dab centred at (cx, cy) into the layer.

        ``tint`` (0..1) marks these pixels for the tinted-noise injection mode.
        """
        r = max(1, int(round(radius)))
        x0, x1 = max(0, int(cx) - r), min(self.w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(self.h, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        ys, xs = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        if brush == "Hard":
            a = (dist <= r).astype(np.float32)
        elif brush == "Spray":
            a = ((dist <= r) & (np.random.random(dist.shape) < 0.06)).astype(np.float32)
        else:  # Soft
            a = np.clip(1.0 - dist / r, 0.0, 1.0).astype(np.float32) ** 1.5
        a *= float(strength)
        if not a.any():
            return
        # Alpha-over composite of `color01` (and the tint flag) onto the existing sub-region.
        sub_rgb = self.rgb[y0:y1, x0:x1]
        sub_a = self.alpha[y0:y1, x0:x1]
        out_a = a + sub_a * (1.0 - a)
        color = np.asarray(color01, dtype=np.float32)
        blended = color[None, None, :] * a[..., None] + sub_rgb * (sub_a * (1.0 - a))[..., None]
        safe = out_a > 1e-6
        blended[safe] /= out_a[safe][..., None]
        self.rgb[y0:y1, x0:x1] = blended
        self.alpha[y0:y1, x0:x1] = out_a
        sub_t = self.tint[y0:y1, x0:x1]
        self.tint[y0:y1, x0:x1] = float(tint) * a + sub_t * (1.0 - a)
        self._nonempty = True
        self.dirty = True
        self._surf_dirty = True

    def stroke(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        radius: float,
        color01: tuple[float, float, float],
        strength: float,
        brush: str,
        tint: float = 0.0,
    ) -> None:
        """Stamp the brush along the segment p0->p1 so fast drags stay continuous."""
        (x0, y0), (x1, y1) = p0, p1
        dist = math.hypot(x1 - x0, y1 - y0)
        spacing = max(1.0, radius * 0.25)
        steps = int(dist / spacing) + 1
        for i in range(steps + 1):
            t = i / max(1, steps)
            self.stamp(
                x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius, color01, strength, brush, tint
            )

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A copy of (rgb, alpha, tint) to hand to the worker thread."""
        return self.rgb.copy(), self.alpha.copy(), self.tint.copy()

    def to_surface(self) -> pygame.Surface:
        """A cached RGBA pygame surface of the layer, for the on-canvas overlay."""
        if self._surface is None or self._surf_dirty:
            rgb = (np.clip(self.rgb, 0, 1) * 255).astype(np.uint8)
            a = (np.clip(self.alpha, 0, 1) * 255).astype(np.uint8)
            rgba = np.dstack([rgb, a])  # (H, W, 4), row-major
            surf = pygame.image.frombuffer(rgba.tobytes(), (self.w, self.h), "RGBA")
            try:
                surf = surf.convert_alpha()  # optimise for blitting (needs a display surface)
            except pygame.error:
                pass  # no display set yet (e.g. headless); the RGBA surface already has alpha
            self._surface = surf
            self._surf_dirty = False
        return self._surface
