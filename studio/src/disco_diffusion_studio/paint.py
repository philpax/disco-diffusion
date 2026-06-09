"""Brushes and the paint subsystem: a paintable RGBA layer + the stroke lifecycle.

The :class:`PaintLayer` is a generation-resolution RGBA buffer the user paints into.
:class:`PaintController` drives the stroke lifecycle — paint into the layer, flush a finished
stroke to the worker as one batch (its own checkpoint), and keep it on-screen as a pending overlay
until a baked frame incorporates it (see ``disco_diffusion.Sampler.paint``). The app keeps the
brush *parameters* (size / strength / colour / kind / noise) and passes them in as a :class:`Brush`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pygame

if TYPE_CHECKING:
    from .worker import GenerationWorker

# Brush kinds (in display order). Selected by name.
BRUSHES = ["Soft", "Hard", "Spray"]

# Noise-mode injection shaping: for noise-mode pixels the *injected* mask is gamma-shaped to
# `NOISE_MAX_INJECT * opacity**gamma`, while the on-screen overlay stays at the raw opacity.
NOISE_MAX_INJECT = 0.2
NOISE_OPACITY_GAMMA = 2.0


class Brush(NamedTuple):
    """The parameters one stroke is painted with (a snapshot of the App's brush state)."""

    type: str
    size: float
    strength: float
    color: tuple[int, int, int]
    noise: bool


class PaintLayer:
    """A generation-resolution RGBA buffer with brush stamping.

    ``rgb`` is ``(H, W, 3)`` in ``[0, 1]`` and ``alpha`` is ``(H, W)`` in ``[0, 1]``.
    ``_surf_dirty`` caches the overlay surface.
    """

    def __init__(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self.rgb = np.zeros((height, width, 3), dtype=np.float32)
        self.alpha = np.zeros((height, width), dtype=np.float32)
        self.tint = np.zeros((height, width), dtype=np.float32)  # per-pixel tinted-noise amount
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


@dataclass
class PaintController:
    """The active paint layer plus the stroke lifecycle (paint -> flush to worker -> overlay)."""

    layer: PaintLayer
    pending_overlays: list[tuple[pygame.Surface, int]] = field(default_factory=list)
    submitted: int = 0  # mirrors the worker's paint_applied_count target for the last flush
    last_gen: tuple[float, float] | None = None  # previous canvas point in the current stroke
    painting: bool = False

    @classmethod
    def for_canvas(cls, width: int, height: int) -> PaintController:
        return cls(layer=PaintLayer(width, height))

    def resize(self, width: int, height: int) -> None:
        """Rebuild the (generation-resolution) layer for a new canvas size; drop stale overlays."""
        self.layer = PaintLayer(width, height)
        self.reset_overlays()

    def reset_overlays(self, submitted: int = 0) -> None:
        """Drop pending overlays and re-baseline the submit counter (fresh run / after a revert)."""
        self.pending_overlays = []
        self.submitted = submitted

    def begin(self) -> None:
        """Start a stroke (left mouse-down on the canvas)."""
        self.painting = True
        self.last_gen = None

    def end(self) -> None:
        """End a stroke (mouse-up); the caller flushes it."""
        self.painting = False
        self.last_gen = None

    def paint_to(self, gen: tuple[float, float], brush: Brush) -> None:
        """Stamp (or stroke from the last point) the brush into the layer at canvas coords."""
        c = brush.color
        color01 = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        tint = 1.0 if brush.noise else 0.0
        if self.last_gen is None:
            self.layer.stamp(gen[0], gen[1], brush.size, color01, brush.strength, brush.type, tint)
        else:
            self.layer.stroke(
                self.last_gen, gen, brush.size, color01, brush.strength, brush.type, tint
            )
        self.last_gen = gen

    def flush(self, worker: GenerationWorker | None, brush: Brush) -> None:
        """Hand the just-finished stroke to the worker as one batch (its own checkpoint).

        The stroke stays on screen as a pending overlay until a baked frame incorporates it (the
        injecting step takes seconds), then clears. Each stroke gets its own batch, so painting
        with different settings yields separate history entries rather than merging.
        """
        layer = self.layer
        if layer.empty():
            return
        if worker is None or not worker.is_alive():
            return  # no run yet: keep the stroke on the active overlay; the next run flushes it
        rgb, alpha, tint = layer.snapshot()
        # Gamma-shape the *injected* mask for noise-mode pixels (tint/alpha = how much of the
        # pixel is noise-mode), keeping the on-screen overlay at the raw opacity.
        frac_noise = np.divide(tint, alpha, out=np.zeros_like(tint), where=alpha > 1e-6)
        shaped = NOISE_MAX_INJECT * alpha**NOISE_OPACITY_GAMMA
        alpha = alpha * (1.0 - frac_noise) + shaped * frac_noise
        label = f"paint {brush.type.lower()} {int(brush.size)}px"
        # Keep `submitted` monotonic even if a frame's count overtook our last target (e.g. after
        # a revert reset it), so an overlay clears only once its own batch has been applied.
        self.submitted = max(self.submitted, worker.paint_applied_count) + 1
        worker.set_paint(rgb, alpha, tint, label)
        self.pending_overlays.append((layer.to_surface().copy(), self.submitted))
        layer.clear()  # next stroke starts fresh; the overlay copy keeps this one visible

    def sync(self, worker: GenerationWorker | None) -> None:
        """Drop pending stroke overlays once a baked frame has incorporated them."""
        if worker is None or not self.pending_overlays:
            return
        frame = worker.latest_frame()
        if frame is not None:
            self.pending_overlays = [o for o in self.pending_overlays if o[1] > frame.paint_applied]
