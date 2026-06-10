"""The img2img init image: the optional seed image, its denoise %, and the on-canvas preview.

Pressing Play seeds a fresh run from :attr:`image`, noised to :meth:`skip_steps` of the way in;
without an image the run starts from scratch. :class:`InitImage` holds that state and the two bits
of derived data — the canvas-resolution preview surface and the skip-steps count — while the App
keeps the surrounding UI (the file dialog, the denoise slider, the status readout).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame
from PIL import Image


@dataclass
class InitImage:
    """An optional img2img seed image, its denoise %, and a cached preview surface."""

    image: Image.Image | None = None
    label: str = "none"
    denoise: int = 60  # % of the schedule to re-diffuse: 100% = from scratch, low = keep the image
    surface: pygame.Surface | None = None  # canvas-resolution preview, shown before Play

    def set(self, image: Image.Image, label: str, canvas_w: int, canvas_h: int) -> None:
        """Adopt a new seed image (rebuilding the preview to the canvas size)."""
        self.image = image
        self.label = label
        self.rebuild_surface(canvas_w, canvas_h)

    def clear(self) -> None:
        """Drop the seed image so the next run starts from scratch."""
        self.image = None
        self.label = "none"
        self.surface = None

    def rebuild_surface(self, canvas_w: int, canvas_h: int) -> None:
        """Rebuild the canvas-resolution preview (called on set + whenever the canvas resizes)."""
        if self.image is None:
            self.surface = None
            return
        pil = self.image.convert("RGB").resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self.surface = pygame.surfarray.make_surface(np.asarray(pil).swapaxes(0, 1))

    def skip_steps(self, steps: int) -> int:
        """The skip_steps the next run starts at for this init + denoise (0 when there's no image).

        Denoise 100% -> skip 0 (full re-diffusion); 0% -> skip steps-1 (keep the init, one step).
        """
        if self.image is None:
            return 0
        skip = round((1 - self.denoise / 100) * steps)
        return max(0, min(steps - 1, skip))
