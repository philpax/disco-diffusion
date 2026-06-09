"""Small shared helpers (parsing + surface conversion), kept out of ``app`` to avoid cycles."""

from __future__ import annotations

import pygame
from PIL import Image

from .constants import STEP_MAX, STEP_MIN


def surface_to_pil(surface: pygame.Surface) -> Image.Image:
    """Copy a pygame surface to a PIL image (transposing pygame's column-major (W, H) layout)."""
    arr = pygame.surfarray.array3d(surface).swapaxes(0, 1).astype("uint8")  # (H, W, 3)
    return Image.fromarray(arr)


def clamp_steps(text: str) -> int:
    """Parse a steps box value and clamp it to the supported range.

    Raises ``ValueError`` on non-integer input so the caller can restore the prior value.
    """
    return max(STEP_MIN, min(STEP_MAX, int(text)))


def int_or(text: str, fallback: int) -> int:
    try:
        return int(text)
    except ValueError:
        return fallback
