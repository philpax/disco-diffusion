"""Sizing tokens and a tiny flow-layout used to position widgets dynamically.

Nothing in the app uses absolute coordinates: a `Stack` hands out horizontal `Row`s
top-to-bottom, and each `Row` packs widgets with `left()` / `right()` / `fill()`. The whole
UI therefore re-flows for any window width.
"""

from __future__ import annotations

import pygame

# Window / image sizing.
PANEL_H = 408  # height of the control panel below the image
MIN_WINDOW_W = 940  # never let the window get narrower than the controls need
MIN_IMAGE_H = 360  # minimum height of the image region (image is centered within it)
DEFAULT_W, DEFAULT_H = 1280, 768
MAX_SIDE = 1280  # cap each dimension (snapped to x64); the UNet accepts any multiple of 64
MIN_SIDE = 64

# Layout tokens (sizes only — positions come from the flow layout).
PAD = 10  # gap between adjacent widgets
MARGIN = 16  # panel inset from the window edge
CTRL_H = 32  # height of a control (button / text box)
LABEL_H = 24  # height of a standalone label line
ROW_PITCH = 46  # vertical pitch of one prompt row


def snap_side(value: int) -> int:
    """Snap a dimension to a multiple of 64, clamped to [MIN_SIDE, MAX_SIDE]."""
    snapped = round(value / 64) * 64
    return max(MIN_SIDE, min(MAX_SIDE, snapped))


class Row:
    """Packs widgets left-to-right (and right-to-left) within a horizontal band.

    `left(w)` / `right(w)` hand out a `pygame.Rect` of width `w` from each end and advance
    the cursor; `fill()` returns whatever span is left between the two cursors.
    """

    def __init__(self, x: int, y: int, width: int, height: int, pad: int = PAD) -> None:
        self._left = x
        self._right = x + width
        self.y = y
        self.h = height
        self.pad = pad

    def left(self, w: int) -> pygame.Rect:
        rect = pygame.Rect(self._left, self.y, w, self.h)
        self._left += w + self.pad
        return rect

    def right(self, w: int) -> pygame.Rect:
        self._right -= w
        rect = pygame.Rect(self._right, self.y, w, self.h)
        self._right -= self.pad
        return rect

    def fill(self) -> pygame.Rect:
        return pygame.Rect(self._left, self.y, max(10, self._right - self._left), self.h)


class Stack:
    """Stacks `Row`s top-to-bottom from a starting point, tracking the y cursor."""

    def __init__(self, x: int, y: int, width: int, pad: int = PAD) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.pad = pad

    def row(self, height: int) -> Row:
        row = Row(self.x, self.y, self.width, height)
        self.y += height + self.pad
        return row
