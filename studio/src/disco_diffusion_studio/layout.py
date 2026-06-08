"""Sizing tokens and a tiny flow-layout used to position widgets dynamically.

Nothing in the app uses absolute coordinates: a `Stack` hands out horizontal `Row`s
top-to-bottom, and each `Row` packs widgets with `left()` / `right()` / `fill()`. The whole
UI therefore re-flows for any window width.
"""

from __future__ import annotations

import pygame

# Layout tokens (sizes only — positions come from the flow layout).
PAD = 10  # gap between adjacent widgets
MARGIN = 16  # panel inset from the window edge
CTRL_H = 32  # height of a control (button / text box)
LABEL_H = 24  # height of a standalone label line
ROW_PITCH = 46  # vertical pitch of one prompt row

# The prompt list is a scrolling area filling the panel below the fixed rows (4 control rows +
# 1 prompt-header label). PANEL_H is the default height (showing ~2 prompt rows); the panel is
# user-resizable via a horizontal divider, down to PANEL_MIN (the fixed rows + one prompt row)
# and up to whatever leaves MIN_IMAGE_H for the image. (The status shares the transport row;
# steps / size and the advanced controls live in the right-hand sidebar.)
PROMPT_LIST_H = 2 * ROW_PITCH + 12
_PANEL_FIXED = PAD + 4 * (CTRL_H + PAD) + 1 * (LABEL_H + PAD) + PAD  # rows above the prompt list
PANEL_H = _PANEL_FIXED + PROMPT_LIST_H
PANEL_MIN = _PANEL_FIXED + ROW_PITCH  # fixed rows + room for a single prompt row

# Right-hand sidebar (size + advanced controls + the "Current" readout). User-resizable via a
# draggable divider; the left column (image + bottom panel) keeps at least MIN_LEFT_PANEL_W.
SIDEBAR_W_DEFAULT = 360
SIDEBAR_W_MIN = 300
SIDEBAR_W_MAX = 680
DIVIDER_W = 6  # width of the draggable divider band between the left column and the sidebar
MIN_LEFT_PANEL_W = 660  # the bottom panel's tool row needs this much

# Window / image sizing.
MIN_WINDOW_W = MIN_LEFT_PANEL_W + DIVIDER_W + SIDEBAR_W_MIN  # left column + divider + sidebar
MIN_IMAGE_H = 320  # minimum height of the image region (image is centered within it)
DEFAULT_W, DEFAULT_H = 1280, 768
MAX_SIDE = 1280  # cap each dimension (snapped to x64); the UNet accepts any multiple of 64
MIN_SIDE = 64


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
