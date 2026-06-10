"""Sizing tokens and a tiny flow-layout used to position widgets dynamically.

Nothing in the app uses absolute coordinates: a `Stack` hands out horizontal `Row`s
top-to-bottom, and each `Row` packs widgets with `left()` / `right()` / `fill()`. The whole
UI therefore re-flows for any window width.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class Layout:
    """Window geometry: the split between the left column (image + bottom panel) and the right
    sidebar, derived from the window size and the two user-draggable divider positions.

    Pure rect maths over four numbers; the setters clamp and report whether anything changed, and
    the App owns the side effects (queueing a relayout). The screen origin is the window top-left.
    """

    win_w: int
    win_h: int
    sidebar_w: int  # right sidebar width (draggable)
    panel_h: int  # bottom-panel height (draggable)

    def panel_w(self) -> int:
        """Width of the left column (image + bottom panel) — everything left of the sidebar."""
        return max(MIN_LEFT_PANEL_W, self.win_w - self.sidebar_w)

    def divider_x(self) -> int:
        """The x of the draggable divider between the left column and the sidebar."""
        return self.panel_w()

    def image_area_h(self) -> int:
        """Height of the image area (the bottom panel sits below it)."""
        return max(0, self.win_h - self.panel_h)

    def sidebar_rect(self) -> pygame.Rect:
        x = self.panel_w() + DIVIDER_W
        return pygame.Rect(x, 0, max(0, self.win_w - x), self.win_h)

    def bottom_panel_rect(self) -> pygame.Rect:
        return pygame.Rect(0, self.image_area_h(), self.panel_w(), self.panel_h)

    def image_region(self) -> pygame.Rect:
        """The screen area the canvas viewport occupies (above the panel, left of the sidebar)."""
        return pygame.Rect(0, 0, self.panel_w(), self.image_area_h())

    def window_size(self) -> tuple[int, int]:
        return (self.win_w, self.win_h)

    def centered_rect(self, w: int, h: int) -> pygame.Rect:
        """A ``w``x``h`` rect centred in the window (for modal dialogs)."""
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (self.win_w // 2, self.win_h // 2)
        return rect

    def set_panel_height(self, height: int) -> bool:
        """Set the bottom-panel height (clamped). Returns True if it changed."""
        max_h = max(PANEL_MIN, self.win_h - MIN_IMAGE_H)
        new_h = max(PANEL_MIN, min(max_h, int(height)))
        changed = new_h != self.panel_h
        self.panel_h = new_h
        return changed

    def set_sidebar_width(self, width: int) -> bool:
        """Set the sidebar width (clamped). Returns True if it changed."""
        max_w = max(SIDEBAR_W_MIN, min(SIDEBAR_W_MAX, self.win_w - MIN_LEFT_PANEL_W - DIVIDER_W))
        new_w = max(SIDEBAR_W_MIN, min(max_w, int(width)))
        changed = new_w != self.sidebar_w
        self.sidebar_w = new_w
        return changed

    def resize(self, w: int, h: int) -> bool:
        """Adopt a new window size, keeping the sidebar + panel within it. False if unchanged."""
        if (w, h) == (self.win_w, self.win_h):
            return False
        self.win_w, self.win_h = w, h
        self.sidebar_w = max(
            SIDEBAR_W_MIN, min(self.sidebar_w, self.win_w - MIN_LEFT_PANEL_W - DIVIDER_W)
        )
        self.panel_h = max(PANEL_MIN, min(self.panel_h, self.win_h - MIN_IMAGE_H))
        return True


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
