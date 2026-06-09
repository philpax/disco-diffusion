"""The scrubbable edit history: the checkpoint list, the scrub cursors, and the scrub maths.

The history slider scrubs over a :class:`Timeline` of checkpoints. This owns the list, the preview
(scrubbed) and undo cursors, the cached preview surface, and the *pure* maths — which step a slider
value snaps to, where the thumb starts, tick colours/positions. The worker/live-frame reads, the
revert orchestration, and the slider widget stay on the App, which passes ``live_index`` /
``live_total`` (the rightmost, in-progress step) into these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame

from .theme import MUTED_COLOR, PENDING_COLOR, READOUT_COLOR
from .worker import HistoryEntry

RGB = tuple[int, int, int]


@dataclass
class Timeline:
    """The checkpoint list + scrub cursors + the pure maths the history slider scrubs over."""

    entries: list[HistoryEntry] = field(default_factory=list)
    preview_index: int | None = None  # the scrubbed checkpoint, or None when on the live frame
    undo_cursor: int | None = None  # how far Ctrl+Z has walked back (reverts don't truncate)
    hist_len: int = 0  # last-seen len(entries), to detect growth/branch in the sync loop
    _surface: pygame.Surface | None = None  # cached render of the previewed checkpoint
    _surface_key: int | None = None

    def previewing(self) -> bool:
        """True when a checkpoint is being previewed (the cursor points into the list)."""
        return self.preview_index is not None and self.preview_index < len(self.entries)

    def preview_entry(self) -> HistoryEntry | None:
        """The previewed checkpoint, or None when on the live frame."""
        return self.entries[self.preview_index] if self.previewing() else None  # type: ignore[index]

    def preview_surface(self) -> pygame.Surface | None:
        """A cached surface of the previewed checkpoint's image, or None when on the live frame."""
        if not self.previewing():
            return None
        entry = self.entries[self.preview_index]  # type: ignore[index]  # previewing() guards it
        if self._surface_key != id(entry.preview):
            self._surface_key = id(entry.preview)
            self._surface = pygame.surfarray.make_surface(entry.preview.swapaxes(0, 1))
        return self._surface

    def total(self, live_total: int) -> int:
        """The run's display-step total (the slider's right edge)."""
        if self.entries:
            return max(1, self.entries[-1].total)
        return max(1, live_total)

    def snap(self, value: float, live_index: int) -> int | None:
        """The checkpoint index nearest a slider step ``value``; None == the live (rightmost)."""
        points: list[tuple[float, int | None]] = [
            (float(cp.index), i) for i, cp in enumerate(self.entries)
        ]
        points.append((float(live_index), None))  # the live frame
        return min(points, key=lambda p: abs(p[0] - value))[1]

    def slider_start(self, live_index: int) -> float:
        """Where the thumb sits: a previewed checkpoint's step, else the live step."""
        if self.previewing():
            return float(self.entries[self.preview_index].index)  # type: ignore[index]
        return float(live_index)

    @staticmethod
    def tick_x(value: float, track: pygame.Rect, button_w: int, total: int) -> int:
        """Screen x where the slider thumb centre sits for a step ``value`` (0..total)."""
        span = max(1, track.width - button_w)
        frac = min(max(value / float(max(total, 1)), 0.0), 1.0)
        return int(track.left + button_w / 2 + frac * span)

    @staticmethod
    def tick_colour(label: str) -> RGB:
        """Colour a checkpoint tick by kind, so the history reads at a glance."""
        if label.startswith("paint"):
            return (118, 200, 140)  # green — painted strokes
        if label.startswith("guidance"):
            return PENDING_COLOR  # amber — guidance retunes
        if label.startswith("preset"):
            return (176, 136, 240)  # violet — preset loads
        if "prompt" in label:
            return READOUT_COLOR  # blue — prompt edits (edit / add / remove)
        return MUTED_COLOR  # grey — the run's baseline ("start")

    @staticmethod
    def brighten(c: RGB, t: float = 0.55) -> RGB:
        """Lerp a colour toward white (for the active / hovered tick), keeping its hue."""
        return tuple(int(v + (255 - v) * t) for v in c)  # type: ignore[return-value]
