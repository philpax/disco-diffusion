"""The scrubbable timeline: snap/total/slider maths, preview cursor, and tick geometry."""

from __future__ import annotations

import numpy as np
import pygame

from disco_diffusion_studio.engine.worker import HistoryEntry
from disco_diffusion_studio.session.timeline import Timeline


def _entry(index: int, total: int = 100, label: str = "edit") -> HistoryEntry:
    img = np.zeros((2, 2, 3), np.uint8)
    return HistoryEntry(latent=None, step=0, index=index, total=total, preview=img, label=label)


def test_total_uses_entries_else_live():
    assert Timeline().total(80) == 80  # no entries -> the live total
    assert Timeline(entries=[_entry(10, total=250)]).total(1) == 250  # entries win


def test_snap_picks_nearest_checkpoint_else_live():
    tl = Timeline(entries=[_entry(2), _entry(60)])
    assert tl.snap(3.0, live_index=95) == 0
    assert tl.snap(58.0, live_index=95) == 1
    assert tl.snap(95.0, live_index=95) is None  # the live (rightmost) point


def test_slider_start_prefers_the_previewed_step():
    tl = Timeline(entries=[_entry(2), _entry(60)], preview_index=1)
    assert tl.slider_start(95) == 60.0
    tl.preview_index = None
    assert tl.slider_start(95) == 95.0  # falls back to the live step


def test_previewing_and_preview_entry():
    tl = Timeline(entries=[_entry(2, label="a")])
    assert not tl.previewing() and tl.preview_entry() is None
    tl.preview_index = 0
    assert tl.previewing()
    entry = tl.preview_entry()
    assert entry is not None and entry.label == "a"
    tl.preview_index = 5  # out of range (e.g. history shrank under it)
    assert not tl.previewing() and tl.preview_entry() is None


def test_tick_x_maps_a_step_onto_the_track():
    track = pygame.Rect(0, 0, 120, 10)
    assert Timeline.tick_x(0, track, button_w=20, total=100) == 10  # left + button/2
    assert Timeline.tick_x(100, track, button_w=20, total=100) == 110  # ...+ full span (120-20)


def test_brighten_lerps_toward_white():
    assert Timeline.brighten((0, 0, 0), 0.5) == (127, 127, 127)
    assert Timeline.brighten((255, 255, 255)) == (255, 255, 255)
