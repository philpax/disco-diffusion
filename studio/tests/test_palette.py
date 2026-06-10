"""The brush colour palette: swatch ordering and the recents model (dedup, cap, skip-fixed)."""

from __future__ import annotations

from disco_diffusion_studio.paint.colours import MAX_RECENT
from disco_diffusion_studio.paint.palette import Palette


def test_swatches_are_fixed_then_recents_deduped():
    p = Palette(fixed=[(0, 0, 0), (1, 1, 1)], recent=[(1, 1, 1), (2, 2, 2)])
    assert p.swatches() == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]  # (1,1,1) not duplicated


def test_default_brush_prefers_fourth_else_first():
    assert Palette([(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)], []).default_brush() == (3, 3, 3)
    assert Palette([(9, 9, 9)], []).default_brush() == (9, 9, 9)


def test_remember_off_palette_colour_prepends():
    p = Palette(fixed=[(0, 0, 0)], recent=[])
    assert p.remember((5, 6, 7)) is True
    assert p.recent == [(5, 6, 7)]


def test_remember_fixed_colour_is_skipped():
    p = Palette(fixed=[(0, 0, 0)], recent=[])
    assert p.remember((0, 0, 0)) is False  # already a fixed swatch
    assert p.recent == []


def test_remember_dedupes_and_moves_to_front():
    p = Palette(fixed=[(0, 0, 0)], recent=[(1, 1, 1), (2, 2, 2)])
    p.remember((2, 2, 2))
    assert p.recent == [(2, 2, 2), (1, 1, 1)]


def test_remember_caps_recents():
    p = Palette(fixed=[(0, 0, 0)], recent=[])
    for i in range(MAX_RECENT + 3):
        p.remember((i, i, i))
    assert len(p.recent) == MAX_RECENT
    assert p.recent[0] == (MAX_RECENT + 2,) * 3  # most-recent-first


def test_load_falls_back_to_seed_palette_in_sandbox():
    assert len(Palette.load().fixed) >= 1  # no config.toml in the sandbox -> default palette
