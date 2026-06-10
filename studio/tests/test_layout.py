"""Pure layout maths: dimension snapping and the flow-layout packing."""

from __future__ import annotations

from disco_diffusion_studio.common.layout import MAX_SIDE, MIN_SIDE, Row, Stack, snap_side


def test_snap_side_snaps_to_multiples_of_64():
    assert snap_side(100) == 128  # 100/64 ≈ 1.56 -> 2 -> 128 (UNet needs multiples of 64)
    assert snap_side(95) == 64  # 95/64 ≈ 1.48 -> 1 -> 64
    assert snap_side(200) == 192  # 200/64 ≈ 3.1 -> 3 -> 192
    assert snap_side(512) == 512  # already a multiple


def test_snap_side_clamps_to_bounds():
    assert snap_side(0) == MIN_SIDE
    assert snap_side(-50) == MIN_SIDE
    assert snap_side(999999) == MAX_SIDE
    assert all(snap_side(v) % 64 == 0 for v in (1, 100, 777, 5000))


def test_row_packs_from_both_ends_then_fills_the_gap():
    row = Row(0, 0, 100, 32, pad=10)
    left = row.left(20)
    assert (left.x, left.width) == (0, 20)
    right = row.right(20)
    assert (right.x, right.width) == (80, 20)  # taken from the right edge
    fill = row.fill()
    # left cursor at 20+10=30; right cursor at 80-10=70 -> the gap between them
    assert (fill.x, fill.width) == (30, 40)
    assert fill.height == 32


def test_row_fill_has_a_minimum_width():
    row = Row(0, 0, 100, 32)
    row.left(100)  # consume the whole band
    assert row.fill().width == 10  # never collapses to zero


def test_stack_advances_y_by_height_plus_pad():
    stack = Stack(5, 10, 200, pad=10)
    first = stack.row(32)
    assert (first.y, first.h) == (10, 32)
    second = stack.row(24)
    assert second.y == 10 + 32 + 10  # previous height + pad
