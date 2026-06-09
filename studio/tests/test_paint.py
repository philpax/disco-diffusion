"""PaintLayer compositing: brush dabs, alpha-over blending, snapshots, and the overlay surface."""

from __future__ import annotations

import numpy as np

from disco_diffusion_studio.paint import BRUSHES, PaintLayer


def test_layer_starts_empty_and_clears():
    layer = PaintLayer(16, 16)
    assert layer.empty()
    layer.stamp(8, 8, 4, (1.0, 0.0, 0.0), 1.0, "Hard")
    assert not layer.empty()
    layer.clear()
    assert layer.empty()


def test_hard_brush_fills_a_disc_with_the_colour():
    layer = PaintLayer(16, 16)
    layer.stamp(8, 8, 4, (1.0, 0.0, 0.0), 1.0, "Hard")
    assert layer.alpha[8, 8] == 1.0  # centre fully painted
    assert tuple(layer.rgb[8, 8]) == (1.0, 0.0, 0.0)
    assert layer.alpha[0, 0] == 0.0  # a corner outside the radius is untouched


def test_soft_brush_falls_off_with_distance():
    layer = PaintLayer(32, 32)
    layer.stamp(16, 16, 10, (1.0, 1.0, 1.0), 1.0, "Soft")
    centre = layer.alpha[16, 16]
    edge = layer.alpha[16, 24]  # ~8px from centre, still within radius 10
    assert centre > edge > 0.0  # the soft brush is strongest at the centre


def test_strength_scales_alpha():
    layer = PaintLayer(16, 16)
    layer.stamp(8, 8, 4, (1.0, 1.0, 1.0), 0.5, "Hard")
    assert layer.alpha[8, 8] == 0.5


def test_snapshot_is_an_independent_copy():
    layer = PaintLayer(16, 16)
    layer.stamp(8, 8, 4, (1.0, 1.0, 1.0), 1.0, "Hard")
    _rgb, alpha, _tint = layer.snapshot()
    layer.clear()  # mutate the layer after snapshotting
    assert alpha[8, 8] == 1.0  # the snapshot is unaffected


def test_tint_marks_painted_pixels():
    layer = PaintLayer(16, 16)
    layer.stamp(8, 8, 4, (1.0, 1.0, 1.0), 1.0, "Hard", tint=1.0)
    assert layer.tint[8, 8] == 1.0


def test_stamp_outside_bounds_is_a_noop():
    layer = PaintLayer(16, 16)
    layer.stamp(-100, -100, 4, (1.0, 1.0, 1.0), 1.0, "Hard")
    assert layer.empty()


def test_spray_is_a_sparse_scatter_within_the_disc():
    np.random.seed(0)
    spray = PaintLayer(40, 40)
    spray.stamp(20, 20, 15, (1.0, 1.0, 1.0), 1.0, "Spray")
    hard = PaintLayer(40, 40)
    hard.stamp(20, 20, 15, (1.0, 1.0, 1.0), 1.0, "Hard")
    assert 0 < int((spray.alpha > 0).sum()) < int((hard.alpha > 0).sum())


def test_all_brushes_stamp_without_error():
    np.random.seed(0)
    for brush in BRUSHES:
        PaintLayer(16, 16).stamp(8, 8, 6, (1.0, 1.0, 1.0), 1.0, brush)


def test_to_surface_matches_layer_size():
    layer = PaintLayer(24, 12)
    assert layer.to_surface().get_size() == (24, 12)
