"""PaintLayer compositing + the PaintController stroke lifecycle (flush to worker, overlay sync)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from disco_diffusion_studio.paint import BRUSHES, Brush, PaintController, PaintLayer


def _brush(color=(255, 0, 0), noise=False) -> Brush:
    return Brush(type="Hard", size=8.0, strength=1.0, color=color, noise=noise)


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


# -- PaintController --


def test_controller_paint_to_marks_the_layer():
    pc = PaintController.for_canvas(16, 16)
    assert pc.layer.empty()
    pc.begin()
    pc.paint_to((8, 8), _brush())
    assert not pc.layer.empty()
    assert pc.last_gen == (8, 8)


def test_flush_is_a_noop_without_a_live_worker(fake_worker):
    pc = PaintController.for_canvas(16, 16)
    pc.flush(None, _brush())  # empty layer + no worker
    assert pc.pending_overlays == []
    pc.paint_to((8, 8), _brush())
    pc.flush(fake_worker(is_alive=lambda: False), _brush())  # painted, but the worker is dead
    assert pc.pending_overlays == [] and not pc.layer.empty()  # stroke kept on the active layer


def test_flush_hands_the_stroke_to_the_worker_and_clears_the_layer(fake_worker):
    calls: list[tuple] = []
    pc = PaintController.for_canvas(16, 16)
    pc.paint_to((8, 8), _brush())
    pc.flush(fake_worker(paint_applied_count=0, set_paint=lambda *a: calls.append(a)), _brush())
    assert len(calls) == 1  # one batch handed off
    assert len(pc.pending_overlays) == 1 and pc.submitted == 1
    assert pc.layer.empty()  # layer cleared; the overlay copy keeps the stroke visible


def test_sync_drops_overlays_once_a_frame_has_applied_them(fake_worker):
    pc = PaintController.for_canvas(16, 16)
    pc.paint_to((8, 8), _brush())
    pc.flush(fake_worker(set_paint=lambda *a: None), _brush())  # submitted == 1
    pc.sync(fake_worker(latest_frame=lambda: SimpleNamespace(paint_applied=0)))
    assert len(pc.pending_overlays) == 1  # not yet applied (0 < 1)
    pc.sync(fake_worker(latest_frame=lambda: SimpleNamespace(paint_applied=1)))
    assert pc.pending_overlays == []  # applied -> dropped


def test_resize_rebuilds_the_layer_and_clears_overlays(fake_worker):
    pc = PaintController.for_canvas(16, 16)
    pc.paint_to((8, 8), _brush())
    pc.flush(fake_worker(set_paint=lambda *a: None), _brush())
    pc.resize(8, 8)
    assert (pc.layer.w, pc.layer.h) == (8, 8)
    assert pc.pending_overlays == [] and pc.submitted == 0
