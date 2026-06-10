"""The canvas view transform: fit, zoom-about-a-point, pan clamping, and screen<->canvas mapping."""

from __future__ import annotations

import pygame

from disco_diffusion_studio.ui.view import MAX_ZOOM, MIN_ZOOM, ViewTransform


def test_fit_centres_the_canvas_in_the_region():
    view = ViewTransform()
    region = pygame.Rect(0, 0, 200, 100)
    view.fit(region, 100, 100)  # canvas is square, region is 2:1
    assert view.zoom == 1.0  # min(200/100, 100/100) -> limited by height
    assert view.canvas_screen_rect(100, 100) == pygame.Rect(50, 0, 100, 100)  # centred in x


def test_screen_to_canvas_inverts_the_mapping():
    view = ViewTransform()
    region = pygame.Rect(0, 0, 200, 100)
    view.fit(region, 100, 100)
    assert view.screen_to_canvas((50, 0), region, 100, 100) == (0.0, 0.0)  # top-left of canvas
    assert view.screen_to_canvas((149, 99), region, 100, 100) == (99.0, 99.0)


def test_screen_to_canvas_returns_none_outside_canvas():
    view = ViewTransform()
    region = pygame.Rect(0, 0, 200, 100)
    view.fit(region, 100, 100)
    assert view.screen_to_canvas((250, 50), region, 100, 100) is None  # outside the region
    assert view.screen_to_canvas((10, 50), region, 100, 100) is None  # in region, left of canvas


def test_zoom_at_keeps_the_anchored_point_fixed():
    view = ViewTransform()
    region = pygame.Rect(0, 0, 200, 100)
    view.fit(region, 100, 100)
    anchor = (100, 50)
    before = view.screen_to_canvas(anchor, region, 100, 100)
    view.zoom_at(anchor, 2.0, region, 100, 100)
    assert view.zoom == 2.0
    after = view.screen_to_canvas(anchor, region, 100, 100)
    assert before is not None and after is not None
    assert after[0] == before[0] and after[1] == before[1]  # the point under the cursor is fixed


def test_zoom_at_clamps_to_bounds():
    view = ViewTransform()
    region = pygame.Rect(0, 0, 200, 100)
    view.fit(region, 100, 100)
    view.zoom_at((100, 50), 1000.0, region, 100, 100)
    assert view.zoom == MAX_ZOOM
    view.zoom_at((100, 50), 0.0001, region, 100, 100)
    assert view.zoom == MIN_ZOOM


def test_clamp_pan_keeps_the_canvas_centre_within_the_region():
    view = ViewTransform(zoom=1.0, pan=pygame.Vector2(9999, -9999))
    region = pygame.Rect(0, 0, 200, 100)
    view.clamp_pan(region, 100, 100)
    rect = view.canvas_screen_rect(100, 100)
    assert region.left <= rect.centerx <= region.right
    assert region.top <= rect.centery <= region.bottom
