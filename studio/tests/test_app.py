"""App-level behaviour with a fake session: construction, history math, modal isolation,
preset/colour wiring, panel resize, geometry, and the loading screen."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pygame
import pygame_gui

from disco_diffusion_studio import app as A
from disco_diffusion_studio.worker import HistoryEntry


def test_app_constructs_and_renders(app):
    app.manager.update(0.016)
    app._draw()
    app.manager.draw_ui(app.screen)
    app._draw_tools()
    app._draw_history_ticks()  # no error with no history


def test_compute_window_size_respects_minimums():
    win_w, win_h = A.compute_window_size(1280, 768, A.SIDEBAR_W_DEFAULT, A.PANEL_H)
    assert win_w >= A.MIN_LEFT_PANEL_W + A.SIDEBAR_W_DEFAULT
    assert win_h >= A.MIN_IMAGE_H + A.PANEL_H


def test_opens_on_default_preset(app):
    assert app._preset_selection == "Default"


def test_editing_guidance_marks_custom_and_arms_checkpoint(app):
    slider = next(iter(app._scale_sliders))
    app._handle_event(
        pygame.event.Event(pygame_gui.UI_HORIZONTAL_SLIDER_MOVED, ui_element=slider, value=9999.0)
    )
    assert app._preset_selection == A.CUSTOM_PRESET
    assert app._guidance_checkpoint_at is not None


def test_apply_preset_sets_guidance_and_requests_checkpoint(app):
    calls: list[str] = []
    app.worker = SimpleNamespace(is_alive=lambda: True, checkpoint=calls.append)
    app._apply_preset("2022 sauce")
    assert app.session.config.clip_guidance_scale == 15000
    assert calls == ["preset 2022 sauce"]


def test_history_snap_picks_nearest_checkpoint(app):
    img = np.zeros((4, 4, 3), np.uint8)
    app._history = [
        HistoryEntry(latent=None, step=0, index=2, total=100, preview=img, label="start"),
        HistoryEntry(latent=None, step=0, index=60, total=100, preview=img, label="paint"),
    ]
    # Live is further along than the last checkpoint, as it is mid-run.
    app.worker = SimpleNamespace(latest_frame=lambda: SimpleNamespace(index=95, total=100))
    assert app._history_snap(3.0) == 0
    assert app._history_snap(58.0) == 1
    assert app._history_snap(95.0) is None  # rightmost == live


def test_modal_blocks_canvas_painting(app):
    app._open_colour_picker()
    assert app._modal_open()
    app._handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=app._image_region().center)
    )
    assert app._painting is False
    assert app._paint_layer.empty()


def test_picked_colour_is_remembered(app):
    before = len(app._recent)
    app._apply_picked_colour((1, 2, 3))
    assert app.brush_color == (1, 2, 3)
    assert (1, 2, 3) in app._recent and len(app._recent) == before + 1
    # a palette colour is already shown, so it doesn't earn a recents slot
    after_custom = len(app._recent)
    app._apply_picked_colour(app._palette[0])
    assert len(app._recent) == after_custom


def test_panel_height_clamps(app):
    app._set_panel_height(10**6)
    assert app._image_area_h() >= A.MIN_IMAGE_H
    app._set_panel_height(0)
    assert app.panel_h == A.PANEL_MIN


def test_history_tick_colours_by_kind():
    paint = A.App._history_tick_colour("paint soft 48px")
    assert paint[1] > paint[0] and paint[1] > paint[2]  # green-dominant
    assert A.App._history_tick_colour("guidance") == A.PENDING_COLOR
    assert A.App._history_tick_colour("preset 2022 sauce") != A.MUTED_COLOR
    assert A.App._history_tick_colour("start") == A.MUTED_COLOR


def test_loading_screen_returns_true_when_done(app):
    state = A._LoadingState(status="CLIP RN50", done=True)
    assert A._loading_screen(app.screen, state) is True


def test_loading_screen_returns_false_on_quit(app):
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    state = A._LoadingState(status="diffusion model")
    assert A._loading_screen(app.screen, state) is False
