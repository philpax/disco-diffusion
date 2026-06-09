"""App-level behaviour with a fake session: construction, history math, modal isolation,
preset/colour wiring, panel resize, geometry, and the loading screen."""

from __future__ import annotations

import zipfile
from types import SimpleNamespace

import numpy as np
import pygame
import pygame_gui
from PIL import Image

from disco_diffusion_studio import app as A
from disco_diffusion_studio.presets import GuidanceSnapshot
from disco_diffusion_studio.timeline import Timeline
from disco_diffusion_studio.worker import HistoryEntry, PromptSpec


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
    slider = next(iter(app.sidebar._scale_sliders))
    app._handle_event(
        pygame.event.Event(pygame_gui.UI_HORIZONTAL_SLIDER_MOVED, ui_element=slider, value=9999.0)
    )
    assert app._preset_selection == A.CUSTOM_PRESET
    assert app._guidance_checkpoint_at is not None


def test_apply_preset_sets_guidance_and_requests_checkpoint(app, fake_worker):
    calls: list[str] = []
    app.worker = fake_worker(checkpoint=calls.append)
    app._apply_preset("2022 sauce")
    assert app.session.config.clip_guidance_scale == 15000
    assert calls == ["preset 2022 sauce"]


def test_history_snap_picks_nearest_checkpoint(app, fake_worker):
    img = np.zeros((4, 4, 3), np.uint8)
    app._timeline.entries = [
        HistoryEntry(latent=None, step=0, index=2, total=100, preview=img, label="start"),
        HistoryEntry(latent=None, step=0, index=60, total=100, preview=img, label="paint"),
    ]
    # Live is further along than the last checkpoint, as it is mid-run.
    app.worker = fake_worker(latest_frame=lambda: SimpleNamespace(index=95, total=100))
    assert app._timeline.snap(3.0, app._live_index()) == 0
    assert app._timeline.snap(58.0, app._live_index()) == 1
    assert app._timeline.snap(95.0, app._live_index()) is None  # rightmost == live


def test_revert_restores_guidance_and_eta(app, fake_worker):
    img = np.zeros((4, 4, 3), np.uint8)
    app._timeline.entries = [
        HistoryEntry(
            latent=None,
            step=0,
            index=5,
            total=100,
            preview=img,
            label="start",
            prompts=[PromptSpec(text="a prompt", weight=1.0, muted=False)],
            config=GuidanceSnapshot(clip_guidance_scale=5000, eta=0.8),
        )
    ]
    app._timeline.hist_len = 1
    app._timeline.preview_index = 0
    app.worker = fake_worker()
    app.session.config.clip_guidance_scale = 20000  # diverge from the checkpoint
    app.session.config.eta = 0.2
    revert = pygame.event.Event(
        pygame_gui.UI_BUTTON_PRESSED, ui_element=app.bottom_bar.revert_button
    )
    app._handle_event(revert)
    assert app.session.config.clip_guidance_scale == 5000
    assert app.session.config.eta == 0.8  # eta restored, not just the live guidance scales


def test_modal_blocks_canvas_painting(app):
    app._open_colour_picker()
    assert app._modal_open()
    app._handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=app._image_region().center)
    )
    assert app.canvas.paint.painting is False
    assert app.canvas.paint.layer.empty()


def test_picked_colour_is_remembered(app):
    before = len(app._palette.recent)
    app._apply_picked_colour((1, 2, 3))
    assert app.brush.color == (1, 2, 3)
    assert (1, 2, 3) in app._palette.recent and len(app._palette.recent) == before + 1
    # a palette colour is already shown, so it doesn't earn a recents slot
    after_custom = len(app._palette.recent)
    app._apply_picked_colour(app._palette.fixed[0])
    assert len(app._palette.recent) == after_custom


def test_mute_button_toggles_and_excludes_from_snapshot(app):
    app.prompts = [A.PromptRow("a", 1.0), A.PromptRow("b", 1.0)]
    app._rebuild_prompt_rows()
    mute_btn = next(b for b, i in app.bottom_bar._mute_buttons.items() if i == 1)
    app._handle_event(pygame.event.Event(pygame_gui.UI_BUTTON_PRESSED, ui_element=mute_btn))
    assert app.prompts[1].muted is True
    # the snapshot carries the muted flag (so the worker excludes it but checkpoints keep it)
    assert app._prompt_snapshot() == [
        PromptSpec(text="a", weight=1.0, muted=False),
        PromptSpec(text="b", weight=1.0, muted=True),
    ]
    app._handle_event(pygame.event.Event(pygame_gui.UI_BUTTON_PRESSED, ui_element=mute_btn))
    assert app.prompts[1].muted is False


def test_seed_for_run_random_fills_field(app):
    app.sidebar.seed_entry.set_text("")
    seed = app._seed_for_run()
    assert 0 <= seed < 2**31
    assert app.sidebar.seed_entry.get_text() == str(seed)  # the used seed is shown (reproducible)


def test_seed_field_is_populated_on_startup(app):
    assert app.sidebar.seed_entry.get_text().isdigit()  # always shows a concrete seed, not empty


def test_seed_for_run_uses_typed_value_and_is_stable_on_replay(app):
    app.sidebar.seed_entry.set_text("12345")
    assert app._seed_for_run() == 12345
    assert app._seed_for_run() == 12345  # replaying reuses the same seed (continuity)


def test_seed_for_run_invalid_is_randomised(app):
    app.sidebar.seed_entry.set_text("not-a-number")
    seed = app._seed_for_run()
    assert app.sidebar.seed_entry.get_text() == str(seed)  # replaced with a real (random) seed


def test_random_seed_button_rerolls_to_a_new_seed(app):
    app.sidebar.seed_entry.set_text("999")
    app._handle_event(
        pygame.event.Event(pygame_gui.UI_BUTTON_PRESSED, ui_element=app.sidebar.random_seed_button)
    )
    assert app.sidebar.seed_entry.get_text().isdigit()
    assert app.sidebar.seed_entry.get_text() != "999"


def test_session_save_load_round_trip(app, tmp_path, stub_dialogs):
    archive = tmp_path / "sess.zip"
    stub_dialogs(save=archive, open=archive)
    app.canvas.frame_surface = pygame.Surface(
        (app.width, app.height)
    )  # a rendered result to bundle
    app.canvas.frame_surface.fill((20, 180, 90))
    app.prompts = [A.PromptRow("castle", 1.3, False), A.PromptRow("bg", 0.4, True)]
    app.steps = 137
    app.sidebar.seed_entry.set_text("424242")
    app._init.denoise = 35
    app.session.config.clip_guidance_scale = 9999
    app._save_session()
    assert archive.exists()
    # mutate everything, then load the session back
    app.steps = 50
    app.sidebar.seed_entry.set_text("1")
    app._init.denoise = 90
    app.session.config.clip_guidance_scale = 100
    app.prompts = [A.PromptRow("x", 1.0)]
    app._init.image = None
    app._load_session()
    assert app.steps == 137
    assert app.sidebar.seed_entry.get_text() == "424242"
    assert app._init.denoise == 35
    assert app.session.config.clip_guidance_scale == 9999
    restored = [(r.text, r.weight, r.muted) for r in app.prompts]
    assert restored == [("castle", 1.3, False), ("bg", 0.4, True)]
    assert app._init.image is not None  # the bundled result became the init image
    assert app._init.label == "session result"


def test_session_restores_scrubbable_history(app, tmp_path, stub_dialogs):
    archive = tmp_path / "s.zip"
    stub_dialogs(save=archive, open=archive)
    img = np.full((app.height, app.width, 3), 7, np.uint8)
    app._timeline.entries = [
        HistoryEntry(
            latent=None,
            step=0,
            index=1,
            total=100,
            preview=img,
            label="start",
            prompts=[PromptSpec("a", 1.0, False)],
            config=GuidanceSnapshot(clip_guidance_scale=5000),
        ),
        HistoryEntry(
            latent=None,
            step=0,
            index=40,
            total=100,
            preview=img,
            label="paint soft 48px",
            prompts=[PromptSpec("a", 1.0, False)],
            config=GuidanceSnapshot(clip_guidance_scale=9000),
        ),
    ]
    app._save_session()
    app._timeline.entries = []
    app._load_session()
    assert [e.label for e in app._timeline.entries] == ["start", "paint soft 48px"]
    assert app._timeline.entries[1].index == 40
    assert app._timeline.entries[0].latent is None  # previews only, no latent


def test_loaded_result_is_rightmost_scrubbable_endpoint(app, tmp_path, stub_dialogs):
    archive = tmp_path / "s.zip"
    stub_dialogs(save=archive, open=archive)
    img = np.full((app.height, app.width, 3), 7, np.uint8)
    app.canvas.frame_surface = pygame.Surface(
        (app.width, app.height)
    )  # a finished result on the canvas
    app.canvas.frame_surface.fill((9, 9, 9))
    app._timeline.entries = [
        HistoryEntry(latent=None, step=0, index=1, total=100, preview=img, label="start"),
        HistoryEntry(latent=None, step=0, index=40, total=100, preview=img, label="paint"),
    ]
    app._save_session()
    app.canvas.frame_surface, app._timeline.entries = None, []
    app._load_session()
    # The result is painted back as a static final frame, and is the timeline's rightmost step.
    assert app.canvas.frame_surface is not None
    assert app._live_index() == 100  # the run's last step, past the index=40 checkpoint
    live = app._live_index()
    assert app._timeline.snap(100.0, live) is None  # rightmost snaps to the result (live), no tick
    assert app._timeline.snap(1.0, live) == 0  # earlier scrubbing still reaches the checkpoints
    assert app._displayed_surface() is app.canvas.frame_surface  # at rest the crisp result shows


def test_session_loaded_via_init_button_shows_error(app, tmp_path):
    bundle = tmp_path / "looks_like.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("session.toml", "x = 1")
    app._load_init_file(str(bundle))
    assert app._message_window is not None and app._message_window.alive()
    assert app._init.image is None  # not mistaken for an image


def test_image_loaded_via_session_button_shows_error(app, tmp_path, stub_dialogs):
    image_file = tmp_path / "pic.png"
    Image.new("RGB", (8, 8)).save(image_file)
    stub_dialogs(open=image_file)
    before = app.steps
    app._load_session()
    assert app._message_window is not None and app._message_window.alive()
    assert app.steps == before  # the session wasn't applied


def test_guidance_snapshot_keeps_int_knobs_int(app):
    # The typed snapshot coerces int knobs back to int even if loaded from floats in JSON;
    # cutn_batches is used as range(cutn_batches), which a float breaks.
    snap = GuidanceSnapshot(cutn_batches=4.0, clip_guidance_scale=5000.0, tv_scale=1.5)
    snap.apply_to(app.session.config)
    cfg = app.session.config
    assert cfg.cutn_batches == 4 and isinstance(cfg.cutn_batches, int)
    assert isinstance(cfg.clip_guidance_scale, int)
    assert cfg.tv_scale == 1.5  # genuine float knobs stay floats
    assert list(range(cfg.cutn_batches)) == [0, 1, 2, 3]


def test_loaded_revert_continues_via_img2img(app, monkeypatch):
    started = []
    monkeypatch.setattr(app, "_start_run", lambda: started.append(True))
    img = np.full((app.height, app.width, 3), 5, np.uint8)
    app.worker = None
    app._timeline.entries = [
        HistoryEntry(
            latent=None,
            step=0,
            index=10,
            total=100,
            preview=img,
            label="prompt",
            prompts=[PromptSpec("castle", 1.0, False)],
            config=GuidanceSnapshot(clip_guidance_scale=3333),
        ),
    ]
    app._timeline.preview_index = 0
    app._do_revert()  # worker is None -> img2img from the checkpoint preview
    assert started == [True]
    assert app._init.image is not None
    assert app._init.label == "history: prompt"
    assert app.session.config.clip_guidance_scale == 3333
    assert [(r.text, r.weight, r.muted) for r in app.prompts] == [("castle", 1.0, False)]


def test_panel_height_clamps(app):
    app._set_panel_height(10**6)
    assert app._image_area_h() >= A.MIN_IMAGE_H
    app._set_panel_height(0)
    assert app.layout.panel_h == A.PANEL_MIN


def test_history_tick_colours_by_kind():
    from disco_diffusion_studio.theme import MUTED_COLOR, PENDING_COLOR

    paint = Timeline.tick_colour("paint soft 48px")
    assert paint[1] > paint[0] and paint[1] > paint[2]  # green-dominant
    assert Timeline.tick_colour("guidance") == PENDING_COLOR
    assert Timeline.tick_colour("preset 2022 sauce") != MUTED_COLOR
    assert Timeline.tick_colour("start") == MUTED_COLOR


def test_loading_screen_returns_true_when_done(app):
    state = A._LoadingState(status="CLIP RN50", done=True)
    assert A._loading_screen(app.screen, state) is True


def test_loading_screen_returns_false_on_quit(app):
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    state = A._LoadingState(status="diffusion model")
    assert A._loading_screen(app.screen, state) is False


def _key(key, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


def test_ctrl_s_saves_via_native_dialog(app, tmp_path, stub_dialogs):
    out = tmp_path / "saved.png"
    stub_dialogs(save=out)
    app.canvas.frame_surface = pygame.Surface((app.width, app.height))
    app._handle_event(_key(pygame.K_s, pygame.KMOD_CTRL))
    assert out.exists()  # the frame was written to the path the native dialog returned


def test_save_image_reports_when_no_backend(app, monkeypatch):
    def _unavailable(**kw):
        raise A.native_dialog.Unavailable("no backend")

    monkeypatch.setattr(A.native_dialog, "save_file", _unavailable)
    app.canvas.frame_surface = pygame.Surface((app.width, app.height))
    app._save_image()
    assert "native dialog" in app.bottom_bar.status_label.text.lower()


def test_bracket_keys_change_brush_size(app):
    app.brush.size = 48.0
    app._handle_event(_key(pygame.K_RIGHTBRACKET))
    assert app.brush.size > 48.0
    app.brush.size = 48.0
    app._handle_event(_key(pygame.K_LEFTBRACKET))
    assert app.brush.size < 48.0


def test_digit_selects_palette_colour(app):
    app._handle_event(_key(pygame.K_3))
    assert app.brush.color == app._palette.swatches()[2]


def test_ctrl_z_reverts_to_latest_checkpoint(app, fake_worker):
    img = np.zeros((4, 4, 3), np.uint8)
    app._timeline.entries = [
        HistoryEntry(
            latent=None,
            step=0,
            index=5,
            total=100,
            preview=img,
            label="start",
            prompts=[PromptSpec(text="p", weight=1.0, muted=False)],
            config=GuidanceSnapshot(clip_guidance_scale=5000),
        )
    ]
    app._timeline.hist_len = 1
    seeked = []
    app.worker = fake_worker(finished=True, seek=seeked.append)
    app.paused = True  # not running -> revert is allowed
    app.session.config.clip_guidance_scale = 20000
    app._handle_event(_key(pygame.K_z, pygame.KMOD_CTRL))
    assert seeked == [0]  # reverted to the latest checkpoint (undo last edit)
    assert app.session.config.clip_guidance_scale == 5000


def test_ctrl_z_walks_back_through_history(app, fake_worker):
    img = np.zeros((4, 4, 3), np.uint8)
    app._timeline.entries = [
        HistoryEntry(
            latent=None,
            step=0,
            index=2,
            total=100,
            preview=img,
            label="start",
            prompts=[PromptSpec(text="p", weight=1.0, muted=False)],
            config=GuidanceSnapshot(),
        ),
        HistoryEntry(
            latent=None,
            step=0,
            index=20,
            total=100,
            preview=img,
            label="prompt",
            prompts=[PromptSpec(text="p", weight=1.0, muted=False)],
            config=GuidanceSnapshot(),
        ),
    ]
    app._timeline.hist_len = 2
    seeked = []

    def seek(i):
        seeked.append(i)
        del app._timeline.entries[i + 1 :]  # mimic the worker's branch-truncation on seek

    app.worker = fake_worker(finished=True, seek=seek)
    app.paused = True
    for _ in range(3):
        app._handle_event(_key(pygame.K_z, pygame.KMOD_CTRL))
    assert seeked == [1, 0, 0]  # latest -> earlier -> clamped at the first checkpoint
