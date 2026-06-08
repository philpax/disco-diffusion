"""img2img: loading an init image, the denoise->skip_steps mapping, and worker forwarding."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pygame
import pygame_gui
from disco_diffusion import RunConfig
from PIL import Image

from disco_diffusion_studio.worker import GenerationWorker


def _png(tmp_path, size=(80, 50)):
    path = tmp_path / "seed.png"
    Image.new("RGB", size, (120, 30, 200)).save(path)
    return path


def test_load_init_file_sets_image_and_preview(app, tmp_path):
    app._load_init_file(str(_png(tmp_path)))
    assert app._init_image is not None
    assert app._init_label == "seed.png"
    assert app._init_surface is not None
    assert app._init_surface.get_size() == (app.width, app.height)  # resized to gen size


def test_denoise_maps_to_skip_steps(app, tmp_path):
    app._load_init_file(str(_png(tmp_path)))
    app.steps = 100
    app._init_denoise = 100
    assert app._init_skip_steps() == 0  # full re-diffusion (ignore init structure)
    app._init_denoise = 0
    assert app._init_skip_steps() == 99  # keep the init (steps - 1)
    app._init_denoise = 60
    assert app._init_skip_steps() == 40


def test_no_init_means_zero_skip(app):
    app._init_image = None
    app._init_denoise = 0
    assert app._init_skip_steps() == 0


def test_use_current_result_as_init(app):
    frame = pygame.Surface((app.width, app.height))
    frame.fill((10, 200, 50))
    app._frame_surface = frame
    app._use_current_as_init()
    assert app._init_image is not None
    assert app._init_label == "current result"


def test_denoise_slider_updates_value(app):
    app._handle_event(
        pygame.event.Event(
            pygame_gui.UI_HORIZONTAL_SLIDER_MOVED, ui_element=app._init_denoise_slider, value=25.0
        )
    )
    assert app._init_denoise == 25


def test_clear_init(app, tmp_path):
    app._load_init_file(str(_png(tmp_path)))
    app._clear_init()
    assert app._init_image is None
    assert app._init_surface is None


def test_init_dialog_is_modal(app):
    app._open_init_dialog()
    assert app._modal_open()


def test_dropfile_loads_init(app, tmp_path):
    app._handle_event(pygame.event.Event(pygame.DROPFILE, file=str(_png(tmp_path))))
    assert app._init_image is not None
    assert app._init_label == "seed.png"


def test_reset_with_nothing_to_clear_opens_no_dialog(app):
    app._open_reset_confirm()
    assert app._confirm_dialog is None


def test_reset_confirm_is_modal_then_clears_frame(app, tmp_path):
    app._load_init_file(str(_png(tmp_path)))
    app._frame_surface = pygame.Surface((app.width, app.height))  # a "rendered" frame
    app._open_reset_confirm()
    assert app._modal_open()  # confirmation is up
    app._reset_canvas()  # confirm
    assert app._frame_surface is None
    assert app._displayed_surface() is None  # so the init preview shows again


def test_worker_forwards_init_to_sampler():
    captured: dict = {}

    class _Stub:
        total = 10

        @property
        def has_output(self):
            return False

        def set_conditioning(self, items):
            pass

        def close(self):
            pass

    def sampler(**kwargs):
        captured.update(kwargs)
        return _Stub()

    session = SimpleNamespace(
        config=RunConfig(),
        diffusion_for=lambda steps: SimpleNamespace(num_timesteps=10),
        sampler=sampler,
    )
    init = Image.new("RGB", (8, 8))
    worker = GenerationWorker(
        session,
        width=64,
        height=64,
        steps=10,
        encode_cache={},
        cache_lock=threading.Lock(),
        init_image=init,
        skip_steps=7,
    )
    worker._start_sampler()  # call directly (no thread) to capture the sampler kwargs
    assert captured["init_image"] is init
    assert captured["skip_steps"] == 7
