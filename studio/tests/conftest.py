"""Shared fixtures: a headless pygame session and a sandboxed copy of the studio's TOML files.

The studio talks to a real window and reads/writes ``studio/presets/*.toml`` and
``studio/config.toml``. These fixtures run pygame against the dummy SDL driver (no window
appears) and redirect the preset/colour IO into a per-test temp copy, so tests are isolated and
never mutate the repo's files.
"""
# ruff: noqa: I001 - the SDL env hints must sit between `import os` and the pygame import, which
# deliberately splits the import block; isort can't reconcile that, so skip import-sorting here.

from __future__ import annotations

import os

# Must be set before pygame imports the video subsystem.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import shutil  # noqa: E402 - after the SDL env hints
import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pygame  # noqa: E402
import pytest  # noqa: E402

from disco_diffusion import RunConfig  # noqa: E402
from disco_diffusion_studio import colours as colours_mod  # noqa: E402
from disco_diffusion_studio import presets as presets_mod  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _pygame_session():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture(autouse=True)
def studio_sandbox(tmp_path, monkeypatch):
    """Point presets/config IO at a tmp copy of the bundled files (saves never touch the repo)."""
    dst_presets = tmp_path / "presets"
    dst_presets.mkdir()
    for toml in presets_mod.PRESETS_DIR.glob("*.toml"):
        shutil.copy(toml, dst_presets / toml.name)
    # Don't copy the repo's config.toml — it carries whatever colours were picked in the real
    # app. Point at a fresh (absent) path so load_colours falls back to the deterministic default
    # palette with empty recents, and saves land in the sandbox.
    dst_config = tmp_path / "config.toml"
    monkeypatch.setattr(presets_mod, "PRESETS_DIR", dst_presets)
    monkeypatch.setattr(colours_mod, "CONFIG_PATH", dst_config)
    return SimpleNamespace(presets_dir=dst_presets, config_path=dst_config)


@pytest.fixture
def fake_session():
    """A stand-in for DiscoSession that's enough to build the UI (config + device, no models)."""
    return SimpleNamespace(config=RunConfig(), device="cpu")


@pytest.fixture
def app(studio_sandbox, fake_session, tmp_path):
    from disco_diffusion_studio.app import App

    return App(session=fake_session, out_dir=tmp_path / "out")


@pytest.fixture
def fake_worker():
    """Factory for a stub worker; sane defaults, override any attribute via kwargs.

    The app reads only a handful of worker attributes off the UI thread, so a ``SimpleNamespace``
    stands in for a real (threaded, sampler-driving) ``GenerationWorker`` in app-level tests.
    """

    def make(**overrides):
        attrs: dict = dict(
            is_alive=lambda: True,
            finished=False,
            seek=lambda i: None,
            set_prompts=lambda p: None,
            checkpoint=lambda label: None,
            paint_applied_count=0,
            latest_frame=lambda: None,
        )
        attrs.update(overrides)
        return SimpleNamespace(**attrs)

    return make


@pytest.fixture
def worker_factory():
    """Build a real ``GenerationWorker`` over a (stub) ``session`` with the standard rigging."""
    from disco_diffusion_studio.worker import GenerationWorker

    def make(session, *, width: int = 64, height: int = 64, steps: int = 20, **kwargs):
        return GenerationWorker(
            session,
            width=width,
            height=height,
            steps=steps,
            encode_cache={},
            cache_lock=threading.Lock(),
            **kwargs,
        )

    return make


@pytest.fixture
def stub_dialogs(monkeypatch):
    """Patch the native Save/Open dialogs to return given paths (str-ified)."""
    from disco_diffusion_studio import app as app_mod

    def patch(*, save: object = None, open: object = None) -> None:
        if save is not None:
            monkeypatch.setattr(app_mod.native_dialog, "save_file", lambda **k: str(save))
        if open is not None:
            monkeypatch.setattr(app_mod.native_dialog, "open_file", lambda **k: str(open))

    return patch
