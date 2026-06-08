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
from types import SimpleNamespace  # noqa: E402

import pygame  # noqa: E402
import pytest  # noqa: E402

from disco_diffusion import RunConfig  # noqa: E402
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
    monkeypatch.setattr(presets_mod, "CONFIG_PATH", dst_config)
    return SimpleNamespace(presets_dir=dst_presets, config_path=dst_config)


@pytest.fixture
def fake_session():
    """A stand-in for DiscoSession that's enough to build the UI (config + device, no models)."""
    return SimpleNamespace(config=RunConfig(), device="cpu")


@pytest.fixture
def app(studio_sandbox, fake_session, tmp_path):
    from disco_diffusion_studio.app import App

    return App(session=fake_session, out_dir=tmp_path / "out")
