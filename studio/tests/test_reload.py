"""The background model reloader: debounce timing and the threaded start/poll handshake."""

from __future__ import annotations

import time

from disco_diffusion import RunConfig

from disco_diffusion_studio import reload as reload_mod
from disco_diffusion_studio.reload import ModelReloader


def _wait_for_result(reloader: ModelReloader, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = reloader.poll()
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("reload did not finish in time")


def test_schedule_cancel_and_due():
    reloader = ModelReloader()
    assert not reloader.queued
    reloader.schedule(1000)
    assert reloader.queued
    assert not reloader.due(999)  # before the fire time
    assert reloader.due(1000)  # at/after the fire time
    reloader.cancel()
    assert not reloader.queued and not reloader.due(5000)


def test_poll_is_none_when_idle():
    assert ModelReloader().poll() is None  # nothing running


def test_start_runs_in_background_and_yields_the_session(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(reload_mod, "DiscoSession", lambda cfg, device=None: sentinel)
    reloader = ModelReloader()
    reloader.schedule(1000)  # a pending reload...
    reloader.start(RunConfig(), None)
    assert reloader.reloading
    assert not reloader.queued  # ...is cleared by start
    assert not reloader.due(9999)  # never fires a second reload while one is running
    assert _wait_for_result(reloader) == {"session": sentinel}
    assert not reloader.reloading
    assert reloader.poll() is None  # the result is consumed exactly once


def test_start_captures_errors(monkeypatch):
    def boom(cfg, device=None):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(reload_mod, "DiscoSession", boom)
    reloader = ModelReloader()
    reloader.start(RunConfig(), None)
    result = _wait_for_result(reloader)
    assert "error" in result and "weights missing" in result["error"]
