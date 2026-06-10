"""The run-lifecycle controller: start / stop / pause, output size + steps, seed, and saving.

:class:`Generation` drives the background :class:`~.worker.GenerationWorker` — starting a fresh
run, stopping it, pausing/resuming, and the reconfigurations that require a fresh run (output size,
step count). It owns the per-run settings snapshot the "Current" sidebar tab shows, picks the seed,
and saves the rendered frame. ``worker`` / ``paused`` stay on the App (the run loop reads them every
frame); this mutates them through the App, reaching its other pieces (canvas, sidebar, timeline,
init image) the same way. The App holds one and forwards the transport buttons to it.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from .layout import snap_side
from .signals import Signals
from .state import PaintState, SharedState
from .util import clamp_steps
from .worker import GenerationWorker

if TYPE_CHECKING:
    from .app import App

log = logging.getLogger("disco_diffusion_studio.generation")


class Generation:
    """Starting, stopping, and reconfiguring the generation run (+ saving its output)."""

    def __init__(self, app: App, signals: Signals, state: SharedState, paint: PaintState) -> None:
        self.app = app
        self.signals = signals
        self.state = state
        self.paint = paint

    def seed_for_run(self) -> int:
        """Seed for the next run: the typed value, or a fresh random one (then shown in the field).

        Filling the field with the seed actually used makes every run reproducible and visible.
        """
        app = self.app
        try:
            seed = int(app.sidebar.seed_text().strip())
            if seed < 0:
                raise ValueError
        except ValueError:
            seed = random.randrange(2**31)  # empty / invalid -> random, then surface it
        self.state.seed_text = str(seed)
        app.sidebar.set_seed_text(self.state.seed_text)
        return seed

    def start(self) -> None:
        """Start a fresh run from the current settings (stopping any run in flight first)."""
        app = self.app
        # Adopt any step count typed into the box but not yet Enter-applied: clicking Play
        # moves focus off the box without firing UI_TEXT_ENTRY_FINISHED, so without this the
        # run would silently use the previous value.
        self.commit_steps()
        self.stop()
        self.state.worker = GenerationWorker(
            self.state.session,
            width=self.state.width,
            height=self.state.height,
            steps=self.state.steps,
            encode_cache=self.state.encode_cache,
            cache_lock=self.state.cache_lock,
            perlin=self.state.session.config.perlin_init,
            init_image=self.state.init.image,
            skip_steps=self.state.init.skip_steps(self.state.steps),
            seed=self.seed_for_run(),
        )
        self.state.worker.set_prompts(app._prompt_snapshot())
        # A fresh worker starts with paint_applied_count == 0; reset the overlay tracking to match
        # and drop stale overlays from the previous run. Any stroke painted before Play (still on
        # the active layer) is flushed as the new run's first paint batch rather than discarded.
        app.canvas.paint.reset_overlays()
        if not app.canvas.paint.layer.empty():
            app.canvas.paint.flush(self.state.worker, self.paint.brush)
        # Freeze the per-run settings this run uses, for the "Current" tab to show while it runs.
        self.state.run_snapshot = app.sidebar.perrun_values(app)
        self.state.timeline.reset()
        self.state.paused = False
        self.state.worker.start()
        self.signals.status("Running")
        self.signals.invalidate()

    def stop(self) -> None:
        """Tear down the worker and clear the timeline (no-op if nothing's running)."""
        if self.state.worker is not None:
            self.state.worker.stop()
            self.state.worker.join(timeout=5.0)
        self.state.worker = None
        self.state.paused = False
        self.state.timeline.reset()
        self.signals.invalidate()

    def toggle_play(self) -> None:
        """Play/Pause: start a fresh run, pause a running one, or resume a paused one."""
        if self.state.timeline.preview_index is not None:
            self.signals.status("Previewing")
            return
        if (
            self.state.worker is None
            or not self.state.worker.is_alive()
            or self.state.worker.finished
        ):
            self.start()  # finished -> Play starts a fresh run (Revert continues a branch)
        elif self.state.paused:
            self.state.paused = False
            self.state.timeline.clear_preview()  # resume from live, drop any history preview
            self.state.timeline.end_undo()  # resuming ends the undo chain
            self.state.worker.resume()
            self.signals.status("Running")
            self.signals.invalidate()
        else:
            self.state.paused = True
            self.state.worker.pause()
            self.signals.status("Paused")
            self.signals.invalidate()

    def apply_size(self, width: int, height: int) -> None:
        """Set the output size (snapped). A shape change needs a fresh run, so it stops the run."""
        app = self.app
        # Changing the output shape requires a fresh run. The window does NOT change — the
        # image is letterboxed into the image region — so orientation flips keep proportions.
        self.stop()
        self.state.width = snap_side(width)
        self.state.height = snap_side(height)
        app.sidebar.set_size_text(self.state.width, self.state.height)
        # Generation size changed: rebuild the init preview (also generation-res), then have the
        # canvas adopt the new size (rebuild the paint layer, drop the stale frame, refit the view).
        self.state.init.rebuild_surface(self.state.width, self.state.height)
        app.canvas.apply_size(self.state.width, self.state.height)
        self.signals.status("Size set")

    def commit_steps(self) -> None:
        """Adopt the steps box value (clamped). Safe to call on Enter, on blur, or at Play.

        Idempotent: re-committing the same value is a no-op, so calling it just before a run
        starts (to pick up a number typed but not Enter-applied) never disturbs anything.
        """
        app = self.app
        try:
            value = clamp_steps(app.sidebar.steps_text())
        except ValueError:
            app.sidebar.set_steps_text(str(self.state.steps))
            return
        app.sidebar.set_steps_text(str(value))
        if value == self.state.steps:
            return
        self.state.steps = value
        # If a run is paused, changing steps abandons it (respacing is fixed per run).
        if self.state.worker is not None and self.state.worker.is_alive():
            self.stop()
            self.signals.status("Steps set")

    def save_image(self) -> None:
        """Save the current frame via the native Save dialog (blocks while it's open)."""
        app = self.app
        surface = app.canvas.frame_for_save()  # freeze; the worker keeps generating
        if surface is None:
            self.signals.status("No frame")
            return
        path_str = app._native_path("save", "Save image")
        if not path_str:
            return  # cancelled
        path = Path(path_str)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, str(path))
        self.signals.status("Saved")
        log.info("saved %s", path)
