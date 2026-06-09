"""Save/load the whole working state as a ``.zip`` (settings + result + scrubbable history).

:class:`SessionIO` is the App's session-persistence controller — it captures the current state
into a :class:`~.presets.Session` (+ the rendered result + the edit history), and on load applies
it back: settings, the result as the init image / static final frame, and the restored timeline.
It takes the App and reaches its pieces through it; the App just holds one and forwards Save/Load.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame
from PIL import Image

from .controls import PromptRow
from .presets import HistoryItem, Session, load_session, save_session
from .util import surface_to_pil
from .worker import HistoryEntry, PromptSpec

if TYPE_CHECKING:
    from .app import App

log = logging.getLogger("disco_diffusion_studio.session_io")


class SessionIO:
    """Captures / restores the whole working state as a session ``.zip``."""

    def __init__(self, app: App) -> None:
        self.app = app

    def save(self) -> None:
        """Save the whole working state + result + history to a .zip via the native Save dialog."""
        app = self.app
        path = app._native_path("save", "Save session")
        if not path:
            return
        try:
            saved = save_session(
                path, self._current_session(), self._current_image(), self._history_for_save()
            )
        except Exception as exc:  # noqa: BLE001 - surface the failure instead of crashing
            log.exception("saving session failed")
            app._status(f"Save failed: {exc}")
            return
        app._status(f"Saved session {saved.name}")

    def load(self) -> None:
        """Load a session .zip via the native Open dialog and apply it (result -> init image)."""
        app = self.app
        path = app._native_path("open", "Open session")
        if not path:
            return
        if not zipfile.is_zipfile(path):  # an image (or other file) picked via the session button
            app._show_message(
                "Not a session",
                f"<b>{Path(path).name}</b> isn't a session bundle (.zip)."
                "<br><br>If it's an image, use the <b>Init image → Open…</b> button to load it.",
            )
            return
        try:
            session, image, history = load_session(path)
        except Exception as exc:  # noqa: BLE001 - bad/old file shouldn't crash the app
            log.exception("loading session failed")
            app._show_message("Bad session", f"Couldn't load <b>{Path(path).name}</b>:<br>{exc}")
            return
        self._apply_session(session)
        # The bundled result becomes the init image so Play continues from it; without one, clear.
        # It's also painted onto the canvas as the (static) final frame, so the timeline treats it
        # as the rightmost endpoint — scrubbing visits it just like a freshly finished run's last
        # step, rather than snapping back to the last recorded checkpoint.
        if image is not None:
            app._set_init_image(image, "session result")
            arr = np.asarray(image.convert("RGB"))  # (H, W, 3)
            app.canvas.frame_surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        else:
            app._clear_init()
            app.canvas.frame_surface = None
        # Restore the scrubbable timeline (previews only, latent=None) — _sync_history keeps it
        # while there's no worker, and Revert continues from a checkpoint's preview via img2img.
        app._timeline.load_entries(
            [
                HistoryEntry(
                    latent=None,
                    step=item.step,
                    index=item.index,
                    total=item.total,
                    preview=np.asarray(preview),
                    label=item.label,
                    prompts=[PromptSpec(t, w, m) for t, w, m in item.prompts],
                    config=item.config,
                )
                for item, preview in history
            ]
        )
        app._sync_enabled()

    def _current_session(self) -> Session:
        """Capture the whole working state (prompts + output + denoise + recipe) as a Session."""
        app = self.app
        recipe = app.recipe.current()
        return Session(
            width=app.width,
            height=app.height,
            steps=app.steps,
            seed=app.generation.seed_for_run(),  # also fills the field with the seed in use
            denoise=app._init.denoise,
            prompts=[PromptSpec(r.text, r.weight, r.muted) for r in app.prompts],
            config=recipe.config,
            clip_models=recipe.clip_models,
            use_secondary_model=recipe.use_secondary_model,
        )

    def _current_image(self) -> Image.Image | None:
        """The rendered result currently on the canvas as a PIL image (for the session zip)."""
        surface = self.app.history.displayed_surface()
        return surface_to_pil(surface) if surface is not None else None

    def _history_for_save(self) -> list[tuple[HistoryItem, Image.Image]]:
        """The current edit history as (metadata, preview image) pairs for the session zip."""
        return [
            (
                HistoryItem(
                    label=e.label,
                    step=e.step,
                    index=e.index,
                    total=e.total,
                    prompts=[PromptSpec(t, w, m) for t, w, m in e.prompts],
                    config=e.config,
                ),
                Image.fromarray(e.preview),
            )
            for e in self.app._timeline.entries
        ]

    def _apply_session(self, session: Session) -> None:
        """Adopt a loaded session: stop the run, set output + prompts + the recipe for next Play."""
        app = self.app
        app.generation.apply_size(
            session.width, session.height
        )  # also stops the run + rebuilds the canvas
        app.steps = session.steps
        app.sidebar.set_steps_text(str(app.steps))
        app._seed_text = str(session.seed)
        app.sidebar.set_seed_text(app._seed_text)
        app._init.denoise = session.denoise
        app.sidebar.set_denoise(app._init.denoise)
        app.prompts = [PromptRow(t, w, m) for t, w, m in session.prompts] or [PromptRow("", 1.0)]
        app.bottom_bar.rebuild_prompt_rows(app)
        app.recipe.apply_recipe(session.config, session.clip_models, session.use_secondary_model)
        app.recipe.selection = app.recipe.detect()
        app.sidebar.spawn_preset_dropdown(app)
        app.generation.run_snapshot = app.sidebar.perrun_values(app)
        app._status("Session loaded — press Play")
