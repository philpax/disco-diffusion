"""The model-set controller: staging the CLIP set / secondary toggle and the debounced reload.

Changing which CLIP models (or the secondary model) guide the run needs a full weight reload, which
takes ~a minute. :class:`Models` owns the *staged* selection (what the toggles show) and the
:class:`~.reload.ModelReloader`: a toggle stages the change and queues a debounced auto-reload that
fires once the user stops fiddling (and un-queues if they land back on the loaded set). The reload
runs off the UI thread; :meth:`poll` swaps the new session in when it's done. It reaches the App's
other pieces (session, sidebar, status) through the App; the App holds one and forwards to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
import pygame_gui
from disco_diffusion.config import AVAILABLE_CLIP_MODELS

from .constants import RELOAD_DEBOUNCE_MS
from .reload import ModelReloader
from .signals import Signals
from .state import SharedState

if TYPE_CHECKING:
    from .app import App


class Models:
    """The staged CLIP set + secondary toggle, and the debounced background weight reload."""

    def __init__(self, app: App, signals: Signals, state: SharedState) -> None:
        self.app = app
        self.signals = signals
        self.state = state
        # The staged selection (state.clip_selected / secondary_on) is shared session state;
        # changing it queues a debounced auto-reload that un-queues if it returns to the loaded set.
        self.reloader = ModelReloader()  # background weight reload + debounced trigger

    @property
    def reloading(self) -> bool:
        """True while a background weight reload is in flight (Play stays disabled)."""
        return self.reloader.reloading

    def toggle_clip(self, button: pygame_gui.elements.UIButton) -> None:
        """Stage a CLIP model in/out of the pending selection (queues the auto-reload)."""
        name = self.app.sidebar.model_name(button)
        self.state.clip_selected.symmetric_difference_update({name})  # toggle in/out
        self.app.sidebar.sync_model_buttons(self.state.clip_selected, self.state.secondary_on)
        self.update_queue()
        self.signals.edited()

    def toggle_secondary(self) -> None:
        """Stage the secondary model in/out of the pending selection (queues the auto-reload)."""
        self.state.secondary_on = not self.state.secondary_on
        self.app.sidebar.sync_model_buttons(self.state.clip_selected, self.state.secondary_on)
        self.update_queue()
        self.signals.edited()

    def matches_session(self) -> bool:
        """True when the staged CLIP set + secondary toggle equal the loaded session's."""
        cfg = self.state.session.config
        return set(self.state.clip_selected) == set(cfg.clip_models) and (
            self.state.secondary_on == cfg.use_secondary_model
        )

    def update_queue(self) -> None:
        """Queue (or cancel) the debounced auto-reload after a model toggle / preset load.

        Changing the CLIP set or secondary toggle needs a full weight reload. Rather than a
        button, we queue the reload to fire shortly after the user stops changing things, and
        cancel it if they land back on the currently-loaded set.
        """
        if self.matches_session():
            if self.reloader.queued:
                self.reloader.cancel()
                self.signals.status("Reload cancelled")
        else:
            self.reloader.schedule(pygame.time.get_ticks() + RELOAD_DEBOUNCE_MS)
            self.signals.status("Reload queued")

    def tick_reload(self, now: int) -> None:
        """Fire the debounced auto-reload once its delay has elapsed (called per frame)."""
        if self.reloader.due(now):
            self.reloader.cancel()  # clear the debounce; start_reload re-validates the change
            self.start_reload()

    def start_reload(self) -> None:
        """Rebuild the session with the staged CLIP set / secondary toggle on a worker thread.

        Reloading weights takes ~a minute, so it runs off the UI thread; the run loop polls
        :meth:`poll` and swaps the session in when it's done. Play stays disabled and the staged
        selection is locked (via the enablement re-sync) until then.
        """
        app = self.app
        if self.reloader.reloading:
            return
        selected = [m for m in AVAILABLE_CLIP_MODELS if m in self.state.clip_selected]
        if not selected:
            self.signals.status("Pick a model")
            return
        cfg = self.state.session.config
        # Order is irrelevant for the CLIP set (guidance sums over all models), so compare as
        # sets — otherwise a reselection in a different order would look like a change.
        if (
            set(selected) == set(cfg.clip_models)
            and self.state.secondary_on == cfg.use_secondary_model
        ):
            self.signals.status("No change")
            return
        app.generation.stop()  # the worker holds the old session; tear it down first
        new_cfg = cfg.model_copy(
            update={"clip_models": selected, "use_secondary_model": self.state.secondary_on}
        )
        self.reloader.start(new_cfg, self.state.session.device)
        self.signals.status("Reloading…")
        self.signals.invalidate()

    def poll(self) -> None:
        """Swap in a reloaded session once the background reload finishes (called per frame)."""
        app = self.app
        result = self.reloader.poll()
        if result is None:
            return
        if result and "session" in result:
            self.state.session = result["session"]  # type: ignore[assignment]
            self.state.encode_cache.clear()  # embeds came from the old CLIP set — now stale
            self.state.clip_selected = set(self.state.session.config.clip_models)
            self.state.secondary_on = self.state.session.config.use_secondary_model
            self.signals.status("Reloaded")
        else:
            self.signals.status("Reload failed")  # the traceback was logged by the reload thread
        app._build_ui()  # rebuild so the advanced controls reflect the (new) session config
        self.signals.invalidate()
