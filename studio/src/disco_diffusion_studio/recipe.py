"""The preset / "recipe" controller: capture, detect, apply, and save guidance recipes.

A *recipe* is the full set of guidance knobs + schedules + the CLIP/secondary model set, bundled as
a :class:`~.presets.Preset`. :class:`Recipe` owns the loaded presets, the dropdown's current
selection, and the save-as modal; it captures the live settings into a Preset, detects which saved
preset (if any) the live settings match, applies a recipe (config now, model change auto-reloads),
and flips the dropdown to "Custom" when the user edits a preset-controlled knob. It reaches the
App's pieces (session, sidebar, models, history, status) through the App; the App holds one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pygame
import pygame_gui
from disco_diffusion.config import AVAILABLE_CLIP_MODELS

from .controls import CUSTOM_PRESET
from .layout import CTRL_H, LABEL_H
from .presets import Preset, PresetConfig, load_presets, match_preset, save_preset
from .signals import Signals
from .state import SharedState

if TYPE_CHECKING:
    from .app import App

log = logging.getLogger("disco_diffusion_studio.recipe")


class Recipe:
    """The loaded presets, the dropdown selection, recipe apply/detect, and the save-as modal."""

    def __init__(self, app: App, signals: Signals, state: SharedState) -> None:
        self.app = app
        self.signals = signals
        self.state = state
        # Presets are loaded from studio/presets/*.toml and surfaced as a dropdown that flips to
        # "Custom" once any preset-controlled knob is edited. ``applying`` suppresses that flip
        # while a preset is being applied (so its own widget updates don't read as edits).
        self.presets: dict[str, Preset] = load_presets()
        self.applying = False
        self.selection = self.detect()  # the preset matching the loaded session (or "Custom")
        # "Save preset" modal (filename prompt). None when closed.
        self.save_window: pygame_gui.elements.UIWindow | None = None
        self.save_entry: pygame_gui.elements.UITextEntryLine | None = None
        self.save_ok: pygame_gui.elements.UIButton | None = None
        self.save_cancel: pygame_gui.elements.UIButton | None = None

    def current(self) -> Preset:
        """Capture the live settings (guidance + per-run + schedules + models) as a Preset."""
        config = PresetConfig.from_run_config(self.state.session.config)
        models = [m for m in AVAILABLE_CLIP_MODELS if m in self.state.clip_selected]
        return Preset(
            config=config, clip_models=models, use_secondary_model=self.state.secondary_on
        )

    def detect(self) -> str:
        """The saved preset whose recipe matches the live settings, else "Custom"."""
        return match_preset(self.presets, self.current()) or CUSTOM_PRESET

    def set_selection(self, name: str) -> None:
        """Set the dropdown's selected entry (rebuilding it, since it has no set-selected API)."""
        if self.selection == name and self.app.sidebar.preset_dropdown is not None:
            return
        self.selection = name
        self.app.sidebar.spawn_preset_dropdown(self.app)

    def mark_custom(self) -> None:
        """Flip the preset dropdown to "Custom" after the user edits a preset-controlled knob."""
        if self.applying:
            return
        self.set_selection(CUSTOM_PRESET)

    def apply_recipe(
        self, config: PresetConfig, clip_models: list[str], use_secondary: bool
    ) -> None:
        """Apply a recipe's config knobs (now) + stage its model set (a change auto-reloads).

        Shared by preset and session loads; wrapped in ``applying`` so the widget updates don't
        read as user edits (which would flip the preset dropdown to "Custom").
        """
        app = self.app
        self.applying = True
        try:
            cfg = self.state.session.config
            for attr, value in config.model_dump().items():
                setattr(cfg, attr, value)
            app.sidebar.refresh_advanced_widgets(app)
            self.state.clip_selected = set(clip_models)
            self.state.secondary_on = use_secondary
            app.sidebar.sync_model_buttons(self.state.clip_selected, self.state.secondary_on)
            app.models.update_queue()  # one-way: re-evaluate the debounced reload for the new set
        finally:
            self.applying = False

    def apply_preset(self, name: str) -> None:
        """Load a full-recipe preset: config knobs apply now; a model change auto-reloads."""
        app = self.app
        preset = self.presets.get(name)
        if preset is None:
            return
        self.apply_recipe(preset.config, preset.clip_models, preset.use_secondary_model)
        self.selection = name
        # A preset retunes the live guidance, so record a revert point (discrete change — no
        # debounce); supersede any pending guidance-drag checkpoint.
        app.history.cancel_guidance_checkpoint()
        app.history.request_checkpoint(f"preset {name}")
        self.signals.status(f"Loaded {name}")

    # -- save-as modal --
    def save_modal_alive(self) -> bool:
        """True while the save-preset modal is open (so App._modal_open suppresses canvas input)."""
        return self.save_window is not None and self.save_window.alive()

    def open_save_dialog(self) -> None:
        """Open a small modal asking for a filename to save the current settings as a preset."""
        app = self.app
        if self.save_modal_alive():
            return
        rect = app.layout.centered_rect(420, 168)
        ui = pygame_gui.elements
        self.save_window = ui.UIWindow(rect, app.manager, window_display_title="Save preset")
        cont = self.save_window
        inner_w = rect.width - 32
        ui.UILabel(pygame.Rect(6, 4, inner_w, LABEL_H), "Filename", app.manager, container=cont)
        self.save_entry = ui.UITextEntryLine(
            pygame.Rect(6, 32, inner_w, CTRL_H), app.manager, container=cont
        )
        self.save_entry.set_text("my-preset")
        self.save_entry.focus()
        self.save_ok = ui.UIButton(
            pygame.Rect(inner_w - 150, 78, 72, CTRL_H),
            "Save",
            app.manager,
            container=cont,
            object_id="#add_button",
        )
        self.save_cancel = ui.UIButton(
            pygame.Rect(inner_w - 70, 78, 72, CTRL_H), "Cancel", app.manager, container=cont
        )

    def close_save_dialog(self) -> None:
        if self.save_window is not None:
            self.save_window.kill()
        self.save_window = None
        self.save_entry = None
        self.save_ok = None
        self.save_cancel = None

    def save_current(self) -> None:
        """Write the current settings to presets/<filename>.toml and select the new preset."""
        if self.save_entry is None:
            return
        filename = self.save_entry.get_text().strip() or "preset"
        try:
            name, _path = save_preset(filename, self.current())
        except Exception as exc:  # noqa: BLE001 - surface the failure instead of crashing
            log.exception("saving preset failed")
            self.signals.status(f"Save failed: {exc}")
            return
        self.presets = load_presets()
        self.close_save_dialog()
        self.set_selection(name)
        self.signals.status(f"Saved {name}")
