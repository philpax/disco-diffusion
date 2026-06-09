"""The right sidebar: the Settings tab + the read-only Current tab + session save/load.

:class:`Sidebar` owns the sidebar's widgets *and* builds them (``build`` + the per-section
helpers), keeping the right column out of the god-class. It takes the App for shared state /
actions, and routes its own widget events via ``handle``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pygame
import pygame_gui
from disco_diffusion.config import AVAILABLE_CLIP_MODELS, parse_schedule
from pygame_gui.elements import (
    UIButton,
    UIDropDownMenu,
    UIHorizontalSlider,
    UILabel,
    UIScrollingContainer,
    UITextEntryLine,
)

from ..constants import GUIDANCE_CHECKPOINT_MS
from ..controls import CURRENT_PERRUN, CUSTOM_PRESET, LIVE_SCALES, SCHEDULES
from ..layout import CTRL_H, LABEL_H, MARGIN, PAD, Row
from ..util import int_or

if TYPE_CHECKING:
    from ..app import App


@dataclass
class Sidebar:
    """Owns the sidebar widgets (tabs, panels, and the Settings/Current controls)."""

    # Tabs + the two scrolling panels they switch between.
    tab_settings: UIButton = field(init=False)
    tab_current: UIButton = field(init=False)
    settings_panel: UIScrollingContainer = field(init=False)
    current_panel: UIScrollingContainer = field(init=False)
    # Session save/load (top of the sidebar, above the tabs).
    save_session_button: UIButton = field(init=False)
    load_session_button: UIButton = field(init=False)
    # Output section.
    steps_entry: UITextEntryLine = field(init=False)
    seed_entry: UITextEntryLine = field(init=False)
    random_seed_button: UIButton = field(init=False)
    width_entry: UITextEntryLine = field(init=False)
    height_entry: UITextEntryLine = field(init=False)
    apply_button: UIButton = field(init=False)
    swap_button: UIButton = field(init=False)
    # Init-image section.
    open_init_button: UIButton = field(init=False)
    use_current_init_button: UIButton = field(init=False)
    clear_init_button: UIButton = field(init=False)
    _init_status_label: UILabel = field(init=False)
    _init_denoise_label: UILabel = field(init=False)
    _init_denoise_slider: UIHorizontalSlider = field(init=False)
    # Preset section.
    save_preset_button: UIButton = field(init=False)
    _preset_dd_rect: pygame.Rect = field(init=False)
    # Per-run section.
    perlin_button: UIButton = field(init=False)
    secondary_button: UIButton = field(init=False)
    _eta_label: UILabel = field(init=False)
    _eta_slider: UIHorizontalSlider = field(init=False)
    # Content width of the settings panel (minus the scrollbar).
    _sb_inner_w: int = field(init=False)

    # State / dicts (populated during build; safe defaults until then).
    preset_dropdown: UIDropDownMenu | None = None
    _sidebar_tab: str = "settings"  # "settings" | "current"
    _scale_sliders: dict[UIHorizontalSlider, tuple[str, bool, UILabel, str]] = field(
        default_factory=dict
    )
    _schedule_entries: dict[UITextEntryLine, str] = field(default_factory=dict)
    _clip_buttons: dict[UIButton, str] = field(default_factory=dict)
    _current_labels: dict[str, UILabel] = field(default_factory=dict)

    def build(self, app: App) -> None:
        """The full-height right sidebar: a Settings / Current tab pair over a scroll area."""
        ui = pygame_gui.elements
        sb = app.layout.sidebar_rect()
        x = sb.x + PAD
        inner = max(40, sb.width - 2 * PAD)
        half = (inner - PAD) // 2

        # Session save/load sits at the very top — a global action (works on either tab), so it's
        # not buried in Settings; what it saves/loads is clear from context (the whole session).
        r = Row(x, MARGIN, inner, CTRL_H)
        self.save_session_button = ui.UIButton(
            r.left(half), "Save…", app.manager, object_id="#add_button"
        )
        self.load_session_button = ui.UIButton(
            r.fill(), "Load…", app.manager, object_id="#add_button"
        )

        tabs_y = MARGIN + CTRL_H + PAD
        r = Row(x, tabs_y, inner, CTRL_H)
        self.tab_settings = ui.UIButton(
            r.left(half), "Settings", app.manager, object_id="#tab_button"
        )
        self.tab_current = ui.UIButton(r.fill(), "Current", app.manager, object_id="#tab_button")

        cont_y = tabs_y + CTRL_H + PAD
        cont_rect = pygame.Rect(x, cont_y, inner, max(60, app.layout.win_h - cont_y - MARGIN))
        self.settings_panel = ui.UIScrollingContainer(cont_rect, app.manager)
        self.current_panel = ui.UIScrollingContainer(cont_rect, app.manager)
        self._sb_inner_w = inner - 24  # leave room for the vertical scrollbar
        self._build_settings_rows(app)
        self._build_current_rows(app)
        self.sync_tabs(app)

    def _section_label(
        self, app: App, container: UIScrollingContainer, inner_w: int, y: int, text: str
    ) -> int:
        """Draw a section heading in the settings panel; return the y below it."""
        pygame_gui.elements.UILabel(
            Row(0, y, inner_w, LABEL_H).fill(),
            text,
            app.manager,
            container=container,
            object_id="#section_label",
        )
        return y + LABEL_H + 6

    def _build_settings_rows(self, app: App) -> None:
        """Build the sidebar Settings tab, one section at a time, threading the y cursor.

        Layout order: output (steps / seed / size) · init image · preset (dropdown + Save) ·
        guidance sliders (retune live) · per-run eta/perlin + cut schedules · model set. Sliders
        write straight to ``session.config`` (read live by the worker each step); schedule boxes
        are validated and applied next Play; toggling a model queues a (debounced) auto-reload.
        """
        self._scale_sliders = {}
        self._schedule_entries = {}
        self._clip_buttons = {}
        self.preset_dropdown = None  # cleared by clear_and_reset; respawned below
        container = self.settings_panel
        inner_w = self._sb_inner_w
        pitch = CTRL_H + 8
        y = 2
        y = self._build_output_section(app, container, inner_w, pitch, y)
        y = self._build_init_section(app, container, inner_w, pitch, y)
        y = self._build_preset_section(app, container, inner_w, pitch, y)
        y = self._build_guidance_section(app, container, inner_w, pitch, y)
        y = self._build_perrun_section(app, container, inner_w, pitch, y)
        y = self._build_models_section(app, container, inner_w, pitch, y)
        container.set_scrollable_area_dimensions((inner_w, y + 8))

    def _build_output_section(
        self, app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
    ) -> int:
        """Steps, seed (+ Rnd), width/height, apply / flip — all per-run (apply on next Play)."""
        ui = pygame_gui.elements
        y = self._section_label(app, container, inner_w, y, "Output — apply on next Play")
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(54), "Steps", app.manager, container=container)
        self.steps_entry = ui.UITextEntryLine(r.fill(), app.manager, container=container)
        self.steps_entry.set_text(str(app.steps))
        y += pitch
        # Seed: always shows the concrete seed in use (so it's reproducible and visible); Play
        # uses whatever's here, and "Rnd" rolls a fresh one.
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(54), "Seed", app.manager, container=container)
        self.random_seed_button = ui.UIButton(r.right(46), "Rnd", app.manager, container=container)
        self.seed_entry = ui.UITextEntryLine(r.fill(), app.manager, container=container)
        self.seed_entry.set_text(app._seed_text)
        y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(20), "W", app.manager, container=container)
        self.width_entry = ui.UITextEntryLine(
            r.left((inner_w - 56) // 2), app.manager, container=container
        )
        self.width_entry.set_text(str(app.width))
        ui.UILabel(r.left(20), "H", app.manager, container=container)
        self.height_entry = ui.UITextEntryLine(r.fill(), app.manager, container=container)
        self.height_entry.set_text(str(app.height))
        y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        self.apply_button = ui.UIButton(
            r.left((inner_w - PAD) // 2), "Apply size", app.manager, container=container
        )
        self.swap_button = ui.UIButton(r.fill(), "Flip W/H", app.manager, container=container)
        return y + pitch

    def _build_init_section(
        self, app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
    ) -> int:
        """Init image (img2img): Open… / Use current / Clear + the denoise slider (per-run)."""
        ui = pygame_gui.elements
        y = self._section_label(app, container, inner_w, y + 6, "Init image — applies on next Play")
        self._init_status_label = ui.UILabel(
            Row(0, y, inner_w, CTRL_H).fill(),
            f"Init: {app._init.label}",
            app.manager,
            container=container,
        )
        y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        third = (inner_w - 2 * PAD) // 3
        self.open_init_button = ui.UIButton(
            r.left(third), "Open…", app.manager, container=container
        )
        self.use_current_init_button = ui.UIButton(
            r.left(third), "Use current", app.manager, container=container
        )
        self.clear_init_button = ui.UIButton(r.fill(), "Clear", app.manager, container=container)
        y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(80), "Denoise", app.manager, container=container)
        self._init_denoise_label = ui.UILabel(r.right(48), "", app.manager, container=container)
        self._init_denoise_slider = ui.UIHorizontalSlider(
            r.fill(),
            start_value=float(app._init.denoise),
            value_range=(0.0, 100.0),
            manager=app.manager,
            container=container,
        )
        self._init_denoise_label.set_text(f"{app._init.denoise}%")
        return y + pitch

    def _build_preset_section(
        self, app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
    ) -> int:
        """A dropdown of saved recipes + Save; selecting one applies the whole recipe."""
        ui = pygame_gui.elements
        y = self._section_label(app, container, inner_w, y + 6, "Preset")
        r = Row(0, y, inner_w, CTRL_H)
        self.save_preset_button = ui.UIButton(
            r.right(72), "Save…", app.manager, container=container, object_id="#add_button"
        )
        self._preset_dd_rect = r.fill()
        self.spawn_preset_dropdown(app)
        return y + pitch

    def _build_guidance_section(
        self, app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
    ) -> int:
        """The live-guidance sliders — each retunes the running step immediately."""
        ui = pygame_gui.elements
        cfg = app.session.config
        y = self._section_label(app, container, inner_w, y + 6, "Guidance — retunes live")
        for sc in LIVE_SCALES:
            r = Row(0, y, inner_w, CTRL_H)
            ui.UILabel(r.left(108), sc.label, app.manager, container=container)
            vlabel = ui.UILabel(r.right(66), "", app.manager, container=container)
            cur = float(getattr(cfg, sc.attr))
            slider = ui.UIHorizontalSlider(
                r.fill(),
                start_value=min(max(cur, sc.lo), sc.hi),
                value_range=(sc.lo, sc.hi),
                manager=app.manager,
                container=container,
            )
            vlabel.set_text(sc.fmt.format(cur))
            self._scale_sliders[slider] = (sc.attr, sc.is_int, vlabel, sc.fmt)
            y += pitch
        return y

    def _build_perrun_section(
        self, app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
    ) -> int:
        """eta + Perlin init, then the raw cut-schedule text boxes (applied on next Play)."""
        ui = pygame_gui.elements
        cfg = app.session.config
        y = self._section_label(app, container, inner_w, y + 6, "Per-run — apply on next Play")
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(40), "eta", app.manager, container=container)
        self.perlin_button = ui.UIButton(
            r.right(96), "Perlin init", app.manager, container=container
        )
        self._eta_label = ui.UILabel(r.right(48), "", app.manager, container=container)
        self._eta_slider = ui.UIHorizontalSlider(
            r.fill(),
            start_value=min(max(float(cfg.eta), 0.0), 1.0),
            value_range=(0.0, 1.0),
            manager=app.manager,
            container=container,
        )
        self._eta_label.set_text(f"{cfg.eta:.2f}")
        if cfg.perlin_init:
            self.perlin_button.select()
        y += pitch
        for sch in SCHEDULES:
            ui.UILabel(
                Row(0, y, inner_w, LABEL_H).left(inner_w),
                sch.label,
                app.manager,
                container=container,
            )
            y += LABEL_H + 2
            entry = ui.UITextEntryLine(
                Row(0, y, inner_w, CTRL_H).fill(), app.manager, container=container
            )
            entry.set_text(str(getattr(cfg, sch.attr)))
            self._schedule_entries[entry] = sch.attr
            y += pitch
        return y

    def _build_models_section(
        self, app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
    ) -> int:
        """CLIP model toggles + the secondary-model toggle (changing these auto-reloads)."""
        ui = pygame_gui.elements
        y = self._section_label(app, container, inner_w, y + 6, "Models — auto-reloads on change")
        per_row = 2
        bw = (inner_w - (per_row - 1) * PAD) // per_row
        r = Row(0, y, inner_w, CTRL_H)
        for i, name in enumerate(AVAILABLE_CLIP_MODELS):
            if i % per_row == 0:
                r = Row(0, y, inner_w, CTRL_H)
            button = ui.UIButton(
                r.left(bw), name, app.manager, container=container, object_id="#brush_button"
            )
            self._clip_buttons[button] = name
            if name in app._clip_selected:
                button.select()
            if i % per_row == per_row - 1:
                y += pitch
        if len(AVAILABLE_CLIP_MODELS) % per_row != 0:
            y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        self.secondary_button = ui.UIButton(
            r.fill(),
            "Secondary model",
            app.manager,
            container=container,
            object_id="#brush_button",
        )
        if app._secondary_on:
            self.secondary_button.select()
        return y + pitch

    def _build_current_rows(self, app: App) -> None:
        """Build the read-only "Current" tab: name + value label per setting."""
        ui = pygame_gui.elements
        container = self.current_panel
        inner_w = self._sb_inner_w
        pitch = CTRL_H + 4
        self._current_labels = {}
        name_w = 116

        def row(y: int, key: str, name: str) -> int:
            ui.UILabel(
                Row(0, y, inner_w, CTRL_H).left(name_w), name, app.manager, container=container
            )
            value = ui.UILabel(
                Row(name_w + PAD, y, inner_w - name_w - PAD, CTRL_H).fill(),
                "",
                app.manager,
                container=container,
            )
            self._current_labels[key] = value
            return y + pitch

        # Headings are plain — the Current tab reflects what the image is generating with, so the
        # values speak for themselves without a "this run / next run" qualifier.
        y = 2
        ui.UILabel(
            Row(0, y, inner_w, LABEL_H).fill(),
            "Guidance",
            app.manager,
            container=container,
            object_id="#section_label",
        )
        y += LABEL_H + 6
        for sc in LIVE_SCALES:
            y = row(y, sc.attr, sc.label)
        y += 6
        ui.UILabel(
            Row(0, y, inner_w, LABEL_H).fill(),
            "Per-run",
            app.manager,
            container=container,
            object_id="#section_label",
        )
        y += LABEL_H + 6
        for key, name in CURRENT_PERRUN:
            y = row(y, key, name)
        container.set_scrollable_area_dimensions((inner_w, y + 8))
        self.refresh_current(app)

    def sync_tabs(self, app: App) -> None:
        """Show exactly one of the Settings / Current panels for the active sidebar tab."""
        settings = self._sidebar_tab == "settings"
        (self.settings_panel.show if settings else self.settings_panel.hide)()
        (self.current_panel.show if not settings else self.current_panel.hide)()
        (self.tab_settings.select if settings else self.tab_settings.unselect)()
        (self.tab_current.select if not settings else self.tab_current.unselect)()

    def set_init_status(self, label: str) -> None:
        """Update the init-image status line (e.g. "Init: foo.png" or "Init: none")."""
        self._init_status_label.set_text(label)

    def steps_text(self) -> str:
        """The raw text in the steps box (parse/clamp is the caller's job)."""
        return self.steps_entry.get_text()

    def set_steps_text(self, text: str) -> None:
        """Show a (clamped) step count in the steps box."""
        self.steps_entry.set_text(text)

    def seed_text(self) -> str:
        """The raw text in the seed box."""
        return self.seed_entry.get_text()

    def set_seed_text(self, text: str) -> None:
        """Show the seed in use in the seed box."""
        self.seed_entry.set_text(text)

    def set_size_text(self, width: int, height: int) -> None:
        """Reflect the (snapped) output size in the width/height boxes."""
        self.width_entry.set_text(str(width))
        self.height_entry.set_text(str(height))

    def set_denoise(self, percent: int) -> None:
        """Reflect the img2img denoise percentage in its slider + label."""
        self._init_denoise_slider.set_current_value(float(percent))
        self._init_denoise_label.set_text(f"{percent}%")

    def model_name(self, button: UIButton) -> str:
        """The CLIP model a toggle button represents."""
        return self._clip_buttons[button]

    def sync_model_buttons(self, selected: set[str], secondary: bool) -> None:
        """Light the CLIP / secondary toggles to match a staged model selection."""
        for button, name in self._clip_buttons.items():
            (button.select if name in selected else button.unselect)()
        (self.secondary_button.select if secondary else self.secondary_button.unselect)()

    def refresh_advanced_widgets(self, app: App) -> None:
        """Re-sync every Advanced widget from the current config (after a preset load)."""
        cfg = app.session.config
        for slider, (attr, _is_int, vlabel, fmt) in self._scale_sliders.items():
            value = float(getattr(cfg, attr))
            slider.set_current_value(value)
            vlabel.set_text(fmt.format(value))
        self._eta_slider.set_current_value(min(max(float(cfg.eta), 0.0), 1.0))
        self._eta_label.set_text(f"{cfg.eta:.2f}")
        (self.perlin_button.select if cfg.perlin_init else self.perlin_button.unselect)()
        for entry, attr in self._schedule_entries.items():
            entry.set_text(str(getattr(cfg, attr)))

    def spawn_preset_dropdown(self, app: App) -> None:
        """(Re)create the preset dropdown in the settings panel at the stored rect/selection."""
        if self.preset_dropdown is not None:
            self.preset_dropdown.kill()
        options: list[str | tuple[str, str]] = [*app._presets.keys(), CUSTOM_PRESET]
        selected = app._preset_selection if app._preset_selection in options else CUSTOM_PRESET
        app._preset_selection = selected
        self.preset_dropdown = UIDropDownMenu(
            options,
            selected,
            self._preset_dd_rect,
            app.manager,
            container=self.settings_panel,
        )

    def handle(self, app: App, event: pygame.event.Event) -> bool:
        """Handle an event targeting a sidebar widget; return True if it was ours."""
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            e = event.ui_element
            if e in (self.tab_settings, self.tab_current):
                self._sidebar_tab = "settings" if e == self.tab_settings else "current"
                self.sync_tabs(app)
            elif e == self.save_preset_button:
                app._open_save_preset_dialog()
            elif e == self.save_session_button:
                app.session_io.save()
            elif e == self.load_session_button:
                app.session_io.load()
            elif e == self.open_init_button:
                app._open_init()
            elif e == self.use_current_init_button:
                app._use_current_as_init()
            elif e == self.clear_init_button:
                app._clear_init()
            elif e == self.perlin_button:
                app.session.config.perlin_init = not app.session.config.perlin_init
                on = app.session.config.perlin_init
                (self.perlin_button.select if on else self.perlin_button.unselect)()
                app._status(f"Perlin {'on' if on else 'off'}")
                app._mark_custom()
            elif e in self._clip_buttons:
                app._toggle_clip_model(e)
            elif e == self.secondary_button:
                app._secondary_on = not app._secondary_on
                (
                    self.secondary_button.select
                    if app._secondary_on
                    else self.secondary_button.unselect
                )()
                app._update_reload_queue()
                app._mark_custom()
            elif e == self.apply_button:
                app.generation.apply_size(
                    int_or(self.width_entry.get_text(), app.width),
                    int_or(self.height_entry.get_text(), app.height),
                )
            elif e == self.swap_button:
                app.generation.apply_size(app.height, app.width)
            elif e == self.random_seed_button:
                app._seed_text = str(random.randrange(2**31))  # roll a fresh, visible seed
                self.seed_entry.set_text(app._seed_text)
                app._status(f"Seed {app._seed_text}")
            else:
                return False
            return True
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            e = event.ui_element
            if e == self._eta_slider:
                # eta is read when the loop's generator is built, so this lands on the next run.
                app.session.config.eta = float(event.value)
                self._eta_label.set_text(f"{event.value:.2f}")
                app._mark_custom()
            elif e == self._init_denoise_slider:
                # img2img strength — converted to skip_steps at the next Play.
                app._init.denoise = int(round(event.value))
                self._init_denoise_label.set_text(f"{app._init.denoise}%")
            elif e in self._scale_sliders:
                attr, is_int, vlabel, fmt = self._scale_sliders[e]
                value: float | int = int(round(event.value)) if is_int else float(event.value)
                # session.config is the live config the running Sampler reads each step, so this
                # retunes guidance on the next step (and seeds the next run when stopped).
                setattr(app.session.config, attr, value)
                vlabel.set_text(fmt.format(value))
                app._mark_custom()
                # Drop a revert point once the drag settles (debounced in run()).
                app.history.arm_guidance_checkpoint(
                    pygame.time.get_ticks() + GUIDANCE_CHECKPOINT_MS
                )
            else:
                return False
            return True
        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.preset_dropdown:
                app._preset_selection = event.text
                if event.text != CUSTOM_PRESET:
                    app._apply_preset(event.text)
                return True
            return False
        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            e = event.ui_element
            if e == self.steps_entry:
                app.generation.commit_steps()
                return True
            if e in (self.width_entry, self.height_entry):
                return True  # applied via the Apply button
            if e in self._schedule_entries:
                self.commit_schedule_entry(app, e)
                return True
        return False

    def perrun_values(self, app: App) -> dict[str, str]:
        """Display strings for every CURRENT_PERRUN key from the current (pending) state.

        Driven by the key list itself: the two synthesised keys are special-cased, the rest are
        read off ``session.config`` by name (so they can't drift from the listed keys).
        """
        cfg = app.session.config
        out: dict[str, str] = {}
        for key, _label in CURRENT_PERRUN:
            if key == "steps":
                out[key] = str(app.steps)
            elif key == "size":
                out[key] = f"{app.width} × {app.height}"
            elif key == "clip_models":
                out[key] = ", ".join(cfg.clip_models)
            else:
                value = getattr(cfg, key)
                if isinstance(value, bool):
                    out[key] = "on" if value else "off"
                elif isinstance(value, float):
                    out[key] = f"{value:.2f}"
                else:
                    out[key] = str(value)
        return out

    def refresh_current(self, app: App) -> None:
        """Update the Current tab: live knobs from session.config, per-run from the snapshot."""
        if not self._current_labels:
            return
        cfg = app.session.config
        for sc in LIVE_SCALES:  # live: reflect session.config as sliders move
            label = self._current_labels.get(sc.attr)
            if label is not None:
                text = sc.fmt.format(float(getattr(cfg, sc.attr)))
                if label.text != text:
                    label.set_text(text)
        # While a run exists (playing, paused, or done) these reflect that run's frozen snapshot —
        # what the image on screen was generated with; only once fully stopped do they show the
        # pending values the next run would use.
        perrun = app.generation.run_snapshot if app.worker is not None else self.perrun_values(app)
        for key, _name in CURRENT_PERRUN:
            label = self._current_labels.get(key)
            text = perrun.get(key, "—")
            if label is not None and label.text != text:
                label.set_text(text)

    def commit_schedule_entry(self, app: App, entry: UITextEntryLine) -> None:
        """Validate a cut-schedule box and store it on the config (applies on next Play).

        Schedules are parsed with the library's own parser; on a malformed string we flag it
        and restore the previous value rather than letting the worker blow up at run start.
        """
        attr = self._schedule_entries.get(entry)
        if attr is None:
            return
        text = entry.get_text().strip()
        if text == str(getattr(app.session.config, attr)):
            return
        try:
            parsed = parse_schedule(text)
        except ValueError:
            app._status("Bad schedule")
            entry.set_text(str(getattr(app.session.config, attr)))
            return
        # cond_fn indexes these over the full 1000-step internal timeline, so a short schedule
        # would IndexError mid-run. Require it to cover 1000 (extra entries are harmless).
        if len(parsed) < 1000:
            app._status("Schedule short")
            entry.set_text(str(getattr(app.session.config, attr)))
            return
        setattr(app.session.config, attr, text)
        app._status("Schedule set")
        app._mark_custom()

    def sync_enabled(self, app: App) -> None:
        """The output boxes (steps / seed / size) are editable only when not generating."""
        editable = not app.running
        for el in (
            self.steps_entry,
            self.seed_entry,
            self.random_seed_button,
            self.width_entry,
            self.height_entry,
            self.apply_button,
            self.swap_button,
        ):
            (el.enable if editable else el.disable)()
